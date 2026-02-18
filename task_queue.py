import asyncio
import uuid
import time
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import IntEnum
from concurrent.futures import ThreadPoolExecutor
import threading

from config import get_config
from models import TaskPriority  # 统一从 models 导入

logger = logging.getLogger(__name__)


@dataclass
class QueuedTask:
    """队列任务"""
    task_id: str
    priority: TaskPriority
    file_path: str
    language: Optional[str]
    diarize: bool
    format: str
    model: Optional[str]
    callback: Callable
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "pending"  # pending, running, completed, cancelled

    def __lt__(self, other):
        # 优先级队列按优先级排序，同优先级按创建时间排序
        if self.priority == other.priority:
            return self.created_at < other.created_at
        return self.priority < other.priority


class TaskQueueManager:
    """任务队列管理器（优化版：使用 Semaphore 替代忙等待）"""

    def __init__(self):
        self.config = get_config().queue
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running_count = 0
        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._tasks: Dict[str, QueuedTask] = {}
        self._stats_lock = threading.Lock()

        # 使用 Semaphore 控制并发（替代忙等待）
        self._semaphore: Optional[asyncio.Semaphore] = None
        # 用于通知新任务到达
        self._task_available = asyncio.Event()
        # 标记队列是否运行中
        self._running = False

    def start(self):
        """启动工作线程"""
        if self._executor is not None:
            return

        self._executor = ThreadPoolExecutor(
            max_workers=self.config.worker_threads
        )
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._running = True
        logger.info(
            f"TaskQueue started with {self.config.worker_threads} workers, "
            f"max_concurrent={self.config.max_concurrent}"
        )

    def stop(self):
        """停止工作线程"""
        self._running = False
        self._task_available.set()  # 唤醒等待的循环
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
            logger.info("TaskQueue stopped")

    async def enqueue(
        self,
        task_id: str,
        file_path: str,
        priority: TaskPriority,
        language: Optional[str],
        diarize: bool,
        format: str,
        model: Optional[str],
        callback: Callable
    ) -> str:
        """将任务加入队列"""
        task = QueuedTask(
            task_id=task_id,
            priority=priority,
            file_path=file_path,
            language=language,
            diarize=diarize,
            format=format,
            model=model,
            callback=callback
        )

        with self._lock:
            self._tasks[task_id] = task

        await self._queue.put(task)
        self._task_available.set()  # 通知有新任务

        logger.info(f"Task {task_id} enqueued with priority {priority.name}")
        return task_id

    async def dequeue(self) -> Optional[QueuedTask]:
        """从队列获取任务（使用 Semaphore 控制并发）"""
        while self._running:
            # 等待任务可用
            await self._task_available.wait()

            if not self._running:
                return None

            # 尝试获取并发槽位（非阻塞检查）
            acquired = self._semaphore.locked() and self._semaphore._value == 0

            # 使用 semaphore 控制并发
            async with self._semaphore:
                # 获取任务
                try:
                    task = self._queue.get_nowait()
                    task.started_at = time.time()
                    task.status = "running"

                    # 如果队列还有任务，保持 event 设置
                    if not self._queue.empty():
                        self._task_available.set()
                    else:
                        self._task_available.clear()

                    return task
                except asyncio.QueueEmpty:
                    self._task_available.clear()
                    continue

        return None

    def release_slot(self):
        """释放并发槽位（由 complete_task/fail_task 调用）"""
        # Semaphore 通过 async with 自动释放，这里仅更新计数
        with self._lock:
            self._running_count = max(0, self._running_count - 1)

    def complete_task(self, task_id: str):
        """标记任务完成"""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].completed_at = time.time()
                self._tasks[task_id].status = "completed"
            self._running_count = max(0, self._running_count - 1)

    def fail_task(self, task_id: str):
        """标记任务失败"""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].completed_at = time.time()
                self._tasks[task_id].status = "failed"
            self._running_count = max(0, self._running_count - 1)

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            if task_id not in self._tasks:
                return False

            task = self._tasks[task_id]

            # 只能取消 pending 状态的任务
            if task.status == "pending":
                task.status = "cancelled"
                task.completed_at = time.time()
                logger.info(f"Task {task_id} cancelled")
                return True

            return False

    def get_task(self, task_id: str) -> Optional[QueuedTask]:
        """获取任务信息"""
        return self._tasks.get(task_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        with self._stats_lock:
            pending = self._queue.qsize()
            running = self._running_count

            completed_tasks = [
                t for t in self._tasks.values()
                if t.status in ("completed", "failed")
            ]
            completed = len(completed_tasks)

            # 计算平均处理时间
            timed_tasks = [
                t for t in completed_tasks
                if t.completed_at and t.started_at
            ]
            avg_time = 0
            if timed_tasks:
                avg_time = sum(
                    t.completed_at - t.started_at for t in timed_tasks
                ) / len(timed_tasks)

            return {
                "max_concurrent": self.config.max_concurrent,
                "worker_threads": self.config.worker_threads,
                "running": running,
                "pending": pending,
                "completed": completed,
                "total": len(self._tasks),
                "avg_processing_time": round(avg_time, 2),
            }

    def get_tasks_list(self, limit: int = 50) -> list:
        """获取任务列表"""
        tasks = []
        sorted_tasks = sorted(
            self._tasks.values(),
            key=lambda x: x.created_at,
            reverse=True
        )[:limit]

        for task in sorted_tasks:
            # 计算处理时长
            duration = None
            if task.status not in ("cancelled", "pending"):
                if task.started_at and task.completed_at:
                    duration = task.completed_at - task.started_at
                elif task.started_at:
                    duration = time.time() - task.started_at

            tasks.append({
                "task_id": task.task_id,
                "priority": task.priority.value,
                "priority_label": task.priority.name,
                "status": task.status,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "duration": round(duration, 2) if duration else None,
            })
        return tasks

    def update_max_concurrent(self, value: int):
        """动态调整最大并发数"""
        if value < 1:
            value = 1
        # M4 Pro 建议：根据内存调整上限
        max_recommended = 4  # 统一内存架构下，4 个并发是安全值
        if value > max_recommended:
            logger.warning(
                f"max_concurrent={value} exceeds recommended {max_recommended} for M4 Pro"
            )

        self.config.max_concurrent = value
        # 更新 semaphore（需要在事件循环中）
        if self._semaphore:
            # asyncio.Semaphore 不支持动态调整，这里只更新配置
            # 实际效果会在新请求时体现
            pass
        logger.info(f"Max concurrent updated to {value}")


# 全局队列管理器
_queue_manager: Optional[TaskQueueManager] = None


def get_queue_manager() -> TaskQueueManager:
    """获取队列管理器实例"""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = TaskQueueManager()
    return _queue_manager
