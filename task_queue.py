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

logger = logging.getLogger(__name__)


class TaskPriority(IntEnum):
    """任务优先级枚举"""
    HIGH = 1    # 高优先级（对话语音，实时）
    MEDIUM = 2  # 中优先级（普通任务）
    LOW = 3     # 低优先级（批处理，定时任务）


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

    def __lt__(self, other):
        # 优先级队列按优先级排序
        return self.priority < other.priority


class TaskQueueManager:
    """任务队列管理器"""

    def __init__(self):
        self.config = get_config().queue
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running_count = 0
        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._workers: list = []
        self._tasks: Dict[str, QueuedTask] = {}
        self._stats_lock = threading.Lock()

    def start(self):
        """启动工作线程"""
        if self._executor is not None:
            return

        self._executor = ThreadPoolExecutor(
            max_workers=self.config.worker_threads
        )
        logger.info(f"TaskQueue started with {self.config.worker_threads} workers, max_concurrent={self.config.max_concurrent}")

    def stop(self):
        """停止工作线程"""
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

        self._tasks[task_id] = task
        await self._queue.put(task)
        logger.info(f"Task {task_id} enqueued with priority {priority}")

        return task_id

    async def dequeue(self) -> Optional[QueuedTask]:
        """从队列获取任务（等待并发槽位）"""
        while True:
            # 等待并发槽位
            with self._lock:
                if self._running_count < self.config.max_concurrent:
                    self._running_count += 1
                    break

            # 等待一段时间后重试
            await asyncio.sleep(0.5)

        # 获取任务
        task = await self._queue.get()
        task.started_at = time.time()
        return task

    def release_slot(self):
        """释放并发槽位"""
        with self._lock:
            self._running_count = max(0, self._running_count - 1)

    def complete_task(self, task_id: str):
        """标记任务完成"""
        if task_id in self._tasks:
            self._tasks[task_id].completed_at = time.time()
        self.release_slot()

    def fail_task(self, task_id: str):
        """标记任务失败"""
        if task_id in self._tasks:
            self._tasks[task_id].completed_at = time.time()
        self.release_slot()

    def get_task(self, task_id: str) -> Optional[QueuedTask]:
        """获取任务信息"""
        return self._tasks.get(task_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        with self._stats_lock:
            pending = self._queue.qsize()
            running = self._running_count
            completed = sum(1 for t in self._tasks.values() if t.completed_at is not None)

            # 计算平均处理时间
            completed_tasks = [t for t in self._tasks.values() if t.completed_at and t.started_at]
            avg_time = 0
            if completed_tasks:
                avg_time = sum(t.completed_at - t.started_at for t in completed_tasks) / len(completed_tasks)

            return {
                "max_concurrent": self.config.max_concurrent,
                "worker_threads": self.config.worker_threads,
                "running": running,
                "pending": pending,
                "completed": completed,
                "total": len(self._tasks),
                "avg_processing_time": avg_time,
            }

    def get_tasks_list(self, limit: int = 50) -> list:
        """获取任务列表"""
        tasks = []
        for task in sorted(self._tasks.values(), key=lambda x: x.created_at, reverse=True)[:limit]:
            status = "pending"
            if task.completed_at:
                status = "completed"
            elif task.started_at:
                status = "running"

            tasks.append({
                "task_id": task.task_id,
                "priority": task.priority,
                "priority_label": TaskPriority(task.priority).name,
                "status": status,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "duration": (task.completed_at or time.time()) - task.started_at if task.started_at else None,
            })
        return tasks

    def update_max_concurrent(self, value: int):
        """动态调整最大并发数"""
        if value < 1:
            value = 1
        if value > 10:
            value = 10
        self.config.max_concurrent = value
        logger.info(f"Max concurrent updated to {value}")


# 全局队列管理器
_queue_manager: Optional[TaskQueueManager] = None


def get_queue_manager() -> TaskQueueManager:
    """获取队列管理器实例"""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = TaskQueueManager()
    return _queue_manager
