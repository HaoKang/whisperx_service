import asyncio
import os
import uuid
import time
import logging
import tempfile
import shutil
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from config import init_config, get_config
from models import (
    RecognizeResponse,
    TaskStatusResponse,
    HealthResponse,
    ModelInfo,
    OutputFormat,
    TaskStatus,
    Segment,
    TaskPriority,
    QueueStats,
    QueueConfigRequest,
    TaskListItem,
    CacheStats,
)
from whisper_service import get_whisper_service
from task_queue import get_queue_manager
from result_cache import get_result_cache
from file_utils import validate_audio_file, ALLOWED_EXTENSIONS
from rate_limiter import get_rate_limiter

# 初始化配置
config = init_config()

# 配置日志
logging.basicConfig(
    level=get_config().logging.level,
    format=get_config().logging.format,
)
logger = logging.getLogger(__name__)

# 请求追踪 ID
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """请求追踪中间件"""

    async def dispatch(self, request: Request, call_next):
        # 生成或获取 request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request_id_var.set(request_id)

        # 处理请求
        response = await call_next(request)

        # 添加到响应头
        response.headers["X-Request-ID"] = request_id
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = request_id_var.get()

        logger.info(f"[{request_id}] {request.method} {request.url.path}")

        response = await call_next(request)

        duration = time.time() - start_time
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"- {response.status_code} - {duration:.3f}s"
        )

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """请求限流中间件"""

    # 不需要限流的路径
    EXEMPT_PATHS = {"/health", "/dashboard", "/api/v1/queue/stats", "/api/v1/cache/stats"}

    async def dispatch(self, request: Request, call_next):
        # 只对 POST 请求限流
        if request.method != "POST":
            return await call_next(request)

        # 跳过豁免路径
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # 获取客户端 IP
        client_ip = request.client.host if request.client else "unknown"
        # 检查代理头
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        # 检查限流
        limiter = get_rate_limiter()
        is_allowed, info = limiter.is_allowed(client_ip, request.url.path)

        if not is_allowed:
            logger.warning(f"Rate limit hit for {client_ip}: {info}")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "请求过于频繁，请稍后重试",
                    "retry_after": info.get("retry_after", 60),
                },
                headers={"Retry-After": str(info.get("retry_after", 60))}
            )

        return await call_next(request)


# 创建 FastAPI 应用
app = FastAPI(
    title="WhisperX 语音识别服务",
    description="本地语音识别服务，基于 WhisperX，针对 Apple Silicon 优化",
    version="1.1.0",
)

# 添加中间件
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(LoggingMiddleware)

# 模板渲染
templates = Jinja2Templates(directory="templates")

# 任务存储
tasks = {}

# 初始化队列管理器
queue_manager = get_queue_manager()


@app.on_event("startup")
async def startup_event():
    """服务启动时预热模型和队列"""
    logger.info("Starting WhisperX service...")

    # 启动队列管理器
    queue_manager.start()

    # 启动模型
    service = get_whisper_service()
    try:
        service.load_model()
        device_info = service.get_device_info()
        logger.info(f"Model loaded: {device_info}")
    except Exception as e:
        logger.warning(f"Failed to load model on startup: {e}")

    # 启动队列处理循环
    asyncio.create_task(queue_processing_loop())


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时清理资源"""
    logger.info("Shutting down WhisperX service...")
    queue_manager.stop()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    service = get_whisper_service()
    device_info = service.get_device_info()

    return HealthResponse(
        status="healthy",
        model_loaded=service._model is not None,
        device=device_info.get("device"),
        is_apple_silicon=device_info.get("is_apple_silicon"),
    )


@app.get("/api/v1/models", response_model=ModelInfo)
async def get_model_info():
    """获取当前模型信息"""
    service = get_whisper_service()
    config = get_config()
    device_info = service.get_device_info()

    return ModelInfo(
        name=config.whisper.model,
        device=device_info.get("device", "unknown"),
        diarize_available=bool(config.whisper.huggingface_token),
        compute_type=device_info.get("compute_type"),
    )


@app.post("/api/v1/recognize", response_model=RecognizeResponse)
async def recognize_speech(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    diarize: bool = Form(False),
    format: OutputFormat = Form(OutputFormat.JSON),
    model: Optional[str] = Form(None),
    priority: int = Form(2),
):
    """语音识别接口（优先级队列）"""
    task_id = str(uuid.uuid4())
    request_id = request_id_var.get()

    # 验证优先级
    if priority not in [1, 2, 3]:
        priority = 2

    priority_enum = TaskPriority(priority)

    # 验证文件扩展名
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}。支持的类型: {ALLOWED_EXTENSIONS}",
        )

    # 保存上传的文件
    storage = get_config().storage
    temp_dir = Path(storage.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    file_path = temp_dir / f"{task_id}{file_ext}"

    try:
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            # 检查文件大小
            max_size = storage.max_file_size * 1024 * 1024
            if len(content) > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"文件大小超过限制: {storage.max_file_size}MB",
                )
            await f.write(content)

        # 验证文件类型（Magic Number 检测）
        is_valid, detected_type = validate_audio_file(str(file_path))
        if not is_valid:
            file_path.unlink()
            raise HTTPException(
                status_code=400,
                detail="无效的音频文件",
            )
        logger.info(f"[{request_id}] File validated: {detected_type}")

        # 检查缓存
        cache = get_result_cache()
        cached_result = cache.get(str(file_path))
        if cached_result:
            logger.info(f"[{request_id}] Cache hit for {task_id}")
            return RecognizeResponse(
                success=True,
                task_id=task_id,
                text=cached_result.get("text"),
                segments=[Segment(**seg) for seg in cached_result.get("segments", [])],
                language=cached_result.get("language"),
                duration=cached_result.get("duration"),
                cached=True,
            )

        # 创建任务记录
        tasks[task_id] = {
            "status": TaskStatus.PENDING,
            "file_path": str(file_path),
            "language": language,
            "diarize": diarize,
            "format": format,
            "model": model,
            "priority": priority,
            "created_at": time.time(),
        }

        # 加入优先级队列
        await queue_manager.enqueue(
            task_id=task_id,
            file_path=str(file_path),
            priority=priority_enum,
            language=language,
            diarize=diarize,
            format=format,
            model=model,
            callback=None
        )

        logger.info(f"[{request_id}] Task {task_id} enqueued with priority {priority_enum.name}")

        priority_label = {1: "高", 2: "中", 3: "低"}[priority]
        return RecognizeResponse(
            success=True,
            task_id=task_id,
            text=f"任务已加入队列，优先级: {priority_label}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Failed to process upload: {e}")
        # 清理临时文件
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


def process_recognition_sync(task_id: str):
    """同步处理识别任务（在工作线程中执行）"""
    task = tasks.get(task_id)
    if not task:
        logger.error(f"Task {task_id} not found")
        return

    request_id = task_id[:8]  # 使用 task_id 作为追踪 ID

    try:
        # 更新状态为处理中
        tasks[task_id]["status"] = TaskStatus.PROCESSING
        tasks[task_id]["started_at"] = time.time()

        logger.info(f"[{request_id}] Processing recognition task")

        service = get_whisper_service()
        result = service.recognize(
            audio_path=task["file_path"],
            language=task["language"],
            diarize=task["diarize"],
            model_name=task["model"],
        )

        # 根据格式转换结果
        format = task["format"]
        text = result.text

        if format == OutputFormat.SRT:
            text = service.generate_srt(result.segments)
        elif format == OutputFormat.VTT:
            text = service.generate_vtt(result.segments)
        elif format == OutputFormat.TEXT:
            text = service.generate_text(result.segments)

        # 缓存原始结果（JSON 格式）
        cache_result = {
            "text": result.text,
            "segments": [seg.model_dump() for seg in result.segments],
            "language": result.language,
            "duration": result.duration,
        }

        # 存储缓存（在文件清理之前）
        cache = get_result_cache()
        cache.set(task["file_path"], cache_result)

        tasks[task_id] = {
            "status": TaskStatus.COMPLETED,
            "result": {
                "task_id": result.task_id,
                "text": text,
                "segments": [seg.model_dump() for seg in result.segments],
                "language": result.language,
                "duration": result.duration,
            },
            "completed_at": time.time(),
        }

        # 标记队列任务完成
        queue_manager.complete_task(task_id)

        duration = time.time() - tasks[task_id]["completed_at"]
        logger.info(f"[{request_id}] Task completed in {duration:.2f}s")

        # 清理临时文件
        file_path = Path(task.get("file_path"))
        if file_path.exists():
            file_path.unlink()

    except Exception as e:
        logger.error(f"[{request_id}] Recognition failed: {e}")
        tasks[task_id] = {
            "status": TaskStatus.FAILED,
            "error": str(e),
            "completed_at": time.time(),
        }
        # 标记队列任务失败
        queue_manager.fail_task(task_id)

        # 清理临时文件
        file_path = Path(task.get("file_path", ""))
        if file_path.exists():
            file_path.unlink()


async def queue_processing_loop():
    """队列处理循环"""
    logger.info("Queue processing loop started")

    loop = asyncio.get_event_loop()

    while True:
        try:
            # 从队列获取任务
            task = await queue_manager.dequeue()

            if task is None:
                # 队列已停止
                break

            logger.info(f"Processing task: {task.task_id}")

            # 使用线程池执行阻塞的识别任务
            await loop.run_in_executor(
                queue_manager._executor,
                process_recognition_sync,
                task.task_id
            )

        except asyncio.CancelledError:
            logger.info("Queue processing loop cancelled")
            break
        except Exception as e:
            logger.error(f"Error in queue processing: {e}")
            await asyncio.sleep(1)


@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """获取任务状态"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    status = task["status"]

    if status == TaskStatus.COMPLETED:
        result = task["result"]
        return TaskStatusResponse(
            task_id=task_id,
            status=status,
            result=result,
        )
    elif status == TaskStatus.FAILED:
        return TaskStatusResponse(
            task_id=task_id,
            status=status,
            error=task.get("error"),
        )
    else:
        # 计算等待时间
        wait_time = None
        if status == TaskStatus.PENDING:
            wait_time = time.time() - task.get("created_at", time.time())

        return TaskStatusResponse(
            task_id=task_id,
            status=status,
            progress=0.5 if status == TaskStatus.PROCESSING else 0.0,
            wait_time=wait_time,
        )


@app.get("/api/v1/result/{task_id}")
async def get_result(task_id: str, format: Optional[OutputFormat] = None):
    """获取识别结果"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Task not completed")

    result = task["result"]
    service = get_whisper_service()
    output_format = format or task.get("format", OutputFormat.JSON)

    if output_format == OutputFormat.JSON:
        return result
    elif output_format == OutputFormat.SRT:
        segments = [
            Segment(**seg) for seg in result.get("segments", [])
        ]
        return PlainTextResponse(service.generate_srt(segments))
    elif output_format == OutputFormat.VTT:
        segments = [
            Segment(**seg) for seg in result.get("segments", [])
        ]
        return PlainTextResponse(service.generate_vtt(segments))
    elif output_format == OutputFormat.TEXT:
        return PlainTextResponse(result.get("text", ""))


# 同步识别接口（适合小文件）
@app.post("/api/v1/recognize/sync", response_model=RecognizeResponse)
async def recognize_speech_sync(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    diarize: bool = Form(False),
    format: OutputFormat = Form(OutputFormat.JSON),
    model: Optional[str] = Form(None),
):
    """同步语音识别接口（适合小文件，使用线程池避免阻塞）"""
    task_id = str(uuid.uuid4())
    request_id = request_id_var.get()

    # 验证文件扩展名
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}",
        )

    # 保存上传的文件
    temp_dir = Path(get_config().storage.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_path = temp_dir / f"{task_id}{file_ext}"

    try:
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        # 验证文件类型（Magic Number 检测）
        is_valid, detected_type = validate_audio_file(str(file_path))
        if not is_valid:
            file_path.unlink()
            raise HTTPException(
                status_code=400,
                detail="无效的音频文件",
            )

        # 使用线程池执行识别（避免阻塞事件循环）
        loop = asyncio.get_event_loop()
        service = get_whisper_service()

        result = await loop.run_in_executor(
            None,
            lambda: service.recognize(
                audio_path=str(file_path),
                language=language,
                diarize=diarize,
                model_name=model,
            )
        )

        # 转换格式
        text = result.text
        if format == OutputFormat.SRT:
            text = service.generate_srt(result.segments)
        elif format == OutputFormat.VTT:
            text = service.generate_vtt(result.segments)
        elif format == OutputFormat.TEXT:
            text = service.generate_text(result.segments)

        logger.info(f"[{request_id}] Sync recognition completed: {task_id}")

        return RecognizeResponse(
            success=True,
            task_id=task_id,
            text=text,
            segments=result.segments if format == OutputFormat.JSON else None,
            language=result.language,
            duration=result.duration,
        )

    except Exception as e:
        logger.error(f"[{request_id}] Sync recognition failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        if file_path.exists():
            file_path.unlink()


# ============== 队列管理 API ==============

@app.get("/api/v1/queue/stats", response_model=QueueStats)
async def get_queue_stats():
    """获取队列统计信息"""
    return queue_manager.get_stats()


@app.post("/api/v1/queue/config")
async def update_queue_config(config_req: QueueConfigRequest):
    """更新队列配置"""
    queue_manager.update_max_concurrent(config_req.max_concurrent)
    return {"success": True, "max_concurrent": config_req.max_concurrent}


@app.get("/api/v1/queue/tasks")
async def get_queue_tasks(limit: int = 50):
    """获取任务列表"""
    return queue_manager.get_tasks_list(limit)


@app.post("/api/v1/queue/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """取消任务"""
    # 从队列中移除
    success = queue_manager.cancel_task(task_id)

    # 从任务存储中移除
    if task_id in tasks:
        del tasks[task_id]

    if success:
        return {"success": True, "message": "任务已取消"}
    else:
        raise HTTPException(status_code=400, detail="任务无法取消（可能已在执行中）")


# ============== 缓存 API ==============

@app.get("/api/v1/cache/stats", response_model=CacheStats)
async def get_cache_stats():
    """获取缓存统计"""
    return get_result_cache().get_stats()


@app.post("/api/v1/cache/clear")
async def clear_cache():
    """清空缓存"""
    get_result_cache().clear()
    return {"success": True, "message": "缓存已清空"}


# ============== 监控页面 ==============

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """监控页面"""
    with open("templates/dashboard.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ============== 设备信息 API ==============

@app.get("/api/v1/device")
async def get_device_info():
    """获取设备信息（M4 Pro 优化）"""
    service = get_whisper_service()
    return service.get_device_info()


if __name__ == "__main__":
    import uvicorn

    cfg = get_config()
    uvicorn.run(
        "app:app",
        host=cfg.server.host,
        port=cfg.server.port,
        workers=cfg.server.workers,
        reload=cfg.server.reload,
    )
