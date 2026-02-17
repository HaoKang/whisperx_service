import os
import logging
import asyncio
import tempfile
from pathlib import Path
from typing import Optional
import aiofiles
import uuid

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
)
from whisper_service import get_whisper_service
from task_queue import get_queue_manager, TaskPriority as QueuePriority

# 初始化配置
config = init_config()

# 配置日志
logging.basicConfig(
    level=get_config().logging.level,
    format=get_config().logging.format,
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="WhisperX 语音识别服务",
    description="本地语音识别服务，基于 WhisperX",
    version="1.0.0",
)

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
        logger.info("Model loaded on startup")
    except Exception as e:
        logger.warning(f"Failed to load model on startup: {e}")

    # 启动队列处理循环
    asyncio.create_task(queue_processing_loop())


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    service = get_whisper_service()
    return HealthResponse(
        status="healthy",
        model_loaded=service._model is not None,
    )


@app.get("/api/v1/models", response_model=ModelInfo)
async def get_model_info():
    """获取当前模型信息"""
    service = get_whisper_service()
    config = get_config()
    return ModelInfo(
        name=config.whisper.model,
        device=service.device,
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

    # 验证优先级
    if priority not in [1, 2, 3]:
        priority = 2

    # 验证文件类型
    allowed_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}。支持的类型: {allowed_extensions}",
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

        # 创建任务记录
        tasks[task_id] = {
            "status": TaskStatus.PENDING,
            "file_path": str(file_path),
            "language": language,
            "diarize": diarize,
            "format": format,
            "model": model,
            "priority": priority,
        }

        # 加入优先级队列
        await queue_manager.enqueue(
            task_id=task_id,
            file_path=str(file_path),
            priority=QueuePriority(priority),
            language=language,
            diarize=diarize,
            format=format,
            model=model,
            callback=None
        )

        priority_label = {1: "高", 2: "中", 3: "低"}[priority]
        return RecognizeResponse(
            success=True,
            task_id=task_id,
            text=f"任务已加入队列，优先级: {priority_label}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_recognition(task_id: str):
    """处理识别任务"""
    task = tasks.get(task_id)
    if not task:
        return

    try:
        # 更新状态为处理中
        tasks[task_id]["status"] = TaskStatus.PROCESSING

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

        tasks[task_id] = {
            "status": TaskStatus.COMPLETED,
            "result": {
                "task_id": result.task_id,
                "text": text,
                "segments": [seg.model_dump() for seg in result.segments],
                "language": result.language,
                "duration": result.duration,
            },
        }

        # 标记队列任务完成
        queue_manager.complete_task(task_id)

        # 清理临时文件
        file_path = Path(task.get("file_path"))
        if file_path.exists():
            file_path.unlink()

    except Exception as e:
        logger.error(f"Recognition failed for task {task_id}: {e}")
        tasks[task_id] = {
            "status": TaskStatus.FAILED,
            "error": str(e),
        }
        # 标记队列任务失败
        queue_manager.fail_task(task_id)


async def queue_processing_loop():
    """队列处理循环"""
    logger.info("Queue processing loop started")

    while True:
        try:
            # 从队列获取任务
            task = await queue_manager.dequeue()

            logger.info(f"Processing task: {task.task_id}")

            # 在线程池中执行识别任务
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: asyncio.run(process_recognition(task.task_id))
            )

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
        return TaskStatusResponse(
            task_id=task_id,
            status=status,
            progress=0.5 if status == TaskStatus.PROCESSING else 0.0,
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


# 同步识别接口（简单场景使用）
@app.post("/api/v1/recognize/sync", response_model=RecognizeResponse)
async def recognize_speech_sync(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    diarize: bool = Form(False),
    format: OutputFormat = Form(OutputFormat.JSON),
    model: Optional[str] = Form(None),
):
    """同步语音识别接口（适合小文件）"""
    task_id = str(uuid.uuid4())

    # 验证文件类型
    allowed_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
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

        # 执行识别
        service = get_whisper_service()
        result = service.recognize(
            audio_path=str(file_path),
            language=language,
            diarize=diarize,
            model_name=model,
        )

        # 转换格式
        text = result.text
        if format == OutputFormat.SRT:
            text = service.generate_srt(result.segments)
        elif format == OutputFormat.VTT:
            text = service.generate_vtt(result.segments)
        elif format == OutputFormat.TEXT:
            text = service.generate_text(result.segments)

        return RecognizeResponse(
            success=True,
            task_id=task_id,
            text=text,
            segments=result.segments if format == OutputFormat.JSON else None,
            language=result.language,
            duration=result.duration,
        )

    except Exception as e:
        logger.error(f"Recognition failed: {e}")
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
async def update_queue_config(config: QueueConfigRequest):
    """更新队列配置"""
    queue_manager.update_max_concurrent(config.max_concurrent)
    return {"success": True, "max_concurrent": config.max_concurrent}


@app.get("/api/v1/queue/tasks")
async def get_queue_tasks(limit: int = 50):
    """获取任务列表"""
    return queue_manager.get_tasks_list(limit)


# ============== 监控页面 ==============

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """监控页面"""
    with open("templates/dashboard.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


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
