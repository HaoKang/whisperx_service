from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class OutputFormat(str, Enum):
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"
    TEXT = "text"


class TaskPriority(int, Enum):
    HIGH = 1    # 高优先级（对话语音，实时）
    MEDIUM = 2  # 中优先级（普通任务）
    LOW = 3     # 低优先级（批处理，定时任务）


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: Optional[List[Dict[str, Any]]] = None


class RecognitionResult(BaseModel):
    task_id: str
    text: str
    segments: List[Segment] = []
    language: Optional[str] = None
    duration: Optional[float] = None


class RecognizeRequest(BaseModel):
    language: Optional[str] = None
    diarize: bool = False
    format: OutputFormat = OutputFormat.JSON
    model: Optional[str] = None


class RecognizeResponse(BaseModel):
    success: bool
    task_id: Optional[str] = None
    text: Optional[str] = None
    segments: Optional[List[Segment]] = None
    language: Optional[str] = None
    duration: Optional[float] = None
    error: Optional[str] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    result: Optional[RecognitionResult] = None
    error: Optional[str] = None
    progress: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    model_loaded: bool = False


class ModelInfo(BaseModel):
    name: str
    device: str
    diarize_available: bool = True


class QueueStats(BaseModel):
    max_concurrent: int
    worker_threads: int
    running: int
    pending: int
    completed: int
    total: int
    avg_processing_time: float = 0.0


class QueueConfigRequest(BaseModel):
    max_concurrent: int = Field(ge=1, le=10)


class TaskListItem(BaseModel):
    task_id: str
    priority: int
    priority_label: str
    status: str
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration: Optional[float] = None


class CacheStats(BaseModel):
    hits: int
    misses: int
    hit_rate: str
    size: int
    max_size: int
    ttl: int
