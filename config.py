import os
from pathlib import Path
from typing import Optional, Any
import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False


class QueueConfig(BaseModel):
    max_concurrent: int = 2
    worker_threads: int = 2


class WhisperConfig(BaseModel):
    model: str = "base"
    device: str = "auto"
    compute_type: str = "float32"
    huggingface_token: Optional[str] = None
    diarize: bool = False
    language: Optional[str] = None
    batch_size: int = 16
    threads: int = 10


class StorageConfig(BaseModel):
    temp_dir: str = "./temp"
    max_file_size: int = 100


class CacheConfig(BaseModel):
    max_size: int = 100
    ttl: int = 6 * 3600  # 6 小时


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class Config(BaseModel):
    server: ServerConfig = ServerConfig()
    queue: QueueConfig = QueueConfig()
    whisper: WhisperConfig = WhisperConfig()
    storage: StorageConfig = StorageConfig()
    cache: CacheConfig = CacheConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """从配置文件加载配置"""
        if config_path is None:
            config_path = os.environ.get("CONFIG_PATH", "config.yaml")

        config_file = Path(config_path)

        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return cls(**data) if data else cls()

        # 如果配置文件不存在，使用默认配置
        return cls()


# 全局配置实例
_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def init_config(config_path: Optional[str] = None) -> Config:
    """初始化配置"""
    global _config
    _config = Config.load(config_path)

    # 确保临时目录存在
    storage = _config.storage
    Path(storage.temp_dir).mkdir(parents=True, exist_ok=True)

    return _config
