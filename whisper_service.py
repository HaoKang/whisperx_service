import os
import uuid
import logging
import platform
from pathlib import Path
from typing import Optional, List

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

from config import get_config
from models import Segment, RecognitionResult

logger = logging.getLogger(__name__)


class WhisperService:
    """WhisperX 服务（针对 Apple Silicon 优化）"""

    def __init__(self):
        self.config = get_config().whisper
        self._model = None
        self._diarize_model = None
        self._align_model = None
        self._align_metadata = None
        self._align_language = None
        self._device = None
        self._model_type = None

        # 检测设备类型
        self._is_apple_silicon = self._detect_apple_silicon()

        # 使用配置文件中的计算类型，默认 int8（CPU 优化）
        self._compute_type = getattr(self.config, 'compute_type', 'int8')
        self._threads = getattr(self.config, 'threads', 8)

        # 设置线程数优化性能
        os.environ["OMP_NUM_THREADS"] = str(self._threads)
        os.environ["MKL_NUM_THREADS"] = str(self._threads)

        logger.info(f"WhisperService initialized: Apple Silicon={self._is_apple_silicon}, "
                    f"compute_type={self._compute_type}, threads={self._threads}")

    def _detect_apple_silicon(self) -> bool:
        """检测是否为 Apple Silicon"""
        return (
            platform.system() == "Darwin" and
            platform.machine() == "arm64" and
            torch.backends.mps.is_available()
        )

    @property
    def device(self):
        """获取运行设备（暂时禁用 MPS，等待 WhisperX 完整支持）"""
        if self._device is None:
            if self.config.device == "auto":
                # MPS 支持尚不完整，暂时使用 CPU
                # TODO: 等 WhisperX MPS 支持完善后启用
                if torch.cuda.is_available():
                    self._device = "cuda"
                    logger.info("Using CUDA for GPU acceleration")
                else:
                    self._device = "cpu"
                    logger.info("Using CPU for inference")
            else:
                self._device = self.config.device
        return self._device

    def load_model(self, model_name: Optional[str] = None):
        """加载 WhisperX 模型"""
        model_name = model_name or self.config.model
        if self._model is not None and self._model_type == model_name:
            return self._model

        logger.info(f"Loading WhisperX model: {model_name} on {self.device}")

        # M4 Pro 优化配置
        if self._is_apple_silicon and self.device == "mps":
            # MPS 后端使用 float16
            compute_type = "float16"
            # M4 Pro 统一内存，可以使用更大的 batch size
            batch_size = getattr(self.config, 'batch_size', 16)
        else:
            compute_type = self._compute_type
            batch_size = getattr(self.config, 'batch_size', 8)

        self._model = whisperx.load_model(
            model_name,
            self.device,
            compute_type=compute_type,
            threads=self._threads,
            language=self.config.language,
        )
        self._model_type = model_name
        logger.info(f"Model loaded: compute_type={compute_type}, batch_size={batch_size}")
        return self._model

    def load_diarize_model(self):
        """加载说话人分离模型"""
        if self._diarize_model is not None:
            return self._diarize_model

        logger.info("Loading diarization model...")
        token = self.config.huggingface_token
        if not token:
            raise ValueError("HuggingFace token is required for diarization")

        # M4 Pro 使用 MPS
        device = self.device
        self._diarize_model = DiarizationPipeline(token=token, device=device)
        logger.info(f"Diarization model loaded on {device}")
        return self._diarize_model

    def load_align_model(self, language: str):
        """加载或获取对齐模型（带缓存）"""
        # 如果已有同语言的对齐模型，直接返回
        if self._align_model is not None and self._align_language == language:
            return self._align_model, self._align_metadata

        logger.info(f"Loading align model for language: {language}")
        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code=language,
            device=self.device
        )
        self._align_language = language
        logger.info("Align model loaded and cached")
        return self._align_model, self._align_metadata

    def recognize(
        self,
        audio_path: str,
        language: Optional[str] = None,
        diarize: bool = False,
        model_name: Optional[str] = None,
    ) -> RecognitionResult:
        """执行语音识别"""
        # 确保临时目录存在
        temp_dir = Path(get_config().storage.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        task_id = str(uuid.uuid4())

        try:
            # 加载模型
            model = self.load_model(model_name)

            # 识别语言
            language = language or self.config.language

            # 1. 语音识别
            logger.info(f"[{task_id}] Starting recognition, language={language}")
            result = model.transcribe(audio_path, language=language)

            detected_language = result.get("language", language)
            logger.info(f"[{task_id}] Detected language: {detected_language}")

            # 2. 对齐音频（使用缓存的对齐模型）
            logger.info(f"[{task_id}] Aligning audio...")
            model_a, metadata = self.load_align_model(detected_language)
            result = whisperx.align(
                result["segments"], model_a, metadata, audio_path, self.device
            )

            # 3. 说话人分离（如果启用）
            if diarize:
                logger.info(f"[{task_id}] Performing diarization...")
                try:
                    # 加载音频
                    audio = whisperx.load_audio(audio_path)
                    # 使用 DiarizationPipeline
                    diarize_model = self.load_diarize_model()
                    diarize_segments = diarize_model(
                        audio,
                        min_speakers=1,
                        max_speakers=10
                    )

                    # 将说话人信息添加到段落
                    result = whisperx.assign_word_speakers(
                        diarize_segments, result
                    )
                except Exception as e:
                    logger.warning(f"[{task_id}] Diarization failed: {e}")

            # 4. 转换为结果对象
            segments = []
            for seg in result.get("segments", []):
                segments.append(
                    Segment(
                        start=seg.get("start", 0.0),
                        end=seg.get("end", 0.0),
                        text=seg.get("text", "").strip(),
                        speaker=seg.get("speaker", None),
                    )
                )

            text = " ".join([seg.text for seg in segments])
            logger.info(f"[{task_id}] Recognition completed: {len(segments)} segments")

            return RecognitionResult(
                task_id=task_id,
                text=text,
                segments=segments,
                language=detected_language,
                duration=result.get("duration"),
            )

        except Exception as e:
            logger.error(f"[{task_id}] Recognition failed: {e}")
            raise

    def generate_srt(self, segments: List[Segment]) -> str:
        """生成 SRT 字幕"""
        srt_lines = []
        for i, seg in enumerate(segments, 1):
            start_time = self._format_srt_time(seg.start)
            end_time = self._format_srt_time(seg.end)
            speaker = f"[{seg.speaker}] " if seg.speaker else ""
            srt_lines.append(f"{i}\n{start_time} --> {end_time}\n{speaker}{seg.text}\n")
        return "\n".join(srt_lines)

    def generate_vtt(self, segments: List[Segment]) -> str:
        """生成 VTT 字幕"""
        vtt_lines = ["WEBVTT", ""]
        for seg in segments:
            start_time = self._format_vtt_time(seg.start)
            end_time = self._format_vtt_time(seg.end)
            speaker = f"[{seg.speaker}] " if seg.speaker else ""
            vtt_lines.append(f"{start_time} --> {end_time}\n{speaker}{seg.text}\n")
        return "\n".join(vtt_lines)

    def generate_text(self, segments: List[Segment]) -> str:
        """生成纯文本"""
        return " ".join([seg.text for seg in segments])

    def _format_srt_time(self, seconds: float) -> str:
        """格式化 SRT 时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _format_vtt_time(self, seconds: float) -> str:
        """格式化 VTT 时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def get_device_info(self) -> dict:
        """获取设备信息"""
        return {
            "device": self.device,
            "is_apple_silicon": self._is_apple_silicon,
            "compute_type": self._compute_type,
            "threads": self._threads,
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        }


# 全局服务实例
_service: Optional[WhisperService] = None


def get_whisper_service() -> WhisperService:
    """获取 WhisperX 服务实例"""
    global _service
    if _service is None:
        _service = WhisperService()
    return _service
