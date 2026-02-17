import os
import uuid
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

import whisperx
import torch
from whisperx.diarize import DiarizationPipeline

from config import get_config
from models import Segment, RecognitionResult

logger = logging.getLogger(__name__)


class WhisperService:
    def __init__(self):
        self.config = get_config().whisper
        self._model = None
        self._diarize_model = None
        self._device = None
        self._model_type = None
        self._compute_type = getattr(self.config, 'compute_type', 'float32')
        self._threads = getattr(self.config, 'threads', 10)

        # 设置线程数优化性能
        os.environ["OMP_NUM_THREADS"] = str(self._threads)
        os.environ["MKL_NUM_THREADS"] = str(self._threads)

    @property
    def device(self):
        if self._device is None:
            if self.config.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.config.device
        return self._device

    def load_model(self, model_name: Optional[str] = None):
        """加载 WhisperX 模型"""
        model_name = model_name or self.config.model
        if self._model is not None and self._model_type == model_name:
            return self._model

        logger.info(f"Loading WhisperX model: {model_name} on {self.device}")

        # 根据配置选择计算类型
        compute_type = self._compute_type

        self._model = whisperx.load_model(
            model_name,
            self.device,
            compute_type=compute_type,
            threads=self._threads
        )
        self._model_type = model_name
        logger.info(f"Model loaded successfully with compute_type={compute_type}")
        return self._model

    def load_diarize_model(self):
        """加载说话人分离模型"""
        if self._diarize_model is not None:
            return self._diarize_model

        logger.info("Loading diarization model...")
        token = self.config.huggingface_token
        if not token:
            raise ValueError("HuggingFace token is required for diarization")

        self._diarize_model = DiarizationPipeline(token=token, device=self.device)
        logger.info("Diarization model loaded")
        return self._diarize_model

        logger.info("Loading diarization model...")
        self._diarize_model = whisperx.load_align_model(language_code="zh")
        logger.info("Diarization model loaded")
        return self._diarize_model

    def recognize(
        self,
        audio_path: str,
        language: Optional[str] = None,
        diarize: bool = False,
        model_name: Optional[str] = None,
    ) -> RecognitionResult:
        """执行语音识别"""
        # 保存临时文件
        temp_dir = Path(get_config().storage.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        task_id = str(uuid.uuid4())

        try:
            # 加载模型
            model = self.load_model(model_name)

            # 识别语言
            language = language or self.config.language

            # 1. 语音识别
            logger.info(f"Starting recognition for task {task_id}")
            result = model.transcribe(audio_path, language=language)

            # 2. 对齐音频
            logger.info("Aligning audio...")
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=self.device
            )
            result = whisperx.align(
                result["segments"], model_a, metadata, audio_path, self.device
            )

            # 3. 说话人分离（如果启用）
            if diarize:
                logger.info("Performing diarization...")
                try:
                    # 加载音频
                    audio = whisperx.load_audio(audio_path)
                    # 使用 DiarizationPipeline
                    diarize_model = self.load_diarize_model()
                    diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=10)

                    # 将说话人信息添加到段落
                    result = whisperx.assign_word_speakers(
                        diarize_segments, result
                    )
                except Exception as e:
                    logger.warning(f"Diarization failed: {e}")

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

            return RecognitionResult(
                task_id=task_id,
                text=text,
                segments=segments,
                language=result.get("language"),
                duration=result.get("duration"),
            )

        except Exception as e:
            logger.error(f"Recognition failed: {e}")
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


# 全局服务实例
_service: Optional[WhisperService] = None


def get_whisper_service() -> WhisperService:
    """获取 WhisperX 服务实例"""
    global _service
    if _service is None:
        _service = WhisperService()
    return _service
