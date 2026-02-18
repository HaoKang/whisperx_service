"""文件类型验证工具"""
import logging
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)

# 音频文件 Magic Numbers（文件头签名）
AUDIO_SIGNATURES = {
    b'\xff\xfb': 'mp3',      # MP3 (MPEG Audio Layer 3)
    b'\xff\xfa': 'mp3',      # MP3 (MPEG Audio Layer 3)
    b'\x49\x44\x33': 'mp3',  # MP3 (ID3 tag)
    b'RIFF': 'wav',          # WAV
    b'fLaC': 'flac',         # FLAC
    b'OggS': 'ogg',          # OGG
    b'\x1a\x45\xdf\xa3': 'webm',  # WebM/MKV
}

# M4A/AAC 的文件类型盒
M4A_BRANDS = {
    b'mp41', b'mp42', b'isom', b'M4A ', b'M4B ', b'M4P ',
    b'iso2', b'iso3', b'iso4', b'iso5', b'iso6',
}

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm'}


def detect_audio_type(file_path: str, read_bytes: int = 12) -> Optional[str]:
    """
    通过文件头检测音频类型

    Args:
        file_path: 文件路径
        read_bytes: 读取的字节数（默认 12 足够检测大多数格式）

    Returns:
        检测到的音频类型，如 'mp3', 'wav', 'm4a' 等
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(read_bytes)

        if len(header) < 4:
            return None

        # 检查 MP3
        if header[:2] in (b'\xff\xfb', b'\xff\xfa'):
            return 'mp3'
        if header[:3] == b'\x49\x44\x33':  # ID3 tag
            return 'mp3'

        # 检查 WAV
        if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
            return 'wav'

        # 检查 FLAC
        if header[:4] == b'fLaC':
            return 'flac'

        # 检查 OGG
        if header[:4] == b'OggS':
            return 'ogg'

        # 检查 WebM
        if header[:4] == b'\x1a\x45\xdf\xa3':
            return 'webm'

        # 检查 M4A（ISO Base Media File Format）
        # M4A 文件有 ftyp 盒，通常在偏移 4 处
        if len(header) >= 12:
            # 检查 ftyp 盒
            if header[4:8] == b'ftyp':
                brand = header[8:12]
                if brand in M4A_BRANDS:
                    return 'm4a'

        return None

    except Exception as e:
        logger.warning(f"Failed to detect file type: {e}")
        return None


def validate_audio_file(file_path: str, expected_ext: Optional[str] = None) -> tuple[bool, Optional[str]]:
    """
    验证音频文件

    Args:
        file_path: 文件路径
        expected_ext: 期望的扩展名（如 '.mp3'）

    Returns:
        (is_valid, detected_type) - 是否有效和检测到的类型
    """
    # 检查扩展名
    ext = Path(file_path).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, None

    # 检测实际文件类型
    detected = detect_audio_type(file_path)

    if detected is None:
        logger.warning(f"Could not detect file type for {file_path}")
        # 对于无法检测的类型，仍然允许（可能是某些特殊格式）
        return True, 'unknown'

    # 验证扩展名与实际类型是否匹配
    ext_to_type = {
        '.mp3': 'mp3',
        '.wav': 'wav',
        '.m4a': 'm4a',
        '.flac': 'flac',
        '.ogg': 'ogg',
        '.webm': 'webm',
    }

    expected_type = ext_to_type.get(ext)
    if expected_type and detected != expected_type:
        logger.warning(
            f"File extension mismatch: {ext} expected {expected_type}, "
            f"but detected {detected}"
        )
        # 仍然返回检测到的类型，让调用者决定如何处理

    return True, detected


def get_file_info(file_path: str) -> dict:
    """
    获取文件信息

    Returns:
        包含文件信息的字典
    """
    path = Path(file_path)
    is_valid, detected_type = validate_audio_file(file_path)

    return {
        'path': str(file_path),
        'name': path.name,
        'extension': path.suffix.lower(),
        'size': path.stat().st_size if path.exists() else 0,
        'is_valid': is_valid,
        'detected_type': detected_type,
    }
