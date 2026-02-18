import hashlib
import time
import logging
from typing import Optional, Dict, Any
from collections import OrderedDict
from pathlib import Path

from config import get_config

logger = logging.getLogger(__name__)


class LRUCache:
    """高效的 LRU 缓存实现（基于 OrderedDict）"""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.cache: OrderedDict[str, tuple] = OrderedDict()  # key -> (value, expire_at)

    def get(self, key: str) -> Optional[Any]:
        """获取缓存（O(1) 复杂度）"""
        if key not in self.cache:
            return None

        value, expire_at = self.cache[key]

        # 检查过期
        if expire_at and time.time() > expire_at:
            del self.cache[key]
            return None

        # 移动到末尾表示最近访问（O(1)）
        self.cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl: int):
        """设置缓存（O(1) 复杂度）"""
        # 如果已存在，先删除
        if key in self.cache:
            del self.cache[key]

        # 如果容量超限，移除最旧的（OrderedDict 的第一个）
        while len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)

        expire_at = time.time() + ttl if ttl > 0 else None
        self.cache[key] = (value, expire_at)

    def clear(self):
        """清空缓存"""
        self.cache.clear()

    def __len__(self):
        return len(self.cache)


class ResultCache:
    """识别结果缓存"""

    def __init__(self, max_size: Optional[int] = None, ttl: Optional[int] = None):
        """
        Args:
            max_size: 最大缓存数量，默认从配置读取
            ttl: 缓存有效期（秒），默认从配置读取
        """
        config = get_config().cache
        self.cache = LRUCache(capacity=max_size or config.max_size)
        self.ttl = ttl or config.ttl
        self._hits = 0
        self._misses = 0
        self._lock = None  # 延迟初始化

    def _get_lock(self):
        """延迟初始化锁（避免在导入时创建）"""
        if self._lock is None:
            import threading
            self._lock = threading.Lock()
        return self._lock

    def compute_hash(self, file_path: str) -> str:
        """计算文件的 SHA256 哈希（取前16位）"""
        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            # 分块读取，避免大文件内存问题
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        return sha256.hexdigest()[:16]

    def get(self, file_path: str) -> Optional[Dict[str, Any]]:
        """获取缓存结果"""
        try:
            hash_key = self.compute_hash(file_path)

            with self._get_lock():
                result = self.cache.get(hash_key)

                if result is not None:
                    self._hits += 1
                    logger.info(f"Cache hit: {hash_key}")
                    return result
                else:
                    self._misses += 1
                    return None

        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
            return None

    def set(self, file_path: str, result: Dict[str, Any]):
        """缓存识别结果"""
        try:
            hash_key = self.compute_hash(file_path)
            with self._get_lock():
                self.cache.set(hash_key, result, self.ttl)
            logger.info(f"Cache set: {hash_key}")
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._get_lock():
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.1f}%",
                "size": len(self.cache),
                "max_size": self.cache.capacity,
                "ttl": self.ttl,
            }

    def clear(self):
        """清空缓存"""
        with self._get_lock():
            self.cache.clear()
            self._hits = 0
            self._misses = 0
        logger.info("Cache cleared")


# 全局缓存实例
_cache: Optional[ResultCache] = None


def get_result_cache() -> ResultCache:
    """获取缓存实例"""
    global _cache
    if _cache is None:
        _cache = ResultCache()
    return _cache
