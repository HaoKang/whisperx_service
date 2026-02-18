"""简单的请求限流器"""
import time
import threading
from collections import defaultdict
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    基于 IP 的滑动窗口限流器

    使用滑动窗口算法，比固定窗口更精确
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour

        # 存储: {ip: [(timestamp, path), ...]}
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()

    def _cleanup_old_requests(self, ip: str, now: float):
        """清理过期的请求记录"""
        hour_ago = now - 3600
        self._requests[ip] = [
            (ts, path) for ts, path in self._requests[ip]
            if ts > hour_ago
        ]

    def is_allowed(self, ip: str, path: str = "") -> Tuple[bool, Dict[str, int]]:
        """
        检查请求是否允许

        Args:
            ip: 客户端 IP
            path: 请求路径

        Returns:
            (is_allowed, info) - 是否允许和当前统计信息
        """
        now = time.time()

        with self._lock:
            # 清理过期记录
            self._cleanup_old_requests(ip, now)

            # 统计最近一分钟和一小时的请求数
            minute_ago = now - 60
            hour_ago = now - 3600

            minute_count = sum(
                1 for ts, _ in self._requests[ip] if ts > minute_ago
            )
            hour_count = len(self._requests[ip])

            # 检查是否超限
            if minute_count >= self.requests_per_minute:
                logger.warning(f"Rate limit exceeded for {ip}: {minute_count}/min")
                return False, {
                    'minute_count': minute_count,
                    'hour_count': hour_count,
                    'limit_per_minute': self.requests_per_minute,
                    'limit_per_hour': self.requests_per_hour,
                    'retry_after': 60 - int(now - minute_ago),
                }

            if hour_count >= self.requests_per_hour:
                logger.warning(f"Hourly rate limit exceeded for {ip}: {hour_count}/hour")
                return False, {
                    'minute_count': minute_count,
                    'hour_count': hour_count,
                    'limit_per_minute': self.requests_per_minute,
                    'limit_per_hour': self.requests_per_hour,
                    'retry_after': 3600 - int(now - hour_ago),
                }

            # 记录请求
            self._requests[ip].append((now, path))

            return True, {
                'minute_count': minute_count + 1,
                'hour_count': hour_count + 1,
                'limit_per_minute': self.requests_per_minute,
                'limit_per_hour': self.requests_per_hour,
            }

    def get_stats(self) -> Dict:
        """获取限流统计"""
        with self._lock:
            total_ips = len(self._requests)
            total_requests = sum(len(reqs) for reqs in self._requests.values())
            return {
                'tracked_ips': total_ips,
                'total_requests': total_requests,
                'limits': {
                    'per_minute': self.requests_per_minute,
                    'per_hour': self.requests_per_hour,
                }
            }


# 全局限流器
_rate_limiter: RateLimiter = None


def get_rate_limiter() -> RateLimiter:
    """获取全局限流器"""
    global _rate_limiter
    if _rate_limiter is None:
        from config import get_config
        config = get_config()
        _rate_limiter = RateLimiter(
            requests_per_minute=getattr(config, 'rate_limit_per_minute', 60),
            requests_per_hour=getattr(config, 'rate_limit_per_hour', 1000),
        )
    return _rate_limiter
