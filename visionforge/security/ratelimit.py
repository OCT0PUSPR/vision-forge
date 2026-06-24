"""In-process token-bucket rate limiter (thread-safe).

A pure-Python implementation so it works with zero external services and is
fully unit-testable. For multi-replica deployments swap in a Redis-backed
limiter (the interface is intentionally tiny); the in-process limiter is the
default and is correct for a single worker.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class _Bucket:
    tokens: float
    last: float
    capacity: float
    refill_per_sec: float
    lock: "threading.Lock" = field(default_factory=threading.Lock)


class TokenBucketRateLimiter:
    """Classic token-bucket keyed by an arbitrary identity string."""

    def __init__(
        self,
        rate_per_min: int = 120,
        burst: int = 0,
        time_func=time.monotonic,
    ) -> None:
        if rate_per_min <= 0:
            raise ValueError("rate_per_min must be positive")
        self.rate_per_min = rate_per_min
        self.refill_per_sec = rate_per_min / 60.0
        # Allow a small burst above the steady rate (defaults to one minute).
        self.capacity = float(burst) if burst > 0 else float(rate_per_min)
        self._time = time_func
        self._buckets: Dict[str, _Bucket] = {}
        self._global_lock = threading.Lock()

    def _get_bucket(self, key: str, rate_per_min: int) -> _Bucket:
        with self._global_lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                cap = float(rate_per_min)
                bucket = _Bucket(
                    tokens=cap,
                    last=self._time(),
                    capacity=cap,
                    refill_per_sec=rate_per_min / 60.0,
                )
                self._buckets[key] = bucket
            return bucket

    def check(self, key: str, rate_per_min: int = 0, cost: float = 1.0) -> Tuple[bool, float]:
        """Try to consume ``cost`` tokens for ``key``.

        Returns ``(allowed, retry_after_seconds)``. ``retry_after_seconds`` is
        0 when allowed.
        """
        effective = rate_per_min or self.rate_per_min
        bucket = self._get_bucket(key, effective)
        with bucket.lock:
            now = self._time()
            elapsed = max(0.0, now - bucket.last)
            bucket.tokens = min(bucket.capacity, bucket.tokens + elapsed * bucket.refill_per_sec)
            bucket.last = now
            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return True, 0.0
            needed = cost - bucket.tokens
            retry_after = needed / bucket.refill_per_sec if bucket.refill_per_sec else 60.0
            return False, retry_after

    def reset(self, key: str) -> None:
        with self._global_lock:
            self._buckets.pop(key, None)

    def clear(self) -> None:
        with self._global_lock:
            self._buckets.clear()
