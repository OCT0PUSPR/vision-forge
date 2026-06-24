"""Reliability primitives: circuit breaker + retry helpers.

Pure-Python and fully unit-testable. The retry helper uses ``tenacity`` when
available (for jittered exponential backoff) and otherwise falls back to a
simple built-in retry loop, so model-download/HF calls are resilient without a
hard dependency.
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"  # healthy, calls flow through
    OPEN = "open"  # failing, calls short-circuit
    HALF_OPEN = "half_open"  # probing whether the dependency recovered


class CircuitBreakerError(RuntimeError):
    """Raised when a call is rejected because the circuit is open."""


class CircuitBreaker:
    """Thread-safe circuit breaker.

    Opens after ``failure_threshold`` consecutive failures; after
    ``recovery_timeout`` seconds it moves to half-open and allows a single trial
    call. A success closes it; a failure re-opens it.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        time_func: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._time = time_func
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self) -> None:
        if self._state == CircuitState.OPEN and self._time() - self._opened_at >= self.recovery_timeout:
            self._state = CircuitState.HALF_OPEN

    def allow(self) -> bool:
        """Return True if a call may proceed right now."""
        with self._lock:
            self._maybe_half_open()
            return self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == CircuitState.HALF_OPEN or self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = self._time()

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute ``func`` through the breaker."""
        if not self.allow():
            raise CircuitBreakerError("Circuit is open")
        try:
            result = func(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result


def retry_call(
    func: Callable[..., T],
    *args,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    exceptions: tuple = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
    **kwargs,
) -> T:
    """Call ``func`` with exponential backoff retries.

    Uses tenacity when available for jittered backoff; otherwise a deterministic
    exponential loop. Re-raises the last exception when attempts are exhausted.
    """
    try:
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        @retry(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=base_delay, max=max_delay),
            retry=retry_if_exception_type(exceptions),
            reraise=True,
        )
        def _wrapped():
            return func(*args, **kwargs)

        return _wrapped()
    except ImportError:  # pragma: no cover - fallback path
        last_exc: Optional[BaseException] = None
        for i in range(attempts):
            try:
                return func(*args, **kwargs)
            except exceptions as exc:  # type: ignore[misc]
                last_exc = exc
                if i < attempts - 1:
                    delay = min(max_delay, base_delay * (2**i))
                    sleep(delay)
        assert last_exc is not None
        raise last_exc
