"""Unit tests for the circuit breaker and retry helper."""

import pytest

from visionforge.reliability import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    retry_call,
)


def test_breaker_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
    assert cb.state == CircuitState.CLOSED
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.allow() is False


def test_breaker_half_opens_after_timeout():
    t = {"v": 0.0}
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5, time_func=lambda: t["v"])
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    t["v"] = 6.0  # past recovery timeout
    assert cb.state == CircuitState.HALF_OPEN
    assert cb.allow() is True


def test_breaker_closes_on_success():
    cb = CircuitBreaker(failure_threshold=2)
    cb.record_failure()
    cb.record_success()
    assert cb.state == CircuitState.CLOSED


def test_breaker_call_success_and_failure():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=100)
    assert cb.call(lambda: 42) == 42

    def boom():
        raise ValueError("x")

    with pytest.raises(ValueError):
        cb.call(boom)
    # now open -> rejects
    with pytest.raises(CircuitBreakerError):
        cb.call(lambda: 1)


def test_retry_call_eventually_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    out = retry_call(flaky, attempts=5, base_delay=0, sleep=lambda _s: None)
    assert out == "ok"
    assert calls["n"] == 3


def test_retry_call_exhausts_and_raises():
    def always_fail():
        raise RuntimeError("perma")

    with pytest.raises(RuntimeError):
        retry_call(always_fail, attempts=2, base_delay=0, sleep=lambda _s: None)
