"""FastAPI dependencies + shared singletons for the API layer.

Builds the authenticator and rate limiter from settings, and exposes FastAPI
dependency callables for auth + per-key rate limiting that raise the structured
:class:`VisionForgeError` subclasses (turned into JSON by the exception handlers).
"""

from __future__ import annotations

from typing import Optional

from fastapi import Request

from visionforge.config import get_settings
from visionforge.errors import RateLimitError
from visionforge.security.auth import (
    API_KEY_HEADER,
    ApiKeyAuthenticator,
    ApiKeyContext,
    parse_env_keys,
)
from visionforge.security.ratelimit import TokenBucketRateLimiter

_authenticator: Optional[ApiKeyAuthenticator] = None
_limiter: Optional[TokenBucketRateLimiter] = None


def _db_lookup(key_hash: str) -> Optional[ApiKeyContext]:
    """Resolve an API key hash against the database, if one is configured."""
    from visionforge.db.session import get_db

    db = get_db()
    if db is None:
        return None
    try:
        from visionforge.db.repository import ApiKeyRepository

        with db.session() as session:
            row = ApiKeyRepository(session).get_by_hash(key_hash)
            if row is None:
                return None
            scopes = row.scopes.split(",") if row.scopes else None
            return ApiKeyContext(
                key_id=row.id,
                name=row.name,
                rate_limit_per_min=row.rate_limit_per_min,
                scopes=scopes,
            )
    except Exception:  # pragma: no cover - DB optional
        return None


def get_authenticator() -> ApiKeyAuthenticator:
    global _authenticator
    if _authenticator is None:
        settings = get_settings()
        _authenticator = ApiKeyAuthenticator(
            env_key_hashes=parse_env_keys(settings.api_keys),
            db_lookup=_db_lookup,
            require_auth=settings.require_auth,
        )
    return _authenticator


def get_limiter() -> TokenBucketRateLimiter:
    global _limiter
    if _limiter is None:
        settings = get_settings()
        _limiter = TokenBucketRateLimiter(rate_per_min=settings.rate_limit_per_min)
    return _limiter


def reset_singletons() -> None:
    """Test helper: rebuild authenticator + limiter from current settings."""
    global _authenticator, _limiter
    _authenticator = None
    _limiter = None


async def require_api_key(request: Request) -> Optional[ApiKeyContext]:
    """FastAPI dependency: authenticate + rate-limit the request.

    Returns the :class:`ApiKeyContext` (or ``None`` when auth is disabled and no
    key was presented). Raises ``AuthError`` / ``RateLimitError``.
    """
    presented = request.headers.get(API_KEY_HEADER)
    ctx = get_authenticator().authenticate(presented)

    # Rate-limit key: per API key when present, else per client IP.
    if ctx is not None:
        rl_key = f"key:{ctx.key_id}:{ctx.name}"
        rate = ctx.rate_limit_per_min
    else:
        client = request.client.host if request.client else "anonymous"
        rl_key = f"ip:{client}"
        rate = get_settings().rate_limit_per_min

    allowed, retry_after = get_limiter().check(rl_key, rate_per_min=rate)
    if not allowed:
        raise RateLimitError(
            "Rate limit exceeded",
            details={"retry_after": round(retry_after, 2), "limit_per_min": rate},
        )
    request.state.api_key = ctx
    return ctx
