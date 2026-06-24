"""API-key authentication.

Keys live in the database (``api_keys`` table) and/or a comma-separated
``VF_API_KEYS`` env var (handy for local dev / bootstrap). Only a SHA-256 hash
of each key is ever stored. The plaintext key is shown to the operator exactly
once at creation time.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import List, Optional, Set

API_KEY_HEADER = "X-API-Key"
_KEY_PREFIX = "vf_"


def generate_api_key(nbytes: int = 24) -> str:
    """Generate a new random API key string (URL-safe)."""
    return _KEY_PREFIX + secrets.token_urlsafe(nbytes)


def hash_key(key: str) -> str:
    """Return the hex SHA-256 of a key (what we persist)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def verify_key(key: str, stored_hash: str) -> bool:
    """Constant-time comparison of a presented key against a stored hash."""
    return hmac.compare_digest(hash_key(key), stored_hash)


def parse_env_keys(raw: Optional[str]) -> Set[str]:
    """Parse a comma/space separated env var into a set of key hashes.

    Accepts either plaintext keys (hashed here) for convenience.
    """
    if not raw:
        return set()
    keys = {k.strip() for k in raw.replace(" ", ",").split(",") if k.strip()}
    return {hash_key(k) for k in keys}


@dataclass
class ApiKeyContext:
    """Resolved identity for an authenticated request."""

    key_id: int
    name: str
    rate_limit_per_min: int = 120
    scopes: Optional[List[str]] = None

    def has_scope(self, scope: str) -> bool:
        return self.scopes is None or scope in self.scopes


class ApiKeyAuthenticator:
    """Validates presented keys against env hashes + a DB lookup callback.

    The DB lookup is injected so this class stays import-light and testable.
    ``db_lookup`` takes a key *hash* and returns an :class:`ApiKeyContext` or
    ``None``.
    """

    def __init__(
        self,
        env_key_hashes: Optional[Set[str]] = None,
        db_lookup=None,
        require_auth: bool = True,
    ) -> None:
        self.env_key_hashes = env_key_hashes or set()
        self.db_lookup = db_lookup
        self.require_auth = require_auth

    def authenticate(self, presented_key: Optional[str]) -> Optional[ApiKeyContext]:
        """Return a context for a valid key, or ``None`` when auth is disabled.

        Raises ``AuthError`` for a missing/invalid key when auth is required.
        """
        from visionforge.errors import AuthError

        if not self.require_auth and not presented_key:
            return None
        if not presented_key:
            raise AuthError("Missing API key", details={"header": API_KEY_HEADER})

        key_hash = hash_key(presented_key)

        if key_hash in self.env_key_hashes:
            return ApiKeyContext(key_id=0, name="env", rate_limit_per_min=120)

        if self.db_lookup is not None:
            ctx = self.db_lookup(key_hash)
            if ctx is not None:
                return ctx

        raise AuthError("Invalid API key")
