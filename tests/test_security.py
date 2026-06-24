"""Unit tests for auth, validation and rate-limiting (pure logic)."""

import pytest

from visionforge.errors import (
    AuthError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
    ValidationError,
)
from visionforge.security.auth import (
    ApiKeyAuthenticator,
    ApiKeyContext,
    generate_api_key,
    hash_key,
    parse_env_keys,
    verify_key,
)
from visionforge.security.ratelimit import TokenBucketRateLimiter
from visionforge.security.validation import (
    sniff_image_mime,
    validate_content_type,
    validate_dimensions,
    validate_image_bytes,
    validate_size,
    validate_task,
    validate_threshold,
)

PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG_HEADER = b"\xff\xd8\xff\xe0" + b"\x00" * 32


# --- auth ---
def test_generate_and_verify_key():
    key = generate_api_key()
    assert key.startswith("vf_")
    h = hash_key(key)
    assert verify_key(key, h)
    assert not verify_key("vf_wrong", h)


def test_parse_env_keys_hashes():
    keys = parse_env_keys("alpha, beta")
    assert hash_key("alpha") in keys
    assert hash_key("beta") in keys
    assert len(keys) == 2
    assert parse_env_keys(None) == set()


def test_authenticator_env_key_accepted():
    auth = ApiKeyAuthenticator(env_key_hashes={hash_key("secret")}, require_auth=True)
    ctx = auth.authenticate("secret")
    assert ctx is not None
    assert ctx.name == "env"


def test_authenticator_missing_key_raises():
    auth = ApiKeyAuthenticator(require_auth=True)
    with pytest.raises(AuthError):
        auth.authenticate(None)


def test_authenticator_invalid_key_raises():
    auth = ApiKeyAuthenticator(env_key_hashes={hash_key("good")}, require_auth=True)
    with pytest.raises(AuthError):
        auth.authenticate("bad")


def test_authenticator_disabled_returns_none():
    auth = ApiKeyAuthenticator(require_auth=False)
    assert auth.authenticate(None) is None


def test_authenticator_db_lookup():
    def lookup(key_hash):
        if key_hash == hash_key("dbkey"):
            return ApiKeyContext(key_id=5, name="db", rate_limit_per_min=10)
        return None

    auth = ApiKeyAuthenticator(db_lookup=lookup, require_auth=True)
    ctx = auth.authenticate("dbkey")
    assert ctx.key_id == 5
    assert ctx.rate_limit_per_min == 10


# --- rate limiting ---
def test_rate_limiter_blocks_after_capacity():
    t = {"v": 0.0}
    rl = TokenBucketRateLimiter(rate_per_min=60, time_func=lambda: t["v"])
    # capacity defaults to rate (60). Consume them all.
    for _ in range(60):
        allowed, _ = rl.check("k")
        assert allowed
    allowed, retry_after = rl.check("k")
    assert not allowed
    assert retry_after > 0


def test_rate_limiter_refills_over_time():
    t = {"v": 0.0}
    rl = TokenBucketRateLimiter(rate_per_min=60, time_func=lambda: t["v"])
    for _ in range(60):
        rl.check("k")
    assert rl.check("k")[0] is False
    # 1 second later -> ~1 token refilled (60/min = 1/s)
    t["v"] = 1.0
    assert rl.check("k")[0] is True


def test_rate_limiter_per_key_isolation():
    rl = TokenBucketRateLimiter(rate_per_min=1)
    assert rl.check("a")[0] is True
    assert rl.check("a")[0] is False
    assert rl.check("b")[0] is True  # different key has its own bucket


def test_rate_limiter_invalid_rate():
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(rate_per_min=0)


# --- validation ---
def test_validate_size_ok_and_too_large():
    validate_size(100, max_mb=1)
    with pytest.raises(PayloadTooLargeError):
        validate_size(2 * 1024 * 1024, max_mb=1)
    with pytest.raises(ValidationError):
        validate_size(0, max_mb=1)


def test_validate_content_type():
    assert validate_content_type("image/png") == "image/png"
    assert validate_content_type("image/jpeg; charset=binary") == "image/jpeg"
    with pytest.raises(UnsupportedMediaTypeError):
        validate_content_type("application/pdf")
    with pytest.raises(UnsupportedMediaTypeError):
        validate_content_type(None)


def test_sniff_image_mime():
    assert sniff_image_mime(PNG_HEADER) == "image/png"
    assert sniff_image_mime(JPEG_HEADER) == "image/jpeg"
    assert sniff_image_mime(b"not an image") is None
    assert sniff_image_mime(b"") is None


def test_validate_image_bytes_full_gauntlet():
    assert validate_image_bytes(PNG_HEADER, max_mb=1) == "image/png"
    with pytest.raises(UnsupportedMediaTypeError):
        validate_image_bytes(b"garbage", max_mb=1)
    with pytest.raises(PayloadTooLargeError):
        validate_image_bytes(PNG_HEADER * 100000, max_mb=0.0001)


def test_validate_dimensions():
    assert validate_dimensions(640, 480) == (640, 480)
    with pytest.raises(ValidationError):
        validate_dimensions(0, 100)
    with pytest.raises(ValidationError):
        validate_dimensions(99999, 99999, max_side=8000)
    with pytest.raises(ValidationError):
        validate_dimensions(7000, 7000, max_pixels=1000)


def test_validate_task():
    assert validate_task("DETECTION", ["detection"]) == "detection"
    assert validate_task(None, ["detection"]) == "detection"
    with pytest.raises(ValidationError):
        validate_task("nope", ["detection"])


def test_validate_threshold():
    assert validate_threshold(0.5, "conf") == 0.5
    assert validate_threshold(None, "conf") is None
    with pytest.raises(ValidationError):
        validate_threshold(1.5, "conf")
    with pytest.raises(ValidationError):
        validate_threshold("abc", "conf")
