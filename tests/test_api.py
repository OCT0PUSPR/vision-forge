"""Integration tests for the FastAPI service via TestClient.

Covers health/ready/metrics/models, auth success+failure, rate-limit 429,
validation 422/413/415, the job lifecycle, the WebSocket protocol, and a real
CPU inference smoke test (skipped if ultralytics is unavailable).
"""

import importlib
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart", reason="python-multipart required for form uploads")
from fastapi.testclient import TestClient  # noqa: E402

_HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None


def _build_client(**env) -> TestClient:
    """Build a fresh app/TestClient with the given env overrides applied."""
    for k, v in env.items():
        os.environ[k] = v
    from visionforge.config import get_settings

    get_settings.cache_clear()
    # Reload modules whose module-level singletons read settings at import.
    import visionforge.api.deps as deps
    import visionforge.api.server as server_mod

    deps.reset_singletons()
    importlib.reload(server_mod)
    return TestClient(server_mod.app)


@pytest.fixture()
def open_client():
    """Auth disabled, generous rate limit."""
    client = _build_client(
        VF_REQUIRE_AUTH="false",
        VF_RATE_LIMIT_PER_MIN="1000",
        VF_API_KEYS="",
        VF_ENV="development",
        VF_CORS_ORIGINS="*",
    )
    with client:
        yield client


# --- health / ready / metrics / models ---
def test_health(open_client):
    r = open_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    # request id echoed back
    assert "X-Request-ID" in r.headers


def test_ready(open_client):
    r = open_client.get("/ready")
    assert r.status_code in (200, 503)
    assert r.json()["status"] in ("ready", "not_ready")


def test_metrics(open_client):
    open_client.get("/health")  # generate a metric
    r = open_client.get("/metrics")
    assert r.status_code == 200
    assert "vf_requests_total" in r.text


def test_models(open_client):
    r = open_client.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert "detection" in body["tasks"]
    assert "onnx" in body["backend_names"]


def test_security_headers_present(open_client):
    r = open_client.get("/health")
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert "Content-Security-Policy" in r.headers


# --- validation ---
def test_infer_rejects_non_image(open_client):
    r = open_client.post(
        "/infer",
        files={"file": ("x.txt", b"not an image", "text/plain")},
        data={"task": "detection"},
    )
    assert r.status_code in (415, 422)
    assert "error" in r.json()


def test_infer_rejects_bad_task(open_client, png_bytes=None):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, format="PNG")
    r = open_client.post(
        "/infer",
        files={"file": ("a.png", buf.getvalue(), "image/png")},
        data={"task": "teleport"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


# --- auth ---
def test_auth_required_rejects_missing_key():
    client = _build_client(
        VF_REQUIRE_AUTH="true",
        VF_API_KEYS="topsecret",
        VF_RATE_LIMIT_PER_MIN="1000",
        VF_ENV="development",
    )
    with client:
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (16, 16)).save(buf, format="PNG")
        r = client.post(
            "/infer",
            files={"file": ("a.png", buf.getvalue(), "image/png")},
            data={"task": "detection"},
        )
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "unauthorized"


def test_auth_accepts_valid_key():
    client = _build_client(
        VF_REQUIRE_AUTH="true",
        VF_API_KEYS="topsecret",
        VF_RATE_LIMIT_PER_MIN="1000",
        VF_ENV="development",
    )
    with client:
        # /models has no auth dep, but /infer does; check the protected route.
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (16, 16)).save(buf, format="PNG")
        r2 = client.post(
            "/infer",
            files={"file": ("a.png", buf.getvalue(), "image/png")},
            data={"task": "detection", "backend": "onnx"},
            headers={"X-API-Key": "topsecret"},
        )
        # auth passes; inference may fail (no onnx weights) but NOT with 401
        assert r2.status_code != 401


# --- rate limiting ---
def test_rate_limit_returns_429():
    client = _build_client(
        VF_REQUIRE_AUTH="false",
        VF_RATE_LIMIT_PER_MIN="3",
        VF_ENV="development",
    )
    with client:
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (16, 16)).save(buf, format="PNG")
        payload = dict(
            files={"file": ("a.png", buf.getvalue(), "image/png")},
            data={"task": "detection", "backend": "onnx"},
        )
        statuses = []
        for _ in range(6):
            statuses.append(client.post("/infer", **payload).status_code)
        assert 429 in statuses


# --- jobs ---
def test_job_submit_and_status(open_client):
    r = open_client.post("/jobs", data={"source": "demo", "task": "detection", "max_frames": "2"})
    assert r.status_code == 202
    job = r.json()["job"]
    assert job["status"] in ("pending", "running", "succeeded")
    jid = job["id"]
    s = open_client.get(f"/jobs/{jid}")
    assert s.status_code == 200
    assert s.json()["job"]["id"] == jid


def test_job_not_found(open_client):
    r = open_client.get("/jobs/does-not-exist")
    assert r.status_code == 404


def test_job_submit_rejects_bad_source(open_client):
    r = open_client.post(
        "/jobs",
        data={"source": "/no/such/path/does-not-exist.mp4", "task": "detection"},
    )
    assert r.status_code == 422


# --- websocket ---
def test_websocket_ping_pong(open_client):
    with open_client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "ping"})
        msg = ws.receive_json()
        assert msg["type"] == "pong"


def test_websocket_bad_frame(open_client):
    with open_client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "frame", "task": "detection", "image": "not-base64!!!"})
        msg = ws.receive_json()
        assert msg["type"] == "error"


# --- real CPU inference smoke test ---
@pytest.mark.skipif(not _HAS_ULTRALYTICS, reason="ultralytics not installed")
def test_real_cpu_inference(open_client):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (320, 240), color=(50, 80, 120)).save(buf, format="JPEG")
    # Exercise the real-inference path via the 'baseline' (Ultralytics YOLO)
    # backend; the default 'centernet' detector needs a trained checkpoint that
    # is not committed, so the baseline backend is the right target here.
    r = open_client.post(
        "/infer",
        files={"file": ("a.jpg", buf.getvalue(), "image/jpeg")},
        data={"task": "detection", "backend": "baseline", "annotate": "true"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "result" in body
    assert body["result"]["task"] == "detection"
    assert "annotated" in body and body["annotated"].startswith("data:image")
