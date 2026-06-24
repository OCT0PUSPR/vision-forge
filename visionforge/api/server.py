"""Production FastAPI application.

Endpoints
---------
* ``GET  /``             -> web GUI
* ``GET  /health``       -> liveness probe (always cheap)
* ``GET  /ready``        -> readiness probe (model loaded, deps ok)
* ``GET  /metrics``      -> Prometheus exposition
* ``GET  /models``       -> available task/backend combinations
* ``POST /infer``        -> single-image inference (auth + rate limited)
* ``POST /infer/batch``  -> multi-image batched inference
* ``POST /jobs``         -> submit an async video/batch job
* ``GET  /jobs/{id}``    -> poll job status/progress/results
* ``DELETE /jobs/{id}``  -> cancel a job
* ``WS   /ws/stream``    -> live frame streaming

Auth: send ``X-API-Key`` (required when ``VF_REQUIRE_AUTH=true``). All errors are
returned as a structured JSON envelope ``{"error": {code, message, details}}``.
The interactive OpenAPI docs are served at ``/docs``.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from visionforge import __version__
from visionforge.api.deps import require_api_key
from visionforge.api.encoding import (
    bytes_to_rgb_array,
    decode_base64,
    encode_data_url,
    rgb_array_to_bytes,
)
from visionforge.api.middleware import (
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from visionforge.config import get_settings
from visionforge.errors import (
    InferenceError,
    ServiceUnavailableError,
    ValidationError,
    VisionForgeError,
)
from visionforge.models.registry import VALID_BACKENDS, VALID_TASKS
from visionforge.observability.logging import configure_logging, get_logger
from visionforge.observability.metrics import CONTENT_TYPE_LATEST, get_metrics
from visionforge.security.validation import (
    validate_dimensions,
    validate_image_bytes,
    validate_task,
)

WEB_DIR = Path(__file__).parent / "web"
_settings = get_settings()
configure_logging(level=_settings.log_level, json_logs=_settings.json_logs)
log = get_logger("visionforge.api.server")


# --------------------------------------------------------------------------- #
# lifespan: startup warmup + graceful shutdown
# --------------------------------------------------------------------------- #
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        settings.validate_startup()
    except Exception as exc:
        log.error("invalid_config", error=str(exc))
        raise
    # Initialize DB (best-effort; service can run stateless).
    with contextlib.suppress(Exception):
        from visionforge.db.session import init_db

        init_db()
        log.info("db_initialized", url=settings.database_url.split("://")[0])
    # Warm up models in a thread so startup is non-blocking-ish.
    from visionforge.models.manager import get_model_manager

    manager = get_model_manager()
    with contextlib.suppress(Exception):
        import anyio

        await anyio.to_thread.run_sync(manager.warmup, settings.warmup_task_list)
    log.info("startup_complete", warmed=list(manager.warmed_tasks))
    try:
        yield
    finally:
        # Graceful shutdown: release models + job threads, and reset the
        # singletons so a subsequent app instance (e.g. in tests) is clean.
        with contextlib.suppress(Exception):
            from visionforge.worker.jobs import reset_job_manager

            reset_job_manager()
        with contextlib.suppress(Exception):
            from visionforge.models.manager import reset_model_manager

            reset_model_manager()
        log.info("shutdown_complete")


app = FastAPI(
    title="vision-forge",
    version=__version__,
    description="Production-grade, real-time, multi-task computer vision platform.",
    lifespan=lifespan,
)

# --- middleware (order matters: last added runs first) ---
if _settings.enable_security_headers:
    app.add_middleware(SecurityHeadersMiddleware, enable_hsts=_settings.enable_hsts)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origin_list or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


# --------------------------------------------------------------------------- #
# exception handlers -> structured JSON
# --------------------------------------------------------------------------- #
def _request_id(request: Request) -> Optional[str]:
    return getattr(request.state, "request_id", None)


@app.exception_handler(VisionForgeError)
async def _handle_vf_error(request: Request, exc: VisionForgeError):
    get_metrics().errors_total.labels(type=exc.code).inc()
    return JSONResponse(
        status_code=exc.status,
        content=exc.to_dict(request_id=_request_id(request)),
    )


@app.exception_handler(Exception)
async def _handle_unexpected(request: Request, exc: Exception):
    get_metrics().errors_total.labels(type="internal_error").inc()
    log.exception("unhandled_exception", path=request.url.path)
    body = {
        "error": {
            "code": "internal_error",
            "message": "An unexpected error occurred.",
            "details": {},
        }
    }
    rid = _request_id(request)
    if rid:
        body["error"]["request_id"] = rid
    return JSONResponse(status_code=500, content=body)


# --------------------------------------------------------------------------- #
# pipeline cache
# --------------------------------------------------------------------------- #
def _decode_image(raw: bytes, declared_type: Optional[str]):
    """Validate + decode upload bytes into a dimension-checked RGB array."""
    settings = get_settings()
    validate_image_bytes(raw, max_mb=settings.max_upload_mb, declared_type=declared_type)
    try:
        image = bytes_to_rgb_array(raw)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"Could not decode image: {exc}") from exc
    h, w = image.shape[:2]
    validate_dimensions(w, h, max_side=settings.max_image_side)
    return image


def _run_inference(image, task: str, backend: Optional[str], frame_index: int = 0):
    from visionforge.models.manager import get_model_manager

    try:
        return get_model_manager().infer(image, task=task, backend=backend, frame_index=frame_index)
    except VisionForgeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(f"Inference failed: {exc}") from exc


def _annotate(image, result):
    from visionforge.core.draw import draw_detections

    return draw_detections(image, result)


# --------------------------------------------------------------------------- #
# static GUI
# --------------------------------------------------------------------------- #
def _serve_static(name: str, media_type: str) -> Response:
    path = WEB_DIR / name
    if not path.exists():
        return Response("", media_type=media_type, status_code=404)
    return Response(path.read_text(encoding="utf-8"), media_type=media_type)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>vision-forge</h1><p>Web GUI not found.</p>")


@app.get("/app.js", include_in_schema=False)
async def app_js() -> Response:
    return _serve_static("app.js", "application/javascript")


@app.get("/style.css", include_in_schema=False)
async def style_css() -> Response:
    return _serve_static("style.css", "text/css")


# --------------------------------------------------------------------------- #
# health / readiness / metrics / models
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": __version__, "device": _settings.resolved_device})


@app.get("/ready")
async def ready() -> JSONResponse:
    from visionforge.models.manager import get_model_manager

    manager = get_model_manager()
    is_ready = manager.is_ready()
    payload = {
        "status": "ready" if is_ready else "not_ready",
        "warmed_tasks": sorted(manager.warmed_tasks),
        "device": _settings.resolved_device,
    }
    if not is_ready:
        return JSONResponse(payload, status_code=503)
    return JSONResponse(payload)


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    if not _settings.enable_metrics:
        raise ServiceUnavailableError("Metrics are disabled")
    return Response(get_metrics().render(), media_type=CONTENT_TYPE_LATEST)


@app.get("/models")
async def models() -> JSONResponse:
    from visionforge.models.registry import get_registry

    registry = get_registry()
    return JSONResponse(
        {
            "tasks": list(VALID_TASKS),
            "backends": registry.available(),
            "backend_names": list(VALID_BACKENDS),
            "defaults": {
                "device": _settings.resolved_device,
                "conf_threshold": _settings.conf_threshold,
                "iou_threshold": _settings.iou_threshold,
                "image_size": _settings.image_size,
            },
        }
    )


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
@app.post("/infer")
async def infer(
    request: Request,
    file: UploadFile = File(...),
    task: str = Form("detection"),
    backend: Optional[str] = Form(None),
    annotate: bool = Form(False),
    _auth=Depends(require_api_key),
) -> JSONResponse:
    task = validate_task(task, VALID_TASKS)
    raw = await file.read()
    image = _decode_image(raw, file.content_type)
    result = _run_inference(image, task, backend)
    payload = {"result": result.to_dict()}
    if annotate:
        with contextlib.suppress(Exception):
            payload["annotated"] = encode_data_url(rgb_array_to_bytes(_annotate(image, result)))
    return JSONResponse(payload)


@app.post("/infer/batch")
async def infer_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    task: str = Form("detection"),
    backend: Optional[str] = Form(None),
    _auth=Depends(require_api_key),
) -> JSONResponse:
    task = validate_task(task, VALID_TASKS)
    if len(files) > 32:
        raise ValidationError("Batch limited to 32 images", details={"got": len(files)})
    images = []
    for f in files:
        raw = await f.read()
        images.append(_decode_image(raw, f.content_type))
    from visionforge.models.manager import get_model_manager

    results = get_model_manager().infer_batch(images, task=task, backend=backend)
    return JSONResponse({"count": len(results), "results": [r.to_dict() for r in results]})


# --------------------------------------------------------------------------- #
# async jobs (video / batch)
# --------------------------------------------------------------------------- #
@app.post("/jobs")
async def submit_job(
    request: Request,
    source: str = Form(...),
    task: str = Form("detection"),
    backend: Optional[str] = Form(None),
    max_frames: Optional[int] = Form(None),
    _auth=Depends(require_api_key),
) -> JSONResponse:
    task = validate_task(task, VALID_TASKS)
    # Only allow the synthetic demo or server-local paths; reject arbitrary URLs.
    if source != "demo" and not Path(source).exists():
        raise ValidationError(
            "source must be 'demo' or an existing server-side file path",
            details={"source": source},
        )
    from visionforge.worker.jobs import get_job_manager

    job = get_job_manager().submit(task=task, source=source, backend=backend, max_frames=max_frames)
    return JSONResponse({"job": job.to_dict()}, status_code=202)


@app.get("/jobs/{job_id}")
async def job_status(job_id: str, _auth=Depends(require_api_key)) -> JSONResponse:
    from visionforge.worker.jobs import get_job_manager

    job = get_job_manager().get(job_id)
    if job is None:
        from visionforge.errors import NotFoundError

        raise NotFoundError("Job not found", details={"job_id": job_id})
    return JSONResponse({"job": job.to_dict()})


@app.get("/jobs/{job_id}/results")
async def job_results(job_id: str, _auth=Depends(require_api_key)) -> JSONResponse:
    from visionforge.worker.jobs import get_job_manager

    job = get_job_manager().get(job_id)
    if job is None:
        from visionforge.errors import NotFoundError

        raise NotFoundError("Job not found", details={"job_id": job_id})
    return JSONResponse({"job_id": job_id, "results": job.results})


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, _auth=Depends(require_api_key)) -> JSONResponse:
    from visionforge.worker.jobs import get_job_manager

    cancelled = get_job_manager().cancel(job_id)
    if not cancelled:
        raise ValidationError(
            "Job cannot be cancelled (not found or already finished)",
            details={"job_id": job_id},
        )
    return JSONResponse({"job_id": job_id, "status": "cancelled"})


# --------------------------------------------------------------------------- #
# WebSocket live streaming
# --------------------------------------------------------------------------- #
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    # Optional auth on the socket via query param or header.
    settings = get_settings()
    if settings.require_auth:
        from visionforge.api.deps import get_authenticator

        key = websocket.headers.get("x-api-key") or websocket.query_params.get("api_key")
        try:
            get_authenticator().authenticate(key)
        except Exception:
            await websocket.close(code=4401)
            return
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            if msg.get("type") != "frame":
                await websocket.send_json({"type": "error", "message": "Unknown message type"})
                continue
            try:
                task = validate_task(msg.get("task"), VALID_TASKS)
                image_bytes = decode_base64(msg["image"])
                validate_image_bytes(image_bytes, max_mb=settings.max_upload_mb)
                image = bytes_to_rgb_array(image_bytes)
                h, w = image.shape[:2]
                validate_dimensions(w, h, max_side=settings.max_image_side)
            except VisionForgeError as exc:
                await websocket.send_json({"type": "error", "message": exc.message})
                continue
            except Exception as exc:  # noqa: BLE001
                await websocket.send_json({"type": "error", "message": f"Bad frame: {exc}"})
                continue

            frame_index = int(msg.get("frame_index", 0))
            want_annotate = bool(msg.get("annotate", False))
            try:
                result = _run_inference(image, task, msg.get("backend"), frame_index)
            except VisionForgeError as exc:
                await websocket.send_json({"type": "error", "message": exc.message})
                continue
            response = {
                "type": "result",
                "frame_index": frame_index,
                "result": result.to_dict(),
            }
            if want_annotate:
                with contextlib.suppress(Exception):
                    response["annotated"] = encode_data_url(rgb_array_to_bytes(_annotate(image, result)))
            await websocket.send_json(response)
    except WebSocketDisconnect:
        return
    except Exception:  # noqa: BLE001
        with contextlib.suppress(Exception):
            await websocket.close()


def run() -> None:  # pragma: no cover - thin uvicorn launcher
    import uvicorn

    uvicorn.run(
        "visionforge.api.server:app",
        host=_settings.host,
        port=_settings.port,
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    run()
