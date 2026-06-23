"""FastAPI application.

Endpoints
---------
* ``GET  /``            -> serves the single-page web GUI
* ``GET  /health``      -> liveness/readiness probe
* ``GET  /models``      -> available task/backend combinations + settings
* ``POST /infer``       -> multipart image upload -> JSON (+ optional annotated b64)
* ``WS   /ws/stream``   -> live frame streaming (base64 in -> detections out)

WebSocket protocol
------------------
Client -> server (JSON text frame)::

    {"type": "frame", "task": "detection", "image": "data:image/jpeg;base64,...",
     "annotate": true, "frame_index": 12}

Server -> client (JSON text frame)::

    {"type": "result", "frame_index": 12, "result": {...FrameResult...},
     "annotated": "data:image/jpeg;base64,..."}   # annotated present iff requested

Errors are returned as ``{"type": "error", "message": "..."}``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from visionforge import __version__
from visionforge.api.encoding import (
    bytes_to_rgb_array,
    decode_base64,
    encode_data_url,
    rgb_array_to_bytes,
    validate_task,
)
from visionforge.config import get_settings
from visionforge.models.registry import VALID_TASKS, get_registry
from visionforge.pipeline import VisionPipeline

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(
    title="vision-forge",
    version=__version__,
    description="Real-time, multi-task computer vision platform.",
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _settings.cors_origins.split(",")] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Small per-process pipeline cache keyed by (task, backend).
_pipelines: dict = {}


def get_pipeline(task: str, backend: Optional[str] = None) -> VisionPipeline:
    registry = get_registry()
    resolved = registry.resolve(task, backend)
    key = f"{resolved}:{task}"
    if key not in _pipelines:
        _pipelines[key] = VisionPipeline(task=task, backend=resolved, registry=registry)
    return _pipelines[key]


# --------------------------------------------------------------------------- #
# static GUI
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>vision-forge</h1><p>Web GUI not found.</p>")


@app.get("/app.js")
async def app_js():
    from fastapi.responses import Response

    path = WEB_DIR / "app.js"
    return Response(
        path.read_text(encoding="utf-8") if path.exists() else "",
        media_type="application/javascript",
    )


@app.get("/style.css")
async def style_css():
    from fastapi.responses import Response

    path = WEB_DIR / "style.css"
    return Response(
        path.read_text(encoding="utf-8") if path.exists() else "",
        media_type="text/css",
    )


# --------------------------------------------------------------------------- #
# JSON endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "version": __version__,
            "device": _settings.resolved_device,
        }
    )


@app.get("/models")
async def models() -> JSONResponse:
    registry = get_registry()
    return JSONResponse(
        {
            "tasks": list(VALID_TASKS),
            "backends": registry.available(),
            "defaults": {
                "device": _settings.resolved_device,
                "conf_threshold": _settings.conf_threshold,
                "iou_threshold": _settings.iou_threshold,
                "image_size": _settings.image_size,
            },
        }
    )


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    task: str = Form("detection"),
    backend: Optional[str] = Form(None),
    annotate: bool = Form(False),
) -> JSONResponse:
    try:
        task = validate_task(task, VALID_TASKS)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    raw = await file.read()
    max_bytes = _settings.max_upload_mb * 1024 * 1024
    if len(raw) > max_bytes:
        return JSONResponse(
            {"error": f"File exceeds {_settings.max_upload_mb} MB limit"},
            status_code=413,
        )

    try:
        image = bytes_to_rgb_array(raw)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"Bad image: {exc}"}, status_code=400)

    try:
        pipeline = get_pipeline(task, backend)
        result = pipeline.infer_array(image, frame_index=0)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"Inference failed: {exc}"}, status_code=500)

    payload = {"result": result.to_dict()}
    if annotate:
        try:
            annotated = pipeline.annotate(image, result)
            payload["annotated"] = encode_data_url(rgb_array_to_bytes(annotated))
        except Exception as exc:  # noqa: BLE001
            payload["annotate_error"] = str(exc)
    return JSONResponse(payload)


# --------------------------------------------------------------------------- #
# WebSocket live streaming
# --------------------------------------------------------------------------- #
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid JSON"}
                )
                continue

            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            if msg.get("type") != "frame":
                await websocket.send_json(
                    {"type": "error", "message": "Unknown message type"}
                )
                continue

            try:
                task = validate_task(msg.get("task"), VALID_TASKS)
                image_bytes = decode_base64(msg["image"])
                image = bytes_to_rgb_array(image_bytes)
            except Exception as exc:  # noqa: BLE001
                await websocket.send_json(
                    {"type": "error", "message": f"Bad frame: {exc}"}
                )
                continue

            frame_index = int(msg.get("frame_index", 0))
            want_annotate = bool(msg.get("annotate", False))
            try:
                pipeline = get_pipeline(task, msg.get("backend"))
                result = pipeline.infer_array(image, frame_index=frame_index)
            except Exception as exc:  # noqa: BLE001
                await websocket.send_json(
                    {"type": "error", "message": f"Inference failed: {exc}"}
                )
                continue

            response = {
                "type": "result",
                "frame_index": frame_index,
                "result": result.to_dict(),
            }
            if want_annotate:
                try:
                    annotated = pipeline.annotate(image, result)
                    response["annotated"] = encode_data_url(
                        rgb_array_to_bytes(annotated)
                    )
                except Exception as exc:  # noqa: BLE001
                    response["annotate_error"] = str(exc)
            await websocket.send_json(response)
    except WebSocketDisconnect:
        return
    except Exception:  # noqa: BLE001 - never crash the server on a bad client
        try:
            await websocket.close()
        except Exception:
            pass


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
