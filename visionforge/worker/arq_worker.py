"""arq + redis worker entrypoint (optional, for horizontal scale).

Run with::

    arq visionforge.worker.arq_worker.WorkerSettings

This mirrors the in-process :class:`JobManager` lifecycle but distributes jobs
across processes/hosts via redis. Guarded imports keep the module importable
without arq/redis installed (the in-process manager remains the default).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from visionforge.observability.logging import get_logger

log = get_logger("visionforge.worker.arq")


async def process_video_job(
    ctx: Dict[str, Any],
    task: str,
    source: str,
    backend: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """arq task: run video inference and persist results.

    Reuses the synchronous frame loop via a thread to avoid blocking the event
    loop, and writes progress/results through the DB repositories.
    """
    import asyncio

    from visionforge.worker.jobs import get_job_manager

    manager = get_job_manager()

    def _run() -> Dict[str, Any]:
        job = manager.submit(task=task, source=source, backend=backend, max_frames=max_frames)
        # Block until terminal (this runs inside a worker thread).
        import time

        while True:
            state = manager.get(job.id)
            if state is None:
                break
            if str(state.status).endswith(("succeeded", "failed", "cancelled")):
                return state.to_dict()
            time.sleep(0.1)
        return {"id": job.id, "status": "unknown"}

    return await asyncio.get_event_loop().run_in_executor(None, _run)


async def startup(ctx: Dict[str, Any]) -> None:  # pragma: no cover - needs redis
    from visionforge.models.manager import get_model_manager

    log.info("arq_worker_startup")
    get_model_manager().warmup(["detection"])


async def shutdown(ctx: Dict[str, Any]) -> None:  # pragma: no cover - needs redis
    from visionforge.models.manager import get_model_manager
    from visionforge.worker.jobs import get_job_manager

    log.info("arq_worker_shutdown")
    get_job_manager().shutdown(wait=True)
    get_model_manager().shutdown()


def _redis_settings():  # pragma: no cover - needs arq/redis
    from arq.connections import RedisSettings

    url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    return RedisSettings.from_dsn(url)


class WorkerSettings:  # pragma: no cover - consumed by the arq CLI
    """arq worker configuration object."""

    functions = [process_video_job]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 4
    job_timeout = 3600

    try:
        redis_settings = _redis_settings()
    except Exception:
        redis_settings = None
