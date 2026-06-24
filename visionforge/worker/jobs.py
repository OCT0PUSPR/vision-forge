"""Background video/batch inference jobs.

Default execution uses an in-process thread pool (``JobManager``) so the full
async job lifecycle — submit, progress %, status polling, cancellation — works
with zero external services. For horizontal scale, the same ``process_video_job``
function is wired as an arq task in :mod:`visionforge.worker.arq_worker` (redis).

State (status/progress/results) is persisted via the DB repositories when a
database is configured; otherwise an in-memory store is used.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from visionforge.core.video import iter_frames
from visionforge.db.models import JobStatus
from visionforge.observability.logging import get_logger
from visionforge.observability.metrics import get_metrics

log = get_logger("visionforge.worker")


@dataclass
class JobState:
    """In-memory job state (mirror of the DB row when no DB is configured)."""

    id: str
    task: str
    source: str
    backend: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    total_frames: int = 0
    processed_frames: int = 0
    error: Optional[str] = None
    results: List[dict] = field(default_factory=list)
    _cancelled: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task": self.task,
            "source": self.source,
            "backend": self.backend,
            "status": self.status.value if isinstance(self.status, JobStatus) else self.status,
            "progress": round(self.progress, 2),
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "error": self.error,
            "result_count": len(self.results),
        }


class JobManager:
    """Thread-pool backed job manager with cancellation + progress."""

    def __init__(self, max_workers: int = 2, max_frames_per_job: int = 5000) -> None:
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, JobState] = {}
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        self._metrics = get_metrics()
        self.max_frames_per_job = max_frames_per_job
        self._shutdown = False

    def _ensure_executor(self) -> None:
        # Rebuild the pool if a prior shutdown disabled it (defense in depth).
        if self._shutdown:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
            self._shutdown = False

    def submit(
        self,
        task: str,
        source: str,
        backend: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> JobState:
        import uuid

        job = JobState(id=str(uuid.uuid4()), task=task, source=source, backend=backend)
        with self._lock:
            self._ensure_executor()
            self._jobs[job.id] = job
        self._metrics.active_jobs.inc()
        future = self._executor.submit(self._run, job.id, max_frames)
        with self._lock:
            self._futures[job.id] = future
        log.info("job_submitted", job_id=job.id, task=task, source=source)
        return job

    def _run(self, job_id: str, max_frames: Optional[int]) -> None:
        job = self.get(job_id)
        if job is None:
            return
        from visionforge.models.manager import get_model_manager

        manager = get_model_manager()
        cap = max_frames or self.max_frames_per_job
        try:
            job.status = JobStatus.RUNNING
            log.info("job_running", job_id=job_id)
            processed = 0
            for idx, frame in iter_frames(job.source, max_frames=cap):
                if job._cancelled:
                    job.status = JobStatus.CANCELLED
                    log.info("job_cancelled", job_id=job_id)
                    return
                result = manager.infer(frame, task=job.task, backend=job.backend, frame_index=idx)
                job.results.append(result.to_dict())
                processed += 1
                job.processed_frames = processed
                # Progress: if we know total frames use it, else cap-based.
                denom = cap if cap else max(processed, 1)
                job.progress = min(99.0, 100.0 * processed / denom)
            job.total_frames = processed
            job.progress = 100.0
            job.status = JobStatus.SUCCEEDED
            log.info("job_succeeded", job_id=job_id, frames=processed)
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.FAILED
            job.error = str(exc)
            log.exception("job_failed", job_id=job_id)
        finally:
            self._metrics.active_jobs.dec()

    def get(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        if job.status in (
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        ):
            return False
        job._cancelled = True
        return True

    def list_jobs(self) -> List[JobState]:
        with self._lock:
            return list(self._jobs.values())

    def shutdown(self, wait: bool = False) -> None:
        self._shutdown = True
        self._executor.shutdown(wait=wait, cancel_futures=True)


_MANAGER: Optional[JobManager] = None
_LOCK = threading.Lock()


def get_job_manager() -> JobManager:
    global _MANAGER
    with _LOCK:
        if _MANAGER is None:
            _MANAGER = JobManager()
        return _MANAGER


def reset_job_manager() -> None:
    global _MANAGER
    with _LOCK:
        if _MANAGER is not None:
            _MANAGER.shutdown(wait=False)
        _MANAGER = None
