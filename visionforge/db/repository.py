"""Repositories: focused data-access objects over the ORM models.

Keeps SQLAlchemy query specifics out of the API/worker layers and gives a small,
testable surface.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from visionforge.db.models import (
    ApiKey,
    AuditLog,
    InferenceJob,
    JobResult,
    JobStatus,
)


class ApiKeyRepository:
    def __init__(self, session) -> None:
        self.session = session

    def create(
        self,
        name: str,
        key_hash: str,
        rate_limit_per_min: int = 120,
        scopes: Optional[str] = None,
    ) -> ApiKey:
        key = ApiKey(
            name=name,
            key_hash=key_hash,
            rate_limit_per_min=rate_limit_per_min,
            scopes=scopes,
            active=1,
        )
        self.session.add(key)
        self.session.flush()
        return key

    def get_by_hash(self, key_hash: str) -> Optional[ApiKey]:
        from sqlalchemy import select

        stmt = select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.active == 1)
        return self.session.execute(stmt).scalar_one_or_none()

    def list(self) -> List[ApiKey]:
        from sqlalchemy import select

        return list(self.session.execute(select(ApiKey)).scalars())

    def deactivate(self, key_id: int) -> bool:
        key = self.session.get(ApiKey, key_id)
        if key is None:
            return False
        key.active = 0
        return True


class JobRepository:
    def __init__(self, session) -> None:
        self.session = session

    def create(
        self,
        task: str,
        source: str,
        backend: Optional[str] = None,
        api_key_id: Optional[int] = None,
    ) -> InferenceJob:
        job = InferenceJob(
            id=str(uuid.uuid4()),
            task=task,
            backend=backend,
            source=source,
            status=JobStatus.PENDING,
            api_key_id=api_key_id,
        )
        self.session.add(job)
        self.session.flush()
        return job

    def get(self, job_id: str) -> Optional[InferenceJob]:
        return self.session.get(InferenceJob, job_id)

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        progress: Optional[float] = None,
        processed_frames: Optional[int] = None,
        total_frames: Optional[int] = None,
        error: Optional[str] = None,
    ) -> Optional[InferenceJob]:
        job = self.get(job_id)
        if job is None:
            return None
        job.status = status
        if progress is not None:
            job.progress = max(0.0, min(100.0, progress))
        if processed_frames is not None:
            job.processed_frames = processed_frames
        if total_frames is not None:
            job.total_frames = total_frames
        if error is not None:
            job.error = error
        return job

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        if job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False
        job.status = JobStatus.CANCELLED
        return True

    def add_result(
        self,
        job_id: str,
        frame_index: int,
        payload: dict,
        artifact_url: Optional[str] = None,
    ) -> JobResult:
        result = JobResult(
            job_id=job_id,
            frame_index=frame_index,
            payload=payload,
            artifact_url=artifact_url,
        )
        self.session.add(result)
        self.session.flush()
        return result

    def results(self, job_id: str) -> List[JobResult]:
        from sqlalchemy import select

        stmt = select(JobResult).where(JobResult.job_id == job_id).order_by(JobResult.frame_index)
        return list(self.session.execute(stmt).scalars())


class AuditRepository:
    def __init__(self, session) -> None:
        self.session = session

    def log(
        self,
        action: str,
        *,
        api_key_id: Optional[int] = None,
        request_id: Optional[str] = None,
        detail: Optional[dict] = None,
    ) -> AuditLog:
        entry = AuditLog(
            action=action,
            api_key_id=api_key_id,
            request_id=request_id,
            detail=detail,
        )
        self.session.add(entry)
        self.session.flush()
        return entry
