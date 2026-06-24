"""SQLAlchemy 2.0 ORM models.

Schema:
    * ``api_keys``       — hashed API keys + per-key rate limits/scopes
    * ``inference_jobs`` — async job records with status + progress
    * ``job_results``    — normalized results attached to a job
    * ``audit_log``      — security/audit trail of authenticated actions

SQLite by default; Postgres via ``DATABASE_URL``. All heavy imports are guarded
so this module imports even when SQLAlchemy is absent (the API degrades to a
stateless mode).
"""

from __future__ import annotations

import datetime as _dt
import enum
from typing import Optional

try:
    from sqlalchemy import (
        JSON,
        DateTime,
        Enum,
        Float,
        ForeignKey,
        Integer,
        String,
        Text,
    )
    from sqlalchemy.orm import (
        DeclarativeBase,
        Mapped,
        mapped_column,
        relationship,
    )

    _HAS_SQLALCHEMY = True
except Exception:  # pragma: no cover - exercised only without the dep
    _HAS_SQLALCHEMY = False


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class JobStatus(str, enum.Enum):
    """Lifecycle states for an async inference job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


if _HAS_SQLALCHEMY:

    class Base(DeclarativeBase):
        pass

    class ApiKey(Base):
        __tablename__ = "api_keys"

        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        name: Mapped[str] = mapped_column(String(128), nullable=False)
        key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
        rate_limit_per_min: Mapped[int] = mapped_column(Integer, default=120)
        scopes: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
        active: Mapped[bool] = mapped_column(Integer, default=1)
        created_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
        last_used_at: Mapped[Optional[_dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    class InferenceJob(Base):
        __tablename__ = "inference_jobs"

        id: Mapped[str] = mapped_column(String(36), primary_key=True)
        task: Mapped[str] = mapped_column(String(32), nullable=False)
        backend: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
        source: Mapped[str] = mapped_column(String(512), nullable=False)
        status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.PENDING, index=True)
        progress: Mapped[float] = mapped_column(Float, default=0.0)
        total_frames: Mapped[int] = mapped_column(Integer, default=0)
        processed_frames: Mapped[int] = mapped_column(Integer, default=0)
        error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
        api_key_id: Mapped[Optional[int]] = mapped_column(ForeignKey("api_keys.id"), nullable=True)
        created_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
        updated_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

        results: Mapped[list] = relationship("JobResult", back_populates="job", cascade="all, delete-orphan")

    class JobResult(Base):
        __tablename__ = "job_results"

        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        job_id: Mapped[str] = mapped_column(ForeignKey("inference_jobs.id"), index=True, nullable=False)
        frame_index: Mapped[int] = mapped_column(Integer, default=0)
        payload: Mapped[dict] = mapped_column(JSON, nullable=False)
        artifact_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
        created_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

        job: Mapped["InferenceJob"] = relationship("InferenceJob", back_populates="results")

    class AuditLog(Base):
        __tablename__ = "audit_log"

        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        api_key_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
        action: Mapped[str] = mapped_column(String(64), nullable=False)
        request_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
        detail: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
        created_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

else:  # pragma: no cover - placeholders so attribute access fails loudly
    Base = None  # type: ignore[misc,assignment]
    ApiKey = None  # type: ignore[misc,assignment]
    InferenceJob = None  # type: ignore[misc,assignment]
    JobResult = None  # type: ignore[misc,assignment]
    AuditLog = None  # type: ignore[misc,assignment]
