"""Integration tests for the SQLAlchemy persistence layer (in-memory SQLite)."""

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

from visionforge.db.models import JobStatus  # noqa: E402
from visionforge.db.repository import (  # noqa: E402
    ApiKeyRepository,
    AuditRepository,
    JobRepository,
)
from visionforge.db.session import Database  # noqa: E402
from visionforge.security.auth import hash_key  # noqa: E402


@pytest.fixture()
def db():
    database = Database(url="sqlite:///:memory:")
    database.create_all()
    yield database
    database.dispose()


def test_api_key_crud(db):
    with db.session() as s:
        repo = ApiKeyRepository(s)
        key = repo.create(name="ci", key_hash=hash_key("k1"), rate_limit_per_min=50)
        assert key.id is not None
    with db.session() as s:
        repo = ApiKeyRepository(s)
        found = repo.get_by_hash(hash_key("k1"))
        assert found is not None
        assert found.name == "ci"
        assert found.rate_limit_per_min == 50
        assert repo.get_by_hash("missing") is None


def test_api_key_deactivate_hides_it(db):
    with db.session() as s:
        repo = ApiKeyRepository(s)
        key = repo.create(name="x", key_hash=hash_key("k2"))
        kid = key.id
    with db.session() as s:
        assert ApiKeyRepository(s).deactivate(kid) is True
    with db.session() as s:
        assert ApiKeyRepository(s).get_by_hash(hash_key("k2")) is None


def test_job_lifecycle(db):
    with db.session() as s:
        repo = JobRepository(s)
        job = repo.create(task="detection", source="demo")
        job_id = job.id
        assert job.status == JobStatus.PENDING
    with db.session() as s:
        repo = JobRepository(s)
        repo.update_status(job_id, JobStatus.RUNNING, progress=50.0, processed_frames=5, total_frames=10)
    with db.session() as s:
        repo = JobRepository(s)
        job = repo.get(job_id)
        assert job.status == JobStatus.RUNNING
        assert job.progress == 50.0
        assert job.processed_frames == 5


def test_job_cancel(db):
    with db.session() as s:
        repo = JobRepository(s)
        job = repo.create(task="detection", source="demo")
        jid = job.id
    with db.session() as s:
        assert JobRepository(s).cancel(jid) is True
    with db.session() as s:
        repo = JobRepository(s)
        assert repo.get(jid).status == JobStatus.CANCELLED
        # cannot cancel a finished job
        assert repo.cancel(jid) is False


def test_job_results(db):
    with db.session() as s:
        repo = JobRepository(s)
        job = repo.create(task="detection", source="demo")
        jid = job.id
        repo.add_result(jid, frame_index=0, payload={"count": 2})
        repo.add_result(jid, frame_index=1, payload={"count": 3})
    with db.session() as s:
        results = JobRepository(s).results(jid)
        assert len(results) == 2
        assert results[0].frame_index == 0
        assert results[1].payload["count"] == 3


def test_audit_log(db):
    with db.session() as s:
        entry = AuditRepository(s).log("infer", api_key_id=1, request_id="req-1", detail={"task": "detection"})
        assert entry.id is not None
        assert entry.action == "infer"
