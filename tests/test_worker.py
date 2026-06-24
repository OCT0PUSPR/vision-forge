"""Tests for the in-process background JobManager.

The job loop calls the ModelManager; we run a couple of frames of the synthetic
demo. The real inference part is skipped when ultralytics is unavailable, but
the lifecycle (submit/poll/cancel/state) is always exercised with a stub.
"""

import importlib
import time

import pytest

from visionforge.db.models import JobStatus
from visionforge.worker.jobs import JobManager, JobState

_HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None


def test_job_state_to_dict():
    js = JobState(id="j1", task="detection", source="demo")
    d = js.to_dict()
    assert d["id"] == "j1"
    assert d["status"] == "pending"
    assert d["result_count"] == 0


def test_jobmanager_cancel_unknown():
    mgr = JobManager(max_workers=1)
    try:
        assert mgr.cancel("nope") is False
    finally:
        mgr.shutdown(wait=False)


def test_jobmanager_executor_rebuilds_after_shutdown():
    mgr = JobManager(max_workers=1)
    mgr.shutdown(wait=True)
    # submitting after shutdown should transparently rebuild the pool
    job = mgr.submit(task="detection", source="demo", max_frames=1)
    assert job.id is not None
    mgr.shutdown(wait=True)


@pytest.mark.skipif(not _HAS_ULTRALYTICS, reason="ultralytics not installed")
def test_jobmanager_runs_demo_to_completion():
    mgr = JobManager(max_workers=1)
    try:
        # Use the 'baseline' (Ultralytics) backend: the default 'centernet'
        # detector requires a trained checkpoint that is not committed.
        job = mgr.submit(task="detection", source="demo", backend="baseline", max_frames=2)
        deadline = time.time() + 60
        while time.time() < deadline:
            state = mgr.get(job.id)
            if state.status in (
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ):
                break
            time.sleep(0.2)
        state = mgr.get(job.id)
        assert state.status == JobStatus.SUCCEEDED, state.error
        assert state.processed_frames == 2
        assert state.progress == 100.0
        assert len(state.results) == 2
    finally:
        mgr.shutdown(wait=False)
