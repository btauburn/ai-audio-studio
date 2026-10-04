"""Atomic project writes survive short Windows reader locks, not persistent errors."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from dubbing_service.app.models.project import Project
from dubbing_service.app.storage import atomic, projects
from tests.audiobook_helpers import use_projects_root


def project_fixture(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    project = Project(id="atomic-retry", name="Before")
    directory = root / project.id
    directory.mkdir()
    projects.save_project(project)
    return project, directory / "project.json"


def windows_error(code):
    error = PermissionError(errno.EACCES, "Windows file is temporarily unavailable")
    error.winerror = code
    return error


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_windows_error_retries_same_complete_stage_and_replaces_atomically(
    monkeypatch, tmp_path, code
):
    project, target = project_fixture(monkeypatch, tmp_path)
    original = target.read_bytes()
    project.name = "After"
    real_replace = os.replace
    stages, delays = [], []

    def replace(source, destination):
        assert Path(destination) == target
        assert target.read_bytes() == original
        assert json.loads(Path(source).read_text(encoding="utf-8"))["name"] == "After"
        stages.append(source)
        if len(stages) <= 2:
            raise windows_error(code)
        real_replace(source, destination)

    monkeypatch.setattr(atomic.os, "replace", replace)
    monkeypatch.setattr(atomic.time, "sleep", delays.append)
    projects.save_project(project)
    assert len(stages) == 3 and len(set(stages)) == 1
    assert delays == [0.05, 0.1]
    assert projects.load_project(project.id).name == "After"
    assert not list(target.parent.glob("project_*.tmp"))


def test_persistent_windows_denial_stops_at_cap_preserves_old_file_and_cleans_stage(
    monkeypatch, tmp_path
):
    project, target = project_fixture(monkeypatch, tmp_path)
    original = target.read_bytes()
    project.name = "Cannot publish"
    failure = windows_error(5)
    calls, delays = [], []

    def denied(source, destination):
        calls.append(source)
        assert target.read_bytes() == original
        raise failure

    monkeypatch.setattr(atomic.os, "replace", denied)
    monkeypatch.setattr(atomic.time, "sleep", delays.append)
    with pytest.raises(PermissionError) as raised:
        projects.save_project(project)
    assert raised.value is failure
    assert len(calls) == 6 and len(delays) == 5
    assert sum(delays) == pytest.approx(1.25)
    assert target.read_bytes() == original
    assert not list(target.parent.glob("project_*.tmp"))


@pytest.mark.parametrize(
    "error",
    [
        PermissionError(errno.EACCES, "generic permission failure without WinError"),
        OSError(errno.ENOSPC, "disk full"),
        OSError(errno.EIO, "device error"),
        FileNotFoundError(errno.ENOENT, "missing destination"),
        windows_error(19),
    ],
)
def test_unrelated_errors_fail_immediately_and_clean_stage(
    monkeypatch, tmp_path, error
):
    project, target = project_fixture(monkeypatch, tmp_path)
    original = target.read_bytes()
    calls, delays = [], []

    def fail(source, destination):
        calls.append(source)
        raise error

    monkeypatch.setattr(atomic.os, "replace", fail)
    monkeypatch.setattr(atomic.time, "sleep", delays.append)
    with pytest.raises(OSError) as raised:
        projects.save_project(project)
    assert raised.value is error
    assert len(calls) == 1 and not delays
    assert target.read_bytes() == original
    assert not list(target.parent.glob("project_*.tmp"))


@pytest.mark.skipif(
    os.name != "nt", reason="exercises a real Windows reader sharing mode"
)
def test_real_windows_reader_released_during_retry(monkeypatch, tmp_path):
    project, target = project_fixture(monkeypatch, tmp_path)
    reader = target.open("rb")
    project.name = "Published after reader release"
    delays = []

    def release_reader(delay):
        delays.append(delay)
        assert json.loads(reader.read())["name"] == "Before"
        reader.close()

    monkeypatch.setattr(atomic.time, "sleep", release_reader)
    try:
        projects.save_project(project)
    finally:
        reader.close()
    assert delays == [0.05]
    assert projects.load_project(project.id).name == "Published after reader release"
    assert not list(target.parent.glob("project_*.tmp"))


# --------------------------------------------------------------------------- #
# Every project-metadata writer must use the retrying replacement.
#
# jobs.json is written by the API process and by casting workers, so its
# replacement can meet another handle holding it open. A bare os.replace there
# turned that transient window into a failed casting round, blocked on
# "[WinError 5] Access is denied" renaming a jobs temp file.
# --------------------------------------------------------------------------- #
def test_every_project_metadata_writer_uses_the_retrying_replacement():
    """A writer that calls os.replace directly re-introduces the sharing failure."""
    import re as _re

    from dubbing_service.app.storage import jobs as jobs_storage
    from dubbing_service.app.storage import segments as segments_storage

    for module, marker in ((jobs_storage, "def save_jobs"),
                           (segments_storage, "def save_segments")):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "replace_metadata_file" in source.split(marker)[0], \
            f"{Path(module.__file__).name} must import the shared retrying replacement"
        body = source.split(marker, 1)[1]
        # Take one function's worth of body: up to the next top-level def.
        body = body.split("\ndef ", 1)[0]
        assert "replace_metadata_file(" in body, \
            f"{marker} in {Path(module.__file__).name} must use the retrying replacement"
        # A bare os.replace of the staged file is the defect being guarded.
        assert not _re.search(r"(?<!replace_metadata_file\()\s*os\.replace\(\s*temp", body), \
            f"{marker} in {Path(module.__file__).name} still calls os.replace directly"


def test_a_locked_jobs_file_is_retried_rather_than_failing_the_write(monkeypatch, tmp_path):
    """The job writer waits out a short lock instead of raising."""
    from dubbing_service.app.models.jobs import SynthesisJob, SynthesisJobStatus
    from dubbing_service.app.storage import jobs as jobs_storage

    root = use_projects_root(monkeypatch, tmp_path)
    project_id = "job-lock-retry"
    directory = root / project_id
    directory.mkdir()
    projects.save_project(Project(id=project_id, name="Job lock fixture"))

    real_replace = os.replace
    calls = []

    def flaky(source, destination):
        assert Path(destination).name == "jobs.json"
        calls.append(destination)
        if len(calls) <= 2:
            raise windows_error(5)
        return real_replace(source, destination)

    monkeypatch.setattr(atomic.os, "replace", flaky)
    monkeypatch.setattr(atomic.time, "sleep", lambda _delay: None)
    job = SynthesisJob(id="job-1", project_id=project_id, project_name="Fixture",
                       job_type="batch_synthesize", request_id="job-1",
                       idempotency_key="job-1", status=SynthesisJobStatus.QUEUED)
    jobs_storage.save_jobs(project_id, [job])
    assert len(calls) == 3, "a winerror-5 lock must be retried"
    assert [row.id for row in jobs_storage.load_jobs(project_id)] == ["job-1"]
    assert not list((tmp_path / "projects" / project_id).glob("jobs_*.tmp"))
