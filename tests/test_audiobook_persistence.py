"""Phase 13: crash-safe commits and second-writer protection.

The crash cases use a *real* child process that hard-exits (``os._exit``) in the
middle of a commit, so no in-process rollback can run.  Whatever the child
leaves on disk is what recovery has to repair.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from dubbing_service.app.audiobook import persistence
from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.storage.process_lock import (
    LOCK_FILENAME,
    LockBusyError,
    LockOwner,
    ProjectsRootLock,
    pid_is_alive,
)
from dubbing_service.app.storage.projects import load_project, save_project
from dubbing_service.app.storage.segments import load_segments
from tests.audiobook_helpers import (
    build_project,
    build_segments,
    use_projects_root,
)

REPO_ROOT = Path(persistence.__file__).resolve().parents[3]

CRASH_CHILD = '''
import os
import sys
from pathlib import Path

from dubbing_service.app.audiobook import persistence

root = Path(sys.argv[1])
new_project = Path(sys.argv[2]).read_bytes()
new_segments = Path(sys.argv[3]).read_bytes()
crash_after = int(sys.argv[4])

_real_replace = os.replace
_calls = {"count": 0}


def crashing_replace(src, dst):
    """Promote for real, then die without unwinding anything."""

    _real_replace(src, dst)
    _calls["count"] += 1
    if _calls["count"] >= crash_after:
        os._exit(9)


os.replace = crashing_replace

persistence.commit_files(
    [
        (root / "project.json", new_project),
        (root / "segments.json", new_segments),
    ],
    root=root,
)
os._exit(0)
'''

HOLD_CHILD = '''
import os
import sys
import time
from pathlib import Path

from dubbing_service.app.storage.process_lock import ProjectsRootLock

root = Path(sys.argv[1])
ready = Path(sys.argv[2])
mode = sys.argv[3]

lock = ProjectsRootLock(root)
lock.acquire()
ready.write_text(str(os.getpid()), encoding="utf-8")

if mode == "die":
    # A writer that dies while holding the lock: no release, no cleanup.
    os._exit(9)

release = Path(str(ready) + ".release")
deadline = time.monotonic() + 120
while not release.exists() and time.monotonic() < deadline:
    time.sleep(0.05)
lock.release()
'''


@pytest.fixture()
def studio(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    return root, tmp_path


def _child_env() -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH") or ""
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    env["PYTHONUTF8"] = "1"
    return env


def _write_script(tmp_path: Path, name: str, source: str) -> Path:
    path = Path(tmp_path) / name
    path.write_text(source.strip() + "\n", encoding="utf-8")
    return path


def _residue(root: Path) -> list:
    """Every file a transaction should have swept: stages, backups, markers."""

    names = [
        path.name
        for path in root.iterdir()
        if path.suffix in {persistence.STAGE_SUFFIX, persistence.BACKUP_SUFFIX}
        or path.name.endswith(".tmp")
    ]
    return sorted(names + [path.name for path in persistence.marker_paths(root)])


def _seed_state(tmp_path: Path, project_bytes: bytes, segments_bytes: bytes) -> None:
    (Path(tmp_path) / "new_project.json").write_bytes(project_bytes)
    (Path(tmp_path) / "new_segments.json").write_bytes(segments_bytes)


def _crash_commit(tmp_path: Path, project_root: Path, crash_after: int):
    """Run a child process that hard-exits in the middle of a commit."""

    script = _write_script(tmp_path, f"crash_after_{crash_after}.py", CRASH_CHILD)
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(project_root),
            str(Path(tmp_path) / "new_project.json"),
            str(Path(tmp_path) / "new_segments.json"),
            str(crash_after),
        ],
        env=_child_env(),
        capture_output=True,
        timeout=180,
    )
    assert completed.returncode == 9, completed.stderr.decode("utf-8", "replace")
    return completed


@pytest.fixture()
def committed(monkeypatch, tmp_path):
    """State A, committed through the public API, plus its exact bytes."""

    use_projects_root(monkeypatch, tmp_path)

    def _build(name: str = "state-a"):
        project = build_project(name=name)
        segments = build_segments(project, ["A only line."])
        persistence.commit_project_state(
            project.id, project=project, segments=segments
        )
        project_root = safe_project_root(project.id)
        return {
            "project": project,
            "segments": segments,
            "root": project_root,
            "project_bytes": (project_root / "project.json").read_bytes(),
            "segments_bytes": (project_root / "segments.json").read_bytes(),
        }

    return _build


def _state_b(tmp_path: Path):
    """A second, independently built state to promote over state A."""

    project_b = build_project(name="state-b")
    project_b.chapters[0].id = "chapB"
    save_project(project_b)
    build_segments(project_b, ["B only line."])
    project_b_root = safe_project_root(project_b.id)
    project_bytes = (project_b_root / "project.json").read_bytes()
    segments_bytes = (project_b_root / "segments.json").read_bytes()
    _seed_state(tmp_path, project_bytes, segments_bytes)
    return project_bytes, segments_bytes


def test_commit_project_state_writes_both_files_and_leaves_no_trace(studio):
    root, _tmp_path = studio
    project = build_project(name="clean")
    segments = build_segments(project, ["Line one.", "Line two."])

    report = persistence.commit_project_state(
        project.id, project=project, segments=segments
    )

    assert report.transaction_id
    assert report.targets == ["project.json", "segments.json"]
    assert persistence.pending_markers(project.id) == []
    assert _residue(safe_project_root(project.id)) == []
    assert persistence.verify_project_state(project.id) == (True, [])
    assert root.is_dir()


def test_verify_project_state_detects_a_hybrid_pair(studio):
    project = build_project(name="hybrid")
    segments = build_segments(project, ["A line."])
    persistence.commit_project_state(project.id, project=project, segments=segments)
    project_root = safe_project_root(project.id)

    payload = json.loads((project_root / "segments.json").read_text(encoding="utf-8"))
    payload[0]["chapter_id"] = "chap_missing"
    (project_root / "segments.json").write_text(json.dumps(payload), encoding="utf-8")

    ok, problems = persistence.verify_project_state(project.id)

    assert ok is False
    assert any("unknown chapter chap_missing" in problem for problem in problems)


def test_verify_project_state_detects_an_unknown_role(studio):
    project = build_project(name="unknown-role")
    segments = build_segments(project, ["A line."])
    persistence.commit_project_state(project.id, project=project, segments=segments)
    project_root = safe_project_root(project.id)

    payload = json.loads((project_root / "segments.json").read_text(encoding="utf-8"))
    payload[0]["role_id"] = "role_missing"
    (project_root / "segments.json").write_text(json.dumps(payload), encoding="utf-8")

    ok, problems = persistence.verify_project_state(project.id)

    assert ok is False
    assert any("unknown role role_missing" in problem for problem in problems)


def test_verify_project_state_reports_missing_files(studio):
    project = build_project(name="missing")
    build_segments(project, ["A line."])
    project_root = safe_project_root(project.id)
    (project_root / "segments.json").unlink()

    ok, problems = persistence.verify_project_state(project.id)

    assert ok is False
    assert any("segments.json is missing" in problem for problem in problems)


def test_hard_kill_before_the_second_promotion_recovers_complete_old_state(
    studio, committed
):
    _root, tmp_path = studio
    state = committed()
    project, project_root = state["project"], state["root"]
    new_project, new_segments = _state_b(tmp_path)

    _crash_commit(tmp_path, project_root, crash_after=2)

    # The killed process left an intent marker and a hybrid pair behind.
    assert persistence.pending_markers(project.id), "the marker must survive the kill"
    assert (project_root / "project.json").read_bytes() == new_project
    assert (project_root / "segments.json").read_bytes() == state["segments_bytes"]
    ok, problems = persistence.verify_project_state(project.id)
    assert ok is False and problems, "a half-promoted pair must be detectably hybrid"

    report = persistence.recover_project_state(project.id)

    assert report.ok, report.as_dict()
    assert set(report.rolled_back) == {"project.json", "segments.json"}
    assert (project_root / "project.json").read_bytes() == state["project_bytes"]
    assert (project_root / "segments.json").read_bytes() == state["segments_bytes"]
    assert persistence.pending_markers(project.id) == []
    assert _residue(project_root) == []
    assert persistence.verify_project_state(project.id) == (True, [])
    reloaded_project = load_project(project.id)
    reloaded_segments = load_segments(project.id)
    assert [chapter.id for chapter in reloaded_project.chapters] == [
        chapter.id for chapter in state["project"].chapters
    ]
    assert [segment.text for segment in reloaded_segments] == [
        segment.text for segment in state["segments"]
    ]
    assert new_segments != state["segments_bytes"], "state B must be a real change"


def test_hard_kill_after_both_promotions_never_leaves_a_hybrid(studio, committed):
    _root, tmp_path = studio
    state = committed("state-a2")
    project, project_root = state["project"], state["root"]
    new_project, new_segments = _state_b(tmp_path)

    _crash_commit(tmp_path, project_root, crash_after=3)

    # Fully promoted but never marked done: complete, not hybrid.
    assert persistence.pending_markers(project.id)
    assert (project_root / "project.json").read_bytes() == new_project
    assert (project_root / "segments.json").read_bytes() == new_segments

    report = persistence.recover_project_state(project.id)

    assert report.ok, report.as_dict()
    assert (project_root / "project.json").read_bytes() == state["project_bytes"]
    assert (project_root / "segments.json").read_bytes() == state["segments_bytes"]
    assert persistence.pending_markers(project.id) == []
    assert _residue(project_root) == []
    assert persistence.verify_project_state(project.id) == (True, [])


def test_recovery_is_idempotent(studio, committed):
    _root, tmp_path = studio
    state = committed("state-a3")
    project_root = state["root"]
    _state_b(tmp_path)
    _crash_commit(tmp_path, project_root, crash_after=2)

    first = persistence.recover_project_state(state["project"].id)
    second = persistence.recover_project_state(state["project"].id)

    assert first.ok and second.ok
    assert first.rolled_back
    assert second.clean
    assert second.recovered == []
    assert second.markers_removed == 0
    assert persistence.pending_markers(state["project"].id) == []


def test_recovery_removes_a_file_the_transaction_created(studio, committed):
    state = committed("created")
    project_root = state["root"]
    transaction_id = "feedfacefeedface"
    backup = project_root / f".project.json.{transaction_id}{persistence.BACKUP_SUFFIX}"
    backup.write_bytes(state["project_bytes"])
    (project_root / "project.json").write_text('{"id": "promoted"}', encoding="utf-8")
    created = project_root / "cues.json"
    created.write_text("[]", encoding="utf-8")
    marker = project_root / f"{persistence.MARKER_PREFIX}{transaction_id}.json"
    marker.write_text(
        json.dumps(
            {
                "transaction_id": transaction_id,
                "targets": [
                    {"name": "project.json", "existed": True},
                    {"name": "cues.json", "existed": False},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = persistence.recover_project_state(state["project"].id)

    assert report.ok, report.as_dict()
    assert created.exists() is False, "a file the transaction created must go away"
    assert (project_root / "project.json").read_bytes() == state["project_bytes"]
    assert not backup.exists()
    assert not marker.exists()
    assert persistence.verify_project_state(state["project"].id) == (True, [])


def test_recovery_reports_an_unreadable_marker_instead_of_guessing(studio, committed):
    state = committed("corrupt")
    project_root = state["root"]
    marker = project_root / f"{persistence.MARKER_PREFIX}notjson.json"
    marker.write_text("{not json", encoding="utf-8")

    report = persistence.recover_project_state(state["project"].id)

    assert report.clean is False
    assert any("notjson.json" in item for item in report.leftovers)
    assert marker.exists(), "an unreadable marker must not be deleted"
    assert (project_root / "segments.json").read_bytes() == state["segments_bytes"]


def test_recovery_reports_a_missing_backup_instead_of_claiming_a_rollback(
    studio, committed
):
    state = committed("lostbackup")
    project_root = state["root"]
    marker = project_root / f"{persistence.MARKER_PREFIX}deadbeefdeadbeef.json"
    marker.write_text(
        json.dumps(
            {
                "transaction_id": "deadbeefdeadbeef",
                "targets": [{"name": "project.json", "existed": True}],
            }
        ),
        encoding="utf-8",
    )

    report = persistence.recover_project_state(state["project"].id)

    assert report.clean is False
    assert any("backup missing" in item for item in report.leftovers)
    assert "project.json" not in report.rolled_back


def test_in_process_failure_rolls_back_and_leaves_no_marker(
    studio, committed, monkeypatch
):
    state = committed("state-a4")
    project_root = state["root"]
    real_replace = os.replace
    calls = {"count": 0}

    def flaky_replace(src, dst):
        calls["count"] += 1
        if calls["count"] == 2:  # marker written, the first promotion fails
            raise OSError("synthetic promotion failure")
        real_replace(src, dst)

    monkeypatch.setattr(persistence.os, "replace", flaky_replace)

    with pytest.raises(OSError):
        persistence.commit_files(
            [
                (project_root / "project.json", b'{"id": "broken"}'),
                (project_root / "segments.json", b"[]"),
            ],
            root=project_root,
        )

    monkeypatch.undo()
    assert (project_root / "project.json").read_bytes() == state["project_bytes"]
    assert (project_root / "segments.json").read_bytes() == state["segments_bytes"]
    assert persistence.pending_markers(state["project"].id) == []
    assert _residue(project_root) == []
    assert persistence.verify_project_state(state["project"].id) == (True, [])


def test_commit_files_rejects_an_empty_transaction(studio, committed):
    state = committed("state-a5")

    with pytest.raises(persistence.PersistenceError):
        persistence.commit_files([], root=state["root"])


def test_there_is_only_one_canonical_state_pair(studio, committed):
    """A commit may not introduce a second source of truth for the project."""

    state = committed("canonical")
    project_root = state["root"]
    visible = {
        path.name for path in project_root.iterdir() if not path.name.startswith(".")
    }

    assert {"project.json", "segments.json"} <= visible
    assert {name for name in visible if name.endswith(".json")} == {
        "project.json",
        "segments.json",
    }, "the root may hold no other authoritative state file"
    databases = [
        path.name
        for path in project_root.rglob("*")
        if path.suffix in {".db", ".sqlite", ".sqlite3"}
    ]
    assert databases == [], "a second canonical database is not allowed"
    assert _residue(project_root) == []


def test_the_render_path_persists_transactionally(studio):
    from dubbing_service.app.audiobook import pipeline, renderer

    project = build_project(name="rendered")
    segments = build_segments(project, ["A short utterance."])
    rendering = renderer.FixtureRenderer(
        project_root=pipeline.project_root(project.id), renderer_id="fixture"
    )
    pipeline.render_utterances(
        project.id, [segment.id for segment in segments], renderer=rendering
    )

    project_root = safe_project_root(project.id)
    assert persistence.pending_markers(project.id) == []
    assert _residue(project_root) == []
    assert persistence.verify_project_state(project.id) == (True, [])
    reloaded = load_segments(project.id)
    assert max(len(segment.takes) for segment in reloaded) >= 1


def _hold_child(tmp_path: Path, root: Path, mode: str):
    """Start a child process that takes the projects-root lock."""

    script = _write_script(tmp_path, "hold_lock.py", HOLD_CHILD)
    ready = tmp_path / f"lock-held-{mode}"
    child = subprocess.Popen(
        [sys.executable, str(script), str(root), str(ready), mode],
        env=_child_env(),
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if ready.exists():
            return child, ready, int(ready.read_text(encoding="utf-8"))
        if child.poll() is not None:
            break
        time.sleep(0.05)
    if child.poll() is None:
        child.terminate()
    raise AssertionError("lock-holding child never acquired the lock")


def test_second_writer_is_refused_while_another_process_holds_the_root(
    studio, tmp_path
):
    root, _tmp_path = studio
    child, ready, child_pid = _hold_child(tmp_path, root, "hold")
    try:
        lock = ProjectsRootLock(root)
        owner = lock.read_owner()
        assert owner is not None and owner.pid == child_pid
        assert owner.pid != os.getpid()
        assert pid_is_alive(owner.pid)

        with pytest.raises(LockBusyError) as excinfo:
            lock.acquire(timeout=0.5)
        message = str(excinfo.value)
        assert str(root) in message
        assert str(child_pid) in message
        assert "另一" in message, "the refusal must explain who holds the root"

        # Nothing in this process may have taken the lock.
        assert lock.read_owner().pid == child_pid
    finally:
        Path(str(ready) + ".release").write_text("go", encoding="utf-8")
        child.wait(timeout=120)

    # A clean exit releases the lock and the next writer gets in.
    lock = ProjectsRootLock(root)
    assert lock.read_owner() is None
    owner = lock.acquire(timeout=1.0)
    assert owner.pid == os.getpid()
    lock.release()


def test_second_writer_refuses_live_local_owner_after_heartbeat_age_expires(
    studio, tmp_path
):
    root, _tmp_path = studio
    child, ready, child_pid = _hold_child(tmp_path, root, "hold")
    try:
        lock = ProjectsRootLock(root, stale_after_seconds=0.01)
        owner = lock.read_owner()
        assert owner is not None and owner.pid == child_pid

        # Simulate a long-lived Studio owner whose process is alive but whose
        # lock heartbeat was not refreshed during an idle period.
        stale = replace(owner, heartbeat_at="2000-01-01T00:00:00+00:00")
        lock.lock_path.write_text(json.dumps(stale.to_payload()), encoding="utf-8")
        assert stale.heartbeat_age() > 0.01
        assert stale.is_live(stale_after_seconds=0.01)
        assert pid_is_alive(child_pid)

        with pytest.raises(LockBusyError):
            lock.acquire(timeout=0.1, poll_interval=0.01)
        assert lock.read_owner().pid == child_pid
    finally:
        Path(str(ready) + ".release").write_text("go", encoding="utf-8")
        child.wait(timeout=120)


@pytest.mark.parametrize("error_code,expected", [(5, True), (87, False), (123, True)])
def test_windows_pid_query_errors_fail_closed_except_missing_pid(
    monkeypatch, error_code, expected
):
    import ctypes
    from types import SimpleNamespace

    from dubbing_service.app.storage import process_lock

    class ApiFunction:
        def __init__(self, callback):
            self.callback = callback

        def __call__(self, *args):
            return self.callback(*args)

    kernel32 = SimpleNamespace(
        OpenProcess=ApiFunction(lambda *args: None),
        GetLastError=ApiFunction(lambda: error_code),
        GetExitCodeProcess=ApiFunction(lambda *args: 0),
        GetProcessTimes=ApiFunction(lambda *args: 0),
        CloseHandle=ApiFunction(lambda *args: 1),
    )
    with monkeypatch.context() as patch:
        patch.setattr(process_lock.os, "name", "nt")
        patch.setattr(ctypes, "windll", SimpleNamespace(kernel32=kernel32), raising=False)
        actual = process_lock.pid_is_alive(987654321)
    assert actual is expected


def test_windows_process_handle_uses_pointer_width_and_reads_creation_time(
    monkeypatch,
):
    import ctypes
    from ctypes import wintypes
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    from dubbing_service.app.storage import process_lock

    class ApiFunction:
        def __init__(self, callback):
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    handle_value = 0x123456789ABC
    creation_ticks = 0x01DD123456789ABC
    observed_handles = []

    def set_dword(handle, output):
        observed_handles.append(handle)
        ctypes.cast(output, ctypes.POINTER(wintypes.DWORD))[0] = 259
        return 1

    def set_times(handle, creation, exit_time, kernel, user):
        observed_handles.append(handle)
        filetime = ctypes.cast(creation, ctypes.POINTER(wintypes.FILETIME))[0]
        filetime.dwLowDateTime = creation_ticks & 0xFFFFFFFF
        filetime.dwHighDateTime = creation_ticks >> 32
        return 1

    kernel32 = SimpleNamespace(
        OpenProcess=ApiFunction(lambda *_args: handle_value),
        GetLastError=ApiFunction(lambda: 0),
        GetExitCodeProcess=ApiFunction(set_dword),
        GetProcessTimes=ApiFunction(set_times),
        CloseHandle=ApiFunction(lambda handle: 1),
    )
    with monkeypatch.context() as patch:
        patch.setattr(process_lock.os, "name", "nt")
        patch.setattr(ctypes, "windll", SimpleNamespace(kernel32=kernel32), raising=False)
        alive, created_at = process_lock._windows_process_identity(2468)

    assert alive is True
    assert created_at == datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(
        microseconds=creation_ticks // 10
    )
    assert kernel32.OpenProcess.argtypes == (
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    )
    assert kernel32.OpenProcess.restype is wintypes.HANDLE
    assert kernel32.GetExitCodeProcess.argtypes[0] is wintypes.HANDLE
    assert kernel32.CloseHandle.argtypes == (wintypes.HANDLE,)
    assert ctypes.sizeof(wintypes.HANDLE) == ctypes.sizeof(ctypes.c_void_p)
    assert observed_handles == [handle_value, handle_value]


def test_windows_pid_reuse_requires_matching_process_creation_identity(monkeypatch):
    from datetime import datetime, timezone

    from dubbing_service.app.storage import process_lock

    # acquired_at can be behind process birth after a system clock adjustment;
    # process instance identity remains stable and is the deciding evidence.
    acquired_at = datetime(2022, 1, 1, tzinfo=timezone.utc)
    recorded_birth = datetime(2024, 1, 1, tzinfo=timezone.utc)
    reused_pid_birth = datetime(2027, 1, 1, tzinfo=timezone.utc)
    owner = LockOwner(
        token="legacy",
        pid=12345,
        host=socket.gethostname(),
        label="test",
        root="root",
        acquired_at=acquired_at.isoformat(),
        heartbeat_at="2000-01-01T00:00:00+00:00",
        process_created_at=recorded_birth.isoformat(),
    )
    with monkeypatch.context() as patch:
        patch.setattr(process_lock.os, "name", "nt")
        patch.setattr(
            process_lock,
            "_windows_process_identity",
            lambda _pid: (True, recorded_birth),
        )
        assert owner.is_live(stale_after_seconds=0.01) is True

        patch.setattr(
            process_lock,
            "_windows_process_identity",
            lambda _pid: (True, reused_pid_birth),
        )
        assert owner.is_live(stale_after_seconds=0.01) is False

        legacy_owner = replace(owner, process_created_at=None)
        patch.setattr(
            process_lock,
            "_windows_process_identity",
            lambda _pid: (True, reused_pid_birth),
        )
        assert legacy_owner.is_live(stale_after_seconds=0.01) is True

        patch.setattr(
            process_lock,
            "_windows_process_identity",
            lambda _pid: (True, None),
        )
        assert owner.is_live(stale_after_seconds=0.01) is True
        patch.setattr(process_lock, "_windows_process_identity", lambda _pid: (False, None))
        assert owner.is_live(stale_after_seconds=0.01) is False


def test_process_creation_identity_survives_nested_lock_and_heartbeat(
    monkeypatch, studio
):
    from datetime import datetime, timezone

    from dubbing_service.app.storage import process_lock

    root, _tmp_path = studio
    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(process_lock.os, "name", "nt")
    monkeypatch.setattr(
        process_lock,
        "_windows_process_identity",
        lambda _pid: (True, created_at),
    )

    lock = ProjectsRootLock(root)
    owner = lock.acquire(timeout=0.5)
    assert owner.process_created_at == created_at.isoformat()
    assert lock.read_owner().process_created_at == owner.process_created_at

    nested = lock.acquire(timeout=0.5)
    assert nested.process_created_at == owner.process_created_at
    refreshed = lock.heartbeat()
    assert refreshed.process_created_at == owner.process_created_at
    assert lock.read_owner().process_created_at == owner.process_created_at

    lock.release()
    assert lock.read_owner().process_created_at == owner.process_created_at
    lock.release()


def test_a_writer_that_dies_holding_the_lock_is_recovered(studio, tmp_path):
    root, _tmp_path = studio
    child, _ready, child_pid = _hold_child(tmp_path, root, "die")
    child.wait(timeout=120)
    assert child.returncode == 9

    lock = ProjectsRootLock(root)
    owner = lock.read_owner()
    assert owner is not None and owner.pid == child_pid
    assert not pid_is_alive(owner.pid), "the previous writer is gone"

    recovered = lock.acquire(timeout=5.0)
    assert recovered.pid == os.getpid()
    assert lock.recovered_stale is not None
    assert lock.recovered_stale.pid == child_pid
    lock.release()
    assert lock.read_owner() is None


def test_a_clean_exit_releases_the_lock(studio, tmp_path):
    root, _tmp_path = studio
    child, ready, _child_pid = _hold_child(tmp_path, root, "hold")
    Path(str(ready) + ".release").write_text("go", encoding="utf-8")

    assert child.wait(timeout=120) == 0
    lock = ProjectsRootLock(root)
    assert lock.read_owner() is None
    assert lock.lock_path.exists() is False
    owner = lock.acquire(timeout=0.5)
    assert owner.pid == os.getpid()
    lock.release()


def test_the_lock_file_lives_in_the_projects_root(studio):
    root, _tmp_path = studio
    lock = ProjectsRootLock(root)

    assert lock.lock_path == root / LOCK_FILENAME
    with lock.hold(timeout=0.5):
        assert lock.lock_path.is_file()
        payload = json.loads(lock.lock_path.read_text(encoding="utf-8"))
        assert payload["root"] == str(root)
        assert payload["pid"] == os.getpid()
    assert lock.lock_path.exists() is False
