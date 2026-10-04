"""Windows reader locks cannot discard a Take or its transaction recovery data."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from dubbing_service.app.audiobook import persistence, pipeline, renderer
from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.storage import atomic
from dubbing_service.app.storage.projects import save_project
from dubbing_service.app.storage.segmentation_commit import (
    _replace_files_transactionally,
)
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audiobook_helpers import build_project, build_segments, use_projects_root


def windows_error(code=5):
    error = PermissionError(errno.EACCES, "metadata is temporarily locked")
    error.winerror = code
    return error


@pytest.fixture
def transaction(tmp_path):
    targets = [tmp_path / "project.json", tmp_path / "segments.json"]
    old = [b'{"state": "old project"}', b'[{"state": "old segments"}]']
    new = [b'{"state": "new project"}', b'[{"state": "new segments"}]']
    for target, data in zip(targets, old):
        target.write_bytes(data)
    return tmp_path, targets, old, new


def residue(root):
    return sorted(path.name for path in root.iterdir() if path.name.startswith("."))


def interrupted_transaction(transaction):
    root, targets, old, new = transaction
    transaction_id = "interrupted"
    backups = []
    for target, before, after in zip(targets, old, new):
        backup = root / f".{target.name}.{transaction_id}.backup"
        backup.write_bytes(before)
        backups.append(backup)
        target.write_bytes(after)
    marker = root / f".commit-{transaction_id}.json"
    marker.write_text(
        json.dumps(
            {
                "transaction_id": transaction_id,
                "targets": [
                    {"name": target.name, "existed": True} for target in targets
                ],
            }
        ),
        encoding="utf-8",
    )
    return marker, backups


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transaction_retries_same_segments_stage_then_publishes_both_files(
    monkeypatch, transaction, code
):
    root, targets, old, new = transaction
    real_replace = os.replace
    attempts, delays = [], []

    def replace(source, target):
        if Path(target) == targets[1]:
            attempts.append(Path(source))
            assert targets[1].read_bytes() == old[1]
            assert Path(source).read_bytes() == new[1]
            assert persistence.marker_paths(root)
            if len(attempts) <= 2:
                raise windows_error(code)
        real_replace(source, target)

    monkeypatch.setattr(atomic.os, "replace", replace)
    monkeypatch.setattr(atomic.time, "sleep", delays.append)
    report = persistence.commit_files(list(zip(targets, new)), root=root)
    assert report.targets == [target.name for target in targets]
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert delays == [0.05, 0.1]
    assert [target.read_bytes() for target in targets] == new
    assert not residue(root)


@pytest.mark.skipif(os.name != "nt", reason="real Windows reader sharing mode")
def test_real_reader_lock_during_take_commit_never_repeats_synthesis(
    monkeypatch, tmp_path
):
    use_projects_root(monkeypatch, tmp_path)
    project = build_project(name="reader-lock")
    segment = build_segments(project, ["A short original test line."])[0]
    root = safe_project_root(project.id)
    target = root / "segments.json"
    readers, delays, calls = [], [], []
    rendering = renderer.FixtureRenderer(project_root=root)
    real_render = rendering.render

    def render_with_reader(request):
        calls.append(request.take_id)
        outcome = real_render(request)
        readers.append(target.open("rb"))
        return outcome

    def release_reader(delay):
        delays.append(delay)
        assert json.load(readers[-1])[0]["selected_take_id"] is None
        readers[-1].close()

    monkeypatch.setattr(rendering, "render", render_with_reader)
    monkeypatch.setattr(atomic.time, "sleep", release_reader)
    try:
        first = pipeline.render_utterances(
            project.id, [segment.id], renderer=rendering, promote=True, resume=True
        )[0]
    finally:
        for reader in readers:
            reader.close()
    assert first.success and first.selected and not first.reused
    assert len(calls) == 1 and delays == [0.05]
    saved = load_segments(project.id)[0]
    assert saved.selected_take_id == first.take_id and len(saved.takes) == 1
    again = pipeline.render_utterances(
        project.id, [segment.id], renderer=rendering, promote=True, resume=True
    )[0]
    assert again.reused and len(calls) == 1
    assert not residue(root)


def test_permanent_promotion_failure_retries_rollback_and_restores_exact_old_state(
    monkeypatch, transaction
):
    root, targets, old, new = transaction
    real_replace = os.replace
    failure = windows_error()
    promotion_attempts, rollback_attempts, delays = [], [], []

    def replace(source, target):
        data = Path(source).read_bytes()
        if Path(target) == targets[1]:
            promotion_attempts.append(source)
            raise failure
        if Path(target) == targets[0] and data == old[0]:
            rollback_attempts.append(source)
            if len(rollback_attempts) <= 2:
                raise windows_error(32)
        real_replace(source, target)

    monkeypatch.setattr(atomic.os, "replace", replace)
    monkeypatch.setattr(atomic.time, "sleep", delays.append)
    with pytest.raises(PermissionError) as raised:
        persistence.commit_files(list(zip(targets, new)), root=root)
    assert raised.value is failure
    assert len(promotion_attempts) == 6 and len(set(promotion_attempts)) == 1
    assert len(rollback_attempts) == 3 and len(set(rollback_attempts)) == 1
    assert sum(delays) == pytest.approx(1.4)
    assert [target.read_bytes() for target in targets] == old
    assert not residue(root)


def test_failed_rollback_keeps_every_backup_and_journal_for_later_recovery(
    monkeypatch, transaction
):
    root, targets, old, new = transaction
    real_replace = os.replace
    failure = windows_error()

    def replace(source, target):
        if Path(target) == targets[1] or (
            Path(target) == targets[0] and Path(source).read_bytes() == old[0]
        ):
            raise failure
        real_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(atomic.os, "replace", replace)
        patch.setattr(atomic.time, "sleep", lambda delay: None)
        with pytest.raises(PermissionError) as raised:
            persistence.commit_files(list(zip(targets, new)), root=root)
        assert raised.value is failure
    assert [target.read_bytes() for target in targets] == [new[0], old[1]]
    assert len(persistence.marker_paths(root)) == 1
    assert len(list(root.glob(".*.backup"))) == 2
    assert not list(root.glob(".*.stage"))
    report = persistence.recover_root(root)
    assert report.ok and set(report.rolled_back) == {target.name for target in targets}
    assert [target.read_bytes() for target in targets] == old
    assert not residue(root)


@pytest.mark.parametrize("code", [5, 32, 33])
def test_recovery_retries_restore_without_consuming_original_backup(
    monkeypatch, transaction, code
):
    root, targets, old, _new = transaction
    marker, backups = interrupted_transaction(transaction)
    real_replace = os.replace
    attempts, delays = [], []

    def replace(source, target):
        if Path(target) == targets[1]:
            attempts.append(source)
            assert marker.exists()
            assert [backup.read_bytes() for backup in backups] == old
            if len(attempts) <= 2:
                raise windows_error(code)
        real_replace(source, target)

    monkeypatch.setattr(atomic.os, "replace", replace)
    monkeypatch.setattr(atomic.time, "sleep", delays.append)
    report = persistence.recover_root(root)
    assert report.ok and report.markers_removed == 1
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert delays == [0.05, 0.1]
    assert [target.read_bytes() for target in targets] == old
    assert not residue(root)


def test_partial_recovery_keeps_restored_backups_until_a_later_complete_recovery(
    monkeypatch, transaction
):
    root, targets, old, new = transaction
    marker, backups = interrupted_transaction(transaction)
    real_replace = os.replace
    attempts = []

    def replace(source, target):
        if Path(target) == targets[1]:
            attempts.append(source)
            raise windows_error()
        real_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(atomic.os, "replace", replace)
        patch.setattr(atomic.time, "sleep", lambda delay: None)
        report = persistence.recover_root(root)
    assert not report.ok and not report.clean and report.markers_removed == 0
    assert len(attempts) == 6
    assert report.rolled_back == [targets[0].name]
    assert [target.read_bytes() for target in targets] == [old[0], new[1]]
    assert marker.exists() and [backup.read_bytes() for backup in backups] == old
    assert not list(root.glob(".*.stage"))
    second = persistence.recover_root(root)
    assert second.ok and second.markers_removed == 1
    assert [target.read_bytes() for target in targets] == old
    assert not residue(root)
    assert persistence.recover_root(root).clean


@pytest.mark.parametrize("operation", ["commit", "recovery", "rollback"])
def test_marker_removal_failure_preserves_backups_and_does_not_claim_clean_success(
    monkeypatch, transaction, operation
):
    root, targets, old, new = transaction
    if operation == "recovery":
        interrupted_transaction(transaction)
    real_unlink = Path.unlink
    real_replace = os.replace

    def replace(source, target):
        if operation == "rollback" and Path(target) == targets[1]:
            raise OSError("promotion failed before marker cleanup")
        return real_replace(source, target)

    def unlink(path, *args, **kwargs):
        if path.name.startswith(".commit-") and path.suffix == ".json":
            raise windows_error()
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", unlink)
        patch.setattr(atomic.os, "replace", replace)
        if operation != "recovery":
            with pytest.raises(OSError):
                persistence.commit_files(list(zip(targets, new)), root=root)
        else:
            report = persistence.recover_root(root)
            assert not report.ok and report.markers_removed == 0
    assert len(persistence.marker_paths(root)) == 1
    assert len(list(root.glob(".*.backup"))) == 2
    # A retained older journal cannot be allowed to erase a later accepted write.
    with pytest.raises(atomic.MetadataRecoveryRequiredError):
        persistence.commit_files(list(zip(targets, new)), root=root)
    assert persistence.recover_root(root).ok
    assert [target.read_bytes() for target in targets] == old
    assert not residue(root)


@pytest.mark.parametrize(
    "writer", ["transaction", "project", "segments", "segmentation"]
)
def test_pending_recovery_blocks_every_canonical_writer_until_explicit_recovery(
    monkeypatch, tmp_path, writer
):
    use_projects_root(monkeypatch, tmp_path)
    project = build_project(name="original")
    segments = build_segments(project, ["Original line."])
    root = safe_project_root(project.id)
    targets = [root / "project.json", root / "segments.json"]
    old = [target.read_bytes() for target in targets]
    project.name = "pending update"
    segments[0].text = "Pending update."
    pending = [
        project.model_dump_json().encode(),
        json.dumps([segment.model_dump(mode="json") for segment in segments]).encode(),
    ]
    marker, backups = interrupted_transaction((root, targets, old, pending))
    project.name = "later update"
    segments[0].text = "Later update."
    later = [
        project.model_dump_json().encode(),
        json.dumps([segment.model_dump(mode="json") for segment in segments]).encode(),
    ]

    def write():
        if writer == "transaction":
            return persistence.commit_files(list(zip(targets, later)), root=root)
        if writer == "project":
            return save_project(project)
        if writer == "segments":
            return save_segments(project.id, segments)
        return _replace_files_transactionally(list(zip(targets, later)))

    with pytest.raises(atomic.MetadataRecoveryRequiredError, match="verify.*--recover"):
        write()
    assert [target.read_bytes() for target in targets] == pending
    assert marker.exists() and [backup.read_bytes() for backup in backups] == old
    assert not list(root.glob(".*.stage")) and not list(root.glob("*.tmp"))
    assert persistence.recover_root(root).ok
    assert [target.read_bytes() for target in targets] == old
    write()
    expected_project = "original" if writer == "segments" else "later update"
    expected_line = "Original line." if writer == "project" else "Later update."
    assert json.loads(targets[0].read_bytes())["name"] == expected_project
    assert json.loads(targets[1].read_bytes())[0]["text"] == expected_line
    assert persistence.recover_root(root).clean
    assert not residue(root)


def test_partial_in_process_rollback_can_repeat_every_restore_later(
    monkeypatch, transaction
):
    root, targets, old, new = transaction
    third = root / "cues.json"
    third.write_bytes(b"[]")
    targets.append(third)
    old.append(b"[]")
    new.append(b'["cue"]')
    real_replace = os.replace

    def replace(source, target):
        if Path(target) == targets[2] or (
            Path(target) == targets[0] and Path(source).read_bytes() == old[0]
        ):
            raise windows_error()
        real_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(atomic.os, "replace", replace)
        patch.setattr(atomic.time, "sleep", lambda delay: None)
        with pytest.raises(PermissionError):
            persistence.commit_files(list(zip(targets, new)), root=root)
    assert [target.read_bytes() for target in targets] == [new[0], old[1], old[2]]
    assert len(list(root.glob(".*.backup"))) == 3
    assert persistence.marker_paths(root)
    assert persistence.recover_root(root).ok
    assert [target.read_bytes() for target in targets] == old
    assert not residue(root)


def test_missing_backup_preserves_journal_and_available_backup(transaction):
    root, _targets, old, _new = transaction
    marker, backups = interrupted_transaction(transaction)
    backups[1].unlink()
    report = persistence.recover_root(root)
    assert not report.ok and report.markers_removed == 0
    assert any("backup missing" in item for item in report.leftovers)
    assert marker.exists() and backups[0].read_bytes() == old[0]


def test_unreadable_journal_does_not_destroy_its_backups(transaction):
    root, _targets, old, _new = transaction
    marker, backups = interrupted_transaction(transaction)
    marker.write_text("{broken", encoding="utf-8")
    report = persistence.recover_root(root)
    assert not report.ok and report.markers_removed == 0
    assert marker.exists() and [backup.read_bytes() for backup in backups] == old


def test_non_windows_promotion_error_is_not_retried(monkeypatch, transaction):
    root, targets, old, new = transaction
    real_replace = os.replace
    calls, delays = [], []
    failure = OSError(errno.ENOSPC, "disk full")

    def replace(source, target):
        if Path(target) == targets[1]:
            calls.append(source)
            raise failure
        real_replace(source, target)

    monkeypatch.setattr(atomic.os, "replace", replace)
    monkeypatch.setattr(atomic.time, "sleep", delays.append)
    with pytest.raises(OSError) as raised:
        persistence.commit_files(list(zip(targets, new)), root=root)
    assert raised.value is failure and len(calls) == 1 and not delays
    assert [target.read_bytes() for target in targets] == old
    assert not residue(root)
