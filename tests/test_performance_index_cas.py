"""M4 index writes: expected revisions, immutable stores and rollback.

Every case runs in a disposable pytest root.  ``test_the_effective_projects_root
_is_the_disposable_pytest_root`` proves that prerequisite before the rest rely on
it, and no case reads or writes a real project.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dubbing_service.app.audiobook import performance_director as director
from dubbing_service.app.audiobook.fingerprint import canonical_json_bytes, sha256_bytes, text_sha256
from dubbing_service.app.models.project import Segment, SourceAnchor
from dubbing_service.app.storage.projects import load_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audiobook_helpers import build_project, build_segments, use_projects_root

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A canonical source revision is a SHA-256, not a label; the scene proofs below
#: refuse anything else, exactly as the production owner does.
SOURCE_REVISION = "c1" * 32


def _capability_sha() -> str:
    return sha256_bytes(canonical_json_bytes({"synthetic": "capability"}))


def _one_row(segment) -> director.UtteranceDirection:
    return director.UtteranceDirection(
        utterance_id=segment.id,
        text_sha256=text_sha256(segment.text),
        role_id=segment.role_id or "",
        sources=director.DirectionSources(capability_sha256=_capability_sha(),
                                          resolver_version=director.RESOLVER_VERSION),
    )


def _direction_set(project_id: str, segments) -> director.DirectionSet:
    return director.DirectionSet(project_id=project_id,
                                 directions=[_one_row(segment) for segment in segments])


@pytest.fixture()
def project(tmp_path, monkeypatch):
    root = use_projects_root(monkeypatch, tmp_path)
    built = build_project(name="direction-book")
    segments = build_segments(built, ("First line.", "Second line.", "Third line."))
    for index, segment in enumerate(segments):
        segment.source_anchor = SourceAnchor(source_revision_id=SOURCE_REVISION)
        segment.index = index
    save_segments(built.id, segments)
    return {"root": root, "project": load_project(built.id),
            "segments": load_segments(built.id)}


def _activate_direction_set(project, direction_set, *, expected_revision, event_id="evt-1",
                           actor="synthetic"):
    stored = director.write_direction_set(project["project"].id, direction_set)

    def change(index: dict) -> None:
        index["active"]["direction_set"] = stored["sha256"]

    return director.commit_index(project["project"].id, expected_revision, change,
                                 actor=actor, event_id=event_id)


# ------------------------------------------------------------------ prerequisites


def test_the_effective_projects_root_is_the_disposable_pytest_root(project, tmp_path) -> None:
    """Every other case depends on the writes landing in the disposable root."""

    from dubbing_service.app.core.config import get_config

    configured = Path(get_config().projects_root_abs).resolve()
    assert configured == Path(project["root"]).resolve()
    assert str(tmp_path) in str(configured)
    assert director._project_root(project["project"].id).is_relative_to(configured)


# ------------------------------------------------------------------- CAS behaviour


def test_an_expected_revision_write_increments_once(project) -> None:
    index = director.read_index(project["project"].metadata) or director.default_index()
    assert index.revision == 0
    committed = _activate_direction_set(project, _direction_set(project["project"].id,
                                                               project["segments"]),
                                        expected_revision=0)
    assert committed["revision"] == 1
    assert committed["active"]["direction_set"]
    assert committed["schema_version"] == "performance-director-index-v1"
    stored = director.read_index(load_project(project["project"].id).metadata)
    assert stored.revision == 1
    assert stored.active.direction_set == committed["active"]["direction_set"]
    assert len(committed["history"]) == 1
    entry = committed["history"][0]
    assert entry["actor"] == "synthetic" and entry["event_id"] == "evt-1"
    assert entry["previous"]["direction_set"] is None
    assert entry["active"]["direction_set"] == committed["active"]["direction_set"]


def test_a_stale_expected_revision_is_refused_and_changes_nothing(project) -> None:
    committed = _activate_direction_set(project, _direction_set(project["project"].id,
                                                               project["segments"]),
                                        expected_revision=0, event_id="evt-1")
    before = load_project(project["project"].id).metadata[director.INDEX_KEY]
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.commit_index(project["project"].id, 0,
                              lambda index: index["active"].update({"direction_set": None}),
                              actor="synthetic", event_id="evt-2")
    assert refused.value.code == "plan_stale"
    after = load_project(project["project"].id).metadata[director.INDEX_KEY]
    assert after == before
    assert after["active"]["direction_set"] == committed["active"]["direction_set"]


def test_two_writers_with_the_same_revision_cannot_both_win(project) -> None:
    first = _direction_set(project["project"].id, project["segments"])
    second = director.DirectionSet(project_id=project["project"].id, directions=[])

    _activate_direction_set(project, first, expected_revision=0, event_id="evt-a")
    # The same expected revision is now stale, so the second writer is refused
    # rather than overwriting the first.
    with pytest.raises(director.PerformanceDirectorError) as refused:
        _activate_direction_set(project, second, expected_revision=0, event_id="evt-b")
    assert refused.value.code == "plan_stale"
    stored = director.read_index(load_project(project["project"].id).metadata)
    assert stored.revision == 1
    assert stored.active.direction_set


def test_a_duplicate_event_id_with_the_same_payload_returns_the_prior_result(project) -> None:
    direction_set = _direction_set(project["project"].id, project["segments"])
    first = _activate_direction_set(project, direction_set, expected_revision=0, event_id="evt-x")
    again = _activate_direction_set(project, direction_set, expected_revision=0, event_id="evt-x")
    assert again == first
    stored = director.read_index(load_project(project["project"].id).metadata)
    assert stored.revision == 1, "an idempotent retry must not increment twice"
    assert len(stored.history) == 1


def test_a_duplicate_event_id_with_a_changed_payload_conflicts(project) -> None:
    direction_set = _direction_set(project["project"].id, project["segments"])
    _activate_direction_set(project, direction_set, expected_revision=0, event_id="evt-y")

    other = director.DirectionSet(project_id=project["project"].id, directions=[])
    stored = director.write_direction_set(project["project"].id, other)
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.commit_index(project["project"].id, 0,
                              lambda index: index["active"].update(
                                  {"direction_set": stored["sha256"]}),
                              actor="synthetic", event_id="evt-y")
    assert refused.value.code == "direction_conflict"
    assert "different payload" in str(refused.value)


def test_the_change_callback_cannot_rewrite_identity_revision_or_history(project) -> None:
    for field, mutate in (("schema_version", {"schema_version": "performance-director-index-v9"}),
                          ("revision", {"revision": 99}),
                          ("history", {"history": [{"event_id": "invented"}]})):
        def change(index: dict, mutate=mutate) -> None:
            index.update(mutate)

        with pytest.raises(director.PerformanceDirectorError) as refused:
            director.commit_index(project["project"].id, 0, change, actor="synthetic",
                                  event_id=f"evt-{field}")
        assert refused.value.code == "direction_conflict"
        assert field in str(refused.value)
    stored = load_project(project["project"].id).metadata.get(director.INDEX_KEY)
    assert stored is None, "a refused change must not have been committed"


def test_a_failed_write_leaves_the_previous_pointer_and_evidence_intact(project, monkeypatch) -> None:
    first = _activate_direction_set(project, _direction_set(project["project"].id,
                                                           project["segments"]),
                                    expected_revision=0, event_id="evt-1")
    before = load_project(project["project"].id).metadata[director.INDEX_KEY]

    def boom(_project) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr("dubbing_service.app.storage.projects.save_project", boom)
    stored = director.write_direction_set(project["project"].id,
                                          director.DirectionSet(project_id=project["project"].id,
                                                                directions=[]))
    with pytest.raises(OSError):
        director.commit_index(project["project"].id, 1,
                              lambda index: index["active"].update(
                                  {"direction_set": stored["sha256"]}),
                              actor="synthetic", event_id="evt-fail")
    monkeypatch.undo()
    after = load_project(project["project"].id).metadata[director.INDEX_KEY]
    assert after == before
    assert after["active"]["direction_set"] == first["active"]["direction_set"]
    # The immutable evidence of the attempt is still there.
    assert Path(stored["path"]).is_file()


def test_the_index_cannot_activate_a_missing_or_mismatched_artifact(project) -> None:
    missing = "a" * 64
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.commit_index(project["project"].id, 0,
                              lambda index: index["active"].update({"direction_set": missing}),
                              actor="synthetic", event_id="evt-missing")
    assert refused.value.code == "plan_stale"
    assert "does not exist" in str(refused.value)

    # A file under the hash name whose content does not hash to that name.
    root = director._project_root(project["project"].id)
    planted = root / director.DIRECTION_DIRECTORY / f"{missing}.json"
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text(json.dumps({"schema_version": "direction-set-v1"}), encoding="utf-8")
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.commit_index(project["project"].id, 0,
                              lambda index: index["active"].update({"direction_set": missing}),
                              actor="synthetic", event_id="evt-mismatch")
    assert refused.value.code == "plan_stale"
    assert "does not match" in str(refused.value)


def test_the_index_cannot_activate_a_malformed_artifact(project) -> None:
    root = director._project_root(project["project"].id)
    payload = {"schema_version": "direction-set-v1", "project_id": project["project"].id,
               "directions": [{"utterance_id": "seg000", "text_sha256": "0" * 64,
                               "role_id": "narrator",
                               "unexpected_field": True,
                               "sources": {"capability_sha256": "0" * 64,
                                           "resolver_version": "x"}}]}
    sha = director.content_sha256(payload)
    path = root / director.DIRECTION_DIRECTORY / f"{sha}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.commit_index(project["project"].id, 0,
                              lambda index: index["active"].update({"direction_set": sha}),
                              actor="synthetic", event_id="evt-malformed")
    assert refused.value.code == "schema_unknown"


def test_a_profile_pointer_is_verified_against_its_own_content(project) -> None:
    profile = director.compile_profile("narrator", narrator=True)
    stored = director.write_delivery_profile(project["project"].id, profile)
    committed = director.commit_index(
        project["project"].id, 0,
        lambda index: index["active"]["profiles"].update({"narrator": stored["sha256"]}),
        actor="synthetic", event_id="evt-profile")
    assert committed["active"]["profiles"]["narrator"] == stored["sha256"]
    read_back = director.read_delivery_profile(project["project"].id, "narrator", stored["sha256"])
    assert read_back.narrator_preservation is True
    assert read_back.role_id == "narrator"

    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.commit_index(project["project"].id, 1,
                              lambda index: index["active"]["profiles"].update(
                                  {"narrator": "b" * 64}),
                              actor="synthetic", event_id="evt-profile-missing")
    assert refused.value.code == "plan_stale"


def test_the_stores_are_immutable_and_reuse_is_verified_by_content(project) -> None:
    direction_set = _direction_set(project["project"].id, project["segments"])
    first = director.write_direction_set(project["project"].id, direction_set)
    assert first["created"] is True
    again = director.write_direction_set(project["project"].id, direction_set)
    assert again["created"] is False and again["sha256"] == first["sha256"]

    # A timestamp is not part of identity, so re-storing with a different one is
    # still verified reuse rather than a byte conflict.
    stamped = director.DirectionSet(project_id=project["project"].id,
                                    directions=direction_set.directions,
                                    created_at="2026-10-07T00:00:00+00:00")
    assert director.content_sha256(stamped) == first["sha256"]
    reused = director.write_direction_set(project["project"].id, stamped)
    assert reused["created"] is False

    # Corrupting the stored artifact is refused, never overwritten.
    Path(first["path"]).write_text(json.dumps({"schema_version": "tampered"}), encoding="utf-8")
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.write_direction_set(project["project"].id, direction_set)
    assert refused.value.code == "direction_conflict"


def test_a_profile_record_is_content_addressed_per_role(project) -> None:
    profile = director.compile_profile("narrator", narrator=False)
    stored = director.write_delivery_profile(project["project"].id, profile)
    assert stored["relative"].startswith("performance/profiles/narrator/")
    assert stored["relative"].endswith(f"{stored['sha256']}.json")
    assert Path(stored["path"]).is_relative_to(director._project_root(project["project"].id))


def test_traversal_separator_and_dot_path_identifiers_are_refused() -> None:
    for bad in ("..", ".", "a/b", "a\\b", "  pad", "pad  ", "pad.", "", "nul\x00",
                "con:name", 'quote"name'):
        with pytest.raises(director.PerformanceDirectorError):
            director._require_component(bad, "role_id")
    assert director._require_component("narrator", "role_id") == "narrator"


def test_a_symlinked_performance_directory_cannot_escape_the_project_root(project, tmp_path) -> None:
    root = director._project_root(project["project"].id)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "performance"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this environment does not allow creating symlinks")
    profile = director.compile_profile("narrator", narrator=False)
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.write_delivery_profile(project["project"].id, profile)
    assert refused.value.code == "plan_stale"
    assert "escapes" in str(refused.value)
    assert not list(outside.iterdir()), "nothing may be written outside the project root"


def test_the_written_index_survives_a_restart(project) -> None:
    committed = _activate_direction_set(project, _direction_set(project["project"].id,
                                                               project["segments"]),
                                        expected_revision=0)
    code = ("import json, sys;"
            "from dubbing_service.app.storage.projects import load_project;"
            "from dubbing_service.app.audiobook.performance_director import read_index;"
            "project = load_project(sys.argv[1]);"
            "print(json.dumps(read_index(project.metadata).model_dump(mode='json'),"
            " sort_keys=True))")
    result = subprocess.run([sys.executable, "-B", "-c", code, project["project"].id],
                            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180,
                            env={**os.environ})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == committed


def test_a_competing_writer_owner_is_refused_across_processes(project) -> None:
    """A second process holding the root lock blocks the commit; nothing changes."""

    from dubbing_service.app.storage.process_lock import LockBusyError

    holder = (
        "import sys, time;"
        "from pathlib import Path;"
        "from dubbing_service.app.storage.process_lock import acquire_projects_root;"
        "lock = acquire_projects_root(Path(sys.argv[1]), label='competing-owner');"
        "print('ready', flush=True);"
        "time.sleep(20);"
        "lock.release()"
    )
    child = subprocess.Popen([sys.executable, "-B", "-c", holder, str(project["root"])],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises((LockBusyError, director.PerformanceDirectorError)):
            director.commit_index(project["project"].id, 0,
                                  lambda index: index.update({"enabled": True}),
                                  actor="synthetic", event_id="evt-blocked")
    finally:
        child.terminate()
        child.wait(timeout=20)
    assert load_project(project["project"].id).metadata.get(director.INDEX_KEY) is None


def test_the_history_is_bounded_and_keeps_the_newest(project, monkeypatch) -> None:
    monkeypatch.setattr(director, "INDEX_HISTORY_LIMIT", 3)
    for step in range(5):
        director.commit_index(project["project"].id, step,
                              lambda index, step=step: index.update({"pilot": f"pilot-{step}"}),
                              actor="synthetic", event_id=f"evt-{step}")
    stored = director.read_index(load_project(project["project"].id).metadata)
    assert stored.revision == 5
    assert len(stored.history) == 3
    assert [entry["event_id"] for entry in stored.history] == ["evt-2", "evt-3", "evt-4"]


def test_the_segment_fixture_keeps_its_canonical_source_anchor(project) -> None:
    """A guard on the fixture itself: the proofs the other files rely on are real."""

    for segment in project["segments"]:
        assert isinstance(segment, Segment)
        assert segment.source_anchor.source_revision_id == SOURCE_REVISION
    proofs = director.scene_proofs(project["project"],
                                   [segment.id for segment in project["segments"]])
    assert proofs["source_revision_sha256"] == SOURCE_REVISION
    assert len(proofs["utterances"]) == 3
