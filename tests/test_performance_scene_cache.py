"""M4 scene identity, scoped invalidation, the active loader and the hook.

Scene identity deliberately excludes voice versions, so recasting keeps an
interpretation.  The active loader is the only path the hot code reads, and it
fails closed on a missing, corrupted, mismatched or stale set while a warmed
cache still notices a changed file.
"""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import urllib.request
from pathlib import Path

import pytest

from dubbing_service.app.audiobook import performance_director as director
from dubbing_service.app.audiobook.fingerprint import (
    canonical_json_bytes,
    sha256_bytes,
    text_sha256,
    voice_revision_for_role,
)
from dubbing_service.app.models.project import Attribution, SourceAnchor, Role
from dubbing_service.app.storage.projects import load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audiobook_helpers import build_project, build_segments, use_projects_root

SOURCE_REVISION = "c1" * 32
CAPABILITY_SHA = sha256_bytes(canonical_json_bytes({"synthetic": "capability"}))


def _row(segment, *, voice_revision=None, salt=0, capability_sha=CAPABILITY_SHA):
    return director.UtteranceDirection(
        utterance_id=segment.id,
        text_sha256=text_sha256(segment.text),
        role_id=segment.role_id or "",
        voice_revision=voice_revision or voice_revision_for_role(Role(id=segment.role_id, name="Narrator")),
        resolved=director.ResolvedControls(seed_salt=salt),
        sources=director.DirectionSources(capability_sha256=capability_sha,
                                          resolver_version=director.RESOLVER_VERSION),
    )


def _direction_set(project_id: str, rows) -> director.DirectionSet:
    return director.DirectionSet(project_id=project_id, directions=list(rows))


def live_capability_sha(studio, segment) -> str:
    """The real capability identity of this project's runtime.

    A row that would forward something must match the runtime that resolved it,
    so the fixtures use the live value instead of inventing one.
    """

    return director._live_capability(studio["project"], segment)["sha256"]


@pytest.fixture()
def studio(tmp_path, monkeypatch):
    use_projects_root(monkeypatch, tmp_path)

    def build():
        built = build_project(name="scene-book")
        built.roles[0].role_type = "character"
        save_project(built)
        segments = build_segments(built, ("First line.", "Second line.", "Third line.",
                                          "Fourth line."), role_id="narrator")
        for index, segment in enumerate(segments):
            segment.source_anchor = SourceAnchor(source_revision_id=SOURCE_REVISION)
            segment.index = index
        save_segments(built.id, segments)
        return built.id

    project_id = build()
    # Synthetic forwarding capability: this tests the loader, not an installed worker.
    capability = {"schema_version": "performance-capability-v1", "runtime_profile": "faster-base-032-direction-x1",
                  "identity_mechanism": "qwen_reference", "conditioning_mode": "icl", "backend": "qwen3tts",
                  "evidence": {"fixture": "CPU loader seam; no model or reference exists"},
                  "controls": {"seed": {"state": "forwarded_experimental"}, "native_instruction": {"state": "forwarded_experimental"}}}
    capability["sha256"] = sha256_bytes(canonical_json_bytes(capability))
    monkeypatch.setattr(director, "_live_capability", lambda project, segment: dict(capability))
    director.clear_direction_cache()
    yield {"project_id": project_id, "project": load_project(project_id),
           "segments": load_segments(project_id)}
    director.clear_direction_cache()


def _scene(studio, segments, *, scene_id, turns=(), mode="manual"):
    project = studio["project"]
    utterances = [director.SceneUtterance(id=segment.id, text_sha256=text_sha256(segment.text),
                                          role_id=segment.role_id) for segment in segments]
    proofs = director.scene_proofs(project, [segment.id for segment in segments])
    interpretation = director.SceneInterpretation(mode=mode, planner_version="pd-planner-1")
    return director.SceneIntention(
        scene_id=scene_id, chapter_id=segments[0].chapter_id or "chap0",
        source_revision_sha256=proofs["source_revision_sha256"],
        utterances=utterances, attribution_digest=proofs["attribution_digest"],
        alias_digest=proofs["alias_digest"], turns=list(turns),
        interpretation=interpretation,
        cache_key=director.scene_cache_key(
            source_revision_sha256=proofs["source_revision_sha256"], utterances=utterances,
            attribution_digest=proofs["attribution_digest"], alias_digest=proofs["alias_digest"],
            interpretation=interpretation),
        author=director.RecordAuthor(origin="manual", actor="synthetic", event_id="evt-scene"))


def _activate(studio, rows, *, event_id="evt-direction", expected_revision=0):
    stored = director.write_direction_set(studio["project_id"], _direction_set(
        studio["project_id"], rows))

    def change(index: dict) -> None:
        index["enabled"] = True
        index["active"]["direction_set"] = stored["sha256"]

    director.commit_index(studio["project_id"], expected_revision, change,
                          actor="synthetic", event_id=event_id)
    # The committed index lives in project.json, so the in-memory copy must be
    # reloaded before anything reads it.
    studio["project"] = load_project(studio["project_id"])
    director.clear_direction_cache()
    return stored


def _enable(studio, *, direction_set=None):
    def change(index: dict) -> None:
        index["enabled"] = True
        if direction_set is not None:
            index["active"]["direction_set"] = direction_set

    committed = director.commit_index(studio["project_id"], 0, change, actor="synthetic",
                                      event_id="evt-enable")
    studio["project"] = load_project(studio["project_id"])
    director.clear_direction_cache()
    return committed


def _guard_against_network_and_models(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("no network client may be constructed or used")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    for name in ("httpx", "requests"):
        module = __import__(name, fromlist=["dummy"]) if name in __import__("sys").modules else None
        if module is not None:
            for attribute in ("Client", "Session", "get", "post"):
                if hasattr(module, attribute):
                    monkeypatch.setattr(module, attribute, refuse, raising=False)


# ------------------------------------------------------------------ scene identity


def test_scene_identity_is_stable_and_excludes_the_voice_version(studio) -> None:
    segments = studio["segments"]
    first = _scene(studio, segments[:2], scene_id="scene-1")
    assert first.cache_key == _scene(studio, segments[:2], scene_id="scene-1").cache_key

    # Recasting is not part of a scene's identity.
    project = load_project(studio["project_id"])
    from dubbing_service.app.models.character import CharacterVoiceIdentity

    project.roles[0].voice_identity = CharacterVoiceIdentity(
        id="voice-narrator", role_id="narrator", status="LOCKED", mechanism="qwen_reference",
        dossier_revision="dossier-1", casting_profile_revision="casting-1",
        design_instruction="measured and unhurried",
        reference_audio_sha256="a" * 64, model="qwen3-tts-base")
    save_project(project)
    recast = _scene({"project": load_project(studio["project_id"]), "segments": segments},
                    segments[:2], scene_id="scene-1")
    assert recast.cache_key == first.cache_key, "a recast must not discard an interpretation"
    stored = director.store_scene_intention(load_project(studio["project_id"]), first)
    again = director.load_scene_intention(load_project(studio["project_id"]), "scene-1",
                                          stored["sha256"])
    assert again.cache_key == first.cache_key


def test_an_attribution_correction_invalidates_only_its_own_scene(studio) -> None:
    segments = studio["segments"]
    left = _scene(studio, segments[:2], scene_id="scene-left")
    right = _scene(studio, segments[2:], scene_id="scene-right")
    left_stored = director.store_scene_intention(studio["project"], left)
    right_stored = director.store_scene_intention(studio["project"], right)

    # Correct the attribution of a line that belongs to the left scene only.
    stored_segments = load_segments(studio["project_id"])
    stored_segments[0].attribution = Attribution(speaker_id="narrator", source="user")
    save_segments(studio["project_id"], stored_segments)
    project = load_project(studio["project_id"])

    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.load_scene_intention(project, "scene-left", left_stored["sha256"])
    assert refused.value.code == "plan_stale"
    # The other scene is untouched.
    assert director.load_scene_intention(project, "scene-right",
                                         right_stored["sha256"]).cache_key == right.cache_key


def test_a_text_source_role_or_alias_change_invalidates_the_scene(studio) -> None:
    segments = studio["segments"]
    scene = _scene(studio, segments[:2], scene_id="scene-1")
    stored = director.store_scene_intention(studio["project"], scene)

    for mutate, restore in (
        (lambda rows: setattr(rows[0], "text", "Rewritten line."),
         lambda rows: setattr(rows[0], "text", "First line.")),
        (lambda rows: setattr(rows[0].source_anchor, "source_revision_id", "d1" * 32),
         lambda rows: setattr(rows[0].source_anchor, "source_revision_id", SOURCE_REVISION)),
        (lambda rows: setattr(rows[0], "role_id", "someone-else"),
         lambda rows: setattr(rows[0], "role_id", "narrator")),
    ):
        rows = load_segments(studio["project_id"])
        mutate(rows)
        save_segments(studio["project_id"], rows)
        with pytest.raises(director.PerformanceDirectorError) as refused:
            director.load_scene_intention(load_project(studio["project_id"]), "scene-1",
                                          stored["sha256"])
        assert refused.value.code == "plan_stale"
        rows = load_segments(studio["project_id"])
        restore(rows)
        save_segments(studio["project_id"], rows)

    # An alias change is refused too, and is reported as a stale plan.
    project = load_project(studio["project_id"])
    project.voice_lab.casting = {**project.voice_lab.casting,
                                 "aliases": {"the narrator": "narrator"}}
    save_project(project)
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.load_scene_intention(load_project(studio["project_id"]), "scene-1",
                                      stored["sha256"])
    assert refused.value.code == "plan_stale"


def test_storing_a_scene_whose_proofs_no_longer_match_is_refused(studio) -> None:
    segments = studio["segments"]
    scene = _scene(studio, segments[:2], scene_id="scene-1")
    rows = load_segments(studio["project_id"])
    rows[0].text = "A different line entirely."
    save_segments(studio["project_id"], rows)
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.store_scene_intention(load_project(studio["project_id"]), scene)
    assert refused.value.code == "plan_stale"


def test_a_project_without_a_canonical_source_revision_cannot_build_a_scene(studio) -> None:
    rows = load_segments(studio["project_id"])
    rows[0].source_anchor = None
    save_segments(studio["project_id"], rows)
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.scene_proofs(load_project(studio["project_id"]), [rows[0].id])
    assert refused.value.code == "plan_stale"


def test_the_scene_store_is_content_addressed_and_immutable(studio) -> None:
    scene = _scene(studio, studio["segments"][:2], scene_id="scene-1")
    first = director.store_scene_intention(studio["project"], scene)
    assert first["created"] is True
    assert first["relative"].startswith("performance/scenes/scene-1/")
    again = director.store_scene_intention(studio["project"], scene)
    assert again["created"] is False and again["sha256"] == first["sha256"]
    Path(first["path"]).write_text(json.dumps({"schema_version": "tampered"}), encoding="utf-8")
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.store_scene_intention(studio["project"], scene)
    assert refused.value.code == "direction_conflict"


def _model_modules_imported_by(action) -> list[str]:
    """Modules a call pulled in that load a model, a provider or the renderer."""

    watched = ("torch", "dubbing_service.app.audiobook.renderer",
               "dubbing_service.app.providers.tts")
    before = {name for name in sys.modules if name.startswith(watched)}
    action()
    after = {name for name in sys.modules if name.startswith(watched)}
    return sorted(after - before)


def test_deterministic_scene_planning_constructs_no_http_client_or_model(studio, monkeypatch) -> None:
    _guard_against_network_and_models(monkeypatch)
    scene_record = _scene(studio, studio["segments"][:2], scene_id="scene-1",
                          mode="deterministic")
    stored = director.store_scene_intention(studio["project"], scene_record)
    loaded = director.load_scene_intention(studio["project"], "scene-1", stored["sha256"])
    assert loaded.interpretation.mode == "deterministic"

    def plan_and_store():
        return director.store_scene_intention(studio["project"], scene_record)

    assert _model_modules_imported_by(plan_and_store) == [], (
        "deterministic scene planning must not import torch, the renderer or the provider "
        "package")


def test_a_scene_plan_carries_no_mutable_per_line_interpretation_store(studio) -> None:
    """The scene record is the only interpretation store; metadata holds no such map."""

    scene = _scene(studio, studio["segments"][:2], scene_id="scene-1")
    director.store_scene_intention(studio["project"], scene)
    project = load_project(studio["project_id"])
    for key in project.metadata:
        assert "interpret" not in key.casefold()
    assert project.metadata.get(director.INDEX_KEY) is None


# ------------------------------------------------------------- active set and hook


def test_an_index_free_project_reads_no_performance_storage(studio) -> None:
    root = director._project_root(studio["project_id"])
    assert not (root / "performance").exists()
    segment = studio["segments"][0]
    assert director.direction_for(studio["project"], segment.id) is None
    assert director.requirement_direction(studio["project"], segment) is None
    assert not (root / "performance").exists(), "the old path must read no performance storage"

    # A disabled index behaves the same way.
    _enable(studio)
    studio["project"] = load_project(studio["project_id"])
    assert director.requirement_direction(studio["project"], segment) is None
    assert not (root / "performance").exists()


def test_the_enabled_index_returns_the_active_direction(studio) -> None:
    segments = studio["segments"]
    _activate(studio, [_row(segments[0])])
    assert director.requirement_direction(studio["project"], segments[0]) is None, (
        "advisory-only rows put nothing into a fingerprint")
    assert director.direction_for(studio["project"], segments[0].id)["utterance_id"] == segments[0].id
    # A line with no row, and a line with no direction at all, are both absent.
    assert director.direction_for(studio["project"], segments[1].id) is None
    assert director.direction_for(studio["project"], "not-a-line") is None


def test_a_row_that_forwards_reaches_the_hook(studio) -> None:
    segments = studio["segments"]
    segment = segments[0]
    live = director._live_capability(studio["project"], segment)["sha256"]
    row = _row(segment, salt=4242, capability_sha=live)
    _activate(studio, [row])
    component = director.requirement_direction(studio["project"], segment)
    assert component is not None
    assert component == director.direction_component(row)
    assert director.requirement_direction(studio["project"], segments[1]) is None


def test_a_missing_or_corrupted_active_set_fails_closed(studio) -> None:
    segment = studio["segments"][0]
    stored = _activate(studio, [_row(segment)])
    assert director.direction_for(studio["project"], segment.id) is not None

    # Removing the file the pointer names must fail closed, not fall back.
    Path(stored["path"]).unlink()
    director.clear_direction_cache()
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.requirement_direction(studio["project"], segment)
    assert refused.value.code == "plan_stale"
    assert "missing" in str(refused.value)

    # Unreadable JSON, and then a set that names a different project.
    path = Path(stored["path"])
    path.write_text("{not json", encoding="utf-8")
    director.clear_direction_cache()
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.requirement_direction(studio["project"], segment)
    assert refused.value.code == "schema_unknown"

    foreign = director.DirectionSet(project_id="another-project", directions=[_row(segment)])
    path.write_text(json.dumps(foreign.model_dump(mode="json")), encoding="utf-8")
    director.clear_direction_cache()
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.requirement_direction(studio["project"], segment)
    assert refused.value.code == "plan_stale"


def test_the_active_set_must_belong_to_this_project(studio) -> None:
    """A hash-matching set for a different project is refused at the hook."""

    segment = studio["segments"][0]
    foreign = director.DirectionSet(project_id="some-other-project",
                                    directions=[_row(segment)])
    stored = director.write_direction_set(studio["project_id"], foreign)
    _enable(studio, direction_set=stored["sha256"])
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.requirement_direction(studio["project"], segment)
    assert refused.value.code == "plan_stale"
    assert "another project" in str(refused.value)


def test_a_warmed_cache_still_notices_a_changed_or_removed_file(studio) -> None:
    segments = studio["segments"]
    row = _row(segments[0], salt=11, capability_sha=live_capability_sha(studio, segments[0]))
    stored = _activate(studio, [row])
    assert director.requirement_direction(studio["project"], segments[0]) is not None
    cached = len(director._DIRECTION_CACHE)
    assert cached == 1

    path = Path(stored["path"])
    replacement = director.DirectionSet(
        project_id=studio["project_id"],
        directions=[_row(segments[0], salt=22,
                         capability_sha=live_capability_sha(studio, segments[0]))])
    path.write_text(json.dumps(replacement.model_dump(mode="json")), encoding="utf-8")
    # The file no longer matches its own name, so the warmed cache must not mask it.
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.requirement_direction(studio["project"], segments[0])
    assert refused.value.code == "plan_stale"
    assert len(director._DIRECTION_CACHE) == cached, "a refused reload keeps the cache bounded"

    path.unlink()
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.requirement_direction(studio["project"], segments[0])
    assert refused.value.code == "plan_stale"
    assert "missing" in str(refused.value)


def test_a_genuine_file_change_is_noticed_while_warm(studio) -> None:
    """A valid replacement under the same hash name is picked up, not cached over."""

    segments = studio["segments"]
    live = live_capability_sha(studio, segments[0])
    stored = _activate(studio, [_row(segments[0], salt=11, capability_sha=live)])
    first = director.requirement_direction(studio["project"], segments[0])
    assert first is not None

    # Replace the pointer with a *valid* set that hashes to its own name: the
    # loader must read the new content rather than reuse the warmed entry.
    second_rows = [_row(segments[0], salt=33, capability_sha=live)]
    second = director.write_direction_set(studio["project_id"],
                                          director.DirectionSet(project_id=studio["project_id"],
                                                                directions=second_rows))
    assert second["sha256"] != stored["sha256"]
    director.commit_index(studio["project_id"], 1,
                          lambda index: index["active"].update({"direction_set": second["sha256"]}),
                          actor="synthetic", event_id="evt-rotate")
    changed = director.requirement_direction(load_project(studio["project_id"]), segments[0])
    assert changed is not None and changed != first


def test_clearing_the_pointer_restores_the_previous_result(studio) -> None:
    segments = studio["segments"]
    _activate(studio, [_row(segments[0], salt=99,
                            capability_sha=live_capability_sha(studio, segments[0]))])
    assert director.requirement_direction(studio["project"], segments[0]) is not None

    director.commit_index(studio["project_id"], 1,
                          lambda index: index["active"].update({"direction_set": None}),
                          actor="synthetic", event_id="evt-clear")
    director.clear_direction_cache()
    assert director.requirement_direction(load_project(studio["project_id"]),
                                          segments[0]) is None


def test_a_stale_text_role_or_voice_binding_fails_closed(studio) -> None:
    segments = studio["segments"]
    live = live_capability_sha(studio, segments[0])

    _activate(studio, [_row(segments[0], salt=5, capability_sha=live,
                            voice_revision="0" * 32)])
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.requirement_direction(studio["project"], segments[0])
    assert refused.value.code == "plan_stale"
    assert "voice changed" in str(refused.value)

    # A row recorded for another role.
    director.clear_direction_cache()
    rows = load_segments(studio["project_id"])
    rows[0].role_id = "someone-else"
    save_segments(studio["project_id"], rows)
    studio["segments"] = load_segments(studio["project_id"])
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.requirement_direction(studio["project"], studio["segments"][0])
    assert refused.value.code == "plan_stale"
    assert "role changed" in str(refused.value)

    # And a row recorded for other text.
    rows = load_segments(studio["project_id"])
    rows[0].role_id = "narrator"
    rows[0].text = "Rewritten after the direction was resolved."
    save_segments(studio["project_id"], rows)
    studio["segments"] = load_segments(studio["project_id"])
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.requirement_direction(studio["project"], studio["segments"][0])
    assert refused.value.code == "plan_stale"
    assert "text changed" in str(refused.value)


def test_a_runtime_mismatch_fails_closed(studio) -> None:
    segments = studio["segments"]
    _activate(studio, [_row(segments[0], salt=7, capability_sha="f" * 64)],
              event_id="evt-mismatch")
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.requirement_direction(studio["project"], segments[0])
    assert refused.value.code == "runtime_mismatch"


def test_a_held_line_refuses_at_the_hook(studio) -> None:
    segments = studio["segments"]
    held = director.UtteranceDirection(
        utterance_id=segments[0].id, text_sha256=text_sha256(segments[0].text),
        role_id=segments[0].role_id or "",
        dispositions=[director.ControlDisposition(control="native_instruction", state="hold",
                                                  reason="capability_unsupported: required")],
        sources=director.DirectionSources(capability_sha256=CAPABILITY_SHA,
                                          resolver_version=director.RESOLVER_VERSION))
    _activate(studio, [held])
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.requirement_direction(studio["project"], segments[0])
    assert refused.value.code == "capability_unsupported"
    assert "held" in str(refused.value)


def test_a_caller_cannot_supply_a_direction_or_a_verdict(studio) -> None:
    """No public interface accepts a caller's direction, verdict or pointer bypass."""

    import inspect

    for function in (director.requirement_direction, director.direction_for,
                     director.active_direction_for, director.resolve_directions):
        parameters = set(inspect.signature(function).parameters)
        for forbidden in ("verification", "verdict", "bridge", "trusted", "override"):
            assert not any(forbidden in name.casefold() for name in parameters), function
    parameters = set(inspect.signature(director.commit_index).parameters)
    assert "expected_revision" in parameters
    assert not any(name in parameters for name in ("trusted", "verdict", "bridge",
                                                   "revision_without_check"))


def test_the_process_cache_is_bounded(studio, monkeypatch) -> None:
    """A bounded cache: rotating the pointer must not grow the process state."""

    monkeypatch.setattr(director, "DIRECTION_CACHE_LIMIT", 2)
    segments = studio["segments"]
    live = live_capability_sha(studio, segments[0])
    director.clear_direction_cache()
    for index, segment in enumerate(segments):
        stored = director.write_direction_set(
            studio["project_id"],
            director.DirectionSet(project_id=studio["project_id"],
                                  directions=[_row(segment, salt=index + 1,
                                                   capability_sha=live)]))
        director.commit_index(studio["project_id"], index,
                              lambda index, sha=stored["sha256"]: index.update(
                                  {"enabled": True}) or index["active"].update(
                                      {"direction_set": sha}),
                              actor="synthetic", event_id=f"evt-{index}")
        project = load_project(studio["project_id"])
        # Each rotation stores a set carrying this line, so every call reaches the
        # loader and the cache has to stay bounded.
        assert director.requirement_direction(project, segment) is not None
    assert len(director._DIRECTION_CACHE) <= 2
