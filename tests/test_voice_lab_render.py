"""Composition tests with non-audio byte fixtures and mocked generation/decoding.

These exercise real Studio metadata, Takes and job persistence. No model, GPU,
playable fixture, chapter excerpt or synthesized listening audio is produced.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.test_listening_excerpt import source  # noqa: F401 - shared pytest fixture
from tests.audiobook_helpers import build_project, use_projects_root
from dubbing_service.app.audiobook import voice_lab_render as render, voice_lab, qa
from dubbing_service.app.audiobook.listening_excerpt import build_excerpt_manifest, text_hash
from dubbing_service.app.models.project import Chapter, Role, Segment, SegmentTake
from dubbing_service.app.models.audiobook import Attribution, SourceAnchor, SourceRevision, QaCheckResult
from dubbing_service.app.models.character import CharacterVoiceIdentity
from dubbing_service.app.storage.projects import load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from dubbing_service.app.storage.jobs import load_jobs
from dubbing_service.app.core.tts_operations import try_register_running_batch, unregister_running_batch


@pytest.fixture
def lab(monkeypatch, tmp_path, source):
    root = use_projects_root(monkeypatch, tmp_path)
    project = build_project(name="isolated metadata fixture, never playable")
    project.voice_lab.scope = "isolated_listening_test"
    chapter = replace(source["chapter"], source_sha256=text_hash(source["source_text"]))
    source["chapter"] = chapter
    project.chapters = [Chapter(id=chapter.chapter_id, title=chapter.chapter_title)]
    project.roles = []
    for role in ["narrator", "alice", "bea", "charlie", "dani"]:
        identity = CharacterVoiceIdentity(id="voice_" + role, role_id=role, mechanism="qwen_reference",
                    dossier_revision="fixture", casting_profile_revision="fixture", design_instruction="fixture only",
                    model="mock-no-model", reference_audio_sha256=text_hash(role), prompt_sha256=text_hash("prompt_" + role))
        project.roles.append(Role(id=role, name=role, voice_identity=identity))
    base = root / project.id
    snapshot = base / "fixture.snapshot"
    snapshot.write_text(source["source_text"], encoding="utf-8", newline="")
    project.source_revisions = [SourceRevision(id=chapter.source_revision_id, uri="fixture:never-render",
                                sha256=chapter.source_sha256, snapshot_file="fixture.snapshot")]
    save_project(project)
    segments = []
    for row in source["utterances"]:
        path = base / f"audio/{row.utterance_id}.wav"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"NON-AUDIO-METADATA-TEST-ONLY-" + row.utterance_id.encode())
        checksum = render.file_sha256(path)
        role = next(r for r in project.roles if r.id == row.role_id)
        identity = role.voice_identity
        take = SegmentTake(id="take_" + row.utterance_id, role_id=role.id,
                           audio_file=path.relative_to(base).as_posix(), audio_sha256=checksum,
                           source_text_hash=row.source_text_sha256, input_fingerprint="fixture-current",
                           fingerprint_components={"voice_revision": "fixture-v1"},
                           generation_metadata={"character_voice_identity": identity.model_dump(mode="json"),
                               "provider_calls": [{"identity_id": identity.id, "prompt_sha256": identity.prompt_sha256,
                                                    "fixture_only": True}]})
        take.qa_report = qa.build_report(take.id, checksum, [QaCheckResult(check_id="decodable", check_version="fixture",
                                                                         status="passed", details={"mocked": True})])
        segment = Segment(id=row.utterance_id, chapter_id=chapter.chapter_id, role_id=row.role_id,
                          text=source["source_text"][row.start_offset:row.end_offset], index=len(segments),
                          source_anchor=SourceAnchor(source_revision_id=chapter.source_revision_id,
                                                   chapter_id=chapter.chapter_id, start_offset=row.start_offset, end_offset=row.end_offset),
                          attribution=Attribution(speaker_id=row.role_id, source="tag"), takes=[take], selected_take_id=take.id)
        segments.append(segment)
    save_segments(project.id, segments)
    manifest = build_excerpt_manifest(**source)
    calls = {"render": 0, "assemble": 0, "mp3": 0}
    monkeypatch.setattr(qa, "read_audio_facts", lambda _: SimpleNamespace(decodable=True, duration_ms=50_000))
    monkeypatch.setattr(render, "_decoded_duration", lambda p: 300 if p.name in {"clean-master.wav", "listening.mp3"} else 50)
    monkeypatch.setattr(render, "_loudness", lambda _: -20.5)
    monkeypatch.setattr(render.pipeline, "_requirement_for_take", lambda *args: SimpleNamespace(fingerprint=lambda: "fixture-current"))
    def render_pipeline(project_id, ids, **kwargs):
        assert ids == manifest["utterance_ids"]
        assert kwargs["resume"] and kwargs["run_qa"] and kwargs["promote"]
        calls["render"] += 1
        return [render.pipeline.RenderRecord(segment_id=s.id, take_id=s.selected_take_id,
                    success=True, selected=True, qa_passed=True, duration_ms=50_000, reused=True) for s in load_segments(project_id)]
    monkeypatch.setattr(render.pipeline, "render_utterances", render_pipeline)
    def assemble(paths, target):
        calls["assemble"] += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"NON-AUDIO-ASSEMBLY-FIXTURE")
    monkeypatch.setattr(render, "_assemble_pcm", assemble)
    def encode(master, target):
        calls["mp3"] += 1
        target.write_bytes(b"NON-AUDIO-MP3-FIXTURE")
    monkeypatch.setattr(render, "_mp3", encode)
    monkeypatch.setattr(qa, "qa_take_audio", lambda **kw: qa.build_report(kw["take_id"], kw["expected_sha256"],
        [QaCheckResult(check_id="decodable", check_version="fixture", status="passed", details={"mocked": True})]))
    return project, manifest, source["source_text"], root, calls


def persist(lab):
    project, manifest, text, _root, _calls = lab
    return render.persist_excerpt_manifest(project.id, manifest, source_text=text)


def run(lab):
    project, manifest, *_ = lab
    return render.render_listening_excerpt(project.id, expected_manifest_id=manifest["manifest_id"], renderer=SimpleNamespace(backend="mock-no-model"))


def test_manifest_must_be_persisted_before_any_render(lab):
    with pytest.raises(render.ListeningRenderError, match="persisted excerpt"):
        run(lab)
    assert lab[-1]["render"] == 0
    state = persist(lab)
    assert state["status"] == "selected_pending_synthesis" and state["audio_sha256"] is None
    assert (lab[3] / lab[0].id / state["manifest_file"]).is_file()


def test_production_scope_cannot_be_implicitly_recast(lab):
    project = load_project(lab[0].id)
    project.voice_lab.scope = "production"
    save_project(project)
    with pytest.raises(render.ListeningRenderError, match="isolated"):
        persist(lab)
    assert lab[-1]["render"] == 0


def test_existing_production_lane_keeps_priority(lab):
    persist(lab)
    lease = try_register_running_batch("other-production")
    assert lease is not None
    try:
        with pytest.raises(render.ListeningRenderError, match="priority"):
            run(lab)
        assert lab[-1]["render"] == 0
    finally:
        unregister_running_batch(lease)


def test_missing_final_identity_does_not_start_an_audition_batch(lab):
    persist(lab)
    project = load_project(lab[0].id)
    project.roles[2].voice_identity = None
    save_project(project)
    with pytest.raises(render.ListeningRenderError, match="no audition batches"):
        run(lab)
    assert lab[-1]["render"] == 0


def test_actual_take_metadata_proves_five_voice_roles_and_publishes_pending_human(lab):
    persist(lab)
    result = run(lab)
    assert result["speaking_character_count"] == 4
    assert result["rendered_voice_role_count"] == 5
    assert result["technical_qa"] == "passed" and result["human_review"] == "pending"
    assert result["audio_file"].endswith("listening.mp3")
    assert result["duration_seconds"] == 300
    assert len({row["conditioning_sha256"] for row in result["map"]}) == 5
    assert result["loudness_measurement"]["audio_sha256"] == result["audio_sha256"]
    assert result["comparisons"][0]["passage_id"] == lab[1]["manifest_id"]
    assert result["comparisons"][0]["utterance_ids"] == lab[1]["utterance_ids"]
    jobs = load_jobs(lab[0].id)
    assert len(jobs) == 1 and jobs[0].status.value == "succeeded" and jobs[0].done == 6


def test_completed_resume_does_not_dispatch_or_reassemble(lab):
    persist(lab)
    first = run(lab)
    second = run(lab)
    assert first["audio_sha256"] == second["audio_sha256"]
    assert lab[-1] == {"render": 1, "assemble": 1, "mp3": 1}
    assert len(load_jobs(lab[0].id)) == 1


@pytest.mark.parametrize("change", ["source", "speaker", "speed", "snapshot"])
def test_manifest_revalidated_against_canonical_source_before_render(lab, change):
    persist(lab)
    project, _, _, root, calls = lab
    segments = load_segments(project.id)
    if change == "source": segments[0].text += " changed"
    elif change == "speaker": segments[0].role_id = "alice"
    elif change == "speed":
        segments[0].playback_speed = 1.1
        segments[0].takes[0].playback_speed = 1.1
    else: (root / project.id / "fixture.snapshot").write_bytes(b"changed fixture")
    save_segments(project.id, segments)
    with pytest.raises(render.ListeningRenderError):
        run(lab)
    assert calls["render"] == 0


def test_card_identity_without_matching_provider_evidence_does_not_prove_voice(lab):
    persist(lab)
    segments = load_segments(lab[0].id)
    segments[2].takes[0].generation_metadata["provider_calls"][0]["identity_id"] = "other"
    save_segments(lab[0].id, segments)
    with pytest.raises(render.ListeningRenderError, match="Provider generation metadata"):
        run(lab)
    assert lab[-1]["assemble"] == 0


def test_stale_take_selection_cannot_publish_older_export(lab, monkeypatch):
    persist(lab)
    original = render._mp3
    def competing_selection(master, destination):
        original(master, destination)
        segments = load_segments(lab[0].id)
        segments[0].selected_take_id = None
        segments[0].take_slots = {}
        segments[0].audio_file = None
        save_segments(lab[0].id, segments)
    monkeypatch.setattr(render, "_mp3", competing_selection)
    with pytest.raises(render.ListeningRenderError):
        run(lab)
    assert load_project(lab[0].id).voice_lab.excerpt.get("audio_file") is None
    assert load_jobs(lab[0].id)[0].status.value == "interrupted"


def test_newer_excerpt_selection_cannot_be_replaced_by_completed_audio(lab, monkeypatch):
    persist(lab)
    original = render._mp3
    def competing_selection(master, destination):
        original(master, destination)
        project = load_project(lab[0].id)
        project.voice_lab.excerpt = {"status": "blocked_source", "manifest_id": "newer-choice"}
        save_project(project)
    monkeypatch.setattr(render, "_mp3", competing_selection)
    with pytest.raises(render.ListeningRenderError, match="newer excerpt"):
        run(lab)
    assert load_project(lab[0].id).voice_lab.excerpt["manifest_id"] == "newer-choice"


def test_qa_failure_never_publishes_and_releases_existing_tts_lease(lab, monkeypatch):
    persist(lab)
    monkeypatch.setattr(qa, "qa_take_audio", lambda **kw: qa.build_report(kw["take_id"], kw["expected_sha256"],
        [QaCheckResult(check_id="peak_ok", check_version="fixture", status="failed")]))
    with pytest.raises(render.ListeningRenderError, match="blocking technical QA"):
        run(lab)
    assert load_project(lab[0].id).voice_lab.excerpt.get("audio_file") is None
    assert lab[-1]["mp3"] == 0
    lease = try_register_running_batch("production-after-failure")
    assert lease is not None
    unregister_running_batch(lease)


def test_wrong_actual_duration_does_not_trigger_another_edition(lab, monkeypatch):
    persist(lab)
    monkeypatch.setattr(render, "_decoded_duration", lambda p: 320 if p.name == "clean-master.wav" else 50)
    with pytest.raises(Exception, match="outside 285"):
        run(lab)
    assert lab[-1]["render"] == 1 and lab[-1]["mp3"] == 0


def test_failed_partial_export_resume_reuses_selected_takes(lab, monkeypatch):
    persist(lab)
    encoder = render._mp3
    monkeypatch.setattr(render, "_mp3", lambda *_: (_ for _ in ()).throw(RuntimeError("mock interrupted encode")))
    with pytest.raises(RuntimeError, match="interrupted"):
        run(lab)
    monkeypatch.setattr(render, "_mp3", encoder)
    result = run(lab)
    assert result["status"] == "ready_for_listening"
    assert "/resume_" in result["audio_file"]
    assert result["generation_totals"]["production_seconds_generated_this_run"] == 0
    assert len(load_jobs(lab[0].id)) == 1


def test_crashed_immutable_publish_leaves_no_partial_canonical_file(tmp_path, monkeypatch):
    target = tmp_path / "manifest.json"
    original = render.os.link
    monkeypatch.setattr(render.os, "link", lambda *_: (_ for _ in ()).throw(OSError("interrupted before publish")))
    with pytest.raises(OSError, match="interrupted"):
        render._immutable(target, b"complete fixture")
    assert not target.exists()
    monkeypatch.setattr(render.os, "link", original)
    render._immutable(target, b"complete fixture")
    assert target.read_bytes() == b"complete fixture"
    render._immutable(target, b"complete fixture")
    with pytest.raises(render.ListeningRenderError, match="different bytes"):
        render._immutable(target, b"cannot overwrite completed fixture")
    assert target.read_bytes() == b"complete fixture"


def test_measured_duration_adjustment_persists_new_manifest_and_reuses_takes(lab, monkeypatch):
    persist(lab)
    original_ids = list(lab[1]["utterance_ids"])
    original_selections = [s.selected_take_id for s in load_segments(lab[0].id)]
    scopes = []
    def pipeline(project_id, ids, **kwargs):
        assert kwargs["resume"] and kwargs["run_qa"]
        assert (load_project(project_id).voice_lab.excerpt or {})["manifest_id"]
        scopes.append(list(ids))
        return [render.pipeline.RenderRecord(segment_id=i, take_id="take_" + i, success=True, selected=True,
                reused=True, duration_ms=30_000 if i == original_ids[0] else 60_000) for i in ids]
    monkeypatch.setattr(render.pipeline, "render_utterances", pipeline)
    monkeypatch.setattr(render, "_decoded_duration", lambda path: 300 if path.name in {"clean-master.wav", "listening.mp3"}
                         else 30 if path.stem == original_ids[0] else 60)
    result = run(lab)
    assert scopes == [original_ids, original_ids[1:]]
    assert result["manifest_id"] != lab[1]["manifest_id"]
    assert result["duration_seconds"] == 300
    assert result["utterance_ids"] == original_ids[1:]
    assert len(result["timing_adjustments"]) == 1
    assert result["timing_adjustments"][0]["decoded_duration_seconds"] == 330
    assert result["generation_totals"]["bounded_timing_repair_seconds_this_run"] == 0
    assert lab[-1]["assemble"] == 1  # No complete off-duration comparison was assembled.
    assert [s.selected_take_id for s in load_segments(lab[0].id)] == original_selections


def test_actual_timing_retry_count_is_bounded(lab, monkeypatch):
    persist(lab)
    calls = []
    def off_duration(project_id, *, expected_manifest_id, renderer=None):
        calls.append(expected_manifest_id)
        manifest = {**lab[1], "manifest_id": expected_manifest_id}
        mapping = [{"utterance_id": u, "start_seconds": i * 60, "end_seconds": (i + 1) * 60} for i, u in enumerate(manifest["utterance_ids"])]
        raise render._TimingAdjustmentNeeded(manifest, mapping, [SimpleNamespace(reused=True, duration_ms=60_000)])
    monkeypatch.setattr(render, "_render_listening_excerpt_once", off_duration)
    monkeypatch.setattr(render, "load_excerpt_manifest", lambda *_: (lab[1], lab[2]))
    monkeypatch.setattr(render, "adjust_excerpt_to_measured_duration", lambda manifest, **_: {**manifest, "manifest_id": manifest["manifest_id"] + "x"})
    monkeypatch.setattr(render, "persist_excerpt_manifest", lambda *_, **__: {})
    with pytest.raises(render.ListeningRenderError, match="after two bounded"):
        run(lab)
    assert len(calls) == 3


def test_saved_reference_generation_receipts_are_separate_from_production(lab):
    project = load_project(lab[0].id)
    project.voice_lab.designs["excerpt_fixture"] = {"status": "ready", "role_id": "alice", "identity_id": "voice_alice",
                                                   "reference_generation_count": 1, "key": "voice_key"}
    project.voice_lab.voices["voice_key"] = {"identity": {"reference_audio": "audio/stable_1.wav",
        "reference_audio_sha256": render.file_sha256(lab[3] / project.id / "audio/stable_1.wav")}}
    totals = render._conditioning_totals(project, {"alice"})
    assert totals["conditioning_reference_count"] == 1
    assert totals["conditioning_reference_seconds"] == 50
    assert totals["conditioning_reference_scope"] == "saved_excerpt_design_receipts_in_this_isolated_project"
