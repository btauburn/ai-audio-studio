"""Real owner persistence with metadata-only fixtures; no GPU or real books."""
from types import SimpleNamespace

import pytest

from tests.audiobook_helpers import build_project, use_projects_root
from dubbing_service.app.audiobook import voice_lab_chapter as chapter
from dubbing_service.app.core.tts_operations import try_register_running_batch, unregister_running_batch
from dubbing_service.app.models.audiobook import Attribution, SourceAnchor, SourceRevision, SpokenTextRevision
from dubbing_service.app.models.character import CharacterVoiceIdentity
from dubbing_service.app.models.project import Segment, SegmentTake
from dubbing_service.app.models.jobs import SynthesisJob, SynthesisJobStatus
from dubbing_service.app.storage.projects import load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from dubbing_service.app.storage.jobs import get_job, load_jobs, upsert_job


@pytest.fixture
def source(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    project = build_project(chapter_titles=["Chapter 257", "Chapter 258"], roles=[("narrator", "Narrator"), ("alice", "Alice")])
    snapshot = b"PRIVATE SYNTHETIC METADATA FIXTURE, NEVER AUDIO"
    (root / project.id / "source.snapshot").write_bytes(snapshot)
    project.source_revisions = [SourceRevision(id="source1", uri="fixture:never-render", snapshot_file="source.snapshot",
        sha256=chapter.hashlib.sha256(snapshot).hexdigest(), byte_length=len(snapshot))]
    project.voice_lab.preferences = {"old": {"favorite": True}}
    rows = []
    for index, text in enumerate(["Fixture narration.", "***", "Fixture dialogue."]):
        role = "alice" if index == 2 else "narrator"
        spoken = SpokenTextRevision(id=f"spoken{index}", source_revision_id="source1", source_text_hash=chapter._text_hash(text),
            spoken_text=text, spoken_text_hash=chapter._text_hash(text))
        project.spoken_text_revisions.append(spoken)
        rows.append(Segment(id=f"seg{index}", chapter_id="chap1", index=index, text=text, role_id=role,
            source_anchor=SourceAnchor(source_revision_id="source1", chapter_id="chap1", occurrence_index=index),
            spoken_text_revision_id=spoken.id, attribution=Attribution(speaker_id=role, source="tag"),
            takes=[SegmentTake(id=f"old{index}", audio_file=f"audio/old{index}.wav")], selected_take_id=f"old{index}",
            audio_file=f"audio/old{index}.wav", review_status="approved"))
    rows.append(Segment(id="other_chapter", chapter_id="chap0", text="Other chapter retained.", role_id="narrator"))
    save_project(project)
    save_segments(project.id, rows)
    return project, root


def fork(source, **kwargs):
    return chapter.fork_chapter(source[0].id, chapter_id="chap1", name="Separate chapter 258", **kwargs)


def ready(source):
    result = fork(source)
    project = load_project(result["project_id"])
    root = source[1] / project.id
    for role in project.roles:
        ref, prompt = role.id + ".wav", role.id + ".npz"
        for relative in [ref, prompt]:
            (root / relative).write_bytes(("NON-AUDIO-TEST-" + relative).encode())
        role.voice_identity = CharacterVoiceIdentity(id="fresh_" + role.id, role_id=role.id, mechanism="qwen_reference",
            model="fixture-never-generate", dossier_revision="fixture", casting_profile_revision="fixture",
            design_instruction="metadata test only", reference_audio=ref, prompt_artifact=prompt,
            reference_audio_sha256=chapter.hashlib.sha256((root / ref).read_bytes()).hexdigest(),
            prompt_sha256=chapter.hashlib.sha256((root / prompt).read_bytes()).hexdigest())
    save_project(project)
    return project


def mock_pipeline(monkeypatch):
    calls, generated = [], []
    def render(project_id, ids, **kwargs):
        assert kwargs["resume"] and kwargs["run_qa"] and kwargs["promote"]
        assert len(ids) == 1
        calls.extend(ids)
        rows = load_segments(project_id)
        row = next(s for s in rows if s.id == ids[0])
        reused = bool(row.selected_take_id)
        if not reused:
            row.selected_take_id = "new_" + row.id
            row.takes.append(SegmentTake(id=row.selected_take_id, audio_file=f"fixture/{row.id}.wav"))
            generated.append(row.id)
            save_segments(project_id, rows)
        return [chapter.pipeline.RenderRecord(segment_id=row.id, take_id=row.selected_take_id,
            success=True, selected=True, qa_passed=True, reused=reused, dispatched=not reused)]
    monkeypatch.setattr(chapter.pipeline, "render_utterances", render)
    return calls, generated


def run(project, **kwargs):
    return chapter.render_chapter(project.id, chapter_id="chap1", request_id="request-one",
                                  renderer=SimpleNamespace(backend="qwen3tts"), **kwargs)


def test_fork_preserves_source_ids_attribution_revisions_and_original_bytes(source):
    original = {p.name: p.read_bytes() for p in (source[1] / source[0].id).iterdir() if p.is_file()}
    result = fork(source)
    project = load_project(result["project_id"])
    rows = load_segments(project.id)
    assert result["speech_utterance_count"] == 2 and result["scene_break_count"] == 1
    assert project.voice_lab.scope == "production" and not project.voice_lab.preferences
    assert not project.character_cast.policy.enabled and project.character_cast.policy.audition_mode == "off"
    assert [s.id for s in rows] == ["seg0", "seg1", "seg2"]
    assert all(not s.takes and not any(slot.take_id for slot in s.take_slots.values()) and s.selected_take_id is None and s.review_status == "draft" for s in rows)
    assert [s.attribution for s in rows] == [s.attribution for s in load_segments(source[0].id) if s.chapter_id == "chap1"]
    assert project.source_revisions == source[0].source_revisions
    assert project.spoken_text_revisions == source[0].spoken_text_revisions
    assert all(r.voice_identity is None and r.speaker_audio is None and r.qwen3_speaker is None for r in project.roles)
    assert original == {p.name: p.read_bytes() for p in (source[1] / source[0].id).iterdir() if p.is_file()}


def test_fork_retry_returns_same_project_and_wrong_index_rejected(source):
    first = fork(source)
    second = fork(source)
    assert first["project_id"] == second["project_id"] and second["reused"]
    with pytest.raises(chapter.ChapterRenditionError, match="Exact canonical chapter"):
        chapter.fork_chapter(source[0].id, chapter_id="258", name="Wrong indexing")


def test_snapshot_tampering_blocks_before_new_project(source):
    (source[1] / source[0].id / "source.snapshot").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="no longer match"):
        fork(source)
    assert len(list(source[1].glob("*/project.json"))) == 1


def test_fork_does_not_touch_active_production(source):
    lease = try_register_running_batch(source[0].id)
    try:
        with pytest.raises(chapter.ChapterRenditionError, match="busy"):
            fork(source)
    finally:
        unregister_running_batch(lease)


def test_render_requires_derived_scope_complete_cast_and_known_attribution(source):
    with pytest.raises(chapter.ChapterRenditionError, match="alternate chapter"):
        run(source[0])
    result = fork(source)
    with pytest.raises(chapter.ChapterRenditionError, match="locked, ready"):
        run(load_project(result["project_id"]))
    project = ready(source)
    rows = load_segments(project.id)
    rows[2].attribution = Attribution(source="unknown")
    save_segments(project.id, rows)
    with pytest.raises(chapter.ChapterRenditionError, match="resolved source-backed"):
        run(project)


def test_render_is_bounded_source_bound_and_scene_breaks_are_not_spoken(source, monkeypatch):
    project = ready(source)
    calls, generated = mock_pipeline(monkeypatch)
    result = run(project)
    assert result["job"]["status"] == "succeeded"
    assert calls == generated == ["seg0", "seg2"]
    assert not load_segments(project.id)[1].takes
    assert result["job"]["job_type"] == "batch_synthesize"
    assert all(s.selected_take_id.startswith("old") for s in load_segments(source[0].id) if s.chapter_id == "chap1")
    with pytest.raises(chapter.ChapterRenditionError, match="distinct speech"):
        run(project, segment_ids=["seg1"])


def test_resume_reuses_takes_without_duplicate_generation(source, monkeypatch):
    project = ready(source)
    calls, generated = mock_pipeline(monkeypatch)
    first = run(project)
    second = run(project)
    assert first["job"]["id"] == second["job"]["id"]
    assert generated == ["seg0", "seg2"] and calls == ["seg0", "seg2", "seg0", "seg2"]
    assert all(row["reused"] for row in second["records"])
    assert len(load_jobs(project.id)) == 1


def test_pending_and_active_production_keep_gpu_priority(source, monkeypatch):
    project = ready(source)
    calls, _ = mock_pipeline(monkeypatch)
    upstream = SynthesisJob(id="oldjob", project_id=source[0].id, job_type="batch_synthesize", segment_ids=["other_chapter"])
    upsert_job(source[0].id, upstream)
    with pytest.raises(chapter.ChapterRenditionError, match="retain GPU priority"):
        run(project)
    assert not calls and not load_jobs(project.id)


def test_intent_refuses_reused_request_for_recast(source, monkeypatch):
    project = ready(source)
    mock_pipeline(monkeypatch)
    run(project)
    changed = load_project(project.id)
    changed.roles[0].voice_identity.id = "newer_voice_version"
    save_project(changed)
    with pytest.raises(chapter.ChapterRenditionError, match="already bound"):
        run(project)


def test_source_edit_before_dispatch_refused(source):
    project = ready(source)
    rows = load_segments(project.id)
    rows[0].text = "Modified source."
    save_segments(project.id, rows)
    with pytest.raises(chapter.ChapterRenditionError, match="Canonical source"):
        run(project)
    assert not load_jobs(project.id)


def test_changed_cast_mid_job_stops_following_utterance(source, monkeypatch):
    project = ready(source)
    calls = []
    def render(project_id, ids, **kwargs):
        calls.extend(ids)
        changed = load_project(project_id)
        changed.roles[0].voice_identity.id = "changed_during_render"
        save_project(changed)
        return [chapter.pipeline.RenderRecord(segment_id=ids[0], take_id="historical-only", success=True,
                    selected=False, qa_passed=True, stale=True, error="stale voice")]
    monkeypatch.setattr(chapter.pipeline, "render_utterances", render)
    result = run(project)
    assert result["job"]["status"] == "interrupted" and calls == ["seg0"]
    assert load_segments(project.id)[0].selected_take_id is None


def test_background_claim_coalesces_concurrent_retry_and_releases_lane(source, monkeypatch):
    project = ready(source)
    calls, _ = mock_pipeline(monkeypatch)
    scheduled = []
    first = run(project, schedule=lambda fn, *args: scheduled.append((fn, args)))
    second = run(project)
    assert second["already_running"] and second["job_id"] == first["job_id"] and not calls
    assert len(scheduled) == 1
    fn, args = scheduled[0]
    assert fn(*args)["job"]["status"] == "succeeded"
    lease = try_register_running_batch(source[0].id)
    assert lease is not None
    unregister_running_batch(lease)


def test_attribution_can_be_corrected_before_first_intent(source, monkeypatch):
    project = ready(source)
    rows = load_segments(project.id)
    rows[2].role_id = "narrator"
    rows[2].attribution = Attribution(speaker_id="narrator", source="user", reviewed=True)
    save_segments(project.id, rows)
    mock_pipeline(monkeypatch)
    assert run(project)["job"]["status"] == "succeeded"


def test_conditioning_tamper_and_legacy_auditions_block_dispatch(source):
    project = ready(source)
    project.character_cast.policy.enabled = True
    save_project(project)
    with pytest.raises(chapter.ChapterRenditionError, match="legacy auditions"):
        run(project)
    project.character_cast.policy.enabled = False
    save_project(project)
    (source[1] / project.id / "narrator.npz").write_bytes(b"tamper")
    with pytest.raises(ValueError, match="no longer match"):
        run(project)


def audit(source):
    project, root = source
    return {"source_project_id": project.id, "chapter_id": "chap1", "source_revision_id": "source1",
        "source_sha256": project.source_revisions[0].sha256,
        "source_project_file_sha256": {"segments.json": chapter.hashlib.sha256((root / project.id / "segments.json").read_bytes()).hexdigest()},
        "corrections": [{"utterance_id": "seg2", "from_role_id": "alice", "role_id": "narrator",
            "source_text_sha256": chapter._text_hash("Fixture dialogue."), "evidence_utterance_ids": ["seg0"],
            "reason": "Synthetic local source audit fixture: quoted narration, no actual person."}],
        "alias_corrections": {}, "scope": "Synthetic metadata fixture, never human preference"}


def test_generic_typed_audit_preserves_original_attribution_and_all_evidence(source):
    from dubbing_service.app.routers.voice_lab import ChapterFork
    request = ChapterFork(chapter_id="chap1", name="Reviewed fixture", attribution_review=audit(source))
    result = chapter.fork_chapter(source[0].id, **request.model_dump())
    project = load_project(result["project_id"])
    reviewed = next(s for s in load_segments(project.id) if s.id == "seg2")
    assert reviewed.role_id == reviewed.attribution.speaker_id == "narrator"
    assert reviewed.attribution.source == "inline" and not reviewed.attribution.reviewed
    meta = project.metadata["chapter_rerun"]
    assert meta["attribution_review"]["scope"] == audit(source)["scope"]
    change = next(row for row in meta["attribution_derivation"] if row["utterance_id"] == "seg2")
    assert change["original_role_id"] == "alice" and change["original_attribution"]["speaker_id"] == "alice"
    assert next(s for s in load_segments(source[0].id) if s.id == "seg2").role_id == "alice"


@pytest.mark.parametrize("change,match", [
    (lambda a: a.update(source_project_id="another_project"), "exact source project"),
    (lambda a: a.update(source_sha256="0" * 64), "another source revision"),
    (lambda a: a["corrections"][0].update(source_text_sha256="0" * 64), "exact source text"),
    (lambda a: a["corrections"][0].update(role_id="invented_person"), "unsupported"),
    (lambda a: a["corrections"][0].update(evidence_utterance_ids=["missing_context"]), "surrounding source evidence"),
    (lambda a: a.update(alias_corrections={"alice": "narrator", "narrator": "alice"}), "cycle"),
])
def test_unbound_audits_rejected_before_project_creation(source, change, match):
    review = audit(source)
    change(review)
    with pytest.raises(chapter.ChapterRenditionError, match=match):
        fork(source, attribution_review=review)
    assert len(list(source[1].glob("*/project.json"))) == 1


def test_audit_context_edit_and_reusing_request_for_changed_review_rejected(source):
    review = audit(source)
    fork(source, attribution_review=review)
    altered = chapter.copy.deepcopy(review)
    altered["corrections"][0]["reason"] = "Changed review meaning"
    with pytest.raises(chapter.ChapterRenditionError, match="changed source"):
        fork(source, attribution_review=altered)
    rows = load_segments(source[0].id)
    next(s for s in rows if s.id == "seg0").text = "Changed surrounding evidence."
    save_segments(source[0].id, rows)
    with pytest.raises(ValueError, match="no longer match"):
        chapter.fork_chapter(source[0].id, chapter_id="chap1", name="Different name", attribution_review=review)
