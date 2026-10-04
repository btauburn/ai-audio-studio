"""Derived chapter composition regression with isolated short signal files.

No speech, model, GPU or user project is read/generated. MP3 is mocked here;
the independent real-codec suite covers the shared encoder/decoder helpers.
"""

from types import SimpleNamespace
import json
import wave

import pytest

from tests.audiobook_helpers import build_project, make_tone_wav, use_projects_root
from dubbing_service.app.audiobook import voice_lab_chapter_export as export, qa
from dubbing_service.app.audiobook.fingerprint import text_sha256, voice_revision_for_role
from dubbing_service.app.models.project import Role, Segment, SegmentTake
from dubbing_service.app.models.audiobook import Attribution, SourceAnchor, SourceRevision, QaCheckResult
from dubbing_service.app.models.character import CharacterVoiceIdentity
from dubbing_service.app.storage.projects import load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from dubbing_service.app.core.tts_operations import try_register_running_batch, unregister_running_batch


@pytest.fixture
def chapter(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    project = build_project(name="isolated signal-only chapter fixture", chapter_titles=["Chapter 258: Fixture"])
    project.voice_lab.scope = "isolated_listening_test"
    base = root / project.id
    snapshot = base / "source.snapshot"
    snapshot.write_bytes(b"ISOLATED FIXTURE ONLY")
    source = SourceRevision(id="source_fixture", uri="fixture:source", sha256=export.file_sha256(snapshot), snapshot_file="source.snapshot")
    project.source_revisions = [source]
    project.roles = []
    for name in ["narrator", "alden", "stuart", "library"]:
        identity = CharacterVoiceIdentity(id="voice_" + name, role_id=name, status="LOCKED", mechanism="qwen_reference",
                 dossier_revision="fixture", casting_profile_revision="fixture", design_instruction="fixture only",
                 model="mock-no-model", reference_audio_sha256=text_sha256(name), prompt_sha256=text_sha256("prompt_" + name))
        project.roles.append(Role(id=name, name=name, voice_identity=identity))
    segments = []
    for i, name in enumerate(["narrator", None, "alden", "stuart", "library"]):
        segment = Segment(id=f"seg{i}", chapter_id="chap0", index=i, role_id=name or "narrator",
                          text="***" if name is None else name + " fixture.",
                          source_anchor=SourceAnchor(source_revision_id=source.id, chapter_id="chap0", occurrence_index=i),
                          attribution=Attribution(speaker_id=name or "narrator", source="user"))
        if name:
            path = make_tone_wav(base / f"audio/{name}.wav", seconds=0.5, frequency=220 + i * 80)
            role = next(r for r in project.roles if r.id == name)
            identity = role.voice_identity
            checksum = export.file_sha256(path)
            take = SegmentTake(id="take_" + name, role_id=name, audio_file=path.relative_to(base).as_posix(),
                     audio_sha256=checksum, source_text_hash=text_sha256(segment.text), input_fingerprint="fixture-current",
                     fingerprint_components={"voice_revision": voice_revision_for_role(role)},
                     generation_metadata={"character_voice_identity": identity.model_dump(mode="json"),
                                          "provider_calls": [{"identity_id": identity.id, "prompt_sha256": identity.prompt_sha256}]})
            take.qa_report = qa.build_report(take.id, checksum, [QaCheckResult(check_id="decodable", check_version="fixture", status="passed")])
            segment.takes = [take]
            segment.selected_take_id = take.id
        segments.append(segment)
    project.metadata["chapter_rerun"] = {
        "authorized": True, "source_project_id": "original-untouched", "chapter_id": "chap0",
        "source_utterance_ids": [s.id for s in segments], "speech_utterance_ids": [s.id for s in segments if s.id != "seg1"],
        "source_revisions": [{"id": source.id, "sha256": source.sha256, "snapshot_file": source.snapshot_file}],
        "source_utterances": [{"utterance_id": s.id, "index": s.index, "chapter_id": s.chapter_id,
                 "source_text_sha256": text_sha256(s.text), "source_anchor": s.source_anchor.model_dump(mode="json"),
                 "spoken_text_revision_id": s.spoken_text_revision_id} for s in segments],
        "scene_breaks": [{"utterance_id": "seg1", "index": 1, "source_text_sha256": text_sha256("***"), "duration_seconds": 1.25}],
    }
    from dubbing_service.app.audiobook.voice_lab_chapter import _hash, _source_proof
    project.metadata["chapter_rerun"]["expected_source_hash"] = _hash(_source_proof(project, project.metadata["chapter_rerun"]["source_utterances"]))
    save_project(project)
    save_segments(project.id, segments)
    monkeypatch.setattr(export.pipeline, "_requirement_for_take", lambda *_: SimpleNamespace(fingerprint=lambda: "fixture-current"))
    calls = {"mp3": 0}
    def encode(master, target):
        calls["mp3"] += 1
        with target.open("xb") as stream:
            stream.write(master.read_bytes())  # signal fixture, never a real MP3
    monkeypatch.setattr(export, "_mp3", encode)
    monkeypatch.setattr(export, "_decoded_duration", lambda path: _duration(path))
    monkeypatch.setattr(export, "_loudness", lambda _: -20.0)
    return project.id, base, calls


def _duration(path):
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def run(chapter):
    return export.export_chapter(chapter[0], "chap0")


def test_complete_chapter_uses_all_actual_roles_no_five_minute_or_four_character_gate(chapter):
    result = run(chapter)
    assert result["duration_seconds"] == 3.25
    assert result["speech_utterance_count"] == 4 and result["scene_break_count"] == 1
    assert result["rendered_voice_role_count"] == 4
    assert result["technical_qa"] == "passed" and result["human_review"] == "pending"
    assert result["status"] == "ready_for_listening"
    assert [r["utterance_id"] for r in result["map"]] == [f"seg{i}" for i in range(5)]
    assert result["map"][1]["kind"] == "scene_break"
    assert result["map"][1]["end_seconds"] - result["map"][1]["start_seconds"] == 1.25
    assert "fixture." not in json.dumps(result)
    assert all("text" not in row for row in result["map"])
    assert result["loudness_measurement"]["audio_sha256"] == result["audio_sha256"]
    assert result["comparisons"][0]["passage_id"] == result["passage_id"]
    assert result["generation_totals"]["tts_generated_by_export"] == 0


def test_resuming_export_preserves_selected_take_files_and_reuses_master(chapter):
    before = {p.name: (export.file_sha256(p), p.stat().st_mtime_ns) for p in (chapter[1] / "audio").glob("*.wav")}
    a = run(chapter)
    b = run(chapter)
    assert a["audio_sha256"] == b["audio_sha256"]
    assert chapter[2]["mp3"] == 1
    assert before == {p.name: (export.file_sha256(p), p.stat().st_mtime_ns) for p in (chapter[1] / "audio").glob("*.wav")}
    with wave.open(str(chapter[1] / a["master_file"]), "rb") as master:
        actual = master.readframes(master.getnframes())
    with wave.open(str(chapter[1] / "audio/narrator.wav"), "rb") as original:
        intro = original.readframes(original.getnframes())
    assert actual.startswith(intro + b"\x00" * 60000)


@pytest.mark.parametrize("change", ["not_authorized", "same_project", "chapter", "missing_source_rows", "source_bytes", "missing_segment", "text", "anchor", "revision", "break_length", "break_hash", "speech_ids"])
def test_source_or_scope_mutation_blocks_export(chapter, change):
    project = load_project(chapter[0]); segments = load_segments(chapter[0]); meta = project.metadata["chapter_rerun"]
    if change == "not_authorized": meta["authorized"] = False
    elif change == "same_project": meta["source_project_id"] = project.id
    elif change == "chapter": meta["chapter_id"] = "other"
    elif change == "missing_source_rows": meta.pop("source_utterances")
    elif change == "source_bytes": (chapter[1] / "source.snapshot").write_bytes(b"changed isolated fixture")
    elif change == "missing_segment": segments.pop()
    elif change == "text": segments[0].text += " altered"
    elif change == "anchor": segments[0].source_anchor.occurrence_index = 10
    elif change == "revision": segments[0].spoken_text_revision_id = "new"
    elif change == "break_length": meta["scene_breaks"][0]["duration_seconds"] = 5
    elif change == "break_hash": meta["scene_breaks"][0]["source_text_sha256"] = "bad"
    elif change == "speech_ids": meta["speech_utterance_ids"].pop()
    save_project(project); save_segments(project.id, segments)
    with pytest.raises(export.ChapterExportError): run(chapter)
    assert chapter[2]["mp3"] == 0


@pytest.mark.parametrize("change", ["no_selected", "missing_hash", "hash", "no_qa", "qa_hash", "qa_failed", "stale", "wrong_role", "wrong_source", "wrong_revision", "speed", "identity", "provider", "voice_version", "unknown_attribution"])
def test_selected_take_proof_is_required(chapter, change):
    segments = load_segments(chapter[0]); segment = segments[0]; take = segment.takes[0]
    if change == "no_selected": segment.selected_take_id = None
    elif change == "missing_hash": take.audio_sha256 = None
    elif change == "hash": take.audio_sha256 = "0" * 64
    elif change == "no_qa": take.qa_report = None
    elif change == "qa_hash": take.qa_report.output_audio_hash = "0" * 64
    elif change == "qa_failed": take.qa_report.results[0].status = "failed"
    elif change == "stale": take.stale = True
    elif change == "wrong_role": take.role_id = "alden"
    elif change == "wrong_source": take.source_text_hash = "bad"
    elif change == "wrong_revision": take.spoken_text_revision_id = "bad"
    elif change == "speed": take.playback_speed = 1.1
    elif change == "identity": take.generation_metadata["character_voice_identity"]["id"] = "bad"
    elif change == "provider": take.generation_metadata["provider_calls"][0]["prompt_sha256"] = "bad"
    elif change == "voice_version": take.fingerprint_components["voice_revision"] = "bad"
    elif change == "unknown_attribution": segment.attribution.source = "unknown"
    save_segments(chapter[0], segments)
    with pytest.raises(export.ListeningRenderError): run(chapter)
    assert chapter[2]["mp3"] == 0


def test_existing_render_lane_prevents_export(chapter):
    lease = try_register_running_batch(chapter[0])
    assert lease is not None
    try:
        with pytest.raises(export.ChapterExportError, match="busy"): run(chapter)
    finally:
        unregister_running_batch(lease)
    assert chapter[2]["mp3"] == 0


@pytest.mark.parametrize("change", ["cast", "source", "selection"])
def test_stale_export_never_replaces_newer_state(chapter, monkeypatch, change):
    encode = export._mp3
    def mutate(master, target):
        encode(master, target)
        project = load_project(chapter[0]); segments = load_segments(chapter[0])
        if change == "cast": project.roles[0].style_instruction = "new casting direction"
        elif change == "source": segments[0].text += " newer revision"
        else:
            take = segments[0].takes[0].model_copy(deep=True)
            take.id = "newer_take"
            take.qa_report.take_id = take.id
            segments[0].takes.append(take)
            segments[0].selected_take_id = take.id
        project.voice_lab.excerpt = {"audio_sha256": "newer-current-selection"}
        save_project(project); save_segments(project.id, segments)
    monkeypatch.setattr(export, "_mp3", mutate)
    with pytest.raises(export.ChapterExportError): run(chapter)
    assert load_project(chapter[0]).voice_lab.excerpt["audio_sha256"] == "newer-current-selection"


def test_interrupted_partial_export_kept_and_resume_is_new_directory(chapter):
    project, segments, source = export._source_snapshot(chapter[0], "chap0")
    _, _, snapshot = export._take_snapshot(project, segments, source)
    partial = chapter[1] / f"voice_lab/chapters/chap0/{snapshot['state_sha256'][:24]}/clean-master.wav"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial isolated fixture")
    result = run(chapter)
    assert "/resume_" in result["audio_file"]
    assert partial.read_bytes() == b"partial isolated fixture"


def test_resume_completes_map_receipt_after_interruption_without_reassembly(chapter, monkeypatch):
    immutable = export._immutable
    def interrupted(path, content):
        if path.name == "source-to-audio-map.json":
            raise OSError("simulated interruption before map publication")
        return immutable(path, content)
    monkeypatch.setattr(export, "_immutable", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        run(chapter)
    assert not load_project(chapter[0]).voice_lab.excerpt
    monkeypatch.setattr(export, "_immutable", immutable)
    result = run(chapter)
    assert chapter[2]["mp3"] == 1
    assert (chapter[1] / result["audio_file"]).with_name("source-to-audio-map.json").is_file()
