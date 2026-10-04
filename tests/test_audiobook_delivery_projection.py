"""Delivery refuses changed selections and unproven chapter completeness."""

from __future__ import annotations

import hashlib
import json
import wave

import pytest

from dubbing_service.app.audiobook.delivery import projection
from dubbing_service.app.models.project import Project, Segment


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def receipt(tmp_path):
    root = tmp_path / "project_example"
    root.mkdir()
    audio = root / "audio"
    audio.mkdir()
    exports = tmp_path / "exports"
    exports.mkdir()
    wav = audio / "take.wav"
    with wave.open(str(wav), "wb") as stream:
        stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x01\x00" * 2400)
    mp3 = exports / "chapter.mp3"
    mp3.write_bytes(b"fixture-receipt-bound-mp3-bytes")
    project = Project(id=root.name, name="Private test book",
                      chapters=[{"id": "chapter_example", "index": 4, "title": "A chapter"}],
                      metadata={"scene_break_manifest": "scene-breaks.json", "source_title": "Fixture book"})
    speech = Segment.model_validate({
        "id": "segment_speech", "chapter_id": "chapter_example", "index": 0,
        "role_id": "narrator", "text": "Private canonical fixture sentence.",
        "selected_take_id": "take_example", "takes": [{
            "id": "take_example", "audio_file": "audio/take.wav", "audio_sha256": _hash(wav),
            "text": "Private canonical fixture sentence.", "role_id": "narrator",
            "qa_report": {"take_id": "take_example", "output_audio_hash": _hash(wav),
                          "qa_check_version": "fixture-v1", "passed": True,
                          "results": [{"check_id": "decodes", "check_version": "v1", "status": "passed"}]},
        }],
    })
    pause = Segment(id="segment_pause", chapter_id="chapter_example", index=1, text="***", role_id="narrator")
    _write(root / "project.json", project.model_dump(mode="json"))
    _write(root / "segments.json", [speech.model_dump(mode="json"), pause.model_dump(mode="json")])
    _write(root / "scene-breaks.json", [{"segment_id": pause.id, "chapter_id": pause.chapter_id,
                                      "index": 1, "source_text": pause.text, "seconds": 1.25}])
    rows = []
    offset = 0.0
    for index, segment in enumerate((speech, pause)):
        duration = 0.1 if index == 0 else 1.25
        row = {"sequence": index, "segment_id": segment.id, "chapter_id": segment.chapter_id,
               "chapter_number": 91, "segment_index": index, "role_id": segment.role_id,
               "canonical_text": segment.text,
               "canonical_text_sha256": hashlib.sha256(segment.text.encode()).hexdigest(),
               "kind": "reused_narration" if index == 0 else "scene_break",
               "take_id": "take_example" if index == 0 else None,
               "duration_seconds": duration, "frames_24000": round(duration * 24000),
               "chapter_start_seconds": offset, "combined_start_seconds": offset,
               "chapter_end_seconds": offset + duration, "combined_end_seconds": offset + duration}
        if index == 0:
            row.update(audio_file="audio/take.wav", audio_sha256=_hash(wav), qa_passed=True,
                       qa_check_version="fixture-v1", qa_executed_checks=["decodes"])
        rows.append(row)
        offset += duration
    manifest = {"schema_version": 1, "status": "verified_listening_export", "project_id": project.id,
                "exporter_version": "three-chapter-listening-export-v3", "settings": {"scene_break_seconds": 1.25},
                "source_hashes": {}, "speaker_corrections_sha256": None, "ordered_rows": rows,
                "chapter_exports": [{"chapter_id": "chapter_example", "chapter_number": 91,
                                     "duration_seconds": offset,
                                     "ordered_segment_ids": [speech.id, pause.id],
                                     "artifacts": [{"file": mp3.name, "sha256": _hash(mp3),
                                                    "kind": "listening_export", "verified_loudness": True,
                                                    "probe": {"codec_name": "mp3", "duration_seconds": offset}}]}]}
    path = exports / "export-manifest.json"

    def save(*, refresh_state=True):
        if refresh_state:
            manifest["rendition_state_hashes"] = {name: _hash(root / name) for name in ("project.json", "segments.json")}
        manifest["source_fingerprint"] = projection._fingerprint(manifest)
        for chapter in manifest["chapter_exports"]:
            for artifact in chapter["artifacts"]:
                artifact["source_fingerprint"] = manifest["source_fingerprint"]
        _write(path, manifest)

    save()
    return root, path, manifest, save


def _discover(receipt):
    return projection.discover_reviewable(receipt[0], [receipt[1]])


def _mutate_segment(receipt, change):
    path = receipt[0] / "segments.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data[0])
    _write(path, data)
    receipt[3]()


def test_projects_exact_existing_audio_without_source_text_or_writes(receipt):
    root, path, manifest, _ = receipt
    files = {p: _hash(p) for p in root.parent.rglob("*") if p.is_file()}
    result = _discover(receipt)
    assert result["project_id"] == root.name
    assert result["book_title"] == "Fixture book"
    chapter = result["chapters"][0]
    assert chapter["chapter_number"] == 91  # No hardcoded production batch.
    assert chapter["audio_path"] == str(path.parent / "chapter.mp3")
    assert chapter["duration_seconds"] == 1.35
    assert chapter["revision"] == manifest["source_fingerprint"]
    assert chapter["selected_takes"] == [{"segment_id": "segment_speech", "take_id": "take_example",
                                          "audio_sha256": _hash(root / "audio/take.wav")}]
    encoded = json.dumps(result)
    assert "Private canonical" not in encoded and "canonical_text" not in encoded
    assert files == {p: _hash(p) for p in root.parent.rglob("*") if p.is_file()}
    assert projection.discover_reviewable(root, [path, path]) == result


@pytest.mark.parametrize("change", [
    lambda s: s.update(selected_take_id=None),
    lambda s: s["takes"][0].update(stale=True),
    lambda s: s["takes"][0].update(missing=True),
    lambda s: s["takes"][0].update(archived_at="2026-01-01T00:00:00Z"),
    lambda s: s["takes"][0].update(text="Changed private text"),
    lambda s: s["takes"][0].update(role_id="another_speaker"),
    lambda s: s.update(spoken_text_revision_id="new_revision"),
    lambda s: s["takes"][0].update(playback_speed=1.5),
    lambda s: s["takes"][0].update(qa_report=None),
    lambda s: s["takes"][0]["qa_report"].update(take_id="another_take"),
    lambda s: s["takes"][0]["qa_report"].update(output_audio_hash="0" * 64),
    lambda s: s["takes"][0]["qa_report"].update(passed=False),
    lambda s: s["takes"][0]["qa_report"]["results"][0].update(status="failed"),
    lambda s: s["takes"][0]["qa_report"]["results"][0].update(status="skipped"),
])
def test_rejects_unreviewable_current_take_even_with_fresh_manifest_state(receipt, change):
    _mutate_segment(receipt, change)
    with pytest.raises(ValueError):
        _discover(receipt)


def test_rejects_changed_canonical_snapshot(receipt):
    root = receipt[0]
    data = json.loads((root / "project.json").read_text(encoding="utf-8"))
    data["name"] = "Changed"
    _write(root / "project.json", data)
    with pytest.raises(ValueError, match="canonical state has changed"):
        _discover(receipt)


@pytest.mark.parametrize("target", ["source", "export"])
def test_rejects_modified_audio_bytes(receipt, target):
    root, path, _, _ = receipt
    audio = root / "audio/take.wav" if target == "source" else path.parent / "chapter.mp3"
    audio.write_bytes(audio.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        _discover(receipt)


@pytest.mark.parametrize("change", [
    lambda m: m["ordered_rows"].pop(),
    lambda m: m["ordered_rows"].reverse(),
    lambda m: m["ordered_rows"][0].update(kind="scene_break", take_id=None),
    lambda m: m["ordered_rows"][1].update(duration_seconds=2),
    lambda m: m["ordered_rows"][0].update(chapter_end_seconds=9),
    lambda m: m["chapter_exports"][0].update(duration_seconds=float("inf")),
    lambda m: m["chapter_exports"][0]["artifacts"][0].update(file="../private.mp3"),
    lambda m: m["chapter_exports"][0]["artifacts"][0].update(file="C:/private.mp3"),
    lambda m: m["chapter_exports"][0]["artifacts"][0].update(verified_loudness=False),
    lambda m: m["ordered_rows"][0].update(qa_executed_checks=[]),
])
def test_rejects_bad_coverage_timeline_receipts_or_path_escape(receipt, change):
    change(receipt[2])
    receipt[3]()
    with pytest.raises(ValueError):
        _discover(receipt)


def test_scene_break_requires_canonical_definition(receipt):
    _write(receipt[0] / "scene-breaks.json", [])
    with pytest.raises(ValueError, match="undeclared scene break"):
        _discover(receipt)


def test_source_audio_escape_is_rejected(receipt):
    _mutate_segment(receipt, lambda s: s["takes"][0].update(audio_file="../outside.wav"))
    receipt[2]["ordered_rows"][0]["audio_file"] = "../outside.wav"
    receipt[3]()
    with pytest.raises(ValueError, match="asset path"):
        _discover(receipt)


def test_corrupt_metadata_errors_never_expose_source_text(receipt):
    (receipt[0] / "segments.json").write_text('[{"text":"PRIVATE SECRET","index":"invalid"}]', encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        _discover(receipt)
    assert "PRIVATE SECRET" not in str(caught.value)
    assert "validation failed" in str(caught.value)


def test_detects_change_during_discovery(receipt, monkeypatch):
    real_sha = projection._sha
    path = receipt[0] / "segments.json"
    calls = 0

    def sha(candidate):
        nonlocal calls
        if candidate == path:
            calls += 1
            return "0" * 64
        return real_sha(candidate)

    monkeypatch.setattr(projection, "_sha", sha)
    with pytest.raises(ValueError, match="changed during discovery"):
        _discover(receipt)
    assert calls == 1


def test_fingerprint_is_not_an_arbitrary_revision_label(receipt):
    receipt[2]["source_fingerprint"] = "0" * 64
    _write(receipt[1], receipt[2])
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _discover(receipt)
