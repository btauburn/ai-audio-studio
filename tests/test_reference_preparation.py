"""Automatic source preparation safety and provenance tests; no GPU or downloads."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np
import pytest

from dubbing_service.app.audiobook import reference_preparation as prep
from dubbing_service.app.audiobook import reference_source_qc as qc


def source_report(pipeline, wave, rate, status="pass"):
    row = {"singleton_count": 2 if status == "review" else 1, "overlap_frames": 0,
           "overlap_peak": .01, "music": .4 if status == "contaminated" else .01,
           "noise": .01, "speech": .8, "start_seconds": 0., "duration_seconds": len(wave) / rate}
    return {"pipeline": pipeline["classifier"], "input_pcm_sha256": prep._sha(wave.tobytes()),
            "sample_rate": rate, "windows": [row], **qc.evaluate_windows([row])}


@pytest.fixture
def setup(monkeypatch, tmp_path):
    pipeline = {"schema": "automatic-reference-preparation-v1", "classifier": {"fixed": True},
                "separator_weights_sha256": prep.reference_separation.MODEL_SHA256,
                "separator_source_sha256": "frozen-v1", "quality": {"fixed_quality": True}}
    monkeypatch.setattr(prep, "_manifest", lambda: pipeline)
    wave = np.full(48000 * 4, .1, dtype=np.float32)
    calls = {"classify": 0, "separate": 0, "quality": 0}
    report = {"status": "pass", "checks": {k: {"status": "pass"} for k in prep._REQUIRED}}
    def classify(wave, rate):
        calls["classify"] += 1
        return source_report(pipeline, wave, rate)
    def quality(directory, wave, rate, text, language, origin):
        calls["quality"] += 1
        return report | {"input_pcm_sha256": prep._sha(wave.tobytes()), "sample_rate": rate,
            "pipeline": pipeline["quality"], "expected_text": text.strip() if text else None,
            "expected_text_sha256": prep._sha(text.strip().encode()) if text else None,
            "language": language, "expected_text_origin": origin}
    def separate(wave, rate):
        calls["separate"] += 1
        return wave.copy(), rate, {"input_pcm_sha256": prep._sha(wave.tobytes()),
            "output_pcm_sha256": prep._sha(wave.tobytes()), "input_sample_rate": rate,
            "output_sample_rate": rate, "output_samples": len(wave), "pipeline_sha256": "frozen-v1",
            "model_sha256": prep.reference_separation.MODEL_SHA256}
    monkeypatch.setattr(prep, "_classify", classify)
    monkeypatch.setattr(prep, "_quality", quality)
    monkeypatch.setattr(prep.reference_separation, "isolate_speech", separate)
    return tmp_path, wave, pipeline, calls, report


def test_clean_reference_skips_isolation_and_cache_skips_all_inference(setup):
    directory, wave, _, calls, _ = setup
    first = prep.prepare(directory, wave, 48000, "expected words")
    second = prep.prepare(directory, wave, 48000, "expected words")
    assert first[3]["status"] == second[3]["status"] == "pass"
    assert first[3] == second[3] and np.array_equal(first[0], second[0])
    assert first[4] is False and second[4] is True
    assert calls == {"classify": 1, "separate": 0, "quality": 1}
    assert second[2] is None and second[3]["processed_reference"] is False


@pytest.mark.parametrize("post_status", ["pass", "contaminated", "review"])
def test_one_contamination_pass_and_post_checks(setup, monkeypatch, post_status):
    directory, wave, pipeline, calls, _ = setup
    def classify(wave, rate):
        calls["classify"] += 1
        status = "contaminated" if calls["classify"] == 1 else post_status
        return source_report(pipeline, wave, rate, status)
    monkeypatch.setattr(prep, "_classify", classify)
    first = prep.prepare(directory, wave, 48000, "expected words")
    second = prep.prepare(directory, wave, 48000, "expected words")
    assert first[3]["status"] == ("pass" if post_status == "pass" else "hold")
    assert first[3]["isolation_attempts"] == 1 and first[3]["processed_reference"] is True
    assert first[3] == second[3] and second[4] is True
    assert calls == {"classify": 2, "separate": 1, "quality": 2}


def test_overlap_is_held_without_separation(setup, monkeypatch):
    directory, wave, _, calls, _ = setup
    monkeypatch.setattr(prep, "_classify", lambda w, r: source_report(setup[2], w, r, "review"))
    result = prep.prepare(directory, wave, 48000, "expected words")
    assert result[3]["status"] == "hold" and calls["separate"] == 0


@pytest.mark.parametrize("failed", ["asr", "transcript_alignment", "crop_coverage", "clipping"])
def test_failed_or_unavailable_checks_hold_without_cleanup(setup, failed):
    directory, wave, _, calls, report = setup
    report["checks"][failed]["status"] = "unavailable"
    result = prep.prepare(directory, wave, 48000, "expected words")
    assert result[3]["status"] == "hold" and calls["separate"] == 0


def test_missing_manifest_fail_closed(setup, monkeypatch):
    directory, wave, _, calls, _ = setup
    monkeypatch.setattr(prep, "_manifest", lambda: (_ for _ in ()).throw(ValueError("missing pinned model")))
    result = prep.prepare(directory, wave, 48000, "expected words")
    assert result[3]["status"] == "hold" and "missing pinned model" in result[3]["reason"]
    assert calls == {"classify": 0, "separate": 0, "quality": 0}


def test_pipeline_change_invalidates_cache(setup):
    directory, wave, pipeline, calls, _ = setup
    prep.prepare(directory, wave, 48000, "expected words")
    pipeline["revision"] = 2
    assert prep.prepare(directory, wave, 48000, "expected words")[4] is False
    assert calls["classify"] == 2


@pytest.mark.parametrize("change", ["audio", "request"])
def test_cache_corruption_holds_without_regeneration(setup, change):
    directory, wave, _, calls, _ = setup
    prep.prepare(directory, wave, 48000, "expected words")
    if change == "audio":
        next((directory / "automatic-preparation").glob("*.npz")).write_bytes(b"bad")
    else:
        path = next((directory / "automatic-preparation").glob("*.json"))
        record = json.loads(path.read_text())
        record["request"]["text"] = "changed"
        path.write_text(json.dumps(record))
    result = prep.prepare(directory, wave, 48000, "expected words")
    assert result[3]["status"] == "hold" and calls["classify"] == 1


def test_changed_text_and_pcm_have_distinct_cache_keys(setup):
    directory, wave, _, calls, _ = setup
    prep.prepare(directory, wave, 48000, "text one")
    prep.prepare(directory, wave, 48000, "text two")
    prep.prepare(directory, wave * 2, 48000, "text two")
    assert calls["classify"] == 3


def test_serial_preparation_and_lock_release(setup, monkeypatch):
    directory, wave, pipeline, _, _ = setup
    active = maximum = 0
    lock = Lock()
    def classify(wave, rate):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(active, maximum)
        time.sleep(.02)
        with lock: active -= 1
        return source_report(pipeline, wave, rate)
    monkeypatch.setattr(prep, "_classify", classify)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda i: prep.prepare(directory, wave, 48000, f"text {i}"), range(3)))
    assert maximum == 1 and all(r[3]["status"] == "pass" for r in results)


@pytest.mark.parametrize("duration", [2, 31])
def test_duration_bounds_before_inference(setup, duration):
    directory, _, _, calls, _ = setup
    with pytest.raises(ValueError):
        prep.prepare(directory, np.zeros(duration * 48000, dtype=np.float32), 48000, "text")
    assert calls["classify"] == 0


def test_classifier_gates_music_noise_overlap_speech_and_unavailable():
    row = {"singleton_count": 1, "overlap_frames": 0, "overlap_peak": .01,
           "music": .01, "noise": .01, "speech": .8, "start_seconds": 0., "duration_seconds": 10.}
    assert qc.evaluate_windows([row])["status"] == "pass"
    assert qc.evaluate_windows([row | {"music": .2}])["status"] == "contaminated"
    assert qc.evaluate_windows([row | {"noise": .2}])["status"] == "contaminated"
    for change in ({"singleton_count": 2}, {"singleton_count": 0}, {"overlap_frames": 1},
                   {"overlap_peak": .1}, {"speech": .5}):
        assert qc.evaluate_windows([row | change])["status"] == "review"
    assert qc.evaluate_windows([])["status"] == "unavailable"



@pytest.mark.parametrize("redigest", [False, True])
def test_cached_hold_cannot_be_promoted_by_status_edit(setup, monkeypatch, redigest):
    directory, wave, pipeline, calls, _ = setup
    monkeypatch.setattr(prep, "_classify", lambda w, r: source_report(pipeline, w, r, "review"))
    assert prep.prepare(directory, wave, 48000, "text")[3]["status"] == "hold"
    path = next((directory / "automatic-preparation").glob("*.json"))
    record = json.loads(path.read_text())
    record["evidence"]["status"] = "pass"
    if redigest:
        record.pop("record_sha256")
        record["record_sha256"] = prep._sha(prep._json(record))
    path.write_text(json.dumps(record))
    result = prep.prepare(directory, wave, 48000, "text")
    assert result[3]["status"] == "hold" and calls["separate"] == 0
    assert "changed" in result[3]["reason"] or "differs" in result[3]["reason"]


def test_failure_after_separation_is_truthful_and_cached(setup, monkeypatch):
    directory, wave, pipeline, calls, _ = setup
    def classify(w, r):
        calls["classify"] += 1
        if calls["classify"] == 2: raise RuntimeError("post classifier unavailable")
        return source_report(pipeline, w, r, "contaminated")
    monkeypatch.setattr(prep, "_classify", classify)
    first = prep.prepare(directory, wave, 48000, "text")
    second = prep.prepare(directory, wave, 48000, "text")
    assert first[3]["status"] == "hold" and first[3]["isolation_attempts"] == 1
    assert first[3]["processing_completed"] is True and first[3]["failure_stage"] == "post_source_qc"
    assert first[3] == second[3] and second[4] is True
    assert first[2] is None and np.array_equal(first[0], wave)
    assert calls["separate"] == 1


def test_pair_eviction_recomputes_prior_entry(setup, monkeypatch):
    directory, wave, _, calls, _ = setup
    monkeypatch.setattr(prep, "_CACHE_BYTES", 2_100_000)
    prep.prepare(directory, wave, 48000, "one")
    prep.prepare(directory, wave, 48000, "two")
    prep.prepare(directory, wave, 48000, "three")
    files = list((directory / "automatic-preparation").iterdir())
    assert len(files) == 4 and {p.suffix for p in files} == {".json", ".npz"}
    assert prep.prepare(directory, wave, 48000, "one")[4] is False
    assert calls["classify"] == 4


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.1, True])
def test_malformed_probabilities_rejected(value):
    row = {"singleton_count": 1, "overlap_frames": 0, "overlap_peak": .01,
           "music": value, "noise": .01, "speech": .8, "start_seconds": 0., "duration_seconds": 10.}
    with pytest.raises(ValueError): qc.evaluate_windows([row])


def test_independent_windows_require_global_speaker_review():
    row = {"singleton_count": 1, "overlap_frames": 0, "overlap_peak": .01,
           "music": .01, "noise": .01, "speech": .8, "start_seconds": 0., "duration_seconds": 10.}
    assert qc.evaluate_windows([row, row | {"start_seconds": 10.}])["status"] == "review"


def test_transcript_origin_never_self_compares_asr_draft():
    original = {"reference_text": "independent text"}
    draft = {"reference_text": "machine text", "transcript_evidence": {"machine_generated": True, "text": "machine text"}}
    assert prep.transcript_origin(original, "independent text") == "provided"
    assert prep.transcript_origin(draft, "machine text") == "machine_draft"
    assert prep.transcript_origin(draft, "new entry") == "unverified_entry"
    assert prep.transcript_origin(draft, "corrected text", {"text": "corrected text"}) == "user_correction"


@pytest.mark.parametrize("field,value", [("evidence", []), ("processing", []), ("processing", None)])
def test_malformed_cache_mapping_holds(setup, field, value):
    directory, wave, _, _, _ = setup
    prep.prepare(directory, wave, 48000, "text")
    path = next((directory / "automatic-preparation").glob("*.json"))
    record = json.loads(path.read_text())
    record[field] = value
    record.pop("record_sha256")
    record["record_sha256"] = prep._sha(prep._json(record))
    path.write_text(json.dumps(record))
    result = prep.prepare(directory, wave, 48000, "text")
    if field == "processing" and value is None:
        assert result[3]["status"] == "pass"  # Correct for unprocessed clean PCM.
    else:
        assert result[3]["status"] == "hold"


def test_compressed_npz_rejected_before_allocation():
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, wave=np.zeros(10000, dtype=np.float32), sample_rate=np.asarray(16000, dtype=np.int64))
    with pytest.raises(ValueError, match="ZIP"):
        prep._cached_pcm(buf.getvalue())


@pytest.mark.parametrize("field,value", [
 ("pipeline", {"changed": True}), ("expected_text", "other"), ("expected_text_origin", "machine_draft"),
 ("expected_text_sha256", "f" * 64), ("language", "de")])
def test_cached_quality_from_other_request_rejected(setup, field, value):
    directory, wave, _, calls, _ = setup
    prep.prepare(directory, wave, 48000, "text")
    path = next((directory / "automatic-preparation").glob("*.json"))
    record = json.loads(path.read_text())
    record["evidence"]["raw_quality"][field] = value
    record.pop("record_sha256")
    record["record_sha256"] = prep._sha(prep._json(record))
    path.write_text(json.dumps(record))
    result = prep.prepare(directory, wave, 48000, "text")
    assert result[3]["status"] == "hold" and calls["classify"] == 1


def test_machine_draft_requires_review_even_if_cached_checks_claim_pass(setup):
    directory, wave, _, _, _ = setup
    result = prep.prepare(directory, wave, 48000, "text", expected_origin="machine_draft")
    assert result[3]["status"] == "hold"
    assert prep.prepare(directory, wave, 48000, "text", expected_origin="machine_draft")[4] is True


def test_incomplete_cached_window_cannot_claim_full_crop(setup):
    directory, wave, _, _, _ = setup
    prep.prepare(directory, wave, 48000, "text")
    path = next((directory / "automatic-preparation").glob("*.json"))
    record = json.loads(path.read_text())
    record["evidence"]["raw_source_qc"]["windows"][0]["duration_seconds"] = 1.
    record.pop("record_sha256")
    record["record_sha256"] = prep._sha(prep._json(record))
    path.write_text(json.dumps(record))
    assert prep.prepare(directory, wave, 48000, "text")[3]["status"] == "hold"


def test_outer_provenance_failure_preserves_attempt_count(setup, monkeypatch):
    directory, wave, pipeline, _, _ = setup
    monkeypatch.setattr(prep, "_classify", lambda w, r: source_report(pipeline, w, r, "contaminated"))
    count = 0
    def manifest():
        nonlocal count
        count += 1
        if count > 1: raise ValueError("final pipeline drift")
        return pipeline
    monkeypatch.setattr(prep, "_manifest", manifest)
    result = prep.prepare(directory, wave, 48000, "text")
    assert result[3]["status"] == "hold" and result[3]["isolation_attempts"] == 1
    assert result[3]["failure_stage"] == "final_provenance"
    assert result[2] is None


def test_cache_write_failure_stage_is_serializable(setup, monkeypatch):
    directory, wave, _, _, _ = setup
    def replace(path, target):
        raise OSError("cache replace blocked")
    monkeypatch.setattr(prep.Path, "replace", replace)
    result = prep.prepare(directory, wave, 48000, "text")
    assert result[3]["status"] == "hold" and result[3]["failure_stage"] == "cache_write"
    json.dumps(result[3])


def test_cache_payload_frame_count_must_match_request(setup):
    import io
    directory, wave, _, _, _ = setup
    prep.prepare(directory, wave, 48000, "text")
    meta = next((directory / "automatic-preparation").glob("*.json"))
    record = json.loads(meta.read_text())
    audio = meta.with_suffix(".npz")
    raw = io.BytesIO()
    shortened = wave[:-1000]
    np.savez(raw, wave=shortened, sample_rate=np.asarray(48000, dtype=np.int64))
    audio.write_bytes(raw.getvalue())
    record["output_sha256"] = prep._sha(raw.getvalue())
    record["output_pcm_sha256"] = prep._sha(shortened.tobytes())
    record.pop("record_sha256")
    record["record_sha256"] = prep._sha(prep._json(record))
    meta.write_text(json.dumps(record))
    assert prep.prepare(directory, wave, 48000, "text")[3]["status"] == "hold"
