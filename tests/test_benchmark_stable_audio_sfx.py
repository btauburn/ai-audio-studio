from __future__ import annotations

import array
import csv
import hashlib
import json
import wave
from pathlib import Path

import pytest

from scripts import benchmark_stable_audio_sfx as benchmark


def _v1_missing_manifest(path: Path) -> None:
    payload = b"model"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "Missing benchmark model",
                "source": "unit test",
                "revision": "pinned",
                "model_id": "missing_benchmark_model",
                "sources": {
                    "model": {
                        "provider": "provenance_only",
                    }
                },
                "files": [
                    {
                        "group": "runtime",
                        "path": "models/missing/model.bin",
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "source": "model",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_benchmark_plan_is_exactly_12_categories_24_quality_and_30_technical() -> None:
    manifest = {
        "model_id": "stable_audio_3_small_sfx",
        "revision": "a" * 40,
        "code_revision": "c" * 40,
        "code_source": {
            "distribution": "stable-audio-3",
            "url": "https://github.com/Stability-AI/stable-audio-3.git",
        },
        "sources": {"model": {"revision": "a" * 40}},
        "files": [
            {
                "group": "runtime",
                "path": "models/stable/model.bin",
                "size": 1,
                "sha256": "b" * 64,
            }
        ],
    }

    plan = benchmark._build_plan(manifest)

    assert plan["code_revision"] == "c" * 40
    assert plan["code_distribution"] == "stable-audio-3"
    assert plan["code_source_url"] == (
        "https://github.com/Stability-AI/stable-audio-3.git"
    )
    assert benchmark._plan_matches_manifest(plan, manifest)
    stale_manifest = dict(manifest, code_revision="d" * 40)
    assert not benchmark._plan_matches_manifest(plan, stale_manifest)
    assert len(benchmark.QUALITY_CATEGORIES) == 12
    assert len({item["category"] for item in plan["quality_cases"]}) == 12
    assert len(plan["quality_cases"]) == 24
    assert all(
        len([case for case in plan["quality_cases"] if case["category"] == category["id"]])
        == 2
        for category in benchmark.QUALITY_CATEGORIES
    )
    assert len(plan["technical_runs"]) == 30
    assert {
        round_number: sum(
            item["round"] == round_number for item in plan["technical_runs"]
        )
        for round_number in range(1, 6)
    } == {1: 6, 2: 6, 3: 6, 4: 6, 5: 6}


def test_wav_analysis_enforces_stereo_44100_pcm16_and_reports_metrics(
    tmp_path: Path,
) -> None:
    wav_path = tmp_path / "valid.wav"
    samples = array.array("h")
    for frame in range(44_100):
        value = 4_000 if frame % 200 < 100 else -4_000
        samples.extend((value, value))
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44_100)
        handle.writeframes(samples.tobytes())

    metrics = benchmark.analyze_wav(wav_path, 1.0)

    assert metrics["passed"] is True
    assert metrics["sample_rate"] == 44_100
    assert metrics["channels"] == 2
    assert metrics["duration_seconds"] == pytest.approx(1.0)
    assert metrics["rms"] > 0.1
    assert metrics["sha256"] == hashlib.sha256(wav_path.read_bytes()).hexdigest()


def test_manual_score_gate_accepts_one_passing_candidate_per_category(tmp_path: Path) -> None:
    manifest = {
        "model_id": "stable_audio_3_small_sfx",
        "revision": "a" * 40,
        "sources": {"model": {"revision": "a" * 40}},
        "files": [
            {
                "group": "runtime",
                "path": "models/stable/model.bin",
                "size": 1,
                "sha256": "b" * 64,
            }
        ],
    }
    plan = benchmark._build_plan(manifest)
    score_path = tmp_path / "manual_scores.csv"
    benchmark._write_score_template(score_path, plan)
    with score_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["semantic_match_1_5"] = "5"
        row["event_timing_1_5"] = "4"
        row["artifact_control_1_5"] = "4"
        row["background_cleanliness_1_5"] = "4"
        row["usable_yes_no"] = "yes"
    rows[0]["semantic_match_1_5"] = "2"
    with score_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=benchmark.SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    scored, categories = benchmark._read_scores(score_path, plan)

    assert len(scored) == 24
    assert sum(item["passed"] for item in categories) == 12
    assert categories[0]["cases_passed"] == 1
    assert categories[0]["passed"] is True


def test_missing_weights_fail_before_worker_and_never_create_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "models" / "manifests" / "missing.json"
    _v1_missing_manifest(manifest_path)
    results_dir = tmp_path / "results"
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("DUBBING_WORKER_INSTANCE_TOKEN", "not-used")

    exit_code = benchmark.main(
        [
            "--generate",
            "--manifest",
            str(manifest_path),
            "--results-dir",
            str(results_dir),
            "--endpoint",
            "http://127.0.0.1:1/should-not-be-called",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "weights are missing or unverified" in captured.err
    assert not results_dir.exists()
    assert not list(tmp_path.rglob("quality_gate.json"))


def test_warm_latency_gate_uses_five_warm_runs_per_handoff_round() -> None:
    technical = []
    for round_number in range(1, 6):
        for slot in range(1, 7):
            technical.append(
                {
                    "id": f"r{round_number}s{slot}",
                    "round": round_number,
                    "slot": slot,
                    "duration_seconds": 3.0,
                    "latency_budget_seconds": 5.0,
                    "metrics": {"elapsed_seconds": 4.0},
                }
            )

    passing = benchmark._warm_latency_gate(technical)
    assert passing["runs_total"] == 25
    assert passing["p95_passed"] is True

    for item in technical:
        if item["slot"] >= 2:
            item["metrics"]["elapsed_seconds"] = 5.5
    failing = benchmark._warm_latency_gate(technical)
    assert failing["p95_budget_ratio"] == pytest.approx(1.1)
    assert failing["p95_passed"] is False


def test_control_endpoints_must_be_explicit_loopback() -> None:
    benchmark._require_loopback_endpoint(
        "http://127.0.0.1:8005/api/v1/sfx/load-model", "SFX load"
    )
    with pytest.raises(benchmark.BenchmarkError, match="loopback"):
        benchmark._require_loopback_endpoint(
            "https://example.com/api/v1/tts/unload-model", "TTS unload"
        )
