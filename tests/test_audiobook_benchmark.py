"""Phase 16: the benchmark corpus is a spec, not a wish list.

The corpus has to be stable (two runs are comparable), small enough to run as a
fixture, and honest about what each case actually exercises — a case labelled
"chunking" that never crosses the chunk threshold is worse than no case at all.
Every claim here is checked against the real pipeline on the fixture renderer;
nothing in this file says anything about RTX performance.
"""

from __future__ import annotations

import json

import pytest

from dubbing_service.app.audiobook import benchmark, pipeline, renderer
from dubbing_service.app.storage.projects import save_project
from dubbing_service.app.storage.segments import save_segments
from tests.audiobook_helpers import build_project, use_projects_root

EXPECTED_FINGERPRINT = "5a5624a739be056cf5c841feca0ccce87d1da3f274983a1f329c10f48f840c76"
EXPECTED_CATEGORIES = {
    "baseline",
    "dialogue",
    "casting",
    "emotion",
    "pronunciation",
    "normalization",
    "dynamics",
    "chunking",
    "identity",
    "structure",
    "attribution",
    "cues",
    "provenance",
}


@pytest.fixture()
def bench(monkeypatch, tmp_path):
    use_projects_root(monkeypatch, tmp_path)
    project = build_project(name="benchmark")
    materialized = benchmark.materialize_benchmark(project)
    save_project(project)
    save_segments(project.id, materialized["segments"])
    return project, materialized


def _fixture_renderer(project):
    return renderer.FixtureRenderer(
        project_root=pipeline.project_root(project.id), renderer_id="fixture"
    )


def test_the_corpus_is_thirteen_cases_and_thirty_six_utterances():
    assert len(benchmark.CORPUS) == 13
    assert benchmark.corpus_utterance_count() == 36
    assert {spec.category for spec in benchmark.CORPUS} == EXPECTED_CATEGORIES


def test_case_ids_and_segment_ids_are_unique():
    case_ids = benchmark.case_ids()
    segment_ids = [item for spec in benchmark.CORPUS for item in spec.segment_ids()]

    assert len(case_ids) == len(set(case_ids))
    assert len(segment_ids) == len(set(segment_ids))
    assert all(spec.utterances for spec in benchmark.CORPUS)


def test_every_utterance_has_a_known_speaker_and_non_empty_text():
    known = set(benchmark.roles_for_corpus())

    for spec in benchmark.CORPUS:
        for item in spec.utterances:
            assert item.speaker in known
            assert item.text.strip()
            assert item.text == item.text.strip()


def test_the_corpus_fingerprint_is_stable_and_pinned():
    first = benchmark.corpus_fingerprint()

    assert first == benchmark.corpus_fingerprint()
    assert len(first) == 64
    assert first == EXPECTED_FINGERPRINT


def test_the_corpus_is_fixture_scale_and_not_borrowed_text():
    total = sum(len(item.text) for spec in benchmark.CORPUS for item in spec.utterances)
    payload = json.dumps(benchmark.corpus_payload(), ensure_ascii=False).lower()

    assert total < 5000, "benchmark must stay fixture-scale, not a whole book"
    assert "©" not in payload
    assert "all rights reserved" not in payload
    assert "http" not in payload


def test_cases_expose_the_agent_facing_schema():
    cases = benchmark.corpus_cases()

    assert [case.case_id for case in cases] == benchmark.case_ids()
    for spec, case in zip(benchmark.CORPUS, cases):
        assert case.expected_speakers == spec.speakers()
        assert case.expectations["chapter"] == spec.chapter
        assert case.expectations["utterance_count"] == len(spec.utterances)
        assert case.text == spec.script()
        assert case.category == spec.category


def test_the_chunking_case_actually_crosses_the_chunk_threshold():
    spec = next(item for item in benchmark.CORPUS if item.case_id == "long_paragraph")
    parts = renderer.plan_chunks(spec.utterances[0].text)

    assert len(parts) > 1
    assert "".join(parts) == spec.utterances[0].text


def test_materialising_the_corpus_creates_roles_chapters_and_utterances(bench):
    project, materialized = bench

    assert materialized["benchmark_version"] == benchmark.BENCHMARK_VERSION
    assert materialized["corpus_fingerprint"] == EXPECTED_FINGERPRINT
    assert materialized["roles"] == benchmark.roles_for_corpus()
    assert {role.name for role in project.roles} >= set(benchmark.roles_for_corpus())
    assert len(materialized["created_chapters"]) == 13
    assert len(materialized["segments"]) == 36
    assert all(segment.role_id for segment in materialized["segments"])
    assert [segment.id for segment in materialized["segments"]] == [
        item for spec in benchmark.CORPUS for item in spec.segment_ids()
    ]


def test_an_unrendered_benchmark_project_reports_every_case_as_missing(bench):
    project, _materialized = bench

    report = benchmark.benchmark_report(project.id)

    assert report["case_count"] == 13
    assert report["utterance_count"] == 36
    assert report["coverage"]["ready"] is False
    assert len(report["cases"]) == 13
    for case_id, payload in report["cases"].items():
        assert payload["takes"] == 0
        assert len(payload["missing"]) == payload["utterances"]
        assert payload["stale"] == []


def test_rendering_the_whole_corpus_covers_every_case(bench):
    project, materialized = bench
    ids = [segment.id for segment in materialized["segments"]]

    records = pipeline.render_utterances(
        project.id, ids, renderer=_fixture_renderer(project)
    )

    assert len(records) == 36
    assert all(record.success for record in records)
    report = benchmark.benchmark_report(project.id)
    assert report["coverage"]["ready"] is True
    assert report["coverage"]["covered"] == 36
    for case_id, payload in report["cases"].items():
        assert payload["missing"] == [], case_id
        assert payload["stale"] == [], case_id
        assert payload["takes"] == payload["utterances"]
        assert all(len(digest) == 64 for digest in payload["audio_hashes"])


def test_the_long_paragraph_stays_one_take_even_when_chunked(bench):
    project, materialized = bench
    long_id = benchmark.benchmark_segment_id("long_paragraph", 0)

    records = pipeline.render_utterances(
        project.id, [long_id], renderer=_fixture_renderer(project)
    )

    assert len(records) == 1
    assert records[0].chunked is True
    from dubbing_service.app.storage.segments import load_segments

    segment = next(item for item in load_segments(project.id) if item.id == long_id)
    assert len(segment.takes) == 1
    assert len(segment.takes[0].chunk_map) > 1


def test_repeated_lines_are_separate_utterances_with_separate_takes(bench):
    project, materialized = bench
    ids = materialized["by_case"]["repeated_short_dialogue"]

    pipeline.render_utterances(project.id, ids, renderer=_fixture_renderer(project))

    report = benchmark.benchmark_report(project.id)["cases"]["repeated_short_dialogue"]
    assert report["utterances"] == 4
    assert report["takes"] == 4


def test_the_benchmark_report_makes_no_runtime_claims(bench):
    project, _materialized = bench

    payload = json.dumps(benchmark.benchmark_report(project.id)).lower()

    for forbidden in ("rtx", "cuda", "gpu", "vram", "tokens_per_second", "latency_ms"):
        assert forbidden not in payload
