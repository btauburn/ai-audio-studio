"""Phases 5/6/8/13: the resolved language is decided once and then kept.

The defect these tests close: the renderer, the synthesis requirement and QA each
re-derived the language with a slightly different rule, and the Take recorded no
proof of which value was used.  A role edited after the audio existed therefore
changed how that audio was interpreted, and the corpus could not be re-audited
from its own evidence.

Nothing here is mocked: real storage, real ``render_utterances``, a real second
interpreter for the reload assertion.  The renderer is ``FixtureRenderer``, which
labels its output ``fixture`` -- these tests are about provenance plumbing and
prove nothing about engine behaviour.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dubbing_service.app.audiobook import (
    fingerprint,
    language,
    pipeline,
    qa,
    renderer,
    takes,
)
from dubbing_service.app.models.audiobook import QaCoverageProfile
from dubbing_service.app.storage.projects import save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audiobook_helpers import build_project, build_segments, use_projects_root

REPO_ROOT = Path(__file__).resolve().parents[1]

ENGLISH_TEXT = "The ledger stayed open on the desk for three days before anyone read it."
CHINESE_TEXT = "他把账本摊在桌上整整三天，直到有人终于敢翻开最后一页。"

RELOAD_CHILD = '''
import json
import sys

from dubbing_service.app.storage.segments import load_segments

payload = {}
for segment in load_segments(sys.argv[1]):
    takes = []
    for take in segment.takes:
        report = take.qa_report
        profile = None if report is None else report.coverage_profile
        takes.append(
            {
                "id": take.id,
                "language": take.language,
                "qwen3_language": take.qwen3_language,
                "input_fingerprint": take.input_fingerprint,
                "stale": take.stale,
                "profile": None if profile is None else profile.profile,
                "profile_language": None if profile is None else profile.language,
            }
        )
    payload[segment.id] = {
        "selected_take_id": segment.selected_take_id,
        "takes": takes,
    }
print(json.dumps(payload))
'''


@pytest.fixture()
def studio(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    project = build_project()
    project.roles[0].qwen3_language = "English"
    save_project(project)
    segments = build_segments(project, [ENGLISH_TEXT, CHINESE_TEXT])
    return root, project, segments


def _fixture(project_id: str, *, ms_per_char: float | None = None):
    kwargs = {} if ms_per_char is None else {"ms_per_char": ms_per_char}
    return renderer.FixtureRenderer(
        project_root=pipeline.project_root(project_id), renderer_id="fixture", **kwargs
    )


def _render(project_id: str, segment_ids, *, ms_per_char: float | None = None):
    return pipeline.render_utterances(
        project_id,
        list(segment_ids),
        renderer=_fixture(project_id, ms_per_char=ms_per_char),
    )


def _reload(projects_root: Path, tmp_path: Path, project_id: str) -> dict:
    """Reload the project in a fresh interpreter and report what it stored."""

    script = Path(tmp_path) / "reload_language.py"
    script.write_text(RELOAD_CHILD.strip() + "\n", encoding="utf-8")
    env = dict(os.environ)
    existing = env.get("PYTHONPATH") or ""
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    env["PYTHONUTF8"] = "1"
    env["DUBBING_PROJECTS_ROOT"] = str(projects_root)
    completed = subprocess.run(
        [sys.executable, str(script), project_id],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _segments_path(projects_root: Path, project_id: str) -> Path:
    return Path(projects_root) / project_id / "segments.json"


def _rewrite_takes(projects_root: Path, project_id: str, mutate) -> None:
    """Edit the committed Takes as a previous version of the code would have."""

    path = _segments_path(projects_root, project_id)
    document = json.loads(path.read_text(encoding="utf-8"))
    for segment in document:
        for take in segment.get("takes", []):
            mutate(take)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


# --- the one rule -----------------------------------------------------------


def test_resolver_reads_segment_then_role_then_default():
    role = SimpleNamespace(qwen3_language="Chinese")
    assert language.resolve_synthesis_language(
        SimpleNamespace(qwen3_language="English"), role
    ) == "English"
    assert language.resolve_synthesis_language(SimpleNamespace(), role) == "Chinese"
    # A declared role outranks a caller's fallback; the fallback is a last resort.
    assert (
        language.resolve_synthesis_language(SimpleNamespace(), role, default="Japanese")
        == "Chinese"
    )
    assert (
        language.resolve_synthesis_language(SimpleNamespace(), None, default="Japanese")
        == "Japanese"
    )
    assert language.resolve_synthesis_language(SimpleNamespace(), None) is None


def test_resolver_folds_case_and_aliases_but_keeps_a_typo_visible():
    assert language.normalize_language(" English ") == "English"
    assert language.normalize_language("english") == "English"
    assert language.normalize_language("zh-CN") == "Chinese"
    assert language.normalize_language("   ") is None
    assert language.normalize_language(None) is None
    # An unrecognized value is reported as declared, not silently dropped.
    assert language.normalize_language(" Klingon ") == "Klingon"
    assert set(language.SUPPORTED_LANGUAGES) >= {"Auto", "Chinese", "English"}


def test_language_changes_the_requirement_fingerprint(studio):
    _root, project, segments = studio
    role = project.roles[0]
    segment = segments[0]

    role.qwen3_language = "English"
    english = fingerprint.requirement_for_segment(
        segment, project_id=project.id, engine_id="fixture", renderer_id="fixture", role=role
    )
    role.qwen3_language = "Chinese"
    chinese = fingerprint.requirement_for_segment(
        segment, project_id=project.id, engine_id="fixture", renderer_id="fixture", role=role
    )

    assert "language" in fingerprint.AUDIO_AFFECTING_COMPONENTS
    assert english.fingerprint() != chinese.fingerprint()
    # ``voice_revision`` also moves, because it folds in the role's own language.
    # The component that matters here is ``language``: it is the value the
    # renderer and QA actually used.
    assert "language" in fingerprint.changed_components(
        english.components(), chinese.components()
    )
    assert english.components()["language"] == "English"
    assert chinese.components()["language"] == "Chinese"


def test_an_already_resolved_language_is_not_re_derived(studio):
    _root, project, segments = studio
    role = project.roles[0]  # declares English
    requirement = fingerprint.requirement_for_segment(
        segments[0],
        project_id=project.id,
        engine_id="fixture",
        renderer_id="fixture",
        role=role,
        language="Chinese",
    )
    assert requirement.language == "Chinese"
    assert requirement.components()["language"] == "Chinese"


# --- persisted on the Take --------------------------------------------------


def test_render_persists_the_language_the_role_declared(studio):
    _root, project, segments = studio
    record = _render(project.id, [segments[0].id])[0]
    assert record.success is True

    segment = load_segments(project.id)[0]
    take = segment.takes[0]
    assert take.language == "English"
    assert take.fingerprint_components["language"] == "English"
    assert take.qa_report is not None
    assert take.qa_report.coverage_profile == QaCoverageProfile(
        language="English", profile="english", profile_version=qa.COVERAGE_PROFILE_VERSION
    )


def test_segment_declaration_overrides_the_role_and_is_persisted(studio):
    _root, project, segments = studio
    target = segments[0]
    target.qwen3_language = "Japanese"
    save_segments(project.id, segments)

    _render(project.id, [target.id])

    take = load_segments(project.id)[0].takes[0]
    assert take.language == "Japanese"
    assert take.qwen3_language == "Japanese"
    assert take.fingerprint_components["language"] == "Japanese"


def test_undeclared_language_is_persisted_as_absent_not_guessed(studio):
    _root, project, segments = studio
    project.roles[0].qwen3_language = None
    save_project(project)

    _render(project.id, [segments[0].id])

    take = load_segments(project.id)[0].takes[0]
    # The engine's own configured default is deliberately not adopted here: the
    # Take records that nothing declared a language.
    assert take.language is None
    assert take.fingerprint_components["language"] == ""
    assert take.qa_report.coverage_profile.language is None
    # ...and QA still names the window it actually applied.
    assert take.qa_report.coverage_profile.profile == "english"


def test_qa_profile_identity_is_recorded_for_the_language_that_was_judged(studio):
    _root, project, segments = studio
    role = project.roles[0]
    role.qwen3_language = "Chinese"
    save_project(project)

    _render(project.id, [segments[0].id], ms_per_char=200.0)

    take = load_segments(project.id)[0].takes[0]
    assert take.language == "Chinese"
    assert take.qa_report.coverage_profile.profile == "chinese"
    assert take.qa_report.coverage_profile.language == "Chinese"
    assert take.qa_report.coverage_profile.profile_version == qa.COVERAGE_PROFILE_VERSION


def test_coverage_profile_identity_names_the_window_even_without_a_language():
    assert qa.coverage_profile_identity("en").profile == "english"
    assert qa.coverage_profile_identity("Chinese").profile == "chinese"
    unnamed = qa.coverage_profile_identity(None)
    assert unnamed.language is None
    assert unnamed.profile == "english"


# --- immutability -----------------------------------------------------------


def test_editing_the_role_does_not_rewrite_an_existing_take(studio):
    _root, project, segments = studio
    _render(project.id, [segments[0].id])

    before = load_segments(project.id)[0].takes[0]
    assert before.language == "English"
    fingerprint_before = before.input_fingerprint

    # The role is re-declared as Chinese *after* the audio existed.
    project.roles[0].qwen3_language = "Chinese"
    save_project(project)
    _render(project.id, [segments[1].id], ms_per_char=200.0)

    segments = load_segments(project.id)
    old_take = segments[0].takes[0]
    new_take = segments[1].takes[0]

    assert old_take.language == "English"
    assert old_take.qwen3_language is None
    assert old_take.input_fingerprint == fingerprint_before
    assert old_take.fingerprint_components["language"] == "English"
    assert old_take.qa_report.coverage_profile.profile == "english"

    assert new_take.language == "Chinese"
    assert new_take.qa_report.coverage_profile.profile == "chinese"

    # The mismatch is *detectable* rather than silently assumed away.
    assert old_take.stale is True
    assert "language" in old_take.stale_components


def test_reaudit_uses_the_take_not_the_current_role(studio):
    _root, project, segments = studio
    project.roles[0].qwen3_language = "Chinese"
    save_project(project)
    _render(project.id, [segments[0].id], ms_per_char=200.0)
    take_id = load_segments(project.id)[0].takes[0].id

    # The role is edited after the fact; the Take's own evidence must still win.
    project.roles[0].qwen3_language = "English"
    save_project(project)

    report = pipeline.reaudit_take(project.id, segments[0].id, take_id)
    assert report.coverage_profile.language == "Chinese"
    assert report.coverage_profile.profile == "chinese"
    assert report.passed is True


def test_reaudit_of_a_take_with_no_language_says_so_instead_of_inventing_one(studio):
    _root, project, segments = studio
    _render(project.id, [segments[0].id])
    take_id = load_segments(project.id)[0].takes[0].id
    _rewrite_takes(_root, project.id, lambda take: take.pop("language", None))

    report = pipeline.reaudit_take(project.id, segments[0].id, take_id)
    assert report.coverage_profile.language is None
    assert report.coverage_profile.profile == "english"
    assert report.notes and "未推断语言" in report.notes


# --- durability -------------------------------------------------------------


def test_take_language_survives_a_real_reload(studio, tmp_path):
    projects_root, project, segments = studio
    _render(project.id, [segments[0].id])
    # Second utterance declares its own language, so one committed file has to
    # carry two different resolved values at once.  Reload first: the in-memory
    # list predates the first render and would drop its Take.
    live = load_segments(project.id)
    live[1].qwen3_language = "Chinese"
    save_segments(project.id, live)
    _render(project.id, [segments[1].id], ms_per_char=200.0)

    reloaded = _reload(projects_root, tmp_path, project.id)

    assert reloaded[segments[0].id]["takes"][0]["language"] == "English"
    assert reloaded[segments[0].id]["takes"][0]["profile"] == "english"
    assert reloaded[segments[0].id]["takes"][0]["profile_language"] == "English"
    assert reloaded[segments[1].id]["takes"][0]["language"] == "Chinese"
    assert reloaded[segments[1].id]["takes"][0]["profile"] == "chinese"


# --- legacy Takes -----------------------------------------------------------


def test_a_take_written_without_a_language_field_still_loads(studio, tmp_path):
    projects_root, project, segments = studio
    _render(project.id, [segments[0].id])
    _rewrite_takes(projects_root, project.id, lambda take: take.pop("language", None))

    take = load_segments(project.id)[0].takes[0]

    assert take.language is None
    # The older record's own evidence is untouched, so nothing is retro-invalidated.
    assert take.fingerprint_components["language"] == "English"
    assert take.qa_report.coverage_profile.language == "English"

    role = project.roles[0]
    live = fingerprint.requirement_for_segment(
        segments[0], project_id=project.id, engine_id="fixture", renderer_id="fixture", role=role
    )
    assert takes.take_is_stale(take, live) == (False, [])


def test_language_evidence_report_reads_takes_only(studio):
    _root, project, segments = studio
    _render(project.id, [segments[0].id])
    project.roles[0].qwen3_language = "Chinese"
    save_project(project)
    _render(project.id, [segments[1].id], ms_per_char=200.0)

    report = pipeline.language_evidence_report(project.id)

    assert report["counts"] == {
        "takes": 2,
        "with_language": 2,
        "undeclared": 0,
        "legacy_unknown": 0,
    }
    rows = {row["persisted_language"]: row for row in report["takes"]}
    assert set(rows) == {"English", "Chinese"}
    assert rows["English"]["coverage_profile"] == {
        "language": "English",
        "profile": "english",
        "profile_version": qa.COVERAGE_PROFILE_VERSION,
    }
    assert rows["Chinese"]["coverage_profile"]["profile"] == "chinese"
    assert all(row["duration_ms"] > 0 for row in report["takes"])
    assert all(row["utterance_id"] in {segments[0].id, segments[1].id} for row in report["takes"])


def test_language_evidence_report_labels_undeclared_and_legacy_takes(studio):
    _root, project, segments = studio
    _render(project.id, [segments[0].id])
    _render(project.id, [segments[1].id])

    committed = load_segments(project.id)
    undeclared_id = committed[0].takes[0].id
    legacy_id = committed[1].takes[0].id

    def strip(take):
        take.pop("language", None)
        if take["id"] == legacy_id:
            # A Take older still: no requirement stamp at all.
            take.pop("input_fingerprint", None)
            take.pop("fingerprint_components", None)

    _rewrite_takes(_root, project.id, strip)

    report = pipeline.language_evidence_report(project.id)
    labels = {row["take_id"]: row["persisted_language"] for row in report["takes"]}

    assert labels[undeclared_id] == "undeclared"
    assert labels[legacy_id] == "legacy_unknown"
    assert report["counts"] == {
        "takes": 2,
        "with_language": 0,
        "undeclared": 1,
        "legacy_unknown": 1,
    }


# --- fixture honesty --------------------------------------------------------


def test_fixture_renders_are_labelled_as_fixture_and_model_identity_is_absent(studio):
    _root, project, segments = studio
    _render(project.id, [segments[0].id])

    take = load_segments(project.id)[0].takes[0]

    # A fixture Take can never be presented as engine output.
    assert take.renderer_id == "fixture"
    assert take.engine_id == "fixture"
    assert take.fingerprint_components["engine_id"] == "fixture"
    # Model identity is still unrecorded; this asserts the status quo rather than
    # claiming a model that was never consulted.
    assert take.model_id is None
    assert take.fingerprint_components["model_id"] == ""
