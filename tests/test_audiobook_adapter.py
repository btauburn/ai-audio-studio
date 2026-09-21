"""Phase 3-4: audiobooker adapter, offline fallback, and the import boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from dubbing_service.app.audiobook import adapter
from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.models.audiobook import Attribution
from dubbing_service.app.storage.projects import load_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audiobook_helpers import (
    build_project,
    build_segments,
    clean_environment,
    use_projects_root,
)

SAMPLE = """Chapter 1 - Arrival

Mara: You moved the chairs.
Tomas: I moved one of them.

[Ilse] The bridge is out past the mill.

The ledger stayed open on the desk for three days.

Chapter 2 - Departure

Narrator: The next morning the ledger was closed.
"""

NARRATION_ONLY = """Chapter 1 - Plain

The harbour was empty. Nobody had touched the nets since the storm.

By evening the lamps were lit anyway.
"""


@pytest.fixture()
def studio(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    clean_environment()
    yield root, tmp_path
    clean_environment()


def _sample_file(tmp_path: Path, text: str = SAMPLE) -> Path:
    path = Path(tmp_path) / "book.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _reload(project_id: str):
    return load_project(project_id), load_segments(project_id)


def test_offline_analysis_is_labelled_and_never_claims_audiobooker(studio):
    analysis = adapter.analyze_text_offline(
        SAMPLE, title="Sample", author="Nobody", language="en"
    )

    assert analysis.title == "Sample"
    assert analysis.author == "Nobody"
    assert analysis.language == "en"
    assert analysis.schema == adapter.ADAPTER_SCHEMA_VERSION
    assert analysis.utterance_count() > 0
    assert any(
        adapter.OFFLINE_ANALYSIS_SOURCE in warning for warning in analysis.warnings
    ), "offline analysis must be identifiable from its warnings"


def test_offline_analysis_detects_chapters_speakers_and_narration(studio):
    analysis = adapter.analyze_text_offline(SAMPLE)

    assert [chapter.title for chapter in analysis.chapters] == [
        "Chapter 1 - Arrival",
        "Chapter 2 - Departure",
    ]
    dialogue = [
        item
        for item in analysis.chapters[0].utterances
        if item.utterance_type == "dialogue"
    ]
    assert [item.speaker for item in dialogue] == ["Mara", "Tomas", "Ilse"]
    assert all(item.attribution_source == "explicit_marker" for item in dialogue)
    assert all(item.confidence == 1.0 for item in dialogue)
    assert set(analysis.speakers) == {"Mara", "Tomas", "Ilse", "Narrator"}


def test_offline_analysis_records_unattributed_text_as_narration(studio):
    analysis = adapter.analyze_text_offline(NARRATION_ONLY)

    utterances = analysis.chapters[0].utterances
    assert utterances, "narration must still produce utterances"
    assert all(item.speaker is None for item in utterances)
    assert all(item.utterance_type == "narration" for item in utterances)
    assert all(item.attribution_source == "unknown" for item in utterances)


def test_offline_analysis_never_invents_chapters_for_plain_text(studio):
    analysis = adapter.analyze_text_offline("One plain line. And a second one.")

    assert len(analysis.chapters) == 1
    assert analysis.chapters[0].utterances[0].attribution_source == "unknown"


def test_analyze_source_file_uses_the_offline_fallback_when_audiobooker_is_absent(
    studio, monkeypatch
):
    _root, tmp_path = studio
    monkeypatch.setattr(
        adapter,
        "availability",
        lambda refresh=False: adapter.Availability(
            available=False, reason="audiobooker not installed in this interpreter"
        ),
    )

    analysis = adapter.analyze_source_file(_sample_file(tmp_path))

    assert analysis.title == "book"
    assert analysis.utterance_count() > 0
    assert analysis.source_path is None
    assert any(
        adapter.OFFLINE_ANALYSIS_SOURCE in warning for warning in analysis.warnings
    )


def test_analyze_source_file_refuses_a_missing_path(studio):
    _root, tmp_path = studio
    with pytest.raises(adapter.AdapterError):
        adapter.analyze_source_file(Path(tmp_path) / "absent.txt")


def test_availability_reports_the_pinned_version_without_claiming_success():
    info = adapter.availability(refresh=True)

    assert info.as_dict()["pinned_version"] == adapter.PINNED_AUDIOBOOKER_VERSION
    if info.available:
        assert info.version == adapter.PINNED_AUDIOBOOKER_VERSION, (
            "a different audiobooker version is not the pinned contract"
        )
    else:
        assert info.reason, "an unavailable adapter must explain itself"
        assert any(
            adapter.OFFLINE_ANALYSIS_SOURCE in warning
            for warning in adapter.analyze_text_offline(SAMPLE).warnings
        ), "the offline path must stay available when audiobooker is missing"


def test_interpreter_search_follows_the_code_not_the_data_directory(
    studio, monkeypatch
):
    """The isolated environment lives beside the code, so relocating projects
    must not make it invisible (tests, smoke runs and deployments all move it).
    """

    _root, tmp_path = studio
    relocated = Path(tmp_path) / "elsewhere" / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(relocated))
    monkeypatch.delenv("DUBBING_AUDIOBOOKER_PYTHON", raising=False)

    candidates = adapter._interpreter_candidates()
    expected = adapter.REPO_ROOT / ".venvs" / "audiobooker" / "Scripts" / "python.exe"

    assert expected in candidates, (
        "the repository-local isolated environment must be searched"
    )
    assert candidates.index(expected) < candidates.index(
        Path(adapter.sys.executable)
    ), "the isolated interpreter must be preferred over the current interpreter"


def test_env_override_still_wins_for_the_interpreter(studio, monkeypatch):
    _root, _tmp = studio
    monkeypatch.setenv("DUBBING_AUDIOBOOKER_PYTHON", adapter.sys.executable)

    assert adapter.isolated_interpreter() == Path(adapter.sys.executable)
    assert adapter._interpreter_candidates()[0] == Path(adapter.sys.executable)


@pytest.mark.skipif(
    not adapter.availability().available,
    reason="the isolated audiobooker environment is not provisioned on this machine",
)
def test_analysis_source_reports_the_analyzer_that_did_the_work(studio):
    offline = adapter.analyze_text_offline(SAMPLE, title="Sample")
    assert adapter.analysis_source(offline) == adapter.OFFLINE_ANALYSIS_SOURCE

    real = adapter.BookAnalysis.from_dict(
        {
            "schema": adapter.ADAPTER_SCHEMA_VERSION,
            "title": "Sample",
            "author": "",
            "language": "",
            "chapters": [],
            "speakers": [],
            "pronunciations": {},
            "warnings": ["audiobooker compiled 1 chapter"],
        }
    )
    assert adapter.analysis_source(real) == adapter.AUDIOBOOKER_ANALYSIS_SOURCE


def test_real_audiobooker_analysis_satisfies_the_adapter_schema(studio):
    """Real integration evidence: the pinned library, not the offline splitter."""

    _root, tmp_path = studio
    analysis = adapter.analyze_source_file(_sample_file(tmp_path))

    assert analysis.source_path is not None, "audiobooker analyses keep their source"
    assert analysis.chapters, "a compiled book must yield chapters"
    assert analysis.utterance_count() > 0
    assert not any(
        adapter.OFFLINE_ANALYSIS_SOURCE in warning for warning in analysis.warnings
    ), "a real analysis must never be labelled as the offline splitter"
    assert adapter.analysis_source(analysis) == adapter.AUDIOBOOKER_ANALYSIS_SOURCE
    for chapter in analysis.chapters:
        for utterance in chapter.utterances:
            assert utterance.text.strip()
            assert utterance.attribution_source in adapter.ATTRIBUTION_SOURCES
            assert utterance.utterance_type


def test_import_creates_project_chapters_roles_and_utterances(studio):
    analysis = adapter.analyze_text_offline(SAMPLE, title="Sample")

    result = adapter.AudiobookerAdapter().reconcile_into_project(
        analysis, project_name="Sample"
    )
    project, segments = _reload(result.project_id)

    assert result.chapters_created == 2
    assert result.roles_created == 4
    assert result.utterances_created == analysis.utterance_count()
    assert {chapter.title for chapter in project.chapters} == {
        "Chapter 1 - Arrival",
        "Chapter 2 - Departure",
    }
    assert {role.name for role in project.roles} == {
        "Mara",
        "Tomas",
        "Ilse",
        "Narrator",
    }
    assert all(segment.role_id for segment in segments)
    narration = [segment for segment in segments if segment.text.startswith("The ledger")]
    assert narration, "plain narration must survive the import"
    narrator = next(role for role in project.roles if role.name == "Narrator")
    assert all(segment.role_id == narrator.id for segment in narration)
    assert any(segment.text == "You moved the chairs." for segment in segments)
    assert project.metadata["audiobooker_pinned_version"] == (
        adapter.PINNED_AUDIOBOOKER_VERSION
    )


def test_import_is_idempotent_and_keeps_utterance_identity(studio):
    analysis = adapter.analyze_text_offline(SAMPLE, title="Sample")
    first = adapter.AudiobookerAdapter().reconcile_into_project(
        analysis, project_name="Sample"
    )
    _project, before = _reload(first.project_id)
    ids_before = [segment.id for segment in before]

    second = adapter.AudiobookerAdapter().reconcile_into_project(
        analysis, project_id=first.project_id
    )
    _project, after = _reload(second.project_id)

    assert [segment.id for segment in after] == ids_before
    assert second.utterances_created == 0
    assert second.chapters_created == 0
    assert second.roles_created == 0


def test_import_keeps_every_utterance_when_the_text_repeats(studio):
    """Position is identity; identical text is two utterances, not one.

    Real books repeat text verbatim — scene-break markers (``******``) and
    short dialogue tags (``said Haoyu.``).  A text-based fallback lookup must
    never let a later occurrence reuse (and so silently swallow) an earlier
    occurrence's segment.
    """

    analysis = adapter.BookAnalysis.from_dict(
        {
            "schema": adapter.ADAPTER_SCHEMA_VERSION,
            "title": "Repeated text",
            "chapters": [
                {
                    "index": 0,
                    "title": "Chapter 1",
                    "utterances": [
                        {
                            "text": "******",
                            "speaker": None,
                            "utterance_type": "narration",
                            "chapter_index": 0,
                            "line_index": 0,
                            "attribution_source": "unknown",
                            "confidence": 0.0,
                        },
                        {
                            "text": "Alden waved.",
                            "speaker": None,
                            "utterance_type": "narration",
                            "chapter_index": 0,
                            "line_index": 1,
                            "attribution_source": "unknown",
                            "confidence": 0.0,
                        },
                        {
                            "text": "******",
                            "speaker": None,
                            "utterance_type": "narration",
                            "chapter_index": 0,
                            "line_index": 2,
                            "attribution_source": "unknown",
                            "confidence": 0.0,
                        },
                    ],
                }
            ],
        }
    )

    first = adapter.AudiobookerAdapter().reconcile_into_project(
        analysis, project_name="Repeated text"
    )
    _project, segments = _reload(first.project_id)

    assert first.utterances_created == analysis.utterance_count()
    assert first.utterances_updated == 0
    assert len(segments) == analysis.utterance_count()
    assert [segment.text for segment in segments].count("******") == 2
    assert sorted(segment.index for segment in segments) == [0, 1, 2]
    assert len({segment.id for segment in segments}) == 3

    ids_before = sorted(segment.id for segment in segments)
    second = adapter.AudiobookerAdapter().reconcile_into_project(
        analysis, project_id=first.project_id
    )
    _project, after = _reload(first.project_id)

    assert second.utterances_created == 0
    assert sorted(segment.id for segment in after) == ids_before, (
        "re-importing repeated text must not mint new segments"
    )


def test_import_never_overwrites_a_human_corrected_utterance(studio):
    analysis = adapter.analyze_text_offline(SAMPLE, title="Sample")
    imported = adapter.AudiobookerAdapter().reconcile_into_project(
        analysis, project_name="Sample"
    )
    project, segments = _reload(imported.project_id)
    target = segments[0]
    target.text = "You moved my chairs."
    target.attribution = Attribution(
        speaker_id="narrator", source="user", reviewed=True, confidence=1.0
    )
    save_segments(project.id, segments)

    storage = adapter.AudiobookerAdapter()
    result = storage.reconcile_into_project(analysis, project_id=project.id)
    _project, after = _reload(project.id)

    assert result.human_corrections_preserved >= 1
    corrected = next(segment for segment in after if segment.id == target.id)
    assert corrected.text == "You moved my chairs."
    assert corrected.attribution is not None
    assert corrected.attribution.human_authoritative


def test_import_records_machine_attribution_without_claiming_certainty(studio):
    analysis = adapter.analyze_text_offline(SAMPLE, title="Sample")
    imported = adapter.AudiobookerAdapter().reconcile_into_project(
        analysis, project_name="Sample"
    )
    _project, segments = _reload(imported.project_id)

    dialogue = next(
        segment for segment in segments if segment.text == "You moved the chairs."
    )
    assert dialogue.attribution is not None
    assert dialogue.attribution.source == "inline"
    assert dialogue.attribution.confidence == 1.0
    assert not dialogue.attribution.human_authoritative


def test_provider_attribution_labels_are_mapped_onto_the_studio_vocabulary():
    known = {
        "explicit_marker": "inline",
        "model": "nlp",
        "heuristic": "nlp",
        "human": "user",
        "nlp": "nlp",
        "turn": "turn",
    }
    for raw, expected in known.items():
        assert adapter.attribution_source(raw) == expected

    assert adapter.attribution_source(None) == "unknown"
    assert adapter.attribution_source("some_future_provider_label") == "unknown"
    assert adapter.attribution_source("human") != "unknown"


def test_scratch_directories_are_inside_the_project_root(studio):
    project = build_project(name="scratchy")
    root = safe_project_root(project.id)

    scratch = adapter.scratch_directories(root)

    assert list(scratch) == [root / "_scratch" / "audiobooker"]
    for path in scratch:
        assert str(path).startswith(str(root)), "scratch must stay inside the project"


def test_adapter_uses_an_injected_runner_and_disposes_its_scratch(studio, tmp_path):
    """The adapter must never import audiobooker into the studio process."""

    calls = {}

    def fake_runner(payload, interpreter):  # noqa: ANN001 - test double
        calls["payload"] = payload
        calls["interpreter"] = interpreter
        Path(payload["scratch_dir"], "analysis.json").write_text(
            "{}", encoding="utf-8"
        )
        return {
            "title": payload.get("title") or "",
            "chapters": [
                {
                    "index": 0,
                    "title": "Chapter 1",
                    "utterances": [
                        {
                            "text": "Isolated line.",
                            "speaker": "Narrator",
                            "utterance_type": "narration",
                            "chapter_index": 0,
                            "line_index": 0,
                            "attribution_source": "model",
                            "confidence": 0.5,
                        }
                    ],
                }
            ],
        }

    scratch_root = Path(tmp_path) / "scratch"
    result = adapter.AudiobookerAdapter(
        runner=fake_runner, scratch_root=scratch_root
    ).analyze_text("Narrator: hello.", title="Isolated")

    assert calls["payload"]["text"].startswith("Narrator")
    assert result.title == "Isolated"
    assert result.chapters[0].utterances[0].text == "Isolated line."
    assert result.chapters[0].utterances[0].confidence == 0.5
    leftovers = [item for item in scratch_root.rglob("*") if item.is_file()]
    assert not leftovers, "scratch data must be disposable"


def test_adapter_keeps_scratch_only_when_asked(studio, tmp_path):
    def fake_runner(payload, interpreter):  # noqa: ANN001 - test double
        Path(payload["scratch_dir"], "analysis.json").write_text(
            "{}", encoding="utf-8"
        )
        return {"title": "kept", "chapters": []}

    scratch_root = Path(tmp_path) / "scratch"
    adapter.AudiobookerAdapter(
        runner=fake_runner, scratch_root=scratch_root
    ).analyze_text("text", title="kept", keep_scratch=True)

    assert list(scratch_root.rglob("analysis.json")), "keep_scratch must preserve it"


def test_adapter_registers_no_audiobooker_payload_as_project_authority(studio):
    """``.audiobooker`` payloads are never project authority."""

    analysis = adapter.analyze_text_offline(SAMPLE, title="Sample")
    imported = adapter.AudiobookerAdapter().reconcile_into_project(
        analysis, project_name="Sample"
    )
    project, segments = _reload(imported.project_id)
    root = safe_project_root(project.id)

    assert project.metadata.get("source_title") == "Sample"
    assert segments and all(segment.text for segment in segments)
    assert "book.audiobooker.json" not in {path.name for path in root.iterdir()}


def test_import_into_a_preexisting_project_does_not_duplicate_segments(studio):
    project = build_project(name="existing", chapter_titles=("Chapter 1 - Arrival",))
    build_segments(project, ["The ledger stayed open on the desk for three days."])
    analysis = adapter.analyze_text_offline(SAMPLE, title="Sample")

    before = len(load_segments(project.id))
    result = adapter.AudiobookerAdapter().reconcile_into_project(
        analysis, project_id=project.id
    )
    after = load_segments(project.id)

    assert result.project_id == project.id
    assert len(after) >= before
    texts = [segment.text for segment in after]
    assert len(texts) == len(set(texts))
