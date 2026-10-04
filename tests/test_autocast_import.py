from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dubbing_service.app.audiobook import adapter, autocast_import
from dubbing_service.app.core.helpers import project_root
from dubbing_service.app.storage.projects import load_project
from dubbing_service.app.storage.segments import load_segments
from tests.audiobook_helpers import use_projects_root


SOURCE = b"Chapter 1\nMara entered the room.\n\"Who is there?\"\n\"It is me, Mara,\" Ivo replied.\n"


def analysis_payload():
    return {
        "title": "Fixture",
        "author": "Test author",
        "language": "en",
        # Nonspeaking listed cast members must not become required slots.
        "speakers": ["Mara", "Ivo", "Elian"],
        "chapters": [{
            "index": 0,
            "title": "Chapter 1",
            "utterances": [
                {"text": "Mara entered the room.", "speaker": "Narrator", "utterance_type": "narration",
                 "chapter_index": 0, "line_index": 0, "start_pos": 10, "end_pos": 33,
                 "attribution_source": "tag", "confidence": 0.99},
                {"text": "Who is there?", "speaker": "Unknown", "utterance_type": "dialogue",
                 "chapter_index": 0, "line_index": 1, "start_pos": 35, "end_pos": 48,
                 "attribution_source": "unknown", "confidence": 0.3},
                {"text": "It is me, Mara,", "speaker": "Mara", "utterance_type": "dialogue",
                 "chapter_index": 0, "line_index": 2, "start_pos": 51, "end_pos": 67,
                 "attribution_source": "tag", "confidence": 0.96},
                {"text": "Ivo replied.", "speaker": "Ivo", "utterance_type": "dialogue",
                 "chapter_index": 0, "line_index": 3, "start_pos": 67, "end_pos": 79,
                 "attribution_source": "turn", "confidence": 0.96},
            ],
        }],
    }


@pytest.fixture
def isolated_adapter(monkeypatch, tmp_path):
    use_projects_root(monkeypatch, tmp_path)
    calls = []

    def fake_runner(payload, interpreter):
        calls.append((Path(payload["source_path"]), interpreter))
        return analysis_payload()

    monkeypatch.setattr(adapter, "availability", lambda refresh=False: adapter.Availability(
        True, "", str(Path(__file__).resolve()), adapter.PINNED_AUDIOBOOKER_VERSION))
    original = adapter.AudiobookerAdapter

    def factory(*, interpreter=None):
        return original(interpreter=Path(interpreter) if interpreter else Path(__file__),
                        runner=fake_runner, scratch_root=tmp_path / "retained-scratch")

    monkeypatch.setattr(autocast_import.adapter, "AudiobookerAdapter", factory)
    return calls


def test_autocast_reconciles_unknown_dialogue_as_held_and_keeps_exact_source(isolated_adapter):
    result = autocast_import.import_autocast(SOURCE, "book.txt", "request-1", "Fixture")

    assert result["mode"] == "autocast"
    assert result["status"] == "blocked"
    assert result["reused"] is False
    assert len(isolated_adapter) == 1
    project = load_project(result["project_id"])
    receipt = project.metadata["autocast_import"]
    assert receipt["schema"] == "autocast-import-v1"
    assert project.character_cast.policy.audition_mode == "off"
    assert receipt["attribution_policy"]["enabled"] is True, "a new book handles lines that would stop it by itself"
    assert receipt["attribution_policy"]["set_by"] == "default"
    assert receipt["source_sha256"] == hashlib.sha256(SOURCE).hexdigest()
    saved = project_root(project.id) / receipt["source_file"]
    assert saved.suffix == ".txt"
    assert saved.read_bytes() == SOURCE
    revision = next(row for row in project.source_revisions if row.id == receipt["raw_source_revision_id"])
    assert (project_root(project.id) / revision.snapshot_file).read_bytes() == SOURCE

    segments = {row.id: row for row in load_segments(project.id)}
    held = set(receipt["unresolved_utterance_ids"])
    assert len(held) == 2
    assert all(segments[segment_id].role_id is None for segment_id in held)
    spoken_ids = {row.id for row in segments.values()
                  if row.source_anchor and row.source_anchor.occurrence_index == 2}
    assert set(receipt["speaking_utterance_ids"]) == spoken_ids
    assert {next(role.name for role in project.roles if role.id == role_id) for role_id in receipt["speaking_role_ids"]} == {"Mara"}
    assert "Elian" not in {next(role.name for role in project.roles if role.id == role_id) for role_id in receipt["speaking_role_ids"]}


def test_autocast_completed_request_is_idempotent_and_hash_bound(isolated_adapter):
    first = autocast_import.import_autocast(SOURCE, "book.txt", "request-2")
    second = autocast_import.import_autocast(SOURCE, "another-name.txt", "request-2")
    assert first["project_id"] == second["project_id"]
    assert second["reused"] is True
    assert len(isolated_adapter) == 1
    with pytest.raises(autocast_import.AutoCastImportError, match="different file or name"):
        autocast_import.import_autocast(SOURCE + b"different", "book.txt", "request-2")


def test_analyzing_receipt_retries_same_project_without_replacing_source(monkeypatch, tmp_path):
    use_projects_root(monkeypatch, tmp_path)
    attempts = 0

    def fake_runner(payload, interpreter):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise adapter.AdapterError("fixture interruption")
        return analysis_payload()

    monkeypatch.setattr(adapter, "availability", lambda refresh=False: adapter.Availability(
        True, "", str(Path(__file__).resolve()), adapter.PINNED_AUDIOBOOKER_VERSION))
    original = adapter.AudiobookerAdapter
    monkeypatch.setattr(autocast_import.adapter, "AudiobookerAdapter", lambda *, interpreter=None: original(
        interpreter=Path(interpreter) if interpreter else Path(__file__), runner=fake_runner,
        scratch_root=tmp_path / "retained-scratch"))
    with pytest.raises(adapter.AdapterError, match="fixture interruption"):
        autocast_import.import_autocast(SOURCE, "book.txt", "request-retry")
    project_ids = [path.parent.name for path in (tmp_path / "projects").glob("*/project.json")]
    assert len(project_ids) == 1
    project_id = project_ids[0]
    receipt = load_project(project_id).metadata["autocast_import"]
    retained = project_root(project_id) / receipt["source_file"]
    before = retained.read_bytes()
    result = autocast_import.import_autocast(SOURCE, "book.txt", "request-retry")
    assert result["project_id"] == project_id
    assert result["reused"] is True
    assert attempts == 2
    assert retained.read_bytes() == before == SOURCE
