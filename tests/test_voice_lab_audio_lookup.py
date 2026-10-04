"""Audio lookups are remembered against the project file, and never outlive a save.

A player issues several ranged requests per audition. Resolving each one used to parse
the whole (multi-megabyte) project file, so the first byte of audio waited behind work
that does not change between requests.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from dubbing_service.app.audiobook import voice_lab as lab


@pytest.fixture
def lookup(tmp_path, monkeypatch):
    meta = tmp_path / "project.json"
    meta.write_text("{}", encoding="utf-8")
    loads = []
    project = SimpleNamespace(voice_lab=SimpleNamespace(excerpt=None))
    monkeypatch.setattr(lab, "_project_meta_path", lambda project_id: meta)
    monkeypatch.setattr(lab, "load_project", lambda project_id: loads.append(project_id) or project)
    monkeypatch.setattr(lab, "comparison_records", lambda current: {})
    monkeypatch.setattr(lab, "_path", lambda project_id, relative: tmp_path / relative)
    monkeypatch.setattr(lab, "_record", lambda current, key: {"audio_file": f"{key}.wav",
                                                              "audio_sha256": "a" * 64, "key": key})
    lab._AUDIO_LOOKUP.clear()
    yield SimpleNamespace(meta=meta, loads=loads, tmp=tmp_path)
    lab._AUDIO_LOOKUP.clear()


def test_repeated_range_requests_do_not_reload_the_project(lookup):
    first = lab.audio_record("project-1", "v_1")
    for _ in range(5):
        assert lab.audio_record("project-1", "v_1") == first
    assert lookup.loads == ["project-1"], "one project read serves every ranged request"
    assert first[0] == lookup.tmp / "v_1.wav" and first[1]["audio_sha256"] == "a" * 64


def test_a_save_invalidates_the_remembered_lookup(lookup):
    lab.audio_record("project-1", "v_1")
    lookup.meta.write_text('{"saved": "a rating was recorded"}', encoding="utf-8")
    lab.audio_record("project-1", "v_1")
    assert lookup.loads == ["project-1", "project-1"], "a changed project file is read again"


def test_each_audition_and_project_is_looked_up_separately(lookup):
    lab.audio_record("project-1", "v_1")
    lab.audio_record("project-1", "v_2")
    lab.audio_record("project-2", "v_1")
    assert lookup.loads == ["project-1", "project-1", "project-2"]


def test_a_missing_audition_is_never_remembered(lookup, monkeypatch):
    def missing(current, key):
        raise lab.VoiceLabError("Voice version no longer available; refresh Voice Lab")

    monkeypatch.setattr(lab, "_record", missing)
    for _ in range(2):
        with pytest.raises(lab.VoiceLabError):
            lab.audio_record("project-1", "v_gone")
    assert lookup.loads == ["project-1", "project-1"], "failures are re-checked, not cached"
    assert ("project-1", "v_gone") not in lab._AUDIO_LOOKUP


def test_a_project_without_a_readable_file_is_not_cached(lookup):
    lookup.meta.unlink()
    lab.audio_record("project-1", "v_1")
    lab.audio_record("project-1", "v_1")
    assert lookup.loads == ["project-1", "project-1"]
