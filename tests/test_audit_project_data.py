from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_project_data import audit_projects


def test_audit_accepts_an_external_projects_root(tmp_path: Path) -> None:
    project = tmp_path / "abc12345"
    audio = project / "audio" / "segments" / "seg.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"RIFF-test")
    (project / "project.json").write_text(
        json.dumps({
            "id": "abc12345",
            "name": "restore drill",
            "roles": [{"id": "role0001", "name": "narrator"}],
        }),
        encoding="utf-8",
    )
    (project / "segments.json").write_text(
        json.dumps([{
            "id": "seg00001",
            "role_id": "role0001",
            "status": "done",
            "audio_file": "audio/segments/seg.wav",
        }]),
        encoding="utf-8",
    )

    result = audit_projects(tmp_path)

    assert result["counts"] == {
        "projects": 1,
        "roles": 1,
        "segments": 1,
        "done_segments": 1,
    }
    assert result["referenced_audio"] == 1
    assert result["errors"] == []


def test_audit_reports_a_missing_root(tmp_path: Path) -> None:
    result = audit_projects(tmp_path / "missing")

    assert result["errors"] == ["projects root does not exist or is not a directory"]
