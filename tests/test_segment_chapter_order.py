"""Regression tests for chapter-aware segment ordering.

``Segment.index`` counts lines inside its own chapter and restarts at 0 in every
chapter, so any ordering that uses ``index`` alone interleaves the chapters of a
multi-chapter project.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.main import create_app
from dubbing_service.app.models.project import Chapter, Segment
from dubbing_service.app.storage.projects import (
    create_project,
    load_project,
    save_project,
)
from dubbing_service.app.storage.segments import (
    load_segments,
    order_segments,
    save_segments,
)
from tests.audio_test_utils import make_wav_bytes


def _ids(segments) -> list[str]:
    return [segment.id for segment in segments]


def test_order_segments_uses_chapter_position_then_in_chapter_index() -> None:
    chapters = [
        Chapter(id="chapter-b", title="Chapter B", index=1),
        Chapter(id="chapter-a", title="Chapter A", index=0),
    ]
    segments = [
        Segment(id="segb0001", chapter_id="chapter-b", index=0, text="b0"),
        Segment(id="sega0001", chapter_id="chapter-a", index=0, text="a0"),
        Segment(id="segb0002", chapter_id="chapter-b", index=1, text="b1"),
        Segment(id="sega0002", chapter_id="chapter-a", index=1, text="a1"),
    ]

    assert _ids(order_segments(segments, chapters)) == [
        "sega0001",
        "sega0002",
        "segb0001",
        "segb0002",
    ]


def test_load_segments_keeps_multi_chapter_projects_in_reading_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(tmp_path / "projects"))
    clear_config_caches()

    project = create_project("chapter-order")
    project.chapters = [
        Chapter(id="chapter-a", title="Chapter A", index=0),
        Chapter(id="chapter-b", title="Chapter B", index=1),
    ]
    save_project(project)

    save_segments(
        project.id,
        [
            Segment(id="segb0001", chapter_id="chapter-b", index=0, text="b0"),
            Segment(id="sega0001", chapter_id="chapter-a", index=0, text="a0"),
            Segment(id="segb0002", chapter_id="chapter-b", index=1, text="b1"),
            Segment(id="sega0002", chapter_id="chapter-a", index=1, text="a1"),
        ],
    )

    assert _ids(load_segments(project.id)) == [
        "sega0001",
        "sega0002",
        "segb0001",
        "segb0002",
    ]

    # Chapter order comes from the project, not from chapter id ordering.
    reordered = load_project(project.id)
    reordered.chapters = [
        Chapter(id="chapter-b", title="Chapter B", index=0),
        Chapter(id="chapter-a", title="Chapter A", index=1),
    ]
    save_project(reordered)

    assert _ids(load_segments(project.id)) == [
        "segb0001",
        "segb0002",
        "sega0001",
        "sega0002",
    ]


def test_export_full_concatenates_segments_in_chapter_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    projects_root = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects_root))
    clear_config_caches()

    with TestClient(create_app()) as client:
        created = client.post(
            "/api/v1/projects",
            json={"name": "chapter-order-export", "description": ""},
        )
        assert created.status_code == 200
        project_id = created.json()["id"]

        imported = client.post(
            f"/api/v1/projects/{project_id}/segments/import-json",
            json={
                "json_text": json.dumps(
                    [
                        {"speaker_name": "旁白", "text": f"line {number}", "emotion_description": "平静"}
                        for number in range(4)
                    ],
                    ensure_ascii=False,
                )
            },
        )
        assert imported.status_code == 200
        segment_ids = [item["id"] for item in imported.json()["segments"]]
        assert len(segment_ids) == 4

        for offset, segment_id in enumerate(segment_ids):
            audio_path = projects_root / project_id / "audio" / "segments" / f"{segment_id}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(make_wav_bytes(sample_value=300 * (offset + 1), frames=3200))

        stored = client.get(f"/api/v1/projects/{project_id}/segments").json()
        assert len(stored) == 4
        for segment in stored:
            segment["status"] = "done"
            segment["audio_file"] = f"audio/segments/{segment['id']}.wav"
        # /segments/restore replaces the whole segment list, so restore every
        # utterance at once and shuffle it to prove the order is derived.
        restored = client.put(
            f"/api/v1/projects/{project_id}/segments/restore",
            json={"segments": [stored[index] for index in (2, 0, 3, 1)]},
        )
        assert restored.status_code == 200

        # The first two utterances belong to chapter A, the last two to chapter
        # B, and each in-chapter index restarts at 0.
        project = load_project(project_id)
        project.chapters = [
            Chapter(id="chapter-a", title="Chapter A", index=0),
            Chapter(id="chapter-b", title="Chapter B", index=1),
        ]
        save_project(project)

        segments = load_segments(project_id)
        layout = {segment.id: position for position, segment in enumerate(segments)}
        for segment in segments:
            position = layout[segment.id]
            segment.chapter_id = "chapter-a" if position < 2 else "chapter-b"
            segment.index = position % 2
        save_segments(project_id, segments)

        exported = client.post(f"/api/v1/projects/{project_id}/export/full")
        assert exported.status_code == 200, exported.text

        timeline = json.loads(
            (projects_root / project_id / "dialogue_timeline.json").read_text(encoding="utf-8")
        )
        assert [item["segment_id"] for item in timeline["items"]] == segment_ids

        starts = [item["start"] for item in timeline["items"]]
        assert starts == sorted(starts)

    clear_config_caches()
