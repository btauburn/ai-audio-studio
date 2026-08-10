from __future__ import annotations

import hashlib
import io
import json
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.main import create_app
from dubbing_service.app.models.project import Segment, SegmentEditRegion, SegmentEditState
from dubbing_service.app.services.takes import (
    create_take,
    ensure_take_state,
    get_slot_take,
    prune_archived_takes,
    set_active_slot,
    update_active_edit_state,
)
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audio_test_utils import make_wav_bytes


def test_legacy_active_b_migrates_without_losing_ab_identity() -> None:
    segment = Segment(
        id="legacy01",
        text="正文",
        active_version="b",
        audio_file="audio/segments/b.wav",
        alt_audio_file="audio/segments/a.wav",
        emotion_text="B 情绪",
        alt_emotion_text="A 情绪",
        tts_backend="qwen3tts",
        alt_tts_backend="indextts",
    )

    ensure_take_state(segment)

    assert get_slot_take(segment, "a").audio_file.endswith("a.wav")
    assert get_slot_take(segment, "b").audio_file.endswith("b.wav")
    assert segment.take_slots["a"].emotion_text == "A 情绪"
    assert segment.take_slots["b"].emotion_text == "B 情绪"
    a_id = segment.take_slots["a"].take_id
    b_id = segment.take_slots["b"].take_id

    set_active_slot(segment, "a")
    assert segment.audio_file.endswith("a.wav")
    assert segment.alt_audio_file.endswith("b.wav")
    set_active_slot(segment, "b")
    assert segment.take_slots["a"].take_id == a_id
    assert segment.take_slots["b"].take_id == b_id


def test_regeneration_archives_old_take_and_keeps_edits_per_slot() -> None:
    segment = Segment(id="take0001", text="测试")
    ensure_take_state(segment)
    first = create_take(
        segment,
        slot="a",
        take_id="takea001",
        audio_file="audio/segments/takea001.wav",
    )
    update_active_edit_state(
        segment,
        SegmentEditState(
            speed=1.1,
            regions=[SegmentEditRegion(id="a-cut", start=0.1, end=0.2)],
        ),
        speed=1.1,
    )

    set_active_slot(segment, "b")
    segment.emotion_text = "B 版本"
    second = create_take(
        segment,
        slot="b",
        take_id="takeb001",
        audio_file="audio/segments/takeb001.wav",
    )
    update_active_edit_state(
        segment,
        SegmentEditState(speed=0.9),
        speed=0.9,
    )

    set_active_slot(segment, "a")
    assert segment.edit_state.regions[0].id == "a-cut"
    assert segment.playback_speed == 1.1
    replacement = create_take(
        segment,
        slot="a",
        take_id="takea002",
        audio_file="audio/segments/takea002.wav",
    )

    assert first.archived_at is not None
    assert second.archived_at is None
    assert replacement.archived_at is None
    assert segment.take_slots["a"].take_id == replacement.id
    assert segment.take_slots["b"].take_id == second.id
    assert segment.selected_take_id == replacement.id
    set_active_slot(segment, "b")
    assert segment.playback_speed == 0.9


def test_archived_take_retention_protects_pinned_and_current_takes() -> None:
    segment = Segment(id="retain01", text="保留策略")
    ensure_take_state(segment)
    created = []
    for index in range(25):
        created.append(
            create_take(
                segment,
                slot="a",
                take_id=f"retain{index:02d}",
                audio_file=f"audio/segments/retain{index:02d}.wav",
            )
        )
    now = datetime.now(timezone.utc)
    for index, take in enumerate(created[:-1]):
        take.archived_at = now - timedelta(days=index + 1)
    created[0].pinned = True
    created[1].archived_at = now - timedelta(days=120)

    removed = prune_archived_takes(segment, now=now, max_archived=20, max_age_days=90)

    remaining_ids = {take.id for take in segment.takes}
    assert created[0].id in remaining_ids
    assert created[1].id not in remaining_ids
    assert created[-1].id in remaining_ids
    assert segment.take_slots["a"].take_id == created[-1].id
    assert all(take.id not in remaining_ids for take in removed)
    unpinned_archived = [
        take
        for take in segment.takes
        if take.archived_at is not None and not take.pinned
    ]
    assert len(unpinned_archived) <= 20


def test_edit_api_saves_recipe_without_modifying_source_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    projects_root = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects_root))
    clear_config_caches()

    with TestClient(create_app()) as client:
        created = client.post("/api/v1/projects", json={"name": "take-edit", "description": ""})
        assert created.status_code == 200
        project_id = created.json()["id"]
        imported = client.post(
            f"/api/v1/projects/{project_id}/segments/import-json",
            json={
                "json_text": json.dumps(
                    [{"speaker_name": "旁白", "text": "测试文本", "emotion_description": "平静"}],
                    ensure_ascii=False,
                )
            },
        )
        assert imported.status_code == 200
        segment_id = imported.json()["segments"][0]["id"]

        audio_path = projects_root / project_id / "audio" / "segments" / f"{segment_id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(make_wav_bytes(sample_value=300, frames=3200))

        segment = client.get(f"/api/v1/projects/{project_id}/segments").json()[0]
        segment["status"] = "done"
        segment["audio_file"] = f"audio/segments/{segment_id}.wav"
        restored = client.put(
            f"/api/v1/projects/{project_id}/segments/restore",
            json={"segments": [segment]},
        )
        assert restored.status_code == 200
        before = hashlib.sha256(audio_path.read_bytes()).hexdigest()

        edited = client.post(
            f"/api/v1/projects/{project_id}/segments/{segment_id}/edit-audio",
            json={
                "speed": 1.1,
                "volume": 0.8,
                "delete_regions": [{"start": 0.02, "end": 0.04}],
                "silence_intervals": [{"at": 0.05, "duration": 0.1}],
            },
        )
        assert edited.status_code == 200, edited.text
        data = edited.json()
        after = hashlib.sha256(audio_path.read_bytes()).hexdigest()

        assert after == before
        assert data["audio_file"] == f"audio/segments/{segment_id}.wav"
        assert data["edit_state"]["speed"] == 1.1
        assert data["edit_state"]["volume"] == 0.8
        assert data["edit_state"]["regions"]
        assert data["edit_state"]["silences"]
        active_take_id = data["take_slots"]["a"]["take_id"]
        active_take = next(item for item in data["takes"] if item["id"] == active_take_id)
        assert active_take["edit_state"] == data["edit_state"]

        exported = client.post(f"/api/v1/projects/{project_id}/export/full-with-edits")
        assert exported.status_code == 200, exported.text
        output_path = projects_root / project_id / exported.json()["output_file"]
        with wave.open(io.BytesIO(output_path.read_bytes()), "rb") as rendered:
            rendered_seconds = rendered.getnframes() / rendered.getframerate()
        # 0.20s source - 0.02s cut + 0.10s silence, then 1.1x speed.
        assert 0.23 <= rendered_seconds <= 0.27

        rendered_take = client.post(
            f"/api/v1/projects/{project_id}/segments/{segment_id}/render-as-new-take"
        )
        assert rendered_take.status_code == 200, rendered_take.text
        rendered_data = rendered_take.json()
        assert audio_path.is_file()
        assert rendered_data["audio_file"] != f"audio/segments/{segment_id}.wav"
        new_audio = projects_root / project_id / rendered_data["audio_file"]
        assert new_audio.is_file()
        new_take_id = rendered_data["take_slots"]["a"]["take_id"]
        new_take = next(item for item in rendered_data["takes"] if item["id"] == new_take_id)
        old_take = next(item for item in rendered_data["takes"] if item["id"] == active_take_id)
        assert new_take["source_take_id"] == active_take_id
        assert new_take["edit_state"] is None
        assert old_take["archived_at"] is not None

    clear_config_caches()


def test_project_cleanup_keeps_archived_take_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    projects_root = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects_root))
    clear_config_caches()

    with TestClient(create_app()) as client:
        project_id = client.post(
            "/api/v1/projects",
            json={"name": "take-cleanup", "description": ""},
        ).json()["id"]
        imported = client.post(
            f"/api/v1/projects/{project_id}/segments/import-json",
            json={
                "json_text": json.dumps(
                    [{"speaker_name": "旁白", "text": "清理测试", "emotion_description": "平静"}],
                    ensure_ascii=False,
                )
            },
        )
        segment_id = imported.json()["segments"][0]["id"]
        audio_dir = projects_root / project_id / "audio" / "segments"
        audio_dir.mkdir(parents=True, exist_ok=True)
        old_path = audio_dir / "old-take.wav"
        current_path = audio_dir / "current-take.wav"
        orphan_path = audio_dir / "orphan.wav"
        for path in (old_path, current_path, orphan_path):
            path.write_bytes(make_wav_bytes())

        segments = load_segments(project_id)
        segment = next(item for item in segments if item.id == segment_id)
        create_take(
            segment,
            slot="a",
            take_id="oldtake01",
            audio_file="audio/segments/old-take.wav",
        )
        create_take(
            segment,
            slot="a",
            take_id="newtake01",
            audio_file="audio/segments/current-take.wav",
        )
        save_segments(project_id, segments)

        cleaned = client.post("/api/v1/projects/cleanup-all")
        assert cleaned.status_code == 200
        assert old_path.is_file()
        assert current_path.is_file()
        assert not orphan_path.exists()

    clear_config_caches()
