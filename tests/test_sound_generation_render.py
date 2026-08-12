from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.core.helpers import sha256_file
from dubbing_service.app.main import create_app
from dubbing_service.app.models.production import (
    AudioAsset,
    AudioAssetOrigin,
    ProductionSettings,
    SoundDesign,
    SoundTrack,
    TimelineClip,
)
from dubbing_service.app.models.sound_generation import (
    GenerationProvenance,
    SemanticAnchor,
    SemanticPlacement,
)
from dubbing_service.app.services.production import (
    plan_ambience_crossfade_loop,
    render_four_track_master,
)
from tests.audio_test_utils import make_wav_bytes


def test_ambience_loop_plan_uses_one_second_overlap() -> None:
    plan = plan_ambience_crossfade_loop(10.0, 20.0)
    assert plan.crossfade_seconds == 1.0
    assert plan.repetitions == 3
    assert plan.repetitions * plan.source_duration - (
        (plan.repetitions - 1) * plan.crossfade_seconds
    ) >= plan.output_duration

    short = plan_ambience_crossfade_loop(0.5, 2.0)
    assert short.crossfade_seconds == pytest.approx(0.25)


def test_render_resolves_semantic_clip_and_builds_ambience_acrossfades(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    asset_path = root / "production" / "assets" / "ambience" / "rain.wav"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(make_wav_bytes(sample_value=200, frames=32000))
    dialogue_path = root / "audio" / "dialogue.wav"
    dialogue_path.parent.mkdir(parents=True)
    dialogue_path.write_bytes(make_wav_bytes(sample_value=400, frames=80000))
    output_path = root / "audio" / "masters" / "master.wav"

    asset = AudioAsset(
        id="rain",
        track=SoundTrack.AMBIENCE,
        name="Rain",
        audio_file="production/assets/ambience/rain.wav",
        audio_sha256=sha256_file(asset_path),
        duration_seconds=2.0,
        origin=AudioAssetOrigin.GENERATED,
        generation=GenerationProvenance(
            provider="test", model_id="model", prompt="steady rain", seed=1,
        ),
    )
    clip = TimelineClip(
        id="rain-bed",
        track=SoundTrack.AMBIENCE,
        asset_id="rain",
        duration=2.0,
        loop=True,
        semantic_placement=SemanticPlacement(
            start=SemanticAnchor(segment_id="seg-1"),
            end=SemanticAnchor(segment_id="seg-2", edge="end"),
        ),
    )
    settings = ProductionSettings(sound_design=SoundDesign(clips=[clip]))
    snapshot = render_four_track_master(
        dialogue_path=dialogue_path,
        output_path=output_path,
        root=root,
        settings=settings,
        assets=[asset],
        dialogue_duration=5.0,
        dialogue_timeline={"items": [
            {"segment_id": "seg-1", "start": 0.0, "end": 2.0},
            {"segment_id": "seg-2", "start": 2.0, "end": 5.0},
        ]},
    )

    assert snapshot["clips"][0]["rendered_duration"] == pytest.approx(5.0)
    assert snapshot["clips"][0]["semantic_resolved"] is True
    assert snapshot["clips"][0]["asset_origin"] == "generated"
    assert snapshot["clips"][0]["generation"]["seed"] == 1
    assert snapshot["clips"][0]["ambience_loop"] == {
        "repetitions": 4,
        "crossfade_seconds": 1.0,
    }
    with wave.open(str(output_path), "rb") as output:
        assert output.getframerate() == 48000
        assert output.getnframes() / output.getframerate() == pytest.approx(5.0, abs=0.02)


def test_multi_segment_full_with_edits_loads_dialogue_crossfade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            project_id = client.post(
                "/api/v1/projects", json={"name": "crossfade", "description": ""},
            ).json()["id"]
            imported = client.post(
                f"/api/v1/projects/{project_id}/segments/import-json",
                json={"json_text": json.dumps([
                    {"speaker_name": "旁白", "text": "第一句", "emotion_description": "平静"},
                    {"speaker_name": "旁白", "text": "第二句", "emotion_description": "平静"},
                ], ensure_ascii=False)},
            )
            segments = imported.json()["segments"]
            for segment in segments:
                path = projects / project_id / "audio" / "segments" / f"{segment['id']}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(make_wav_bytes(sample_value=300, frames=8000))
                segment.update({
                    "status": "done",
                    "audio_file": f"audio/segments/{segment['id']}.wav",
                })
            assert client.put(
                f"/api/v1/projects/{project_id}/segments/restore", json={"segments": segments},
            ).status_code == 200
            settings = client.put(
                f"/api/v1/projects/{project_id}/production/sound-design",
                json={"clips": [], "mix": {"dialogue_crossfade_seconds": 0.05}},
            )
            assert settings.status_code == 200, settings.text

            exported = client.post(f"/api/v1/projects/{project_id}/export/full-with-edits")
            assert exported.status_code == 200, exported.text
            output_path = projects / project_id / exported.json()["output_file"]
            with wave.open(io.BytesIO(output_path.read_bytes()), "rb") as output:
                assert output.getnframes() / output.getframerate() == pytest.approx(0.95, abs=0.03)
            timeline_path = projects / project_id / exported.json()["dialogue_timeline_file"]
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
            assert timeline["items"][1]["start"] == pytest.approx(0.45, abs=0.02)
    finally:
        clear_config_caches()
