from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from dubbing_service.app.core.tts_operations import (
    release_tts_operation,
    try_acquire_tts_operation,
)
from dubbing_service.app.main import create_app
from dubbing_service.app.models.subtitles import SubtitleFile
from dubbing_service.app.providers import subtitles as provider
from dubbing_service.app.routers import settings, subtitles as subtitles_router


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHISPER_MANIFESTS = {
    "whisper_large_v3.json": 3_090_835_702,
    "whisper_large_v3_turbo.json": 1_621_665_983,
    "whisper_distil_large_v3.json": 1_516_479_628,
    "whisperx_zh_alignment.json": 1_276_342_318,
}


def _materialize_required_files(root: Path, relative: str, names: tuple[str, ...]) -> Path:
    model_dir = root / relative
    model_dir.mkdir(parents=True)
    for name in names:
        target = model_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"test")
    return model_dir


def test_whisper_manifests_are_modelscope_only_and_exactly_sized() -> None:
    manifest_root = PROJECT_ROOT / "models" / "manifests"
    for filename, expected_bytes in WHISPER_MANIFESTS.items():
        payload = json.loads((manifest_root / filename).read_text(encoding="utf-8"))
        assert payload["runtime_policy"]["network_access"] == "forbidden"
        assert {source["provider"] for source in payload["sources"].values()} == {
            "modelscope"
        }
        assert sum(item["size"] for item in payload["files"]) == expected_bytes
        assert all(len(item["sha256"]) == 64 for item in payload["files"])
        assert {item["source_revision"] for item in payload["files"]} == {"master"}
        assert all(
            len(item["source_file_revision"]) == 40
            for item in payload["files"]
        )

    alignment = json.loads(
        (manifest_root / "whisperx_zh_alignment.json").read_text(encoding="utf-8")
    )
    alignment_paths = {item["source_path"] for item in alignment["files"]}
    assert "pytorch_model.bin" in alignment_paths
    assert "flax_model.msgpack" not in alignment_paths


def test_whisper_model_ids_resolve_only_to_project_local_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(provider, "PROJECT_ROOT", tmp_path)
    expected = _materialize_required_files(
        tmp_path,
        provider.WHISPER_MODEL_DIRECTORIES["large-v3-turbo"],
        provider.WHISPER_MODEL_REQUIRED_FILES,
    )

    resolved = provider.resolve_whisper_model_path("large-v3-turbo")

    assert resolved == expected.resolve()
    with pytest.raises(provider.SubtitleModelUnavailableError, match="不支持"):
        provider.resolve_whisper_model_path("Systran/faster-whisper-large-v3")


def test_provider_passes_a_local_path_to_faster_whisper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_model = tmp_path / "models" / "whisper" / "turbo"
    local_model.mkdir(parents=True)
    loaded = object()
    load_model = Mock(return_value=loaded)
    monkeypatch.setattr(provider, "resolve_whisper_model_path", lambda _model_id: local_model)
    monkeypatch.setattr(provider, "_detect_device_and_compute_type", lambda: ("cpu", "int8"))
    monkeypatch.setattr(provider, "_load_whisper_model", load_model)

    instance = provider.FastWhisperSubtitleProvider()

    assert instance.model_size == "large-v3-turbo"
    assert instance._model is loaded
    load_model.assert_called_once_with(str(local_model), "cpu", "int8")


def test_subtitle_caches_hold_one_model_and_clear_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert provider._load_whisper_model.cache_parameters()["maxsize"] == 1
    assert provider._load_whisperx_align_model.cache_parameters()["maxsize"] == 1
    assert provider._load_whisperx_audio.cache_parameters()["maxsize"] == 1

    asr_cache = Mock()
    align_cache = Mock()
    audio_cache = Mock()
    monkeypatch.setattr(provider, "_load_whisper_model", asr_cache)
    monkeypatch.setattr(provider, "_load_whisperx_align_model", align_cache)
    monkeypatch.setattr(provider, "_load_whisperx_audio", audio_cache)
    monkeypatch.setattr(provider.gc, "collect", Mock())

    provider.clear_whisper_model_cache()

    asr_cache.cache_clear.assert_called_once_with()
    align_cache.cache_clear.assert_called_once_with()
    audio_cache.cache_clear.assert_called_once_with()


def test_missing_chinese_alignment_model_falls_back_without_loading_remote(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    project_root = tmp_path / "p1"
    audio_path = project_root / "audio" / "final" / "book.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"test")
    subtitle = SubtitleFile(
        id="s1",
        project_id="p1",
        audio_rel_path="audio/final/book.wav",
        audio_filename="book.wav",
        language="zh",
        segments=[],
    )
    monkeypatch.setattr(provider, "safe_project_root", lambda _project_id: project_root)
    monkeypatch.setattr(provider, "_detect_device_and_compute_type", lambda: ("cpu", "int8"))
    monkeypatch.setattr(
        provider,
        "resolve_whisperx_zh_model_path",
        Mock(side_effect=provider.SubtitleModelUnavailableError("missing local align model")),
    )
    load_audio = Mock()
    load_align = Mock()
    monkeypatch.setattr(provider, "_load_whisperx_audio", load_audio)
    monkeypatch.setattr(provider, "_load_whisperx_align_model", load_align)

    result = provider._init_whisperx_alignment_resources(subtitle)

    assert result == ("cpu", None, None, None)
    load_audio.assert_not_called()
    load_align.assert_not_called()
    assert "deterministic timestamp fallback" in caplog.text


def test_subtitle_generation_route_is_sync_and_respects_gpu_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not inspect.iscoroutinefunction(
        subtitles_router.api_generate_subtitle_from_final
    )
    monkeypatch.setattr(
        subtitles_router.FastWhisperSubtitleProvider,
        "resolve_model_path",
        Mock(return_value=PROJECT_ROOT / "models" / "whisper" / "placeholder"),
    )
    lease = try_acquire_tts_operation(kind="test_busy")
    assert lease is not None
    try:
        response = TestClient(create_app()).post(
            "/api/v1/projects/p1/subtitles/generate-from-final",
            json={"model_size": "large-v3-turbo"},
        )
    finally:
        release_tts_operation(lease)

    assert response.status_code == 409


def test_distil_subtitle_model_requires_explicit_english_language() -> None:
    client = TestClient(create_app())

    rejected = client.post(
        "/api/v1/projects/p1/subtitles/generate-from-final",
        json={"model_size": "distil-large-v3", "language": "zh"},
    )

    assert rejected.status_code == 400
    assert "仅支持" in rejected.text


def test_gpu_handoff_unloads_only_a_running_managed_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_endpoint = Mock(return_value={"status": "unloaded"})
    monkeypatch.setattr(settings, "_call_tts_endpoint_json", call_endpoint)
    monkeypatch.setattr(settings, "_get_tts_runtime_state", lambda: (False, None, None))
    assert settings.unload_managed_tts_model_for_gpu_handoff() is None
    call_endpoint.assert_not_called()

    monkeypatch.setattr(
        settings,
        "_get_tts_runtime_state",
        lambda: (True, 1234, "indextts"),
    )
    assert settings.unload_managed_tts_model_for_gpu_handoff() == {
        "status": "unloaded"
    }
    call_endpoint.assert_called_once_with(
        "indextts",
        "/api/v1/tts/unload-model",
        method="POST",
        timeout=60,
    )
