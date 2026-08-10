from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dubbing_service.app.core.config import clear_config_caches, safe_project_root
from dubbing_service.app.main import create_app
from dubbing_service.app.models.project import Segment
from dubbing_service.app.routers import segments as segments_router
from dubbing_service.app.routers import synthesis as synthesis_router


@pytest.fixture
def isolated_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(tmp_path / "projects"))
    clear_config_caches()
    try:
        with TestClient(create_app()) as client:
            yield client
    finally:
        clear_config_caches()


def _create_project_role_and_segment(client: TestClient) -> tuple[str, str, dict]:
    project_response = client.post("/api/v1/projects", json={"name": "path-safety"})
    assert project_response.status_code == 200
    project_id = project_response.json()["id"]

    role_response = client.post(
        f"/api/v1/projects/{project_id}/roles",
        json={"name": "narrator", "role_type": "narrator"},
    )
    assert role_response.status_code == 200
    role_id = role_response.json()["id"]

    segment = Segment(id="seg00001", role_id=role_id, index=0, text="安全边界测试")
    return project_id, role_id, segment.model_dump(mode="json")


@pytest.mark.parametrize(
    "project_id",
    [
        r"alias\..\target",
        "..",
        ".",
        "with/slash",
        "with space",
    ],
)
def test_safe_project_root_rejects_non_simple_component_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    project_id: str,
) -> None:
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(tmp_path / "projects"))
    clear_config_caches()
    try:
        with pytest.raises(ValueError, match="project_id"):
            safe_project_root(project_id)
    finally:
        clear_config_caches()


def test_restore_rejects_non_simple_segment_id(isolated_client: TestClient) -> None:
    project_id, _role_id, segment = _create_project_role_and_segment(isolated_client)
    segment["id"] = r"..\..\..\..\service\escaped"

    response = isolated_client.put(
        f"/api/v1/projects/{project_id}/segments/restore",
        json={"segments": [segment]},
    )

    assert response.status_code == 400
    assert "id" in str(response.json()).lower()


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    [
        ("duplicate_id", "重复"),
        ("duplicate_index", "index"),
        ("non_contiguous_index", "index"),
        ("dangling_role", "角色"),
    ],
)
def test_restore_rejects_invalid_segment_identity_graph(
    isolated_client: TestClient,
    mutation: str,
    expected_fragment: str,
) -> None:
    project_id, role_id, segment = _create_project_role_and_segment(isolated_client)
    second = Segment(
        id="seg00002",
        role_id=role_id,
        index=1,
        text="第二段",
    ).model_dump(mode="json")

    payload = [segment, second]
    if mutation == "duplicate_id":
        second["id"] = segment["id"]
    elif mutation == "duplicate_index":
        second["index"] = segment["index"]
    elif mutation == "non_contiguous_index":
        second["index"] = 2
    elif mutation == "dangling_role":
        second["role_id"] = "missing-role"

    response = isolated_client.put(
        f"/api/v1/projects/{project_id}/segments/restore",
        json={"segments": payload},
    )

    assert response.status_code == 400
    assert expected_fragment.lower() in str(response.json()).lower()


@pytest.mark.parametrize(
    "field_value",
    [
        "project.json",
        "roles/prompts/narrator.wav",
        "audio/segments/../../project.json",
        "audio/segments/not-audio.txt",
        r"E:\outside.wav",
    ],
)
def test_restore_restricts_segment_audio_paths_to_audio_segments(
    isolated_client: TestClient,
    field_value: str,
) -> None:
    project_id, _role_id, segment = _create_project_role_and_segment(isolated_client)
    segment["audio_file"] = field_value

    response = isolated_client.put(
        f"/api/v1/projects/{project_id}/segments/restore",
        json={"segments": [segment]},
    )

    assert response.status_code == 400
    assert "audio_file" in str(response.json())


def test_restore_clears_missing_primary_audio_and_resets_done_status(
    isolated_client: TestClient,
) -> None:
    project_id, _role_id, segment = _create_project_role_and_segment(isolated_client)
    segment["status"] = "done"
    segment["audio_file"] = "audio/segments/missing.wav"

    response = isolated_client.put(
        f"/api/v1/projects/{project_id}/segments/restore",
        json={"segments": [segment]},
    )

    assert response.status_code == 200
    restored = response.json()[0]
    assert restored["audio_file"] is None
    assert restored["status"] == "pending"


def test_restore_keeps_existing_primary_but_clears_missing_alt_audio(
    isolated_client: TestClient,
) -> None:
    project_id, _role_id, segment = _create_project_role_and_segment(isolated_client)
    project_root = safe_project_root(project_id)
    primary = project_root / "audio" / "segments" / "seg00001.wav"
    primary.parent.mkdir(parents=True, exist_ok=True)
    primary.write_bytes(b"test-audio-placeholder")

    segment["status"] = "done"
    segment["audio_file"] = "audio/segments/seg00001.wav"
    segment["alt_audio_file"] = "audio/segments/missing_alt.wav"

    response = isolated_client.put(
        f"/api/v1/projects/{project_id}/segments/restore",
        json={"segments": [segment]},
    )

    assert response.status_code == 200
    restored = response.json()[0]
    assert restored["audio_file"].replace("\\", "/") == "audio/segments/seg00001.wav"
    assert restored["alt_audio_file"] is None
    assert restored["status"] == "done"


def test_segment_audio_gc_cannot_delete_project_metadata(
    isolated_client: TestClient,
) -> None:
    project_id, _role_id, _segment = _create_project_role_and_segment(isolated_client)
    project_meta = safe_project_root(project_id) / "project.json"
    assert project_meta.is_file()

    segments_router._delete_segment_audio_file(project_id, "project.json")
    synthesis_router._unlink_project_relative_file(project_meta.parent, "project.json")

    assert project_meta.is_file()


def test_synthesis_rejects_unsafe_segment_id_before_copy(
    isolated_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id, role_id, _segment = _create_project_role_and_segment(isolated_client)
    unsafe = Segment(
        id=r"..\..\..\..\service\escaped",
        role_id=role_id,
        index=0,
        text="不得写出项目目录",
    )

    def _unexpected_copy(*_args, **_kwargs):
        raise AssertionError("unsafe segment ID reached shutil.copy2")

    monkeypatch.setattr(synthesis_router.shutil, "copy2", _unexpected_copy)

    with pytest.raises(ValueError, match="segment_id"):
        synthesis_router._copy_audio_to_project(
            project_id,
            unsafe,
            "outputs/nonexistent.wav",
        )
