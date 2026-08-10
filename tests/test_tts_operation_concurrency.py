from __future__ import annotations

import json
import threading
from collections.abc import Callable
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.core.tts_operations import (
    has_active_tts_operation,
    release_project_export,
    release_project_mutation,
    try_acquire_project_export,
    try_acquire_project_mutation,
    try_register_project_deletion,
    try_register_running_batch,
    unregister_project_deletion,
    unregister_running_batch,
)
from dubbing_service.app.main import create_app
from dubbing_service.app.models.enums import SegmentStatus
from dubbing_service.app.routers import projects as projects_router
from dubbing_service.app.routers import synthesis as synthesis_router
from dubbing_service.app.routers import ws_synthesis as ws_router
from dubbing_service.app.routers.synthesis import BatchSynthesisRequest
from dubbing_service.app.storage.projects import load_project
from dubbing_service.app.storage.segments import load_segments


def _create_synthesis_project(
    client: TestClient,
    name: str,
    texts: list[str],
) -> tuple[str, list[str]]:
    created = client.post(
        "/api/v1/projects",
        json={"name": name, "description": "concurrency regression"},
    )
    assert created.status_code == 200, created.text
    project_id = created.json()["id"]

    role = client.post(
        f"/api/v1/projects/{project_id}/roles",
        json={"name": "旁白", "role_type": "narrator"},
    )
    assert role.status_code == 200, role.text

    imported = client.post(
        f"/api/v1/projects/{project_id}/segments/import-json",
        json={
            "json_text": json.dumps(
                [
                    {
                        "speaker_name": "旁白",
                        "text": text,
                        "emotion_description": "平静",
                    }
                    for text in texts
                ],
                ensure_ascii=False,
            )
        },
    )
    assert imported.status_code == 200, imported.text
    return project_id, [item["id"] for item in imported.json()["segments"]]


@pytest.fixture
def isolated_projects(tmp_path, monkeypatch):
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(tmp_path / "projects"))
    clear_config_caches()
    try:
        yield
    finally:
        clear_config_caches()


def test_project_update_holds_lock_across_load_modify_save(isolated_projects) -> None:
    app = create_app()
    first_in_save = threading.Event()
    release_first_save = threading.Event()
    second_loaded = threading.Event()
    load_count = 0
    load_count_lock = threading.Lock()
    responses: dict[str, object] = {}

    with TestClient(app) as client:
        created = client.post("/api/v1/projects", json={"name": "original"})
        assert created.status_code == 200
        project_id = created.json()["id"]

        real_load = projects_router.load_project
        real_save = projects_router.save_project

        def observed_load(pid: str):
            nonlocal load_count
            with load_count_lock:
                load_count += 1
                if load_count >= 2:
                    second_loaded.set()
            return real_load(pid)

        def blocking_save(project):
            if project.name == "first":
                first_in_save.set()
                assert release_first_save.wait(timeout=5)
            return real_save(project)

        def update(label: str) -> None:
            responses[label] = client.put(
                f"/api/v1/projects/{project_id}",
                json={"name": label, "description": label},
            )

        with patch.object(projects_router, "load_project", side_effect=observed_load), patch.object(
            projects_router,
            "save_project",
            side_effect=blocking_save,
        ):
            first_thread = threading.Thread(target=update, args=("first",), daemon=True)
            second_thread = threading.Thread(target=update, args=("second",), daemon=True)
            first_thread.start()
            assert first_in_save.wait(timeout=3)
            second_thread.start()

            # The second request may enter FastAPI, but it must not reach load_project
            # until the first request has completed its save under the same lock.
            assert not second_loaded.wait(timeout=0.25)
            release_first_save.set()
            first_thread.join(timeout=5)
            second_thread.join(timeout=5)

        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        assert responses["first"].status_code == 200
        assert responses["second"].status_code == 200
        assert load_project(project_id).name == "second"


def test_rest_batch_is_global_and_blocks_ws_and_service_lifecycle(
    isolated_projects,
) -> None:
    app = create_app()
    entered_synthesis = threading.Event()
    release_synthesis = threading.Event()
    first_response: dict[str, object] = {}

    def blocking_synthesis(*args, **kwargs):
        entered_synthesis.set()
        assert release_synthesis.wait(timeout=5)
        return {"success": False, "error": "intentional test failure"}

    with TestClient(app) as client:
        first_project, first_segments = _create_synthesis_project(
            client,
            "first-project",
            ["第一段"],
        )
        second_project, second_segments = _create_synthesis_project(
            client,
            "second-project",
            ["第二段"],
        )

        def run_first_batch() -> None:
            first_response["response"] = client.post(
                f"/api/v1/projects/{first_project}/segments/synthesize",
                json={"segment_ids": first_segments},
            )

        with patch(
            "dubbing_service.app.routers.settings._load_raw_file_config",
            return_value={"tts_backend": "qwen3tts_cv"},
        ), patch.object(
            synthesis_router,
            "_synthesize_one",
            side_effect=blocking_synthesis,
        ):
            worker = threading.Thread(target=run_first_batch, daemon=True)
            worker.start()
            try:
                assert entered_synthesis.wait(timeout=3)

                duplicate = client.post(
                    f"/api/v1/projects/{second_project}/segments/synthesize",
                    json={"segment_ids": second_segments},
                )
                assert duplicate.status_code == 409

                with client.websocket_connect(
                    f"/api/v1/ws/synthesis/{second_project}"
                ) as websocket:
                    websocket.send_json(
                        {"action": "start", "segment_ids": second_segments}
                    )
                    ws_result = websocket.receive_json()
                assert ws_result["type"] == "error"
                assert "正在执行" in ws_result["message"]

                start = client.post(
                    "/api/v1/settings/tts-service/start",
                    json={"backend": "indextts"},
                )
                stop = client.post("/api/v1/settings/tts-service/stop")
                assert start.status_code == 409
                assert stop.status_code == 409

                delete_active = client.delete(f"/api/v1/projects/{first_project}")
                assert delete_active.status_code == 409
                assert "正在批量合成" in delete_active.text

                blocked_patch = client.patch(
                    f"/api/v1/projects/{first_project}/segments/{first_segments[0]}",
                    json={"text": "不能覆盖长合成中的快照"},
                )
                blocked_delete = client.delete(
                    f"/api/v1/projects/{first_project}/segments/{first_segments[0]}"
                )
                blocked_import = client.post(
                    f"/api/v1/projects/{first_project}/segments/import-json",
                    json={
                        "json_text": json.dumps(
                            [{"speaker_name": "旁白", "text": "不能导入", "emotion_description": "平静"}],
                            ensure_ascii=False,
                        )
                    },
                )
                blocked_export = client.post(
                    f"/api/v1/projects/{first_project}/export/full"
                )
                role_id = load_segments(first_project)[0].role_id
                blocked_role_voice = client.patch(
                    f"/api/v1/projects/{first_project}/roles/{role_id}/voice-profile",
                    json={"tts_backend": "qwen3tts"},
                )
                assert blocked_patch.status_code == 409
                assert blocked_delete.status_code == 409
                assert blocked_import.status_code == 409
                assert blocked_export.status_code == 409
                assert blocked_role_voice.status_code == 409

                other_project_edit = client.patch(
                    f"/api/v1/projects/{second_project}/segments/{second_segments[0]}",
                    json={"text": "不同项目仍可安全编辑"},
                )
                assert other_project_edit.status_code == 200
            finally:
                release_synthesis.set()
                worker.join(timeout=5)

        assert not worker.is_alive()
        assert first_response["response"].status_code == 200
        assert not has_active_tts_operation()


def test_rest_batch_persists_each_segment_before_advancing(
    isolated_projects,
) -> None:
    app = create_app()
    second_entered = threading.Event()
    release_second = threading.Event()
    response_holder: dict[str, object] = {}
    call_count = 0
    call_lock = threading.Lock()

    def staged_synthesis(*args, **kwargs):
        nonlocal call_count
        with call_lock:
            call_count += 1
            current_call = call_count
        if current_call == 2:
            second_entered.set()
            assert release_second.wait(timeout=5)
        return {"success": False, "error": f"failure-{current_call}"}

    with TestClient(app) as client:
        project_id, segment_ids = _create_synthesis_project(
            client,
            "incremental-persistence",
            ["第一段", "第二段"],
        )

        def run_batch() -> None:
            response_holder["response"] = client.post(
                f"/api/v1/projects/{project_id}/segments/synthesize",
                json={"segment_ids": segment_ids},
            )

        with patch(
            "dubbing_service.app.routers.settings._load_raw_file_config",
            return_value={"tts_backend": "qwen3tts_cv"},
        ), patch.object(
            synthesis_router,
            "_synthesize_one",
            side_effect=staged_synthesis,
        ):
            worker = threading.Thread(target=run_batch, daemon=True)
            worker.start()
            try:
                assert second_entered.wait(timeout=3)
                persisted = {item.id: item for item in load_segments(project_id)}
                assert persisted[segment_ids[0]].status == SegmentStatus.FAILED
                assert persisted[segment_ids[1]].status == SegmentStatus.GENERATING
            finally:
                release_second.set()
                worker.join(timeout=5)

        assert not worker.is_alive()
        response = response_holder["response"]
        assert response.status_code == 200
        assert [item["success"] for item in response.json()["results"]] == [False, False]
        assert not has_active_tts_operation()


@pytest.mark.parametrize(
    ("failed_stage", "synthesis_result", "expected_disk_status"),
    [
        ("GENERATING", {"success": True, "audio_file": "unused.wav"}, SegmentStatus.PENDING),
        ("DONE", {"success": True, "audio_file": "audio/segments/test.wav"}, SegmentStatus.GENERATING),
        ("FAILED", {"success": False, "error": "tts failed"}, SegmentStatus.GENERATING),
    ],
)
def test_websocket_persistence_failure_never_reports_success(
    isolated_projects,
    failed_stage: str,
    synthesis_result: dict[str, object],
    expected_disk_status: SegmentStatus,
) -> None:
    app = create_app()

    with TestClient(app) as client:
        project_id, segment_ids = _create_synthesis_project(
            client,
            f"ws-durability-{failed_stage.lower()}",
            ["持久化故障注入"],
        )
        segment_id = segment_ids[0]
        real_save: Callable = ws_router._save_segment_state
        synth_calls = 0

        def failing_save(pid, sid, segment_status, **kwargs):
            if segment_status.value.upper() == failed_stage:
                raise OSError(f"injected {failed_stage} persistence failure")
            return real_save(pid, sid, segment_status, **kwargs)

        def fake_synthesis(*args, **kwargs):
            nonlocal synth_calls
            synth_calls += 1
            return synthesis_result

        with patch(
            "dubbing_service.app.routers.settings._load_raw_file_config",
            return_value={"tts_backend": "qwen3tts_cv"},
        ), patch.object(
            ws_router,
            "_save_segment_state",
            side_effect=failing_save,
        ), patch.object(
            ws_router,
            "_synthesize_one",
            side_effect=fake_synthesis,
        ):
            with client.websocket_connect(
                f"/api/v1/ws/synthesis/{project_id}"
            ) as websocket:
                websocket.send_json({"action": "start", "segment_ids": segment_ids})
                batch_start = websocket.receive_json()
                segment_result = websocket.receive_json()
                batch_complete = websocket.receive_json()

        assert batch_start == {
            "type": "batch_start",
            "total": 1,
            "segment_ids": [segment_id],
        }
        assert segment_result["type"] == "segment_result"
        assert segment_result["success"] is False
        assert segment_result["audio_file"] is None
        assert failed_stage in segment_result["error"]
        assert segment_result["done"] == 1
        assert batch_complete == {
            "type": "batch_complete",
            "total": 1,
            "succeeded": 0,
            "failed": 1,
            "stopped": False,
        }
        if failed_stage == "GENERATING":
            assert synth_calls == 0
        else:
            assert synth_calls == 1
        assert load_segments(project_id)[0].status == expected_disk_status
        assert not has_active_tts_operation()


def test_rest_batch_releases_lease_when_setup_raises() -> None:
    with patch.object(
        synthesis_router,
        "_select_segments_for_synthesis",
        side_effect=RuntimeError("injected setup failure"),
    ):
        with pytest.raises(RuntimeError, match="injected setup failure"):
            synthesis_router.api_batch_synthesize_segments(
                "broken-project",
                BatchSynthesisRequest(),
            )

    probe_lease = try_register_running_batch("lease-probe")
    assert probe_lease is not None
    unregister_running_batch(probe_lease)
    assert not has_active_tts_operation()


def test_stale_batch_token_cannot_release_newer_owner() -> None:
    first = try_register_running_batch("same-project")
    assert first is not None
    unregister_running_batch(first)

    second = try_register_running_batch("same-project")
    assert second is not None
    unregister_running_batch(first)
    assert has_active_tts_operation()

    unregister_running_batch(second)
    assert not has_active_tts_operation()


def test_project_deletion_reservation_closes_batch_registration_race() -> None:
    deletion = try_register_project_deletion("deleting-project")
    assert deletion is not None
    try:
        assert try_register_running_batch("deleting-project") is None
        other_batch = try_register_running_batch("other-project")
        assert other_batch is not None
        unregister_running_batch(other_batch)
    finally:
        unregister_project_deletion(deletion)

    batch = try_register_running_batch("deleting-project")
    assert batch is not None
    unregister_running_batch(batch)
    assert not has_active_tts_operation()


def test_project_mutation_and_export_reservations_are_token_owned_and_scoped() -> None:
    mutation = try_acquire_project_mutation("reserved-project")
    assert mutation is not None
    try:
        assert try_register_running_batch("reserved-project") is None
        assert try_acquire_project_export("reserved-project") is None

        other_export = try_acquire_project_export("other-project")
        assert other_export is not None
        release_project_export(other_export)
    finally:
        release_project_mutation(mutation)

    export = try_acquire_project_export("reserved-project")
    assert export is not None
    stale_export = try_acquire_project_export("other-project")
    assert stale_export is not None
    release_project_export(stale_export)
    try:
        assert try_register_running_batch("reserved-project") is None
        assert try_acquire_project_mutation("reserved-project") is None
    finally:
        release_project_export(export)

    probe = try_acquire_project_mutation("reserved-project")
    assert probe is not None
    release_project_mutation(export)
    assert try_register_running_batch("reserved-project") is None
    release_project_mutation(probe)

    batch = try_register_running_batch("reserved-project")
    assert batch is not None
    unregister_running_batch(batch)
    assert not has_active_tts_operation()


def test_synthesize_alt_owns_gpu_and_project_segment_until_request_finishes(
    isolated_projects,
) -> None:
    app = create_app()
    entered = threading.Event()
    release = threading.Event()
    response_holder: dict[str, object] = {}

    def blocking_alt(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return {"success": False, "error": "intentional alt stop"}

    with TestClient(app) as client:
        project_id, segment_ids = _create_synthesis_project(
            client,
            "standalone-alt",
            ["B 版本并发保护"],
        )
        other_project, other_segments = _create_synthesis_project(
            client,
            "standalone-alt-other",
            ["其他项目"],
        )

        def run_alt() -> None:
            response_holder["response"] = client.post(
                f"/api/v1/projects/{project_id}/segments/{segment_ids[0]}/synthesize-alt",
                json={"tts_backend": "qwen3tts_cv"},
            )

        with patch(
            "dubbing_service.app.routers.settings._load_raw_file_config",
            return_value={"tts_backend": "qwen3tts_cv"},
        ), patch.object(
            synthesis_router,
            "_synthesize_one",
            side_effect=blocking_alt,
        ):
            worker = threading.Thread(target=run_alt, daemon=True)
            worker.start()
            try:
                assert entered.wait(timeout=3)
                same_project_edit = client.patch(
                    f"/api/v1/projects/{project_id}/segments/{segment_ids[0]}",
                    json={"text": "必须被拒绝"},
                )
                same_project_export = client.post(
                    f"/api/v1/projects/{project_id}/export/full"
                )
                other_project_edit = client.patch(
                    f"/api/v1/projects/{other_project}/segments/{other_segments[0]}",
                    json={"text": "其他项目可编辑"},
                )
                competing_batch = client.post(
                    f"/api/v1/projects/{other_project}/segments/synthesize",
                    json={"segment_ids": other_segments},
                )
                assert same_project_edit.status_code == 409
                assert same_project_export.status_code == 409
                assert other_project_edit.status_code == 200
                assert competing_batch.status_code == 409
            finally:
                release.set()
                worker.join(timeout=5)

    assert not worker.is_alive()
    assert response_holder["response"].status_code == 502
    assert not has_active_tts_operation()
    probe = try_register_running_batch("post-alt-probe")
    assert probe is not None
    unregister_running_batch(probe)


def test_service_start_failure_releases_lifecycle_lease() -> None:
    app = create_app()
    with patch(
        "dubbing_service.app.routers.settings.build_backend_command",
        side_effect=FileNotFoundError("missing isolated environment"),
    ):
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/settings/tts-service/start",
                json={"backend": "indextts"},
            )

    assert response.status_code == 503
    probe_lease = try_register_running_batch("post-start-failure")
    assert probe_lease is not None
    unregister_running_batch(probe_lease)
    assert not has_active_tts_operation()
