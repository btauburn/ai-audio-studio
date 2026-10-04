import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from dubbing_service.app.main import create_app
from dubbing_service.app.storage import projects as project_storage
from dubbing_service.app.storage.projects import ProjectStorageError, list_project_summaries, list_projects


class TestDubbingProjectsApi(unittest.TestCase):
    def test_list_projects_accepts_legacy_and_utc_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                ("legacy01", "Legacy", "2026-01-01T08:00:00"),
                ("current1", "Current", "2026-01-01T09:00:00Z"),
            ]
            for project_id, name, created_at in rows:
                project_dir = root / project_id
                project_dir.mkdir()
                (project_dir / "project.json").write_text(
                    json.dumps(
                        {
                            "id": project_id,
                            "name": name,
                            "description": None,
                            "created_at": created_at,
                            "updated_at": created_at,
                            "chapters": [],
                            "roles": [],
                        }
                    ),
                    encoding="utf-8",
                )

            with patch(
                "dubbing_service.app.storage.projects._ensure_projects_root",
                return_value=root,
            ):
                projects = list_projects()

        self.assertEqual([project.id for project in projects], ["legacy01", "current1"])

    def test_create_project_storage_error_returns_500(self) -> None:
        app = create_app()

        with patch(
            "dubbing_service.app.routers.projects.create_project",
            side_effect=ProjectStorageError("create failed"),
        ):
            with TestClient(app) as client:
                resp = client.post(
                    "/api/v1/projects",
                    json={"name": "demo", "description": "x"},
                )

        self.assertEqual(resp.status_code, 500)
        self.assertIn("create failed", resp.text)


def _write_project(root: Path, project_id: str, name: str, created_at: str, *, extra: dict | None = None) -> Path:
    project_dir = root / project_id
    project_dir.mkdir(exist_ok=True)
    path = project_dir / "project.json"
    path.write_text(json.dumps({"id": project_id, "name": name, "description": None, "created_at": created_at,
                                "updated_at": created_at, "chapters": [], "roles": [], **(extra or {})}), encoding="utf-8")
    return path


def test_project_summaries_list_every_readable_project_oldest_first_and_reread_only_what_changed(tmp_path):
    _write_project(tmp_path, "legacy01", "Legacy", "2026-01-01T08:00:00")
    changing = _write_project(tmp_path, "current1", "Current", "2026-01-01T09:00:00Z")
    (tmp_path / "broken01").mkdir()
    (tmp_path / "broken01" / "project.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "not-a-project").mkdir()
    reads = []
    real_load = json.load
    with patch("dubbing_service.app.storage.projects._ensure_projects_root", return_value=tmp_path),             patch.object(project_storage.json, "load", lambda handle: reads.append(handle.name) or real_load(handle)):
        project_storage._SUMMARIES.clear()
        first = list_project_summaries()
        assert [(row["id"], row["name"]) for row in first] == [("legacy01", "Legacy"), ("current1", "Current")]
        assert len(reads) == 3, "each file is read once (the broken one too, and skipped)"
        reads.clear()
        assert list_project_summaries() == first
        assert len(reads) == 1, "only the unreadable file is tried again; the good ones come from memory"
        reads.clear()
        _write_project(tmp_path, "current1", "Renamed", "2026-01-01T09:00:00Z", extra={"description": "grew a little"})
        assert [row["name"] for row in list_project_summaries()] == ["Legacy", "Renamed"]
        assert sorted(Path(name).parent.name for name in reads) == ["broken01", "current1"], "a changed file is read again, an unchanged one is not"
        changing.unlink()
        (tmp_path / "current1").rmdir() if not any((tmp_path / "current1").iterdir()) else None
        assert [row["id"] for row in list_project_summaries()] == ["legacy01"], "a deleted project drops out"


def test_project_summaries_route_is_light_and_is_not_shadowed_by_the_project_routes(tmp_path):
    _write_project(tmp_path, "aaaa0001", "First", "2026-01-01T08:00:00Z")
    with patch("dubbing_service.app.storage.projects._ensure_projects_root", return_value=tmp_path):
        project_storage._SUMMARIES.clear()
        with TestClient(create_app()) as client:
            response = client.get("/api/v1/projects/summaries")
    assert response.status_code == 200
    assert response.json() == [{"id": "aaaa0001", "name": "First", "created_at": "2026-01-01T08:00:00Z"}]
