import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from dubbing_service.app.main import create_app
from dubbing_service.app.storage.projects import ProjectStorageError, list_projects


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


if __name__ == "__main__":
    unittest.main()
