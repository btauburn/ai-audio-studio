import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from dubbing_service.app.main import create_app
from dubbing_service.app.storage.projects import ProjectStorageError


class TestDubbingProjectsApi(unittest.TestCase):
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
