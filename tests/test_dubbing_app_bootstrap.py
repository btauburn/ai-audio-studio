import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.main import create_app


class TestDubbingAppBootstrap(unittest.TestCase):
    def test_create_app_works_when_cwd_is_not_repo_root(self) -> None:
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.chdir(tmp_dir)
            try:
                clear_config_caches()
                app = create_app()
                with TestClient(app) as client:
                    resp = client.get("/")
            finally:
                os.chdir(old_cwd)

        self.assertEqual(resp.status_code, 200)
        self.assertIn("<!doctype html>", resp.text.lower())

    def test_registers_v1_api_routes(self) -> None:
        clear_config_caches()
        app = create_app()
        paths = {route.path for route in app.routes}

        expected_paths = {
            "/api/v1/ready",
            "/api/v1/projects/{project_id}/subtitles/generate-from-final",
            "/api/v1/qwen3tts/status",
            "/api/v1/health",
        }

        for path in expected_paths:
            self.assertIn(path, paths)

        # 确认不存在无版本号的 /api/ 路由（防止误注册双前缀）
        legacy_paths = [p for p in paths if p.startswith("/api/") and not p.startswith("/api/v1/")]
        self.assertEqual(legacy_paths, [], f"不应存在 /api/ 无版本前缀路由: {legacy_paths}")

    def test_webui_entry_document_is_never_served_from_stale_cache(self) -> None:
        clear_config_caches()
        app = create_app()
        with TestClient(app) as client:
            root_response = client.get("/")
            web_response = client.get("/web/")

        self.assertEqual(root_response.status_code, 200)
        self.assertEqual(web_response.status_code, 200)
        self.assertEqual(root_response.headers.get("cache-control"), "no-store, max-age=0")
        self.assertEqual(web_response.headers.get("cache-control"), "no-store, max-age=0")


if __name__ == "__main__":
    unittest.main()
