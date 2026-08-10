"""Live DeepSeek segmentation smoke test using an isolated temporary project."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from dubbing_service.app.core.config import clear_config_caches  # noqa: E402
from dubbing_service.app.main import create_app  # noqa: E402
from dubbing_service.app.storage.llm_configs import load_llm_configs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send one real DeepSeek segmentation request using an isolated temporary project.",
        epilog="This command may incur API charges and sends the built-in sample text to DeepSeek.",
    )
    parser.parse_args()

    default = next((item for item in load_llm_configs() if item.is_default), None)
    if default is None:
        print("FAIL: no default LLM configuration")
        return 1

    source = "夜色慢慢降临。远处的灯火依次亮起，微风从安静的街道上轻轻吹过。"
    old_root = os.environ.get("DUBBING_PROJECTS_ROOT")
    try:
        with tempfile.TemporaryDirectory(prefix="dubbing-llm-smoke-") as temporary:
            os.environ["DUBBING_PROJECTS_ROOT"] = temporary
            clear_config_caches()
            app = create_app()
            with TestClient(app) as client:
                created = client.post("/api/v1/projects", json={"name": "LLM smoke"})
                created.raise_for_status()
                project_id = created.json()["id"]
                response = client.post(
                    f"/api/v1/projects/{project_id}/segments/auto-generate",
                    json={"text": source, "llm_config_id": default.id},
                    timeout=700,
                )
                response.raise_for_status()
                segments = response.json().get("segments") or []
                if not segments:
                    print("FAIL: DeepSeek returned no validated segments")
                    return 1
                print(
                    f"PASS: model={default.model}, segments={len(segments)}, "
                    f"new_roles={response.json().get('new_roles_count', 0)}"
                )
                return 0
    finally:
        if old_root is None:
            os.environ.pop("DUBBING_PROJECTS_ROOT", None)
        else:
            os.environ["DUBBING_PROJECTS_ROOT"] = old_root
        clear_config_caches()


if __name__ == "__main__":
    raise SystemExit(main())
