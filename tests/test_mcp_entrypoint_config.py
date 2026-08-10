from __future__ import annotations

from service.setup_mcp import create_mcp_config


def test_mcp_config_uses_locked_offline_project_runtime() -> None:
    server = create_mcp_config()["indextts2"]

    assert server["command"].endswith((".venv\\Scripts\\python.exe", ".venv/bin/python"))
    assert server["args"][0].endswith("service\\mcp_server.py") or server["args"][0].endswith(
        "service/mcp_server.py"
    )
    assert server["args"][1] == "--model_dir"
    assert server["env"]["HF_HUB_OFFLINE"] == "1"
    assert server["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert server["env"]["PYTHONIOENCODING"] == "utf-8"
