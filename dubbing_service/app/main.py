"""FastAPI entrypoint for AI Audio Studio.

提供项目管理、角色管理、LLM 分段、批量合成、字幕生成等功能的后端 API。
API 前缀：/api/v1/
"""

from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response

from dubbing_service.app.core.config import (
    _load_file_config,
    get_config,
)
from dubbing_service.app.core.errors import install_exception_handlers
from dubbing_service.app.core.helpers import expose_secrets as _expose_secrets
from dubbing_service.app.core.logging_config import setup_logging
from dubbing_service.app.providers.subtitles import (
    clear_whisper_model_cache,
    is_whisper_model_loaded,
)
from dubbing_service.app.routers import projects as projects_router
from dubbing_service.app.routers import roles as roles_router
from dubbing_service.app.routers import segments as segments_router
from dubbing_service.app.routers import llm_configs as llm_configs_router
from dubbing_service.app.routers import llm_providers as llm_providers_router
from dubbing_service.app.routers import voices as voices_router
from dubbing_service.app.routers import settings as settings_router
from dubbing_service.app.routers import synthesis as synthesis_router
from dubbing_service.app.routers import subtitles as subtitles_router
from dubbing_service.app.routers import qwen3tts as qwen3tts_router
from dubbing_service.app.routers import ws_synthesis as ws_synthesis_router
from dubbing_service.app.routers import jobs as jobs_router
from dubbing_service.app.routers import masters as masters_router
from dubbing_service.app.routers import production as production_router
from dubbing_service.app.routers import sound_generation as sound_generation_router

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API 前缀
# ---------------------------------------------------------------------------
API_V1 = "/api/v1"


class WebUIStaticFiles(StaticFiles):
    """Serve the WebUI while preventing a stale entry document.

    Vendor assets use versioned paths and may be cached normally.  The entry
    document must be revalidated on every visit, otherwise a browser can keep
    an older HTML shell that references obsolete scripts after an upgrade.
    """

    async def get_response(self, path: str, scope: dict) -> Response:
        response = await super().get_response(path, scope)
        if path.replace("\\", "/").strip("/") in {"", ".", "index.html"}:
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期：启动时初始化日志，关闭时清理 TTS 子进程。"""
    setup_logging()
    logger.info("Dubbing service starting up")
    # A browser request is not the owner of a sound-generation job.  Jobs are
    # durable, but Python threads are not; make an interrupted restart explicit
    # instead of leaving a project permanently stuck in "running".
    try:
        from dubbing_service.app.storage.projects import list_projects
        from dubbing_service.app.storage.sound_jobs import recover_interrupted_sound_jobs

        for project in list_projects():
            recover_interrupted_sound_jobs(project.id)
    except Exception:
        logger.warning("Failed to recover interrupted sound-generation jobs", exc_info=True)
    yield
    # ── Shutdown ──────────────────────────────────────────────────────
    logger.info("Dubbing service shutting down, cleaning up ...")
    try:
        from dubbing_service.app.routers.settings import shutdown_tts_process
        shutdown_tts_process()
        logger.info("TTS subprocess cleaned up")
    except Exception:
        logger.warning("Failed to clean up TTS subprocess", exc_info=True)
    try:
        clear_whisper_model_cache()
        logger.info("Subtitle model caches cleaned up")
    except Exception:
        logger.warning("Failed to clean up subtitle model caches", exc_info=True)
    try:
        from dubbing_service.app.services.sound_jobs import shutdown_sound_jobs

        shutdown_sound_jobs()
        logger.info("Stable Audio SFX worker cleaned up")
    except Exception:
        logger.warning("Failed to clean up Stable Audio SFX worker", exc_info=True)


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。"""

    app = FastAPI(
        title="AI Audio Studio API",
        description="多 TTS 后端的本地有声作品制作服务。",
        version="1.0.0",
        lifespan=lifespan,
    )

    # 安装全局异常处理器（统一错误响应格式）
    install_exception_handlers(app)

    # ── API 路由 ────────────────────────────────────────────
    app.include_router(projects_router.router, prefix=f"{API_V1}/projects", tags=["projects"])
    app.include_router(roles_router.router, prefix=f"{API_V1}/projects", tags=["roles"])
    app.include_router(segments_router.router, prefix=API_V1, tags=["segments"])
    app.include_router(llm_providers_router.router, prefix=API_V1, tags=["llm-providers"])
    app.include_router(llm_configs_router.router, prefix=API_V1, tags=["llm-configs"])
    app.include_router(voices_router.router, prefix=API_V1, tags=["voices"])
    app.include_router(settings_router.router, prefix=API_V1, tags=["settings"])
    app.include_router(synthesis_router.router, prefix=API_V1, tags=["synthesis"])
    app.include_router(subtitles_router.router, prefix=API_V1, tags=["subtitles"])
    app.include_router(qwen3tts_router.router, prefix=API_V1, tags=["qwen3tts"])
    app.include_router(ws_synthesis_router.router, prefix=API_V1, tags=["ws-synthesis"])
    app.include_router(jobs_router.router, prefix=API_V1, tags=["jobs"])
    app.include_router(masters_router.router, prefix=API_V1, tags=["masters"])
    app.include_router(production_router.router, prefix=API_V1, tags=["production"])
    app.include_router(sound_generation_router.router, prefix=API_V1, tags=["sound-generation"])

    # Web 管理界面静态文件（使用绝对路径，避免受当前工作目录影响）
    webui_dir = Path(__file__).resolve().parents[1] / "webui"
    app.mount("/web", WebUIStaticFiles(directory=str(webui_dir), html=True), name="webui")

    @app.get("/", response_class=HTMLResponse)
    async def index_page() -> HTMLResponse:
        """首页：简单重定向到 Web 管理界面的入口 HTML。

        这里直接读取打包在仓库中的 dubbing_service/webui/index.html。
        """

        try:
            with (webui_dir / "index.html").open("r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            return HTMLResponse("<h1>Web UI 未找到</h1><p>请确认 dubbing_service/webui/index.html 是否存在。</p>", status_code=500)

        return HTMLResponse(
            content,
            headers={
                "Cache-Control": "no-store, max-age=0",
                "Pragma": "no-cache",
            },
        )

    @app.get(f"{API_V1}/ready")
    async def ready_check() -> dict[str, object]:
        """轻量就绪检查接口。

        仅用于启动器或外部探针判断 Web 平台是否已可访问，
        不执行 TTS/GPU 等较重的附加诊断。
        """

        return {"status": "ok"}

    @app.get(f"{API_V1}/health")
    def health_check() -> dict[str, object]:
        """健康检查接口。

        返回服务状态以及部分关键配置，方便排查部署问题。
        """

        cfg_local = get_config()

        # 额外探测 TTS 服务健康与模型加载状态（根据当前配置的后端选择正确的端口）
        tts_model_loaded = None
        tts_device = None
        tts_health_ok = False
        tts_backend = None  # 从 TTS 服务获取当前后端类型
        tts_runtime_backend = None
        tts_config_backend = None
        tts_service_running = False

        # 探测当前进程所在机器的 GPU 显存占用情况（如果安装了 torch 且 CUDA 可用）
        gpu_name: str | None = None
        gpu_mem_total_gb: float | None = None
        gpu_mem_used_gb: float | None = None

        # 获取当前配置的 TTS 后端，选择对应的健康检查 URL
        file_cfg_for_health = _load_file_config()
        configured_backend = settings_router._normalize_tts_backend(file_cfg_for_health.get("tts_backend"))
        running, _, runtime_backend = settings_router._get_tts_runtime_state()
        tts_service_running = running
        tts_runtime_backend = runtime_backend if running else None
        tts_config_backend = configured_backend

        probe_backend = runtime_backend if running and runtime_backend else configured_backend
        base_url = settings_router._get_tts_backend_base_url(probe_backend)
        health_url = f"{base_url.rstrip('/')}/health"
        
        logger.debug(
            "健康检查探测 TTS: probe_backend=%s runtime_backend=%s configured_backend=%s url=%s",
            probe_backend,
            runtime_backend,
            configured_backend,
            health_url,
        )
        try:
            with urllib.request.urlopen(health_url, timeout=3) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(body)
            except Exception:
                data = {}
            if isinstance(data, dict):
                # 如果响应中明确包含 model_loaded 字段，使用其值；
                # 否则服务能正常响应即视为已就绪（Qwen3 等外部服务可能不返回此字段）
                if "model_loaded" in data:
                    tts_model_loaded = bool(data["model_loaded"])
                else:
                    tts_model_loaded = True
                tts_device = str(data.get("device") or "unknown")
                tts_backend = data.get("backend") or probe_backend
                tts_health_ok = True
            logger.debug("TTS 健康检查成功: model_loaded=%s backend=%s", tts_model_loaded, tts_backend)
        except Exception as exc:
            logger.debug("TTS 健康检查失败: %s (url=%s)", exc, health_url)
            tts_model_loaded = None
            tts_device = None
            tts_health_ok = False

        # 本地 GPU 显存（如果有 GPU 且安装了 torch），单位 GiB，保留一位小数
        try:  # pragma: no cover - 与实际部署环境强相关
            import torch  # type: ignore

            if torch.cuda.is_available():
                # 默认查看 0 号设备的显存占用情况
                free_bytes, total_bytes = torch.cuda.mem_get_info()  # type: ignore[assignment]
                used_bytes = max(0, int(total_bytes) - int(free_bytes))
                gb = 1024**3
                gpu_mem_total_gb = round(total_bytes / gb, 1)
                gpu_mem_used_gb = round(used_bytes / gb, 1)
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    gpu_name = None
        except Exception:
            gpu_name = None
            gpu_mem_total_gb = None
            gpu_mem_used_gb = None

        file_cfg = file_cfg_for_health
        show_secrets = _expose_secrets()

        if not show_secrets and isinstance(file_cfg, dict):
            safe_file_cfg = dict(file_cfg)
            for key in (
                "llm_api_key",
                "tts_api_key",
                "llm_api_key_ref",
                "tts_api_key_ref",
            ):
                if safe_file_cfg.get(key):
                    safe_file_cfg[key] = "********"
            file_cfg_out: object = safe_file_cfg
        else:
            file_cfg_out = file_cfg

        projects_root_out: str
        if show_secrets:
            projects_root_out = str(cfg_local.projects_root_abs)
        else:
            try:
                pr = Path(cfg_local.projects_root)
                projects_root_out = pr.name if pr.is_absolute() else str(pr)
            except Exception:
                projects_root_out = "********"

        return {
            "status": "ok",
            "tts_base_url": cfg_local.tts_base_url,
            "projects_root": projects_root_out,
            "llm_configured": bool(cfg_local.llm_base_url and cfg_local.llm_api_key),
            # 下面几个字段仅用于调试配置问题，后续可以去掉
            "llm_base_url": cfg_local.llm_base_url,
            "llm_api_key_present": bool(cfg_local.llm_api_key),
            "file_config": file_cfg_out,
            # 新增：TTS / Whisper 模型加载状态
            "tts_model_loaded": tts_model_loaded,
            "tts_device": tts_device,
            "tts_health_ok": tts_health_ok,
            "tts_backend": tts_backend,  # 当前 TTS 后端类型 (indextts/cosyvoice3)
            "tts_runtime_backend": tts_runtime_backend,
            "tts_config_backend": tts_config_backend,
            "tts_service_running": tts_service_running,
            "whisper_model_loaded": is_whisper_model_loaded(),
            # GPU 显存占用（GiB），如果无法探测则为 null
            "gpu_name": gpu_name,
            "gpu_mem_total_gb": gpu_mem_total_gb,
            "gpu_mem_used_gb": gpu_mem_used_gb,
        }

    return app


app = create_app()


if __name__ == "__main__":
    """允许通过 `uv run dubbing_service/app/main.py` 直接启动服务。"""

    import uvicorn

    uvicorn.run(
        "dubbing_service.app.main:app",
        host="127.0.0.1",
        port=9000,
        reload=False,
    )
