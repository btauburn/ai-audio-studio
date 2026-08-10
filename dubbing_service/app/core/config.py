"""Application configuration for the dubbing service.

配置来源优先级：环境变量 > 默认值。

环境变量：
- DUBBING_TTS_BASE_URL: IndexTTS2 REST 服务地址，默认 http://127.0.0.1:8001
- DUBBING_COSYVOICE_BASE_URL: CosyVoice3 REST 服务地址（可选）
- DUBBING_QWEN3_TTS_BASE_URL: Qwen3-TTS VoiceDesign REST 服务地址（可选）
- DUBBING_QWEN3_TTS_CV_BASE_URL: Qwen3-TTS CustomVoice REST 服务地址（可选）
- DUBBING_LLM_BASE_URL: 兼容 OpenAI API 协议的 LLM Base URL，可为空
- DUBBING_LLM_API_KEY: LLM API Key，可为空
- DUBBING_PROJECTS_ROOT: 配音项目根目录，默认 dubbing_projects
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, field_validator

from dubbing_service.app.core.path_safety import validate_simple_component_id
from dubbing_service.app.core.secret_store import SecretStoreError, get_secret

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _secret_from_file_config(file_cfg: Dict[str, Any], key: str) -> str | None:
    reference = file_cfg.get(f"{key}_ref")
    if isinstance(reference, str) and reference:
        try:
            return get_secret(reference)
        except SecretStoreError:
            return None
    # Legacy compatibility until ``scripts/migrate_llm_secrets.py`` is run.
    value = file_cfg.get(key)
    if isinstance(value, str) and value:
        return value
    return None


@lru_cache(maxsize=1)
def _load_file_config() -> Dict[str, Any]:
    """从 dubbing_service/config.json 读取配置（如果存在）。

    允许用户通过配置文件而不是系统环境变量来配置 TTS/LLM。
    配置优先级：环境变量 > 配置文件 > 默认值。
    """

    # config.py 位于 dubbing_service/app/core/，向上 3 级到 dubbing_service 目录
    base_dir = Path(__file__).resolve().parents[2]
    cfg_path = base_dir / "config.json"

    if not cfg_path.is_file():
        return {}

    try:
        import json

        with cfg_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    if isinstance(data, dict):
        return data
    return {}


class AppConfig(BaseModel):
    """全局配置对象。

    通过 `get_config()` 获取单例实例，以便在 FastAPI 路由和服务中使用。
    """

    # 重要：开启默认值校验，这样即便未显式传入字段，field_validator 也会运行，
    # 从环境变量或 config.json 中加载配置。
    model_config = ConfigDict(validate_default=True)

    tts_base_url: str = "http://127.0.0.1:8001"
    cosyvoice_base_url: str | None = None
    qwen3tts_base_url: str | None = None
    qwen3tts_cv_base_url: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    # 可选：TTS 服务的 API Key，用于需要鉴权的 TTS REST 服务
    tts_api_key: str | None = None
    projects_root: Path = Path("dubbing_projects")

    @field_validator("tts_base_url", mode="before")
    @classmethod
    def _from_env_tts_base_url(cls, v: str | None) -> str:
        env_value = os.getenv("DUBBING_TTS_BASE_URL")
        if env_value:
            return env_value

        file_cfg = _load_file_config()
        file_value = file_cfg.get("tts_base_url")
        if isinstance(file_value, str) and file_value:
            return file_value

        return v or "http://127.0.0.1:8001"

    @field_validator("qwen3tts_base_url", mode="before")
    @classmethod
    def _from_env_qwen3tts_base_url(cls, v: str | None) -> str | None:
        env_value = os.getenv("DUBBING_QWEN3_TTS_BASE_URL")
        if env_value:
            return env_value

        file_cfg = _load_file_config()
        file_value = file_cfg.get("qwen3tts_base_url")
        if isinstance(file_value, str) and file_value:
            return file_value

        return v

    @field_validator("cosyvoice_base_url", mode="before")
    @classmethod
    def _from_env_cosyvoice_base_url(cls, v: str | None) -> str | None:
        env_value = os.getenv("DUBBING_COSYVOICE_BASE_URL")
        if env_value:
            return env_value

        file_cfg = _load_file_config()
        file_value = file_cfg.get("cosyvoice_base_url")
        if isinstance(file_value, str) and file_value:
            return file_value

        return v

    @field_validator("qwen3tts_cv_base_url", mode="before")
    @classmethod
    def _from_env_qwen3tts_cv_base_url(cls, v: str | None) -> str | None:
        env_value = os.getenv("DUBBING_QWEN3_TTS_CV_BASE_URL")
        if env_value:
            return env_value

        file_cfg = _load_file_config()
        file_value = file_cfg.get("qwen3tts_cv_base_url")
        if isinstance(file_value, str) and file_value:
            return file_value

        return v

    @field_validator("llm_base_url", mode="before")
    @classmethod
    def _from_env_llm_base_url(cls, v: str | None) -> str | None:
        env_value = os.getenv("DUBBING_LLM_BASE_URL")
        if env_value:
            return env_value

        file_cfg = _load_file_config()
        file_value = file_cfg.get("llm_base_url")
        if isinstance(file_value, str) and file_value:
            return file_value

        return v

    @field_validator("llm_api_key", mode="before")
    @classmethod
    def _from_env_llm_api_key(cls, v: str | None) -> str | None:
        env_value = os.getenv("DUBBING_LLM_API_KEY")
        if env_value:
            return env_value

        file_cfg = _load_file_config()
        file_value = _secret_from_file_config(file_cfg, "llm_api_key")
        if file_value:
            return file_value

        return v

    @field_validator("tts_api_key", mode="before")
    @classmethod
    def _from_env_tts_api_key(cls, v: str | None) -> str | None:
        env_value = os.getenv("DUBBING_TTS_API_KEY")
        if env_value:
            return env_value

        file_cfg = _load_file_config()
        file_value = _secret_from_file_config(file_cfg, "tts_api_key")
        if file_value:
            return file_value

        return v

    @field_validator("projects_root", mode="before")
    @classmethod
    def _from_env_projects_root(cls, v: str | os.PathLike[str] | None) -> Path:
        env_value = os.getenv("DUBBING_PROJECTS_ROOT")
        if env_value:
            return Path(env_value)

        file_cfg = _load_file_config()
        file_value = file_cfg.get("projects_root")
        if isinstance(file_value, str) and file_value:
            return Path(file_value)

        if v is None:
            return Path("dubbing_projects")
        return Path(v)

    @property
    def projects_root_abs(self) -> Path:
        """返回项目根目录的绝对路径。"""

        try:
            pr = Path(self.projects_root)
        except Exception:
            pr = Path("dubbing_projects")

        if pr.is_absolute():
            return pr
        return _REPO_ROOT / pr


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """获取全局配置单例。

    使用 lru_cache 避免重复解析环境变量。
    """

    return AppConfig()


def clear_config_caches() -> None:
    """清理配置相关的缓存，使 config.json 和环境变量的变更能在运行时立即生效。"""

    # 清理文件配置缓存
    try:
        _load_file_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        # 失败时忽略，至少不影响后续正常读取
        pass

    # 清理 AppConfig 单例缓存
    try:
        get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        # 失败时忽略，让下次进程重启后至少能生效
        pass


def safe_project_root(project_id: str) -> Path:
    validate_simple_component_id(project_id, label="project_id")
    cfg = get_config()
    base = cfg.projects_root_abs
    base_resolved = base.resolve()

    try:
        candidate = (base / project_id).resolve()
    except Exception as exc:
        raise ValueError("invalid project_id") from exc

    if not candidate.is_relative_to(base_resolved):
        raise ValueError("invalid project_id")

    return candidate


def _normalize_base_url(value: str | None) -> str | None:
    if not value:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    return text.rstrip("/") if text else None


def _replace_port(base_url: str, source_ports: tuple[str, ...], target_port: str) -> str:
    for port in source_ports:
        token = f":{port}"
        if token in base_url:
            return base_url.replace(token, f":{target_port}")
    return base_url


def get_qwen3tts_base_url() -> str:
    """获取 Qwen3-TTS VoiceDesign 服务地址（可通过配置覆盖）。"""
    cfg = get_config()
    override = _normalize_base_url(cfg.qwen3tts_base_url)
    if override:
        return override
    base_url = _normalize_base_url(cfg.tts_base_url) or "http://127.0.0.1:8001"
    return _replace_port(base_url, ("8001", "8002"), "8003")


def get_cosyvoice_base_url() -> str:
    """获取 CosyVoice3 服务地址（可通过配置覆盖）。"""
    cfg = get_config()
    override = _normalize_base_url(cfg.cosyvoice_base_url)
    if override:
        return override
    base_url = _normalize_base_url(cfg.tts_base_url) or "http://127.0.0.1:8001"
    return _replace_port(base_url, ("8001", "8003", "8004"), "8002")


def get_qwen3tts_cv_base_url() -> str:
    """获取 Qwen3-TTS CustomVoice 服务地址（可通过配置覆盖）。"""
    cfg = get_config()
    override = _normalize_base_url(cfg.qwen3tts_cv_base_url)
    if override:
        return override
    base_url = _normalize_base_url(cfg.tts_base_url) or "http://127.0.0.1:8001"
    return _replace_port(base_url, ("8001", "8002", "8003"), "8004")
