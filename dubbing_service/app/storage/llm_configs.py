"""File-based helpers for managing LLM model configurations."""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Generator, List, Literal

from pydantic import BaseModel, ValidationError


class LLMConfigStorageError(Exception):
    """与 LLM 配置存储相关的异常。"""


_LLM_CONFIGS_LOCK = Lock()


@contextmanager
def llm_configs_lock() -> Generator[None, None, None]:
    """LLM 配置读写锁的上下文管理器，保护 llm_models.json 的读-改-写操作。"""
    _LLM_CONFIGS_LOCK.acquire()
    try:
        yield
    finally:
        _LLM_CONFIGS_LOCK.release()


class LLMConfig(BaseModel):
    """单个 LLM 配置项。

    - id: 配置 ID（短 UUID）。
    - name: 在前端下拉菜单中显示的名称。
    - model: 发送给 LLM 服务的模型名称（如 deepseek-v4-flash）。
    - max_tokens: 可选，覆盖模型默认的 max_tokens；为 None 时表示使用模型默认值。
    - description: 可选描述。
    - is_default: 是否为默认配置，用于前端初始选中。
    """

    id: str
    name: str
    # 可选：关联的 LLM 提供商 ID；为空时表示使用默认提供商
    provider_id: str | None = None
    model: str
    max_tokens: int | None = None
    description: str | None = None
    # 是否在调用时启用 response_format={"type": "json_object"}
    response_format_json: bool = False
    # DeepSeek V4 thinking 参数；为空时不显式覆盖服务端默认值
    thinking: Literal["enabled", "disabled"] | None = None
    reasoning_effort: Literal["high", "max"] | None = None
    # 可选的一次性模型回退；仅由显式启用的配置触发
    fallback_model: str | None = None
    fallback_enabled: bool = False
    is_default: bool = False


def _llm_models_path() -> Path:
    """返回全局 LLM 配置文件路径（位于 dubbing_service/llm_models.json）。"""

    base_dir = Path(__file__).resolve().parents[2]
    return base_dir / "llm_models.json"


def _migrate_deepseek_v4_configs(configs: List[LLMConfig]) -> bool:
    """Migrate retired DeepSeek aliases in place while preserving IDs/providers."""

    changed = False
    for cfg in configs:
        if cfg.model == "deepseek-chat":
            cfg.name = "DeepSeek V4 Flash"
            cfg.model = "deepseek-v4-flash"
            cfg.max_tokens = 64000
            cfg.description = (
                "默认分段模型：JSON 模式、关闭 thinking；"
                "校验或临时故障时仅回退一次 V4 Pro。"
            )
            cfg.response_format_json = True
            cfg.thinking = "disabled"
            cfg.reasoning_effort = None
            cfg.fallback_model = "deepseek-v4-pro"
            cfg.fallback_enabled = True
            changed = True
        elif cfg.model == "deepseek-reasoner":
            cfg.name = "DeepSeek V4 Pro"
            cfg.model = "deepseek-v4-pro"
            cfg.max_tokens = 64000
            cfg.description = (
                "V4 Flash 的备用模型，也可手动选择；默认关闭 thinking 以控制延迟与成本。"
            )
            cfg.response_format_json = True
            cfg.thinking = "disabled"
            cfg.reasoning_effort = None
            cfg.fallback_model = None
            cfg.fallback_enabled = False
            changed = True

    if not changed:
        return False

    flash = next((cfg for cfg in configs if cfg.model == "deepseek-v4-flash"), None)
    if flash is None:
        pro = next((cfg for cfg in configs if cfg.model == "deepseek-v4-pro"), None)
        flash = LLMConfig(
            id=uuid.uuid4().hex[:8],
            name="DeepSeek V4 Flash",
            provider_id=pro.provider_id if pro else None,
            model="deepseek-v4-flash",
            max_tokens=64000,
            description=(
                "默认分段模型：JSON 模式、关闭 thinking；"
                "校验或临时故障时仅回退一次 V4 Pro。"
            ),
            response_format_json=True,
            thinking="disabled",
            fallback_model="deepseek-v4-pro",
            fallback_enabled=True,
            is_default=True,
        )
        configs.insert(0, flash)

    for cfg in configs:
        cfg.is_default = cfg.id == flash.id
    return True


def load_llm_configs() -> List[LLMConfig]:
    """加载所有 LLM 配置。

    如果文件不存在，则返回空列表。
    """

    path = _llm_models_path()
    if not path.is_file():
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise LLMConfigStorageError(
            f"llm_models.json 内容损坏，无法解析 JSON: {exc}"
        ) from exc

    if not isinstance(raw, list):
        raise LLMConfigStorageError("llm_models.json 内容格式错误：应为列表")

    configs: List[LLMConfig] = []
    for item in raw:
        try:
            cfg = LLMConfig(**item)
        except ValidationError as exc:
            raise LLMConfigStorageError("解析 LLM 配置失败") from exc
        else:
            configs.append(cfg)

    if _migrate_deepseek_v4_configs(configs):
        save_llm_configs(configs)

    return configs


def save_llm_configs(configs: List[LLMConfig]) -> None:
    """将 LLM 配置列表保存到 llm_models.json，覆盖原有内容。"""

    path = _llm_models_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.model_dump(mode="json") for c in configs]

    tmp_path = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex}")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
    except OSError as exc:
        raise LLMConfigStorageError(f"保存 llm_models.json 失败: {exc}") from exc
    finally:
        if tmp_path.is_file():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def generate_llm_config_id() -> str:
    """生成短 UUID 作为 LLM 配置 ID。"""

    return uuid.uuid4().hex[:8]
