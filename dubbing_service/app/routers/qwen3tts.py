"""Qwen3-TTS VoiceDesign API 路由。

提供 AI 生成音色的接口，调用本地 Qwen3-TTS VoiceDesign 服务。
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dubbing_service.app.core.config import get_qwen3tts_base_url

router = APIRouter(prefix="/qwen3tts", tags=["qwen3tts"])


def _get_qwen3tts_base_url() -> str:
    """获取 Qwen3-TTS 服务的 base URL。"""
    return get_qwen3tts_base_url()


def _normalize_audio_path(audio_file: Optional[str]) -> Optional[str]:
    if not audio_file:
        return None

    repo_root = Path(__file__).resolve().parents[3]
    allowed_roots = [
        (repo_root / "outputs").resolve(),
        (repo_root / "service" / "outputs").resolve(),
    ]

    try:
        raw_path = Path(audio_file)
        if raw_path.is_absolute():
            candidate = raw_path.resolve()
        else:
            candidate = (repo_root / raw_path).resolve()
    except Exception:
        return audio_file

    for root in allowed_roots:
        if candidate.is_relative_to(root):
            try:
                rel = candidate.relative_to(repo_root)
            except Exception:
                return audio_file
            return rel.as_posix()

    return audio_file


class GenerateRequest(BaseModel):
    """AI 生成音色请求"""
    text: str = Field(..., description="要合成的文本（试听文本）")
    instruct: str = Field(..., description="音色描述")
    language: str = Field("Chinese", description="语言")


class GenerateResponse(BaseModel):
    """AI 生成音色响应"""
    success: bool
    message: str
    audio_file: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate: int = 24000


@router.post("/generate", response_model=GenerateResponse)
def api_generate_voice(request: GenerateRequest) -> GenerateResponse:
    """
    使用 Qwen3-TTS VoiceDesign 生成音色。
    
    根据文本描述生成音色并合成试听音频。
    """
    base_url = _get_qwen3tts_base_url()
    url = f"{base_url}/api/v1/tts/synthesize"
    
    payload = {
        "text": request.text,
        "instruct": request.instruct,
        "language": request.language,
        "return_base64": False,
        "save_file": True,
    }
    
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            resp_text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        detail = None
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict) and parsed.get("detail"):
                detail = str(parsed["detail"])
        except json.JSONDecodeError:
            pass
        
        msg = detail if detail else f"Qwen3-TTS 服务错误: {exc.code}"
        return GenerateResponse(success=False, message=msg)
    except urllib.error.URLError:
        return GenerateResponse(
            success=False,
            message=f"无法连接 Qwen3-TTS 服务 ({base_url})，请确保服务已启动"
        )
    except Exception as exc:
        return GenerateResponse(success=False, message=f"请求失败: {str(exc)}")
    
    try:
        raw = json.loads(resp_text)
    except json.JSONDecodeError:
        return GenerateResponse(success=False, message="响应解析失败")
    
    if not isinstance(raw, dict):
        return GenerateResponse(success=False, message="响应格式错误")
    
    if raw.get("success") is False:
        return GenerateResponse(
            success=False,
            message=raw.get("message") or raw.get("error") or "生成失败"
        )
    
    audio_file = None
    for key in ("audio_file", "audio_path", "output_path", "file_path", "path"):
        if key in raw and isinstance(raw[key], str):
            audio_file = raw[key]
            break
    
    return GenerateResponse(
        success=True,
        message="生成成功",
        audio_file=_normalize_audio_path(audio_file),
        duration_seconds=raw.get("duration_seconds"),
        sample_rate=raw.get("sample_rate", 24000),
    )


@router.get("/audio")
def api_get_audio(path: str = Query(..., description="音频文件路径")):
    """
    获取 AI 生成的音频文件。
    
    用于前端预览 AI 生成的音频。
    """
    try:
        repo_root = Path(__file__).resolve().parents[3]
        allowed_roots = [
            (repo_root / "outputs").resolve(),
            (repo_root / "service" / "outputs").resolve(),
        ]

        raw_path = Path(path)
        if raw_path.is_absolute():
            candidate = raw_path.resolve()
        else:
            candidate = (repo_root / raw_path).resolve()

        def _is_allowed(p: Path) -> bool:
            return any(p.is_relative_to(root) for root in allowed_roots)

        if not _is_allowed(candidate):
            if Path(path).name == path:
                fallback = (allowed_roots[0] / path).resolve()
                if _is_allowed(fallback):
                    candidate = fallback
                else:
                    raise HTTPException(status_code=400, detail="音频路径非法")
            else:
                raise HTTPException(status_code=400, detail="音频路径非法")

        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="音频文件不存在")
        
        return FileResponse(
            str(candidate),
            media_type="audio/wav",
            filename=candidate.name,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"获取音频失败: {str(exc)}")


@router.get("/status")
def api_get_status():
    """
    获取 Qwen3-TTS 服务状态。
    """
    base_url = _get_qwen3tts_base_url()
    url = f"{base_url}/health"
    
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp_text = resp.read().decode("utf-8", errors="replace")
            raw = json.loads(resp_text)
            return {
                "available": True,
                "model_loaded": raw.get("model_loaded", False),
                "backend": raw.get("backend", "qwen3tts"),
            }
    except Exception:
        return {
            "available": False,
            "model_loaded": False,
            "backend": "qwen3tts",
        }
