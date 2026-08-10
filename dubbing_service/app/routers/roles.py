"""Role management APIs for the dubbing service."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import List
import re
import shutil

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dubbing_service.app.core.audio_validation import (
    AudioProbeUnavailableError,
    AudioValidationError,
    normalize_audio_extension,
    validate_decodable_audio_async,
)
from dubbing_service.app.core.helpers import (
    get_max_upload_bytes as _get_max_upload_bytes,
    project_root,
    resolve_role_reference_audio,
    sha256_file as _sha256_file,
)
from dubbing_service.app.core.path_safety import (
    SEGMENT_AUDIO_EXTENSIONS,
    resolve_scoped_project_asset,
)
from dubbing_service.app.models.enums import RoleType
from dubbing_service.app.models.project import Role
from dubbing_service.app.routers.operation_guards import project_mutation_guard
from dubbing_service.app.routers.segments import remap_segments_for_project
from dubbing_service.app.storage import roles as roles_storage
from dubbing_service.app.storage.roles import RoleStorageError
from dubbing_service.app.storage.segmentation_commit import (
    commit_role_deletion,
    commit_role_synthesis_update,
)
from dubbing_service.app.storage.projects import (
    ProjectStorageError,
    load_project,
    project_meta_lock,
    save_project,
)
from dubbing_service.app.storage.voices import (
    VoiceStorageError,
    load_voices,
    voices_audio_dir,
)

router = APIRouter()


def _project_root(project_id: str) -> Path:
    root = project_root(project_id, ensure_dir=False)
    if not (root / "project.json").is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定项目不存在",
        )
    return root


def _cleanup_superseded_role_prompts(
    prompts_dir: Path,
    role_id: str,
    keep_name: str,
) -> None:
    """Best-effort GC after the new prompt path is durably committed."""
    try:
        prompts_root = prompts_dir.resolve()
    except Exception:
        return
    prefix = f"{role_id}_"
    try:
        candidates = list(prompts_dir.iterdir())
    except Exception:
        return
    for candidate in candidates:
        if candidate.name == keep_name or not candidate.name.startswith(prefix):
            continue
        try:
            resolved = candidate.resolve()
            if resolved.is_relative_to(prompts_root) and resolved.is_file():
                resolved.unlink(missing_ok=True)
        except Exception:
            continue


def _cleanup_committed_role_prompts(
    project_id: str,
    prompts_dir: Path,
    role_id: str,
) -> None:
    """GC prompts while protecting the currently committed path from races."""

    try:
        with project_meta_lock(project_id):
            project = load_project(project_id)
            role = next((item for item in project.roles if item.id == role_id), None)
            if role is None:
                return
            normalized = str(role.speaker_audio or "").replace("\\", "/")
            if normalized:
                parts = normalized.split("/")
                if (
                    len(parts) != 3
                    or parts[:2] != ["roles", "prompts"]
                    or parts[2] in {"", ".", ".."}
                ):
                    return
                keep_name = parts[2]
            else:
                keep_name = ""
            _cleanup_superseded_role_prompts(prompts_dir, role_id, keep_name)
    except (ProjectStorageError, OSError):
        # Cleanup is post-commit GC. On uncertainty, retaining an orphan is
        # safer than deleting the prompt referenced by durable metadata.
        return


def _delete_invalidated_segment_outputs(project_id: str, paths: tuple[str, ...]) -> None:
    """Best-effort GC after role and segment metadata commit successfully."""

    if not paths:
        return
    try:
        root = _project_root(project_id)
    except HTTPException:
        return
    for rel_path in paths:
        try:
            target = resolve_scoped_project_asset(
                root,
                rel_path,
                required_subdir="audio/segments",
                allowed_extensions=SEGMENT_AUDIO_EXTENSIONS,
            )
            if target.is_file():
                target.unlink(missing_ok=True)
        except Exception:
            continue


async def _require_decodable_audio(path: Path) -> None:
    try:
        await validate_decodable_audio_async(path)
    except AudioValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except AudioProbeUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


class RoleCreateRequest(BaseModel):
    """创建角色的请求体。"""

    name: str = Field(..., min_length=1, description="角色名称")
    role_type: RoleType = Field(
        RoleType.CHARACTER,
        description="角色类型：narrator=旁白，character=角色，other=其他",
    )
    description: str | None = Field(None, description="角色简介或备注")


class RoleVoiceProfileRequest(BaseModel):
    """角色级后端和音色配置；未提供的字段保持不变。"""

    tts_backend: str | None = None
    qwen3_speaker: str | None = None
    qwen3_language: str | None = None
    voice_description: str | None = None
    style_instruction: str | None = None
    preferred_speed_min: float | None = Field(default=None, ge=0.5, le=2.0)
    preferred_speed_max: float | None = Field(default=None, ge=0.5, le=2.0)
    voice_source: str | None = None
    voice_license_name: str | None = None
    voice_commercial_use: bool | None = None
    voice_rights_note: str | None = None


class BindVoiceRequest(BaseModel):
    """为角色绑定音色库中的音色。"""

    voice_id: str = Field(..., description="音色库条目的 ID")


class CleanupPromptsResponse(BaseModel):
    deleted_files: int
    updated_roles: int


@router.post(
    "/{project_id}/roles/cleanup-prompts",
    response_model=CleanupPromptsResponse,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_cleanup_role_prompts(project_id: str) -> CleanupPromptsResponse:
    project_root = _project_root(project_id)
    prompts_dir = project_root / "roles" / "prompts"
    try:
        prompts_dir.mkdir(parents=True, exist_ok=True)
        prompts_dir_resolved = prompts_dir.resolve()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="无法访问 roles/prompts 目录",
        )

    with project_meta_lock(project_id):
        try:
            project = load_project(project_id)
        except ProjectStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

        roles: List[Role] = project.roles or []

        role_prefix_re = re.compile(r"^[0-9a-fA-F]{8}$")

        prompt_files: list[Path] = []
        prefix_to_names: dict[str, list[str]] = {}
        try:
            for fp in prompts_dir.iterdir():
                if not fp.is_file():
                    continue
                name = fp.name
                if "_" not in name:
                    continue
                prefix = name.split("_", 1)[0]
                if not role_prefix_re.match(prefix):
                    continue
                prompt_files.append(fp)
                prefix_to_names.setdefault(prefix, []).append(name)
        except Exception:
            prompt_files = []
            prefix_to_names = {}

        keep_files: set[str] = set()
        updated_roles = 0

        for r in roles:
            if not r or not getattr(r, "id", None):
                continue

            rid = str(r.id)
            if getattr(r, "speaker_voice_id", None):
                if getattr(r, "speaker_audio", None):
                    r.speaker_audio = None
                    updated_roles += 1
                continue

            bound_name: str | None = None
            if getattr(r, "speaker_audio", None):
                try:
                    rp = (project_root / str(r.speaker_audio)).resolve()
                except Exception:
                    rp = None
                if rp is not None and rp.is_relative_to(prompts_dir_resolved) and rp.is_file():
                    bound_name = rp.name

            if bound_name is None:
                cands = prefix_to_names.get(rid) or []
                if cands:
                    best = max(cands)
                    r.speaker_audio = str(Path("roles") / "prompts" / best)
                    updated_roles += 1
                    bound_name = best

            if bound_name is not None:
                keep_files.add(bound_name)

        if updated_roles:
            try:
                project.updated_at = datetime.now(timezone.utc)
            except Exception:
                pass
            try:
                save_project(project)
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"保存项目失败: {exc}",
                ) from exc

        deleted = 0
        for fp in prompt_files:
            name = fp.name
            if name in keep_files:
                continue
            try:
                rp = fp.resolve()
            except Exception:
                continue
            if not rp.is_relative_to(prompts_dir_resolved):
                continue
            try:
                rp.unlink(missing_ok=True)
                deleted += 1
            except Exception:
                continue

    return CleanupPromptsResponse(deleted_files=deleted, updated_roles=updated_roles)


@router.post(
    "/{project_id}/roles",
    response_model=Role,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_create_role(project_id: str, payload: RoleCreateRequest) -> Role:
    """在指定项目下创建一个角色。"""

    try:
        with project_meta_lock(project_id):
            role = roles_storage.create_role(
                project_id=project_id,
                name=payload.name,
                role_type=payload.role_type,
                description=payload.description,
            )
        remap_segments_for_project(project_id)
        return role
    except RoleStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get("/{project_id}/roles/{role_id}/speaker-audio")
async def api_get_role_speaker_audio(project_id: str, role_id: str) -> FileResponse:
    """返回指定角色绑定的参考音频文件，用于在前端试听。

    音频文件路径存储在 Role.speaker_audio 中，为项目根目录下的相对路径。
    """

    if Path(role_id).name != role_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法角色 ID",
        )

    try:
        project = load_project(project_id)
    except ProjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    target: Role | None = None
    for r in project.roles:
        if r.id == role_id:
            target = r
            break

    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="角色未配置参考音频",
        )

    if not getattr(target, "speaker_audio", None) and not getattr(target, "speaker_voice_id", None):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="角色未配置参考音频",
        )

    audio_path = resolve_role_reference_audio(
        project_id,
        getattr(target, "speaker_audio", None),
        getattr(target, "speaker_voice_id", None),
    )

    if audio_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="参考音频文件不存在",
        )

    suffix = audio_path.suffix.lower()
    mime = "audio/wav"
    if suffix in {".mp3"}:
        mime = "audio/mpeg"
    elif suffix in {".ogg"}:
        mime = "audio/ogg"
    elif suffix in {".m4a"}:
        mime = "audio/mp4"

    return FileResponse(audio_path, media_type=mime, filename=audio_path.name)


@router.put(
    "/{project_id}/roles/{role_id}",
    response_model=Role,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_update_role(project_id: str, role_id: str, payload: RoleCreateRequest) -> Role:
    """更新指定项目下某个角色的基础信息。"""

    if Path(role_id).name != role_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法角色 ID",
        )

    try:
        with project_meta_lock(project_id):
            role = roles_storage.update_role(
                project_id=project_id,
                role_id=role_id,
                name=payload.name,
                role_type=payload.role_type,
                description=payload.description,
            )
    except RoleStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return role


@router.patch(
    "/{project_id}/roles/{role_id}/voice-profile",
    response_model=Role,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_update_role_voice_profile(
    project_id: str,
    role_id: str,
    payload: RoleVoiceProfileRequest,
) -> Role:
    """Update stable role-level TTS defaults without touching reference audio."""

    if Path(role_id).name != role_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="非法角色 ID")

    updates = payload.model_dump(exclude_unset=True)
    if "tts_backend" in updates:
        backend = str(updates["tts_backend"] or "").strip()
        if backend and backend not in {"indextts", "cosyvoice3", "qwen3tts", "qwen3tts_cv"}:
            raise HTTPException(status_code=400, detail="不支持的角色 TTS 后端")
    if "qwen3_language" in updates:
        language = str(updates["qwen3_language"] or "").strip()
        allowed_languages = {
            "Auto", "Chinese", "English", "Japanese", "Korean", "German",
            "French", "Russian", "Portuguese", "Spanish", "Italian",
        }
        if language and language not in allowed_languages:
            raise HTTPException(status_code=400, detail="不支持的 Qwen3 语言")
    speed_min = updates.get("preferred_speed_min")
    speed_max = updates.get("preferred_speed_max")
    if speed_min is not None or speed_max is not None:
        with project_meta_lock(project_id):
            current_project = load_project(project_id)
            current_role = next((item for item in current_project.roles if item.id == role_id), None)
        if current_role is None:
            raise HTTPException(status_code=404, detail="角色不存在")
        resolved_min = float(speed_min if speed_min is not None else current_role.preferred_speed_min)
        resolved_max = float(speed_max if speed_max is not None else current_role.preferred_speed_max)
        if resolved_min > resolved_max:
            raise HTTPException(status_code=400, detail="建议最低语速不能高于最高语速")

    try:
        result = commit_role_synthesis_update(project_id, role_id, updates)
    except (LookupError, ProjectStorageError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存角色音色配置失败，旧配置已保留: {exc}",
        ) from exc

    _delete_invalidated_segment_outputs(project_id, result.segment_audio_to_delete)
    return result.role


@router.delete(
    "/{project_id}/roles/{role_id}",
    dependencies=[Depends(project_mutation_guard)],
)
async def api_delete_role(project_id: str, role_id: str) -> dict[str, bool]:
    """删除指定项目下的一个角色，并清理所有段落中的 role_id 引用。"""

    if Path(role_id).name != role_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法角色 ID",
        )

    try:
        result = commit_role_deletion(project_id, role_id)
    except (LookupError, ProjectStorageError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除角色事务失败: {exc}",
        ) from exc

    # project.json 与 segments.json 已一并提交，此后清理失败只会留下孤儿文件。
    if result.prompt_to_delete:
        try:
            project_root = _project_root(project_id)
            project_root_resolved = project_root.resolve()
            prompt_path = (project_root / result.prompt_to_delete).resolve()
            if prompt_path.is_relative_to(project_root_resolved) and prompt_path.is_file():
                prompt_path.unlink(missing_ok=True)
        except Exception:
            pass

    return {"success": True}


@router.post(
    "/{project_id}/roles/{role_id}/bind-voice",
    response_model=Role,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_bind_role_voice(project_id: str, role_id: str, payload: BindVoiceRequest) -> Role:
    """从全局音色库中选择一个音色，并为角色绑定。

    实现方式：
    - 从 voices.json 中查找对应音色，获取其 audio_path；
    - 将该文件复制到项目的 roles/prompts/ 目录；
    - 更新角色的 speaker_audio 字段为项目内相对路径；
    - 原子更新角色配置，并使该角色已生成的 A/B 音频失效。
    """

    if Path(role_id).name != role_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法角色 ID",
        )

    try:
        voices = load_voices()
    except VoiceStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    target = next((v for v in voices if v.id == payload.voice_id), None)
    if target is None or not target.audio_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定的音色不存在或尚未上传参考音频",
        )

    base_dir = voices_audio_dir()
    try:
        base_dir_resolved = base_dir.resolve()
        src_path = Path(target.audio_path)
        if not src_path.is_absolute():
            src_path = (base_dir / str(target.audio_path)).resolve()
        else:
            src_path = src_path.resolve()
    except Exception:
        src_path = None

    if (
        src_path is None
        or not src_path.is_relative_to(base_dir_resolved)
        or not src_path.is_file()
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="音色对应的音频文件不存在，请在音色库中检查后重试。",
        )

    await _require_decodable_audio(src_path)

    project_root = _project_root(project_id)
    prompts_dir = project_root / "roles" / "prompts"
    try:
        prompts_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        prompts_dir_resolved = prompts_dir.resolve()
    except Exception:
        prompts_dir_resolved = None

    try:
        ext = normalize_audio_extension(src_path.name)
    except AudioValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    target_name = f"{role_id}_{ts}{ext}"
    target_path = prompts_dir / target_name

    try:
        if prompts_dir_resolved is not None:
            rp = target_path.resolve()
            if not rp.is_relative_to(prompts_dir_resolved):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="角色参考音频保存路径非法",
                )
        shutil.copy2(src_path, target_path)
        with target_path.open("rb+") as copied:
            os.fsync(copied.fileno())
    except HTTPException:
        raise
    except Exception as exc:
        try:
            target_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"复制音色参考音频失败: {exc}",
        ) from exc

    rel_path: str | None = str(Path("roles") / "prompts" / target_name)

    try:
        result = commit_role_synthesis_update(
            project_id,
            role_id,
            {
                "speaker_audio": rel_path,
                "speaker_voice_id": payload.voice_id,
            },
        )
        _cleanup_committed_role_prompts(project_id, prompts_dir, role_id)
        _delete_invalidated_segment_outputs(project_id, result.segment_audio_to_delete)
        return result.role
    except (LookupError, ProjectStorageError) as exc:
        try:
            target_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        try:
            target_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存角色音色绑定失败: {exc}",
        ) from exc


@router.get("/{project_id}/roles", response_model=List[Role])
async def api_list_roles(project_id: str) -> List[Role]:
    """列出项目下的所有角色。"""

    try:
        project = load_project(project_id)
    except ProjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    roles: List[Role] = project.roles or []

    # 兼容历史项目：过去绑定音色库时未持久化 speaker_voice_id，导致前端只能显示路径。
    # 这里按音频内容哈希做一次回填：若角色参考音频与某个音色库音频完全一致，则写回该 voice_id。
    try:
        need_backfill = any(
            (getattr(r, "speaker_audio", None) and not getattr(r, "speaker_voice_id", None))
            for r in roles
        )
    except Exception:
        need_backfill = False

    if need_backfill:
        try:
            voices = load_voices()
        except VoiceStorageError:
            voices = []

        voice_hash_map: dict[str, str] = {}
        base_dir = voices_audio_dir()
        try:
            base_dir_resolved = base_dir.resolve()
        except Exception:
            base_dir_resolved = None
        for v in voices:
            if not v or not getattr(v, "audio_path", None):
                continue
            try:
                vp = Path(v.audio_path)
                if not vp.is_absolute():
                    vp = (base_dir / str(v.audio_path)).resolve()
                else:
                    vp = vp.resolve()
            except Exception:
                continue

            if base_dir_resolved is None or not vp.is_relative_to(base_dir_resolved):
                continue
            if not vp.is_file():
                continue
            try:
                voice_hash_map[v.id] = _sha256_file(vp)
            except Exception:
                continue

        project_root = _project_root(project_id)
        changed = False

        for role in roles:
            if not getattr(role, "speaker_audio", None):
                continue
            if getattr(role, "speaker_voice_id", None):
                continue

            try:
                project_root_resolved = project_root.resolve()
                rp = (project_root / str(role.speaker_audio)).resolve()
            except Exception:
                continue
            if not rp.is_relative_to(project_root_resolved):
                continue
            if not rp.is_file():
                continue
            try:
                role_hash = _sha256_file(rp)
            except Exception:
                continue

            matched_voice_id = next(
                (vid for vid, vh in voice_hash_map.items() if vh == role_hash),
                None,
            )
            if matched_voice_id:
                role.speaker_voice_id = matched_voice_id
                changed = True

        if changed:
            try:
                project.updated_at = datetime.now(timezone.utc)
            except Exception:
                pass
            save_project(project)

    return roles


@router.post(
    "/{project_id}/roles/{role_id}/speaker-audio",
    response_model=Role,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_upload_role_speaker_audio(
    project_id: str,
    role_id: str,
    file: UploadFile = File(..., description="角色参考音频文件，建议 3-10 秒"),
) -> Role:
    """上传并绑定角色参考音频。

    音频文件会被保存到项目目录下的 `roles/prompts/` 子目录中，
    并在角色对象中记录相对路径，后续调用 TTS 时会使用该路径。
    """

    if Path(role_id).name != role_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法角色 ID",
        )

    project_root = _project_root(project_id)
    prompts_dir = project_root / "roles" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or f"{role_id}.wav"
    try:
        ext = normalize_audio_extension(filename)
    except AudioValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    # 使用带时间戳的文件名，避免复用旧的参考音频路径导致下游 TTS 缓存错误地复用旧音色
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    target_name = f"{role_id}_{ts}{ext}"
    target_path = prompts_dir / target_name

    # 读取上传大小限制（默认 20MB）
    max_bytes = _get_max_upload_bytes()

    written = 0
    try:
        with target_path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="上传文件过大",
                    )
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            target_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    try:
        await _require_decodable_audio(target_path)
    except HTTPException:
        try:
            target_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    rel_path = str(Path("roles") / "prompts" / target_name)

    try:
        result = commit_role_synthesis_update(
            project_id,
            role_id,
            {
                "speaker_audio": rel_path,
                "speaker_voice_id": None,
            },
        )
        _cleanup_committed_role_prompts(project_id, prompts_dir, role_id)
        _delete_invalidated_segment_outputs(project_id, result.segment_audio_to_delete)
        return result.role
    except (LookupError, ProjectStorageError) as exc:
        try:
            target_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        try:
            target_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存角色参考音频失败: {exc}",
        ) from exc
