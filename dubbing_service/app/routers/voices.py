"""Voice (tone) library management APIs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import uuid
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dubbing_service.app.core.audio_validation import (
    AudioProbeUnavailableError,
    AudioValidationError,
    UPLOAD_AUDIO_EXTENSIONS,
    normalize_audio_extension,
    validate_decodable_audio_async,
)
from dubbing_service.app.core.config import get_config
from dubbing_service.app.core.helpers import get_max_upload_bytes as _get_max_upload_bytes
from dubbing_service.app.core.path_safety import validate_simple_component_id
from dubbing_service.app.storage.projects import (
    ProjectStorageError,
    load_project,
)
from dubbing_service.app.storage.segments import SegmentStorageError, load_segments

from dubbing_service.app.storage.voices import (
    Voice,
    VoiceStorageError,
    generate_voice_id,
    load_voices,
    save_voices,
    voices_audio_dir,
    voices_lock,
)


logger = logging.getLogger(__name__)


router = APIRouter()


class VoicePayload(BaseModel):
    """创建/更新音色元数据的请求体。"""

    name: str = Field(..., min_length=1, max_length=100, description="音色名称")
    tags: list[str] = Field(
        default_factory=list,
        description="标签列表，例如 ['女', '青年']",
    )
    description: str | None = Field(
        default=None,
        max_length=500,
        description="可选：备注说明，例如适合什么场景",
    )


class VoiceResponse(BaseModel):
    id: str
    name: str
    tags: list[str] = Field(default_factory=list)
    audio_path: str | None = None
    description: str | None = None
    created_at: datetime
    updated_at: datetime


def _normalize_tags(tags: list[str]) -> list[str]:
    return sorted({t.strip() for t in tags if t and t.strip()})


@dataclass(frozen=True)
class _VoiceDependencies:
    role_count: int
    emotion_count: int
    projects: tuple[dict[str, object], ...]
    emotion_filenames: frozenset[str]


def _validate_voice_id(voice_id: str) -> None:
    try:
        validate_simple_component_id(voice_id, label="voice_id")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法音色 ID",
        ) from exc


def _voice_relative_filename(raw_path: object, voices_root: Path) -> str | None:
    """Return a direct voices-directory filename for a persisted reference."""

    value = str(raw_path or "").strip()
    if not value:
        return None
    try:
        candidate = Path(value)
        if candidate.is_absolute():
            resolved = candidate.resolve()
            root = voices_root.resolve()
            if resolved.is_relative_to(root) and resolved.parent == root:
                return resolved.name
            return None
    except (OSError, ValueError):
        return None

    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    if len(parts) == 1 and parts[0] not in {"", ".", ".."}:
        return parts[0]
    if len(parts) == 2 and parts[0] == "voices" and parts[1] not in {"", ".", ".."}:
        return parts[1]
    return None


def _scan_voice_dependencies(
    voice_id: str,
    *,
    owned_audio_names: set[str],
) -> _VoiceDependencies:
    """Fail closed while scanning every readable project for voice references."""

    root = get_config().projects_root_abs
    voices_root = voices_audio_dir()
    role_count = 0
    emotion_count = 0
    projects: list[dict[str, object]] = []
    emotion_filenames: set[str] = set()

    try:
        project_dirs = sorted(
            item for item in root.iterdir() if item.is_dir() and (item / "project.json").is_file()
        ) if root.is_dir() else []
    except OSError as exc:
        raise RuntimeError(f"无法扫描项目目录: {exc}") from exc

    for project_dir in project_dirs:
        project_id = project_dir.name
        try:
            project = load_project(project_id)
            segments = load_segments(project_id)
        except (ProjectStorageError, SegmentStorageError, OSError, ValueError) as exc:
            raise RuntimeError(f"无法完整扫描项目 {project_id}: {exc}") from exc

        project_role_count = sum(
            1
            for role in (project.roles or [])
            if str(getattr(role, "speaker_voice_id", "") or "").strip() == voice_id
        )
        project_emotion_count = 0
        for segment in segments:
            filename = _voice_relative_filename(segment.emotion_audio, voices_root)
            if filename is None or filename not in owned_audio_names:
                continue
            project_emotion_count += 1
            emotion_filenames.add(filename)

        if project_role_count or project_emotion_count:
            projects.append(
                {
                    "id": project.id,
                    "name": project.name,
                    "role_dependencies": project_role_count,
                    "emotion_dependencies": project_emotion_count,
                }
            )
        role_count += project_role_count
        emotion_count += project_emotion_count

    return _VoiceDependencies(
        role_count=role_count,
        emotion_count=emotion_count,
        projects=tuple(projects),
        emotion_filenames=frozenset(emotion_filenames),
    )


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


@router.get("/voices", response_model=list[VoiceResponse])
async def api_list_voices() -> list[VoiceResponse]:
    try:
        voices = load_voices()
    except VoiceStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return [VoiceResponse(**v.model_dump()) for v in voices]


@router.post("/voices", response_model=VoiceResponse)
async def api_create_voice(payload: VoicePayload) -> VoiceResponse:
    with voices_lock():
        try:
            voices = load_voices()
        except VoiceStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        now = datetime.now(timezone.utc)
        voice = Voice(
            id=generate_voice_id(),
            name=payload.name.strip(),
            tags=_normalize_tags(payload.tags),
            audio_path=None,
            description=payload.description.strip() if payload.description else None,
            created_at=now,
            updated_at=now,
        )

        voices.append(voice)

        try:
            save_voices(voices)
        except VoiceStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    return VoiceResponse(**voice.model_dump())


@router.put("/voices/{voice_id}", response_model=VoiceResponse)
async def api_update_voice(voice_id: str, payload: VoicePayload) -> VoiceResponse:
    _validate_voice_id(voice_id)

    with voices_lock():
        try:
            voices = load_voices()
        except VoiceStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        target: Voice | None = None
        for v in voices:
            if v.id == voice_id:
                target = v
                break

        if target is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="指定的音色不存在",
            )

        target.name = payload.name.strip()
        target.tags = _normalize_tags(payload.tags)
        target.description = payload.description.strip() if payload.description else None
        target.updated_at = datetime.now(timezone.utc)

        try:
            save_voices(voices)
        except VoiceStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    return VoiceResponse(**target.model_dump())


@router.delete("/voices/{voice_id}")
async def api_delete_voice(voice_id: str) -> dict[str, bool]:
    _validate_voice_id(voice_id)

    with voices_lock():
        try:
            voices = load_voices()
        except VoiceStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        new_voices: list[Voice] = []
        target: Voice | None = None
        for v in voices:
            if v.id == voice_id:
                target = v
                continue
            new_voices.append(v)

        if target is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="指定的音色不存在",
            )

        audio_dir = voices_audio_dir()
        owned_audio_names: set[str] = {
            f"{voice_id}{extension}" for extension in UPLOAD_AUDIO_EXTENSIONS
        }
        owned_audio_names.update({
            path.name
            for path in audio_dir.glob(f"{voice_id}.*")
            if path.is_file() and path.parent.resolve() == audio_dir.resolve()
        })
        current_name = _voice_relative_filename(target.audio_path, audio_dir)
        if current_name:
            owned_audio_names.add(current_name)

        try:
            dependencies = _scan_voice_dependencies(
                voice_id,
                owned_audio_names=owned_audio_names,
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"删除前无法完成依赖扫描，音色未删除: {exc}",
            ) from exc

        if dependencies.role_count or dependencies.emotion_count:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": "音色仍被项目引用，请先解除角色绑定和情感音频引用",
                    "voice_id": voice_id,
                    "role_dependency_count": dependencies.role_count,
                    "emotion_dependency_count": dependencies.emotion_count,
                    "projects": list(dependencies.projects),
                },
            )

        try:
            save_voices(new_voices)
        except VoiceStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        # Metadata no longer references this voice. File cleanup is now safe;
        # failures only leave removable orphans.
        for filename in owned_audio_names:
            try:
                candidate = (audio_dir / filename).resolve()
                if candidate.parent == audio_dir.resolve() and candidate.is_file():
                    candidate.unlink(missing_ok=True)
            except Exception:
                logger.debug("删除音色关联音频失败: voice_id=%s", voice_id, exc_info=True)

    return {"success": True}


@router.post("/voices/{voice_id}/audio", response_model=VoiceResponse)
async def api_upload_voice_audio(voice_id: str, file: UploadFile = File(...)) -> VoiceResponse:
    """上传或替换指定音色的参考音频文件。"""

    _validate_voice_id(voice_id)

    try:
        voices = load_voices()
    except VoiceStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    target: Voice | None = None
    for v in voices:
        if v.id == voice_id:
            target = v
            break

    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定的音色不存在",
        )

    audio_dir = voices_audio_dir()
    try:
        ext = normalize_audio_extension(file.filename or f"{voice_id}.wav")
    except AudioValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    target_path = audio_dir / f"{voice_id}{ext}"
    transaction_id = uuid.uuid4().hex
    stage_path = audio_dir / f".{voice_id}.{transaction_id}.upload{ext}"
    backup_path = audio_dir / f".{target_path.name}.{transaction_id}.backup"

    max_bytes = _get_max_upload_bytes()

    written = 0
    try:
        with stage_path.open("xb") as f:
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
            stage_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    try:
        await _require_decodable_audio(stage_path)
    except HTTPException:
        try:
            stage_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    promoted = False
    had_previous_target = False
    old_filename: str | None = None
    try:
        # Promotion and voices.json update are rollback-coupled under the same
        # lock. A metadata failure restores the exact previous target file.
        with voices_lock():
            try:
                voices = load_voices()
            except VoiceStorageError as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(exc),
                ) from exc

            target = next((voice for voice in voices if voice.id == voice_id), None)
            if target is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="指定的音色不存在",
                )

            old_filename = _voice_relative_filename(target.audio_path, audio_dir)
            try:
                if target_path.is_file():
                    os.replace(target_path, backup_path)
                    had_previous_target = True
                os.replace(stage_path, target_path)
                promoted = True
            except Exception as exc:
                if had_previous_target and backup_path.is_file() and not target_path.exists():
                    os.replace(backup_path, target_path)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"保存音色音频失败，旧文件已保留: {exc}",
                ) from exc

            target.audio_path = target_path.name
            target.updated_at = datetime.now(timezone.utc)
            try:
                save_voices(voices)
            except Exception as exc:
                rollback_error: Exception | None = None
                try:
                    if promoted:
                        target_path.unlink(missing_ok=True)
                    if had_previous_target and backup_path.is_file():
                        os.replace(backup_path, target_path)
                except Exception as rollback_exc:  # pragma: no cover - catastrophic I/O
                    rollback_error = rollback_exc
                detail = f"保存音色元数据失败，旧音频和元数据已保留: {exc}"
                if rollback_error is not None:
                    detail += f"；自动回滚文件失败，请保留备份并检查磁盘: {rollback_error}"
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=detail,
                ) from exc

            # Preserve old extensions that are still referenced directly by a
            # segment. Unreferenced historical files can be collected now that
            # the new metadata is durable.
            cleanup_candidates = {
                path.name
                for path in audio_dir.glob(f"{voice_id}.*")
                if path.is_file()
            }
            if old_filename:
                cleanup_candidates.add(old_filename)
            keep_names = {target_path.name}
            try:
                dependencies = _scan_voice_dependencies(
                    voice_id,
                    owned_audio_names=cleanup_candidates,
                )
                keep_names.update(dependencies.emotion_filenames)
            except RuntimeError:
                # A failed scan must never risk deleting a possibly referenced
                # old file. Leave all candidates as harmless orphans instead.
                keep_names.update(cleanup_candidates)
                logger.warning(
                    "替换音色后依赖扫描失败，保留旧音频: voice_id=%s",
                    voice_id,
                    exc_info=True,
                )

            for filename in cleanup_candidates - keep_names:
                try:
                    candidate = (audio_dir / filename).resolve()
                    if candidate.parent == audio_dir.resolve() and candidate.is_file():
                        candidate.unlink(missing_ok=True)
                except OSError:
                    logger.debug(
                        "清理旧音色音频失败: voice_id=%s file=%s",
                        voice_id,
                        filename,
                        exc_info=True,
                    )
    finally:
        try:
            stage_path.unlink(missing_ok=True)
        except OSError:
            pass

    try:
        backup_path.unlink(missing_ok=True)
    except OSError:
        pass
    return VoiceResponse(**target.model_dump())


@router.get("/voices/{voice_id}/audio")
async def api_get_voice_audio(voice_id: str) -> FileResponse:
    """返回指定音色的音频文件用于试听。"""

    _validate_voice_id(voice_id)

    try:
        voices = load_voices()
    except VoiceStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    target: Voice | None = None
    for v in voices:
        if v.id == voice_id:
            target = v
            break

    if target is None or not target.audio_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="音色或音频文件不存在",
        )

    base_dir = voices_audio_dir()
    try:
        base_dir_resolved = base_dir.resolve()
        audio_path = Path(target.audio_path)
        if not audio_path.is_absolute():
            audio_path = (base_dir / str(target.audio_path)).resolve()
        else:
            audio_path = audio_path.resolve()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="音频文件不存在",
        )

    if not audio_path.is_relative_to(base_dir_resolved) or not audio_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="音频文件不存在",
        )

    # 简单根据后缀推断 MIME 类型
    suffix = audio_path.suffix.lower()
    mime = "audio/wav"
    if suffix in {".mp3"}:
        mime = "audio/mpeg"
    elif suffix in {".ogg"}:
        mime = "audio/ogg"

    return FileResponse(audio_path, media_type=mime, filename=audio_path.name)
