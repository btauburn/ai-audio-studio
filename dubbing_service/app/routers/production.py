"""Pronunciation, four-track sound design, QC and release-package APIs."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import ffmpeg  # type: ignore
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dubbing_service.app.core.audio_validation import (
    AudioProbeUnavailableError,
    AudioValidationError,
    normalize_audio_extension,
    validate_decodable_audio_async,
)
from dubbing_service.app.core.helpers import get_max_upload_bytes, project_root, sha256_file
from dubbing_service.app.models.production import (
    AssetLicense,
    AudioAsset,
    ProductionSettings,
    PronunciationEntry,
    QualityReport,
    ReleaseMetadata,
    ReleasePackage,
    SoundDesign,
    SoundTrack,
)
from dubbing_service.app.routers.operation_guards import project_mutation_guard
from dubbing_service.app.routers.synthesis import _probe_audio_duration_seconds
from dubbing_service.app.services.production import run_project_quality_checks
from dubbing_service.app.storage.masters import MasterStorageError, load_master
from dubbing_service.app.storage.production import (
    ProductionStorageError,
    load_assets,
    load_releases,
    load_settings,
    production_lock,
    production_root,
    save_assets,
    save_releases,
    save_settings,
)
from dubbing_service.app.storage.projects import load_project

router = APIRouter()


class PronunciationPayload(BaseModel):
    entries: List[PronunciationEntry] = Field(default_factory=list, max_length=2000)


class ContentProfilePayload(BaseModel):
    content_mode: str = Field(pattern="^(story|information|knowledge|podcast)$")


def _root(project_id: str) -> Path:
    root = project_root(project_id, ensure_dir=False)
    if not (root / "project.json").is_file():
        raise HTTPException(status_code=404, detail="项目不存在")
    return root


def _safe_file(root: Path, rel: str, prefix: str) -> Path:
    try:
        target = (root / rel).resolve()
        base = root.resolve()
        relative = target.relative_to(base).as_posix()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="文件路径非法") from exc
    if not relative.startswith(prefix.rstrip("/") + "/") or not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return target


@router.get("/projects/{project_id}/production", response_model=ProductionSettings)
async def api_get_production(project_id: str) -> ProductionSettings:
    _root(project_id)
    return load_settings(project_id)


@router.put(
    "/projects/{project_id}/production/profile",
    response_model=ProductionSettings,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_put_content_profile(project_id: str, payload: ContentProfilePayload) -> ProductionSettings:
    _root(project_id)
    with production_lock(project_id):
        settings = load_settings(project_id)
        settings.content_mode = payload.content_mode
        settings.updated_at = datetime.now(timezone.utc)
        save_settings(project_id, settings)
    return settings


@router.put(
    "/projects/{project_id}/production/pronunciations",
    response_model=ProductionSettings,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_put_pronunciations(project_id: str, payload: PronunciationPayload) -> ProductionSettings:
    _root(project_id)
    ids = [item.id for item in payload.entries]
    terms = [item.source.casefold() for item in payload.entries]
    if len(ids) != len(set(ids)) or len(terms) != len(set(terms)):
        raise HTTPException(status_code=400, detail="发音词条 ID 或原词不能重复")
    with production_lock(project_id):
        settings = load_settings(project_id)
        settings.pronunciation_dictionary = payload.entries
        settings.updated_at = datetime.now(timezone.utc)
        save_settings(project_id, settings)
    return settings


@router.put(
    "/projects/{project_id}/production/sound-design",
    response_model=ProductionSettings,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_put_sound_design(project_id: str, payload: SoundDesign) -> ProductionSettings:
    _root(project_id)
    with production_lock(project_id):
        assets = {item.id: item for item in load_assets(project_id)}
        for clip in payload.clips:
            asset = assets.get(clip.asset_id)
            if asset is None or asset.track != clip.track:
                raise HTTPException(status_code=400, detail=f"片段 {clip.id} 引用的素材不存在或轨道不匹配")
        settings = load_settings(project_id)
        payload.version = max(settings.sound_design.version + 1, payload.version)
        payload.updated_at = datetime.now(timezone.utc)
        settings.sound_design = payload
        settings.updated_at = datetime.now(timezone.utc)
        save_settings(project_id, settings)
    return settings


@router.put(
    "/projects/{project_id}/production/release-metadata",
    response_model=ProductionSettings,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_put_release_metadata(project_id: str, payload: ReleaseMetadata) -> ProductionSettings:
    root = _root(project_id)
    if payload.cover_file:
        _safe_file(root, payload.cover_file, "production/cover")
    with production_lock(project_id):
        settings = load_settings(project_id)
        settings.release = payload
        settings.updated_at = datetime.now(timezone.utc)
        save_settings(project_id, settings)
    return settings


async def _save_upload(upload: UploadFile, staged: Path, max_bytes: int) -> int:
    total = 0
    with staged.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(status_code=413, detail="上传文件过大")
            handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    return total


@router.post(
    "/projects/{project_id}/production/assets",
    response_model=AudioAsset,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_upload_asset(
    project_id: str,
    file: UploadFile = File(...),
    track: SoundTrack = Form(...),
    name: str = Form(...),
    tags: str = Form(""),
    source: str = Form(""),
    author: str = Form(""),
    license_name: str = Form(""),
    commercial_use: bool | None = Form(None),
    proof_note: str = Form(""),
) -> AudioAsset:
    _root(project_id)
    extension = normalize_audio_extension(file.filename or "")
    if extension not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}:
        raise HTTPException(status_code=400, detail="仅支持 WAV/MP3/FLAC/OGG/M4A/AAC 音频")
    asset_id = uuid.uuid4().hex[:16]
    directory = production_root(project_id, ensure=True) / "assets" / track.value
    fd, staged_name = tempfile.mkstemp(dir=directory, prefix="upload_", suffix=extension)
    os.close(fd)
    staged = Path(staged_name)
    final = directory / f"{asset_id}{extension}"
    try:
        if await _save_upload(file, staged, get_max_upload_bytes(default=500 * 1024 * 1024)) <= 0:
            raise HTTPException(status_code=400, detail="上传文件为空")
        try:
            await validate_decodable_audio_async(staged)
        except AudioValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AudioProbeUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        duration = _probe_audio_duration_seconds(staged)
        if duration <= 0:
            raise HTTPException(status_code=400, detail="无法读取素材时长")
        os.replace(staged, final)
        root = _root(project_id)
        asset = AudioAsset(
            id=asset_id,
            track=track,
            name=name.strip() or Path(file.filename or "素材").stem,
            audio_file=str(final.relative_to(root)),
            audio_sha256=sha256_file(final),
            duration_seconds=duration,
            tags=[item.strip() for item in tags.split(",") if item.strip()][:30],
            license=AssetLicense(
                source=source.strip() or None,
                author=author.strip() or None,
                license_name=license_name.strip() or None,
                commercial_use=commercial_use,
                proof_note=proof_note.strip() or None,
            ),
        )
        with production_lock(project_id):
            assets = load_assets(project_id)
            assets.append(asset)
            save_assets(project_id, assets)
        return asset
    except Exception:
        staged.unlink(missing_ok=True)
        if final.is_file() and not any(item.id == asset_id for item in load_assets(project_id)):
            final.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@router.get("/projects/{project_id}/production/assets", response_model=List[AudioAsset])
async def api_list_assets(project_id: str) -> List[AudioAsset]:
    _root(project_id)
    return load_assets(project_id)


@router.get("/projects/{project_id}/production/assets/{asset_id}/audio")
async def api_get_asset_audio(project_id: str, asset_id: str) -> FileResponse:
    root = _root(project_id)
    asset = next((item for item in load_assets(project_id) if item.id == asset_id), None)
    if asset is None:
        raise HTTPException(status_code=404, detail="素材不存在")
    path = _safe_file(root, asset.audio_file, "production/assets")
    if sha256_file(path) != asset.audio_sha256:
        raise HTTPException(status_code=409, detail="素材校验失败")
    return FileResponse(path, filename=path.name)


@router.delete(
    "/projects/{project_id}/production/assets/{asset_id}",
    dependencies=[Depends(project_mutation_guard)],
)
async def api_delete_asset(project_id: str, asset_id: str) -> dict[str, bool]:
    root = _root(project_id)
    with production_lock(project_id):
        settings = load_settings(project_id)
        if any(item.asset_id == asset_id for item in settings.sound_design.clips):
            raise HTTPException(status_code=409, detail="素材仍被当前时间线引用，请先移除片段")
        assets = load_assets(project_id)
        asset = next((item for item in assets if item.id == asset_id), None)
        if asset is None:
            raise HTTPException(status_code=404, detail="素材不存在")
        # Immutable Masters carry the asset hash in their sound-design snapshot.
        from dubbing_service.app.storage.masters import list_masters
        if any(asset_id in json.dumps(item.sound_design_snapshot or {}) for item in list_masters(project_id)):
            raise HTTPException(status_code=409, detail="素材已被正式 Master 引用，不能删除")
        save_assets(project_id, [item for item in assets if item.id != asset_id])
    path = _safe_file(root, asset.audio_file, "production/assets")
    path.unlink(missing_ok=True)
    return {"success": True}


@router.post(
    "/projects/{project_id}/production/cover",
    response_model=ProductionSettings,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_upload_cover(project_id: str, file: UploadFile = File(...)) -> ProductionSettings:
    root = _root(project_id)
    extension = Path(file.filename or "").suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=400, detail="封面仅支持 JPG、PNG 或 WebP")
    directory = production_root(project_id, ensure=True) / "cover"
    fd, staged_name = tempfile.mkstemp(dir=directory, prefix="cover_", suffix=extension)
    os.close(fd)
    staged = Path(staged_name)
    final = directory / f"cover{extension}"
    try:
        size = await _save_upload(file, staged, 20 * 1024 * 1024)
        if size < 32:
            raise HTTPException(status_code=400, detail="封面文件无效")
        magic = staged.read_bytes()[:12]
        if not (magic.startswith(b"\xff\xd8\xff") or magic.startswith(b"\x89PNG\r\n\x1a\n") or magic.startswith(b"RIFF") and magic[8:12] == b"WEBP"):
            raise HTTPException(status_code=400, detail="封面文件内容与扩展名不符")
        os.replace(staged, final)
        with production_lock(project_id):
            settings = load_settings(project_id)
            old_cover = settings.release.cover_file
            settings.release.cover_file = str(final.relative_to(root))
            settings.updated_at = datetime.now(timezone.utc)
            save_settings(project_id, settings)
        if old_cover and old_cover != settings.release.cover_file:
            try:
                _safe_file(root, old_cover, "production/cover").unlink(missing_ok=True)
            except HTTPException:
                pass
        return settings
    finally:
        staged.unlink(missing_ok=True)
        await file.close()


@router.get("/projects/{project_id}/quality-check", response_model=QualityReport)
async def api_quality_check(project_id: str) -> QualityReport:
    return run_project_quality_checks(project_id, _root(project_id))


def _subtitle_timestamp(seconds: float, *, vtt: bool = False) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, rem = divmod(milliseconds, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    separator = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _timeline_subtitles(timeline: dict, *, vtt: bool = False) -> str:
    blocks = ["WEBVTT", ""] if vtt else []
    for index, item in enumerate(timeline.get("items") or [], 1):
        if not vtt:
            blocks.append(str(index))
        blocks.append(
            f"{_subtitle_timestamp(float(item['start']), vtt=vtt)} --> "
            f"{_subtitle_timestamp(float(item['end']), vtt=vtt)}"
        )
        blocks.extend([str(item.get("text") or ""), ""])
    return "\n".join(blocks)


@router.post(
    "/projects/{project_id}/release-packages/{master_id}",
    response_model=ReleasePackage,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_create_release_package(project_id: str, master_id: str) -> ReleasePackage:
    root = _root(project_id)
    try:
        master = load_master(project_id, master_id)
    except MasterStorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    master_path = _safe_file(root, master.audio_file, "audio/masters")
    if sha256_file(master_path) != master.audio_sha256:
        raise HTTPException(status_code=409, detail="Master 缺失或校验失败")
    frozen_quality = master.quality_report or {}
    if frozen_quality and frozen_quality.get("passed") is False:
        raise HTTPException(status_code=400, detail="该 Master 的冻结质检未通过，不能生成发布包")
    settings = load_settings(project_id)
    release_id = uuid.uuid4().hex[:16]
    release_dir = production_root(project_id, ensure=True) / "releases"
    staging = Path(tempfile.mkdtemp(dir=release_dir, prefix=f"{release_id}_"))
    archive = release_dir / f"release_{release_id}.zip"
    try:
        wav_path = staging / "master.wav"
        shutil.copy2(master_path, wav_path)
        mp3_path = staging / "publish.mp3"
        ffmpeg.run(
            ffmpeg.input(str(master_path)).output(str(mp3_path), acodec="libmp3lame", audio_bitrate="192k"),
            overwrite_output=True, capture_stdout=True, capture_stderr=True,
        )
        timeline = {"items": []}
        if master.dialogue_timeline_file:
            timeline_path = _safe_file(root, master.dialogue_timeline_file, "audio/masters")
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        (staging / "subtitles.srt").write_text(_timeline_subtitles(timeline), encoding="utf-8-sig")
        (staging / "subtitles.vtt").write_text(_timeline_subtitles(timeline, vtt=True), encoding="utf-8")
        project = load_project(project_id)
        chapters = [item.model_dump(mode="json") for item in project.chapters]
        (staging / "chapters.json").write_text(json.dumps(chapters, ensure_ascii=False, indent=2), encoding="utf-8")
        metadata = settings.release.model_dump(mode="json")
        metadata.update({"project_id": project_id, "master_id": master_id, "master_name": master.name})
        (staging / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        credits = {
            "tts": [item.model_dump(mode="json") for item in master.segments],
            "sound_design": master.sound_design_snapshot,
            "quality_report": master.quality_report,
        }
        (staging / "credits_and_licenses.json").write_text(
            json.dumps(credits, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        if settings.release.cover_file:
            cover = _safe_file(root, settings.release.cover_file, "production/cover")
            shutil.copy2(cover, staging / f"cover{cover.suffix.lower()}")
        checksums = []
        for path in sorted(item for item in staging.iterdir() if item.is_file()):
            checksums.append(f"{sha256_file(path)}  {path.name}")
        (staging / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="ascii")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in sorted(staging.iterdir()):
                if path.is_file():
                    bundle.write(path, arcname=path.name)
        package = ReleasePackage(
            id=release_id,
            project_id=project_id,
            master_id=master_id,
            archive_file=str(archive.relative_to(root)),
            archive_sha256=sha256_file(archive),
            files=sorted(path.name for path in staging.iterdir() if path.is_file()),
        )
        with production_lock(project_id):
            releases = load_releases(project_id)
            releases.append(package)
            save_releases(project_id, releases)
        return package
    except ffmpeg.Error as exc:  # type: ignore[attr-defined]
        archive.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="发布版 MP3 转码失败") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)


@router.get("/projects/{project_id}/release-packages", response_model=List[ReleasePackage])
async def api_list_release_packages(project_id: str) -> List[ReleasePackage]:
    _root(project_id)
    return load_releases(project_id)


@router.get("/projects/{project_id}/release-packages/{release_id}/download")
async def api_download_release_package(project_id: str, release_id: str) -> FileResponse:
    root = _root(project_id)
    package = next((item for item in load_releases(project_id) if item.id == release_id), None)
    if package is None:
        raise HTTPException(status_code=404, detail="发布包不存在")
    path = _safe_file(root, package.archive_file, "production/releases")
    if sha256_file(path) != package.archive_sha256:
        raise HTTPException(status_code=409, detail="发布包校验失败")
    return FileResponse(path, media_type="application/zip", filename=path.name)
