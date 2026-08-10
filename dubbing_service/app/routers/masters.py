"""Reviewed immutable Master creation and download APIs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

import ffmpeg  # type: ignore
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dubbing_service.app.core.helpers import project_root, sha256_file
from dubbing_service.app.core.path_safety import resolve_scoped_project_asset
from dubbing_service.app.models.enums import SegmentReviewStatus
from dubbing_service.app.models.master import AudioMaster, MasterSegmentSnapshot
from dubbing_service.app.routers.operation_guards import project_export_guard
from dubbing_service.app.routers.synthesis import (
    _probe_audio_duration_seconds,
    api_export_full_project_with_edits,
)
from dubbing_service.app.services.takes import get_selected_take
from dubbing_service.app.services.production import (
    render_four_track_master,
    run_project_quality_checks,
)
from dubbing_service.app.storage.masters import (
    MasterStorageError,
    list_masters,
    load_master,
    save_master,
)
from dubbing_service.app.storage.segments import load_segments
from dubbing_service.app.storage.production import load_assets, load_settings
from dubbing_service.app.storage.projects import load_project

router = APIRouter()


class CreateMasterRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=120)


def _production_inputs_payload(settings, assets) -> dict:
    settings_data = settings.model_dump(mode="json")
    settings_data.pop("updated_at", None)
    if isinstance(settings_data.get("sound_design"), dict):
        settings_data["sound_design"].pop("updated_at", None)
    return {
        "settings": settings_data,
        "assets": [item.model_dump(mode="json") for item in assets],
    }


def _master_inputs(project_id: str) -> tuple[list[MasterSegmentSnapshot], str]:
    segments = load_segments(project_id)
    project = load_project(project_id)
    if not segments:
        raise HTTPException(status_code=400, detail="项目没有可制作 Master 的分段")
    snapshots = []
    failures = []
    for segment in sorted(segments, key=lambda item: item.index):
        if segment.review_status != SegmentReviewStatus.APPROVED:
            failures.append(f"第 {segment.index + 1} 段尚未通过审听")
            continue
        take = get_selected_take(segment)
        if take is None or take.missing:
            failures.append(f"第 {segment.index + 1} 段缺少可用最终 Take")
            continue
        unresolved = [issue for issue in segment.review_issues if not issue.resolved]
        if unresolved:
            failures.append(f"第 {segment.index + 1} 段仍有未解决问题")
            continue
        snapshots.append(
            MasterSegmentSnapshot(
                segment_id=segment.id,
                index=segment.index,
                text=segment.text,
                role_id=segment.role_id,
                take_id=take.id,
                take_audio_sha256=take.audio_sha256,
                tts_backend=take.tts_backend,
                qwen3_speaker=take.qwen3_speaker,
                qwen3_language=take.qwen3_language,
                voice_license_snapshot=(
                    {
                        "source": role.voice_source,
                        "license_name": role.voice_license_name,
                        "commercial_use": role.voice_commercial_use,
                        "rights_note": role.voice_rights_note,
                    }
                    if (role := next((item for item in project.roles if item.id == segment.role_id), None))
                    else None
                ),
                edit_state=(
                    take.edit_state.model_copy(deep=True)
                    if take.edit_state is not None
                    else None
                ),
            )
        )
    if failures:
        raise HTTPException(status_code=400, detail="；".join(failures[:10]))
    payload = [item.model_dump(mode="json") for item in snapshots]
    fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return snapshots, fingerprint


@router.post(
    "/projects/{project_id}/masters",
    response_model=AudioMaster,
    dependencies=[Depends(project_export_guard)],
)
async def api_create_master(project_id: str, payload: CreateMasterRequest) -> AudioMaster:
    snapshots, fingerprint = _master_inputs(project_id)
    root = project_root(project_id, ensure_dir=False)
    quality = run_project_quality_checks(project_id, root)
    if not quality.passed:
        errors = [item.message for item in quality.issues if item.severity == "error"]
        raise HTTPException(status_code=400, detail="；".join(errors[:10]))
    settings = load_settings(project_id)
    assets = load_assets(project_id)
    production_payload = _production_inputs_payload(settings, assets)
    production_fingerprint = hashlib.sha256(
        json.dumps(production_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    fingerprint = hashlib.sha256(f"{fingerprint}:{production_fingerprint}".encode()).hexdigest()
    export = api_export_full_project_with_edits(project_id)
    source_audio = (root / export.output_file).resolve()
    if not source_audio.is_relative_to(root.resolve()) or not source_audio.is_file():
        raise HTTPException(status_code=500, detail="Master 临时导出文件不存在")

    # Detect review/selection/edit changes that happened while FFmpeg rendered.
    latest_snapshots, latest_segment_fingerprint = _master_inputs(project_id)
    latest_settings = load_settings(project_id)
    latest_assets = load_assets(project_id)
    latest_production_payload = _production_inputs_payload(latest_settings, latest_assets)
    latest_production_fingerprint = hashlib.sha256(
        json.dumps(latest_production_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    latest_fingerprint = hashlib.sha256(
        f"{latest_segment_fingerprint}:{latest_production_fingerprint}".encode()
    ).hexdigest()
    if latest_fingerprint != fingerprint:
        raise HTTPException(status_code=409, detail="制作 Master 期间项目发生变化，请重新生成")

    master_id = uuid.uuid4().hex[:16]
    audio_rel = Path("audio") / "masters" / f"{master_id}.wav"
    audio_path = root / audio_rel
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    fd, staged_name = tempfile.mkstemp(dir=audio_path.parent, prefix="master_", suffix=".wav")
    os.close(fd)
    staged = Path(staged_name)
    timeline_rel: str | None = None
    try:
        sound_snapshot = render_four_track_master(
            dialogue_path=source_audio,
            output_path=staged,
            root=root,
            settings=latest_settings,
            assets=latest_assets,
            dialogue_duration=_probe_audio_duration_seconds(source_audio),
        )
        os.replace(staged, audio_path)
        if export.dialogue_timeline_file:
            source_timeline = (root / export.dialogue_timeline_file).resolve()
            if source_timeline.is_relative_to(root.resolve()) and source_timeline.is_file():
                timeline_path = root / "audio" / "masters" / f"{master_id}.timeline.json"
                shutil.copy2(source_timeline, timeline_path)
                timeline_rel = str(timeline_path.relative_to(root))
        master = AudioMaster(
            id=master_id,
            project_id=project_id,
            name=(payload.name or "").strip() or f"Master {master_id[:8]}",
            audio_file=str(audio_rel),
            audio_sha256=sha256_file(audio_path),
            dialogue_timeline_file=timeline_rel,
            duration_seconds=_probe_audio_duration_seconds(audio_path),
            input_fingerprint=fingerprint,
            segments=latest_snapshots,
            sound_design_snapshot=sound_snapshot,
            quality_report=quality.model_dump(mode="json"),
        )
        save_master(master)
        return master
    except Exception:
        staged.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)
        if timeline_rel:
            (root / timeline_rel).unlink(missing_ok=True)
        raise


@router.get("/projects/{project_id}/masters", response_model=List[AudioMaster])
async def api_list_masters(project_id: str) -> List[AudioMaster]:
    try:
        return list_masters(project_id)
    except MasterStorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/projects/{project_id}/masters/{master_id}/audio")
async def api_download_master(project_id: str, master_id: str) -> FileResponse:
    try:
        master = load_master(project_id, master_id)
    except MasterStorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    root = project_root(project_id, ensure_dir=False)
    try:
        path = resolve_scoped_project_asset(
            root,
            master.audio_file,
            required_subdir="audio/masters",
            allowed_extensions={".wav"},
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Master 音频路径非法") from exc
    if not path.is_file() or sha256_file(path) != master.audio_sha256:
        raise HTTPException(status_code=409, detail="Master 音频缺失或校验失败")
    return FileResponse(path, media_type="audio/wav", filename=path.name)
