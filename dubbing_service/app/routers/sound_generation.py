"""SoundPlan, Stable Audio candidates, semantic adoption and preview APIs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from dubbing_service.app.core.helpers import project_root, sha256_file
from dubbing_service.app.core.path_safety import resolve_scoped_project_asset
from dubbing_service.app.core.sfx_worker import SFXWorkerError, worker_status
from dubbing_service.app.core.tts_operations import (
    release_gpu_operation,
    try_acquire_gpu_operation,
)
from dubbing_service.app.models.production import AssetLicense, AudioAsset, SoundDesign
from dubbing_service.app.models.sound_generation import (
    SemanticPlacement,
    SoundCandidateStatus,
    SoundCue,
    SoundCueReviewStatus,
    SoundPlan,
    SoundPlanDensity,
)
from dubbing_service.app.models.sound_jobs import SoundGenerationJob
from dubbing_service.app.providers.llm import OpenAIStyleLLMProvider
from dubbing_service.app.routers.operation_guards import project_export_guard, project_mutation_guard
from dubbing_service.app.routers.synthesis import (
    _probe_audio_duration_seconds,
    api_export_full_project_with_edits,
)
from dubbing_service.app.services.production import (
    _current_stable_audio_confirmation,
    render_four_track_master,
    run_project_quality_checks,
    stable_audio_license_snapshot_matches_current,
)
from dubbing_service.app.services.sound_generation import (
    SoundCandidateAdoptionError,
    adopt_sound_candidate,
    fingerprint_segment_source,
    fingerprint_sound_cue,
    fingerprint_sound_source,
    refresh_sound_plan_staleness,
)
from dubbing_service.app.services.sound_jobs import (
    SoundJobError,
    launch_sound_job,
    prepare_sound_job,
    request_sound_job_cancel,
)
from dubbing_service.app.services.sound_planner import (
    SoundPlannerError,
    default_sound_density,
    draft_sound_plan,
)
from dubbing_service.app.storage.llm_configs import LLMConfigStorageError, load_llm_configs
from dubbing_service.app.storage.llm_providers import LLMProviderStorageError, load_llm_providers
from dubbing_service.app.storage.production import (
    ProductionStorageError,
    load_assets,
    load_settings,
    production_lock,
    production_root,
    save_assets,
    save_settings,
)
from dubbing_service.app.storage.projects import ProjectStorageError, load_project
from dubbing_service.app.storage.segments import load_segments
from dubbing_service.app.storage.sound_generation import (
    SoundGenerationStorageError,
    delete_sound_plan,
    list_sound_plans,
    load_sound_plan,
    save_sound_plan,
    sound_plan_lock,
)
from dubbing_service.app.storage.sound_jobs import (
    SoundJobStorageError,
    generate_sound_job_id,
    list_sound_jobs,
    load_sound_job,
)


router = APIRouter()
REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "models" / "manifests" / "stable_audio_3_small_sfx.json"


class DraftSoundPlanRequest(BaseModel):
    density: SoundPlanDensity | None = None
    llm_config_id: str | None = Field(default=None, max_length=160)


class CueUpdateRequest(BaseModel):
    track: Literal["sfx", "ambience"] | None = None
    description_cn: str | None = Field(default=None, max_length=1000)
    prompt: str | None = Field(default=None, min_length=1, max_length=1000)
    placement: SemanticPlacement | None = None
    requested_duration_seconds: float | None = None
    playback_duration_seconds: float | None = None
    gain_db: float | None = Field(default=None, ge=-60.0, le=18.0)
    fade_in: float | None = Field(default=None, ge=0.0, le=30.0)
    fade_out: float | None = Field(default=None, ge=0.0, le=30.0)
    loop: bool | None = None
    enabled: bool | None = None
    review_status: SoundCueReviewStatus | None = None


class CreateSoundJobRequest(BaseModel):
    cue_ids: list[str] = Field(..., min_length=1, max_length=50)
    candidate_count: int = Field(default=2, ge=1, le=4)


class LicenseConfirmationRequest(BaseModel):
    license_tier: Literal["community", "enterprise"]
    stability_registered: bool
    stability_terms_accepted: bool
    gemma_terms_accepted: bool

    @model_validator(mode="after")
    def all_terms_are_explicit(self) -> "LicenseConfirmationRequest":
        if not (
            self.stability_registered
            and self.stability_terms_accepted
            and self.gemma_terms_accepted
        ):
            raise ValueError("必须逐项确认 Stability 注册、Stability 条款和 Gemma 条款")
        return self


class PreviewResponse(BaseModel):
    audio_url: str
    audio_sha256: str
    duration_seconds: float
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    render_snapshot: dict[str, Any]


def _project(project_id: str):
    try:
        return load_project(project_id)
    except ProjectStorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _provider_from_config(llm_config_id: str | None) -> tuple[OpenAIStyleLLMProvider, str | None, str]:
    try:
        configs = load_llm_configs()
        providers = load_llm_providers()
    except (LLMConfigStorageError, LLMProviderStorageError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    config = None
    if llm_config_id:
        config = next((item for item in configs if item.id == llm_config_id), None)
        if config is None:
            raise HTTPException(status_code=400, detail="指定的 LLM 配置不存在")
    elif configs:
        config = next((item for item in configs if item.is_default), configs[0])
    selected_provider = None
    if providers:
        provider_id = config.provider_id if config else None
        if provider_id:
            selected_provider = next((item for item in providers if item.id == provider_id), None)
            if selected_provider is None:
                raise HTTPException(status_code=400, detail="LLM 配置关联的 Provider 不存在")
        else:
            selected_provider = next((item for item in providers if item.is_default), providers[0])
    try:
        provider = OpenAIStyleLLMProvider(
            model=config.model if config else "gpt-4o-mini",
            default_temperature=None,
            base_url=selected_provider.base_url if selected_provider else None,
            api_key=selected_provider.api_key if selected_provider else None,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return provider, selected_provider.name if selected_provider else None, provider.model


def _fresh_plan(project_id: str, plan_id: str, *, persist: bool = False) -> SoundPlan:
    with sound_plan_lock(project_id, plan_id):
        try:
            plan = load_sound_plan(project_id, plan_id)
            refreshed = refresh_sound_plan_staleness(plan, load_segments(project_id))
            if persist and refreshed.model_dump(mode="json") != plan.model_dump(mode="json"):
                save_sound_plan(project_id, refreshed)
            return refreshed
        except SoundGenerationStorageError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sound-generation/model")
def api_sound_model_status() -> dict[str, Any]:
    state = worker_status()
    manifest: dict[str, Any] = {}
    if MANIFEST_PATH.is_file():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    return {
        **asdict(state),
        "manifest": MANIFEST_PATH.name if MANIFEST_PATH.is_file() else None,
        "model_revision": manifest.get("revision") or manifest.get("model_revision"),
        "code_revision": "a0b57f5483c4588f827f3552b7d5c6ca2a9687be",
        "port": 8005,
        "enabled": bool(state.model_ready and state.license_accepted and state.quality_gate_passed),
        "powered_by": "Stability AI",
    }


@router.post("/sound-generation/license-confirmation")
def api_confirm_sound_model_license(payload: LicenseConfirmationRequest) -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise HTTPException(status_code=503, detail="Stable Audio manifest 尚未安装")
    try:
        from scripts.model_preflight import load_manifest, record_license_acceptance

        manifest = load_manifest(MANIFEST_PATH)
        receipt = record_license_acceptance(
            manifest,
            license_tier=payload.license_tier,
            stability_registered=payload.stability_registered,
            stability_terms_accepted=payload.stability_terms_accepted,
            gemma_terms_accepted=payload.gemma_terms_accepted,
            project_root=REPO_ROOT,
        )
        return {
            "accepted": True,
            "receipt": str(receipt.relative_to(REPO_ROOT)),
            "license_tier": payload.license_tier,
        }
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/projects/{project_id}/sound-plans",
    response_model=SoundPlan,
    dependencies=[Depends(project_mutation_guard)],
)
def api_draft_sound_plan(project_id: str, payload: DraftSoundPlanRequest) -> SoundPlan:
    project = _project(project_id)
    segments = load_segments(project_id)
    settings = load_settings(project_id)
    density = payload.density or default_sound_density(settings.content_mode)
    provider, provider_name, model = _provider_from_config(payload.llm_config_id)
    try:
        plan = draft_sound_plan(
            project=project,
            segments=segments,
            density=density,
            provider=provider,
            planner_provider=provider_name,
            planner_model=model,
        )
        save_sound_plan(project_id, plan)
        return plan
    except SoundPlannerError as exc:
        # draft_sound_plan is all-or-nothing: there is deliberately no partial
        # plan file to clean up on provider or schema failure.
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/projects/{project_id}/sound-plans", response_model=list[SoundPlan])
def api_list_sound_plans(project_id: str) -> list[SoundPlan]:
    _project(project_id)
    return [
        refresh_sound_plan_staleness(plan, load_segments(project_id))
        for plan in list_sound_plans(project_id)
    ]


@router.get("/projects/{project_id}/sound-plans/{plan_id}", response_model=SoundPlan)
def api_get_sound_plan(project_id: str, plan_id: str) -> SoundPlan:
    _project(project_id)
    return _fresh_plan(project_id, plan_id)


@router.delete(
    "/projects/{project_id}/sound-plans/{plan_id}",
    dependencies=[Depends(project_mutation_guard)],
)
def api_delete_sound_plan(project_id: str, plan_id: str) -> dict[str, bool]:
    with sound_plan_lock(project_id, plan_id):
        plan = _fresh_plan(project_id, plan_id, persist=False)
        if any(
            cue.adopted_asset_id
            or cue.adopted_clip_id
            or cue.selected_candidate_id
            or any(
                candidate.status == SoundCandidateStatus.ADOPTED
                for candidate in cue.candidates
            )
            for cue in plan.cues
        ):
            raise HTTPException(status_code=409, detail="计划已有采用或半完成采用状态，不能直接删除")
        if any(
            candidate.status in {SoundCandidateStatus.QUEUED, SoundCandidateStatus.GENERATING}
            for cue in plan.cues
            for candidate in cue.candidates
        ):
            raise HTTPException(status_code=409, detail="计划仍有生成中的候选")
        root = project_root(project_id, ensure_dir=False)
        paths: list[Path] = []
        for cue in plan.cues:
            for candidate in cue.candidates:
                if not candidate.audio_file:
                    continue
                try:
                    path = resolve_scoped_project_asset(
                        root,
                        candidate.audio_file,
                        required_subdir="production/sound_generation/candidates",
                        allowed_extensions={".wav"},
                    )
                except Exception as exc:
                    raise HTTPException(status_code=409, detail="计划包含非法候选路径，拒绝删除") from exc
                if path not in paths:
                    paths.append(path)
        moved: list[tuple[Path, Path]] = []
        try:
            for path in paths:
                if not path.is_file():
                    continue
                quarantine = path.with_name(f".{path.name}.delete-{uuid.uuid4().hex}.tmp")
                os.replace(path, quarantine)
                moved.append((path, quarantine))
            deleted = delete_sound_plan(project_id, plan_id)
            if not deleted:
                raise SoundGenerationStorageError("音效计划在删除事务中消失")
            for _path, quarantine in moved:
                try:
                    quarantine.unlink(missing_ok=True)
                except OSError:
                    # The metadata commit is authoritative. A later maintenance
                    # sweep may remove a quarantined file; restoring it here
                    # would create an untracked candidate WAV.
                    pass
            return {"deleted": True}
        except Exception:
            for path, quarantine in reversed(moved):
                if quarantine.exists() and not path.exists():
                    try:
                        os.replace(quarantine, path)
                    except OSError:
                        pass
            raise


@router.patch(
    "/projects/{project_id}/sound-plans/{plan_id}/cues/{cue_id}",
    response_model=SoundPlan,
    dependencies=[Depends(project_mutation_guard)],
)
def api_update_sound_cue(
    project_id: str,
    plan_id: str,
    cue_id: str,
    payload: CueUpdateRequest,
) -> SoundPlan:
    # Keep the global lock order stable whenever a SoundPlan mutation may also
    # update the production timeline: production first, then the plan.
    with production_lock(project_id), sound_plan_lock(project_id, plan_id):
        plan = _fresh_plan(project_id, plan_id, persist=False)
        cue = next((item for item in plan.cues if item.id == cue_id), None)
        if cue is None:
            raise HTTPException(status_code=404, detail="cue 不存在")
        before = fingerprint_sound_cue(cue)
        updates = payload.model_dump(exclude_unset=True)
        if payload.placement is not None:
            updates["segment_id"] = payload.placement.start.segment_id
        try:
            replacement = SoundCue.model_validate({**cue.model_dump(mode="python"), **updates})
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        changed_fields = {
            name
            for name in SoundCue.model_fields
            if getattr(replacement, name) != getattr(cue, name)
        }
        if cue.adopted_asset_id:
            # An adopted WAV is immutable, but its semantic position and mix are
            # editorial data and must remain adjustable. Generation semantics,
            # approval state, source segment, and enabled state require an
            # explicit unadopt operation first.
            editable_after_adoption = {
                "description_cn",
                "placement",
                "playback_duration_seconds",
                "gain_db",
                "fade_in",
                "fade_out",
                "loop",
            }
            forbidden = changed_fields - editable_after_adoption
            if "placement" in changed_fields:
                # segment_id is a denormalized projection of placement.start,
                # never an independently editable field in this API.
                forbidden.discard("segment_id")
            if forbidden:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "该 cue 已采用为正式素材；生成语义或审核状态不能直接修改。"
                        "请先解除采用后再编辑（冲突字段: "
                        + ", ".join(sorted(forbidden))
                        + "）"
                    ),
                )
        segments = load_segments(project_id)
        allowed_segment_ids = {item.id for item in segments}
        referenced_segment_ids = {
            replacement.segment_id,
            replacement.placement.start.segment_id,
        }
        if replacement.placement.end is not None:
            referenced_segment_ids.add(replacement.placement.end.segment_id)
        unknown = referenced_segment_ids - allowed_segment_ids
        if unknown:
            raise HTTPException(status_code=422, detail=f"cue 引用了未知句段: {', '.join(sorted(unknown))}")
        if replacement.placement.end is not None:
            order = {item.id: index for index, item in enumerate(segments)}
            start_anchor = replacement.placement.start
            end_anchor = replacement.placement.end
            start_index = order[start_anchor.segment_id]
            end_index = order[end_anchor.segment_id]
            if end_index < start_index:
                raise HTTPException(status_code=422, detail="Ambience 结束锚点不能位于开始句段之前")
            if end_index == start_index:
                if start_anchor.edge.value == "end" and end_anchor.edge.value == "start":
                    raise HTTPException(status_code=422, detail="Ambience 同句段的结束锚点必须晚于开始锚点")
                if (
                    start_anchor.edge == end_anchor.edge
                    and end_anchor.offset_seconds <= start_anchor.offset_seconds
                ):
                    raise HTTPException(status_code=422, detail="Ambience 同边锚点的结束偏移必须大于开始偏移")
        if cue.adopted_asset_id and replacement.segment_id != cue.segment_id:
            rebound_segment = next(item for item in segments if item.id == replacement.segment_id)
            replacement.source_fingerprint = fingerprint_segment_source(rebound_segment)
            replacement.stale = False
        elif not cue.adopted_asset_id and replacement.segment_id != cue.segment_id:
            # A draft moved to another sentence has not yet been semantically
            # approved against that sentence, even though its WAV-generation
            # parameters may be byte-for-byte identical.
            replacement.stale = True
            for candidate in replacement.candidates:
                candidate.stale = True
        generation_fields_changed = fingerprint_sound_cue(replacement) != before
        review_fields_changed = bool(changed_fields & {
            "track",
            "prompt",
            "negative_prompt",
            "placement",
            "requested_duration_seconds",
            "playback_duration_seconds",
            "gain_db",
            "fade_in",
            "fade_out",
            "loop",
        })
        if generation_fields_changed and not cue.adopted_asset_id:
            replacement.review_status = SoundCueReviewStatus.DRAFT
            for candidate in replacement.candidates:
                candidate.stale = True
        elif review_fields_changed and not cue.adopted_asset_id:
            replacement.review_status = SoundCueReviewStatus.DRAFT
        if payload.review_status == SoundCueReviewStatus.APPROVED:
            segment = next((item for item in segments if item.id == replacement.segment_id), None)
            if segment is None:
                raise HTTPException(status_code=409, detail="cue 锚定的句段已删除")
            replacement.source_fingerprint = fingerprint_segment_source(segment)
            replacement.stale = False
            replacement.review_status = SoundCueReviewStatus.APPROVED
            for candidate in replacement.candidates:
                candidate.stale = candidate.cue_fingerprint != fingerprint_sound_cue(replacement)
            plan.source_fingerprint = fingerprint_sound_source(segments)
        plan.cues = [replacement if item.id == cue_id else item for item in plan.cues]
        current_segments = segments
        plan.stale = (
            plan.source_fingerprint != fingerprint_sound_source(current_segments)
            or any(item.stale for item in plan.cues)
        )
        plan.updated_at = datetime.now(timezone.utc)
        if cue.adopted_asset_id:
            settings = load_settings(project_id)
            old_settings = settings.model_copy(deep=True)
            clip = next(
                (item for item in settings.sound_design.clips if item.id == cue.adopted_clip_id),
                None,
            )
            if clip is None:
                raise HTTPException(
                    status_code=409,
                    detail="已采用 cue 的时间线片段已缺失；请先解除采用以修复状态",
                )
            selected = next(
                (item for item in replacement.candidates if item.id == replacement.selected_candidate_id),
                None,
            )
            clip.semantic_placement = replacement.placement.model_copy(deep=True)
            clip.duration = (
                replacement.playback_duration_seconds
                or (selected.duration_seconds if selected is not None else clip.duration)
            )
            clip.gain_db = replacement.gain_db
            clip.fade_in = replacement.fade_in
            clip.fade_out = replacement.fade_out
            clip.loop = replacement.loop
            now = datetime.now(timezone.utc)
            settings.sound_design.updated_at = now
            settings.updated_at = now
            save_settings(project_id, settings)
            try:
                save_sound_plan(project_id, plan)
            except Exception:
                try:
                    save_settings(project_id, old_settings)
                except Exception:
                    pass
                raise
        else:
            save_sound_plan(project_id, plan)
        return plan


@router.post(
    "/projects/{project_id}/sound-plans/{plan_id}/jobs",
    response_model=SoundGenerationJob,
    status_code=202,
    dependencies=[Depends(project_mutation_guard)],
)
def api_create_sound_job(
    project_id: str,
    plan_id: str,
    payload: CreateSoundJobRequest,
) -> SoundGenerationJob:
    _project(project_id)
    lease = try_acquire_gpu_operation(kind="sound_generation", project_id=project_id)
    if lease is None:
        raise HTTPException(status_code=409, detail="生成音频 GPU 正忙；本请求未排队，请稍后重试")
    try:
        job = prepare_sound_job(
            project_id,
            plan_id,
            payload.cue_ids,
            candidate_count=payload.candidate_count,
            job_id=generate_sound_job_id(),
        )
        launch_sound_job(job, lease)
        return job
    except (SoundJobError, SoundGenerationStorageError, SoundJobStorageError) as exc:
        release_gpu_operation(lease)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception:
        release_gpu_operation(lease)
        raise


@router.get("/projects/{project_id}/sound-jobs", response_model=list[SoundGenerationJob])
def api_list_sound_jobs(project_id: str) -> list[SoundGenerationJob]:
    _project(project_id)
    return list_sound_jobs(project_id)


@router.get("/projects/{project_id}/sound-jobs/{job_id}", response_model=SoundGenerationJob)
def api_get_sound_job(project_id: str, job_id: str) -> SoundGenerationJob:
    try:
        return load_sound_job(project_id, job_id)
    except SoundJobStorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/projects/{project_id}/sound-jobs/{job_id}/cancel", response_model=SoundGenerationJob)
def api_cancel_sound_job(project_id: str, job_id: str) -> SoundGenerationJob:
    try:
        return request_sound_job_cancel(project_id, job_id)
    except SoundJobStorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _find_candidate(plan: SoundPlan, cue_id: str, candidate_id: str):
    cue = next((item for item in plan.cues if item.id == cue_id), None)
    if cue is None:
        raise HTTPException(status_code=404, detail="cue 不存在")
    candidate = next((item for item in cue.candidates if item.id == candidate_id), None)
    if candidate is None:
        raise HTTPException(status_code=404, detail="候选不存在")
    return cue, candidate


@router.get(
    "/projects/{project_id}/sound-plans/{plan_id}/cues/{cue_id}/candidates/{candidate_id}/audio"
)
def api_get_sound_candidate_audio(
    project_id: str,
    plan_id: str,
    cue_id: str,
    candidate_id: str,
) -> FileResponse:
    plan = _fresh_plan(project_id, plan_id)
    _cue, candidate = _find_candidate(plan, cue_id, candidate_id)
    if not candidate.audio_file or not candidate.audio_sha256:
        raise HTTPException(status_code=409, detail="候选音频尚未生成")
    root = project_root(project_id, ensure_dir=False)
    try:
        path = resolve_scoped_project_asset(
            root,
            candidate.audio_file,
            required_subdir="production/sound_generation/candidates",
            allowed_extensions={".wav"},
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail="候选音频路径非法") from exc
    if not path.is_file() or sha256_file(path) != candidate.audio_sha256:
        raise HTTPException(status_code=409, detail="候选音频缺失或 SHA-256 不匹配")
    return FileResponse(path, media_type="audio/wav", filename=f"{candidate.id}.wav")


@router.post(
    "/projects/{project_id}/sound-plans/{plan_id}/cues/{cue_id}/candidates/{candidate_id}/adopt",
    response_model=SoundPlan,
    dependencies=[Depends(project_mutation_guard)],
)
def api_adopt_sound_candidate(
    project_id: str,
    plan_id: str,
    cue_id: str,
    candidate_id: str,
) -> SoundPlan:
    root = project_root(project_id, ensure_dir=False)
    with production_lock(project_id), sound_plan_lock(project_id, plan_id):
        plan = _fresh_plan(project_id, plan_id, persist=False)
        cue, candidate = _find_candidate(plan, cue_id, candidate_id)
        assets = load_assets(project_id)
        settings = load_settings(project_id)
        if cue.selected_candidate_id == candidate_id and cue.adopted_asset_id and cue.adopted_clip_id:
            adopted_asset = next((item for item in assets if item.id == cue.adopted_asset_id), None)
            adopted_clip = next(
                (item for item in settings.sound_design.clips if item.id == cue.adopted_clip_id),
                None,
            )
            if adopted_asset is None or adopted_clip is None:
                raise HTTPException(status_code=409, detail="cue 的采用状态不完整，请先修复或解除采用")
            try:
                adopted_path = resolve_scoped_project_asset(
                    root,
                    adopted_asset.audio_file,
                    required_subdir="production/assets",
                    allowed_extensions={".wav"},
                )
            except Exception as exc:
                raise HTTPException(status_code=409, detail="已采用素材路径非法") from exc
            if (
                adopted_clip.asset_id != adopted_asset.id
                or adopted_clip.track.value != cue.track.value
                or adopted_asset.track.value != cue.track.value
                or adopted_clip.semantic_placement != cue.placement
                or adopted_asset.audio_sha256 != candidate.audio_sha256
                or adopted_asset.generation != candidate.provenance
                or not adopted_path.is_file()
                or sha256_file(adopted_path) != adopted_asset.audio_sha256
            ):
                raise HTTPException(status_code=409, detail="cue 的已采用素材或时间线状态不一致")
            return plan
        if cue.adopted_asset_id:
            raise HTTPException(status_code=409, detail="该 cue 已采用另一候选；请先解除采用")
        if not candidate.audio_file or not candidate.audio_sha256:
            raise HTTPException(status_code=409, detail="候选尚无可采用音频")
        current_confirmation = _current_stable_audio_confirmation()
        if current_confirmation is None:
            raise HTTPException(status_code=409, detail="当前 Stable Audio/Gemma 许可确认缺失或已过期")
        if (
            candidate.provenance.commercial_use is not True
            or not candidate.provenance.license_confirmation_id
        ):
            raise HTTPException(status_code=409, detail="候选缺少生成时的可商用许可快照，不能采用")
        if not stable_audio_license_snapshot_matches_current(
            candidate.provenance.license_confirmation_id,
            candidate.provenance.license_snapshot,
            current_confirmation,
        ):
            raise HTTPException(status_code=409, detail="候选绑定的是旧版或其他 Stable Audio/Gemma 许可回执")
        try:
            source = resolve_scoped_project_asset(
                root,
                candidate.audio_file,
                required_subdir="production/sound_generation/candidates",
                allowed_extensions={".wav"},
            )
        except Exception as exc:
            raise HTTPException(status_code=404, detail="候选路径非法") from exc
        if not source.is_file() or sha256_file(source) != candidate.audio_sha256:
            raise HTTPException(status_code=409, detail="候选文件缺失或 SHA-256 不匹配")

        asset_id = f"asset-{uuid.uuid4().hex[:12]}"
        clip_id = f"clip-{uuid.uuid4().hex[:12]}"
        try:
            adoption = adopt_sound_candidate(
                plan,
                cue_id,
                candidate_id,
                asset_id=asset_id,
                clip_id=clip_id,
                asset_name=cue.description_cn or f"Generated {cue.track.value}",
            )
        except SoundCandidateAdoptionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        target_dir = production_root(project_id, ensure=True) / "assets" / cue.track.value
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{asset_id}.wav"
        fd, staged_name = tempfile.mkstemp(dir=target_dir, prefix="adopt_", suffix=".tmp")
        os.close(fd)
        staged = Path(staged_name)
        old_assets = [item.model_copy(deep=True) for item in assets]
        old_settings = settings.model_copy(deep=True)
        try:
            shutil.copy2(source, staged)
            if sha256_file(staged) != candidate.audio_sha256:
                raise RuntimeError("采用素材复制后 SHA-256 不一致")
            os.replace(staged, target)
            adoption.asset.audio_file = str(target.relative_to(root))
            if adoption.asset.generation is not None:
                adoption.asset.generation.audio_sha256 = candidate.audio_sha256
            adopted_cue = next(item for item in adoption.plan.cues if item.id == cue_id)
            adopted_candidate = next(
                item for item in adopted_cue.candidates if item.id == candidate_id
            )
            adopted_candidate.provenance.audio_sha256 = candidate.audio_sha256
            assets.append(adoption.asset)
            settings.sound_design.clips.append(adoption.clip)
            now = datetime.now(timezone.utc)
            settings.sound_design.updated_at = now
            settings.updated_at = now
            save_assets(project_id, assets)
            save_settings(project_id, settings)
            save_sound_plan(project_id, adoption.plan)
            return adoption.plan
        except Exception:
            try:
                save_assets(project_id, old_assets)
                save_settings(project_id, old_settings)
            except Exception:
                pass
            staged.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise


@router.delete(
    "/projects/{project_id}/sound-plans/{plan_id}/cues/{cue_id}/adoption",
    response_model=SoundPlan,
    dependencies=[Depends(project_mutation_guard)],
)
def api_unadopt_sound_candidate(
    project_id: str,
    plan_id: str,
    cue_id: str,
) -> SoundPlan:
    """Atomically detach one generated asset while preserving its candidate."""

    root = project_root(project_id, ensure_dir=False)
    with production_lock(project_id), sound_plan_lock(project_id, plan_id):
        plan = _fresh_plan(project_id, plan_id, persist=False)
        cue = next((item for item in plan.cues if item.id == cue_id), None)
        if cue is None:
            raise HTTPException(status_code=404, detail="cue 不存在")
        adopted_candidates = [
            item for item in cue.candidates
            if item.status == SoundCandidateStatus.ADOPTED
        ]
        if (
            not cue.adopted_asset_id
            and not cue.adopted_clip_id
            and not cue.selected_candidate_id
            and not adopted_candidates
        ):
            return plan
        if not cue.adopted_asset_id and not cue.adopted_clip_id:
            raise HTTPException(status_code=409, detail="cue 的候选采用状态不完整，拒绝空操作")
        if not cue.adopted_asset_id or not cue.adopted_clip_id:
            raise HTTPException(status_code=409, detail="cue 的采用状态不完整，拒绝自动清理")

        assets = load_assets(project_id)
        settings = load_settings(project_id)
        asset = next((item for item in assets if item.id == cue.adopted_asset_id), None)
        clip = next(
            (item for item in settings.sound_design.clips if item.id == cue.adopted_clip_id),
            None,
        )
        if asset is None or clip is None or clip.asset_id != asset.id:
            raise HTTPException(status_code=409, detail="采用素材或时间线片段已缺失，拒绝自动清理")

        from dubbing_service.app.storage.masters import list_masters

        from dubbing_service.app.routers.production import _master_snapshot_uses_asset

        if any(
            _master_snapshot_uses_asset(item.sound_design_snapshot, asset.id)
            for item in list_masters(project_id)
        ):
            raise HTTPException(status_code=409, detail="素材已被冻结进正式 Master，不能解除采用")
        try:
            asset_path = resolve_scoped_project_asset(
                root,
                asset.audio_file,
                required_subdir="production/assets",
                allowed_extensions={".wav"},
            )
        except Exception as exc:
            raise HTTPException(status_code=409, detail="采用素材路径非法，拒绝解除") from exc
        if not asset_path.is_file() or sha256_file(asset_path) != asset.audio_sha256:
            raise HTTPException(status_code=409, detail="采用素材缺失或 SHA-256 不匹配")

        selected_candidate = next(
            (item for item in cue.candidates if item.id == cue.selected_candidate_id),
            None,
        )
        if (
            selected_candidate is None
            or selected_candidate.status != SoundCandidateStatus.ADOPTED
            or not selected_candidate.audio_file
            or selected_candidate.audio_sha256 != asset.audio_sha256
        ):
            raise HTTPException(status_code=409, detail="已采用候选记录不完整，拒绝解除采用")
        try:
            candidate_path = resolve_scoped_project_asset(
                root,
                selected_candidate.audio_file,
                required_subdir="production/sound_generation/candidates",
                allowed_extensions={".wav"},
            )
        except Exception as exc:
            raise HTTPException(status_code=409, detail="已采用候选路径非法，拒绝解除") from exc

        # The production asset must never be the only healthy copy. Rebuild a
        # missing/corrupt candidate staging WAV from the verified immutable
        # adopted asset before deleting that production copy.
        candidate_needs_repair = (
            not candidate_path.is_file()
            or sha256_file(candidate_path) != selected_candidate.audio_sha256
        )
        if candidate_needs_repair:
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            repair = candidate_path.with_name(
                f".{candidate_path.name}.repair-{uuid.uuid4().hex}.tmp"
            )
            try:
                shutil.copy2(asset_path, repair)
                if sha256_file(repair) != selected_candidate.audio_sha256:
                    raise RuntimeError("重建候选后 SHA-256 不一致")
                os.replace(repair, candidate_path)
            except Exception as exc:
                repair.unlink(missing_ok=True)
                raise HTTPException(status_code=409, detail=f"无法重建已采用候选音频: {exc}") from exc

        old_plan = plan.model_copy(deep=True)
        old_assets = [item.model_copy(deep=True) for item in assets]
        old_settings = settings.model_copy(deep=True)
        quarantine = asset_path.with_name(f".{asset_path.name}.unadopt-{uuid.uuid4().hex}.tmp")
        os.replace(asset_path, quarantine)
        try:
            save_assets(project_id, [item for item in assets if item.id != asset.id])
            settings.sound_design.clips = [
                item for item in settings.sound_design.clips if item.id != clip.id
            ]
            now = datetime.now(timezone.utc)
            settings.sound_design.updated_at = now
            settings.updated_at = now
            save_settings(project_id, settings)
            for item in cue.candidates:
                if item.id == cue.selected_candidate_id:
                    item.status = SoundCandidateStatus.READY
                    item.updated_at = now
            cue.selected_candidate_id = None
            cue.adopted_asset_id = None
            cue.adopted_clip_id = None
            plan.updated_at = now
            save_sound_plan(project_id, plan)
            quarantine.unlink()
            return plan
        except Exception:
            try:
                save_assets(project_id, old_assets)
                save_settings(project_id, old_settings)
                save_sound_plan(project_id, old_plan)
            except Exception:
                pass
            if quarantine.exists() and not asset_path.exists():
                try:
                    os.replace(quarantine, asset_path)
                except OSError:
                    pass
            raise


@router.delete(
    "/projects/{project_id}/sound-plans/{plan_id}/cues/{cue_id}/candidates/{candidate_id}",
    dependencies=[Depends(project_mutation_guard)],
)
def api_delete_sound_candidate(
    project_id: str,
    plan_id: str,
    cue_id: str,
    candidate_id: str,
) -> dict[str, bool]:
    with sound_plan_lock(project_id, plan_id):
        plan = _fresh_plan(project_id, plan_id, persist=False)
        cue, candidate = _find_candidate(plan, cue_id, candidate_id)
        if candidate.status in {SoundCandidateStatus.QUEUED, SoundCandidateStatus.GENERATING}:
            raise HTTPException(status_code=409, detail="候选仍在生成")
        if cue.selected_candidate_id == candidate_id or candidate.status == SoundCandidateStatus.ADOPTED:
            raise HTTPException(status_code=409, detail="已采用候选不可清理")
        path: Path | None = None
        quarantine: Path | None = None
        if candidate.audio_file:
            root = project_root(project_id, ensure_dir=False)
            try:
                path = resolve_scoped_project_asset(
                    root,
                    candidate.audio_file,
                    required_subdir="production/sound_generation/candidates",
                    allowed_extensions={".wav"},
                )
            except Exception as exc:
                raise HTTPException(status_code=409, detail="候选路径非法，拒绝清理") from exc
            if path.is_file():
                quarantine = path.with_name(f".{path.name}.delete-{uuid.uuid4().hex}.tmp")
                os.replace(path, quarantine)
        cue.candidates = [item for item in cue.candidates if item.id != candidate_id]
        try:
            save_sound_plan(project_id, plan)
            if quarantine is not None:
                try:
                    quarantine.unlink(missing_ok=True)
                except OSError:
                    pass
            return {"deleted": True}
        except Exception:
            if path is not None and quarantine is not None and quarantine.exists() and not path.exists():
                try:
                    os.replace(quarantine, path)
                except OSError:
                    pass
            raise


@router.post(
    "/projects/{project_id}/sound-preview",
    response_model=PreviewResponse,
    dependencies=[Depends(project_export_guard)],
)
def api_create_sound_preview(project_id: str) -> PreviewResponse:
    root = project_root(project_id, ensure_dir=False)
    quality = run_project_quality_checks(project_id, root, purpose="preview")
    export = api_export_full_project_with_edits(project_id)
    dialogue = (root / export.output_file).resolve()
    timeline: object | None = None
    if export.dialogue_timeline_file:
        timeline_path = (root / export.dialogue_timeline_file).resolve()
        if timeline_path.is_relative_to(root.resolve()) and timeline_path.is_file():
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    preview_dir = production_root(project_id, ensure=True) / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    final = preview_dir / "sound_mix_latest.wav"
    fd, staged_name = tempfile.mkstemp(dir=preview_dir, prefix="preview_", suffix=".wav")
    os.close(fd)
    staged = Path(staged_name)
    try:
        snapshot = render_four_track_master(
            dialogue_path=dialogue,
            output_path=staged,
            root=root,
            settings=load_settings(project_id),
            assets=load_assets(project_id),
            dialogue_duration=_probe_audio_duration_seconds(dialogue),
            dialogue_timeline=timeline,
        )
        os.replace(staged, final)
    except Exception:
        staged.unlink(missing_ok=True)
        raise
    return PreviewResponse(
        audio_url=f"/api/v1/projects/{project_id}/sound-preview/audio",
        audio_sha256=sha256_file(final),
        duration_seconds=_probe_audio_duration_seconds(final),
        warnings=[item.model_dump(mode="json") for item in quality.issues if item.severity != "info"],
        render_snapshot=snapshot,
    )


@router.get("/projects/{project_id}/sound-preview/audio")
def api_get_sound_preview(project_id: str) -> FileResponse:
    path = production_root(project_id) / "previews" / "sound_mix_latest.wav"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="尚未生成声音设计 Preview")
    return FileResponse(path, media_type="audio/wav", filename="sound_mix_preview.wav")
