"""Quality checks and the deliberately small four-track render engine."""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import ffmpeg  # type: ignore

from dubbing_service.app.core.helpers import sha256_file
from dubbing_service.app.models.enums import SegmentReviewStatus
from dubbing_service.app.models.production import (
    AudioAsset,
    ProductionSettings,
    QualityIssue,
    QualityReport,
    SoundTrack,
)
from dubbing_service.app.services.takes import get_selected_take
from dubbing_service.app.services.sound_generation import resolve_sound_design
from dubbing_service.app.storage.masters import list_masters
from dubbing_service.app.storage.production import load_assets, load_settings
from dubbing_service.app.storage.segments import load_segments
from dubbing_service.app.storage.subtitles import list_subtitles
from dubbing_service.app.storage.projects import load_project


def apply_pronunciation_dictionary(project_id: str, text: str | None) -> str:
    """Apply enabled project terms to a generation-only text copy."""

    result = str(text or "")
    entries = sorted(
        (item for item in load_settings(project_id).pronunciation_dictionary if item.enabled),
        key=lambda item: len(item.source),
        reverse=True,
    )
    for entry in entries:
        result = result.replace(entry.source, entry.pronunciation)
    return result


def _safe_file(root: Path, rel: str, required_prefix: str | None = None) -> Path | None:
    try:
        target = (root / rel).resolve()
        base = root.resolve()
    except Exception:
        return None
    if not target.is_relative_to(base):
        return None
    if required_prefix:
        try:
            relative = target.relative_to(base).as_posix()
        except ValueError:
            return None
        if not relative.startswith(required_prefix.rstrip("/") + "/"):
            return None
    return target if target.is_file() else None


def _audio_levels(path: Path) -> tuple[float | None, float | None, float]:
    """Return max dBFS, mean dBFS and longest detected silence."""

    command = [
        "ffmpeg", "-hide_banner", "-nostdin", "-i", str(path),
        "-af", "volumedetect,silencedetect=noise=-45dB:d=1.0", "-f", "null", "NUL",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None, None, 0.0
    output = result.stderr or ""
    max_match = re.findall(r"max_volume:\s*(-?[0-9.]+) dB", output)
    mean_match = re.findall(r"mean_volume:\s*(-?[0-9.]+) dB", output)
    starts = [float(item) for item in re.findall(r"silence_start:\s*([0-9.]+)", output)]
    ends = [float(item) for item in re.findall(r"silence_end:\s*([0-9.]+)", output)]
    longest = max((end - start for start, end in zip(starts, ends)), default=0.0)
    return (
        float(max_match[-1]) if max_match else None,
        float(mean_match[-1]) if mean_match else None,
        longest,
    )


def _duration(path: Path) -> float:
    try:
        probe = ffmpeg.probe(str(path))
        return float(probe.get("format", {}).get("duration") or 0.0)
    except Exception:
        return 0.0


def _formal_severity(purpose: str) -> str:
    return "warning" if purpose == "preview" else "error"


@dataclass(frozen=True)
class StableAudioLicenseConfirmation:
    """Validated current receipt fields used by generated-audio gates."""

    operator_confirmation_id: str
    confirmation_sha256: str
    license_tier: str


def stable_audio_license_snapshot_matches_current(
    historical_confirmation_id: object,
    historical_snapshot: object,
    current_confirmation: StableAudioLicenseConfirmation | None,
) -> bool:
    """Match immutable provenance to the current terms, without binding receipt IDs.

    A restore deliberately creates a new operator confirmation ID.  The old ID
    remains required in provenance for auditability, while the terms hash and
    license tier determine whether the new, valid receipt is equivalent.
    """

    if (
        current_confirmation is None
        or not str(historical_confirmation_id or "").strip()
        or not isinstance(historical_snapshot, dict)
    ):
        return False
    return (
        historical_snapshot.get("confirmation_sha256")
        == current_confirmation.confirmation_sha256
        and historical_snapshot.get("license_tier")
        == current_confirmation.license_tier
    )


def _current_stable_audio_confirmation() -> StableAudioLicenseConfirmation | None:
    """Return the identity, terms hash and tier from the current valid receipt."""

    repo_root = Path(__file__).resolve().parents[3]
    manifest_path = repo_root / "models" / "manifests" / "stable_audio_3_small_sfx.json"
    if not manifest_path.is_file():
        return None
    try:
        from scripts.model_preflight import (
            has_license_acceptance,
            license_receipt_path,
            load_manifest,
        )

        manifest = load_manifest(manifest_path)
        if not has_license_acceptance(manifest, project_root=repo_root):
            return None
        receipt_path = license_receipt_path(manifest, project_root=repo_root)
        if receipt_path is None or not receipt_path.is_file():
            return None
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_id = payload.get("operator_confirmation_id")
        confirmation_sha256 = payload.get("confirmation_sha256")
        license_tier = payload.get("license_tier")
        if (
            not receipt_id
            or not confirmation_sha256
            or license_tier not in {"community", "enterprise"}
        ):
            return None
        return StableAudioLicenseConfirmation(
            operator_confirmation_id=str(receipt_id),
            confirmation_sha256=str(confirmation_sha256),
            license_tier=str(license_tier),
        )
    except Exception:
        return None


def run_project_quality_checks(
    project_id: str,
    root: Path,
    *,
    purpose: str = "preview",
) -> QualityReport:
    if purpose not in {"preview", "master", "release"}:
        raise ValueError("purpose 必须为 preview、master 或 release")
    issues: list[QualityIssue] = []
    segments = sorted(load_segments(project_id), key=lambda item: item.index)
    if not segments:
        issues.append(QualityIssue(code="no_segments", severity="error", message="项目没有分段"))
    last_index = -1
    previous_mean: float | None = None
    previous_text: str | None = None
    project = load_project(project_id)
    roles = {item.id: item for item in project.roles}
    for segment in segments:
        if segment.index <= last_index:
            issues.append(QualityIssue(
                code="segment_order", severity="error", segment_id=segment.id,
                message=f"第 {segment.index + 1} 段顺序重复或倒置",
            ))
        last_index = segment.index
        take = get_selected_take(segment)
        if take is None or take.missing:
            issues.append(QualityIssue(
                code="missing_take", severity="error", segment_id=segment.id,
                message=f"第 {segment.index + 1} 段缺少最终 Take",
            ))
            continue
        path = _safe_file(root, take.audio_file, "audio/segments")
        if path is None:
            issues.append(QualityIssue(
                code="missing_audio", severity="error", segment_id=segment.id,
                message=f"第 {segment.index + 1} 段音频文件缺失或路径非法",
            ))
            continue
        if take.audio_sha256 and sha256_file(path) != take.audio_sha256:
            issues.append(QualityIssue(
                code="take_hash_mismatch", severity="error", segment_id=segment.id,
                message=f"第 {segment.index + 1} 段原始 Take 校验失败",
            ))
        duration = _duration(path)
        char_count = len("".join(str(segment.text or "").split()))
        if duration <= 0:
            issues.append(QualityIssue(
                code="invalid_duration", severity="error", segment_id=segment.id,
                message=f"第 {segment.index + 1} 段无法读取有效时长",
            ))
        elif char_count >= 4:
            chars_per_second = char_count / duration
            if chars_per_second < 1.0 or chars_per_second > 12.0:
                issues.append(QualityIssue(
                    code="abnormal_duration", severity="warning", segment_id=segment.id,
                    message=f"第 {segment.index + 1} 段时长与文本长度明显不匹配（约 {chars_per_second:.1f} 字/秒）",
                ))
        normalized_text = "".join(str(segment.text or "").split())
        if normalized_text and normalized_text == previous_text:
            issues.append(QualityIssue(
                code="duplicate_segment", severity="warning", segment_id=segment.id,
                message=f"第 {segment.index + 1} 段与上一段文本重复",
            ))
        previous_text = normalized_text
        # A Take freezes the role used for synthesis. Legacy Takes without that
        # field fall back to the segment's current role; an explicit missing or
        # deleted role must never make formal voice-rights checks disappear.
        effective_role_id = take.role_id if take.role_id is not None else segment.role_id
        role = roles.get(effective_role_id or "")
        if role is None:
            issues.append(QualityIssue(
                code="voice_license_missing",
                severity=_formal_severity(purpose),
                segment_id=segment.id,
                message=f"第 {segment.index + 1} 段最终 Take 无法解析生成时的声音授权角色",
            ))
        elif role.voice_commercial_use is not True:
            issues.append(QualityIssue(
                code="voice_license_unconfirmed", severity=_formal_severity(purpose), segment_id=segment.id,
                message=f"角色“{role.name}”的声音商业授权尚未明确确认",
            ))
        if segment.review_status != SegmentReviewStatus.APPROVED:
            issues.append(QualityIssue(
                code="not_approved", severity="error", segment_id=segment.id,
                message=f"第 {segment.index + 1} 段尚未通过审听",
            ))
        for review_issue in segment.review_issues:
            if not review_issue.resolved:
                issues.append(QualityIssue(
                    code="unresolved_review_issue", severity="error", segment_id=segment.id,
                    message=f"第 {segment.index + 1} 段仍有未解决的 {review_issue.category} 问题",
                ))
        max_db, mean_db, silence = _audio_levels(path)
        if max_db is not None and max_db >= -0.1:
            issues.append(QualityIssue(
                code="possible_clipping", severity="warning", segment_id=segment.id,
                message=f"第 {segment.index + 1} 段峰值 {max_db:.1f} dBFS，可能削波",
            ))
        if silence >= 3.0:
            issues.append(QualityIssue(
                code="long_silence", severity="warning", segment_id=segment.id,
                message=f"第 {segment.index + 1} 段检测到约 {silence:.1f} 秒连续静音",
            ))
        if mean_db is not None and previous_mean is not None and abs(mean_db - previous_mean) >= 8.0:
            issues.append(QualityIssue(
                code="volume_jump", severity="warning", segment_id=segment.id,
                message=f"第 {segment.index + 1} 段与上一段平均响度相差 {abs(mean_db - previous_mean):.1f} dB",
            ))
        if mean_db is not None:
            previous_mean = mean_db

    settings = load_settings(project_id)
    assets = {item.id: item for item in load_assets(project_id)}
    for clip in settings.sound_design.clips:
        asset = assets.get(clip.asset_id)
        if asset is None:
            issues.append(QualityIssue(
                code="missing_asset", severity="error", asset_id=clip.asset_id,
                message=f"时间线引用了不存在的 {clip.track.value} 素材",
            ))
            continue
        path = _safe_file(root, asset.audio_file, "production/assets")
        if path is None or sha256_file(path) != asset.audio_sha256:
            issues.append(QualityIssue(
                code="asset_hash_mismatch", severity="error", asset_id=asset.id,
                message=f"素材“{asset.name}”缺失或校验失败",
            ))
        if asset.license.commercial_use is not True:
            issues.append(QualityIssue(
                code="license_unconfirmed", severity=_formal_severity(purpose), asset_id=asset.id,
                message=f"素材“{asset.name}”尚未明确确认可商用",
            ))

        if getattr(asset.origin, "value", asset.origin) == "generated":
            provenance = asset.generation
            confirmation = _current_stable_audio_confirmation()
            if provenance is None:
                issues.append(QualityIssue(
                    code="generated_provenance_missing",
                    severity=_formal_severity(purpose),
                    asset_id=asset.id,
                    message=f"生成素材“{asset.name}”缺少模型 provenance",
                ))
            elif (
                provenance.commercial_use is not True
                or not stable_audio_license_snapshot_matches_current(
                    provenance.license_confirmation_id,
                    provenance.license_snapshot,
                    confirmation,
                )
                or provenance.audio_sha256 != asset.audio_sha256
            ):
                issues.append(QualityIssue(
                    code="generated_license_stale",
                    severity=_formal_severity(purpose),
                    asset_id=asset.id,
                    message=f"生成素材“{asset.name}”的 Stable Audio/Gemma 许可确认缺失或已过期",
                ))

    referenced_assets = {clip.asset_id for clip in settings.sound_design.clips}
    try:
        from dubbing_service.app.services.sound_generation import refresh_sound_plan_staleness
        from dubbing_service.app.storage.sound_generation import list_sound_plans

        settings_clips = {clip.id: clip for clip in settings.sound_design.clips}
        for plan in list_sound_plans(project_id):
            refreshed = refresh_sound_plan_staleness(plan, segments)
            for cue in refreshed.cues:
                pointers = (
                    cue.selected_candidate_id,
                    cue.adopted_asset_id,
                    cue.adopted_clip_id,
                )
                selected = next(
                    (item for item in cue.candidates if item.id == cue.selected_candidate_id),
                    None,
                )
                adopted_asset = assets.get(cue.adopted_asset_id or "")
                adopted_clip = settings_clips.get(cue.adopted_clip_id or "")
                adopted_candidates = [
                    item for item in cue.candidates if item.status.value == "adopted"
                ]
                adoption_inconsistent = (any(pointers) or adopted_candidates) and (
                    not all(pointers)
                    or len(adopted_candidates) != 1
                    or adopted_candidates[0].id != cue.selected_candidate_id
                    or selected is None
                    or selected.status.value != "adopted"
                    or adopted_asset is None
                    or adopted_clip is None
                    or adopted_clip.asset_id != cue.adopted_asset_id
                    or adopted_clip.track.value != cue.track.value
                    or adopted_asset.track.value != cue.track.value
                    or adopted_clip.semantic_placement != cue.placement
                    or adopted_clip.duration
                    != (cue.playback_duration_seconds or selected.duration_seconds)
                    or adopted_clip.gain_db != cue.gain_db
                    or adopted_clip.fade_in != cue.fade_in
                    or adopted_clip.fade_out != cue.fade_out
                    or adopted_clip.loop != cue.loop
                    or adopted_asset.audio_sha256 != selected.audio_sha256
                    or adopted_asset.generation != selected.provenance
                )
                if adoption_inconsistent:
                    issues.append(QualityIssue(
                        code="sound_adoption_inconsistent",
                        severity=_formal_severity(purpose),
                        asset_id=cue.adopted_asset_id,
                        segment_id=cue.segment_id,
                        message=f"已采用音效 cue“{cue.description_cn or cue.id}”与候选、素材或时间线绑定不一致",
                    ))
                if cue.adopted_asset_id in referenced_assets and cue.stale:
                    issues.append(QualityIssue(
                        code="sound_cue_stale",
                        severity=_formal_severity(purpose),
                        asset_id=cue.adopted_asset_id,
                        segment_id=cue.segment_id,
                        message=f"已采用音效 cue“{cue.description_cn or cue.id}”与当前对白文本不一致",
                    ))
    except Exception as exc:
        issues.append(QualityIssue(
            code="sound_plan_check_failed",
            severity=_formal_severity(purpose),
            message=f"无法验证 SoundPlan 语义锚点与许可状态: {exc}",
        ))

    for master in list_masters(project_id):
        path = _safe_file(root, master.audio_file, "audio/masters")
        if path is None or sha256_file(path) != master.audio_sha256:
            issues.append(QualityIssue(
                code="master_hash_mismatch", severity="error", master_id=master.id,
                message=f"Master“{master.name}”缺失或校验失败",
            ))
    for subtitle in list_subtitles(project_id):
        path = _safe_file(root, subtitle.audio_rel_path)
        if path is None:
            issues.append(QualityIssue(
                code="subtitle_source_missing", severity="warning",
                message=f"字幕 {subtitle.id} 的源音频已不存在",
            ))
        elif subtitle.audio_sha256 and sha256_file(path) != subtitle.audio_sha256:
            issues.append(QualityIssue(
                code="subtitle_hash_mismatch", severity="warning",
                message=f"字幕 {subtitle.id} 与当前源音频哈希不匹配",
            ))
    return QualityReport(
        project_id=project_id,
        purpose=purpose,
        passed=not any(item.severity == "error" for item in issues),
        issues=issues,
    )


def _mix_many(streams: Iterable, *, duration: str = "longest"):
    items = list(streams)
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    return ffmpeg.filter(items, "amix", inputs=len(items), duration=duration, dropout_transition=0, normalize=0)


@dataclass(frozen=True)
class AmbienceLoopPlan:
    source_duration: float
    output_duration: float
    repetitions: int
    crossfade_seconds: float


def plan_ambience_crossfade_loop(
    source_duration: float,
    output_duration: float,
    *,
    crossfade_seconds: float = 1.0,
) -> AmbienceLoopPlan:
    """Plan the minimum overlapping repetitions required for an ambience bed."""

    if not math.isfinite(source_duration) or source_duration <= 0:
        raise ValueError("环境声源时长必须大于 0")
    if not math.isfinite(output_duration) or output_duration <= 0:
        raise ValueError("环境声输出时长必须大于 0")
    if not math.isfinite(crossfade_seconds) or crossfade_seconds <= 0:
        raise ValueError("环境声循环交叉淡化时长必须大于 0")
    if output_duration <= source_duration:
        return AmbienceLoopPlan(source_duration, output_duration, 1, 0.0)
    overlap = min(crossfade_seconds, source_duration / 2.0)
    effective_step = source_duration - overlap
    repetitions = max(2, math.ceil((output_duration - overlap) / effective_step))
    return AmbienceLoopPlan(source_duration, output_duration, repetitions, overlap)


def _ambience_loop_stream(
    asset_path: Path,
    *,
    source_offset: float,
    source_duration: float,
    output_duration: float,
):
    plan = plan_ambience_crossfade_loop(source_duration, output_duration)
    source = (
        ffmpeg.input(str(asset_path)).audio
        .filter("atrim", start=source_offset, duration=source_duration)
        .filter("asetpts", "PTS-STARTPTS")
        .filter("aresample", 48000)
        .filter("aformat", channel_layouts="stereo")
    )
    if plan.repetitions == 1:
        parts = [source]
    else:
        split = source.filter_multi_output("asplit", plan.repetitions)
        parts = [split[index] for index in range(plan.repetitions)]
    stream = parts[0]
    for part in parts[1:]:
        stream = ffmpeg.filter(
            [stream, part],
            "acrossfade",
            d=plan.crossfade_seconds,
            c1="tri",
            c2="tri",
        )
    return stream.filter("atrim", duration=output_duration).filter("asetpts", "PTS-STARTPTS"), plan


def render_four_track_master(
    *,
    dialogue_path: Path,
    output_path: Path,
    root: Path,
    settings: ProductionSettings,
    assets: list[AudioAsset],
    dialogue_duration: float,
    dialogue_timeline: object | None = None,
) -> dict[str, object]:
    """Render Dialogue plus BGM/Ambience/SFX, returning the frozen sound snapshot."""

    by_id = {item.id: item for item in assets}
    sound_design = settings.sound_design
    if any(item.semantic_placement is not None for item in sound_design.clips):
        if dialogue_timeline is None:
            timeline_path = root / "dialogue_timeline.json"
            try:
                dialogue_timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("存在语义定位音效，但最新 dialogue_timeline 不可用") from exc
        sound_design = resolve_sound_design(sound_design, dialogue_timeline)
    mix = sound_design.mix
    dialogue = (
        ffmpeg.input(str(dialogue_path)).audio
        .filter("aresample", 48000)
        .filter("aformat", channel_layouts="stereo")
    )
    if mix.preset == "spoken_word":
        dialogue = dialogue.filter("highpass", f=65).filter(
            "acompressor", threshold=0.16, ratio=1.8, attack=20, release=220,
        )
    elif mix.preset == "audio_story":
        dialogue = dialogue.filter("highpass", f=55).filter(
            "acompressor", threshold=0.14, ratio=2.2, attack=18, release=260,
        )
    else:  # clear_podcast
        dialogue = dialogue.filter("highpass", f=80).filter("lowpass", f=15500).filter(
            "acompressor", threshold=0.12, ratio=3.0, attack=12, release=180,
        )
    dialogue = dialogue.filter("volume", f"{mix.dialogue_gain_db}dB")
    needs_sidechain = mix.bgm_ducking and any(
        item.track == SoundTrack.BGM for item in sound_design.clips
    )
    if needs_sidechain:
        dialogue_parts = dialogue.filter_multi_output("asplit", 2)
        dialogue_main = dialogue_parts[0]
        sidechain = dialogue_parts[1]
    else:
        dialogue_main = dialogue
        sidechain = None
    tracks: dict[SoundTrack, list] = {kind: [] for kind in SoundTrack}
    snapshot_clips: list[dict[str, object]] = []
    total_duration = dialogue_duration

    for clip in sound_design.clips:
        asset = by_id.get(clip.asset_id)
        if asset is None or asset.track != clip.track:
            raise ValueError(f"时间线片段 {clip.id} 的素材不存在或轨道类型不匹配")
        asset_path = _safe_file(root, asset.audio_file, "production/assets")
        if asset_path is None or sha256_file(asset_path) != asset.audio_sha256:
            raise ValueError(f"素材 {asset.name} 缺失或校验失败")
        available = max(asset.duration_seconds - clip.source_offset, 0.01)
        duration = clip.duration
        if duration is None:
            duration = max(dialogue_duration - clip.start, 0.01) if clip.loop else available
        ambience_loop_plan: AmbienceLoopPlan | None = None
        if clip.track == SoundTrack.AMBIENCE and clip.loop:
            stream, ambience_loop_plan = _ambience_loop_stream(
                asset_path,
                source_offset=clip.source_offset,
                source_duration=available,
                output_duration=duration,
            )
        else:
            input_kwargs = {"stream_loop": -1} if clip.loop else {}
            stream = ffmpeg.input(str(asset_path), **input_kwargs).audio
            stream = stream.filter("atrim", start=clip.source_offset, duration=duration).filter("asetpts", "PTS-STARTPTS")
        if clip.fade_in > 0:
            stream = stream.filter("afade", type="in", start_time=0, duration=min(clip.fade_in, duration))
        if clip.fade_out > 0:
            stream = stream.filter(
                "afade", type="out", start_time=max(duration - clip.fade_out, 0), duration=min(clip.fade_out, duration),
            )
        stream = (
            stream.filter("aresample", 48000)
            .filter("aformat", channel_layouts="stereo")
            .filter("volume", f"{clip.gain_db}dB")
            .filter("adelay", f"{round(clip.start * 1000)}|{round(clip.start * 1000)}")
        )
        tracks[clip.track].append(stream)
        total_duration = max(total_duration, clip.start + duration)
        snapshot_clips.append({
            **clip.model_dump(mode="json"),
            "rendered_duration": duration,
            "asset_name": asset.name,
            "asset_file": asset.audio_file,
            "asset_sha256": asset.audio_sha256,
            "asset_origin": asset.origin.value,
            "license": asset.license.model_dump(mode="json"),
            "origin": asset.origin.value,
            "generation": (
                asset.generation.model_dump(mode="json")
                if asset.generation is not None
                else None
            ),
            "semantic_resolved": clip.semantic_placement is not None,
            "ambience_loop": (
                {
                    "repetitions": ambience_loop_plan.repetitions,
                    "crossfade_seconds": ambience_loop_plan.crossfade_seconds,
                }
                if ambience_loop_plan is not None
                else None
            ),
        })

    mixed_inputs = [dialogue_main]
    global_gains = {
        SoundTrack.BGM: mix.bgm_gain_db,
        SoundTrack.AMBIENCE: mix.ambience_gain_db,
        SoundTrack.SFX: mix.sfx_gain_db,
    }
    for track in SoundTrack:
        track_mix = _mix_many(tracks[track])
        if track_mix is None:
            continue
        track_mix = track_mix.filter("volume", f"{global_gains[track]}dB")
        if track == SoundTrack.BGM and mix.bgm_ducking and sidechain is not None:
            ratio = max(2.0, 1.0 + mix.ducking_amount_db / 2.0)
            track_mix = ffmpeg.filter(
                [track_mix, sidechain], "sidechaincompress",
                threshold=0.04, ratio=ratio, attack=20, release=450, makeup=1,
            )
        mixed_inputs.append(track_mix)
    rendered = _mix_many(mixed_inputs)
    if mix.fade_in > 0:
        rendered = rendered.filter("afade", type="in", start_time=0, duration=min(mix.fade_in, total_duration))
    if mix.fade_out > 0:
        rendered = rendered.filter(
            "afade", type="out", start_time=max(total_duration - mix.fade_out, 0), duration=min(mix.fade_out, total_duration),
        )
    rendered = rendered.filter(
        "loudnorm", I=mix.loudness_lufs, TP=mix.true_peak_db, LRA=11,
    ).filter("alimiter", limit=10 ** (mix.true_peak_db / 20.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = ffmpeg.output(
        rendered, str(output_path), ar=48000, ac=mix.output_channels, acodec="pcm_s24le",
    )
    ffmpeg.run(output, overwrite_output=True, capture_stdout=True, capture_stderr=True)
    return {
        "timeline_version": sound_design.version,
        "clips": snapshot_clips,
        "mix": mix.model_dump(mode="json"),
    }
