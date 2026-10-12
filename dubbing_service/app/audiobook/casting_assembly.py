"""Strict frozen-cast checks shared by audio assembly boundaries."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.services.takes import take_is_selectable
from . import takes
from .fingerprint import file_sha256


class CastingAssemblyError(ValueError):
    """Assembly cannot prove every selected Take belongs to an approved cast."""


def _canonical_role_id(project: Any, role_id: str | None) -> str | None:
    aliases = project.character_cast.role_aliases
    seen = set()
    while role_id in aliases:
        if role_id in seen:
            raise CastingAssemblyError("Character alias cycle prevents strict audio assembly")
        seen.add(role_id)
        role_id = aliases[role_id]
    return role_id


def strict_casting_enabled(project: Any) -> bool:
    metadata = project.metadata or {}
    return bool(project.voice_lab.casting.get("enabled")) or (
        (metadata.get("autocast_import") or {}).get("mode") == "autocast"
    )


def current_frozen_binding(project: Any, segments: list, *, allow_experimental: bool = False) -> dict | None:
    """Validate selected Takes against one current frozen manifest, if required.

    Legacy projects return ``None`` and preserve their existing export behavior.
    Strict projects must prove current source/cast approvals, exact per-utterance
    requirement fingerprints, playable Take provenance, and unchanged bytes.
    """
    if not allow_experimental:
        from .runtime_binding import require_production_take, RuntimeBindingError
        try:
            for segment in segments:
                require_production_take(takes.selected_take(segment))
        except RuntimeBindingError as exc:
            raise CastingAssemblyError(str(exc)) from exc
    if not strict_casting_enabled(project):
        return None
    from . import casting_review

    book_creator = (project.metadata or {}).get("book_creator") or {}
    scene_breaks = set(book_creator.get("scene_breaks", []))
    left_out = set(((project.metadata or {}).get("autocast_import") or {}).get("skipped") or {})
    speech = [segment for segment in segments
        if segment.id not in scene_breaks and segment.id not in left_out and segment.text.strip()]
    if not speech:
        raise CastingAssemblyError("Strict audio assembly has no speaking utterances")
    wanted = {segment.id for segment in speech}
    state = project.voice_lab.casting
    candidates = []
    for receipt in state.get("productions", {}).values():
        manifests = [receipt.get("manifest"), *(receipt.get("manifest_history") or [])]
        for manifest in manifests:
            if not isinstance(manifest, dict) or not wanted.issubset(set(manifest.get("utterance_ids", []))):
                continue
            if manifest not in candidates:
                candidates.append(manifest)
    selected_manifest = next((manifest for manifest in candidates
        if casting_review.manifest_current(project.id, manifest)), None)
    if selected_manifest is None:
        raise CastingAssemblyError("Strict audio assembly requires a current frozen cast manifest covering every utterance")

    root = safe_project_root(project.id).resolve()
    selections = []
    for segment in speech:
        role_id = _canonical_role_id(project, segment.role_id)
        assignment = (state.get("roles", {}).get(role_id, {}).get("assignment") or {}) if role_id else {}
        frozen_assignment = (selected_manifest.get("cast") or {}).get(role_id)
        if (not role_id or not casting_review.approved(project, role_id)
                or assignment != frozen_assignment):
            raise CastingAssemblyError(f"Utterance {segment.id} lacks the exact current approved frozen assignment")
        take = takes.selected_take(segment)
        requirement = (selected_manifest.get("requirements") or {}).get(segment.id)
        if (not take or not requirement or take.input_fingerprint != requirement
                # A take stores the role id its line carried when it was recorded, which is the old id once a character
                # has been merged into another. The requirement fingerprint above already binds it to the voice.
                or _canonical_role_id(project, take.role_id) != role_id or not take.audio_sha256 or not take.audio_file
                or take_is_selectable(take) is not None
                or not take.qa_report or not take.qa_report.passed
                or not take.qa_report.matches(take.id, take.audio_sha256)):
            raise CastingAssemblyError(f"Selected Take for {segment.id} is not the frozen approved, QA-passing requirement")
        try:
            path = (root / take.audio_file).resolve()
            path.relative_to(root)
        except (OSError, ValueError) as exc:
            raise CastingAssemblyError(f"Selected Take for {segment.id} is outside the project audio root") from exc
        if not path.is_file() or file_sha256(path) != take.audio_sha256:
            raise CastingAssemblyError(f"Selected Take bytes for {segment.id} do not match the frozen selection")
        selections.append({"utterance_id": segment.id, "role_id": role_id, "take_id": take.id,
            "audio_sha256": take.audio_sha256, "input_fingerprint": take.input_fingerprint})
    return {"manifest_id": selected_manifest.get("id"), "selections": selections}


def revalidate_binding(project: Any, segments: list, original: dict | None) -> None:
    """Refuse publication if strict approvals or selected source Takes changed."""
    if original is None:
        return
    current = current_frozen_binding(project, segments)
    if current != original:
        raise CastingAssemblyError("Source, approved cast or selected Takes changed during strict audio assembly")
