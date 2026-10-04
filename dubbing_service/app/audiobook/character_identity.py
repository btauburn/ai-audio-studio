"""The one authoritative speaker -> canonical character identity resolution.

Every speaking segment must resolve through speaker attribution to exactly one
canonical ``characterId``.  Historically two representations of the same
identity lived side by side: ``Segment.role_id`` held a canonical ``Role.id``
while ``Attribution.speaker_id`` held the analyzer's raw speaker *name*.  Read
sites compared the two for equality, so every correctly attributed segment
looked like a conflict.

This module is the single place that decides identity:

* :func:`canonical_role_id` follows the project's alias map to the canonical role;
* :func:`role_id_for_label` maps a legacy raw label onto a role only when exactly
  one role can mean it, so a genuine ambiguity fails closed instead of guessing;
* :func:`resolve_segment` reports, for one segment, whether the two stored
  representations agree, merely differ representationally, or really disagree;
* :func:`reconcile_identities` repairs representational differences in place and
  returns the entries that a human still has to decide.

A representational difference is never a reason to block casting.  A real
ambiguity isolates its own segment and is reported with evidence; it never
blocks the other characters.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from dubbing_service.app.models.project import Project, Segment

# Analyzer labels that are attribution *slots*, never people.  Kept in one place
# so the adapter, the importer and the caster agree on what a character is not.
SPEAKER_SENTINELS = frozenset(
    {
        "narrator",
        "narration",
        "unknown",
        "unattributed",
        "none",
        "null",
        "speaker",
        "speaker 1",
        "speaker_1",
        "voice 1",
        "voice_1",
        "",
    }
)

# Why a segment did not reach one canonical character.
UNRESOLVED_NO_EVIDENCE = "no_speaker_evidence"
UNRESOLVED_REAL_CONFLICT = "identity_conflict"
UNRESOLVED_HELD_BY_IMPORT = "held_for_attribution_review"


@dataclass(frozen=True)
class IdentityResolution:
    """What one segment's identity resolves to, and the evidence for it."""

    segment_id: str
    role_id: str | None
    status: str  # "resolved" | "reconciled" | "needs_attention"
    reason: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def settled(self) -> bool:
        return self.role_id is not None and self.status in {"resolved", "reconciled"}


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def canonical_role_id(project: Project, role_id: str | None) -> str | None:
    """Follow the project alias map to the canonical role, cycle-safely.

    An alias chain that loops is itself a defect, so it stops at the first
    repeated id rather than raising: the caller sees a real id, and the loop is
    reported by the research layer that owns the alias map.
    """

    current = _clean(role_id)
    if current is None:
        return None
    aliases = project.character_cast.role_aliases or {}
    seen = {current}
    while current in aliases:
        nxt = _clean(aliases[current])
        if nxt is None or nxt in seen:
            break
        seen.add(nxt)
        current = nxt
    return current


def _roles_by_id(project: Project) -> dict[str, Any]:
    return {role.id: role for role in project.roles}


def role_ids_for_label(project: Project, label: str | None) -> set[str]:
    """Every canonical role a raw analyzer label could mean.

    Returns the exact id match first; otherwise matches the project's alias map,
    the role display name, and the role's research aliases.  A label that matches
    more than one distinct canonical role is returned as a multi-element set so
    the caller can fail closed.
    """

    cleaned = _clean(label)
    if cleaned is None:
        return set()

    by_id = _roles_by_id(project)
    if cleaned in by_id:
        canonical = canonical_role_id(project, cleaned)
        return {canonical} if canonical else set()

    aliases = project.character_cast.role_aliases or {}
    if cleaned in aliases:
        canonical = canonical_role_id(project, cleaned)
        return {canonical} if canonical else set()

    folded = cleaned.casefold()
    matches: set[str] = set()
    for role in project.roles:
        if (role.name or "").strip().casefold() == folded:
            canonical = canonical_role_id(project, role.id)
            if canonical:
                matches.add(canonical)

    from .character_research import _active_aliases

    for dossier in project.character_cast.dossiers.values():
        owner = canonical_role_id(project, dossier.role_id)
        if not owner:
            continue
        for alias in _active_aliases(dossier, project.character_cast.policy):
            if (alias or "").strip().casefold() == folded:
                matches.add(owner)
    return matches


def role_id_for_label(project: Project, label: str | None) -> str | None:
    """The one canonical role a raw label means, or ``None`` when ambiguous."""

    matches = role_ids_for_label(project, label)
    if len(matches) == 1:
        return next(iter(matches))
    return None


def identity_for_token(project: Project, token: str | None) -> str | None:
    """Resolve *either* stored representation of a speaker onto one canonical id.

    ``Segment.role_id`` and ``Attribution.speaker_id`` are both supposed to hold
    a canonical ``Role.id``, but historical projects stored the analyzer's raw
    speaker name in the attribution.  A token that is a known role id or alias
    key resolves through the alias map; anything else is resolved as a display
    name, and only when exactly one role can mean it.
    """

    cleaned = _clean(token)
    if cleaned is None:
        return None
    if cleaned in _roles_by_id(project) or cleaned in (project.character_cast.role_aliases or {}):
        return canonical_role_id(project, cleaned)
    return role_id_for_label(project, cleaned)


def resolve_segment(
    project: Project, segment: Segment, *, held: bool = False
) -> IdentityResolution:
    """Resolve one segment to exactly one canonical character, or explain why not.

    ``held`` marks a segment the importer deliberately parked for human
    attribution review; the evidence is still reported so the review screen can
    show what the analyzer proposed.
    """

    assigned = identity_for_token(project, segment.role_id)
    attribution = segment.attribution
    raw_speaker = _clean(attribution.speaker_id) if attribution else None
    raw_label = _clean(attribution.label) if attribution else None

    attributed = identity_for_token(project, raw_speaker)

    evidence: dict[str, Any] = {
        "assigned_role_id": segment.role_id,
        "attribution_speaker_id": raw_speaker,
        "attribution_label": raw_label,
        "source": getattr(attribution, "source", None),
        "confidence": getattr(attribution, "confidence", None),
    }

    if held:
        return IdentityResolution(
            segment_id=segment.id,
            role_id=assigned or attributed,
            status="needs_attention",
            reason=UNRESOLVED_HELD_BY_IMPORT,
            evidence=evidence,
        )

    if assigned and attributed:
        if assigned == attributed:
            changed = segment.role_id != assigned or raw_speaker != assigned
            return IdentityResolution(
                segment_id=segment.id,
                role_id=assigned,
                status="reconciled" if changed else "resolved",
                evidence=evidence,
            )
        return IdentityResolution(
            segment_id=segment.id,
            role_id=None,
            status="needs_attention",
            reason=UNRESOLVED_REAL_CONFLICT,
            evidence=evidence,
        )

    if assigned:
        return IdentityResolution(
            segment_id=segment.id,
            role_id=assigned,
            status="reconciled",
            evidence=evidence,
        )

    if attributed:
        return IdentityResolution(
            segment_id=segment.id,
            role_id=attributed,
            status="reconciled",
            evidence=evidence,
        )

    return IdentityResolution(
        segment_id=segment.id,
        role_id=None,
        status="needs_attention",
        reason=UNRESOLVED_NO_EVIDENCE,
        evidence=evidence,
    )


@dataclass
class ReconcileReport:
    """What :func:`reconcile_identities` changed and what still needs a human."""

    resolved: int = 0
    reconciled: int = 0
    needs_attention: list[IdentityResolution] = field(default_factory=list)
    conflicts: list[IdentityResolution] = field(default_factory=list)
    held: list[IdentityResolution] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return self.reconciled

    @property
    def settled(self) -> int:
        return self.resolved + self.reconciled


def reconcile_identities(
    project: Project,
    segments: Sequence[Segment],
    *,
    held_ids: Iterable[str] | None = None,
    apply: bool = True,
) -> ReconcileReport:
    """Restore the identity invariant across live segments.

    ``segment.role_id`` and ``attribution.speaker_id`` end up carrying the same
    canonical id whenever that is knowable.  A human decision is never
    overwritten; a real conflict keeps both stored values as evidence.
    """

    held = set(held_ids or ())
    report = ReconcileReport()
    for segment in segments:
        if segment.lineage is not None and segment.lineage.retired_at is not None:
            continue
        resolution = resolve_segment(project, segment, held=segment.id in held)
        if resolution.status == "resolved":
            report.resolved += 1
            continue
        if resolution.status == "reconciled":
            report.reconciled += 1
            if apply and resolution.role_id:
                segment.role_id = resolution.role_id
                if segment.attribution is None:
                    continue
                if segment.attribution.human_authoritative:
                    # A human decided the speaker; representation is aligned but
                    # the decision itself is never rewritten.
                    continue
                segment.attribution.speaker_id = resolution.role_id
            continue
        report.needs_attention.append(resolution)
        if resolution.reason == UNRESOLVED_REAL_CONFLICT:
            report.conflicts.append(resolution)
        elif resolution.reason == UNRESOLVED_HELD_BY_IMPORT:
            report.held.append(resolution)
    return report


def needs_attention_roles(report: ReconcileReport) -> dict[str, dict[str, Any]]:
    """Group unresolved segments by the character they most plausibly concern.

    Evidence-only names are deliberately included so a human can see which
    character a held segment may belong to even though nothing is asserted.
    """

    grouped: dict[str, dict[str, Any]] = {}
    for resolution in report.needs_attention:
        candidates = []
        for key in ("assigned_role_id", "attribution_speaker_id", "attribution_label"):
            value = resolution.evidence.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
        for candidate in candidates or [None]:
            bucket = grouped.setdefault(
                candidate or "",
                {"role_id": candidate, "segment_ids": [], "reasons": set()},
            )
            bucket["segment_ids"].append(resolution.segment_id)
            if resolution.reason:
                bucket["reasons"].add(resolution.reason)
    for bucket in grouped.values():
        bucket["segment_ids"] = sorted(set(bucket["segment_ids"]))
        bucket["reasons"] = sorted(bucket["reasons"])
    return grouped
