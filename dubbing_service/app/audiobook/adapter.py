"""Audiobooker adapter (Phase 3).

Audiobooker is used in **library mode**: analysis is derived/scratch, and AI
Audio Studio stays the canonical production authority.  Two hard rules:

* the platform must work with audiobooker absent — :func:`availability` reports
  ``available=False`` with the exact reason and the manual import path still
  works;
* nothing derived from audiobooker is written outside a disposable scratch
  directory.

Because audiobooker is installed in its own isolated environment (it must not
be merged into the TTS environments), analysis runs through a small subprocess
bridge.  :class:`AudiobookerAdapter` accepts an injected ``runner`` so the
adapter itself is fully testable without audiobooker installed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from dubbing_service.app.core.config import get_config
from dubbing_service.app.models.audiobook import (
    Attribution,
    AttributionSource,
    SourceAnchor,
)
from dubbing_service.app.audiobook import persistence as persistence_module
from dubbing_service.app.models.project import (
    Chapter,
    Project,
    Role,
    Segment,
)
from dubbing_service.app.storage.projects import (
    create_project,
    load_project,
    save_project,
)
from dubbing_service.app.storage.segments import load_segments

# Pinned for reproducibility; the isolated environment installs exactly this.
PINNED_AUDIOBOOKER_VERSION = "3.0.0"
ADAPTER_SCHEMA_VERSION = "audiobooker-adapter-v1"

# The isolated audiobooker environment is provisioned inside this repository.
REPO_ROOT = Path(__file__).resolve().parents[3]

RUNNER_MODULE = "dubbing_service.app.audiobook._audiobooker_runner"

ATTRIBUTION_SOURCES = frozenset({"tag", "turn", "nlp", "inline", "user", "unknown"})


class AdapterError(RuntimeError):
    """Raised when an analysis or import cannot be completed."""


@dataclass
class AnalyzedUtterance:
    text: str
    speaker: Optional[str] = None
    utterance_type: str = "narration"
    chapter_index: int = 0
    line_index: int = 0
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    attribution_source: str = "unknown"
    confidence: Optional[float] = None
    emotion: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AnalyzedUtterance":
        source = str(raw.get("attribution_source") or "unknown")
        if source not in ATTRIBUTION_SOURCES:
            source = "unknown"
        confidence = raw.get("confidence")
        return cls(
            text=str(raw.get("text") or ""),
            speaker=raw.get("speaker") or None,
            utterance_type=str(raw.get("utterance_type") or "narration"),
            chapter_index=int(raw.get("chapter_index") or 0),
            line_index=int(raw.get("line_index") or 0),
            start_pos=raw.get("start_pos"),
            end_pos=raw.get("end_pos"),
            attribution_source=source,
            confidence=None if confidence is None else float(confidence),
            emotion=raw.get("emotion") or None,
        )


@dataclass
class AnalyzedChapter:
    index: int
    title: str
    utterances: List[AnalyzedUtterance] = field(default_factory=list)
    source_file: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AnalyzedChapter":
        return cls(
            index=int(raw.get("index") or 0),
            title=str(raw.get("title") or ""),
            source_file=raw.get("source_file") or None,
            utterances=[
                AnalyzedUtterance.from_dict(item)
                for item in raw.get("utterances") or []
            ],
        )


@dataclass
class BookAnalysis:
    """Derived, disposable analysis of one source document."""

    title: str = ""
    author: Optional[str] = None
    language: Optional[str] = None
    chapters: List[AnalyzedChapter] = field(default_factory=list)
    speakers: List[str] = field(default_factory=list)
    pronunciations: Dict[str, str] = field(default_factory=dict)
    source_path: Optional[str] = None
    schema: str = ADAPTER_SCHEMA_VERSION
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BookAnalysis":
        chapters = [AnalyzedChapter.from_dict(item) for item in raw.get("chapters") or []]
        speakers = list(raw.get("speakers") or [])
        if not speakers:
            seen: List[str] = []
            for chapter in chapters:
                for utterance in chapter.utterances:
                    if utterance.speaker and utterance.speaker not in seen:
                        seen.append(utterance.speaker)
            speakers = seen
        return cls(
            title=str(raw.get("title") or ""),
            author=raw.get("author") or None,
            language=raw.get("language") or None,
            chapters=chapters,
            speakers=speakers,
            pronunciations=dict(raw.get("pronunciations") or {}),
            source_path=raw.get("source_path") or None,
            warnings=list(raw.get("warnings") or []),
        )

    def utterance_count(self) -> int:
        return sum(len(chapter.utterances) for chapter in self.chapters)


@dataclass
class ImportResult:
    project_id: str
    chapters_created: int = 0
    roles_created: int = 0
    utterances_created: int = 0
    utterances_updated: int = 0
    human_corrections_preserved: int = 0
    skipped: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "chapters_created": self.chapters_created,
            "roles_created": self.roles_created,
            "utterances_created": self.utterances_created,
            "utterances_updated": self.utterances_updated,
            "human_corrections_preserved": self.human_corrections_preserved,
            "skipped": list(self.skipped),
        }


@dataclass
class Availability:
    available: bool
    reason: str = ""
    interpreter: Optional[str] = None
    version: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "interpreter": self.interpreter,
            "version": self.version,
            "pinned_version": PINNED_AUDIOBOOKER_VERSION,
        }


def _interpreter_candidates() -> List[Path]:
    """Candidate interpreters for the isolated audiobooker environment.

    The isolated environment is provisioned next to the *code*, so the search
    must not depend on the projects data directory (which tests, smoke runs and
    deployments all relocate).  ``DUBBING_AUDIOBOOKER_PYTHON`` still wins, and
    the current interpreter is the last resort so the probe can report a
    precise reason instead of guessing.
    """

    candidates: List[Path] = []
    env_value = os.environ.get("DUBBING_AUDIOBOOKER_PYTHON")
    if env_value:
        candidates.append(Path(env_value))
    for root in (REPO_ROOT, Path(get_config().projects_root_abs).parent):
        candidates.extend(
            [
                root / ".venvs" / "audiobooker" / "Scripts" / "python.exe",
                root / ".venvs" / "audiobooker" / "bin" / "python",
            ]
        )
    candidates.append(Path(sys.executable))

    unique: List[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def isolated_interpreter() -> Optional[Path]:
    """The isolated audiobooker interpreter, if it has been provisioned."""

    for candidate in _interpreter_candidates():
        if candidate.is_file():
            return candidate
    return None


def _probe_version(interpreter: Path) -> Availability:
    try:
        completed = subprocess.run(
            [
                str(interpreter),
                "-c",
                "import audiobooker, json; print(json.dumps(getattr(audiobooker, '__version__', 'unknown')))",
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Availability(False, f"Audiobooker 解释器无法启动: {exc}", str(interpreter))
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        reason = detail[-1] if detail else "import audiobooker 失败"
        return Availability(False, f"Audiobooker 不可用: {reason}", str(interpreter))
    version = completed.stdout.strip().strip('"') or "unknown"
    return Availability(True, "", str(interpreter), version)


def availability(*, refresh: bool = False) -> Availability:
    """Report audiobooker availability without making it a startup dependency."""

    global _AVAILABILITY_CACHE
    if _AVAILABILITY_CACHE is not None and not refresh:
        return _AVAILABILITY_CACHE
    interpreter = isolated_interpreter()
    if interpreter is None:
        result = Availability(False, "未找到隔离的 Audiobooker 解释器")
    else:
        result = _probe_version(interpreter)
    _AVAILABILITY_CACHE = result
    return result


_AVAILABILITY_CACHE: Optional[Availability] = None


def _default_runner(payload: Dict[str, Any], interpreter: Path) -> Dict[str, Any]:
    """Run the analyzer inside the isolated environment and read JSON back."""

    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["DUBBING_AUDIOBOOKER_PAYLOAD"] = json.dumps(payload, ensure_ascii=False)
    repo_root = Path(__file__).resolve().parents[3]
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(repo_root) if not existing else os.pathsep.join([str(repo_root), existing])
    )
    try:
        completed = subprocess.run(
            [str(interpreter), "-m", RUNNER_MODULE],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=1800,
            env=environment,
            cwd=str(payload.get("scratch_dir") or repo_root),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AdapterError(f"Audiobooker 子进程失败: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        raise AdapterError(
            "Audiobooker 分析失败: " + (detail[-1] if detail else str(completed.returncode))
        )
    try:
        payload_out = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AdapterError(f"Audiobooker 返回的 JSON 无效: {exc}") from exc
    if isinstance(payload_out, dict):
        # audiobooker reports its real quality caveats on stderr — "42% of
        # DIALOGUE lines ... were inferred by alternating turns — a guess, not a
        # reading".  Dropping them on a successful exit hid the fact that a
        # large share of the attributions were guesses, so they are folded into
        # the analysis instead of discarded.
        diagnostics = [
            line.strip()
            for line in (completed.stderr or "").splitlines()
            if line.strip()
        ]
        if diagnostics:
            warnings = payload_out.get("warnings")
            if not isinstance(warnings, list):
                warnings = []
            payload_out["warnings"] = warnings + diagnostics
    return payload_out if isinstance(payload_out, dict) else {}


@dataclass
class AudiobookerAdapter:
    """Analyze a source document and reconcile it into a studio project."""

    runner: Optional[Callable[[Dict[str, Any], Path], Dict[str, Any]]] = None
    interpreter: Optional[Path] = None
    scratch_root: Optional[Path] = None

    def resolve_interpreter(self) -> Optional[Path]:
        if self.interpreter is not None:
            return self.interpreter
        return isolated_interpreter()

    def scratch_directory(self, label: str = "") -> Path:
        base = self.scratch_root or (Path(get_config().projects_root_abs) / "_scratch")
        target = base / "audiobooker" / f"{label or 'analysis'}-{uuid.uuid4().hex[:12]}"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def analyze_source(
        self,
        source_path: Path | str,
        *,
        title: Optional[str] = None,
        author: Optional[str] = None,
        language: Optional[str] = None,
        keep_scratch: bool = False,
    ) -> BookAnalysis:
        """Analyze an epub/pdf/docx/txt source.  Raises :class:`AdapterError`."""

        path = Path(source_path)
        if not path.is_file():
            raise AdapterError(f"源文件不存在: {path}")
        interpreter = self.resolve_interpreter()
        runner = self.runner or _default_runner
        if interpreter is None and self.runner is None:
            raise AdapterError(availability(refresh=True).reason or "Audiobooker 不可用")

        scratch = self.scratch_directory(path.stem)
        payload = {
            "source_path": str(path),
            "title": title,
            "author": author,
            "language": language,
            "scratch_dir": str(scratch),
        }
        try:
            raw = runner(payload, interpreter)  # type: ignore[arg-type]
        finally:
            if not keep_scratch:
                shutil.rmtree(scratch, ignore_errors=True)
        if not isinstance(raw, dict):
            raise AdapterError("Audiobooker 返回的分析结果不是对象")
        analysis = BookAnalysis.from_dict(raw)
        analysis.source_path = str(path)
        return analysis

    def analyze_text(
        self,
        text: str,
        *,
        title: str = "",
        author: Optional[str] = None,
        language: Optional[str] = None,
        keep_scratch: bool = False,
    ) -> BookAnalysis:
        """Analyze raw text (used by the CLI and by tests)."""

        interpreter = self.resolve_interpreter()
        runner = self.runner or _default_runner
        if interpreter is None and self.runner is None:
            raise AdapterError(availability(refresh=True).reason or "Audiobooker 不可用")
        scratch = self.scratch_directory("text")
        payload = {
            "text": text,
            "title": title,
            "author": author,
            "language": language,
            "scratch_dir": str(scratch),
        }
        try:
            raw = runner(payload, interpreter)  # type: ignore[arg-type]
        finally:
            if not keep_scratch:
                shutil.rmtree(scratch, ignore_errors=True)
        if not isinstance(raw, dict):
            raise AdapterError("Audiobooker 返回的分析结果不是对象")
        return BookAnalysis.from_dict(raw)

    # ---- import boundary -------------------------------------------------

    def reconcile_into_project(
        self,
        analysis: BookAnalysis,
        *,
        project_id: Optional[str] = None,
        project_name: Optional[str] = None,
        source_revision_id: Optional[str] = None,
    ) -> ImportResult:
        """Create or update the canonical studio project from an analysis.

        Import never overwrites a human-attributed utterance; it records the
        new proposal as derived data and reports how many were preserved.
        """

        if project_id is None:
            project = create_project(project_name or analysis.title or "audiobook")
            project_id = project.id
        else:
            project = load_project(project_id)

        project.metadata = dict(project.metadata or {})
        if analysis.title:
            project.metadata.setdefault("source_title", analysis.title)
        if analysis.author:
            project.metadata.setdefault("source_author", analysis.author)
        if analysis.language:
            project.metadata.setdefault("source_language", analysis.language)
        if analysis.source_path:
            project.metadata["source_path"] = analysis.source_path
        project.metadata["audiobooker_schema"] = analysis.schema
        project.metadata["audiobooker_pinned_version"] = PINNED_AUDIOBOOKER_VERSION
        record_pronunciation_suggestions(project, analysis)

        result = ImportResult(project_id=project_id)
        existing_roles = {role.name: role for role in project.roles}
        role_by_speaker: Dict[str, Role] = {}
        for speaker in _declared_speakers(analysis):
            name = speaker.strip() or "未指定"
            role = existing_roles.get(name)
            if role is None:
                role = Role(id=_role_id(name), name=name)
                project.roles.append(role)
                existing_roles[name] = role
                result.roles_created += 1
            role_by_speaker[speaker] = role
        narrator_role = _narrator_role(project, existing_roles)

        segments = load_segments(project_id)
        by_id = {segment.id: segment for segment in segments}
        chapter_by_index = {chapter.index: chapter for chapter in project.chapters}
        # Position is identity: each analyzed utterance binds to a segment at
        # most once per import pass.  Without this, two utterances that happen
        # to share the same text (scene breaks, ``said Haoyu.``) both resolve to
        # the first one's unanchored segment, and the later speaker is silently
        # dropped from the project.
        claimed: Set[str] = set()
        for chapter in analysis.chapters:
            target = chapter_by_index.get(chapter.index)
            if target is None:
                target = Chapter(
                    id=_chapter_id(chapter.index, chapter.title),
                    index=chapter.index,
                    title=chapter.title or f"Chapter {chapter.index + 1}",
                )
                project.chapters.append(target)
                chapter_by_index[chapter.index] = target
                result.chapters_created += 1
            elif chapter.title and not target.title:
                target.title = chapter.title
            for utterance in chapter.utterances:
                if not utterance.text.strip():
                    continue
                _merge_utterance(
                    segments,
                    by_id,
                    utterance,
                    chapter=target,
                    role_by_speaker=role_by_speaker,
                    default_role=narrator_role,
                    source_revision_id=source_revision_id,
                    result=result,
                    claimed=claimed,
                )

        persistence_module.commit_project_state(
            project_id, project=project, segments=segments
        )
        return result


# Audiobooker (and any future provider) names its attribution methods its own
# way.  The studio has one controlled vocabulary for *why* a speaker was
# chosen, so foreign labels are mapped onto it rather than stored verbatim.
_ATTRIBUTION_SOURCES = {
    "tag": "tag",
    "turn": "turn",
    "nlp": "nlp",
    "inline": "inline",
    "user": "user",
    "unknown": "unknown",
    "explicit_marker": "inline",
    "explicit": "inline",
    "marker": "inline",
    "model": "nlp",
    "ml": "nlp",
    "heuristic": "nlp",
    "human": "user",
}


def attribution_source(value: Optional[str]) -> "AttributionSource":
    """Map a provider attribution label onto the studio's controlled vocabulary."""

    if not value:
        return "unknown"
    return _ATTRIBUTION_SOURCES.get(str(value).strip().lower(), "unknown")  # type: ignore[return-value]


def _role_id(name: str) -> str:
    import hashlib

    return "role_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]


def _chapter_id(index: int, title: str) -> str:
    import hashlib

    return "chap_" + hashlib.sha256(f"{index}:{title}".encode("utf-8")).hexdigest()[:12]


def _segment_id(chapter: Chapter, line_index: int) -> str:
    """Stable utterance identity: chapter + position, never the text.

    An ordinary text edit (fixing a typo, changing wording) must preserve
    ``segment.id`` — that is the whole point of the source/spoken-text revision
    split.  Structural changes (split, merge, reorder) mint new ids through
    :mod:`dubbing_service.app.audiobook.identity` instead.
    """

    import hashlib

    payload = f"{chapter.id}:{int(line_index)}"
    return "seg_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _lookup_segment(
    segments: List[Segment],
    by_id: Dict[str, Segment],
    *,
    chapter: Chapter,
    utterance: AnalyzedUtterance,
    claimed: "Set[str]",
) -> Optional[Segment]:
    """Find the segment an analyzed utterance belongs to.

    Position first, then the recorded anchor, then an unanchored text match so
    projects imported before anchors existed still reconcile in place.

    A segment already resolved by an earlier utterance in this same pass is
    never re-matched by text: text is not identity, so the target position gets
    its own segment instead of the earlier position's segment being reused and
    the utterance lost.
    """

    position_id = _segment_id(chapter, utterance.line_index)
    existing = by_id.get(position_id)
    if existing is not None:
        return existing
    for candidate in segments:
        if candidate.chapter_id != chapter.id:
            continue
        anchor = candidate.source_anchor
        if anchor is not None and anchor.occurrence_index == utterance.line_index:
            return candidate
    for candidate in segments:
        if candidate.chapter_id != chapter.id or candidate.source_anchor is not None:
            continue
        if candidate.text == utterance.text and candidate.id not in claimed:
            return candidate
    return None


def _narrator_role(project: Project, existing_roles: Dict[str, Role]) -> Role:
    """The role unattributed narration is spoken by.

    Unattributed text stays *narration* (nothing infers a character), but it
    still needs a voice before it can be rendered, so it is routed to the
    project's narrator role instead of being left unroutable.
    """

    for role in project.roles:
        if getattr(role.role_type, "value", role.role_type) == "narrator":
            return role
    role = existing_roles.get("Narrator")
    if role is not None:
        return role
    role = Role(id=_role_id("Narrator"), name="Narrator", role_type="narrator")
    project.roles.append(role)
    existing_roles["Narrator"] = role
    return role


def _declared_speakers(analysis: BookAnalysis) -> List[str]:
    """Every speaker name that must own a role, in first-seen order.

    ``analysis.speakers`` is the provider's *casting table*, which is not
    guaranteed to list the characters its own attributions name — audiobooker
    3.0.0 discovers character names during ``compile()`` but leaves the casting
    table holding only the narrator.  A speaker named by an utterance but
    missing from the casting table would get no role, and an unrouted utterance
    cannot be cast, so it could never be rendered with that character's voice.
    Declared and attributed speakers are therefore unioned.
    """

    names: List[str] = []
    seen: Set[str] = set()
    attributed = [
        utterance.speaker
        for chapter in analysis.chapters
        for utterance in chapter.utterances
    ]
    for raw in list(analysis.speakers) + attributed:
        name = (raw or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _merge_utterance(
    segments: List[Segment],
    by_id: Dict[str, Segment],
    utterance: AnalyzedUtterance,
    *,
    chapter: Chapter,
    role_by_speaker: Dict[str, Role],
    default_role: Optional[Role],
    source_revision_id: Optional[str],
    result: ImportResult,
    claimed: "Set[str]",
) -> None:
    segment_id = _segment_id(chapter, utterance.line_index)
    existing = _lookup_segment(
        segments, by_id, chapter=chapter, utterance=utterance, claimed=claimed
    )
    role = role_by_speaker.get(utterance.speaker or "", None)
    if role is None and not utterance.speaker:
        role = default_role
    source = attribution_source(utterance.attribution_source)
    attribution = Attribution(
        speaker_id=utterance.speaker,
        source=source,
        confidence=utterance.confidence,
        reviewed=source == "user",
    )

    if existing is None:
        existing = Segment(
            id=segment_id,
            chapter_id=chapter.id,
            role_id=role.id if role is not None else None,
            index=utterance.line_index,
            text=utterance.text,
            source_anchor=SourceAnchor(
                source_revision_id=source_revision_id or "",
                chapter_id=chapter.id,
                start_offset=utterance.start_pos,
                end_offset=utterance.end_pos,
                occurrence_index=utterance.line_index,
            )
            if source_revision_id
            else None,
            attribution=attribution,
        )
        segments.append(existing)
        by_id[existing.id] = existing
        result.utterances_created += 1
    else:
        previous = existing.attribution
        if previous is not None and previous.human_authoritative:
            result.human_corrections_preserved += 1
        else:
            existing.attribution = attribution
            if role is not None:
                existing.role_id = role.id
            result.utterances_updated += 1
        existing.index = utterance.line_index

    # Either way this position now owns this segment: a later utterance with the
    # same text must not be able to claim it back through the text fallback.
    claimed.add(existing.id)

    if source_revision_id:
        existing.source_anchor = SourceAnchor(
            source_revision_id=source_revision_id,
            chapter_id=chapter.id,
            start_offset=utterance.start_pos,
            end_offset=utterance.end_pos,
            occurrence_index=utterance.line_index,
        )
    if utterance.emotion and not existing.emotion_text:
        existing.emotion_text = utterance.emotion
    if existing.role_id is None and role is not None:
        existing.role_id = role.id


def pronunciation_suggestions(analysis: BookAnalysis, project: Project) -> Dict[str, str]:
    """Pronunciation suggestions recorded as derived metadata.

    Suggestions never silently rewrite the canonical pronunciation dictionary:
    the studio's ``ProductionSettings.pronunciation_dictionary`` stays under
    human control and is applied through :func:`apply_pronunciation_suggestions`.
    """

    merged = dict(project.metadata.get("pronunciation_suggestions") or {})
    merged.update(analysis.pronunciations)
    return merged


def record_pronunciation_suggestions(
    project: Project, analysis: BookAnalysis
) -> List[str]:
    """Store suggestions in project metadata and return the new keys."""

    existing = dict(project.metadata.get("pronunciation_suggestions") or {})
    added = [key for key in analysis.pronunciations if key not in existing]
    existing.update(analysis.pronunciations)
    project.metadata["pronunciation_suggestions"] = existing
    return added


def apply_pronunciation_suggestions(project_id: str) -> List[str]:
    """Move reviewed suggestions into the production pronunciation dictionary."""

    from dubbing_service.app.storage.production import load_settings, save_settings

    project = load_project(project_id)
    suggestions = dict(project.metadata.get("pronunciation_suggestions") or {})
    if not suggestions:
        return []
    settings = load_settings(project_id)
    settings.pronunciation_dictionary = {
        **suggestions,
        **dict(settings.pronunciation_dictionary or {}),
    }
    save_settings(project_id, settings)
    project.metadata["pronunciation_applied"] = sorted(suggestions)
    save_project(project)
    return sorted(suggestions)


def scratch_directories(project_root: Path) -> Sequence[Path]:
    """Disposable audiobooker scratch locations inside one project root."""

    return [Path(project_root) / "_scratch" / "audiobooker"]


# ---------------------------------------------------------------------------
# Deterministic offline analysis
# ---------------------------------------------------------------------------
#
# Audiobooker may be absent (it is a separately licensed dependency, ADAPTER
# section of the brief).  The CLI still needs a file/fixture input path, so this
# analyzer produces the *same* ``BookAnalysis`` shape with plain, dependency-free
# rules.  Every utterance it emits is labelled ``offline_splitter`` so downstream
# code and reviewers can always tell it apart from audiobooker output; it never
# pretends to be semantic attribution.

OFFLINE_ANALYSIS_SOURCE = "offline_splitter"
AUDIOBOOKER_ANALYSIS_SOURCE = "audiobooker"


def analysis_source(analysis: BookAnalysis) -> str:
    """Report which analyzer actually produced an analysis.

    Callers must not infer this from ``availability()``: the library can be
    installed and still fail, in which case the offline fallback did the work.
    """

    if any(OFFLINE_ANALYSIS_SOURCE in warning for warning in analysis.warnings):
        return OFFLINE_ANALYSIS_SOURCE
    return AUDIOBOOKER_ANALYSIS_SOURCE


_CHAPTER_PATTERNS = (
    re.compile(r"^(?:chapter|part|section|book)\b[^a-z0-9]*([0-9ivxlcdm]+)?", re.I),
    re.compile(r"^[0-9]+\.\s+\S"),
    re.compile(r"^#{1,3}\s+\S"),
)
_BRACKET_SPEAKER_PATTERN = re.compile(
    r"^\[(?P<speaker>[^\]]{1,40})\](?:\s*[:—–-]\s*|\s+)(?P<line>.+)$"
)
_PLAIN_SPEAKER_PATTERN = re.compile(
    r"^(?P<speaker>[A-Z][\w'’-]*(?: [A-Z][\w'’-]*){0,2})\s*[:—–-]\s+(?P<line>.+)$"
)


def _match_speaker(line: str) -> Optional[tuple[str, str]]:
    """Return ``(speaker, spoken line)`` for an explicitly marked line."""

    for pattern in (_BRACKET_SPEAKER_PATTERN, _PLAIN_SPEAKER_PATTERN):
        match = pattern.match(line)
        if match is not None:
            speaker = (match.group("speaker") or "").strip()
            body = match.group("line").strip()
            if speaker and body:
                return speaker, body
    return None
_QUOTE_PATTERN = re.compile(r'[“"]([^”"]+)[”"]')


def _looks_like_chapter(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    return any(pattern.match(stripped) for pattern in _CHAPTER_PATTERNS)


def _split_sentences(block: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", block.strip())
    return [part for part in (item.strip() for item in parts) if part]


def analyze_text_offline(
    text: str,
    *,
    title: str = "",
    author: Optional[str] = None,
    language: Optional[str] = None,
) -> BookAnalysis:
    """Analyze text without audiobooker using deterministic, transparent rules.

    Chapter headings are recognised by common prefixes; dialogue is recognised by
    ``Speaker: line`` or bracketed ``[Speaker] line`` prefixes; everything else
    stays narration.  Attribution confidence is left unset except for explicit
    speaker prefixes, which are recorded as ``explicit_marker``.
    """

    chapters: List[AnalyzedChapter] = []
    speakers: List[str] = []
    current: Optional[AnalyzedChapter] = None
    line_index = 0

    def ensure_chapter() -> AnalyzedChapter:
        nonlocal current
        if current is None:
            current = AnalyzedChapter(index=0, title=title or "Chapter 1")
            chapters.append(current)
        return current

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if _looks_like_chapter(line):
            current = AnalyzedChapter(index=len(chapters), title=line)
            chapters.append(current)
            line_index = 0
            continue
        chapter = ensure_chapter()
        marked = _match_speaker(line)
        speaker: Optional[str] = None
        body = line
        source = "unknown"
        confidence: Optional[float] = None
        if marked is not None:
            speaker, body = marked
            source = "explicit_marker"
            confidence = 1.0
        for index, sentence in enumerate(_split_sentences(body)):
            if not sentence:
                continue
            chapter.utterances.append(
                AnalyzedUtterance(
                    text=sentence,
                    speaker=speaker,
                    utterance_type="dialogue" if speaker else "narration",
                    chapter_index=chapter.index,
                    line_index=line_index + index,
                    start_pos=None,
                    end_pos=None,
                    attribution_source=source,
                    confidence=confidence,
                )
            )
            if speaker and speaker not in speakers:
                speakers.append(speaker)
        line_index += max(1, len(_split_sentences(body)))

    if not chapters:
        chapters.append(AnalyzedChapter(index=0, title=title or "Chapter 1"))

    return BookAnalysis(
        title=title,
        author=author,
        language=language,
        chapters=chapters,
        speakers=speakers,
        pronunciations={},
        warnings=[f"analysis produced by {OFFLINE_ANALYSIS_SOURCE}, not audiobooker"],
    )


def analyze_source_file(
    path: Path | str,
    *,
    title: str = "",
    author: Optional[str] = None,
    language: Optional[str] = None,
    prefer_audiobooker: bool = True,
    adapter: Optional["AudiobookerAdapter"] = None,
) -> BookAnalysis:
    """Analyze a file, using audiobooker when available and falling back offline."""

    source = Path(path)
    if not source.is_file():
        raise AdapterError(f"源文件不存在: {source}")
    reader = adapter or AudiobookerAdapter()
    if prefer_audiobooker and availability().available:
        return reader.analyze_source(
            source, title=title or source.stem, author=author, language=language
        )
    text = source.read_text(encoding="utf-8")
    return analyze_text_offline(
        text, title=title or source.stem, author=author, language=language
    )
