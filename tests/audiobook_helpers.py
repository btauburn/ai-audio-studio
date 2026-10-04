"""Shared fixtures for the audiobook v1.2 test suites.

Everything here builds *real* studio objects through the public storage API, so
the tests exercise the same canonical project/segment files the service uses.
"""

from __future__ import annotations

import os
import struct
import wave
from pathlib import Path
from typing import List, Optional, Sequence

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.models.project import Chapter, Project, Role, Segment
from dubbing_service.app.storage.projects import create_project, save_project
from dubbing_service.app.storage.segments import save_segments


def use_projects_root(monkeypatch, tmp_path: Path) -> Path:
    """Point the service at a disposable projects root."""

    root = Path(tmp_path) / "projects"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(root))
    clear_config_caches()
    return root


def make_tone_wav(
    path: Path,
    *,
    seconds: float = 0.5,
    rate: int = 24000,
    amplitude: float = 0.4,
    frequency: float = 220.0,
) -> Path:
    """A short audible PCM WAV (a pure silence blob would fail QA by design)."""

    import math

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(rate * seconds)
    samples = [
        int(amplitude * 32767 * math.sin(2 * math.pi * frequency * index / rate))
        for index in range(frames)
    ]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack(f"<{frames}h", *samples))
    return path


def make_silence_wav(path: Path, *, seconds: float = 0.5, rate: int = 24000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * frames)
    return path


def build_project(
    *,
    name: str = "audiobook",
    chapter_titles: Sequence[str] = ("Chapter 1",),
    roles: Optional[Sequence[tuple[str, str]]] = None,
) -> Project:
    project = create_project(name)
    project.chapters = [
        Chapter(id=f"chap{index}", title=title, index=index)
        for index, title in enumerate(chapter_titles)
    ]
    for role_id, role_name in roles or [("narrator", "Narrator")]:
        project.roles.append(
            Role(id=role_id, name=role_name, role_type="narrator")
        )
    save_project(project)
    return project


def build_segments(
    project: Project,
    texts: Sequence[str],
    *,
    role_id: str = "narrator",
    chapter_index: int = 0,
    start_index: int = 0,
) -> List[Segment]:
    chapter = project.chapters[chapter_index]
    segments = [
        Segment(
            id=f"seg{start_index + offset:03d}",
            chapter_id=chapter.id,
            role_id=role_id,
            index=start_index + offset,
            text=text,
        )
        for offset, text in enumerate(texts)
    ]
    save_segments(project.id, segments)
    return segments


def anchored_candidate(candidate_id: str = "original", *, index: int = 0, count: int = 1,
                       design_instruction: str | None = None, **extra) -> dict:
    """One shortlist candidate in the shape production now creates.

    Since the performer-reference layer, a generated candidate is always cast
    against a specific performance and always carries the acoustic
    characterization derived from it; a fixture that omits both is a shape
    production can no longer emit. The values come from the real anchoring,
    inference and reference modules, so a test exercises the same numbers.
    """
    from dubbing_service.app.audiobook import casting_anchors, performer_reference

    anchor = casting_anchors.anchored_plan(None, None, count=count)[index]
    concept = anchor["concept"]
    evidence = performer_reference.resolve(anchor, concept)
    candidate = {
        "candidate_id": candidate_id,
        "original_casting": True,
        "design_instruction": design_instruction or (
            f"Create an original synthetic voice. {casting_anchors.design_fragment(anchor)}"),
        "creative_direction": "Clear natural delivery.",
        "anchor": anchor,
        "concept": concept,
        "reference": performer_reference.describe(evidence),
        "conditioning": performer_reference.conditioning_payload(evidence),
        "match_rationale": evidence["rationale"],
        "inspiration": {"performer": anchor["performer"], "performance": anchor["performance"],
                        "source_urls": list(anchor.get("source_urls") or [])},
        "tags": list(anchor.get("tags") or []),
        "source_urls": list(anchor.get("source_urls") or []),
        "score": 1,
    }
    candidate.update(extra)
    return candidate


def reload(project: Project):
    from dubbing_service.app.storage.projects import load_project
    from dubbing_service.app.storage.segments import load_segments

    return load_project(project.id), load_segments(project.id)


def clean_environment() -> None:
    os.environ.pop("DUBBING_AUDIOBOOKER_PAYLOAD", None)
    clear_config_caches()
