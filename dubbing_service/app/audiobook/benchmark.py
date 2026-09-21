"""Synthetic, public-safe benchmark corpus for renderer comparison (Phase 16).

Thirteen cases cover the linguistic and structural situations that change what a
renderer must do.  Every string below was written for this project, so the corpus
can be committed, shared, and replayed against a future renderer without any
third-party licensing question.

The corpus is data plus a small materialiser and report: it lands in the normal
project authority (chapters, roles, segments, Takes), which is what makes it
reusable for comparing renderers on identical input.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.models.audiobook import BenchmarkCase
from dubbing_service.app.models.project import Chapter, Project, Role, Segment

BENCHMARK_VERSION = "corpus-v1"

NARRATOR = "Narrator"
MARA = "Mara"
TOMAS = "Tomas"
ILSE = "Ilse"
OSKAR = "Oskar"


@dataclass(frozen=True)
class BenchmarkUtterance:
    speaker: str
    text: str
    emotion: Optional[str] = None
    expected_speaker: Optional[str] = None


@dataclass(frozen=True)
class BenchmarkCaseSpec:
    case_id: str
    category: str
    title: str
    chapter: str
    expectation: str
    utterances: Sequence[BenchmarkUtterance] = field(default_factory=tuple)

    def script(self) -> str:
        return "\n".join(f"{item.speaker}: {item.text}" for item in self.utterances)

    def speakers(self) -> List[str]:
        seen: List[str] = []
        for item in self.utterances:
            if item.speaker not in seen:
                seen.append(item.speaker)
        return seen

    def segment_ids(self) -> List[str]:
        return [
            benchmark_segment_id(self.case_id, index)
            for index in range(len(self.utterances))
        ]

    def as_case(self) -> BenchmarkCase:
        return BenchmarkCase(
            case_id=self.case_id,
            category=self.category,
            title=self.title,
            text=self.script(),
            expected_speakers=self.speakers(),
            expectations={
                "behavior": self.expectation,
                "chapter": self.chapter,
                "utterance_count": len(self.utterances),
            },
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "title": self.title,
            "chapter": self.chapter,
            "expectation": self.expectation,
            "utterances": [
                {
                    "speaker": item.speaker,
                    "text": item.text,
                    "emotion": item.emotion,
                    "expected_speaker": item.expected_speaker or item.speaker,
                }
                for item in self.utterances
            ],
        }


CORPUS: Sequence[BenchmarkCaseSpec] = (
    BenchmarkCaseSpec(
        case_id="plain_narration",
        category="baseline",
        title="Plain narration",
        chapter="Chapter 1 - The Quiet Ledger",
        expectation="Unbroken narration renders as one utterance per sentence group.",
        utterances=(
            BenchmarkUtterance(
                NARRATOR,
                "The ledger stayed open on the desk for three days before anyone "
                "dared to read the last entry.",
            ),
            BenchmarkUtterance(
                NARRATOR,
                "Rain worked at the window. The lamp hummed. Nothing in the room "
                "suggested that a decision had been made.",
            ),
        ),
    ),
    BenchmarkCaseSpec(
        case_id="two_person_dialogue",
        category="dialogue",
        title="Two-person dialogue",
        chapter="Chapter 2 - Two Chairs",
        expectation="Alternating speakers keep separate voices across a short exchange.",
        utterances=(
            BenchmarkUtterance(MARA, "You moved the chairs."),
            BenchmarkUtterance(TOMAS, "I moved one of them."),
            BenchmarkUtterance(MARA, "You moved both."),
            BenchmarkUtterance(TOMAS, "The second one followed on its own."),
        ),
    ),
    BenchmarkCaseSpec(
        case_id="four_speakers",
        category="casting",
        title="Four speakers in one scene",
        chapter="Chapter 3 - Four Corners",
        expectation="Four distinct roles remain stable inside a single chapter.",
        utterances=(
            BenchmarkUtterance(ILSE, "The bridge is out past the mill."),
            BenchmarkUtterance(OSKAR, "Then we cross at the shallow bank."),
            BenchmarkUtterance(TOMAS, "The bank is worse than the bridge."),
            BenchmarkUtterance(MARA, "We cross at the bank and argue about it there."),
        ),
    ),
    BenchmarkCaseSpec(
        case_id="emotion_range",
        category="emotion",
        title="Emotion and intensity",
        chapter="Chapter 4 - Measured Fire",
        expectation="Emotional direction varies per utterance without changing identity.",
        utterances=(
            BenchmarkUtterance(NARRATOR, "She read the letter twice.", emotion="calm"),
            BenchmarkUtterance(MARA, "Read it again.", emotion="anger"),
            BenchmarkUtterance(
                TOMAS, "It says the same thing it said last night.", emotion="sadness"
            ),
            BenchmarkUtterance(
                MARA,
                "Then it will say something different tomorrow.",
                emotion="determination",
            ),
        ),
    ),
    BenchmarkCaseSpec(
        case_id="difficult_names",
        category="pronunciation",
        title="Difficult names and places",
        chapter="Chapter 5 - Proper Nouns",
        expectation="Pronunciations stay consistent for non-English names.",
        utterances=(
            BenchmarkUtterance(
                NARRATOR,
                "Zbigniew Kowalczyk-Nowak met Aisling Niamh O Suilleabhain at the "
                "harbour in Vranov nad Dyji.",
            ),
            BenchmarkUtterance(
                NARRATOR,
                "The registry spelled both names differently the following spring.",
            ),
        ),
    ),
    BenchmarkCaseSpec(
        case_id="dates_numbers_currency",
        category="normalization",
        title="Dates, numbers, and currency",
        chapter="Chapter 6 - Figures",
        expectation="Numerals expand predictably across units and formats.",
        utterances=(
            BenchmarkUtterance(
                NARRATOR,
                "On 14 March 1897 the account held 1,204 pounds 7 shillings 3 pence, "
                "or roughly 31.5 percent of the estate.",
            ),
            BenchmarkUtterance(
                NARRATOR,
                "By 1901 it had grown to 2,480 dollars and 55 cents across three "
                "separate obligations.",
            ),
        ),
    ),
    BenchmarkCaseSpec(
        case_id="quiet_whisper",
        category="dynamics",
        title="Quiet and whispered delivery",
        chapter="Chapter 7 - Below the Floorboards",
        expectation="Low-intensity lines stay intelligible and are not normalised away.",
        utterances=(
            BenchmarkUtterance(
                NARRATOR, "Someone was awake below the floorboards.", emotion="whisper"
            ),
            BenchmarkUtterance(ILSE, "Do not move.", emotion="whisper"),
            BenchmarkUtterance(OSKAR, "I heard it too.", emotion="whisper"),
        ),
    ),
    BenchmarkCaseSpec(
        case_id="long_paragraph",
        category="chunking",
        title="Long paragraph",
        chapter="Chapter 8 - The Long Entry",
        expectation="A paragraph beyond one breath is chunked without dropping text.",
        utterances=(
            BenchmarkUtterance(
                NARRATOR,
                "The argument ran the length of the corridor and back again, past "
                "the pantry, past the room where the maps were kept, past the door "
                "that nobody repaired, and it did not resolve, because neither of "
                "them had come to resolve anything; they had come to be right in "
                "front of a witness, and the witness, who had been folding linen "
                "the entire time, kept folding linen, which is its own kind of "
                "verdict. Later that evening the corridor was swept, the maps were "
                "rolled and tied, and the door that nobody repaired was still not "
                "repaired, though someone had chalked a small mark on the frame to "
                "show that it had been considered.",
            ),
        ),
    ),
    BenchmarkCaseSpec(
        case_id="repeated_short_dialogue",
        category="identity",
        title="Repeated short dialogue",
        chapter="Chapter 9 - Again",
        expectation="Repeated identical lines stay separate utterances with separate takes.",
        utterances=(
            BenchmarkUtterance(MARA, "Again."),
            BenchmarkUtterance(TOMAS, "Again."),
            BenchmarkUtterance(MARA, "Again."),
            BenchmarkUtterance(TOMAS, "Again."),
        ),
    ),
    BenchmarkCaseSpec(
        case_id="chapter_boundary",
        category="structure",
        title="Chapter boundary",
        chapter="Chapter 10 - New Leaf",
        expectation="Crossing a chapter boundary opens a new chapter without merging voices.",
        utterances=(
            BenchmarkUtterance(NARRATOR, "The next morning the ledger was closed."),
            BenchmarkUtterance(MARA, "Whose hand closed it?"),
        ),
    ),
    BenchmarkCaseSpec(
        case_id="attribution_ambiguity",
        category="attribution",
        title="Ambiguous attribution",
        chapter="Chapter 11 - Someone Said",
        expectation="Ambiguous attribution is recorded as low confidence, never as fact.",
        utterances=(
            BenchmarkUtterance(TOMAS, "No.", expected_speaker=MARA),
            BenchmarkUtterance(MARA, "Say that again, slowly.", expected_speaker=TOMAS),
            BenchmarkUtterance(NARRATOR, "Neither of them would repeat it."),
        ),
    ),
    BenchmarkCaseSpec(
        case_id="ambience_sfx_anchor",
        category="cues",
        title="Ambience and SFX anchors",
        chapter="Chapter 12 - Weather and Doors",
        expectation="Cues anchor to utterance boundaries and stay separate from dialogue.",
        utterances=(
            BenchmarkUtterance(NARRATOR, "The storm arrived before the coach."),
            BenchmarkUtterance(ILSE, "Shut the door before the lamps go out."),
            BenchmarkUtterance(NARRATOR, "The door shut itself."),
        ),
    ),
    BenchmarkCaseSpec(
        case_id="source_correction",
        category="provenance",
        title="One deliberate source correction",
        chapter="Chapter 13 - Correction",
        expectation="A corrected line changes spoken text while the source revision survives.",
        utterances=(
            BenchmarkUtterance(NARRATOR, "He counted eleven crates at the dock."),
            BenchmarkUtterance(NARRATOR, "The manifest listed the same eleven."),
        ),
    ),
)


def benchmark_segment_id(case_id: str, line_index: int) -> str:
    return f"seg_bench_{case_id}_{line_index:02d}"


def corpus_cases() -> List[BenchmarkCase]:
    return [spec.as_case() for spec in CORPUS]


def corpus_payload() -> Dict[str, object]:
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "case_count": len(CORPUS),
        "cases": [spec.as_dict() for spec in CORPUS],
    }


def corpus_fingerprint() -> str:
    """Stable hash of the corpus so two renderer runs are provably comparable."""

    return hashlib.sha256(
        json.dumps(corpus_payload(), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def corpus_utterance_count() -> int:
    return sum(len(spec.utterances) for spec in CORPUS)


def case_ids() -> List[str]:
    return [spec.case_id for spec in CORPUS]


def roles_for_corpus() -> List[str]:
    names: List[str] = []
    for spec in CORPUS:
        for speaker in spec.speakers():
            if speaker not in names:
                names.append(speaker)
    return names


def _role_id(name: str) -> str:
    return "role_bench_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]


def materialize_benchmark(project: Project) -> Dict[str, object]:
    """Add the corpus to ``project`` as chapters, roles, and segments.

    Returns the new segments plus the identifiers needed to render them, so the
    caller persists through the normal project authority.
    """

    role_by_name: Dict[str, Role] = {role.name: role for role in project.roles}
    for speaker in roles_for_corpus():
        if speaker in role_by_name:
            continue
        role = Role(id=_role_id(speaker), name=speaker)
        project.roles.append(role)
        role_by_name[speaker] = role

    chapter_by_title: Dict[str, Chapter] = {
        chapter.title: chapter for chapter in project.chapters
    }
    created_chapters: List[str] = []
    segments: List[Segment] = []
    by_case: Dict[str, List[str]] = {}

    for position, spec in enumerate(CORPUS):
        chapter = chapter_by_title.get(spec.chapter)
        if chapter is None:
            chapter = Chapter(
                id=f"chap_bench_{spec.case_id}",
                index=len(project.chapters),
                title=spec.chapter,
            )
            project.chapters.append(chapter)
            chapter_by_title[spec.chapter] = chapter
            created_chapters.append(chapter.id)

        case_segments: List[str] = []
        for line_index, item in enumerate(spec.utterances):
            segment = Segment(
                id=benchmark_segment_id(spec.case_id, line_index),
                chapter_id=chapter.id,
                role_id=role_by_name[item.speaker].id,
                index=position * 100 + line_index,
                text=item.text,
                emotion_text=item.emotion,
            )
            segments.append(segment)
            case_segments.append(segment.id)
        by_case[spec.case_id] = case_segments

    return {
        "segments": segments,
        "by_case": by_case,
        "roles": roles_for_corpus(),
        "created_chapters": created_chapters,
        "corpus_fingerprint": corpus_fingerprint(),
        "benchmark_version": BENCHMARK_VERSION,
    }


def benchmark_report(project_id: str) -> Dict[str, object]:
    """Summarise what a benchmark project currently holds, per case."""

    from dubbing_service.app.audiobook import release as release_module
    from dubbing_service.app.audiobook import takes as takes_module
    from dubbing_service.app.storage.projects import load_project
    from dubbing_service.app.storage.segments import load_segments

    project = load_project(project_id)
    segments = load_segments(project_id)
    coverage = release_module.release_report(
        segments,
        cues=project.audio_cues,
        project_root=safe_project_root(project_id),
    )

    per_case: Dict[str, Dict[str, object]] = {}
    for spec in CORPUS:
        case_ids = set(spec.segment_ids())
        case_segments = [segment for segment in segments if segment.id in case_ids]
        hashes: List[str] = []
        stale: List[str] = []
        missing: List[str] = []
        for segment in case_segments:
            take = takes_module.selected_take(segment)
            if take is None:
                missing.append(segment.id)
                continue
            hashes.append(str(take.audio_sha256 or ""))
            if getattr(take, "stale", False):
                stale.append(segment.id)
        per_case[spec.case_id] = {
            "category": spec.category,
            "utterances": len(case_segments),
            "takes": len(hashes),
            "missing": missing,
            "stale": stale,
            "audio_hashes": sorted(hashes),
        }

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "corpus_fingerprint": corpus_fingerprint(),
        "case_count": len(CORPUS),
        "utterance_count": corpus_utterance_count(),
        "cases": per_case,
        "coverage": coverage.as_dict(),
    }
