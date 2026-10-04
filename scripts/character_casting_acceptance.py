"""Resumable bounded acceptance through production character/casting APIs.

Run with the root environment. Start the existing qwen3tts worker before audio.
All private text, reports, and generated audio stay outside versioned source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubbing_service.app.audiobook.adapter import (
    AudiobookerAdapter,
    BookAnalysis,
    analyze_text_offline,
)
from dubbing_service.app.audiobook.character_casting import (
    cast_report,
    markdown_report,
    prepare_cast,
)
from dubbing_service.app.audiobook.pipeline import render_utterances
from dubbing_service.app.audiobook.renderer import BackendRenderer
from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.storage.projects import (
    create_project,
    load_project,
    save_project,
)
from dubbing_service.app.storage.segments import load_segments

SYNTHETIC = """Chapter 1
Morgan was a woman. Morgan was thirty years old. Morgan's voice was clear and high-pitched.
Morgan spoke slowly. Morgan was a cautious scholar.
Morgan: The door is open. We can leave when you are ready.
Jules was a man. Jules was sixty years old. Jules's voice was deep and rough.
Jules was a confident teacher.
Jules: We should take the old path through the valley.
Morgan: I remember that day. We have time to make things right.
Jules: Let us go together. Nobody should have to walk alone.
Chapter 2
Morgan: The morning is bright. I think we will reach home soon.
Jules: I can see the bridge from here. There is our way home.
Morgan: Please wait a moment. I want to remember this view.
Jules: Take your time. We are finally safe now.
"""


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_suffix(path.suffix + ".stage")
    stage.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    stage.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--audio-characters", type=int, default=2, choices=range(9))
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.file if args.file and args.file.is_file() else None
    digest = hashlib.sha256(
        source.read_bytes() if source else SYNTHETIC.encode()
    ).hexdigest()
    record = out / "acceptance-state.json"
    state = (
        json.loads(record.read_text(encoding="utf-8"))
        if record.exists()
        else {
            "source_sha256": digest,
            "source": str(source) if source else "legal synthetic fixture",
            "substitution": source is None,
            "projects": {},
        }
    )
    if state["source_sha256"] != digest:
        raise ValueError(
            "Acceptance directory contains a different source; choose a new output directory"
        )
    analysis_path = out / "analysis.json"
    if analysis_path.exists():
        analysis = BookAnalysis.from_dict(
            json.loads(analysis_path.read_text(encoding="utf-8"))
        )
    else:
        print("Analyzing source through existing Audiobooker adapter", flush=True)
        analysis = (
            AudiobookerAdapter().analyze_source(source, language="en")
            if source
            else analyze_text_offline(
                SYNTHETIC, title="Original synthetic casting fixture", language="en"
            )
        )
        write(analysis_path, asdict(analysis))
    print(
        f"Analysis: {len(analysis.chapters)} chapters, {analysis.utterance_count()} utterances",
        flush=True,
    )
    summaries = {}
    for mode, palette in (("default", None), ("inspired", "harry-potter-film-cast")):
        pid = state["projects"].get(mode)
        if pid is None:
            project = create_project(f"Casting acceptance {mode}: {analysis.title}")
            project.character_cast.policy.inspiration_palette = palette
            project.character_cast.policy.max_candidates = 2
            project.character_cast.policy.max_audition_characters = (
                args.audio_characters
            )
            project.character_cast.policy.max_chapter = max(
                c.index for c in analysis.chapters
            )
            save_project(project)
            pid = project.id
            state["projects"][mode] = pid
            write(record, state)
        previous = load_project(pid)
        preexisting = {
            r.id: r.voice_identity.model_dump(mode="json")
            for r in previous.roles
            if r.voice_identity
        }
        print(f"{mode}: reconcile {pid}", flush=True)
        AudiobookerAdapter().reconcile_into_project(analysis, project_id=pid)
        project = load_project(pid)
        roles = sorted(
            project.character_cast.profiles,
            key=lambda r: (-project.character_cast.profiles[r].dialogue_count, r),
        )[: args.audio_characters]
        if not args.no_audio:
            print(f"{mode}: bounded auditions for {len(roles)} characters", flush=True)
            prepare_cast(pid, role_ids=roles, max_characters=args.audio_characters)
        project = load_project(pid)
        locked_before = {
            r.id: r.voice_identity.model_dump(mode="json")
            for r in project.roles
            if r.voice_identity
        }
        reloaded = load_project(pid)
        locked_after = {
            r.id: r.voice_identity.model_dump(mode="json")
            for r in reloaded.roles
            if r.voice_identity
        }
        assert locked_after == locked_before
        assert all(
            locked_after.get(r) == identity for r, identity in preexisting.items()
        ), "Reprocessing replaced an accepted identity"
        render_checks = []
        if not args.no_audio:
            renderer = BackendRenderer(
                backend="qwen3tts",
                engine_id="qwen3tts",
                project_root=safe_project_root(pid),
            )
            for role_id in roles:
                if role_id not in locked_after:
                    continue
                choices = [
                    s
                    for s in load_segments(pid)
                    if s.role_id == role_id and 25 <= len(s.text) <= 110
                ]
                if not choices:
                    continue
                segment = min(choices, key=lambda s: abs(len(s.text) - 60))
                print(f"{mode}: reload/render one short line for {role_id}", flush=True)
                records = render_utterances(
                    pid, [segment.id], renderer=renderer, resume=True
                )
                take = (
                    next(s for s in load_segments(pid) if s.id == segment.id).takes[-1]
                    if records[0].success
                    else None
                )
                identity_matches = bool(
                    take
                    and take.generation_metadata.get(
                        "character_voice_identity", {}
                    ).get("id")
                    == locked_after[role_id]["id"]
                )
                render_checks.append(
                    {
                        "role_id": role_id,
                        "identity_matches": identity_matches,
                        "records": [r.as_dict() for r in records],
                        "audio": take.audio_file if take else None,
                    }
                )
        report = cast_report(pid)
        write(out / f"{mode}-cast.json", report)
        (out / f"{mode}-cast.md").write_text(markdown_report(report), encoding="utf-8")
        summaries[mode] = {
            "project_id": pid,
            "project_root": str(safe_project_root(pid)),
            "characters": len(report["characters"]),
            "locked": report["locked"],
            "collision_count": len(report["collisions"]),
            "reload_equal": locked_before == locked_after,
            "reload_population": len(locked_before),
            "render_checks": render_checks,
            "preexisting_identities_preserved": True,
            "preexisting_population": len(preexisting),
            "requested_audio_characters": len(roles),
            "pending_audio_roles": [r for r in roles if r not in locked_after],
            "audition_count": sum(len(r["auditions"]) for r in report["characters"]),
            "research_facts": sum(
                sum(f["level"] != "UNKNOWN" for f in r["research"]["facts"])
                for r in report["characters"]
            ),
        }
        write(out / "acceptance-results.json", {**state, "results": summaries})
    lines = [
        "# Character casting acceptance",
        "",
        f"Source: {state['source']}",
        f"Synthetic substitution: {state['substitution']}",
        "",
        "| Mode | Dossiers | Evidenced facts | Locked voices | Audition candidates | Collisions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, row in summaries.items():
        lines.append(
            f"| {mode} | {row['characters']} | {row['research_facts']} | {row['locked']} | {row['audition_count']} | {row['collision_count']} |"
        )
    lines.extend(
        [
            "",
            "No complete chapter renders were requested. Full evidence and rationale are in the adjacent cast reports.",
            "Acoustic similarity is a review proxy, not a learned speaker embedding. Qwen Base does not support natural-language emotion instructions.",
            "Characters outside the audio budget retain dossiers/profiles and PENDING_VOICE status. Unknown facts remain unknown.",
            "Requested characters rejected by the audition/collision gate remain pending and are identified in acceptance-results.json.",
        ]
    )
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)
    if args.no_audio:
        return 0
    return (
        0
        if all(
            r["locked"]
            and r["render_checks"]
            and all(c["identity_matches"] for c in r["render_checks"])
            for r in summaries.values()
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
