"""make-audiobook: thin CLI over the AI Audio Studio audiobook authority.

The CLI is a boundary, not an authority: every operation calls the same
``dubbing_service.app`` code the HTTP API uses, so project state on disk stays
the single source of truth.

Examples
--------
::

    make-audiobook ingest --file book.txt --project-id my_book
    make-audiobook render --project-id my_book --renderer fixture
    make-audiobook status --project-id my_book
    make-audiobook export --project-id my_book --output report.json
    make-audiobook verify --project-id my_book --recover
    make-audiobook benchmark --project-id bench_run --render

Fixture rendering is the default renderer: it never claims to be engine output
(``renderer_id="fixture"``), so a fixture run can never be reported as RTX
hardware evidence.

The same ``main`` is reachable as the installed ``make-audiobook`` console
script and, from a checkout, as ``python scripts/make_audiobook.py``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _emit(payload: Dict[str, Any]) -> None:
    # The CLI is a machine-readable subprocess boundary.  ASCII-escaped JSON
    # survives the default Windows code page even when an error contains CJK,
    # while json.loads reconstructs the original Unicode for every caller.
    print(json.dumps(payload, ensure_ascii=True, indent=2))


def _load_app() -> None:
    try:
        import dubbing_service.app  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            f"无法导入 dubbing_service（缺少 {exc.name}）。请在项目虚拟环境中运行。"
        ) from exc


def cmd_ingest(args: argparse.Namespace) -> int:
    from dubbing_service.app.audiobook import adapter as adapter_module
    from dubbing_service.app.storage.projects import create_project, load_project

    source = Path(args.file)
    if not source.is_file():
        _emit({"ok": False, "operation": "ingest", "error": f"源文件不存在: {source}"})
        return 1

    try:
        analysis = adapter_module.analyze_source_file(
            source,
            title=args.title or source.stem,
            author=args.author,
            language=args.language,
            prefer_audiobooker=not args.offline,
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports the failure surface
        _emit({"ok": False, "operation": "ingest", "error": str(exc)})
        return 1

    project_id = args.project_id
    if project_id:
        try:
            project = load_project(project_id)
        except Exception:  # noqa: BLE001 - an unknown id must not silently become a new project
            _emit(
                {
                    "ok": False,
                    "operation": "ingest",
                    "project_id": project_id,
                    "error": (
                        f"未知项目: {project_id}; 省略 --project-id 可新建项目"
                    ),
                }
            )
            return 1
    else:
        project = create_project(args.title or source.stem)
        project_id = project.id

    from dubbing_service.app.storage.projects import save_project
    if getattr(args, "palette", None):
        project.character_cast.policy.inspiration_palette = args.palette
    if getattr(args, "max_chapter", None) is not None:
        project.character_cast.policy.max_chapter = args.max_chapter
    save_project(project)

    try:
        result = adapter_module.AudiobookerAdapter().reconcile_into_project(
            analysis, project_id=project_id, project_name=args.title or source.stem
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports the failure surface
        _emit(
            {
                "ok": False,
                "operation": "ingest",
                "project_id": project_id,
                "error": str(exc),
            }
        )
        return 1
    payload = result.as_dict()
    payload["operation"] = "ingest"
    payload["ok"] = True
    payload["analysis_source"] = adapter_module.analysis_source(analysis)
    payload["chapters_in_analysis"] = len(analysis.chapters)
    payload["utterances_in_analysis"] = analysis.utterance_count()
    payload["warnings"] = list(analysis.warnings)
    refreshed = load_project(project_id)
    payload["character_dossiers"] = len(refreshed.character_cast.dossiers)
    payload["casting_profiles"] = len(refreshed.character_cast.profiles)
    _emit(payload)
    return 0


def _renderer_for(args: argparse.Namespace, project_id: str) -> Any:
    from dubbing_service.app.audiobook import engines as engines_module
    from dubbing_service.app.audiobook import pipeline as pipeline_module
    from dubbing_service.app.audiobook import renderer as renderer_module

    root = pipeline_module.project_root(project_id)
    if args.renderer == "fixture":
        return renderer_module.FixtureRenderer(project_root=root)
    if not args.backend:
        # Never guess an engine: the caller names the backend explicitly.
        raise ValueError(
            "--renderer backend requires --backend "
            "(indextts / cosyvoice3 / qwen3tts / qwen3tts_cv)"
        )
    return renderer_module.BackendRenderer(
        backend=args.backend,
        engine_id=engines_module.engine_id_for_backend(args.backend),
        project_root=root,
    )


def _target_utterances(
    args: argparse.Namespace,
    project_id: str,
    *,
    default_all: bool,
) -> tuple[List[str], List[str]]:
    """Resolve explicit utterance/chapter selectors without guessing scope."""

    from dubbing_service.app.storage.projects import load_project
    from dubbing_service.app.storage.segments import load_segments
    from dubbing_service.app.audiobook import regeneration as regeneration_module

    segments = load_segments(project_id)
    utterance_ids = list(getattr(args, "utterance", []) or [])
    chapter_refs = list(getattr(args, "chapter_id", []) or [])
    resolved_chapters: List[str] = []
    if chapter_refs:
        project = load_project(project_id)
        chapters = list(project.chapters)
        for reference in chapter_refs:
            matches = [chapter for chapter in chapters if chapter.id == reference]
            if not matches:
                matches = [
                    chapter
                    for chapter in chapters
                    if chapter.title.strip().casefold() == reference.strip().casefold()
                ]
            if not matches and reference.isdigit():
                numeric = re.compile(rf"(?<!\d){re.escape(reference)}(?!\d)")
                matches = [
                    chapter
                    for chapter in chapters
                    if numeric.search(chapter.title)
                    or _chapter_title_has_number(chapter.title, int(reference))
                ]
            if len(matches) != 1:
                raise ValueError(
                    f"chapter selector {reference!r} resolved to {len(matches)} chapters; "
                    "use an exact chapter ID or title"
                )
            resolved_chapters.append(matches[0].id)
        chapter_targets = [
            segment.id
            for segment in segments
            if segment.chapter_id in set(resolved_chapters)
        ]
        utterance_ids.extend(chapter_targets)

    explicit = bool(getattr(args, "utterance", []) or chapter_refs)
    if not explicit and default_all:
        utterance_ids = regeneration_module.live_utterance_ids(segments)
    targets = list(dict.fromkeys(utterance_ids))
    return targets, resolved_chapters


def _english_number_words(value: int) -> List[str]:
    if value < 0 or value > 9999:
        return []
    ones = [
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    words: List[str] = []
    if value >= 1000:
        words.extend(_english_number_words(value // 1000))
        words.append("thousand")
        value %= 1000
    if value >= 100:
        words.append(ones[value // 100])
        words.append("hundred")
        value %= 100
    if value >= 20:
        words.append(tens[value // 10])
        value %= 10
    if value:
        words.append(ones[value])
    return words


def _chapter_title_has_number(title: str, value: int) -> bool:
    wanted = _english_number_words(value)
    title_words = re.findall(r"[a-z]+|\d+", title.casefold())
    # Book titles may use the optional British-style "and" after hundreds.
    title_words = [word for word in title_words if word != "and"]
    if not wanted:
        return False
    return any(
        title_words[index : index + len(wanted)] == wanted
        for index in range(len(title_words) - len(wanted) + 1)
    )


def cmd_render(args: argparse.Namespace) -> int:
    from dubbing_service.app.audiobook import pipeline as pipeline_module

    mode = getattr(args, "mode", "new-take")
    if mode == "resume" and not (args.utterance or args.chapter_id):
        _emit({
            "ok": False,
            "operation": "render",
            "project_id": args.project_id,
            "error": "resume requires explicit --utterance or --chapter-id selectors; whole-project resume is refused",
        })
        return 1
    if mode == "resume" and (args.no_promote or args.no_qa):
        _emit({
            "ok": False,
            "operation": "render",
            "project_id": args.project_id,
            "error": "resume requires QA and promotion enabled; use new-take for an explicit non-promoting render",
        })
        return 1
    try:
        targets, resolved_chapters = _target_utterances(
            args, args.project_id, default_all=(mode == "new-take")
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports selection errors as JSON
        _emit({
            "ok": False,
            "operation": "render",
            "project_id": args.project_id,
            "error": str(exc),
        })
        return 1
    if not targets:
        _emit(
            {
                "ok": False,
                "operation": "render",
                "error": "项目中没有可渲染的 utterance",
            }
        )
        return 1

    try:
        renderer = _renderer_for(args, args.project_id)
        records = pipeline_module.render_utterances(
            args.project_id,
            targets,
            renderer=renderer,
            slot=args.slot,
            promote=not args.no_promote,
            run_qa=not args.no_qa,
            resume=(mode == "resume"),
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports the failure surface
        _emit(
            {
                "ok": False,
                "operation": "render",
                "project_id": args.project_id,
                "requested_renderer": args.renderer,
                "renderer": args.renderer,
                "error": str(exc),
            }
        )
        return 1

    failures = [record for record in records if not record.success]
    # A Take that was written but refused by blocking QA is not a success: it
    # stays on disk as a candidate, and the aggregate says so.
    blocked = [
        record
        for record in records
        if record.success and not record.selected and record.blocking_qa
    ]
    payload = {
        "ok": not failures and not blocked,
        "operation": "render",
        "mode": mode,
        "project_id": args.project_id,
        "chapter_ids": resolved_chapters,
        "requested_renderer": args.renderer,
        "renderer": getattr(renderer, "renderer_id", args.renderer),
        "engine": getattr(renderer, "engine_id", None),
        "rendered": sum(record.dispatched for record in records),
        "reused": sum(record.reused for record in records),
        "reaudited": sum(record.reaudited for record in records),
        "succeeded": len(records) - len(failures),
        "qa_blocked": len(blocked),
        "takes": [
            {
                "segment_id": record.segment_id,
                "take_id": record.take_id,
                "rendered": record.dispatched,
                "reused": record.reused,
                "resume_reason": record.resume_reason,
                "candidate_created": record.candidate_created,
                "qa_passed": record.qa_passed,
                "selected": record.selected,
                "blocking_qa": list(record.blocking_qa),
                "duration_ms": record.duration_ms,
                "chunked": record.chunked,
                "stage_timings": record.stage_timings,
                "stale_components": list(record.stale_components),
            }
            for record in records
            if record.success
        ],
        "blocked": [
            {
                "segment_id": record.segment_id,
                "take_id": record.take_id,
                "selected": record.selected,
                "qa_passed": record.qa_passed,
                "blocking_qa": list(record.blocking_qa),
                "error": record.error,
                "stage_timings": record.stage_timings,
            }
            for record in blocked
        ],
        "failed": [
            {
                "segment_id": record.segment_id,
                "take_id": record.take_id,
                "error": record.error,
                "stage_timings": record.stage_timings,
                "resume_reason": record.resume_reason,
                "dispatched": record.dispatched,
            }
            for record in failures
        ],
        "status": pipeline_module.status_report(args.project_id),
    }
    if mode == "resume":
        payload["resume_summary"] = {
            "requested": len(records),
            "reused": sum(record.reused for record in records),
            "generated": sum(record.dispatched for record in records),
            "reaudited": sum(record.reaudited for record in records),
            "actionable_failures": len(failures),
        }
    if failures:
        payload["error"] = f"{len(failures)} of {len(records)} renders failed"
    elif blocked:
        payload["error"] = (
            f"{len(blocked)} of {len(records)} renders were refused by blocking QA"
        )
    _emit(payload)
    return 0 if not failures and not blocked else 1


def cmd_status(args: argparse.Namespace) -> int:
    from dubbing_service.app.audiobook import interface as interface_module

    if getattr(args, "resume", False):
        from dubbing_service.app.audiobook import pipeline as pipeline_module

        if not (args.utterance or args.chapter_id):
            _emit({
                "ok": False,
                "operation": "status",
                "error": "status --resume requires explicit --utterance or --chapter-id selectors",
            })
            return 1
        try:
            targets, chapters = _target_utterances(
                args, args.project_id, default_all=False
            )
            result = pipeline_module.resume_status(args.project_id, targets)
        except Exception as exc:  # noqa: BLE001 - CLI reports selection errors as JSON
            _emit({"ok": False, "operation": "status", "error": str(exc)})
            return 1
        result["chapter_ids"] = chapters
        _emit({"ok": True, "operation": "status", "resume": result})
        return 0

    try:
        result = interface_module.get_status(args.project_id, job_id=args.job_id)
    except interface_module.InterfaceError as exc:
        _emit({"ok": False, "operation": "status", "error": str(exc)})
        return 1
    _emit(result.as_dict())
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from dubbing_service.app.audiobook import interface as interface_module

    result = interface_module.export(args.project_id)
    payload = result.as_dict()
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        payload["written_to"] = str(target)
    _emit(payload)
    return 0 if payload.get("ready") else 1


def cmd_select(args: argparse.Namespace) -> int:
    from dubbing_service.app.audiobook import interface as interface_module

    try:
        if args.reject:
            result = interface_module.reject_take(
                args.project_id,
                segment_id=args.segment_id,
                take_id=args.take_id,
                reason=args.reason or "",
            )
        else:
            result = interface_module.select_take(
                args.project_id,
                segment_id=args.segment_id,
                take_id=args.take_id,
                slot=args.slot,
            )
    except interface_module.InterfaceError as exc:
        _emit({"ok": False, "operation": "select", "error": str(exc)})
        return 1
    _emit(result.as_dict())
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from dubbing_service.app.audiobook import persistence as persistence_module
    from dubbing_service.app.audiobook import pipeline as pipeline_module

    recovery: Optional[Dict[str, object]] = None
    if args.recover:
        recovery = persistence_module.recover_project_state(args.project_id).as_dict()
    ok, problems = persistence_module.verify_project_state(args.project_id)
    try:
        status = pipeline_module.status_report(args.project_id)
    except Exception as exc:  # noqa: BLE001 - verify reports, it does not crash
        status = {"coverage": None, "release_ready": False, "error": str(exc)}
    _emit(
        {
            "ok": ok,
            "operation": "verify",
            "project_id": args.project_id,
            "problems": problems,
            "pending_markers": persistence_module.pending_markers(args.project_id),
            "recovery": recovery,
            # State integrity alone must never read as "ready to release": a
            # live utterance whose Takes are all QA-blocked is reported here.
            "coverage": status.get("coverage"),
            "release_ready": status.get("release_ready", False),
            "qa_blocked": status.get("qa_blocked", 0),
            "blocked_utterances": status.get("blocked_utterances", []),
        }
    )
    return 0 if ok else 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    from dubbing_service.app.audiobook import benchmark as benchmark_module
    from dubbing_service.app.audiobook import pipeline as pipeline_module
    from dubbing_service.app.audiobook import persistence as persistence_module
    from dubbing_service.app.storage.projects import create_project, load_project

    project_id = args.project_id
    if project_id:
        try:
            project = load_project(project_id)
        except Exception:  # noqa: BLE001 - a missing project is created below
            project = create_project(args.title or "benchmark-corpus")
    else:
        project = create_project(args.title or "benchmark-corpus")

    materialized = benchmark_module.materialize_benchmark(project)
    segments = materialized["segments"]
    persistence_module.commit_project_state(
        project.id, project=project, segments=segments  # type: ignore[arg-type]
    )

    payload: Dict[str, Any] = {
        "ok": True,
        "operation": "benchmark",
        "project_id": project.id,
        "cases": benchmark_module.case_ids(),
        "roles": materialized["roles"],
        "utterances": benchmark_module.corpus_utterance_count(),
        "corpus_fingerprint": benchmark_module.corpus_fingerprint(),
    }
    if args.render:
        try:
            renderer = _renderer_for(args, project.id)
            records = pipeline_module.render_utterances(
                project.id,
                [segment.id for segment in segments],  # type: ignore[union-attr]
                renderer=renderer,
            )
        except Exception as exc:  # noqa: BLE001 - CLI reports the failure surface
            payload["ok"] = False
            payload["error"] = str(exc)
            payload["report"] = _jsonable(benchmark_module.benchmark_report(project.id))
            _emit(payload)
            return 1
        payload["rendered"] = len(records)
        payload["succeeded"] = sum(1 for record in records if record.success)
        payload["renderer"] = getattr(renderer, "renderer_id", args.renderer)
        payload["engine"] = getattr(renderer, "engine_id", None)
    payload["report"] = _jsonable(benchmark_module.benchmark_report(project.id))
    _emit(payload)
    return 0


def _jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def cmd_cast(args: argparse.Namespace) -> int:
    from dubbing_service.app.audiobook.character_casting import (
        cast_report, markdown_report, prepare_cast, refresh_cast,
    )
    from dubbing_service.app.storage.projects import load_project, project_meta_lock, save_project

    try:
        with project_meta_lock(args.project_id):
            project = load_project(args.project_id)
            policy = project.character_cast.policy
            if args.palette is not None:
                policy.inspiration_palette = None if args.palette == "default" else args.palette
            if args.max_chapter is not None:
                policy.max_chapter = args.max_chapter
            if args.adaptation:
                policy.adaptation_requested = True
            if args.allow_spoilers:
                policy.allow_spoilers = True
            if args.external_research:
                policy.external_research = True
            if args.max_candidates is not None:
                policy.max_candidates = args.max_candidates
            save_project(project)
        if args.no_audition:
            refresh_cast(args.project_id)
            report = cast_report(args.project_id)
        else:
            report = prepare_cast(args.project_id, role_ids=args.role or None, max_characters=args.max_characters)
        if args.report:
            target = Path(args.report)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            target.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8")
        _emit({"ok": True, "operation": "cast", **report})
        return 0
    except Exception as exc:
        _emit({"ok": False, "operation": "cast", "project_id": args.project_id, "error": str(exc)})
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make-audiobook",
        description="Compile a source document into an AI Audio Studio audiobook project",
    )
    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser("ingest", help="create/update a project from a file")
    ingest.add_argument("--file", required=True, help="epub/pdf/docx/txt source path")
    ingest.add_argument("--project-id", default=None, help="existing project to update")
    ingest.add_argument("--title", default=None)
    ingest.add_argument("--author", default=None)
    ingest.add_argument("--language", default=None)
    ingest.add_argument("--palette", default=None, help="optional performance inspiration palette")
    ingest.add_argument("--max-chapter", type=int, default=None, help="zero-based research spoiler boundary")
    ingest.add_argument(
        "--offline",
        action="store_true",
        help="force the deterministic offline analyzer even if audiobooker is present",
    )
    ingest.set_defaults(func=cmd_ingest)

    cast = subparsers.add_parser("cast", help="research, design, audition and persist original character voices")
    cast.add_argument("--project-id", required=True)
    cast.add_argument("--palette", default=None, help="default or harry-potter-film-cast")
    cast.add_argument("--adaptation", action="store_true", help="allow direct adaptation character mappings")
    cast.add_argument("--max-chapter", type=int, default=None, help="zero-based inclusive spoiler boundary")
    cast.add_argument("--allow-spoilers", action="store_true")
    cast.add_argument("--external-research", action="store_true", help="try configured provider; unavailable remains UNKNOWN")
    cast.add_argument("--no-audition", action="store_true", help="produce research/profiles without GPU work")
    cast.add_argument("--role", action="append", default=[], help="bound voice work to these role IDs")
    cast.add_argument("--max-characters", type=int, default=None)
    cast.add_argument("--max-candidates", type=int, choices=(1, 2, 3, 4), default=None)
    cast.add_argument("--report", default=None, help="save JSON plus human-readable Markdown")
    cast.set_defaults(func=cmd_cast)

    render = subparsers.add_parser("render", help="render utterances")
    render.add_argument("--project-id", required=True)
    render.add_argument("--utterance", action="append", default=[])
    render.add_argument(
        "--chapter-id",
        action="append",
        default=[],
        help="bounded chapter selector (exact project ID/title, or a unique numeric chapter number)",
    )
    render.add_argument(
        "--mode",
        choices=("new-take", "resume"),
        default="new-take",
        help="new-take always synthesizes; resume reuses only valid selected Takes",
    )
    render.add_argument("--renderer", choices=("fixture", "backend"), default="fixture")
    render.add_argument(
        "--backend",
        default=None,
        choices=("indextts", "cosyvoice3", "qwen3tts", "qwen3tts_cv"),
        help="engine backend; required with --renderer backend, since voices "
        "come from the project role configuration rather than the CLI",
    )
    render.add_argument("--slot", default="a")
    render.add_argument("--no-promote", action="store_true")
    render.add_argument("--no-qa", action="store_true")
    render.set_defaults(func=cmd_render)

    status = subparsers.add_parser("status", help="project and job status")
    status.add_argument("--project-id", required=True)
    status.add_argument("--job-id", default=None)
    status.add_argument("--resume", action="store_true", help="show read-only resume eligibility for an explicit scope")
    status.add_argument("--utterance", action="append", default=[])
    status.add_argument("--chapter-id", action="append", default=[])
    status.set_defaults(func=cmd_status)

    export = subparsers.add_parser("export", help="release coverage manifest")
    export.add_argument("--project-id", required=True)
    export.add_argument("--output", default=None, help="write the manifest as JSON")
    export.set_defaults(func=cmd_export)

    select = subparsers.add_parser("select", help="select or reject one Take")
    select.add_argument("--project-id", required=True)
    select.add_argument("--segment-id", required=True)
    select.add_argument("--take-id", required=True)
    select.add_argument("--slot", default="a")
    select.add_argument("--reject", action="store_true", help="reject instead of select")
    select.add_argument("--reason", default=None)
    select.set_defaults(func=cmd_select)

    verify = subparsers.add_parser("verify", help="project state integrity check")
    verify.add_argument("--project-id", required=True)
    verify.add_argument("--recover", action="store_true", help="repair an interrupted commit")
    verify.set_defaults(func=cmd_verify)

    benchmark = subparsers.add_parser("benchmark", help="materialise the 13-case corpus")
    benchmark.add_argument("--project-id", default=None)
    benchmark.add_argument("--title", default=None)
    benchmark.add_argument("--render", action="store_true")
    benchmark.add_argument("--renderer", choices=("fixture", "backend"), default="fixture")
    benchmark.add_argument(
        "--backend",
        default=None,
        choices=("indextts", "cosyvoice3", "qwen3tts", "qwen3tts_cv"),
        help="engine backend; required with --renderer backend",
    )
    benchmark.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    _load_app()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 - the CLI boundary never leaks a traceback
        _emit({"ok": False, "operation": args.command, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
