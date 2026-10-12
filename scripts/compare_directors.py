#!/usr/bin/env python3
"""Run the four-arm director audio QA comparison from a manifest.

Reads audio and receipts that already exist; never synthesizes, never uses the
GPU or a model, never touches live stores.  Results are published exclusively:
re-running with identical inputs reuses the files, differing inputs against an
existing output directory are refused.

    compare_directors.py MANIFEST --output-dir OUT [--human-review REVIEW.json]
                         [--stage-blind-audio DIR]
    compare_directors.py --write-fixture DIR     # labeled artificial tones

Exit codes: 0 evaluated, 2 invalid/untrusted input or output conflict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dubbing_service.app.audiobook import director_evaluation as de  # noqa: E402


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise de.EvaluationError("manifest_invalid", f"cannot read {path}: {exc}") from exc


def _load_json(path: Path, what: str) -> dict:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise de.EvaluationError("manifest_invalid", f"cannot read {what} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise de.EvaluationError("manifest_invalid", f"{what} must be a JSON object")
    return value


def _summary(result: dict) -> dict:
    """Terminal output: never includes the sealed ballot key."""
    return {
        "comparison_id": result["comparison_id"],
        "evaluation_id": result["evaluation_id"],
        "status": result["status"],
        "fixture": result["fixture"],
        "production_eligible": result["production_eligible"],
        "arm_status": result["arm_status"],
        "ballot": result["ballot_ref"],
        "human_gate": {k: result["human_gate"][k] for k in ("status", "reasons", "approval")},
        "warnings": result["warnings"],
        "outputs": result.get("outputs"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", nargs="?", type=Path, help="comparison manifest JSON")
    parser.add_argument("--output-dir", type=Path, help="publish evaluation/ballot files here (exclusive)")
    parser.add_argument("--human-review", type=Path, help="completed ballot review JSON to score")
    parser.add_argument("--stage-blind-audio", type=Path, help="copy ballot audio to arm-blind file names here")
    parser.add_argument("--write-fixture", type=Path, help="write a labeled artificial-tone fixture and manifest, then exit")
    args = parser.parse_args(argv)

    try:
        if args.write_fixture:
            manifest = de.build_synthetic_fixture(args.write_fixture)
            target = args.write_fixture / "manifest.json"
            de._publish_file(target, de._pretty(manifest))
            print(json.dumps({"fixture": True, "label": de.FIXTURE_LABEL, "manifest": str(target)}, indent=2))
            return 0
        if not args.manifest:
            parser.error("a manifest is required")
        manifest_path = args.manifest.resolve()
        before = _sha256(manifest_path)
        manifest = _load_json(manifest_path, "manifest")
        manifest.setdefault("base_dir", str(manifest_path.parent))
        review_path = args.human_review.resolve() if args.human_review else None
        review_before = _sha256(review_path) if review_path else None
        if review_path:
            manifest["human_review"] = _load_json(review_path, "human review")
        result = de.evaluate_comparison(manifest, output_dir=args.output_dir)
        if args.stage_blind_audio:
            result["staged_blind_audio"] = de.stage_blind_audio(result, args.stage_blind_audio)
        if _sha256(manifest_path) != before or (review_path and _sha256(review_path) != review_before):
            raise de.EvaluationError("source_changed", "the manifest or review changed while evaluating")
    except de.EvaluationError as exc:
        print(json.dumps({"error": exc.code, "message": exc.message, "details": exc.details}, indent=2, default=str),
              file=sys.stderr)
        return 2
    summary = _summary(result)
    if "staged_blind_audio" in result:
        summary["staged_blind_audio"] = result["staged_blind_audio"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
