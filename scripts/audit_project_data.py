"""Read-only integrity audit for project metadata and referenced files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "dubbing_projects"


def audit_projects(projects_root: Path) -> dict[str, object]:
    errors: list[str] = []
    counts = {"projects": 0, "roles": 0, "segments": 0, "done_segments": 0}
    referenced_audio: set[Path] = set()

    projects_root = projects_root.resolve()
    if not projects_root.is_dir():
        return {
            "projects_root": str(projects_root),
            "counts": counts,
            "referenced_audio": 0,
            "errors": ["projects root does not exist or is not a directory"],
        }

    for project_dir in sorted(path for path in projects_root.iterdir() if path.is_dir()):
        project_path = project_dir / "project.json"
        if not project_path.is_file():
            continue
        counts["projects"] += 1
        try:
            project = json.loads(project_path.read_text(encoding="utf-8"))
            segments_path = project_dir / "segments.json"
            segments = json.loads(segments_path.read_text(encoding="utf-8")) if segments_path.is_file() else []
        except Exception as exc:
            errors.append(f"{project_dir.name}: invalid JSON: {exc}")
            continue

        roles = project.get("roles", []) if isinstance(project, dict) else []
        counts["roles"] += len(roles) if isinstance(roles, list) else 0
        role_ids = {str(role.get("id")) for role in roles if isinstance(role, dict) and role.get("id")}

        if not isinstance(segments, list):
            errors.append(f"{project_dir.name}: segments.json is not a list")
            continue
        counts["segments"] += len(segments)
        for segment in segments:
            if not isinstance(segment, dict):
                errors.append(f"{project_dir.name}: non-object segment")
                continue
            if segment.get("status") == "done":
                counts["done_segments"] += 1
            role_id = segment.get("role_id")
            if role_id and str(role_id) not in role_ids:
                errors.append(f"{project_dir.name}/{segment.get('id')}: missing role {role_id}")
            for field in ("audio_file", "alt_audio_file", "emotion_audio"):
                value = segment.get(field)
                if not value:
                    continue
                candidate = Path(str(value))
                if not candidate.is_absolute():
                    candidate = project_dir / candidate
                candidate = candidate.resolve()
                try:
                    candidate.relative_to(project_dir.resolve())
                except ValueError:
                    errors.append(f"{project_dir.name}/{segment.get('id')}: {field} escapes project")
                    continue
                referenced_audio.add(candidate)
                if not candidate.is_file():
                    errors.append(f"{project_dir.name}/{segment.get('id')}: missing {field} {value}")

    return {
        "projects_root": str(projects_root),
        "counts": counts,
        "referenced_audio": len(referenced_audio),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only project data integrity audit")
    parser.add_argument(
        "--projects-root",
        type=Path,
        default=PROJECTS,
        help="project data root to audit (defaults to repository dubbing_projects)",
    )
    args = parser.parse_args()
    result = audit_projects(args.projects_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
