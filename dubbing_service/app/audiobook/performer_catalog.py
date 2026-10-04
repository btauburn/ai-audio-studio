"""The performer-reference pool: the shipped catalogue plus every sourced row.

The casting pipeline used to choose from a handful of hand-written rows, so the
same few performers won every character. The pool is now the shipped seeds plus a
much larger body of documented credits, and it can grow: a performer whose credit
evidence has been retrieved from a primary source can be adopted into the pool
without editing code.

Two rules keep a larger pool from becoming a looser one:

* every row is validated against the same shape the seeds use, so a malformed or
  half-populated candidate is refused rather than ranked into a cast;
* adoption requires a retrieved, cited credit. A name with no performance, no
  source URL or no stated fact is not a reference — it is the generic "AI voice"
  problem wearing a famous name, and it is rejected here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "performer-catalog-sources-v1"
ADOPTED_SCHEMA = "performer-catalog-adopted-v1"

#: Shipped, reviewable data: documented credits only, no acoustic measurements.
SOURCES_PATH = Path(__file__).with_name("performer_catalog_sources.json")
#: Performers adopted from retrieved evidence at runtime, kept apart from the
#: shipped file so an upgrade never overwrites discovered work.
ADOPTED_PATH = Path(__file__).with_name("performer_catalog_adopted.json")

#: The exact key set every catalogue row must carry. A row missing one of these is
#: refused: the ranker reads all of them, and a silent default would let an
#: unfounded candidate reach a cast.
ROW_KEYS = frozenset({
    "candidate_id", "performer", "performance", "tags", "creative_direction",
    "source_urls", "verified_facts", "facts_verified_at", "language",
    "evidence_level", "audio_observations", "creative_interpretation",
})

CREATIVE_INTERPRETATION = ("Delivery, vocal age and accent tags are creative casting "
                           "directions, not acoustic measurements or biographical facts.")

#: Tags a row may carry. An unknown tag is dropped rather than ranked, so a typo
#: cannot silently invent a casting dimension.
KNOWN_TAGS = frozenset({
    "male", "female", "young", "teen", "adult", "mature", "narrator", "warm",
    "measured", "confident", "dry", "resonant", "clear", "theatrical", "lively",
    "expressive", "energetic", "earnest", "vulnerable", "thoughtful", "soft",
    "playful", "curious", "reassuring", "grounded", "direct", "determined",
    "british", "american", "irish", "scottish", "australian", "canadian",
    "african", "indian", "documentary", "conversational", "versatile", "gravel",
    "airy", "precise", "brisk", "slow", "comic", "authoritative",
})

_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


class CatalogError(ValueError):
    """A catalogue row or file that must not enter the casting pool."""


def _text(value: Any, *, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit]


def _urls(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    found = []
    for item in value:
        text = _text(item, limit=500)
        if text.startswith("https://") and len(text) > len("https://"):
            found.append(text)
    return tuple(found)


def normalise(row: Mapping[str, Any], *, require_https: bool = True) -> dict:
    """One validated catalogue row, or :class:`CatalogError` naming the gap.

    ``require_https`` is on for shipped and adopted data alike: a reference whose
    credit cannot be re-checked at a URL is not evidence, and accepting it here
    would put an unverifiable performer into a cast.
    """
    if not isinstance(row, Mapping):
        raise CatalogError("a catalogue row must be an object")
    missing = sorted(ROW_KEYS - set(row))
    if missing:
        raise CatalogError("catalogue row is missing: " + ", ".join(missing))
    unknown = sorted(set(row) - ROW_KEYS)
    if unknown:
        raise CatalogError("catalogue row carries unsupported keys: " + ", ".join(unknown))
    candidate_id = _text(row["candidate_id"], limit=64)
    if not _ID.match(candidate_id):
        raise CatalogError(f"invalid candidate id: {candidate_id!r}")
    performer = _text(row["performer"], limit=120)
    performance = _text(row["performance"], limit=200)
    if not performer or not performance:
        raise CatalogError(f"{candidate_id}: a reference needs both a performer and a specific performance")
    tags = [tag for tag in (_text(item, limit=40).casefold() for item in row["tags"] or [])
            if tag in KNOWN_TAGS]
    if not tags:
        raise CatalogError(f"{candidate_id}: a reference needs at least one recognised casting tag")
    if "male" in tags and "female" in tags:
        raise CatalogError(f"{candidate_id}: contradictory presentation tags")
    direction = _text(row["creative_direction"], limit=400)
    if not direction:
        raise CatalogError(f"{candidate_id}: a reference needs a delivery direction")
    urls = _urls(row["source_urls"])
    if not urls:
        if require_https:
            raise CatalogError(f"{candidate_id}: a reference needs a retrievable credit URL")
        raise CatalogError(f"{candidate_id}: a reference needs a source")
    facts = [_text(item, limit=400) for item in row["verified_facts"] or []]
    facts = [fact for fact in facts if fact]
    if not facts:
        raise CatalogError(f"{candidate_id}: a reference needs a stated credit fact")
    if row["audio_observations"]:
        raise CatalogError(f"{candidate_id}: shipped rows carry no audio observations")
    return {
        "candidate_id": candidate_id,
        "performer": performer,
        "performance": performance,
        "tags": tags,
        "creative_direction": direction,
        "source_urls": list(urls),
        "verified_facts": facts,
        "facts_verified_at": _text(row["facts_verified_at"], limit=32),
        "language": _text(row["language"], limit=16) or "en",
        "evidence_level": _text(row["evidence_level"], limit=64) or "metadata_only_exploratory",
        "audio_observations": [],
        "creative_interpretation": _text(row["creative_interpretation"], limit=400)
                                   or CREATIVE_INTERPRETATION,
    }


def _load_rows(path: Path, *, schema: str) -> tuple[list[dict], list[str]]:
    """Validated rows from one catalogue file, plus the reasons rows were refused.

    A missing file is an empty catalogue, not an error: the seeds still work. A
    file that is present but unreadable is reported, because silently ranking a
    smaller pool is the failure this module exists to prevent.
    """
    if not path.is_file():
        return [], []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [], [f"{path.name} could not be read: {type(exc).__name__}"]
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        return [], [f"{path.name} is not a {schema} document"]
    rows = payload.get("performers")
    if not isinstance(rows, list):
        return [], [f"{path.name} has no performers list"]
    accepted, refused = [], []
    for index, row in enumerate(rows):
        try:
            accepted.append(normalise(row))
        except CatalogError as exc:
            refused.append(str(exc) or f"{path.name} row {index} was refused")
    return accepted, refused


def load_pool(seeds: tuple[Mapping[str, Any], ...]) -> dict:
    """The whole casting pool: shipped seeds plus sourced and adopted rows.

    Returns the rows in a stable order with the reasons any row was refused, so a
    caller can report a shrinking pool instead of quietly casting from less.
    """
    shipped, refused = _load_rows(SOURCES_PATH, schema=SCHEMA)
    adopted, adopted_refused = _load_rows(ADOPTED_PATH, schema=ADOPTED_SCHEMA)
    seen: dict[str, dict] = {}
    for row in seeds:
        normalised = normalise(dict(row), require_https=True)
        seen.setdefault(normalised["candidate_id"], normalised)
    for row in [*shipped, *adopted]:
        seen.setdefault(row["candidate_id"], row)
    return {"performers": tuple(seen.values()), "shipped": len(shipped),
            "adopted": len(adopted), "seeds": len(seeds),
            "refused": refused + adopted_refused}


def pool_version(pool: Mapping[str, Any]) -> str:
    """A version string that changes whenever the pool's contents do."""
    import hashlib

    digest = hashlib.sha256()
    for row in pool["performers"]:
        digest.update(row["candidate_id"].encode())
        digest.update(row["performer"].encode())
        digest.update(row["performance"].encode())
    return f"{pool['seeds']}+{pool['shipped']}+{pool['adopted']}:{digest.hexdigest()[:12]}"


def adopt(row: Mapping[str, Any], *, path: Path | None = None) -> dict:
    """Add one performer to the adopted pool, or refuse it with the reason.

    Idempotent by candidate id, and never rewrites an existing row: a second
    adoption of the same identity is a no-op rather than a silent edit of the
    evidence a cast was built on.
    """
    path = Path(path or ADOPTED_PATH)
    normalised = normalise(row)
    document = {"schema": ADOPTED_SCHEMA, "version": "1", "performers": []}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CatalogError(f"the adopted catalogue is unreadable: {type(exc).__name__}") from exc
        if not isinstance(existing, dict) or existing.get("schema") != ADOPTED_SCHEMA:
            raise CatalogError("the adopted catalogue is not a performer-catalog-adopted-v1 document")
        document = existing
        document.setdefault("performers", [])
        for current in document["performers"]:
            if current.get("candidate_id") == normalised["candidate_id"]:
                return {"adopted": False, "reason": "already adopted", "performer": normalised["performer"]}
    document["performers"].append(normalised)
    staged = path.with_name(path.name + ".stage")
    staged.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8", newline="\n")
    staged.replace(path)
    return {"adopted": True, "performer": normalised["performer"],
            "candidate_id": normalised["candidate_id"],
            "pool_size": len(document["performers"])}
