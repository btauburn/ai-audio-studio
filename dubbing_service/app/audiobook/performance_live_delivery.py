"""Private, verified delivery of one pilot's review audio to Brian only.

The M3 assembler publishes a review-only edition: a `review-master.wav`, a
`listening.mp3`, a `source-to-audio-map.json` and an immutable `receipt.json`
that binds them to the exact Takes, QA reports, cast and runtime the assembly
actually used.  This module turns such an edition into something Brian can
play, and it never weakens what M3 established.

Three properties matter here.

* **Verify, then publish.**  Every byte this module offers is re-hashed from the
  edition itself and re-checked against the receipt before the batch is written,
  and the published batch is re-read and re-verified before anyone is told it is
  ready.  An edition whose receipt does not verify is refused, never repaired.
* **Do not inflate the claim.**  An M3 receipt is an *assembly* receipt.  It is
  not the delivery server's unrelated `verified_listening_export` schema, and
  this module never relabels it as one, never sets `verified_loudness`, and never
  asserts that audio was heard, matched or preferred.  Human review stays
  `pending` until a real authenticated human says otherwise.
* **Keep the key private.**  The batch carries opaque X/Y labels only.  Which
  label is the baseline and which is the variant lives in the pilot's own
  private key store and is never written into, or served from, the batch.

Notification reuses the existing durable intent/outcome owner unchanged, so a
duplicate, a pending attempt or an unknown outcome suppresses automatic resend
instead of producing a second push.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dubbing_service.app.audiobook.delivery import store as delivery_store
from dubbing_service.app.audiobook.performance_director import (
    PerformanceDirectorError,
    content_sha256,
)

BATCH_SCHEMA = "performance-live-delivery-v1"
PILOT_PROJECTS_DIR = "performance/pilots"
RECIPIENT_ID = "brian"
MAX_CHAPTERS = 8

#: The delivery server's own, unrelated export schema.  Named here so this
#: module can prove it never claims it.
FOREIGN_EXPORT_SCHEMA = "verified_listening_export"

DELIVERY_LIMITATION = (
    "this batch is a private review projection of an assembly-edition receipt.  It is not a "
    "verified listening export, it carries no loudness verification, and it states nothing about "
    "whether the audio is good, preferred or heard.  Human review remains pending."
)

ArmNames = ("baseline", "variant")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


class LiveDeliveryError(PerformanceDirectorError):
    """A private-delivery refusal.  Same hold codes as M1-M3."""


def _refuse(message: str, code: str = "input_ineligible") -> LiveDeliveryError:
    return LiveDeliveryError(code, message)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _resolve_within(root: Path, candidate: Path, what: str) -> Path:
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise _refuse(f"{what} is missing or unreadable: {exc}") from exc
    if resolved != root and not resolved.is_relative_to(root):
        raise _refuse(f"{what} escapes its pilot root")
    return resolved


# --------------------------------------------------------------------------- the edition


def edition_directory(pilot_root: Path, pilot_project_id: str, arm: str,
                      timeline_sha256: str) -> Path:
    """The one directory M3 publishes an edition into, by its own layout."""

    if arm not in ArmNames:
        raise _refuse(f"arm must be one of {list(ArmNames)}")
    if not isinstance(timeline_sha256, str) or not _SHA256.fullmatch(timeline_sha256):
        raise _refuse("timeline_sha256 must be a lowercase SHA-256 digest")
    if not _IDENTIFIER.fullmatch(pilot_project_id or ""):
        raise _refuse("pilot_project_id is not a valid project identifier")
    root = pilot_root.resolve()
    return root / PILOT_PROJECTS_DIR / pilot_project_id / arm / timeline_sha256


def verify_assembly_receipt(directory: Path, *, arm: str, timeline_sha256: str) -> dict:
    """Re-derive every fact an M3 assembly receipt claims, from the bytes on disk."""

    from dubbing_service.app.audiobook.performance_pilot import (
        ASSEMBLY_SCHEMA,
        MAP_SCHEMA,
        REQUIRED_ARTIFACTS,
    )

    directory = Path(directory)
    receipt_path = directory / "receipt.json"
    if not receipt_path.is_file():
        raise _refuse("no assembly receipt exists for this edition", "measurement_uncertain")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _refuse(f"the assembly receipt is unreadable: {exc}") from exc
    if not isinstance(receipt, Mapping):
        raise _refuse("the assembly receipt is not an object")
    if receipt.get("schema_version") == FOREIGN_EXPORT_SCHEMA or receipt.get("status") == FOREIGN_EXPORT_SCHEMA:
        raise _refuse(
            "an assembly receipt must never be presented as the delivery server's "
            "verified listening export schema", "direction_conflict")
    if receipt.get("schema_version") != ASSEMBLY_SCHEMA:
        raise _refuse("the edition does not carry an M3 assembly receipt", "schema_unknown")
    if receipt.get("kind") != "review_only":
        raise _refuse("only a review-only assembly may be delivered")
    if receipt.get("arm") != arm:
        raise _refuse("the assembly receipt belongs to another arm", "plan_stale")
    if receipt.get("timeline_sha256") != timeline_sha256:
        raise _refuse("the assembly receipt belongs to another timeline", "plan_stale")
    if receipt.get("production_enablement") is not False:
        raise _refuse("an assembly receipt with production enablement is refused",
                      "direction_conflict")
    if receipt.get("human_review") != "pending":
        raise _refuse("this module never presents an assembly as human-reviewed",
                      "direction_conflict")
    recorded = receipt.get("receipt_sha256")
    if not isinstance(recorded, str) or recorded != content_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}):
        raise _refuse("the assembly receipt's own digest does not verify", "plan_stale")
    declared = receipt.get("artifacts")
    if not isinstance(declared, Mapping) or set(declared) != set(REQUIRED_ARTIFACTS):
        raise _refuse("the assembly receipt does not declare the complete artifact set")
    for name in REQUIRED_ARTIFACTS:
        digest = declared.get(name)
        path = directory / name
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise _refuse(f"the receipt's {name} digest is malformed")
        if not path.is_file() or _sha256_file(path) != digest:
            raise _refuse(f"declared artifact {name} does not match its receipt", "plan_stale")

    total_frames = receipt.get("total_frames")
    sample_rate = receipt.get("sample_rate")
    if (isinstance(total_frames, bool) or not isinstance(total_frames, int) or total_frames <= 0
            or isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0):
        raise _refuse("the assembly receipt's frame accounting is malformed")
    master = directory / "review-master.wav"
    try:
        with wave.open(str(master), "rb") as handle:
            frames, rate = handle.getnframes(), handle.getframerate()
    except (wave.Error, OSError) as exc:
        raise _refuse(f"the review master cannot be read as PCM: {exc}") from exc
    if frames != total_frames or rate != sample_rate:
        raise _refuse("the review master does not match the receipt's frame accounting",
                      "plan_stale")

    try:
        mapping = json.loads((directory / "source-to-audio-map.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _refuse(f"the source-to-audio map is unreadable: {exc}") from exc
    if not isinstance(mapping, Mapping) or mapping.get("schema_version") != MAP_SCHEMA:
        raise _refuse("the source-to-audio map has an unknown schema", "schema_unknown")
    if content_sha256(mapping) != receipt.get("map_sha256"):
        raise _refuse("the source-to-audio map does not match its receipt", "plan_stale")
    if mapping.get("timeline_sha256") != timeline_sha256:
        raise _refuse("the source-to-audio map belongs to another timeline", "plan_stale")
    rows = mapping.get("rows")
    if not isinstance(rows, list) or not rows:
        raise _refuse("the source-to-audio map has no rows")
    if sum(int(row.get("frames", 0)) for row in rows) != total_frames:
        raise _refuse("the source-to-audio map does not cover the master's frames",
                      "plan_stale")

    proofs = receipt.get("inputs")
    if not isinstance(proofs, list) or not proofs:
        raise _refuse("the assembly receipt names no input proofs")
    for proof in proofs:
        if not isinstance(proof, Mapping):
            raise _refuse("an assembly input proof is malformed")
        for field in ("utterance_id", "take_id", "audio_sha256", "requirement_fingerprint",
                      "source_state_sha256"):
            value = proof.get(field)
            if field in ("audio_sha256", "requirement_fingerprint", "source_state_sha256"):
                if not isinstance(value, str) or not _SHA256.fullmatch(value):
                    raise _refuse(f"assembly proof field {field} is malformed")
            elif not isinstance(value, str) or not value:
                raise _refuse(f"assembly proof field {field} is missing")
    originals = receipt.get("originals")
    speech_ids = [row.get("utterance_id") for row in rows if row.get("kind") == "speech"]
    if not isinstance(originals, Mapping) or not originals:
        raise _refuse("the assembly receipt declares no original copies")
    if sorted(originals) != sorted(speech_ids):
        raise _refuse("the original copies do not cover exactly the mapped speech")
    if sorted(originals) != sorted(proof["utterance_id"] for proof in proofs):
        raise _refuse("the original copies do not cover exactly the assembly proofs")
    for utterance_id, digest in originals.items():
        if not isinstance(utterance_id, str) or not _IDENTIFIER.fullmatch(utterance_id):
            raise _refuse("an original copy has an unsafe utterance identity")
        copy = _resolve_within(directory, directory / "originals" / f"{utterance_id}.wav",
                               "original Take copy")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise _refuse(f"the original copy digest for {utterance_id} is malformed")
        if not copy.is_file() or _sha256_file(copy) != digest:
            raise _refuse(f"the original copy for {utterance_id} does not match its Take",
                          "plan_stale")

    listening = directory / "listening.mp3"
    from dubbing_service.app.audiobook.voice_lab_render import _decoded_duration
    try:
        duration = _decoded_duration(listening)
    except Exception as exc:
        raise _refuse(f"the listening derivative cannot be decoded: {exc}",
                      "capability_unsupported") from exc
    expected = total_frames / sample_rate
    if not math.isfinite(duration) or abs(duration - expected) > 0.1:
        raise _refuse("the listening derivative does not decode to the master duration",
                      "plan_stale")
    return {
        "directory": str(directory),
        "receipt": dict(receipt),
        "receipt_sha256": recorded,
        "arm": arm,
        "timeline_sha256": timeline_sha256,
        "listening_mp3": listening,
        "listening_mp3_sha256": declared["listening.mp3"],
        "review_master_sha256": declared["review-master.wav"],
        "map_sha256": declared["source-to-audio-map.json"],
        "total_frames": total_frames,
        "sample_rate": sample_rate,
        "duration_seconds": expected,
        "human_review": "pending",
        "production_enablement": False,
        "limitation": DELIVERY_LIMITATION,
    }


def locate_edition(pilot_root: Path, pilot_project_id: str, audio_sha256: str) -> dict:
    """Find the one verified edition whose published audio has this digest."""

    if not isinstance(audio_sha256, str) or not _SHA256.fullmatch(audio_sha256):
        raise _refuse("the audio digest to resolve is malformed")
    found: list[dict] = []
    for arm in ArmNames:
        base = Path(pilot_root).resolve() / PILOT_PROJECTS_DIR / pilot_project_id / arm
        if not base.is_dir():
            continue
        for timeline in sorted(path.name for path in base.iterdir() if path.is_dir()):
            try:
                edition = verify_assembly_receipt(base / timeline, arm=arm,
                                                  timeline_sha256=timeline)
            except PerformanceDirectorError:
                # An edition that does not verify is never a candidate; a
                # different edition may still be the one under this digest.
                continue
            if edition["listening_mp3_sha256"] == audio_sha256:
                found.append(edition)
    if not found:
        raise _refuse("no verified edition publishes this audio digest", "measurement_uncertain")
    if len(found) > 1:
        raise _refuse("more than one verified edition publishes this audio digest",
                      "direction_conflict")
    return found[0]


# --------------------------------------------------------------------------- the batch


def _opaque_label(label: Any) -> str:
    text = str(label or "")
    lowered = text.lower()
    if not text or any(arm in lowered for arm in ArmNames) or "pair" not in lowered:
        raise _refuse("a pilot label must be opaque and must not name an arm")
    return text


def build_pilot_projection(*, pilot_root: Path, pilot_project_id: str, book_title: str,
                           index: Mapping[str, Any]) -> dict:
    """Build the Brian-only delivery projection from a verified pilot pair index.

    The index is the pilot's *public* half: opaque labels and audio digests.  It
    carries no arm, and this function refuses an index that has been contaminated
    with the private key, because the batch it produces is served over HTTP.
    """

    if not isinstance(index, Mapping):
        raise _refuse("the pilot pair index must be an object")
    from dubbing_service.app.audiobook.performance_pilot import PAIR_INDEX_SCHEMA
    if index.get("schema_version") != PAIR_INDEX_SCHEMA:
        raise _refuse("the pilot pair index has an unknown schema", "schema_unknown")
    if index.get("pilot_id") != pilot_project_id:
        raise _refuse("the pilot pair index belongs to another pilot")
    for forbidden in ("key", "pairs_key"):
        if forbidden in index:
            raise _refuse("the private arm key must never enter a served batch",
                          "direction_conflict")
    pairs = index.get("pairs")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)) or not pairs:
        raise _refuse("the pilot pair index has no pairs")
    if len(pairs) * 2 > MAX_CHAPTERS:
        raise _refuse(f"a delivery batch carries at most {MAX_CHAPTERS} chapters")
    if not isinstance(book_title, str) or not book_title.strip():
        raise _refuse("the delivery batch needs a book title")

    chapters, labels = [], set()
    for number, row in enumerate(pairs, start=1):
        if not isinstance(row, Mapping):
            raise _refuse("a pilot pair row is malformed")
        for forbidden in ("x_arm", "y_arm", "edition_sha256"):
            if forbidden in row:
                raise _refuse("the private arm key must never enter a served batch",
                              "direction_conflict")
        label = _opaque_label(row.get("label"))
        if label in labels:
            raise _refuse("pilot labels must be unique within a batch")
        labels.add(label)
        for suffix, key in (("x", "x_audio_sha256"), ("y", "y_audio_sha256")):
            digest = row.get(key)
            edition = locate_edition(pilot_root, pilot_project_id, digest)
            chapter_id = f"{label}-{suffix}"
            chapters.append({
                "chapter_id": chapter_id,
                "chapter_number": len(chapters) + 1,
                "chapter_label": f"{label} {suffix.upper()}",
                "order": len(chapters) + 1,
                "title": f"{label} {suffix.upper()}",
                "duration_seconds": edition["duration_seconds"],
                "audio_path": str(edition["listening_mp3"]),
                "audio_sha256": edition["listening_mp3_sha256"],
                "revision": edition["receipt_sha256"],
                "export_manifest_sha256": edition["receipt_sha256"],
                "selected_takes": [],
                "assembly_receipt_sha256": edition["receipt_sha256"],
                "assembly_timeline_sha256": edition["timeline_sha256"],
                "human_review": "pending",
            })
    return {
        "schema_version": BATCH_SCHEMA,
        "project_id": pilot_project_id,
        "book_title": f"{book_title.strip()} — private timing review",
        "chapters": chapters,
        "recipient_id": RECIPIENT_ID,
        "production_enablement": False,
        "human_review": "pending",
        "limitation": DELIVERY_LIMITATION,
    }


def publish_pilot_batch(*, state_dir: Path, projection: Mapping[str, Any]) -> dict:
    """Publish the verified projection through the existing delivery store."""

    if not isinstance(projection, Mapping) or not projection.get("chapters"):
        raise _refuse("there is nothing to publish")
    if projection.get("recipient_id") != RECIPIENT_ID:
        raise _refuse("a pilot batch is published to Brian only")
    if projection.get("production_enablement") is not False:
        raise _refuse("a pilot batch is never a production enablement",
                      "direction_conflict")
    for chapter in projection["chapters"]:
        for field in ("title", "chapter_label"):
            text = str(chapter.get(field) or "").lower()
            if any(arm in text for arm in ArmNames):
                raise _refuse("a served label must never name an arm", "direction_conflict")
    return delivery_store.publish_batch(Path(state_dir), dict(projection), RECIPIENT_ID)


def verify_published_batch(*, state_dir: Path, batch: Mapping[str, Any],
                           audio_roots: Sequence[Any]) -> dict:
    """Re-read the published batch and prove the exact bytes it will serve."""

    if not isinstance(batch, Mapping) or not isinstance(batch.get("batch_id"), str):
        raise _refuse("a published batch is required")
    stored = delivery_store.load_batch(Path(state_dir), batch["batch_id"])
    if dict(stored) != dict(batch):
        raise _refuse("the stored batch differs from the batch under review", "plan_stale")
    roots = [Path(root).resolve() for root in audio_roots]
    if not roots:
        raise _refuse("explicit audio roots are required")
    verified = []
    for chapter in stored["chapters"]:
        path = Path(chapter["audio_path"])
        if not path.is_absolute():
            raise _refuse("a published chapter must name an absolute path")
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise _refuse(f"published chapter audio is missing: {exc}") from exc
        if not any(resolved.is_relative_to(root) for root in roots):
            raise _refuse("published chapter audio is outside the configured roots",
                          "direction_conflict")
        if resolved.suffix.lower() not in {".mp3", ".wav", ".m4a"}:
            raise _refuse("published chapter audio is not a playable format")
        digest = _sha256_file(resolved)
        if digest != chapter.get("audio_sha256"):
            raise _refuse("published chapter audio does not match the batch digest",
                          "plan_stale")
        size = resolved.stat().st_size
        if size <= 0:
            raise _refuse("published chapter audio is empty", "plan_stale")
        with resolved.open("rb") as handle:
            head = handle.read(1024)
        if len(head) != min(1024, size):
            raise _refuse("published chapter audio cannot be read fully", "plan_stale")
        verified.append({
            "chapter_id": chapter["chapter_id"],
            "audio_sha256": digest,
            "size_bytes": size,
            "revision": chapter.get("revision"),
            "export_manifest_sha256": chapter.get("export_manifest_sha256"),
            "first_1024_sha256": hashlib.sha256(head).hexdigest(),
            "range_contract": f"bytes 0-{min(1023, size - 1)}/{size}",
        })
    return {"batch_id": stored["batch_id"], "project_id": stored["project_id"],
            "recipient_id": stored["recipient_id"], "chapters": verified,
            "verified": True, "human_review": "pending",
            "production_enablement": False, "limitation": DELIVERY_LIMITATION}


def notify_pilot_ready(*, state_dir: Path, batch: Mapping[str, Any],
                       verification: Mapping[str, Any], recipient: Mapping[str, Any],
                       transport_root: Path, listening_url: str) -> dict:
    """Send at most one completion notification, through the existing owner.

    Readiness is a precondition, not a hint: the caller must present the
    verification produced for *this* batch.  Automatic resend is never requested
    here, so a duplicate, a pending attempt or an unknown outcome suppresses a
    second push rather than producing one.
    """

    if not isinstance(recipient, Mapping) or recipient.get("id") != RECIPIENT_ID:
        raise _refuse("the completion notification goes to Brian only")
    if recipient.get("enabled") is not True:
        raise _refuse("Brian's private recipient is not enabled")
    if not isinstance(verification, Mapping) or verification.get("verified") is not True:
        raise _refuse("the published bytes must be verified before notifying",
                      "review_pending")
    if verification.get("batch_id") != batch.get("batch_id"):
        raise _refuse("the verification belongs to another batch", "plan_stale")
    digests = [chapter["audio_sha256"] for chapter in verification.get("chapters") or []]
    if not digests or digests != [chapter["audio_sha256"] for chapter in batch.get("chapters") or []]:
        raise _refuse("the verification does not cover exactly this batch", "plan_stale")
    from dubbing_service.app.audiobook.delivery.notify import send_notification
    result = send_notification(dict(batch), dict(recipient), state_dir=Path(state_dir),
                               transport_root=Path(transport_root),
                               listening_url=str(listening_url), force=False)
    return {"attempted": True, "device_delivery_confirmed": False,
            "human_listening": "pending", "result": result}


def build_current_pilot_projection(*, pilot_root: Path, pilot_project_id: str,
                                   book_title: str, index: Mapping[str, Any]) -> dict:
    """Verify canonical owner state and eligibility before live publication.

    The lower receipt checker measures saved bytes. It cannot establish current
    source, cast or QA eligibility by itself. Live publishing uses this boundary
    to bind the public pair index to the owner's private edition keys and to
    re-read every selected Take through M3's canonical eligibility owner.
    """
    from . import performance_pilot as pilot
    from dubbing_service.app.storage.projects import load_project
    from dubbing_service.app.storage.segments import load_segments
    from dubbing_service.app.core.config import safe_project_root
    expected_root = safe_project_root(pilot_project_id).resolve()
    if Path(pilot_root).resolve() != expected_root:
        raise _refuse("the live projection root differs from its canonical pilot", "plan_stale")
    pilot.read_pilot(pilot_project_id)
    if not isinstance(index, Mapping) or not isinstance(index.get("pairs"), list):
        raise _refuse("the live projection needs the canonical pair index")
    for row in index["pairs"]:
        key = pilot._pair_key(pilot_project_id, row.get("pair_id"))
        if not key.get("resolved"):
            raise _refuse("a live pair has no resolved owner editions", "review_pending")
        for field in ("label", "x_audio_sha256", "y_audio_sha256"):
            if key.get(field) != row.get(field):
                raise _refuse("the public pair differs from its canonical owner key", "plan_stale")
        arm_proofs = []
        for suffix in ("x", "y"):
            digest, arm = key[f"{suffix}_audio_sha256"], key[f"{suffix}_arm"]
            if pilot._verified_audio_edition(pilot_project_id, digest, arm) is None:
                raise _refuse("a live pair edition is no longer published", "plan_stale")
            edition = locate_edition(expected_root, pilot_project_id, digest)
            if edition["arm"] != arm:
                raise _refuse("a live edition has another arm identity", "plan_stale")
            proofs = edition["receipt"]["inputs"]
            for proof in proofs:
                source = load_project(proof["project_id"])
                eligible = pilot._eligible_source(source, load_segments(source.id),
                                                  proof["utterance_id"],
                                                  expected_audio=proof["audio_sha256"])
                actual = {"take_id": eligible["take"].id,
                          "qa_report_sha256": pilot._qa_digest(eligible["take"]),
                          "requirement_fingerprint": eligible["requirement_fingerprint"],
                          "source_state_sha256": eligible["source_state_sha256"]}
                if any(proof.get(field) != value for field, value in actual.items()):
                    raise _refuse("an edition no longer matches its current source/cast/QA proof",
                                  "plan_stale")
            arm_proofs.append(proofs)
        if arm_proofs[0] != arm_proofs[1]:
            raise _refuse("a timing pair must use exactly the same ordered eligible Takes",
                          "direction_conflict")
    return build_pilot_projection(pilot_root=expected_root, pilot_project_id=pilot_project_id,
                                  book_title=book_title, index=index)
