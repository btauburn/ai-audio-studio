"""Reviewed exact public recording records, independent of synthetic-use approval.

No crawler, credentials or general government-host allowlist. An external model
cannot add records here or turn a recording license into performer consent.
Observed hashes pin retrieved originals; they are not publisher checksums.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping

_LITHGOW = {
    "source_id": "white-house-2011-john-lithgow-reading",
    "provider": "reviewed_official_recording",
    "performer": "John Lithgow",
    "source_page": "https://obamawhitehouse.archives.gov/photos-and-video/video/2011/04/25/2011-white-house-easter-egg-roll-john-lithgow-reads-his-books",
    "media_url": "https://obamawhitehouse.archives.gov/videos/2011/April/042511_EER_Lithgow_Read.mp3",
    "attribution": "The White House archive; John Lithgow reading at the 2011 Easter Egg Roll",
    "license": "Public Domain (recording page label)",
    "license_url": "https://obamawhitehouse.archives.gov/copyright",
    "terms_url": "https://obamawhitehouse.archives.gov/copyright",
    "recording_license_evidence": "Official recording page displays Public Domain and offers an MP3 download. Copyright policy distinguishes government-produced and third-party material.",
    "rights_scope": "Recording-page evidence only; no performer consent, endorsement or rights claim about underlying books.",
    "facts_verified_at": "2026-10-02",
    "language": "en",
    "duration_seconds": 1335.312189,
    "size_bytes": 21550980,
    "mime": "audio/mpeg",
    "reviewed_original_sha256": "00090f5ffa569683f24875c11a9b2e550dff50ad7da527798e8bbdd9794ad9f0",
    "checksum_origin": "locally_observed_download_not_publisher_checksum",
    "performance": "2011 White House Easter Egg Roll, reading from his books",
    "performance_linkage": "Official page identifies actor John Lithgow as the reader; no Dumbledore performance claim.",
    "preprocessing": [], "audio_observations": [], "original_path": None,
}
# Exact locally acquired corpus samples; no general host or download allowlist.
_dataset_raw = Path(__file__).with_name("reviewed_dataset_recordings.json").read_bytes()
if hashlib.sha256(_dataset_raw).hexdigest() != "059b8f40fbd4bb9d82ba7ba8329e630fbe0dac427101c957f9fad6658533f5fe":
    raise ValueError("Reviewed corpus registry differs from its pinned source")
_DATASETS = tuple(json.loads(_dataset_raw)["records"])
_RECORDS = (_LITHGOW, *_DATASETS)
_BOUND_FIELDS = tuple(_LITHGOW)


def records_for(performer: str) -> list[dict]:
    return [copy.deepcopy(row) for row in _RECORDS
            if row["performer"].casefold() == str(performer).strip().casefold()]


def reviewed(source: Mapping) -> dict | None:
    for row in _RECORDS:
        if all(source.get(key) == row[key] for key in row):
            return copy.deepcopy(row)
    return None


def reviewed_url(url: str) -> dict | None:
    return next((copy.deepcopy(row) for row in _RECORDS if row["media_url"] == url and not row.get("requires_cached_original")), None)


def verify_original(source: Mapping, raw: bytes) -> None:
    record = reviewed(source)
    if record is None:
        if source.get("provider") in {"reviewed_official_recording", "reviewed_dataset_recording"}:
            raise ValueError("Official recording does not match a reviewed exact source")
        return
    if (len(raw) != record["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != record["reviewed_original_sha256"]):
        raise ValueError("Official recording differs from the reviewed immutable original")
