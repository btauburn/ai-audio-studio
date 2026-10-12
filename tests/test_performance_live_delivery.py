"""CPU contract tests for verified private pilot delivery.

Every edition here is produced by the real M3 producers — the real receipt
builder, the real source-to-audio map builder and the real ffmpeg encode — over
synthetic Takes.  Every batch is published and served by the real delivery store
and the real private listener.  Only the Pushover transport is a double, because
sending a live push from a test would be exactly the thing this node must not do.
"""
from __future__ import annotations

import array
import hashlib
import json
import math
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dubbing_service.app.audiobook import performance_live_delivery as delivery
from dubbing_service.app.audiobook import performance_pilot as pilot
from dubbing_service.app.audiobook.delivery import notify, store as delivery_store
from dubbing_service.app.audiobook.delivery.server import create_app
from tests.test_performance_pilot_assembly import studio  # noqa: F401

PILOT_ID = "elf6pilot01"
LISTENING_URL = "https://desktop.example.ts.net:8444"
PORT = 8779
RATE = 24000
SECONDS = 2.0
FRAMES = int(RATE * SECONDS)


def tone(frames: int, frequency: float) -> bytes:
    values = array.array("h", (int(6000 * math.sin(2 * math.pi * frequency * i / RATE))
                               for i in range(frames)))
    return values.tobytes()


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(pcm)


def make_edition(pilot_root: Path, *, arm: str, timeline: str, frequency: float,
                 utterance: str) -> dict:
    """Build one genuine M3 edition with the real producers."""

    directory = (pilot_root / "performance" / "pilots" / PILOT_ID / arm / timeline)
    directory.mkdir(parents=True)
    master_pcm = tone(FRAMES, frequency)
    write_wav(directory / "review-master.wav", master_pcm)
    original_take = directory / "originals" / f"{utterance}.wav"
    write_wav(original_take, tone(FRAMES, frequency))
    take_sha256 = hashlib.sha256(original_take.read_bytes()).hexdigest()
    staging = directory / "staging"
    staging.mkdir()
    mp3_bytes = pilot._encode_mp3(directory / "review-master.wav", staging)
    (directory / "listening.mp3").write_bytes(mp3_bytes)

    compiled = {"timeline_sha256": timeline, "policy_sha256": "9" * 64,
                "classifier_version": "m2-v1", "assembler_version": "m3-v1",
                "sample_rate": RATE, "sample_domain": "int16"}
    placements = [{"kind": "speech", "utterance_id": utterance, "take_id": f"take-{utterance}",
                   "audio_sha256": take_sha256,
                   "source_start_frame": 0, "source_end_frame": FRAMES,
                   "offset_frames": 0, "derived_start_frame": 0, "derived_frames": FRAMES}]
    mapping, total = pilot._build_map(
        {"placements": placements, "timeline_sha256": timeline}, {},
        (1, 2, RATE))
    (directory / "source-to-audio-map.json").write_bytes(pilot._json_bytes(mapping))
    proofs = [{"utterance_id": utterance, "take_id": f"take-{utterance}",
               "audio_sha256": take_sha256,
               "role_id": "role-narrator", "requirement_fingerprint": "7" * 64,
               "source_revision_id": "rev-1", "spoken_text_revision_id": "spoken-1",
               "source_state_sha256": "8" * 64}]
    receipt = pilot._receipt({"timeline_sha256": timeline, "policy_sha256": "9" * 64,
                              "classifier_version": "m2-v1", "assembler_version": "m3-v1",
                              "sample_rate": RATE, "sample_domain": "int16"},
                             proofs, mapping, total,
                             (directory / "review-master.wav").read_bytes(), mp3_bytes,
                             directory, arm)
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    staging.rename(directory / "staging-done")
    return {"directory": directory, "receipt": receipt,
            "audio_sha256": hashlib.sha256(mp3_bytes).hexdigest(),
            "utterance": utterance, "arm": arm, "timeline": timeline}


@pytest.fixture
def editions(tmp_path):
    pilot_root = tmp_path / "projects" / PILOT_ID
    pilot_root.mkdir(parents=True)
    baseline = make_edition(pilot_root, arm="baseline", timeline="a" * 64,
                            frequency=220.0, utterance="utt-baseline")
    variant = make_edition(pilot_root, arm="variant", timeline="b" * 64,
                           frequency=330.0, utterance="utt-variant")
    assert baseline["audio_sha256"] != variant["audio_sha256"]
    index = {
        "schema_version": pilot.PAIR_INDEX_SCHEMA,
        "pilot_id": PILOT_ID,
        "seed": 1,
        "pairs": [{"label": "pair-001", "pair_id": "exchange-1",
                   "x_audio_sha256": variant["audio_sha256"],
                   "y_audio_sha256": baseline["audio_sha256"]}],
        "editions": {"exchange-1": "e" * 64},
        "production_enablement": False,
    }
    return {"pilot_root": pilot_root, "baseline": baseline, "variant": variant,
            "index": index, "state_dir": tmp_path / "pilot-state",
            "book_title": "ELF — Chapter 6 private timing review"}


@pytest.fixture
def published(editions):
    projection = delivery.build_pilot_projection(
        pilot_root=editions["pilot_root"], pilot_project_id=PILOT_ID,
        book_title=editions["book_title"], index=editions["index"])
    batch = delivery.publish_pilot_batch(state_dir=editions["state_dir"],
                                        projection=projection)
    verification = delivery.verify_published_batch(
        state_dir=editions["state_dir"], batch=batch,
        audio_roots=[str(editions["pilot_root"])])
    return {**editions, "projection": projection, "batch": batch,
            "verification": verification}


def _current_pair(studio):
    from tests.test_performance_pilot_assembly import build_inputs
    from dubbing_service.app.audiobook import performance_boundaries as boundaries
    inputs = build_inputs(studio, baseline_spacer_frames=12000)
    pilot.store_timeline_inputs(studio["pilot_id"], inputs)
    pilot.reserve_operation(studio["pilot_id"], studio["manifest"]["operation_id"], assemblies=2)
    timeline = boundaries.compile_timeline(inputs["rows"], policy=inputs["policy"],
                                           hints=inputs["hints"],
                                           allow_lead_offset=inputs["allow_lead_offset"])
    baseline = pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    variant = pilot.assemble_review(studio["pilot_id"], timeline, arm="variant")
    index, _ = pilot.blind_pairs(studio["pilot_id"], [{"pair_id": "source-exchange",
        "baseline_audio_sha256": baseline["artifacts"]["listening.mp3"],
        "variant_audio_sha256": variant["artifacts"]["listening.mp3"]}], seed=7)
    return index


def test_actual_m3_assembly_pair_publishes_only_with_current_eligible_takes(studio):
    from dubbing_service.app.storage.segments import load_segments, save_segments
    index = _current_pair(studio)
    root = pilot.safe_project_root(studio["pilot_id"])
    projection = delivery.build_current_pilot_projection(pilot_root=root,
        pilot_project_id=studio["pilot_id"], book_title="Real owner CPU contract", index=index)
    assert len(projection["chapters"]) == 2
    assert all(arm not in json.dumps(projection["chapters"]).lower()
               for arm in ("x_arm", "y_arm"))
    corrupted = json.loads(json.dumps(index))
    corrupted["pairs"][0]["x_audio_sha256"] = corrupted["pairs"][0]["y_audio_sha256"]
    with pytest.raises(delivery.LiveDeliveryError):
        delivery.build_current_pilot_projection(pilot_root=root,
            pilot_project_id=studio["pilot_id"], book_title="CPU", index=corrupted)
    source_rows = load_segments(studio["project"].id)
    selected = next(t for t in source_rows[0].takes if t.id == source_rows[0].selected_take_id)
    selected.stale = True
    save_segments(studio["project"].id, source_rows)
    with pytest.raises(pilot.PerformanceDirectorError):
        delivery.build_current_pilot_projection(pilot_root=root,
            pilot_project_id=studio["pilot_id"], book_title="CPU", index=index)


# --------------------------------------------------------------------------- the edition


def test_the_real_producers_build_an_edition_this_module_accepts(editions) -> None:
    edition = delivery.verify_assembly_receipt(editions["baseline"]["directory"],
                                               arm="baseline", timeline_sha256="a" * 64)
    assert edition["human_review"] == "pending"
    assert edition["production_enablement"] is False
    assert edition["total_frames"] == FRAMES
    assert abs(edition["duration_seconds"] - SECONDS) < 0.05
    assert edition["receipt"]["schema_version"] == pilot.ASSEMBLY_SCHEMA
    assert edition["receipt"]["schema_version"] != delivery.FOREIGN_EXPORT_SCHEMA


def test_tampered_or_missing_edition_bytes_are_refused(editions) -> None:
    baseline = editions["baseline"]["directory"]
    listening = baseline / "listening.mp3"
    original = listening.read_bytes()
    listening.write_bytes(original + b"\x00")
    with pytest.raises(delivery.LiveDeliveryError):
        delivery.verify_assembly_receipt(baseline, arm="baseline", timeline_sha256="a" * 64)
    listening.write_bytes(original)

    original_copy = baseline / "originals" / "utt-baseline.wav"
    saved = original_copy.read_bytes()
    original_copy.unlink()
    with pytest.raises(delivery.LiveDeliveryError):
        delivery.verify_assembly_receipt(baseline, arm="baseline", timeline_sha256="a" * 64)
    original_copy.write_bytes(saved)

    receipt_path = baseline / "receipt.json"
    saved_receipt = receipt_path.read_bytes()
    receipt_path.unlink()
    with pytest.raises(delivery.LiveDeliveryError):
        delivery.verify_assembly_receipt(baseline, arm="baseline", timeline_sha256="a" * 64)
    receipt_path.write_bytes(saved_receipt)
    assert delivery.verify_assembly_receipt(baseline, arm="baseline",
                                           timeline_sha256="a" * 64)


def test_a_changed_map_or_wrong_arm_or_timeline_is_refused(editions) -> None:
    baseline = editions["baseline"]["directory"]
    with pytest.raises(delivery.LiveDeliveryError):
        delivery.verify_assembly_receipt(baseline, arm="variant", timeline_sha256="a" * 64)
    with pytest.raises(delivery.LiveDeliveryError):
        delivery.verify_assembly_receipt(baseline, arm="baseline", timeline_sha256="c" * 64)

    map_path = baseline / "source-to-audio-map.json"
    saved = map_path.read_bytes()
    mapping = json.loads(saved.decode("utf-8"))
    mapping["rows"][0]["frames"] = mapping["rows"][0]["frames"] - 1
    map_path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(delivery.LiveDeliveryError):
        delivery.verify_assembly_receipt(baseline, arm="baseline", timeline_sha256="a" * 64)
    map_path.write_bytes(saved)


def test_an_assembly_receipt_is_never_relabelled_as_a_verified_listening_export(editions) -> None:
    directory = editions["baseline"]["directory"]
    receipt_path = directory / "receipt.json"
    saved = receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(saved)
    receipt.pop("receipt_sha256")
    receipt["schema_version"] = delivery.FOREIGN_EXPORT_SCHEMA
    receipt["status"] = delivery.FOREIGN_EXPORT_SCHEMA
    receipt["verified_loudness"] = True
    receipt["receipt_sha256"] = pilot.content_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(delivery.LiveDeliveryError) as refused:
        delivery.verify_assembly_receipt(directory, arm="baseline", timeline_sha256="a" * 64)
    assert refused.value.code == "direction_conflict"
    receipt_path.write_text(saved, encoding="utf-8")


def test_a_receipt_claiming_review_or_production_is_refused(editions) -> None:
    directory = editions["baseline"]["directory"]
    receipt_path = directory / "receipt.json"
    saved = receipt_path.read_text(encoding="utf-8")
    for field, value in (("human_review", "reviewed"), ("production_enablement", True)):
        receipt = json.loads(saved)
        receipt.pop("receipt_sha256")
        receipt[field] = value
        receipt["receipt_sha256"] = pilot.content_sha256(receipt)
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with pytest.raises(delivery.LiveDeliveryError):
            delivery.verify_assembly_receipt(directory, arm="baseline", timeline_sha256="a" * 64)
    receipt_path.write_text(saved, encoding="utf-8")


def test_only_a_verified_edition_resolves_for_a_digest(editions) -> None:
    resolved = delivery.locate_edition(editions["pilot_root"], PILOT_ID,
                                       editions["variant"]["audio_sha256"])
    assert resolved["arm"] == "variant"
    with pytest.raises(delivery.LiveDeliveryError) as refused:
        delivery.locate_edition(editions["pilot_root"], PILOT_ID, "0" * 64)
    assert refused.value.code == "measurement_uncertain"


# --------------------------------------------------------------------------- the batch


def test_the_batch_is_brian_only_and_carries_no_arm(published) -> None:
    batch = published["batch"]
    assert batch["recipient_id"] == "brian"
    assert batch["production_enablement"] is False
    assert batch["schema_version"] == delivery.BATCH_SCHEMA
    assert len(batch["chapters"]) == 2
    for chapter in batch["chapters"]:
        assert "verified_loudness" not in chapter
        assert chapter["human_review"] == "pending"
        assert chapter["export_manifest_sha256"] == chapter["assembly_receipt_sha256"]
        assert chapter["selected_takes"] == []
    text = json.dumps([{key: chapter[key] for key in
                        ("chapter_id", "title", "chapter_label", "revision")}
                        | {"labels_are_opaque": not any(
                            arm in str(chapter.get(key, "")).lower()
                            for arm in delivery.ArmNames for key in ("title", "chapter_label"))}
                       for chapter in batch["chapters"]]).lower()
    assert "baseline" not in text and "variant" not in text and "x_arm" not in text
    assert "x_arm" not in json.dumps(batch["chapters"])
    assert published["verification"]["verified"] is True
    assert published["verification"]["human_review"] == "pending"


def test_a_contaminated_index_is_refused(editions) -> None:
    with_key = {**editions["index"], "key": [{"x_arm": "variant"}]}
    with pytest.raises(delivery.LiveDeliveryError) as refused:
        delivery.build_pilot_projection(pilot_root=editions["pilot_root"],
                                       pilot_project_id=PILOT_ID,
                                       book_title=editions["book_title"], index=with_key)
    assert refused.value.code == "direction_conflict"

    contaminated = json.loads(json.dumps(editions["index"]))
    contaminated["pairs"][0]["x_arm"] = "variant"
    with pytest.raises(delivery.LiveDeliveryError):
        delivery.build_pilot_projection(pilot_root=editions["pilot_root"],
                                       pilot_project_id=PILOT_ID,
                                       book_title=editions["book_title"],
                                       index=contaminated)

    labelled = json.loads(json.dumps(editions["index"]))
    labelled["pairs"][0]["label"] = "variant"
    with pytest.raises(delivery.LiveDeliveryError):
        delivery.build_pilot_projection(pilot_root=editions["pilot_root"],
                                       pilot_project_id=PILOT_ID,
                                       book_title=editions["book_title"], index=labelled)


def test_a_batch_is_never_a_production_enablement(editions) -> None:
    projection = delivery.build_pilot_projection(
        pilot_root=editions["pilot_root"], pilot_project_id=PILOT_ID,
        book_title=editions["book_title"], index=editions["index"])
    with pytest.raises(delivery.LiveDeliveryError) as refused:
        delivery.publish_pilot_batch(state_dir=editions["state_dir"],
                                    projection={**projection, "production_enablement": True})
    assert refused.value.code == "direction_conflict"
    with pytest.raises(delivery.LiveDeliveryError):
        delivery.publish_pilot_batch(state_dir=editions["state_dir"],
                                    projection={**projection, "recipient_id": "mk"})


def test_published_bytes_must_match_and_stay_inside_the_roots(published) -> None:
    chapter = published["batch"]["chapters"][0]
    audio = Path(chapter["audio_path"])
    saved = audio.read_bytes()
    audio.write_bytes(saved + b"\x00")
    with pytest.raises(delivery.LiveDeliveryError) as refused:
        delivery.verify_published_batch(state_dir=published["state_dir"],
                                       batch=published["batch"],
                                       audio_roots=[str(published["pilot_root"])])
    assert refused.value.code == "plan_stale"
    audio.write_bytes(saved)

    with pytest.raises(delivery.LiveDeliveryError) as refused:
        delivery.verify_published_batch(state_dir=published["state_dir"],
                                       batch=published["batch"],
                                       audio_roots=[str(published["pilot_root"] / "elsewhere")])
    assert refused.value.code == "direction_conflict"

    stored_path = published["state_dir"] / "manifests" / (published["batch"]["batch_id"] + ".json")
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    tampered = json.loads(json.dumps(stored))
    tampered["chapters"][0]["audio_sha256"] = "0" * 64
    stored_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises((delivery.LiveDeliveryError, ValueError)):
        delivery.verify_published_batch(state_dir=published["state_dir"],
                                       batch=published["batch"],
                                       audio_roots=[str(published["pilot_root"])])
    stored_path.write_text(json.dumps(stored), encoding="utf-8")
    assert delivery.verify_published_batch(state_dir=published["state_dir"],
                                          batch=published["batch"],
                                          audio_roots=[str(published["pilot_root"])])


def test_the_real_private_listener_serves_only_the_opaque_batch(published, editions) -> None:
    config = {"project_dir": str(editions["pilot_root"]),
              "state_dir": str(editions["state_dir"]), "port": PORT,
              "listening_base_url": LISTENING_URL,
              "allowed_audio_roots": [str(editions["pilot_root"])],
              "recipients": {"brian": {"id": "brian", "enabled": True,
                                       "tailscale_login": "brian@example.test",
                                       "allow_download": False, "allow_feedback": False},
                             "mk": {"id": "mk", "enabled": False}}}
    batch = published["batch"]
    with TestClient(create_app(config), base_url=f"http://127.0.0.1:{PORT}",
                    client=("127.0.0.1", 54321),
                    headers={"Tailscale-User-Login": "brian@example.test"}) as client:
        manifest = client.get(f"/api/batches/{batch['batch_id']}")
        assert manifest.status_code == 200
        text = manifest.text.lower()
        assert "baseline" not in text and "variant" not in text and "x_arm" not in text
        assert "audio_path" not in text and "selected_takes" not in text
        chapter_id = batch["chapters"][0]["chapter_id"]
        head = client.get(f"/audio/{batch['batch_id']}/{chapter_id}",
                          headers={"Range": "bytes=0-1023"})
        assert head.status_code == 206
        assert head.headers["content-range"].startswith("bytes 0-1023/")
        with Path(batch["chapters"][0]["audio_path"]).open("rb") as handle:
            assert head.content == handle.read(1024)
        full = client.get(f"/audio/{batch['batch_id']}/{chapter_id}")
        assert full.status_code == 200
        assert hashlib.sha256(full.content).hexdigest() == batch["chapters"][0]["audio_sha256"]


# --------------------------------------------------------------------------- notification


class FakePublisher:
    def __init__(self) -> None:
        self.sent: list = []

    def send(self, rendered, timeout=0):
        self.sent.append(rendered)
        return FakeResponse()


class FakeResponse:
    state = "accepted"
    attempts = 1
    http_status = 200
    request_id = "fake-request"


class FakeConfig:
    app_token = "fake-token"
    user_key = "fake-user"
    timeout_seconds = 1


class FakeMessage:
    def __init__(self, **fields) -> None:
        self.fields = fields


@pytest.fixture
def transport(monkeypatch):
    publisher = FakePublisher()
    monkeypatch.setattr(notify, "_load_transport",
                        lambda root: (FakeConfig(), publisher, FakeMessage, None))
    return publisher


def recipient() -> dict:
    return {"id": "brian", "enabled": True, "notification_transport": "pushover"}


def test_one_notification_then_duplicate_suppression(published, transport) -> None:
    first = delivery.notify_pilot_ready(
        state_dir=published["state_dir"], batch=published["batch"],
        verification=published["verification"], recipient=recipient(),
        transport_root=Path("unused"), listening_url=LISTENING_URL)
    assert first["attempted"] is True
    assert first["result"]["state"] == "accepted"
    assert first["device_delivery_confirmed"] is False
    assert first["human_listening"] == "pending"
    assert len(transport.sent) == 1

    second = delivery.notify_pilot_ready(
        state_dir=published["state_dir"], batch=published["batch"],
        verification=published["verification"], recipient=recipient(),
        transport_root=Path("unused"), listening_url=LISTENING_URL)
    assert second["result"].get("duplicate") is True
    assert second["result"]["accepted"] is False
    assert len(transport.sent) == 1


def test_an_unknown_outcome_blocks_an_automatic_resend(published, transport) -> None:
    directory = published["state_dir"] / "notifications"
    (directory / "intents").mkdir(parents=True, exist_ok=True)
    (directory / "outcomes").mkdir(parents=True, exist_ok=True)
    message, artifacts = notify._message(published["batch"], recipient(), LISTENING_URL)
    attempt_id = "0" * 32
    (directory / "intents" / f"{attempt_id}.json").write_text(json.dumps({
        "schema_version": 1, "attempt_id": attempt_id, "artifact_ids": artifacts,
        "recipient_id": "brian", "project_id": published["batch"]["project_id"],
        "batch_id": published["batch"]["batch_id"]}), encoding="utf-8")
    (directory / "outcomes" / f"{attempt_id}.json").write_text(json.dumps({
        "schema_version": 1, "attempt_id": attempt_id, "state": "delivery_unknown"}),
        encoding="utf-8")
    result = delivery.notify_pilot_ready(
        state_dir=published["state_dir"], batch=published["batch"],
        verification=published["verification"], recipient=recipient(),
        transport_root=Path("unused"), listening_url=LISTENING_URL)
    assert result["result"].get("duplicate") is True
    assert result["result"]["previous_state"] == "delivery_unknown"
    assert transport.sent == []


def test_notification_refuses_unverified_or_foreign_batches(published) -> None:
    with pytest.raises(delivery.LiveDeliveryError) as refused:
        delivery.notify_pilot_ready(
            state_dir=published["state_dir"], batch=published["batch"],
            verification={**published["verification"], "verified": False},
            recipient=recipient(), transport_root=Path("unused"),
            listening_url=LISTENING_URL)
    assert refused.value.code == "review_pending"

    with pytest.raises(delivery.LiveDeliveryError) as refused:
        delivery.notify_pilot_ready(
            state_dir=published["state_dir"], batch=published["batch"],
            verification={**published["verification"], "batch_id": "0" * 64},
            recipient=recipient(), transport_root=Path("unused"),
            listening_url=LISTENING_URL)
    assert refused.value.code == "plan_stale"

    with pytest.raises(delivery.LiveDeliveryError):
        delivery.notify_pilot_ready(
            state_dir=published["state_dir"], batch=published["batch"],
            verification=published["verification"],
            recipient={"id": "mk", "enabled": True, "notification_transport": "pushover"},
            transport_root=Path("unused"), listening_url=LISTENING_URL)


def test_a_bad_transport_never_reports_a_send(published) -> None:
    result = delivery.notify_pilot_ready(
        state_dir=published["state_dir"], batch=published["batch"],
        verification=published["verification"], recipient=recipient(),
        transport_root=Path("no-such-transport-root"), listening_url=LISTENING_URL)
    assert result["result"]["accepted"] is False
    assert result["result"]["state"] == "skipped"
    assert result["device_delivery_confirmed"] is False
    assert not list((published["state_dir"] / "notifications" / "intents").glob("*.json"))
