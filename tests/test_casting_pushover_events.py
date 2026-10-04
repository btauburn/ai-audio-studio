"""Auto Cast human-review Pushover events: content, aggregation and durability.

Every case drives the real ``casting_review`` owner over the canonical project
store and replaces only the two external boundaries: the private delivery
configuration and the Pushover transport itself.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from dubbing_service.app.audiobook import book_creator_notifications
from dubbing_service.app.audiobook import casting_review as casting
from dubbing_service.app.audiobook import voice_lab as lab
from dubbing_service.app.audiobook.delivery import notify, server
from dubbing_service.app.audiobook.fingerprint import file_sha256, voice_revision_for_role
from dubbing_service.app.models.project import Role, Segment
from dubbing_service.app.storage.projects import create_project, load_project, save_project
from dubbing_service.app.storage.segments import save_segments
from tests.audiobook_helpers import make_tone_wav, use_projects_root

TAILNET = "https://brian-desktop.tail3e7427.ts.net:8443"
ROLES = (("role-narrator", "Narrator"), ("role-saphienne", "Saphienne"),
         ("role-kylantha", "Kylantha"), ("role-unresolved", "Elara"))


class Transport:
    """A scripted Pushover boundary that records exactly what it was asked to send."""

    def __init__(self, decisions=None):
        self.sent = []
        self.decisions = list(decisions or [])

    def __call__(self, path):
        outer = self

        class Message:
            def __init__(self, **values):
                self.__dict__.update(values)

        class Publisher:
            def send(self, message, *, timeout):
                outer.sent.append(message)
                outcome = outer.decisions.pop(0) if outer.decisions else {"state": "accepted"}
                if outcome == "raise":
                    raise ConnectionError("TEST transport unavailable")
                return SimpleNamespace(attempts=1, http_status=200, request_id="safe-request", **outcome)

        return (SimpleNamespace(app_token="test-token", user_key="test-key", device=None,
                                timeout_seconds=2.0),
                Publisher(), Message, object())


@pytest.fixture
def world(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    # This fixture's projects are production-scope, so a 4- or 5-star rating
    # publishes to the global voice library. Point that store at a disposable
    # directory: a test must never write to the machine's real voice library.
    voices = tmp_path / "voice-library"
    voices.mkdir()
    monkeypatch.setattr("dubbing_service.app.storage.voices.voices_audio_dir", lambda: voices)
    monkeypatch.setattr("dubbing_service.app.storage.voices._voices_file_path",
                        lambda: voices.parent / "voices.json")
    project = create_project("Elf_1_10 AutoCast Acceptance")
    project.metadata["autocast_import"] = {
        "schema": "autocast-import-v1", "mode": "autocast", "status": "ready",
        "private_review_recipient": "brian",
    }
    project.roles = [Role(id=role_id, name=name, role_type="character", tts_backend="qwen3tts")
                     for role_id, name in ROLES]
    save_project(project)
    save_segments(project.id, [
        Segment(id=f"line-{index}", index=index, text=f"{name} speaks in this fixture.",
                role_id=role_id)
        for index, (role_id, name) in enumerate(ROLES)])
    transport = Transport()
    monkeypatch.setattr(notify, "_load_transport", transport)
    config = {"notification_transport_root": str(tmp_path / "transport"),
              "listening_base_url": TAILNET, "project_dir": str(root / project.id)}
    monkeypatch.setattr(book_creator_notifications, "_private_config",
                        lambda: (dict(config), {"id": "brian", "enabled": True,
                                                "notification_transport": "pushover"}))
    monkeypatch.setattr(server, "shared_lab_project", lambda *args: True)
    monkeypatch.setattr(casting, "COALESCE_SECONDS", 0)
    return SimpleNamespace(root=root, project_id=project.id, sent=transport.sent,
                           config=config, transport=transport, seq=0)


def prepare(world, request_id="prepare-all"):
    return casting.prepare(world.project_id, request_id=request_id)


def _record(root, project, role_id, key, *, fixture_only=False):
    # Each candidate is a different sound, as a real replacement round is; the
    # delivery ledger treats identical bytes as the same already-reviewed audio.
    tone = 180.0 + (sum((index + 1) * ord(char) for index, char in enumerate(key)) % 120)
    audio = make_tone_wav(root / project.id / "voices" / f"{key}.wav", seconds=0.8, frequency=tone)
    reference = make_tone_wav(root / project.id / "voices" / f"{key}-ref.wav", seconds=0.8,
                              frequency=tone + 40)
    prompt = root / project.id / "voices" / f"{key}.npz"
    prompt.write_bytes(b"SYNTHETIC-FIXTURE-PROMPT-CONDITIONING")
    identity = {
        "id": f"identity-{key}", "role_id": role_id, "status": "LOCKED", "mechanism": "qwen_reference",
        "dossier_revision": "fixture-dossier", "casting_profile_revision": "fixture-profile",
        "design_instruction": "Synthetic fixture voice.",
        "reference_audio": reference.relative_to(root / project.id).as_posix(),
        "reference_audio_sha256": file_sha256(reference), "reference_text": "Synthetic fixture reference.",
        "prompt_artifact": prompt.relative_to(root / project.id).as_posix(),
        "prompt_sha256": file_sha256(prompt),
        "backend": "qwen3tts", "model": "SYNTHETIC-FIXTURE-ONLY", "model_revision": "fixture",
        "runtime_version": "fixture",
        "provenance": {"origin": "generated_original_voice", "fixture_only": fixture_only},
    }
    audio_hash = file_sha256(audio)
    return {"key": key, "voice_id": f"voice-id-{key}",
            "name": f"Audition {key}", "kind": "original_inspired", "synthetic": True,
            "version": voice_revision_for_role(next(r for r in project.roles if r.id == role_id)),
            "audio_file": audio.relative_to(root / project.id).as_posix(),
            "audio_sha256": audio_hash, "identity": identity, "backend": "qwen3tts",
            "technical_qa": "passed", "take_id": f"take-{key}", "sections": [], "duration_seconds": 0.8}


def ready_candidate(world, role_id, *, index=0, key=None, fixture_only=False):
    """Mark one already-queued candidate as genuinely playable audio."""
    project = load_project(world.project_id)
    item = project.voice_lab.casting["roles"][role_id]
    current = item["round"]
    key = key or f"voice-{role_id}-{index}"
    record = _record(world.root, project, role_id, key, fixture_only=fixture_only)
    candidate = current["candidates"][index]
    candidate.update(status="ready", key=key, audio_sha256=record["audio_sha256"],
                     identity_id=f"effective-{key}", duration_seconds=0.8, technical_qa="passed",
                     sections=[], voice_version=record["version"], ready_at=lab.now())
    project.voice_lab.voices[key] = record
    save_project(project)
    return {"candidate": candidate, "record": record, "round_revision": current["revision"],
            "assignment_revision": item["assignment_revision"]}


def finish_round(world, role_id):
    project = load_project(world.project_id)
    current = project.voice_lab.casting["roles"][role_id]["round"]
    current["status"] = "awaiting_review"
    current["completed_at"] = lab.now()
    save_project(project)


def fill(world, role_id, *, count=3, prefix=None):
    """Ready ``count`` candidates and leave the round exactly as a worker would."""
    project = load_project(world.project_id)
    round_revision = project.voice_lab.casting["roles"][role_id]["round"]["revision"]
    prefix = prefix or f"{role_id}-r{round_revision}"
    ready = [ready_candidate(world, role_id, index=index, key=f"{prefix}-{index}")
             for index in range(count)]
    finish_round(world, role_id)
    return ready


def decision(world, role_id, *, stars, action="save", raw_direction="", index=0, event_id=None):
    project = load_project(world.project_id)
    item = project.voice_lab.casting["roles"][role_id]
    current = item["round"]
    candidate = current["candidates"][index]
    world.seq += 1
    return {"event_id": event_id or f"evt-{world.seq}",
            "role_id": role_id, "candidate_id": candidate["candidate_id"],
            "round_revision": current["revision"], "assignment_revision": item["assignment_revision"],
            "key": candidate["key"], "audio_sha256": candidate["audio_sha256"],
            "action": action, "stars": stars, "raw_direction": raw_direction}


def notification(world, role_id):
    return load_project(world.project_id).voice_lab.casting["roles"][role_id]["round"]["notification"]


def make_due(world, role_id):
    project = load_project(world.project_id)
    record = project.voice_lab.casting["roles"][role_id]["round"]["notification"]
    record["next_retry_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    save_project(project)


def dispatch(world):
    """Run the maintenance owner with its workers stubbed out.

    The real worker releases its own active flag when it finishes; a stubbed
    dispatch must do the same or the next pass would skip this project.
    """
    from dubbing_service.app.storage.jobs import load_jobs, mark_job_active
    for job in load_jobs(world.project_id):
        if job.id.startswith(("cast_", "strict_")):
            mark_job_active(world.project_id, job.id, False)
    casting.maintenance_tick(project_ids=[world.project_id], schedule=lambda *args: None)


def receipts(world):
    return sorted((world.root / world.project_id / "delivery/notifications/outcomes").glob("*.json"))


# --------------------------------------------------------------------------
# A / B - one useful review push per reviewable character
# --------------------------------------------------------------------------

def test_auditions_ready_sends_exactly_one_deep_linked_review_push(world):
    prepare(world)
    fill(world, "role-saphienne", count=3)
    result = casting.notify_ready(world.project_id)

    assert result["state"] == "accepted"
    assert len(world.sent) == 1, "three auditions for one character is one review, not three pushes"
    message = world.sent[0]
    assert message.title == "🎙 Voice ready: Saphienne"
    assert message.body == ("3 auditions are ready.\nTap to listen, rate, and choose a voice.\n"
                            "Project: Elf_1_10 AutoCast Acceptance\n"
                            "3 / 4 characters remaining.")
    assert message.url == f"{TAILNET}/voice-lab?project={world.project_id}&character=role-saphienne"
    assert message.url_title == "Review Saphienne"
    assert message.priority == 0
    assert "127.0.0.1" not in message.url and "localhost" not in message.url


# --------------------------------------------------------------------------
# C / D - reading the review never notifies, and state is processed once
# --------------------------------------------------------------------------

def test_reading_the_review_page_never_notifies_and_duplicate_state_is_one_push(world):
    prepare(world)
    fill(world, "role-saphienne", count=2)
    for _ in range(4):
        casting.view(world.project_id)
    assert world.sent == [], "opening or refreshing Voice Lab has zero push effect"

    assert casting.notify_ready(world.project_id)["state"] == "accepted"
    assert casting.notify_ready(world.project_id)["state"] == "idle"
    assert len(world.sent) == 1

    # A crash after the ledger accepted the push but before the project
    # projection was written must still not submit the same audio twice.
    project = load_project(world.project_id)
    project.voice_lab.casting["roles"]["role-saphienne"]["round"]["notification"] = {"state": "pending"}
    save_project(project)
    assert casting.notify_ready(world.project_id)["state"] == "accepted"
    assert len(world.sent) == 1


# --------------------------------------------------------------------------
# E / F - durable outbox and persisted receipt
# --------------------------------------------------------------------------

def test_pending_outbox_record_resumes_and_receipt_is_persisted(world):
    prepare(world)
    fill(world, "role-saphienne", count=1)
    assert casting.notify_ready(world.project_id)["state"] == "accepted"

    stored = notification(world, "role-saphienne")
    assert stored["state"] == "accepted"
    assert stored["event"] == "CHARACTER_REVIEW_READY"
    assert stored["delivery_status"] == "accepted" and stored["delivered_at"]
    assert stored["review_url"].endswith("&character=role-saphienne")
    assert stored["device_delivery"] == "unobserved" and stored["human_feedback"] == "pending"
    outcomes = receipts(world)
    assert len(outcomes) == 1 and len(list(outcomes[0].parent.parent.glob("intents/*.json"))) == 1
    assert "test-token" not in outcomes[0].read_text(encoding="utf-8")
    assert "test-key" not in outcomes[0].read_text(encoding="utf-8")
    assert "test-token" not in json.dumps(stored)

    # A second character proves an unsent durable event still resumes, because
    # the claim is written before any network call.
    fill(world, "role-kylantha", count=1)
    assert casting.notify_ready(world.project_id)["state"] == "accepted"
    assert len(world.sent) == 2
    assert world.sent[1].url.endswith("&character=role-kylantha")


# --------------------------------------------------------------------------
# G / H - bounded retries, and a permanent failure stops
# --------------------------------------------------------------------------

def test_transient_failure_retries_then_succeeds(world):
    world.transport.decisions = [{"state": "failed"}, {"state": "accepted"}]
    prepare(world)
    fill(world, "role-saphienne", count=1)
    assert casting.notify_ready(world.project_id)["state"] == "failed"
    assert notification(world, "role-saphienne")["next_retry_at"] is not None
    make_due(world, "role-saphienne")
    assert casting.notify_ready(world.project_id)["state"] == "accepted"
    assert len(world.sent) == 2
    assert world.sent[0].body == world.sent[1].body


def test_a_provider_outage_never_abandons_a_character_waiting_for_feedback(world):
    """A review the human is waiting on keeps being announced until it is accepted.

    The early attempts back off quickly; once they are spent the interval stops at a
    capped value instead of the notification being dropped, so an outage can never
    leave a waiting character that nobody was told about.
    """
    world.transport.decisions = [{"state": "failed"}] * 5 + [{"state": "accepted"}]
    prepare(world)
    fill(world, "role-saphienne", count=1)
    for attempt in range(5):
        assert casting.notify_ready(world.project_id)["state"] == "failed"
        record = notification(world, "role-saphienne")
        assert record["next_retry_at"] is not None, f"attempt {attempt + 1} must keep a retry clock"
        if attempt >= casting.MAX_NOTIFICATION_ATTEMPTS:
            wait = (datetime.fromisoformat(record["next_retry_at"]) - datetime.now(timezone.utc)).total_seconds()
            assert wait <= casting.EXHAUSTED_RETRY_SECONDS + 5, "the interval is capped, not unbounded"
        make_due(world, "role-saphienne")
    assert casting.notify_ready(world.project_id)["state"] == "accepted"
    assert len(world.sent) == 6
    make_due(world, "role-saphienne")
    assert casting.notify_ready(world.project_id)["state"] == "idle", "once accepted it is not repeated"
    assert len(world.sent) == 6


def test_playable_auditions_are_announced_even_when_a_sibling_is_still_generating(world):
    """One slow or stuck voice must not hold back the review of the ones that are ready."""
    prepare(world)
    ready_candidate(world, "role-saphienne", index=0, key="saphienne-partial-0")
    ready_candidate(world, "role-saphienne", index=1, key="saphienne-partial-1")
    project = load_project(world.project_id)
    round_ = project.voice_lab.casting["roles"]["role-saphienne"]["round"]
    assert round_["status"] != "awaiting_review", "the round is still producing a sibling"
    assert casting.notify_ready(world.project_id)["state"] == "idle", "a fresh partial round is not announced yet"
    assert not world.sent
    for candidate in round_["candidates"]:
        if candidate.get("ready_at"):
            candidate["ready_at"] = (datetime.now(timezone.utc)
                                     - timedelta(seconds=casting.PARTIAL_REVIEW_SECONDS + 60)).isoformat()
    save_project(project)
    assert casting.notify_ready(world.project_id)["state"] == "accepted"
    assert len(world.sent) == 1
    assert "Saphienne" in world.sent[0].title or "Saphienne" in world.sent[0].body


def test_a_new_round_is_a_new_notification_even_when_its_audio_is_identical():
    """The ledger dedupes by artifact identity, so the round must be part of it.

    Otherwise a re-cast whose audio happens to match an earlier round is reported as
    "already notified" and the human is never told a new round is waiting.
    """
    sha = "a" * 64

    def identities(round_id):
        batch = {"project_id": "project-1", "event": "CHARACTER_REVIEW_READY", "book_title": "Book",
                 "characters_total": 1,
                 "characters": [{"role_id": "role-a", "name": "A", "options": 3,
                                 "audio_sha256": sha, "round_id": round_id}]}
        return notify._casting_message(batch, TAILNET + "/voice-lab?project=project-1")[1]

    assert identities("round-1") != identities("round-2")
    assert identities("round-1") == identities("round-1"), "the same round is still deduplicated"
    assert identities(None) != identities("round-1")


# --------------------------------------------------------------------------
# I / J / K - the human feedback loop decides what becomes notifiable
# --------------------------------------------------------------------------

def test_low_rating_queues_a_replacement_that_notifies_once_ready(world):
    prepare(world)
    fill(world, "role-saphienne", count=3)
    assert casting.notify_ready(world.project_id)["state"] == "accepted"

    casting.feedback(world.project_id, decision(world, "role-saphienne", stars=1))

    # The rating is recorded and a replacement requested, but the reviewed round
    # is left intact so the remaining options can still be scored.
    project = load_project(world.project_id)
    reviewed = project.voice_lab.casting["roles"]["role-saphienne"]["round"]
    assert reviewed["replacement"] is False and reviewed["status"] == "awaiting_review"
    assert reviewed["candidates"][0]["rating"] == 1
    assert reviewed["candidates"][0]["needs_replacement"]["rating"] == 1
    request = project.voice_lab.casting["replacement_queue"]["role-saphienne"]
    assert request["status"] == "waiting_for_review"
    assert casting.notify_ready(world.project_id)["state"] == "idle"

    # Scoring the rest of the heard set releases the replacement round.
    for index in (1, 2):
        casting.feedback(world.project_id, decision(world, "role-saphienne", stars=4, index=index))
    dispatch(world)
    replacement = load_project(world.project_id).voice_lab.casting["roles"]["role-saphienne"]["round"]
    assert replacement["replacement"] is True
    assert replacement["status"] == "pending"
    assert casting.notify_ready(world.project_id)["state"] == "idle", "queued work is not reviewable work"

    fill(world, "role-saphienne", count=3)
    assert casting.notify_ready(world.project_id)["state"] == "accepted"
    assert len(world.sent) == 2
    newest = world.sent[1]
    assert newest.title == "🎙 New Saphienne voice ready"
    assert newest.url.endswith("&character=role-saphienne")
    rejected = (load_project(world.project_id).voice_lab.casting["roles"]["role-saphienne"]["assignment"] or {})
    assert rejected.get("rejected") is True, "the rejected candidate stays historically recorded"


def test_qualitative_feedback_is_preserved_and_summarized_safely(world):
    feedback = "deeper, more British, less nasal"
    prepare(world)
    fill(world, "role-saphienne", count=3)
    casting.notify_ready(world.project_id)
    result = casting.feedback(world.project_id, decision(world, "role-saphienne", stars=2,
                                                         raw_direction=feedback))
    assert result["direction"]["requested"] == [feedback]

    project = load_project(world.project_id)
    stored = project.voice_lab.casting["roles"]["role-saphienne"]["round"]
    assert stored["direction"]["requested"] == [feedback]
    assert stored["replacement"] is False, "qualitative feedback never rewrites the round being scored"
    for index in (1, 2):
        casting.feedback(world.project_id, decision(world, "role-saphienne", stars=4, index=index))
    dispatch(world)
    fill(world, "role-saphienne", count=3)
    assert casting.notify_ready(world.project_id)["state"] == "accepted"

    newest = world.sent[1]
    assert newest.title == "🎙 New Saphienne voice ready"
    assert f'Your feedback:\n"{feedback}"' in newest.body
    assert len(newest.body) <= 1024

    # A very long free-text direction must still yield a bounded push. Driving a
    # second replacement proves the same contract on a later round.
    long_text = "warmer and slower " * 200
    casting.feedback(world.project_id, decision(world, "role-saphienne", stars=2,
                                                raw_direction=long_text))
    for index in (1, 2):
        casting.feedback(world.project_id, decision(world, "role-saphienne", stars=4, index=index))
    dispatch(world)
    fill(world, "role-saphienne", count=3)
    assert casting.notify_ready(world.project_id)["state"] == "accepted"
    long_body = world.sent[2].body
    assert len(long_body) <= 1024
    assert "warmer and slower" in long_body


def test_five_stars_and_approval_never_queue_a_replacement_push(world):
    prepare(world)
    fill(world, "role-saphienne", count=3)
    casting.notify_ready(world.project_id)
    before = len(world.sent)

    casting.feedback(world.project_id, decision(world, "role-saphienne", stars=5, action="save"))
    project = load_project(world.project_id)
    assert project.voice_lab.casting["roles"]["role-saphienne"]["round_revision"] == 1
    assert casting.notify_ready(world.project_id)["state"] == "idle"

    casting.feedback(world.project_id, decision(world, "role-saphienne", stars=5, action="use"))
    project = load_project(world.project_id)
    assert casting.approved(project, "role-saphienne")
    casting.notify_ready(world.project_id)
    assert len(world.sent) == before, "an approved voice must not produce a replacement push"


# --------------------------------------------------------------------------
# L / M - aggregation, and exactly one completion push
# --------------------------------------------------------------------------

def test_characters_finishing_together_send_one_aggregated_push(world):
    prepare(world)
    fill(world, "role-saphienne", count=3)
    fill(world, "role-kylantha", count=3)

    assert casting.notify_ready(world.project_id)["state"] == "accepted"
    assert len(world.sent) == 1
    message = world.sent[0]
    assert message.title == "🎭 2 characters ready to cast"
    assert message.body == ("Saphienne and Kylantha have new auditions ready.\n"
                            "6 voice options are waiting for your feedback.")
    assert message.url == f"{TAILNET}/voice-lab?project={world.project_id}&view=casting"
    assert message.url_title == "Open Voice Lab"


def test_all_characters_approved_sends_exactly_one_completion_push(world):
    prepare(world)
    for role_id, _name in ROLES:
        fill(world, role_id, count=1)
        casting.feedback(world.project_id, decision(world, role_id, stars=5, action="use"))
    project = load_project(world.project_id)
    assert all(casting.approved(project, role_id) for role_id, _name in ROLES)
    assert world.sent == [], "approving a voice is not itself a review notification"

    # Approving the last character is the last human action: the same maintenance
    # owner that starts casting also starts the render and announces completion.
    casting.maintenance_tick(project_ids=[world.project_id], schedule=lambda *a: None)
    state = casting._state(load_project(world.project_id))
    receipt = state["productions"][casting.AUTOMATIC_PRODUCTION_ID]
    assert receipt["automatic"] is True and receipt["required_roles"]

    assert len(world.sent) == 1, "completion is pushed exactly once"
    final = world.sent[0]
    assert final.title == "✅ Casting complete: Elf_1_10 AutoCast Acceptance"
    assert final.body == ("All 4 required characters have approved voices.\n"
                          "Tap to see how your book is coming along.")
    assert final.url == f"{TAILNET}/voice-lab?project={world.project_id}&view=casting"
    assert final.url_title == "Review final cast"
    complete_key = next(key for key in state["notifications"] if key.startswith("complete:"))
    assert state["notifications"][complete_key]["event"] == "CASTING_COMPLETE"
    assert state["notifications"][complete_key]["state"] == "accepted"

    sent = len(world.sent)
    for _ in range(3):
        casting.notify_ready(world.project_id)
        casting.maintenance_tick(project_ids=[world.project_id], schedule=lambda *a: None)
    assert len(world.sent) == sent, "completion is announced exactly once"


def test_a_finished_render_sends_exactly_one_production_push(world):
    """The end of the automatic render is its own event, once per receipt state."""
    prepare(world)
    for role_id, _name in ROLES:
        fill(world, role_id, count=1)
        casting.feedback(world.project_id, decision(world, role_id, stars=5, action="use"))
    casting.maintenance_tick(project_ids=[world.project_id], schedule=lambda *a: None)
    world.sent.clear()

    def finish(status, *, reason=None, manifest=None):
        with casting.project_meta_lock(world.project_id):
            project = load_project(world.project_id)
            receipt = casting._state(project)["productions"][casting.AUTOMATIC_PRODUCTION_ID]
            receipt.update(status=status, reason=reason, manifest=manifest or {
                "manifest_id": "manifest_fixture_0001"})
            casting._save(project)

    # A render still in flight is not announced.
    assert casting._production_event(load_project(world.project_id)) is None
    assert (casting.notify_ready(world.project_id).get("terminal") or {}).get("PRODUCTION_READY") is None
    assert world.sent == [], "a running render must not be announced"

    finish("succeeded")
    result = casting.notify_ready(world.project_id)
    assert result["terminal"]["PRODUCTION_READY"] == "accepted"
    ready = next(m for m in world.sent if m.title.startswith("🎧"))
    assert ready.title == "🎧 Audiobook ready: Elf_1_10 AutoCast Acceptance"
    assert "ready to listen to" in ready.body
    assert "manifest_fixture_0001" in ready.body
    assert ready.url == f"{TAILNET}/voice-lab?project={world.project_id}&view=casting"

    sent = len(world.sent)
    for _ in range(3):
        casting.notify_ready(world.project_id)
    assert len(world.sent) == sent, "a finished render is announced exactly once"


def test_a_stopped_render_asks_for_a_resume_instead_of_claiming_completion(world):
    prepare(world)
    for role_id, _name in ROLES:
        fill(world, role_id, count=1)
        casting.feedback(world.project_id, decision(world, role_id, stars=5, action="use"))
    casting.maintenance_tick(project_ids=[world.project_id], schedule=lambda *a: None)
    world.sent.clear()
    with casting.project_meta_lock(world.project_id):
        project = load_project(world.project_id)
        casting._state(project)["productions"][casting.AUTOMATIC_PRODUCTION_ID].update(
            status="failed", reason="Bounded production attempts exhausted")
        casting._save(project)

    result = casting.notify_ready(world.project_id)
    assert result["terminal"]["PRODUCTION_READY"] == "accepted"
    stopped = next(m for m in world.sent if "stopped" in m.title)
    assert stopped.title == "⚠️ Audiobook generation stopped: Elf_1_10 AutoCast Acceptance"
    assert stopped.body == ("Bounded production attempts exhausted\n"
                            "Tap to see what your book needs.")
    assert stopped.url_title == "Open Voice Lab"


def test_an_auto_started_render_is_not_queued_twice(world):
    """The auto-start is idempotent: later ticks reuse the existing receipt."""
    prepare(world)
    for role_id, _name in ROLES:
        fill(world, role_id, count=1)
        casting.feedback(world.project_id, decision(world, role_id, stars=5, action="use"))
    for _ in range(4):
        casting.maintenance_tick(project_ids=[world.project_id], schedule=lambda *a: None)
    state = casting._state(load_project(world.project_id))
    automatic = [key for key, row in state["productions"].items() if row.get("automatic")]
    assert automatic == [casting.AUTOMATIC_PRODUCTION_ID]


def test_an_incomplete_cast_is_never_rendered(world):
    """Auto-start requires every required character; a partial cast is left alone."""
    prepare(world)
    role_id, _name = ROLES[0]
    fill(world, role_id, count=1)
    casting.feedback(world.project_id, decision(world, role_id, stars=5, action="use"))
    project = load_project(world.project_id)
    assert casting.approved(project, role_id)
    assert not all(casting.approved(project, r) for r, _ in ROLES)
    casting.maintenance_tick(project_ids=[world.project_id], schedule=lambda *a: None)
    state = casting._state(load_project(world.project_id))
    assert casting.AUTOMATIC_PRODUCTION_ID not in state["productions"], \
        "a partly-reviewed cast must not be rendered with unapproved voices"


# --------------------------------------------------------------------------
# N - a stuck character notifies without stopping the rest
# --------------------------------------------------------------------------

def test_unresolved_character_gets_actionable_attention_and_the_rest_continue(world):
    prepare(world)
    fill(world, "role-saphienne", count=2)
    project = load_project(world.project_id)
    project.metadata["autocast_import"]["unresolved_utterance_ids"] = ["line-3"]
    project.voice_lab.casting["roles"]["role-unresolved"].pop("round", None)
    save_project(project)

    result = casting.notify_ready(world.project_id)
    assert result["state"] == "accepted", "valid characters keep casting independently"
    assert result["terminal"]["CASTING_NEEDS_ATTENTION"] == "accepted"
    titles = [message.title for message in world.sent]
    assert "🎙 Voice ready: Saphienne" in titles
    attention = next(message for message in world.sent if message.title.startswith("⚠️"))
    assert attention.title == "⚠️ Casting needs your decision: Elf_1_10 AutoCast Acceptance"
    assert "utterance(s) need speaker attribution review" in attention.body
    assert attention.url == f"{TAILNET}/voice-lab?project={world.project_id}&view=casting"

    sent = len(world.sent)
    for _ in range(3):
        casting.notify_ready(world.project_id)
    assert len(world.sent) == sent, "an unresolved character never becomes a repeating push"


# --------------------------------------------------------------------------
# O - delivery never gates casting
# --------------------------------------------------------------------------

def test_pushover_unavailable_never_invalidates_casting_work(world):
    world.transport.decisions = ["raise"] * 8
    prepare(world)
    fill(world, "role-saphienne", count=3)
    result = casting.notify_ready(world.project_id)
    assert result["state"] == "delivery_unknown"

    project = load_project(world.project_id)
    item = project.voice_lab.casting["roles"]["role-saphienne"]
    assert item["round"]["status"] == "awaiting_review"
    assert [c["status"] for c in item["round"]["candidates"]] == ["ready"] * 3
    view = casting.view(world.project_id)
    role = next(row for row in view["roles"] if row["role_id"] == "role-saphienne")
    assert role["round_status"] == "awaiting_review"

    casting.feedback(world.project_id, decision(world, "role-saphienne", stars=2,
                                                raw_direction="warmer"))
    project = load_project(world.project_id)
    item = project.voice_lab.casting["roles"]["role-saphienne"]
    assert item["round"]["candidates"][0]["rating"] == 2
    assert project.voice_lab.casting["replacement_queue"]["role-saphienne"]["status"] == "waiting_for_review"
    assert project.voice_lab.casting["events"], "human feedback and history survive a delivery outage"


# --------------------------------------------------------------------------
# P - a localhost review link is refused, never faked
# --------------------------------------------------------------------------

@pytest.mark.parametrize("base", ["", "http://brian-desktop.tail3e7427.ts.net:8443",
                                  "https://127.0.0.1:8443", "https://localhost:8443"])
def test_missing_mobile_review_link_is_an_explicit_configuration_failure(world, base):
    world.config["listening_base_url"] = base
    prepare(world)
    fill(world, "role-saphienne", count=1)
    result = casting.notify_ready(world.project_id)
    assert result["state"] == "blocked"
    assert "MOBILE REVIEW LINK NOT CONFIGURED" in result["reason"]
    assert world.sent == []
    assert notification(world, "role-saphienne")["state"] == "blocked"


def test_review_link_helpers_keep_the_deep_link_contract():
    assert casting.review_link("https://host.ts.net:8443/", "abc", role_id="role-a") == \
        "https://host.ts.net:8443/voice-lab?project=abc&character=role-a"
    assert casting.review_link("https://host.ts.net:8443", "abc", view="casting") == \
        "https://host.ts.net:8443/voice-lab?project=abc&view=casting"
    assert casting.review_base_url({"listening_base_url": TAILNET + "/"}) == TAILNET


# --------------------------------------------------------------------------
# a push for every finished chapter, so listening can start right away
# --------------------------------------------------------------------------

def _publish_chapters(world, numbers, *, audio="e"):
    """Record chapters as the strict production worker does, without any synthesis."""
    from dubbing_service.app.models.project import Chapter
    from dubbing_service.app.storage.segments import load_segments
    project = load_project(world.project_id)
    project.chapters = [Chapter(id="ch1", title="The Gate", index=0), Chapter(id="ch2", title="The Road", index=1)]
    save_project(project)
    rows = load_segments(world.project_id)
    for position, row in enumerate(rows):
        row.chapter_id = "ch1" if position < 2 else "ch2"
    save_segments(world.project_id, rows)
    project = load_project(world.project_id)
    state = casting._state(project)
    receipt = state["productions"].setdefault(casting.AUTOMATIC_PRODUCTION_ID, {
        "request_id": casting.AUTOMATIC_PRODUCTION_ID, "status": "queued", "job_id": "strict_chapters",
        "utterance_ids": [row.id for row in rows], "required_roles": [], "chapters": {}})
    for number in numbers:
        chapter_id = f"ch{number}"
        receipt["chapters"][chapter_id] = {
            "status": "ready", "chapter_id": chapter_id, "title": "The Gate" if number == 1 else "The Road",
            "number": number, "export_id": "cast_" + str(number) * 28, "audio_sha256": audio * 64,
            "duration_seconds": 754.0}
    save_project(project)


def _chapter_pushes(world):
    return [message for message in world.sent if message.title.startswith("🎧 New chapter ready")]


def test_each_finished_chapter_sends_its_own_push_with_a_link_to_listen(world):
    _publish_chapters(world, [1])
    casting.notify_ready(world.project_id)
    (first,) = _chapter_pushes(world)
    assert first.title == "🎧 New chapter ready: Elf_1_10 AutoCast Acceptance"
    assert first.body == "The Gate\n~13 min · 1 of 2 chapters ready.\nTap to listen now."
    assert first.url == f"{TAILNET}/voice-lab?project={world.project_id}&view=casting&chapter=ch1"
    assert first.url_title == "Listen now"
    _publish_chapters(world, [1, 2])
    casting.notify_ready(world.project_id)
    pushes = _chapter_pushes(world)
    assert [message.body.splitlines()[0] for message in pushes] == ["The Gate", "The Road"], "each names the book's own chapter"
    assert "2 of 2 chapters ready" in pushes[1].body and pushes[1].url.endswith("&chapter=ch2")


def test_a_chapter_is_announced_exactly_once_however_often_the_tick_runs(world):
    _publish_chapters(world, [1, 2])
    for _ in range(4):
        casting.notify_ready(world.project_id)
    assert len(_chapter_pushes(world)) == 2


def test_a_chapter_whose_audio_changed_is_announced_again(world):
    _publish_chapters(world, [1])
    casting.notify_ready(world.project_id)
    _publish_chapters(world, [1], audio="f")
    casting.notify_ready(world.project_id)
    assert len(_chapter_pushes(world)) == 2, "different audio is a different chapter to listen to"


def test_a_chapter_push_that_fails_keeps_retrying_until_accepted(world):
    world.transport.decisions = [{"state": "failed"}] * 4 + [{"state": "accepted"}]
    _publish_chapters(world, [1])
    for _ in range(5):
        casting.notify_ready(world.project_id)
        project = load_project(world.project_id)
        records = [row for name, row in casting._state(project).get("notifications", {}).items() if name.startswith("chapter:")]
        assert len(records) == 1
        if records[0]["state"] == "accepted":
            break
        assert records[0]["next_retry_at"] is not None
        project = load_project(world.project_id)
        casting._state(project)["notifications"][next(iter(casting._state(project)["notifications"]))]["next_retry_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        save_project(project)
    assert len(_chapter_pushes(world)) == 5 and records[0]["state"] == "accepted"


def test_a_superseded_production_never_announces_its_chapters(world):
    _publish_chapters(world, [1])
    project = load_project(world.project_id)
    casting._state(project)["productions"][casting.AUTOMATIC_PRODUCTION_ID]["status"] = "retracted"
    save_project(project)
    casting.notify_ready(world.project_id)
    assert _chapter_pushes(world) == []


def test_the_whole_book_push_reports_the_chapters(world):
    _publish_chapters(world, [1, 2])
    project = load_project(world.project_id)
    casting._state(project)["productions"][casting.AUTOMATIC_PRODUCTION_ID].update(status="succeeded")
    save_project(project)
    casting.notify_ready(world.project_id)
    book = [message for message in world.sent if "Audiobook ready" in message.title]
    assert len(book) == 1 and "2 chapters" in book[0].body
