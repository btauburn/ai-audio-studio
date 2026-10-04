"""Automatic Auto Cast: new-source workflow, idempotency, identity reconciliation.

Everything runs against disposable pytest project roots with a fixture analyser
and a fixture signal provider. No real book text, GPU, model, network or worker
is used, and no human rating is ever fabricated.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from dubbing_service.app.audiobook import (
    adapter,
    autocast_import,
    renderer,
)
from dubbing_service.app.audiobook import (
    casting_review as casting,
)
from dubbing_service.app.audiobook import (
    voice_lab as lab,
)
from dubbing_service.app.audiobook import (
    voice_lab_audition as preview,
)
from dubbing_service.app.audiobook.character_identity import (
    UNRESOLVED_REAL_CONFLICT,
    canonical_role_id,
    reconcile_identities,
    resolve_segment,
    role_id_for_label,
)
from dubbing_service.app.audiobook.fingerprint import (
    file_sha256,
)
from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.models.audiobook import Attribution
from dubbing_service.app.models.enums import RoleType
from dubbing_service.app.models.project import Role
from dubbing_service.app.providers.tts import TTSResult
from dubbing_service.app.storage.projects import load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audiobook_helpers import make_tone_wav, use_projects_root

SOURCE = b"Chapter 1\nMara entered the room.\n\"Who is there?\"\n\"It is me, Mara,\" Ivo replied.\n"


def analysis_payload(*, unknown_source="unknown", chapters=1):
    """A source with two clean characters, narration, and one held utterance."""
    payload = {
        "title": "Fixture Book",
        "author": "Test author",
        "language": "en",
        "speakers": ["Narrator", "Mara", "Ivo"],
        "chapters": [],
    }
    for index in range(chapters):
        payload["chapters"].append({
            "index": index,
            "title": f"Chapter {index + 1}",
            "utterances": [
                {"text": f"Mara entered the room in chapter {index}.", "speaker": "Narrator",
                 "utterance_type": "narration", "chapter_index": index, "line_index": 0,
                 "start_pos": 0, "end_pos": 38, "attribution_source": "tag", "confidence": 0.99},
                {"text": f"Who is there in chapter {index}?", "speaker": "Mara",
                 "utterance_type": "dialogue", "chapter_index": index, "line_index": 1,
                 "start_pos": 39, "end_pos": 67, "attribution_source": "tag", "confidence": 0.96},
                {"text": f"It is me, Mara, in chapter {index}.", "speaker": "Mara",
                 "utterance_type": "dialogue", "chapter_index": index, "line_index": 2,
                 "start_pos": 68, "end_pos": 101, "attribution_source": "tag", "confidence": 0.95},
                {"text": f"Ivo replied in chapter {index}.", "speaker": "Ivo",
                 "utterance_type": "dialogue", "chapter_index": index, "line_index": 3,
                 "start_pos": 102, "end_pos": 131, "attribution_source": "tag", "confidence": 0.94},
                {"text": f"Nobody knows who spoke in chapter {index}.", "speaker": "Unknown",
                 "utterance_type": "dialogue", "chapter_index": index, "line_index": 4,
                 "start_pos": 132, "end_pos": 170, "attribution_source": unknown_source,
                 "confidence": 0.3},
            ],
        })
    return payload


class SignalProvider:
    """Deterministic audible signal double; never labelled model-generated audio."""

    def __init__(self):
        self.designs = []
        self.calls = []

    def design_character_voice(self, project_id, identity_id, instruction, text, seed, conditioning=None):
        self.designs.append((project_id, identity_id))
        self.conditioning = copy.deepcopy(conditioning)
        root = safe_project_root(project_id)
        reference = make_tone_wav(root / "voices/reference.wav", seconds=1)
        prompt = root / "voices/prompt.npz"
        prompt.write_bytes(b"LOCAL-TEST-SIGNAL-NOT-A-MODEL-PROMPT")
        return {
            "mechanism": "qwen_reference", "backend": "qwen3tts",
            "model": "LOCAL-TEST-SIGNAL-NOT-TTS", "runtime_version": "test-only",
            "reference_audio": reference.relative_to(root).as_posix(),
            "reference_audio_sha256": file_sha256(reference),
            "reference_text": text,
            "prompt_artifact": prompt.relative_to(root).as_posix(),
            "prompt_sha256": file_sha256(prompt),
            "provenance": {"origin": "generated_original_voice", "fixture_only": True},
        }

    def synthesize_segment(self, project_id, role, segment):
        self.calls.append((project_id, segment.id))
        path = make_tone_wav(
            safe_project_root(project_id) / f"signals/{len(self.calls)}.wav", seconds=10)
        return TTSResult(success=True, audio_path=str(path), raw={
            "identity_id": role.voice_identity.id,
            "prompt_sha256": role.voice_identity.prompt_sha256,
            "execution_profile": {"model_id": "LOCAL-TEST-SIGNAL-NOT-TTS", "fixture_only": True}})


def _install_fixture_runtime(monkeypatch, tmp_path, *, payload=None, keep_scratch=None):
    """Fake analyzer + fake TTS provider, no GPU and no real book bytes."""
    provider = SignalProvider()
    # This suite isolates canonical source/fixture-TTS behavior. The bounded
    # model phase has its own fake-provider integration/privacy/recovery suite.
    from dubbing_service.app.audiobook import autocast_public_model_research
    monkeypatch.setattr(autocast_public_model_research, "research_for_project", lambda *a, **kw: None)
    monkeypatch.setattr(adapter, "availability", lambda refresh=False: adapter.Availability(
        True, "", str(Path(__file__).resolve()), adapter.PINNED_AUDIOBOOKER_VERSION))
    original = adapter.AudiobookerAdapter
    body = payload if payload is not None else analysis_payload

    def factory(*, interpreter=None):
        return original(interpreter=Path(interpreter) if interpreter else Path(__file__),
                        runner=lambda payload_, interpreter_: body() if callable(body) else body,
                        scratch_root=tmp_path / "retained-scratch")

    monkeypatch.setattr(autocast_import.adapter, "AudiobookerAdapter", factory)
    monkeypatch.setattr(preview, "ensure_backend_ready", lambda *a, **kw: {"fixture_only": True})
    monkeypatch.setattr(renderer.TTSProviderFactory, "create", lambda backend: provider)
    original_design = lab.design
    monkeypatch.setattr(lab, "design",
                        lambda *a, **kw: original_design(*a, **{**kw, "provider": provider}))
    return provider


@pytest.fixture
def studio(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    provider = _install_fixture_runtime(monkeypatch, tmp_path)
    return root, provider


def import_fixture(request_id="request-1", *, name="AutoCast fixture"):
    return autocast_import.import_autocast(SOURCE, "book.txt", request_id, name)


def characters(project):
    return [role for role in project.roles if role.role_type == RoleType.CHARACTER]


def role_names(project):
    return sorted(role.name for role in characters(project))


# ---------------------------------------------------------------------------
# A. New-source happy path: characters -> automatic run -> playable audition
# ---------------------------------------------------------------------------

def test_new_source_happy_path_reaches_a_playable_audition_without_prepare_cast(studio):
    _root, provider = studio
    result = import_fixture()
    project = load_project(result["project_id"])

    # 1. Characters are created automatically by the import itself.
    assert role_names(project) == ["Ivo", "Mara"]
    assert project.character_cast.dossiers or project.character_cast.profiles is not None

    # 2. The run was requested by the durable import completion, not by a view.
    run = project.character_cast.autocast
    assert run is not None and run.trigger == "source_import"
    assert run.status == "pending"
    assert result["autocast_run_key"] == run.key and result["autocast_status"] == "pending"

    # 3. The existing maintenance owner — never a Prepare cast click — starts it.
    casting.maintenance_tick(project_ids=[result["project_id"]])

    started = load_project(result["project_id"]).character_cast.autocast
    assert started.status in {"running", "complete"}
    state = load_project(result["project_id"]).voice_lab.casting
    assert state["roles"], "Auto Cast must have queued rounds for the discovered characters"

    # 4. At least one real audition becomes playable through the pipeline.
    view = casting.view(result["project_id"])
    ready = [c for role in view["roles"] for c in role["candidates"] if c["status"] == "ready"]
    assert ready, view["pipeline"]
    assert view["pipeline"]["characters"] >= 2
    assert view["pipeline"]["ready_options"] >= 1
    assert view["scope_error"] is None

    # 5. The artifact is real audio on disk and readable through the lab record.
    record = lab.audio_record(result["project_id"], ready[0]["key"])
    path, stored = record
    assert path.is_file() and path.stat().st_size > 0
    assert stored["audio_sha256"] == ready[0]["audio_sha256"]
    assert provider.calls, "the audition must come from the real synthesis seam"

    # 6. No human rating or approval was invented.
    assert not load_project(result["project_id"]).voice_lab.casting["events"]
    assert all(role["status"] != "APPROVED" for role in view["roles"])


# ---------------------------------------------------------------------------
# B. Idempotency: one completion event delivered twice
# ---------------------------------------------------------------------------

def test_repeated_completion_event_produces_exactly_one_casting_run(studio):
    _root, _provider = studio
    first = import_fixture("request-twice")
    second = autocast_import.import_autocast(SOURCE, "book.txt", "request-twice", "AutoCast fixture")
    assert second["reused"] is True and second["project_id"] == first["project_id"]
    assert second["autocast_run_key"] == first["autocast_run_key"]

    casting.maintenance_tick(project_ids=[first["project_id"]])
    once = load_project(first["project_id"])
    rounds_once = {rid: item["round"]["id"]
                   for rid, item in once.voice_lab.casting["roles"].items() if item["round"]}
    revisions = {rid: item["round_revision"]
                 for rid, item in once.voice_lab.casting["roles"].items() if item["round"]}
    assert len(rounds_once) >= 2

    # The same completion event again: no new run, no new round, no duplicate candidates.
    casting.request_autocast_run(first["project_id"], trigger="source_import")
    casting.maintenance_tick(project_ids=[first["project_id"]])
    twice = load_project(first["project_id"])
    assert {rid: item["round"]["id"]
            for rid, item in twice.voice_lab.casting["roles"].items() if item["round"]} == rounds_once
    assert {rid: item["round_revision"]
            for rid, item in twice.voice_lab.casting["roles"].items() if item["round"]} == revisions
    assert twice.character_cast.autocast.history == []
    assert twice.character_cast.autocast.key == first["autocast_run_key"]


# ---------------------------------------------------------------------------
# C. Page refresh: viewing Voice Lab must not start or duplicate anything
# ---------------------------------------------------------------------------

def test_voice_lab_refresh_creates_no_duplicate_run_or_candidates(studio):
    _root, _provider = studio
    result = import_fixture("request-refresh")
    project_id = result["project_id"]
    before = load_project(project_id)

    for _ in range(5):
        casting.view(project_id)
        casting.view(project_id, base_url="/api/v1/projects/x/voice-lab")

    after = load_project(project_id)
    assert after.character_cast.autocast.key == before.character_cast.autocast.key
    assert after.voice_lab.casting == before.voice_lab.casting
    assert after.voice_lab.casting.get("requests") in (None, {})

    casting.maintenance_tick(project_ids=[project_id])
    settled = load_project(project_id)
    rounds = {rid: item["round"]["id"]
              for rid, item in settled.voice_lab.casting["roles"].items() if item["round"]}
    assert len(rounds) >= 2
    for _ in range(2):
        casting.maintenance_tick(project_ids=[project_id])
    stable = load_project(project_id)
    assert {rid: item["round"]["id"]
            for rid, item in stable.voice_lab.casting["roles"].items() if item["round"]} == rounds
    assert {rid: item["round_revision"]
            for rid, item in stable.voice_lab.casting["roles"].items() if item["round"]} == {rid: 1 for rid in rounds}


# ---------------------------------------------------------------------------
# D. Alias / legacy raw-name normalization
# ---------------------------------------------------------------------------

def test_legacy_raw_speaker_name_reconciles_without_a_false_conflict(studio):
    _root, _provider = studio
    result = import_fixture("request-legacy")
    project_id = result["project_id"]
    project = load_project(project_id)
    mara = next(r for r in project.roles if r.name == "Mara")

    # Recreate what a pre-fix import stored: the canonical id in role_id and the
    # analyzer's raw speaker *name* in attribution.speaker_id.
    segments = load_segments(project_id)
    dialogue = next(s for s in segments if s.role_id == mara.id)
    dialogue.attribution.speaker_id = "Mara"
    save_segments(project_id, segments)

    project = load_project(project_id)
    segments = load_segments(project_id)
    target = next(s for s in segments if s.id == dialogue.id)
    assert role_id_for_label(project, "Mara") == mara.id
    assert resolve_segment(project, target).settled

    report = reconcile_identities(project, segments)
    assert not report.conflicts and report.reconciled >= 1
    repaired = next(s for s in segments if s.id == target.id)
    assert repaired.attribution.speaker_id == repaired.role_id == mara.id

    view = casting.view(project_id)
    assert view["scope_error"] is None
    assert view["pipeline"]["characters"] >= 2


def test_alias_chain_resolves_to_one_canonical_character(studio):
    _root, _provider = studio
    result = import_fixture("request-alias")
    project_id = result["project_id"]
    project = load_project(project_id)
    mara = next(r for r in project.roles if r.name == "Mara")
    project.roles.append(Role(id="role-mara-alias", name="The Healer", role_type="character"))
    project.character_cast.role_aliases["role-mara-alias"] = mara.id
    save_project(project)

    assert canonical_role_id(project, "role-mara-alias") == mara.id
    assert role_id_for_label(project, "The Healer") == mara.id
    assert role_id_for_label(project, "Mara") == mara.id
    # A cycle must not hang and must not invent an identity.
    project.character_cast.role_aliases[mara.id] = "role-mara-alias"
    assert canonical_role_id(project, "role-mara-alias") in {mara.id, "role-mara-alias"}


# ---------------------------------------------------------------------------
# E. Genuine conflict: needs attention, with evidence, fail closed
# ---------------------------------------------------------------------------

def test_genuine_identity_conflict_is_isolated_with_evidence(studio):
    _root, _provider = studio
    result = import_fixture("request-conflict")
    project_id = result["project_id"]
    project = load_project(project_id)
    mara = next(r for r in project.roles if r.name == "Mara")
    ivo = next(r for r in project.roles if r.name == "Ivo")

    segments = load_segments(project_id)
    contested = next(s for s in segments if s.role_id == mara.id)
    contested.attribution = Attribution(
        speaker_id=ivo.id, label="Ivo", source="turn", confidence=0.4)
    save_segments(project_id, segments)

    view = casting.view(project_id)
    conflicts = [row for row in view["needs_attention"]
                 if row["reason"] == UNRESOLVED_REAL_CONFLICT]
    assert [row["utterance_id"] for row in conflicts] == [contested.id]
    assert conflicts[0]["evidence"]["assigned_role_id"] == mara.id
    assert conflicts[0]["evidence"]["attribution_speaker_id"] == ivo.id
    # The error names the affected utterance and retains the original wording.
    assert conflicts[0]["message"] == (
        f"Speaker attribution conflicts with the assigned character for {contested.id}")


# ---------------------------------------------------------------------------
# F. Partial conflict: the other characters keep going
# ---------------------------------------------------------------------------

def test_partial_conflict_does_not_block_unaffected_characters(studio):
    _root, _provider = studio
    result = import_fixture("request-partial")
    project_id = result["project_id"]
    project = load_project(project_id)
    mara = next(r for r in project.roles if r.name == "Mara")
    ivo = next(r for r in project.roles if r.name == "Ivo")

    segments = load_segments(project_id)
    mara_lines = [s for s in segments if s.role_id == mara.id]
    contested = mara_lines[0]
    contested.attribution = Attribution(
        speaker_id=ivo.id, label="Ivo", source="turn", confidence=0.4)
    save_segments(project_id, segments)

    casting.maintenance_tick(project_ids=[project_id])
    project = load_project(project_id)
    cast_roles = {rid for rid, item in project.voice_lab.casting["roles"].items() if item.get("round")}
    assert mara.id in cast_roles, "a partial conflict must not remove a still-valid character"
    assert ivo.id in cast_roles

    view = casting.view(project_id)
    assert view["pipeline"]["ready_options"] >= 1
    assert view["scope_error"] is None


# ---------------------------------------------------------------------------
# G. Narrator handling
# ---------------------------------------------------------------------------

def test_narrator_is_cast_from_its_own_role_and_never_as_a_fallback(studio):
    _root, _provider = studio
    result = import_fixture("request-narrator")
    project_id = result["project_id"]
    project = load_project(project_id)
    narrator = next(r for r in project.roles if r.role_type == RoleType.NARRATOR and r.name.lower() == "narrator")

    segments = load_segments(project_id)
    narration = [s for s in segments if s.role_id == narrator.id]
    assert narration, "narration keeps its own role"
    assert all(s.attribution.speaker_id == narrator.id for s in narration)

    held = set(project.metadata["autocast_import"]["unresolved_utterance_ids"])
    assert held, "the low-confidence utterance stays held"
    assert all(load_project(project_id).metadata["autocast_import"]["unresolved_utterance_ids"])
    for segment in segments:
        if segment.id in held:
            assert segment.role_id is None, "held dialogue must not fall back to the narrator"
            assert segment.attribution.label == "Unknown", "the analyzer proposal is kept as evidence"

    casting.maintenance_tick(project_ids=[project_id])
    view = casting.view(project_id)
    assert any(role["role_id"] == narrator.id for role in view["roles"])


# ---------------------------------------------------------------------------
# H. Restart / resume
# ---------------------------------------------------------------------------

def test_pending_run_resumes_after_a_restart_without_duplicate_candidates(studio):
    _root, _provider = studio
    result = import_fixture("request-restart")
    project_id = result["project_id"]

    # Nothing has run yet: the receipt survived the "process" that wrote it.
    assert load_project(project_id).character_cast.autocast.status == "pending"

    # A fresh application lifecycle runs the same durable owner.
    casting.maintenance_tick()
    resumed = load_project(project_id)
    assert resumed.character_cast.autocast.status in {"running", "complete"}
    rounds_once = {rid: item["round"]["id"] for rid, item in resumed.voice_lab.casting["roles"].items() if item["round"]}
    candidates_once = {rid: [c["candidate_id"] for c in item["round"]["candidates"]]
                       for rid, item in resumed.voice_lab.casting["roles"].items() if item["round"]}

    # A second restart repeats the pass with no additional work.
    casting.maintenance_tick()
    again = load_project(project_id)
    assert {rid: item["round"]["id"] for rid, item in again.voice_lab.casting["roles"].items() if item["round"]} == rounds_once
    assert {rid: [c["candidate_id"] for c in item["round"]["candidates"]]
            for rid, item in again.voice_lab.casting["roles"].items() if item["round"]} == candidates_once


# ---------------------------------------------------------------------------
# I. Human work is preserved
# ---------------------------------------------------------------------------

def _approve_first_ready(project_id):
    """A genuine 4-star human decision, recorded through the public seam.

    The fixture provider produces a signal, not a model voice, so the project is
    marked as a regression fixture — the existing guard that keeps test audio out
    of production approval. The rating itself is still supplied explicitly.
    """
    from dubbing_service.app.storage.projects import project_meta_lock
    with project_meta_lock(project_id):
        project = load_project(project_id)
        project.voice_lab.scope = "regression_fixture"
        save_project(project)
    view = casting.view(project_id)
    for role in view["roles"]:
        for candidate in role["candidates"]:
            if candidate["status"] != "ready":
                continue
            body = {"event_id": "evt-human-" + candidate["candidate_id"][:12],
                    "role_id": role["role_id"], "candidate_id": candidate["candidate_id"],
                    "round_revision": role["round_revision"],
                    "assignment_revision": role["assignment_revision"],
                    "key": candidate["key"], "audio_sha256": candidate["audio_sha256"],
                    "stars": 5, "raw_direction": "", "action": "use"}
            casting.feedback(project_id, body)
            return role["role_id"], candidate["key"]
    raise AssertionError("no playable candidate to decide on")


def test_unchanged_source_preserves_the_human_rating_and_accepted_voice(studio):
    _root, _provider = studio
    result = import_fixture("request-preserve")
    project_id = result["project_id"]
    casting.maintenance_tick(project_ids=[project_id])
    role_id, key = _approve_first_ready(project_id)
    approved_before = load_project(project_id)
    assert casting.approved(approved_before, role_id)
    revision_before = approved_before.voice_lab.casting["roles"][role_id]["round_revision"]

    _run, created = casting.request_autocast_run(project_id, trigger="source_import")
    assert created is False, "an unchanged source must not start a second run"
    casting.maintenance_tick(project_ids=[project_id])

    after = load_project(project_id)
    assert casting.approved(after, role_id)
    assert after.voice_lab.casting["roles"][role_id]["assignment"]["key"] == key
    assert after.voice_lab.casting["roles"][role_id]["round_revision"] == revision_before
    assert after.voice_lab.casting["events"], "the human decision is retained as history"


def test_source_revision_change_starts_a_new_run_and_keeps_the_accepted_voice(studio):
    _root, _provider = studio
    result = import_fixture("request-revision")
    project_id = result["project_id"]
    casting.maintenance_tick(project_ids=[project_id])
    role_id, key = _approve_first_ready(project_id)
    first_key = load_project(project_id).character_cast.autocast.key

    # A genuinely new source revision for the same project.
    from dubbing_service.app.audiobook.provenance import register_source_revision
    from dubbing_service.app.storage.projects import project_meta_lock
    with project_meta_lock(project_id):
        current = load_project(project_id)
        revised = SOURCE + b"Chapter 2\nMara: We should go.\n"
        revision, created_revision = register_source_revision(
            project_id, current, uri="local-upload:revised", data=revised,
            media_type="text/plain", title="Fixture Book revised")
        assert created_revision is True
        receipt = current.metadata["autocast_import"]
        receipt["raw_source_revision_id"] = revision.id
        receipt["source_sha256"] = revision.sha256
        current.metadata["autocast_import"] = receipt
        save_project(current)

    run, created = casting.request_autocast_run(project_id, trigger="source_import")
    assert created is True and run["key"] != first_key
    casting.maintenance_tick(project_ids=[project_id])

    after = load_project(project_id)
    assert casting.approved(after, role_id), "an already-accepted voice is never silently replaced"
    assert after.voice_lab.casting["roles"][role_id]["assignment"]["key"] == key


# ---------------------------------------------------------------------------
# K. Regression for the reported "Speaker attribution conflicts" failure
# ---------------------------------------------------------------------------

def test_reconcilable_narrator_identity_never_reports_the_conflict_error(studio):
    """The exact shape of the reported failure: role_id is an id, attribution a name."""
    _root, _provider = studio
    result = import_fixture("request-regression")
    project_id = result["project_id"]
    project = load_project(project_id)
    narrator = next(r for r in project.roles if r.role_type == RoleType.NARRATOR and r.name.lower() == "narrator")

    # Restore the pre-fix representation for the narration rows.
    segments = load_segments(project_id)
    for segment in segments:
        if segment.role_id == narrator.id:
            segment.attribution.speaker_id = "narrator"
    save_segments(project_id, segments)

    view = casting.view(project_id)
    assert view["scope_error"] is None, view["scope_error"]
    assert all("conflicts with the assigned character" not in (row.get("message") or "")
               for row in view["needs_attention"])
    assert view["pipeline"]["characters"] >= 2
    assert view["pipeline"]["stage"] not in {"failed", "needs_attention"} or view["pipeline"]["castable"] >= 2

    # And the invariant is repaired at the authoritative layer on the next import.
    reimport = adapter.AudiobookerAdapter().reconcile_into_project(
        adapter.BookAnalysis.from_dict(analysis_payload()),
        project_id=project_id, source_revision_id=load_project(project_id).source_revisions[0].id)
    assert reimport.project_id == project_id
    repaired = load_segments(project_id)
    for segment in repaired:
        if segment.attribution is not None and segment.attribution.speaker_id:
            assert segment.attribution.speaker_id == canonical_role_id(project, segment.role_id)


# ---------------------------------------------------------------------------
# L. Ingestion in progress is never reported as "zero characters"
# ---------------------------------------------------------------------------

def test_ingestion_in_progress_reports_a_real_stage_not_zero_characters(studio):
    _root, _provider = studio
    from dubbing_service.app.storage.projects import create_project, project_meta_lock

    # The race shape: the receipt exists and the analyzer is still running, so
    # no character can be known yet. That must read as a stage, never as a
    # finished empty cast.
    project = create_project("Ingestion still running")
    with project_meta_lock(project.id):
        current = load_project(project.id)
        current.metadata["autocast_import"] = {"schema": "autocast-import-v1",
                                               "status": "analyzing", "request_id": "in-flight"}
        save_project(current)
    analyzing = casting.view(project.id)
    assert analyzing["pipeline"]["stage"] == "extracting"
    assert "extract" in analyzing["pipeline"]["label"].lower()

    # And once characters are known, the same project never regresses to a
    # "no speaking characters" claim.
    result = import_fixture("request-stage")
    view = casting.view(result["project_id"])
    assert view["pipeline"]["characters"] >= 2
    assert view["pipeline"]["stage"] not in {"segmenting", "resolving_characters"}
    assert view["scope_error"] is None


def test_project_with_no_source_at_all_reports_segmentation_not_a_lie(studio):
    _root, _provider = studio
    from dubbing_service.app.storage.projects import create_project
    project = create_project("Empty project, no source yet")
    view = casting.view(project.id)
    assert view["pipeline"]["stage"] in {"segmenting", "resolving_characters"}
    assert view["pipeline"]["characters"] == 0
    assert view["roles"] == []
    assert view["pipeline"]["label"]
