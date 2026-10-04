"""Famous-performer anchoring, non-blocking low ratings, preferred-voice memory.

Every case runs against a disposable projects root and a disposable global voice
library, with a fixture signal provider instead of a model. No real book text,
GPU, network or human rating is used, and nothing is written to the machine's
real voice library.
"""
from __future__ import annotations

import copy

import pytest

from dubbing_service.app.audiobook import (
    casting_anchors,
    casting_inference,
    preferred_voices,
)
from dubbing_service.app.audiobook import casting_review as casting
from dubbing_service.app.audiobook import voice_lab as lab
from dubbing_service.app.audiobook.casting_review import round_is_stale
from dubbing_service.app.audiobook.fingerprint import (
    file_sha256,
    voice_revision_for_role,
)
from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.models.audiobook import Attribution
from dubbing_service.app.models.character import (
    AUTOCAST_PIPELINE_VERSION,
    CastingDecision,
    CastingProfile,
    CharacterVoiceIdentity,
)
from dubbing_service.app.models.enums import RoleType
from dubbing_service.app.models.project import Role, Segment
from dubbing_service.app.providers.tts import TTSResult
from dubbing_service.app.storage import voices as library
from dubbing_service.app.storage.projects import (
    create_project,
    load_project,
    save_project,
)
from dubbing_service.app.storage.segments import save_segments
from tests.audiobook_helpers import make_tone_wav, use_projects_root

#: The requirement's own example character: an older British male narrator.
#: Evidence ids are present because a real profile decision always cites the
#: dossier fact it was interpreted from.
ELDER_TRAITS = (
    CastingDecision(dimension="presentation", value="masculine", reason="established gender",
                    level="SUPPORTED", evidence_ids=["fact-gender"]),
    CastingDecision(dimension="perceived_vocal_age", value="older adult", reason="apparent age",
                    level="INFERRED", evidence_ids=["fact-age"]),
    CastingDecision(dimension="accent", value="British", reason="accent evidence", level="SUPPORTED",
                    evidence_ids=["fact-accent"]),
    CastingDecision(dimension="warmth", value="warm", reason="temperament evidence", level="INFERRED",
                    evidence_ids=["fact-temperament"]),
    CastingDecision(dimension="cadence", value="deliberate", reason="cadence evidence",
                    level="INFERRED", evidence_ids=["fact-cadence"]),
    # A neutral editorial placeholder: it must not become an anchor rationale.
    CastingDecision(dimension="pitch", value="mid", reason="Neutral register pending audition",
                    weight=0.5),
)

#: A character that shares no vocal dimension with the elder narrator.
YOUNG_TRAITS = (
    CastingDecision(dimension="presentation", value="feminine", reason="established gender",
                    level="SUPPORTED", evidence_ids=["fact-gender"]),
    CastingDecision(dimension="perceived_vocal_age", value="youthful", reason="apparent age",
                    level="INFERRED", evidence_ids=["fact-age"]),
    CastingDecision(dimension="accent", value="American", reason="accent evidence", level="SUPPORTED",
                    evidence_ids=["fact-accent"]),
    CastingDecision(dimension="intensity", value="high energy", reason="temperament evidence",
                    level="INFERRED", evidence_ids=["fact-temperament"]),
)


class SignalProvider:
    """Deterministic audible signal double; never labelled model-generated audio.

    It stands in for the real worker's design -> acoustics -> prepare sequence,
    including the acoustic shaping pass: the reference it hands back is shaped
    onto the conditioning it received, so a test can verify that the candidate's
    acoustic target reached the generation request and moved the actual audio.
    """

    def __init__(self):
        self.designs = []

    def design_character_voice(self, project_id, identity_id, instruction, text, seed, conditioning=None):
        self.designs.append({"identity_id": identity_id, "instruction": instruction,
                             "conditioning": copy.deepcopy(conditioning)})
        root = safe_project_root(project_id)
        source = make_tone_wav(root / f"voices/{identity_id}-reference.wav", seconds=0.8,
                               frequency=200 + (len(self.designs) * 17) % 120)
        reference = source
        acoustics = None
        if conditioning:
            from service import voice_acoustics
            before = voice_acoustics.measure(source)
            plan = voice_acoustics.plan_shaping(conditioning["target_profile"], before)
            shaped = root / f"voices/{identity_id}-conditioned.wav"
            voice_acoustics.shape(source, shaped, semitones=plan["semitones"], rate=plan["rate"])
            after = voice_acoustics.measure(shaped)
            acoustics = {"schema": "identity-acoustics-v1", "identity_id": identity_id,
                         "reference_audio": shaped.relative_to(root).as_posix(),
                         "reference_audio_sha256": file_sha256(shaped),
                         "conditioning": copy.deepcopy(conditioning), "shaping_plan": plan,
                         "profile_before": before, "profile_after": after,
                         "achieved": True}
            reference = shaped
        prompt = root / f"voices/{identity_id}-prompt.npz"
        prompt.write_bytes(b"LOCAL-TEST-SIGNAL-NOT-A-MODEL-PROMPT")
        return {
            "mechanism": "qwen_reference", "backend": "qwen3tts",
            "model": "LOCAL-TEST-SIGNAL-NOT-TTS", "runtime_version": "test-only",
            "reference_audio": reference.relative_to(root).as_posix(),
            "reference_audio_sha256": file_sha256(reference),
            "reference_text": text,
            "prompt_artifact": prompt.relative_to(root).as_posix(),
            "prompt_sha256": file_sha256(prompt),
            "acoustics": acoustics,
            "provenance": {"origin": "generated_original_voice", "fixture_only": False},
        }

    def synthesize_segment(self, project_id, role, segment):
        path = make_tone_wav(safe_project_root(project_id) / f"signals/{segment.id}.wav", seconds=0.8)
        return TTSResult(success=True, audio_path=str(path), raw={
            "identity_id": role.voice_identity.id,
            "prompt_sha256": role.voice_identity.prompt_sha256,
            "execution_profile": {"model_id": "LOCAL-TEST-SIGNAL-NOT-TTS", "fixture_only": False}})


@pytest.fixture
def studio(monkeypatch, tmp_path):
    """Disposable projects root plus a disposable canonical voice library."""
    root = use_projects_root(monkeypatch, tmp_path)
    voices = tmp_path / "voice-library"
    voices.mkdir()
    monkeypatch.setattr(library, "voices_audio_dir", lambda: voices)
    monkeypatch.setattr(library, "_voices_file_path", lambda: voices.parent / "voices.json")
    return root


def build(studio, name, *, traits=ELDER_TRAITS, role_id="role-lead", role_name="Narrator"):
    project = create_project(name)
    project.roles = [Role(id=role_id, name=role_name, role_type="character", tts_backend="qwen3tts")]
    project.character_cast.profiles[role_id] = CastingProfile(
        role_id=role_id, revision="fixture-profile", dossier_revision="fixture-dossier",
        design_instruction="Create an original, consistent character voice.", decisions=list(traits))
    save_project(project)
    save_segments(project.id, [Segment(id="line-1", index=0, text="A fixture line.",
                                       role_id=role_id)])
    return project.id


def make_ready(project_id, role_id, *, index=0, voice_id=None, acoustics=None):
    """Give one queued candidate genuinely playable, publishable audio.

    Keys are formed exactly as the production record reader forms them, so the
    voice library validates them the way it validates real voices.
    """
    project = load_project(project_id)
    role = next(item for item in project.roles if item.id == role_id)
    round_ = project.voice_lab.casting["roles"][role_id]["round"]
    candidate = round_["candidates"][index]
    voice_id = voice_id or f"voice_{project_id}_{role_id}_{round_['revision']}_{index}"
    root = safe_project_root(project_id)
    # Each candidate is its own voice revision, exactly as a distinct generated
    # identity is in production; a shared revision would misrepresent casting.
    identity = {
        "id": voice_id, "role_id": role_id, "status": "LOCKED", "mechanism": "qwen_reference",
        "dossier_revision": "fixture-dossier", "casting_profile_revision": "fixture-profile",
        "design_instruction": "Synthetic fixture voice.",
        "reference_audio": None, "reference_audio_sha256": None,
        "reference_text": "Synthetic fixture reference.",
        "prompt_artifact": None, "prompt_sha256": None,
        "backend": "qwen3tts", "model": "SYNTHETIC-FIXTURE-ONLY", "model_revision": "fixture",
        "runtime_version": "fixture",
        "provenance": {"origin": "generated_original_voice", "fixture_only": False},
    }
    probe = role.model_copy(update={"voice_identity": CharacterVoiceIdentity.model_validate(identity)})
    version = voice_revision_for_role(probe)
    key = "v_" + lab.digest([voice_id, version])[:28]
    audio = make_tone_wav(root / f"voices/{key}.wav", seconds=0.8, frequency=190 + index * 13)
    reference = make_tone_wav(root / f"voices/{key}-ref.wav", seconds=0.8, frequency=250 + index * 13)
    prompt = root / f"voices/{key}.npz"
    prompt.write_bytes(b"SYNTHETIC-FIXTURE-PROMPT-CONDITIONING")
    identity["reference_audio"] = reference.relative_to(root).as_posix()
    identity["reference_audio_sha256"] = file_sha256(reference)
    identity["prompt_artifact"] = prompt.relative_to(root).as_posix()
    identity["prompt_sha256"] = file_sha256(prompt)
    inspiration = None
    if candidate.get("inspiration"):
        inspiration = {
            "palette": "catalogue-performer-anchors-v1",
            "character": candidate["inspiration"]["performance"],
            "performer": candidate["inspiration"]["performer"],
            "mode": "archetype", "traits": {}, "reasons": [candidate.get("match_rationale") or ""],
            "score": 0, "source": "; ".join(candidate["inspiration"].get("source_urls") or []),
            "original_voice_only": True,
        }
        identity["inspiration"] = inspiration
    candidate.update(status="ready", key=key, audio_sha256=file_sha256(audio),
                     identity_id=f"effective-{voice_id}", duration_seconds=0.8, technical_qa="passed",
                     sections=[], voice_version=version, ready_at=lab.now())
    project.voice_lab.voices[key] = {
        "generation_language": role.qwen3_language,
        "key": key, "voice_id": voice_id, "version": version,
        "name": f"Synthetic {voice_id}", "kind": "original_inspired", "synthetic": True,
        "role_id": role_id, "role_name": role.name, "tags": [],
        "inspiration": ({"performer": inspiration["performer"], "performance": inspiration["character"],
                         "source_urls": [], "rationale": candidate.get("match_rationale")}
                        if inspiration else None),
        "anchor": copy.deepcopy(candidate.get("anchor")),
        "audio_file": audio.relative_to(root).as_posix(), "audio_sha256": file_sha256(audio),
        "identity": identity, "backend": "qwen3tts", "technical_qa": "passed",
        "take_id": f"take-{voice_id}", "sections": [], "duration_seconds": 0.8,
        "acoustics": acoustics}
    save_project(project)
    return load_project(project_id).voice_lab.casting["roles"][role_id]["round"]["candidates"][index]


def complete_round(project_id, role_id):
    """The terminal projection the round worker writes once auditions exist."""
    project = load_project(project_id)
    project.voice_lab.casting["roles"][role_id]["round"].update(
        status="awaiting_review", completed_at=lab.now())
    save_project(project)


def prepare_round(project_id, role_id, *, request_id="prepare-1", count=3):
    casting.prepare(project_id, request_id=request_id)
    for index in range(count):
        make_ready(project_id, role_id, index=index)
    complete_round(project_id, role_id)
    return load_project(project_id).voice_lab.casting["roles"][role_id]["round"]


def body_for(project_id, role_id, index, *, stars, action="save", direction="", event_id=None):
    item = load_project(project_id).voice_lab.casting["roles"][role_id]
    candidate = item["round"]["candidates"][index]
    return {"event_id": event_id or f"evt-{role_id}-{index}-{stars}-{action}",
            "role_id": role_id, "candidate_id": candidate["candidate_id"],
            "round_revision": item["round"]["revision"],
            "assignment_revision": item["assignment_revision"],
            "key": candidate["key"], "audio_sha256": candidate["audio_sha256"],
            "action": action, "stars": stars, "raw_direction": direction}


def rate(project_id, role_id, index, stars, *, direction="", action="save", event_id=None):
    return casting.feedback(project_id, body_for(project_id, role_id, index, stars=stars,
                                                 direction=direction, action=action,
                                                 event_id=event_id))


def dispatch(project_id):
    """The real maintenance owner, with its worker dispatch stubbed out.

    A stubbed dispatch leaves the queued job marked active, and the owner skips a
    project whose casting job is active — so the next tick must release it, exactly
    as a finished worker would.
    """
    from dubbing_service.app.storage.jobs import load_jobs, mark_job_active
    for job in load_jobs(project_id):
        if job.id.startswith(("cast_", "strict_", "audition_")):
            mark_job_active(project_id, job.id, False)
    casting.maintenance_tick(project_ids=[project_id], schedule=lambda *args: None)


def casting_role(project_id, role_id="role-lead"):
    return next(row for row in casting.view(project_id)["roles"] if row["role_id"] == role_id)


# --------------------------------------------------------------------------
# 1. Every normal generated candidate records a reference and a rationale
# --------------------------------------------------------------------------

def test_generated_candidates_are_anchored_to_a_specific_performance(studio):
    project_id = build(studio, "Anchored casting")
    prepare_round(project_id, "role-lead")
    generated = [c for c in casting_role(project_id)["candidates"] if not c["prior_match"]]
    assert len(generated) == 3
    for candidate in generated:
        assert candidate["provenance"] == "generated"
        anchor = candidate["anchor"]
        assert anchor["anchored"] is True
        assert anchor["performer"] and anchor["performance"]
        assert anchor["matched_traits"] and anchor["rationale"]
        assert candidate["inspiration"]["performer"] == anchor["performer"]
        assert candidate["match_rationale"] == candidate["reference"]["rationale"]
        assert anchor["performer"] in candidate["match_rationale"]
        assert candidate["conditioning"]["reference"]["performer"] == anchor["performer"]
    # The requirement's own example: an older British male narrator anchors on the
    # documentary narrator rather than on an unrelated catalogue row.
    assert generated[0]["anchor"]["performer"] == "David Attenborough"
    assert "Planet Earth" in generated[0]["anchor"]["performance"]
    assert {"british", "male", "mature"}.issubset(set(generated[0]["anchor"]["matched_traits"]))
    assert len({c["anchor"]["performer"] for c in generated}) == len(generated), \
        "the options in one round must be genuinely different performances"
    # The stored proposal names its reference and still demands an original voice.
    matches = [m for m in load_project(project_id).voice_lab.shortlists["role-lead"]["matches"]
               if m["candidate_id"] == generated[0]["candidate_id"]]
    assert "David Attenborough, Planet Earth (2006), UK narration" in matches[0]["design_instruction"]
    assert "do not reproduce the reference recording" in matches[0]["design_instruction"]


def test_a_heard_generated_voice_records_its_reference_and_rationale(studio):
    project_id = build(studio, "Anchored design")
    round_ = prepare_round(project_id, "role-lead")
    project = load_project(project_id)
    role = next(item for item in project.roles if item.id == "role-lead")
    expected = role.voice_identity.id if role.voice_identity else voice_revision_for_role(role)
    candidate = round_["candidates"][0]
    designed = lab.design(project_id, "role-lead", candidate["candidate_id"], expected,
                          "design-anchored", provider=SignalProvider())
    record = load_project(project_id).voice_lab.voices[designed["key"]]
    assert record["inspiration"]["performer"] == candidate["inspiration"]["performer"]
    assert record["inspiration"]["performance"] == candidate["inspiration"]["performance"]
    assert record["inspiration"]["rationale"] == candidate["match_rationale"]
    identity = record["identity"]
    assert identity["inspiration"]["performer"] == candidate["inspiration"]["performer"]
    assert identity["inspiration"]["mode"] == "archetype"
    assert identity["inspiration"]["original_voice_only"] is True
    assert identity["inspiration"]["reasons"][0] == candidate["match_rationale"]
    assert "David Attenborough" in identity["design_instruction"]


def test_sparse_character_evidence_produces_labelled_casting_inference(studio):
    """A character the book barely describes is cast, not left unanchored.

    Sparse evidence used to produce an unanchored candidate, which is the generic
    "AI voice" this layer exists to remove. It now produces several deliberately
    differentiated cast concepts, every one of them anchored on a real
    performance, with the interpolated dimensions labelled as casting inference
    and never presented as something the book said.
    """
    project_id = build(studio, "Sparse casting", traits=())
    prepare_round(project_id, "role-lead")
    project = load_project(project_id)
    profile = project.character_cast.profiles["role-lead"]
    assert casting_anchors.vocal_traits(profile, None) == {}
    candidates = casting_role(project_id)["candidates"]
    assert len(candidates) >= 3
    concepts = set()
    for candidate in candidates:
        assert candidate["anchor"]["anchored"] is True
        assert candidate["anchor"]["performer"] and candidate["anchor"]["performance"]
        assert candidate["reference"] and candidate["conditioning"]
        # Nothing the book did not say may appear as a book fact.
        assert candidate["supported_traits"] == []
        assert candidate["inferred_traits"], "an interpolated dimension must be labelled"
        assert all(row["level"] == casting_inference.INFERENCE_LEVEL
                   for row in candidate["inferred_traits"])
        assert all(row["evidence_ids"] == [] for row in candidate["inferred_traits"])
        assert all(row["basis"].startswith("Casting inference:")
                   for row in candidate["inferred_traits"])
        concepts.add(candidate["concept"]["key"])
    assert len(concepts) == len(candidates), \
        "each candidate must audition a different casting concept"
    # The interpolated dimensions are genuinely different between options, which
    # is what makes the human rating informative.
    signatures = {frozenset(row["dimension"] for row in candidate["inferred_traits"])
                  for candidate in candidates}
    assert len(signatures) == len(candidates)


def test_a_narrator_role_contributes_its_narrative_function_to_anchoring(studio):
    """Narration is known structure, so it may inform an anchor; a name may not."""
    project_id = build(studio, "Narrator anchoring", role_id="role-narr", role_name="Narrator")
    project = load_project(project_id)
    project.roles[0].role_type = RoleType.NARRATOR
    save_project(project)
    project = load_project(project_id)
    profile = project.character_cast.profiles["role-narr"]
    role = project.roles[0]
    assert "narration" in casting_anchors.anchor_query(profile, None, role)
    assert "narration" not in casting_anchors.anchor_query(profile, None, None)
    anchored = casting_anchors.anchors_for_role(profile, None, limit=1, role=role)[0]
    assert "narrator" in anchored["matched_traits"]
    assert anchored["performer"] in {"David Attenborough", "Stephen Fry", "Jim Dale", "LeVar Burton"}
    # The character's name never reaches the query that selects a reference.
    assert "Narrator" not in casting_anchors.anchor_query(profile, None, role)


def test_editorial_placeholders_alone_do_not_justify_a_reference(studio):
    """A neutral placeholder is not evidence, so it cannot be a matched trait.

    The candidate is still anchored — the casting-inference layer supplies a cast
    instead of leaving the voice generic — but the placeholder values must not
    appear as though the book had established them.
    """
    placeholder = CastingProfile(
        role_id="role-lead", revision="fixture-profile", dossier_revision="fixture-dossier",
        design_instruction="Create an original, consistent character voice.",
        decisions=[CastingDecision(dimension="pitch", value="mid", weight=0.5,
                                   reason="Neutral register pending audition; no explicit pitch evidence"),
                   CastingDecision(dimension="texture", value="clear", weight=0.5,
                                   reason="Neutral intelligible texture pending audition")])
    plan = casting_anchors.anchored_plan(placeholder, None, count=2)
    assert casting_anchors.vocal_traits(placeholder, None) == {}
    assert [row["anchored"] for row in plan] == [True, True]
    for row in plan:
        assert row["performer"] and row["performance"]
        assert "clear" not in row["matched_traits"] and "mid" not in row["matched_traits"]


def test_a_round_records_the_pipeline_version_that_produced_it(studio):
    project_id = build(studio, "Versioned round")
    round_ = prepare_round(project_id, "role-lead")
    assert round_["pipeline_version"] == AUTOCAST_PIPELINE_VERSION
    assert casting.round_is_stale(None) is False
    assert casting.round_is_stale({"pipeline_version": AUTOCAST_PIPELINE_VERSION}) is False
    assert casting.round_is_stale({"pipeline_version": "autocast-pipeline-v1"}) is True
    assert casting.round_is_stale({"id": "round_legacy"}) is True, \
        "a round with no recorded version came from an earlier pipeline"


def test_a_new_pipeline_version_recasts_unapproved_characters_and_keeps_their_history(studio):
    """Re-submitting under a new pipeline version must actually re-cast.

    The version bump is the documented signal that automatic casting's observable
    behaviour changed, so the previously reviewed candidates are no longer what the
    pipeline would propose. They are superseded for characters the human has not
    approved, and the old round with its ratings and direction stays in history.
    """
    project_id = build(studio, "Pipeline recast")
    round_ = prepare_round(project_id, "role-lead")
    original_revision = round_["revision"]
    old_candidates = [row["candidate_id"] for row in round_["candidates"]]
    # A low rating is recorded immediately and defers its replacement, so the round
    # the human was reviewing stays exactly as it was.
    rate(project_id, "role-lead", 0, 2, direction="more British")

    # Age the round as though an earlier pipeline had produced it.
    project = load_project(project_id)
    state = casting._state(project)
    state["roles"]["role-lead"]["round"]["pipeline_version"] = "autocast-pipeline-v1"
    save_project(project)

    casting.prepare(project_id, request_id="resubmitted-run")
    item = casting._state(load_project(project_id))["roles"]["role-lead"]
    fresh = item["round"]
    assert fresh["pipeline_version"] == AUTOCAST_PIPELINE_VERSION
    assert fresh["revision"] > original_revision, "the character must be re-cast"
    assert {row["candidate_id"] for row in fresh["candidates"]} & set(old_candidates) == set(), \
        "the new round must propose different candidates"
    # The superseded round is preserved, with the rating and the direction on it.
    archived = [row for row in item["history"] if row.get("revision") == original_revision]
    assert archived, "the reviewed round must move to history rather than being discarded"
    kept = next(row for row in archived[0]["candidates"] if row["candidate_id"] == old_candidates[0])
    assert kept["rating"] == 2
    assert archived[0]["direction"]["requested"] == ["more British"]
    # The re-submission names the characters it re-cast, so the reason is auditable.
    receipt = casting._state(load_project(project_id))["requests"]["resubmitted-run"]
    assert receipt["recast_role_ids"] == ["role-lead"]

    # Re-preparing the same re-submitted run is idempotent: one re-cast, not two.
    casting.prepare(project_id, request_id="resubmitted-run")
    again = casting._state(load_project(project_id))["roles"]["role-lead"]
    assert again["round_revision"] == item["round_revision"]


def test_a_current_pipeline_never_disturbs_a_round_under_review(studio):
    project_id = build(studio, "Current pipeline")
    round_ = prepare_round(project_id, "role-lead")
    assert round_["pipeline_version"] == AUTOCAST_PIPELINE_VERSION
    casting.prepare(project_id, request_id="second-prepare")
    after = casting._state(load_project(project_id))["roles"]["role-lead"]
    assert after["round_revision"] == 1
    assert after["round"]["revision"] == round_["revision"]


def test_an_approved_voice_is_never_recast_by_a_pipeline_version_change(studio):
    """An approval is the human's decision; a pipeline bump must not overturn it."""
    project_id = build(studio, "Approved survives", traits=())
    prepare_round(project_id, "role-lead")
    rate(project_id, "role-lead", 0, 5, direction="keep this", action="use")
    assert casting.approved(load_project(project_id), "role-lead")
    accepted = next(row for row in casting.view(project_id)["roles"] if row["role_id"] == "role-lead")
    assert accepted["status"] == "APPROVED"
    approved_revision = casting._state(load_project(project_id))["roles"]["role-lead"]["round_revision"]

    casting.prepare(project_id, request_id="pipeline-bump")
    item = casting._state(load_project(project_id))["roles"]["role-lead"]
    assert item["round_revision"] == approved_revision
    assert casting.approved(load_project(project_id), "role-lead")


def test_a_ready_candidate_carries_the_achieved_acoustic_receipt(studio, monkeypatch):
    """The audible candidate reports what conditioning achieved, not only the target.

    The target conditioning is the request; the achieved profile is the result. Only
    the round worker can attach the result, because only it sees the finished voice
    record — so this asserts the copy rather than the target that was already there.
    """
    project_id = build(studio, "Achieved acoustics")
    project = load_project(project_id)
    casting._new_round(project, "role-lead")
    save_project(project)
    state = casting._state(load_project(project_id))
    intent_id = state["roles"]["role-lead"]["round"]["intent_id"]

    receipt = {"schema": "identity-acoustics-v1", "achieved": True,
               "reach": {"verdict": "reached", "reached": ["f0_median_hz"], "shortfall": [],
                         "out_of_scope": [], "reached_count": 1, "shortfall_count": 0,
                         "out_of_scope_count": 0},
               "shaping_plan": {"semitones": -1.25, "rate": 0.95},
               "profile_before": {"f0_median_hz": 150.0}, "profile_after": {"f0_median_hz": 138.0}}

    def fake_audition(project_id, *, role_id, candidate_id, expected_identity_id, request_id,
                      script_version=None, reuse_key=None):
        make_ready(project_id, role_id, index=0, acoustics=copy.deepcopy(receipt))
        candidate = load_project(project_id).voice_lab.casting["roles"][role_id]["round"]["candidates"][0]
        return {"status": "succeeded", "key": candidate["key"], "audio_sha256": candidate["audio_sha256"],
                "duration_seconds": candidate["duration_seconds"]}

    from dubbing_service.app.audiobook import voice_lab_audition
    monkeypatch.setattr(voice_lab_audition, "audition", fake_audition)
    casting._round_worker(project_id, intent_id)

    fresh = load_project(project_id)
    ready = [c for c in casting._state(fresh)["roles"]["role-lead"]["round"]["candidates"]
             if c["status"] == "ready"]
    assert ready, "the stubbed audition must produce a ready candidate"
    assert ready[0]["acoustics"] == receipt
    # The target conditioning the request was built from is still there beside it.
    assert ready[0]["conditioning"]["parameters"]["pitch_target_hz"] > 0
    # An audible candidate exposes the achieved receipt through the review view too.
    row = next(r for r in casting.view(project_id)["roles"] if r["role_id"] == "role-lead")
    shown = next(c for c in row["candidates"] if c["status"] == "ready")
    assert shown["acoustics"]["shaping_plan"]["semitones"] == -1.25


def test_the_casting_pool_is_far_larger_than_the_hand_written_seeds():
    """The pool must not be the handful of names the seeds contain.

    A small pool is why the same few performers appeared in every cast. The
    shipped, reviewable sources file extends it, and every row it contributes is
    validated into the same shape the seeds use.
    """
    from dubbing_service.app.audiobook import performer_catalog, performer_discovery

    pool = performer_discovery.CATALOG_POOL
    assert pool["seeds"] == 10, "the hand-written seeds are the base of the pool"
    assert pool["shipped"] >= 60, f"the shipped sources file must be substantial, got {pool['shipped']}"
    assert pool["refused"] == [], f"no shipped row may be refused: {pool['refused']}"
    assert len(performer_discovery.CATALOG) >= 70
    names = [row["performer"] for row in performer_discovery.CATALOG]
    assert len(set(names)) == len(names), "one performer must appear once in the pool"
    assert len(set(names)) > pool["seeds"], "the pool must contain more than the original names"

    # The diversity the wider pool exists to provide is actually present.
    tags = [tag for row in performer_discovery.CATALOG for tag in row["tags"]]
    accents = {"british", "american", "irish", "scottish", "australian", "canadian",
               "african", "indian"}
    found = accents & set(tags)
    assert len(found) >= 6, f"the pool must cover several accent families, found {sorted(found)}"
    assert "male" in tags and "female" in tags
    assert {"young", "adult", "mature"} <= set(tags)

    # The pool version tracks the contents, so a changed pool is a visible change.
    assert ":" in performer_catalog.pool_version(pool)


def test_a_catalogue_row_without_retrievable_credit_evidence_is_refused():
    """A famous name is not a reference: the credit has to be there."""
    from dubbing_service.app.audiobook import performer_catalog

    good = {"candidate_id": "fixture-performer", "performer": "Fixture Person",
            "performance": "A Fixture Production (2020), narration",
            "tags": ["male", "narrator", "measured"],
            "creative_direction": "A measured fixture delivery.",
            "source_urls": ["https://example.test/credits"],
            "verified_facts": ["The credits page lists Fixture Person as narrator."],
            "facts_verified_at": "2026-09-29", "language": "en",
            "evidence_level": "metadata_only_exploratory", "audio_observations": [],
            "creative_interpretation": performer_catalog.CREATIVE_INTERPRETATION}
    assert performer_catalog.normalise(good)["candidate_id"] == "fixture-performer"

    for field, value, match in (
            ("source_urls", [], "retrievable credit URL"),
            ("performance", "", "specific performance"),
            ("performer", "", "performer"),
            ("verified_facts", [], "stated credit fact"),
            ("creative_direction", "", "delivery direction"),
            ("tags", ["not-a-real-tag"], "recognised casting tag"),
            ("tags", ["male", "female"], "contradictory presentation"),
    ):
        broken = {**good, field: value}
        with pytest.raises(performer_catalog.CatalogError, match=match):
            performer_catalog.normalise(broken)

    missing = {key: value for key, value in good.items() if key != "performance"}
    with pytest.raises(performer_catalog.CatalogError, match="missing: performance"):
        performer_catalog.normalise(missing)
    extra = {**good, "unexpected": 1}
    with pytest.raises(performer_catalog.CatalogError, match="unsupported keys: unexpected"):
        performer_catalog.normalise(extra)
    # A local path is not a URL a reviewer can re-check.
    with pytest.raises(performer_catalog.CatalogError, match="retrievable credit URL"):
        performer_catalog.normalise({**good, "source_urls": ["file:///tmp/credits.html"]})


def test_a_performer_source_can_be_adopted_into_the_pool_and_is_idempotent(tmp_path):
    """The pool grows from retrieved evidence without editing code."""
    from dubbing_service.app.audiobook import performer_catalog

    path = tmp_path / "adopted.json"
    row = {"candidate_id": "newly-sourced-performer", "performer": "Newly Sourced",
           "performance": "A Sourced Production (2024), narration",
           "tags": ["female", "mature", "narrator", "warm", "irish"],
           "creative_direction": "Warm, unhurried Irish narration.",
           "source_urls": ["https://example.test/sourced-credits"],
           "verified_facts": ["The credits page lists Newly Sourced as narrator."],
           "facts_verified_at": "2026-09-29", "language": "en",
           "evidence_level": "metadata_only_exploratory", "audio_observations": [],
           "creative_interpretation": performer_catalog.CREATIVE_INTERPRETATION}
    first = performer_catalog.adopt(row, path=path)
    assert first["adopted"] is True and first["pool_size"] == 1
    again = performer_catalog.adopt(row, path=path)
    assert again["adopted"] is False and again["reason"] == "already adopted"

    # The adopted row joins the pool and is ranked like any other reference.
    seeds = ({**row},)
    pool = performer_catalog.load_pool(seeds)
    assert "newly-sourced-performer" in {r["candidate_id"] for r in pool["performers"]}
    with pytest.raises(performer_catalog.CatalogError):
        performer_catalog.adopt({**row, "source_urls": []}, path=path)


def test_a_malformed_catalogue_file_is_reported_rather_than_quietly_shrinking_the_pool(tmp_path):
    """A pool that silently shrank is the defect; the report is the guard."""
    from dubbing_service.app.audiobook import performer_catalog

    seeds = ({"candidate_id": "fixture-seed", "performer": "Fixture Seed",
              "performance": "Fixture (2020)", "tags": ["male", "narrator"],
              "creative_direction": "Measured.", "source_urls": ["https://example.test/a"],
              "verified_facts": ["Listed as narrator."], "facts_verified_at": "2026-09-29",
              "language": "en", "evidence_level": "metadata_only_exploratory",
              "audio_observations": [],
              "creative_interpretation": performer_catalog.CREATIVE_INTERPRETATION},)
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(performer_catalog, "SOURCES_PATH", broken)
        patch.setattr(performer_catalog, "ADOPTED_PATH", tmp_path / "absent.json")
        pool = performer_catalog.load_pool(seeds)
    assert len(pool["performers"]) == 1
    assert pool["refused"] and "could not be read" in pool["refused"][0]

    alien = tmp_path / "alien.json"
    alien.write_text('{"schema": "something-else", "performers": []}', encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(performer_catalog, "SOURCES_PATH", alien)
        patch.setattr(performer_catalog, "ADOPTED_PATH", tmp_path / "absent.json")
        pool = performer_catalog.load_pool(seeds)
    assert pool["refused"] and "performer-catalog-sources-v1" in pool["refused"][0]


def test_two_characters_in_one_book_are_not_cast_the_same_performers(studio):
    """A book's characters should be read by different people.

    Without this, the highest-scoring performer for a trait set is proposed to
    every similar character, so a ten-character book can end up with the same
    three names repeated. The spread is a preference, never a hard rule: a cast
    large enough to exhaust the pool still gets anchored candidates.
    """
    from dubbing_service.app.audiobook import casting_anchors, performer_discovery

    decisions = (
        CastingDecision(dimension="presentation", value="masculine", reason="gender",
                        level="SUPPORTED", evidence_ids=["fact-gender"]),
        CastingDecision(dimension="perceived_vocal_age", value="adult", reason="age",
                        level="SUPPORTED", evidence_ids=["fact-age"]),
    )
    project_id = build(studio, "Spread across characters", traits=decisions)
    project = load_project(project_id)
    profile = project.character_cast.profiles["role-lead"]
    second = CastingProfile(role_id="role-other", revision="fixture-profile-2",
                            dossier_revision="fixture-dossier",
                            design_instruction="Create an original, consistent character voice.",
                            decisions=[copy.deepcopy(row) for row in decisions])

    first = casting_anchors.anchored_plan(profile, None, count=3)
    taken = [row["performer"] for row in first]
    second_plan = casting_anchors.anchored_plan(second, None, count=3, avoid_performers=taken)
    second_names = [row["performer"] for row in second_plan]
    assert second_names and all(row["anchored"] for row in second_plan)
    assert not set(second_names) & set(taken), \
        "the second character must not be offered the first character's performers"
    # Two concepts inside one round still differ from each other, and the two
    # identical characters produce different casts.
    assert len(set(taken)) == len(taken) == 3
    assert len(set(second_names)) == 3
    assert set(taken) != set(second_names)
    assert len(performer_discovery.CATALOG) >= 12, \
        "the pool must be deep enough for the spread to be possible"


def test_the_spread_is_a_preference_and_never_leaves_a_character_unanchored(studio):
    """An exhausted pool must still produce anchored candidates."""
    from dubbing_service.app.audiobook import casting_anchors, performer_discovery

    decisions = (CastingDecision(dimension="presentation", value="masculine", reason="gender",
                                 level="SUPPORTED", evidence_ids=["fact-gender"]),)
    project_id = build(studio, "Exhausted pool", traits=decisions)
    profile = load_project(project_id).character_cast.profiles["role-lead"]
    everyone = [row["performer"] for row in performer_discovery.CATALOG]
    plan = casting_anchors.anchored_plan(profile, None, count=3, avoid_performers=everyone)
    assert len(plan) == 3
    assert all(row["anchored"] for row in plan), \
        "a reused performer is a worse outcome than an unanchored candidate, but neither may be forced"
    assert {row["performer"] for row in plan} <= set(everyone)


def test_the_round_planner_spreads_a_books_cast_across_the_pool(studio):
    """The spread is applied by the round planner, not only by a direct call."""
    from dubbing_service.app.audiobook import performer_discovery

    decisions = (
        CastingDecision(dimension="presentation", value="masculine", reason="gender",
                        level="SUPPORTED", evidence_ids=["fact-gender"]),
        CastingDecision(dimension="perceived_vocal_age", value="adult", reason="age",
                        level="SUPPORTED", evidence_ids=["fact-age"]),
    )
    project_id = build(studio, "Planner spread", traits=decisions)
    project = load_project(project_id)
    for extra in ("role-two", "role-three"):
        project.character_cast.profiles[extra] = CastingProfile(
            role_id=extra, revision=f"fixture-{extra}", dossier_revision="fixture-dossier",
            design_instruction="Create an original, consistent character voice.",
            decisions=[copy.deepcopy(row) for row in decisions])
    save_project(project)

    # Plan the three similar characters in order, feeding each earlier cast back
    # into the state the planner reads — exactly what a real round does.
    project = load_project(project_id)
    state = casting._state(project)
    state["roles"] = {}
    used = {}
    for role_id in ("role-lead", "role-two", "role-three"):
        plan = casting._proposal_plan(project, role_id,
                                      project.character_cast.profiles[role_id], None,
                                      "Create an original, consistent character voice.", "")
        used[role_id] = [row["anchor"]["performer"] for row in plan if row.get("anchor")]
        state["roles"][role_id] = {"round_revision": 0, "history": [], "assignment": None,
                                   "rejections": [],
                                   "round": {"candidates": [{"anchor": row.get("anchor"),
                                                             "inspiration": row.get("inspiration")}
                                                            for row in plan]}}

    every = [name for names in used.values() for name in names]
    assert all(len(names) == 3 for names in used.values())
    assert len(set(every)) == len(every), \
        f"three similar characters must not share a cast, got {used}"
    assert len(set(every)) <= len(performer_discovery.CATALOG)


def test_an_explicit_pipeline_recast_recasts_approved_characters_and_keeps_history(studio):
    """A whole-pipeline re-run can opt in to re-casting approved characters.

    The default stays safe — a routine version bump never overturns an approval —
    so the opt-in is the only way to see what the current pool would propose for a
    character the human already accepted. The prior approval, its rating and its
    round all stay in history, and the approval itself is untouched until a new
    voice is chosen.
    """
    project_id = build(studio, "Opt-in recast")
    prepare_round(project_id, "role-lead")
    rate(project_id, "role-lead", 0, 5, direction="keep this", action="use")
    assert casting.approved(load_project(project_id), "role-lead")
    before = casting._state(load_project(project_id))["roles"]["role-lead"]
    approved_revision = before["round_revision"]
    old_candidates = [c["candidate_id"] for c in before["round"]["candidates"]]
    approved_key = (before.get("assignment") or {}).get("key")
    assert approved_key

    assert round_is_stale(before["round"]) is False, "the round is current for this pipeline"
    casting.prepare(project_id, request_id="plain-run")
    assert casting._state(load_project(project_id))["roles"]["role-lead"]["round_revision"] == \
        approved_revision, "a plain re-submission must leave the approval alone"

    casting.prepare(project_id, request_id="opt-in-recast", recast_approved=True)
    after = casting._state(load_project(project_id))["roles"]["role-lead"]
    assert after["round_revision"] > approved_revision, "the opt-in must actually re-cast"
    assert {c["candidate_id"] for c in after["round"]["candidates"]} & set(old_candidates) == set()
    archived = [row for row in after["history"] if row.get("revision") == before["round"]["revision"]]
    assert archived, "the accepted round must be preserved"
    kept = next(c for c in archived[0]["candidates"] if c["candidate_id"] == old_candidates[0])
    assert kept["rating"] == 5
    assert archived[0]["direction"]["requested"] == ["keep this"]
    # The approval is still recorded until a different voice is accepted.
    assert casting.approved(load_project(project_id), "role-lead")
    assert (after.get("assignment") or {}).get("key") == approved_key
    receipt = casting._state(load_project(project_id))["requests"]["opt-in-recast"]
    assert receipt["recast_approved"] is True
    assert receipt["recast_role_ids"] == ["role-lead"]


def test_a_recast_run_mints_its_own_run_rather_than_joining_a_completed_one(studio):
    """The opt-in is a different intent, so it cannot be absorbed by a finished run."""
    project_id = build(studio, "Recast run key")
    prepare_round(project_id, "role-lead")
    rate(project_id, "role-lead", 0, 5, action="use")
    first, created_first = casting.request_autocast_run(
        project_id, trigger="resubmit", reason="plain re-run")
    assert created_first is True
    assert first["recast_approved"] is False

    same, created_same = casting.request_autocast_run(
        project_id, trigger="resubmit", reason="plain re-run")
    assert created_same is False and same["key"] == first["key"], \
        "the same intent must be one effective run"

    second, created_second = casting.request_autocast_run(
        project_id, trigger="resubmit", reason="explicit re-cast", recast_approved=True)
    assert created_second is True
    assert second["key"] != first["key"], "a different intent needs its own run"
    assert second["recast_approved"] is True
    project = load_project(project_id)
    assert project.character_cast.autocast.key == second["key"]


def test_a_recast_in_flight_keeps_the_run_open_and_does_not_start_production(studio):
    """An explicit re-cast must not be mistaken for a finished cast.

    The approval from the previous pipeline still stands — correctly, because the
    human's decision is theirs — so a settle or auto-start that reads the approval
    alone closes the run and queues a render of the voices being replaced.
    """
    project_id = build(studio, "Recast in flight")
    prepare_round(project_id, "role-lead")
    rate(project_id, "role-lead", 0, 5, action="use")
    assert casting.approved(load_project(project_id), "role-lead")
    dispatch(project_id)

    # An explicit re-cast supersedes the round while the approval remains.
    casting.prepare(project_id, request_id="recast-in-flight", recast_approved=True)
    project = load_project(project_id)
    item = casting._state(project)["roles"]["role-lead"]
    assert item["round"]["status"] in {"pending", "running"}
    assert casting.approved(project, "role-lead"), "the approval stands until a new voice is chosen"

    # A run over that state must stay open, and no production may start.
    casting.request_autocast_run(project_id, trigger="resubmit", recast_approved=True)
    project = load_project(project_id)
    project.character_cast.autocast.required_role_ids = ["role-lead"]
    project.character_cast.autocast.status = "running"
    save_project(project)
    casting.settle_autocast_run(load_project(project_id))
    project = load_project(project_id)
    project.character_cast.autocast.status = "running"
    save_project(project)
    casting.settle_autocast_run(load_project(project_id))
    assert load_project(project_id).character_cast.autocast.status == "running", \
        "a round still generating keeps the run open"

    dispatch(project_id)
    state = casting._state(load_project(project_id))
    receipt = state["productions"].get(casting.AUTOMATIC_PRODUCTION_ID)
    assert receipt is None or receipt["status"] == "retracted", \
        "a re-cast in flight must not leave a renderable production receipt"
    # Completion is not announced for a cast that is no longer settled.
    events = casting.notification_events(load_project(project_id))
    assert not [row for row in events if row["event"] == casting.CASTING_COMPLETE]
    assert not [row for row in events if row["event"] == casting.PRODUCTION_READY]

def test_completion_is_not_announced_while_a_character_is_still_being_produced(studio):
    """The completion event means "nothing left to decide", not "approved once"."""
    project_id = build(studio, "Completion timing")
    prepare_round(project_id, "role-lead")
    rate(project_id, "role-lead", 0, 5, action="use")
    assert casting.approved(load_project(project_id), "role-lead")
    settled, required = casting._cast_settled(load_project(project_id))
    assert settled is True and required == ["role-lead"]

    # Supersede the round with an explicit re-cast: the cast is no longer settled.
    casting.prepare(project_id, request_id="recast-timing", recast_approved=True)
    settled, _ = casting._cast_settled(load_project(project_id))
    assert settled is False, "a round in flight means the cast is not settled"
    events = casting.notification_events(load_project(project_id))
    assert not [row for row in events if row["event"] == casting.CASTING_COMPLETE]

    # Completion is keyed by the cast too, so a re-cast can announce its own.
    project = load_project(project_id)
    first_cast = casting._cast_binding(project, ["role-lead"])
    assert first_cast
    assert casting._round_in_flight(project, "role-lead") is True


def test_a_superseded_production_receipt_is_not_promoted_or_resumed(studio):
    """A retracted receipt belongs to a replaced cast, so it must not render."""
    project_id = build(studio, "Retracted production")
    prepare_round(project_id, "role-lead")
    rate(project_id, "role-lead", 0, 5, action="use")
    dispatch(project_id)
    state = casting._state(load_project(project_id))
    # The same tick that creates the receipt also promotes it: the cast is settled.
    assert state["productions"][casting.AUTOMATIC_PRODUCTION_ID]["status"] == "queued"

    casting.prepare(project_id, request_id="recast-production", recast_approved=True)
    dispatch(project_id)
    receipt = casting._state(load_project(project_id))["productions"][casting.AUTOMATIC_PRODUCTION_ID]
    assert receipt["status"] == "retracted", \
        "a receipt whose cast is being replaced must not stay renderable"
    assert "Superseded" in receipt["reason"]
    # A retraction is not a production outcome, so nothing is announced for it.
    assert casting._production_event(load_project(project_id)) is None
    # Resuming it by hand is refused while its cast is still being replaced.
    with pytest.raises(casting.CastingError, match="cast that has since changed"):
        casting.resume_request(project_id, casting.AUTOMATIC_PRODUCTION_ID)


def test_a_production_worker_starting_after_the_cast_changed_does_not_publish(studio, monkeypatch):
    """The worker is the last defence: a replaced cast must not be published.

    A re-cast can begin between the tick that queued the render and the worker
    actually starting, so the worker re-checks rather than trusting the queue.
    """
    from dubbing_service.app.audiobook import pipeline

    project_id = build(studio, "Worker guard")
    prepare_round(project_id, "role-lead")
    rate(project_id, "role-lead", 0, 5, action="use")
    dispatch(project_id)
    receipt = casting._state(load_project(project_id))["productions"][casting.AUTOMATIC_PRODUCTION_ID]
    assert receipt["status"] == "queued"

    rendered = []
    monkeypatch.setattr(pipeline, "render_utterances",
                        lambda *a, **kw: rendered.append(a) or [])
    # The cast is replaced before the worker gets to run.
    casting.prepare(project_id, request_id="recast-worker", recast_approved=True)
    casting._production_worker(project_id, casting.AUTOMATIC_PRODUCTION_ID)

    assert rendered == [], "a superseded render must not produce audio"
    parked = casting._state(load_project(project_id))["productions"][casting.AUTOMATIC_PRODUCTION_ID]
    assert parked["status"] == "retracted"
    assert "cast changed before this render started" in parked["reason"]


def test_a_recast_that_settles_again_starts_a_fresh_render_and_keeps_the_retraction(studio):
    """A retraction must not block the render forever once the cast settles again.

    The receipt is keyed by a fixed request id, so treating any existing receipt as
    final would mean a project could never render after a re-cast — the retraction
    is a statement about one superseded cast, not a permanent veto.
    """
    project_id = build(studio, "Retract then settle")
    prepare_round(project_id, "role-lead")
    rate(project_id, "role-lead", 0, 5, action="use")
    dispatch(project_id)
    assert casting._state(load_project(project_id))["productions"][
        casting.AUTOMATIC_PRODUCTION_ID]["status"] == "queued"

    casting.prepare(project_id, request_id="recast-then-settle", recast_approved=True)
    dispatch(project_id)
    retracted = casting._state(load_project(project_id))["productions"][casting.AUTOMATIC_PRODUCTION_ID]
    assert retracted["status"] == "retracted"
    assert retracted.get("cast_binding")
    assert casting._automatic_production(load_project(project_id)) is False, \
        "a re-cast in flight must not start a render"

    # The re-cast produces a new round and the human approves it.
    make_ready(project_id, "role-lead", index=0, voice_id="voice_after_recast")
    complete_round(project_id, "role-lead")
    rate(project_id, "role-lead", 0, 5, action="use", event_id="evt-after-recast")
    assert casting.approved(load_project(project_id), "role-lead")
    dispatch(project_id)

    fresh = casting._state(load_project(project_id))["productions"][casting.AUTOMATIC_PRODUCTION_ID]
    assert fresh["status"] == "queued", "the settled cast must be renderable again"
    assert fresh["cast_binding"] != retracted["cast_binding"], \
        "the new receipt belongs to the new cast"
    assert fresh["superseded"], "the retraction is carried forward as history"
    assert "Superseded" in fresh["superseded"][-1]["retracted_reason"]
    # Idempotent: repeated ticks do not queue a second render.
    for _ in range(3):
        dispatch(project_id)
    again = casting._state(load_project(project_id))["productions"][casting.AUTOMATIC_PRODUCTION_ID]
    assert again["job_id"] == fresh["job_id"]


def test_an_unattributed_utterance_blocks_production_instead_of_crashing_the_tick(studio, monkeypatch):
    """A scope that cannot be frozen must be recorded, not re-raised every tick.

    Freezing the cast manifest requires a resolvable role for every utterance, and a
    partially attributed project has utterances that do not resolve. Promotion then
    raised, so the maintenance owner logged and retried it on every tick and the
    human saw nothing. The block is now recorded with its reason, which is the honest
    outcome: rendering only the resolved lines would silently ship a partial book.
    """
    project_id = build(studio, "Unfreezable scope")
    prepare_round(project_id, "role-lead")
    rate(project_id, "role-lead", 0, 5, action="use")
    dispatch(project_id)
    receipt = casting._state(load_project(project_id))["productions"][casting.AUTOMATIC_PRODUCTION_ID]
    assert receipt["status"] == "queued"

    # Re-open the receipt, then make manifest freezing fail the way an unresolved
    # utterance does on the real project.
    with casting.project_meta_lock(project_id):
        project = load_project(project_id)
        casting._state(project)["productions"][casting.AUTOMATIC_PRODUCTION_ID].update(
            status="awaiting_review", manifest=None)
        casting._save(project)

    def boom(*args, **kwargs):
        raise casting.CastingError("Unknown canonical speaking character")

    monkeypatch.setattr(casting, "freeze_manifest", boom)
    dispatch(project_id)
    blocked = casting._state(load_project(project_id))["productions"][casting.AUTOMATIC_PRODUCTION_ID]
    assert blocked["status"] == "blocked"
    assert "Unknown canonical speaking character" in blocked["reason"]

    # A later tick neither raises nor flips the receipt back to queued.
    dispatch(project_id)
    again = casting._state(load_project(project_id))["productions"][casting.AUTOMATIC_PRODUCTION_ID]
    assert again["status"] == "blocked"


def test_a_blocked_receipt_refreshes_a_stale_reason_once(studio, monkeypatch):
    """A block is terminal for promotion, so an outdated reason would persist forever.

    The stored reason is the only thing the human's notification carries, and it
    was written by whichever code first blocked the receipt. Re-deriving once per
    reason version is what lets an improved diagnosis reach them; the notification
    key stays stable so the corrected text never re-announces a delivered event.
    """
    project_id = build(studio, "Stale block reason")
    prepare_round(project_id, "role-lead")
    rate(project_id, "role-lead", 0, 5, action="use")
    dispatch(project_id)

    with casting.project_meta_lock(project_id):
        project = load_project(project_id)
        casting._state(project)["productions"][casting.AUTOMATIC_PRODUCTION_ID].update(
            status="blocked", manifest=None, reason_version=1,
            reason="Unknown canonical speaking character")
        casting._save(project)

    def boom(*args, **kwargs):
        calls.append(1)
        raise casting.CastingError(
            "871 of 1822 source utterance(s) in scope have no resolved speaking character")

    calls = []
    monkeypatch.setattr(casting, "freeze_manifest", boom)
    dispatch(project_id)
    refreshed = casting._state(load_project(project_id))["productions"][casting.AUTOMATIC_PRODUCTION_ID]
    assert refreshed["status"] == "blocked"
    assert refreshed["reason_version"] == casting.BLOCK_REASON_VERSION
    assert "871 of 1822" in refreshed["reason"]
    assert len(calls) == 1

    # The re-derivation is once per version, so a later tick neither re-reads the
    # scope nor asks the freezability question again.
    dispatch(project_id)
    assert len(calls) == 1
    assert refreshed["reason_version"] == casting.BLOCK_REASON_VERSION


def test_a_blocked_receipt_clears_itself_when_the_scope_becomes_freezable(studio):
    """The block must not outlive the condition that caused it.

    ``blocked`` is terminal for the promotion loop, so once it is recorded only this
    re-derivation can notice that the human attributed the utterances and let the
    render start without a manual resume.
    """
    project_id = build(studio, "Recovered block")
    prepare_round(project_id, "role-lead")
    rate(project_id, "role-lead", 0, 5, action="use")
    dispatch(project_id)

    with casting.project_meta_lock(project_id):
        project = load_project(project_id)
        casting._state(project)["productions"][casting.AUTOMATIC_PRODUCTION_ID].update(
            status="blocked", manifest=None, reason_version=1, reason="stale")
        casting._save(project)

    dispatch(project_id)
    recovered = casting._state(load_project(project_id))["productions"][casting.AUTOMATIC_PRODUCTION_ID]
    assert recovered["status"] == "queued"
    assert recovered["manifest"]["schema"] == "casting-production-manifest-v1"


def test_freezing_a_partially_attributed_book_names_the_population(studio):
    """The refusal must say what a human has to do, not raise a bare lookup error.

    Freezing looks a role up per utterance, so an unresolved one used to surface
    as "Unknown canonical speaking character" -- a message that names no count and
    no next action, on a project where the real answer is "871 lines need
    attribution". The two unresolved populations need different human actions, so
    the message separates them rather than reporting one undifferentiated total.
    """
    project_id = build(studio, "Partially attributed")
    prepare_round(project_id, "role-lead")
    rate(project_id, "role-lead", 0, 5, action="use")

    project = load_project(project_id)
    project.metadata["autocast_import"] = {"unresolved_utterance_ids": ["line-2", "line-3"]}
    save_project(project)
    save_segments(project_id, [
        Segment(id="line-1", index=0, text="A fixture line.", role_id="role-lead"),
        Segment(id="line-2", index=1, text="Who says this?", role_id=None,
                attribution=Attribution(label="Almon")),
        Segment(id="line-3", index=2, text="And this one?", role_id=None,
                attribution=Attribution(label="unknown")),
    ])
    project = load_project(project_id)
    _, rows, required, _ = casting._scope(project)
    assert len(rows) == 3

    with pytest.raises(casting.CastingError) as excinfo:
        casting.freeze_manifest(project, rows, required)
    message = str(excinfo.value)
    assert "2 of 3 source utterance(s) in scope have no resolved speaking character" in message
    assert "1 carry a proposed speaker awaiting attribution review" in message
    assert "1 have no speaker evidence" in message

    # Dropping the unresolved lines from the scope freezes normally, which is what
    # proves the refusal was about the unresolved population and nothing else.
    only_settled = [s for s in rows if s.id == "line-1"]
    manifest = casting.freeze_manifest(project, only_settled, required)
    assert manifest["utterance_ids"] == ["line-1"]


def test_anchoring_is_deterministic_for_the_same_character(studio):
    project_id = build(studio, "Deterministic anchoring")
    profile = load_project(project_id).character_cast.profiles["role-lead"]
    first = casting_anchors.anchored_plan(profile, None, count=3)
    second = casting_anchors.anchored_plan(profile, None, count=3)
    assert [(row["performer"], row["performance"], row["rationale"]) for row in first] == \
           [(row["performer"], row["performance"], row["rationale"]) for row in second]


# --------------------------------------------------------------------------
# 2 / 3 / 4. Low ratings never interrupt the session, and never lose evidence
# --------------------------------------------------------------------------

def test_low_rating_leaves_the_remaining_options_playable_and_scoreable(studio):
    project_id = build(studio, "Non-blocking rating")
    before = prepare_round(project_id, "role-lead")
    result = rate(project_id, "role-lead", 0, 2, direction="warmer")
    item = load_project(project_id).voice_lab.casting["roles"]["role-lead"]
    assert result["conflict"] is False and result["round_revision"] == item["round_revision"]

    # The batch is untouched: same round, same candidates, same audio, and the
    # unrated options are still playable and still scoreable.
    assert item["round"]["id"] == before["id"] and item["round_revision"] == before["revision"]
    assert item["round"]["status"] == "awaiting_review"
    for index, original in ((1, before["candidates"][1]), (2, before["candidates"][2])):
        candidate = item["round"]["candidates"][index]
        assert candidate["candidate_id"] == original["candidate_id"]
        assert candidate["status"] == "ready" and candidate["rating"] is None
        assert candidate["key"] == original["key"]
        assert candidate["audio_sha256"] == original["audio_sha256"]
        path, stored = lab.audio_record(project_id, candidate["key"])
        assert path.is_file() and stored["audio_sha256"] == candidate["audio_sha256"]
    assert item["round"]["candidates"][0]["rating"] == 2

    # And the remaining options can still be scored afterwards.
    rate(project_id, "role-lead", 1, 5)
    rate(project_id, "role-lead", 2, 1)
    ratings = [c["rating"] for c in
               load_project(project_id).voice_lab.casting["roles"]["role-lead"]["round"]["candidates"]]
    assert ratings == [2, 5, 1]


def test_low_rating_defers_one_replacement_instead_of_replacing_the_candidate(studio):
    project_id = build(studio, "Deferred replacement")
    before = prepare_round(project_id, "role-lead")
    rate(project_id, "role-lead", 0, 1, direction="deeper and more British")
    rate(project_id, "role-lead", 0, 3, direction="less nasal", event_id="evt-second")

    item = load_project(project_id).voice_lab.casting["roles"]["role-lead"]
    assert item["round_revision"] == before["revision"] and item["history"] == []
    request = load_project(project_id).voice_lab.casting["replacement_queue"]["role-lead"]
    assert request["schema"] == "casting-replacement-v1"
    assert request["round_id"] == before["id"] and request["status"] == "waiting_for_review"
    # Both ratings are retained, and no replacement round exists yet.
    assert [row["rating"] for row in request["candidates"]] == [1, 3]
    assert item["round"]["candidates"][0]["rating"] == 3
    assert item["round"]["candidates"][0]["needs_replacement"]["feedback"] == "less nasal"
    for index in (1, 2):
        assert item["round"]["candidates"][index]["needs_replacement"] is None

    # Only after the whole heard set is scored does one replacement round appear,
    # and it is a new round rather than a rewrite of the reviewed one.
    rate(project_id, "role-lead", 1, 4)
    rate(project_id, "role-lead", 2, 4)
    dispatch(project_id)
    after = load_project(project_id).voice_lab.casting["roles"]["role-lead"]
    assert after["round_revision"] == before["revision"] + 1
    assert after["round"]["replacement"] is True and after["round"]["id"] != before["id"]
    assert len(after["history"]) == 1 and after["history"][0]["id"] == before["id"]
    assert [c["rating"] for c in after["history"][0]["candidates"]] == [3, 4, 4]
    assert all(c["rating"] is None for c in after["round"]["candidates"])


def test_an_approved_voice_releases_its_deferred_replacement_request(studio):
    project_id = build(studio, "Approved before replacement")
    prepare_round(project_id, "role-lead")
    rate(project_id, "role-lead", 0, 2)
    rate(project_id, "role-lead", 1, 1)
    rate(project_id, "role-lead", 2, 5, action="use")
    assert casting.approved(load_project(project_id), "role-lead")
    dispatch(project_id)

    item = load_project(project_id).voice_lab.casting["roles"]["role-lead"]
    assert item["round_revision"] == 1, "an approved character must not be recast by a stale request"
    request = load_project(project_id).voice_lab.casting["replacement_queue"]["role-lead"]
    assert request["status"] == "superseded"
    assert request["reason"] == "This character already has an approved voice"


def test_qualitative_feedback_survives_into_the_replacement_round(studio):
    project_id = build(studio, "Retained direction")
    prepare_round(project_id, "role-lead")
    rate(project_id, "role-lead", 0, 2, direction="more British, deeper, less nasal")
    rate(project_id, "role-lead", 1, 1, direction="warmer")
    rate(project_id, "role-lead", 2, 4)
    dispatch(project_id)

    item = load_project(project_id).voice_lab.casting["roles"]["role-lead"]
    assert item["round"]["replacement"] is True
    assert item["round"]["direction"]["requested"] == ["more British, deeper, less nasal", "warmer"]
    assert item["round"]["proposal_count"] == 3
    instructions = [m["design_instruction"] for m in
                    load_project(project_id).voice_lab.shortlists["role-lead"]["matches"]
                    if m["round_id"] == item["round"]["id"]]
    assert instructions
    for instruction in instructions:
        assert "Human casting direction:" in instruction
        assert "more British, deeper, less nasal" in instruction
        assert "warmer" in instruction


# --------------------------------------------------------------------------
# 5 / 6 / 7. Cross-project preferred-voice memory
# --------------------------------------------------------------------------

def rate_single_voice(studio, *, name, stars, traits=ELDER_TRAITS, role_id="role-a",
                      role_name="Narrator", direction="") -> str:
    """One project with one heard voice, rated, published to the voice library."""
    project_id = build(studio, name, traits=traits, role_id=role_id, role_name=role_name)
    prepare_round(project_id, role_id, count=1, request_id=f"{project_id}-prepare")
    rate(project_id, role_id, 0, stars, direction=direction)
    return project_id


def test_a_four_star_voice_from_one_project_is_proposed_to_a_similar_character_elsewhere(studio):
    rate_single_voice(studio, name="Project A", stars=4, direction="measured and warm")

    target_id = build(studio, "Project C", traits=ELDER_TRAITS, role_id="role-b",
                      role_name="The Librarian")
    casting.prepare(target_id, request_id="c-prepare")
    role = casting_role(target_id, "role-b")
    prior = [c for c in role["candidates"] if c["prior_match"]]
    assert prior, "a sufficiently similar character must retrieve the approved voice"
    first = prior[0]
    assert first["prior_evidence"]["prior_rating"] == 4
    assert first["prior_evidence"]["match_percent"] >= preferred_voices.MATCH_THRESHOLD
    assert "presentation: masculine" in first["prior_evidence"]["match_reasons"]
    assert "accent: British" in first["prior_evidence"]["match_reasons"]
    assert "perceived vocal age: older adult" in first["prior_evidence"]["match_reasons"]
    assert "previously inspired by" in " ".join(first["prior_evidence"]["match_reasons"])
    assert first["prior_evidence"]["previously_used"]
    assert first["library_voice_id"] and first["library_version"]
    assert first["match_rationale"].startswith("Previously approved voice")
    # Hear preserved rated samples first; exploration is an explicit next step.
    assert all(c["prior_match"] for c in role["candidates"])

    # A character with opposed vocal traits does not retrieve it.
    other_id = build(studio, "Project D", traits=YOUNG_TRAITS, role_id="role-c", role_name="Sela")
    casting.prepare(other_id, request_id="d-prepare")
    other = casting_role(other_id, "role-c")
    assert not [c for c in other["candidates"] if c["prior_match"]]
    assert other["candidates"] and all(c["anchor"]["anchored"] for c in other["candidates"])


def test_a_five_star_voice_carries_a_stronger_reuse_signal_than_a_four_star_voice(studio):
    rate_single_voice(studio, name="Signal four", stars=4)
    rate_single_voice(studio, name="Signal five", stars=5)

    entries = {row["signal"]: row for row in preferred_voices.preferred_entries()}
    assert set(entries) == {4, 5}
    profile = CastingProfile(role_id="role-x", revision="r", dossier_revision="d",
                             design_instruction="x", decisions=list(ELDER_TRAITS))
    proposals = preferred_voices.match_preferred(casting_anchors.vocal_traits(profile, None))
    assert len(proposals) == 2
    assert [row["signal"] for row in proposals] == [5, 4], \
        "all else equal, a five-star prior voice must outrank a four-star one"
    assert proposals[0]["match_percent"] > proposals[1]["match_percent"]
    assert proposals[0]["match_reasons"] == proposals[1]["match_reasons"]


def test_a_low_rated_voice_is_retained_as_history_but_never_promoted(studio):
    project_id = rate_single_voice(studio, name="Low rated", stars=2, direction="sounds robotic")

    entries = preferred_voices.preferred_entries(include_retired=True)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["signal"] is None and entry["promoted"] is False
    assert entry["highest_rating"] == 2
    assert entry["ratings"][0]["rating"] == 2
    assert entry["ratings"][0]["feedback"] == "sounds robotic"
    assert entry["ratings"][0]["project_id"] == project_id
    assert preferred_voices.preferred_entries() == []
    assert preferred_voices.match_preferred({"presentation": "masculine", "accent": "British"}) == []

    target_id = build(studio, "Low rated target", traits=ELDER_TRAITS, role_id="role-b")
    casting.prepare(target_id, request_id="target-prepare")
    assert not [c for c in casting_role(target_id, "role-b")["candidates"] if c["prior_match"]]


def test_a_prior_voice_is_proposed_and_never_silently_assigned(studio):
    rate_single_voice(studio, name="Assign source", stars=5)

    target_id = build(studio, "Assign target", traits=ELDER_TRAITS, role_id="role-b")
    casting.prepare(target_id, request_id="assign-prepare")
    role = casting_role(target_id, "role-b")
    prior = [c for c in role["candidates"] if c["prior_match"]]
    assert prior, "the proposal must exist for this assertion to mean anything"
    project = load_project(target_id)
    assert role["status"] != "APPROVED"
    assert casting.approved(project, "role-b") is False
    assert project.voice_lab.casting["roles"]["role-b"].get("assignment") is None
    assert next(item for item in project.roles if item.id == "role-b").voice_identity is None
    # Nothing is audible or reusable until the option itself is requested, and
    # even after the import step the voice is only a candidate.
    assert all(c["reuse_key"] is None for c in role["candidates"])
    dispatch(target_id)
    role = casting_role(target_id, "role-b")
    assert role["status"] != "APPROVED"
    assert casting.approved(load_project(target_id), "role-b") is False
    assert all(c["status"] != "resolved" for c in role["candidates"])


def test_historical_ratings_stay_attached_to_the_voice_after_a_later_recast(studio):
    project_id = build(studio, "Recast history")
    prepare_round(project_id, "role-lead")
    rate(project_id, "role-lead", 0, 5, direction="keep this", action="use")
    assert casting.approved(load_project(project_id), "role-lead")
    approved_key = load_project(project_id).voice_lab.casting["roles"]["role-lead"]["assignment"]["key"]

    # A later explicit recast and a low rating must not rewrite the approval or
    # the rating that produced it.
    casting.feedback(project_id, body_for(project_id, "role-lead", 0, stars=None,
                                          action="recast", event_id="evt-recast"))
    make_ready(project_id, "role-lead", index=0, voice_id="voice_second")
    complete_round(project_id, "role-lead")
    rate(project_id, "role-lead", 0, 1, direction="too different")

    item = load_project(project_id).voice_lab.casting["roles"]["role-lead"]
    assert item["round_revision"] == 2 and len(item["history"]) == 1
    assert [c["rating"] for c in item["history"][0]["candidates"]] == [5, None, None]
    assert item["assignment"]["approved"] is True
    assert item["assignment"]["stars"] == 5 and item["assignment"]["key"] == approved_key
    assert any(row.get("rating") == 1 for row in item["rejections"])

    entries = preferred_voices.preferred_entries(include_retired=True)
    assert len(entries) == 2, "each rated voice keeps its own library identity"
    approved_entry = next(row for row in entries if row["signal"] == 5)
    later_entry = next(row for row in entries if row["signal"] is None)
    assert [row["rating"] for row in approved_entry["ratings"]] == [5]
    assert approved_entry["ratings"][0]["feedback"] == "keep this"
    assert approved_entry["highest_rating"] == 5 and approved_entry["promoted"] is True
    assert [row["rating"] for row in later_entry["ratings"]] == [1]
    assert later_entry["ratings"][0]["feedback"] == "too different"
    assert later_entry["promoted"] is False
    assert preferred_voices.preferred_entries() == [approved_entry]


def test_a_voice_identity_is_stable_across_projects_and_characters(studio):
    rate_single_voice(studio, name="Identity source", stars=5)
    entry = preferred_voices.preferred_entries()[0]
    library_voice_id, version = entry["library_voice_id"], entry["version"]
    assert entry["characters"][0]["character"] == "Narrator"
    assert entry["characters"][0]["role_id"] == "role-a"
    identity = preferred_voices.stable_voice_id(library_voice_id, version)
    assert identity not in {library_voice_id, version}

    target_id = build(studio, "Identity target", traits=ELDER_TRAITS, role_id="role-z",
                      role_name="The Sculptor")
    casting.prepare(target_id, request_id="identity-prepare")
    prior = next(c for c in casting_role(target_id, "role-z")["candidates"] if c["prior_match"])
    assert prior["library_voice_id"] == library_voice_id
    assert prior["library_version"] == version
    assert preferred_voices.stable_voice_id(prior["library_voice_id"],
                                            prior["library_version"]) == identity

    # Importing the voice into another project and character keeps that identity.
    dispatch(target_id)
    candidate = next(c for c in casting_role(target_id, "role-z")["candidates"]
                     if c["candidate_id"] == prior["candidate_id"])
    assert candidate["import_state"] == "ready"
    imported = load_project(target_id).voice_lab.voices[candidate["reuse_key"]]
    assert preferred_voices.stable_voice_id(imported["library_voice_id"],
                                            imported["library_version"]) == identity
    assert imported["origin_project_id"] == entry["projects"][0]
    # The first character's use of the voice is still recorded against the voice.
    recorded = preferred_voices.library_preferred(library_voice_id, version)
    assert recorded["characters"][0]["character"] == "Narrator"
    assert recorded["generation"]["design_instruction"]


# --------------------------------------------------------------------------
# Import path: the proposed prior voice becomes the exact heard voice
# --------------------------------------------------------------------------

def test_a_proposed_prior_voice_is_imported_as_the_exact_approved_voice(studio):
    source_id = rate_single_voice(studio, name="Import source", stars=5)
    source_key, source_record = next(iter(load_project(source_id).voice_lab.voices.items()))

    target_id = build(studio, "Import target", traits=ELDER_TRAITS, role_id="role-b")
    casting.prepare(target_id, request_id="import-prepare")
    prior = next(c for c in casting_role(target_id, "role-b")["candidates"] if c["prior_match"])
    dispatch(target_id)

    candidate = next(c for c in casting_role(target_id, "role-b")["candidates"]
                     if c["candidate_id"] == prior["candidate_id"])
    assert candidate["import_state"] == "ready" and candidate["reuse_key"]
    # The import is project-local: the target project now owns its own copy of the
    # approved voice, while the source project's record is untouched.
    target_voices = load_project(target_id).voice_lab.voices
    assert candidate["reuse_key"] in target_voices
    imported = target_voices[candidate["reuse_key"]]
    # The copy is project-local: its artifact paths point inside the target
    # project, and only the copy knows which library voice it came from.
    assert imported["audio_file"] != source_record["audio_file"]
    assert imported["identity"]["reference_audio"] != source_record["identity"]["reference_audio"]
    assert imported["library_voice_id"] == prior["library_voice_id"]
    assert imported["library_version"] == prior["library_version"]
    assert source_record.get("library_voice_id") is None
    assert imported["audio_sha256"] == source_record["audio_sha256"]
    assert imported["identity"]["id"] == source_record["identity"]["id"]
    assert imported["origin_project_id"] == source_id
    assert load_project(source_id).voice_lab.voices[source_key] == source_record
    assert load_project(target_id).voice_lab.casting["roles"]["role-b"].get("assignment") is None


def test_an_unavailable_prior_voice_blocks_that_option_instead_of_inventing_a_voice(studio, monkeypatch):
    rate_single_voice(studio, name="Blocked source", stars=5)

    target_id = build(studio, "Blocked target", traits=ELDER_TRAITS, role_id="role-b")
    casting.prepare(target_id, request_id="blocked-prepare")
    monkeypatch.setattr(preferred_voices, "prepare_prior_candidate",
                        lambda *args, **kwargs: {"state": "blocked", "reason": "TEST library unavailable"})
    dispatch(target_id)
    prior = next(c for c in casting_role(target_id, "role-b")["candidates"] if c["prior_match"])
    assert prior["import_state"] == "waiting" and prior["reuse_key"] is None
    assert prior["import_reason"] == "TEST library unavailable"

    for _ in range(casting.MAX_IMPORT_ATTEMPTS):
        casting._prepare_prior_imports(target_id)
    prior = next(c for c in casting_role(target_id, "role-b")["candidates"] if c["prior_match"])
    assert prior["import_state"] == "blocked" and prior["reuse_key"] is None
    assert prior["import_attempts"] == casting.MAX_IMPORT_ATTEMPTS
    # No substitute voice is generated for the option the human chose to hear.
    assert prior["anchor"] is None and prior["provenance"] == "prior_voice"
    assert prior["status"] == "pending"


def test_a_round_worker_refuses_to_generate_a_substitute_for_an_unavailable_prior_voice(studio, monkeypatch):
    rate_single_voice(studio, name="Refuse source", stars=5)
    target_id = build(studio, "Refuse target", traits=ELDER_TRAITS, role_id="role-b")
    casting.prepare(target_id, request_id="refuse-prepare")

    def explode(*args, **kwargs):
        raise AssertionError("a substitute voice must never be generated here")

    monkeypatch.setattr("dubbing_service.app.audiobook.voice_lab_audition.audition", explode)
    item = load_project(target_id).voice_lab.casting["roles"]["role-b"]
    casting._round_worker(target_id, item["round"]["intent_id"])

    round_ = load_project(target_id).voice_lab.casting["roles"]["role-b"]["round"]
    assert round_["status"] == "pending", "one blocked option must not fail the whole round"
    blocked = next(c for c in round_["candidates"]
                   if c["library_voice_id"] and c["status"] == "blocked")
    assert blocked["import_state"] == "blocked"
    assert "not available in this project yet" in blocked["reason"]
