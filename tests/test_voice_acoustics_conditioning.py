"""The performer-reference layer must change the SOUND, not only the metadata.
Two layers are proven here.

* The worker layer (``CharacterVoiceRuntime``): a generated original reference is
  measured, shaped toward its reference-derived acoustic target, re-measured, and
  only then used to build the Base prompt. The prompt therefore encodes the
  conditioned audio, and a conditioned identity cannot be prepared without its
  conditioning pass.
* The production layer (``casting_review`` + ``voice_lab``): every freshly
  generated candidate carries a specific performer and performance, the reference
  evidence behind it, the acoustic characterization derived from that reference,
  and the numeric parameters the generation request actually receives.

Everything runs against disposable project roots with fixture audio and a fixture
model. No real book text, GPU, network, performer recording or human rating is
used.
"""
from __future__ import annotations

import contextlib
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from dubbing_service.app.audiobook import casting_anchors
from dubbing_service.app.audiobook import casting_inference
from dubbing_service.app.audiobook import casting_review as casting
from dubbing_service.app.audiobook import performer_reference
from dubbing_service.app.audiobook import voice_lab as lab
from dubbing_service.app.audiobook.fingerprint import file_sha256, voice_revision_for_role
from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.models.character import CastingDecision, CastingProfile
from dubbing_service.app.models.project import Role, Segment
from dubbing_service.app.providers.tts import TTSResult
from dubbing_service.app.storage.projects import create_project, load_project, save_project
from dubbing_service.app.storage.segments import save_segments
from service import qwen_character_voice as voice
from tests.audiobook_helpers import make_tone_wav, use_projects_root


# --------------------------------------------------------------------------- #
# worker layer: the shaped reference is the one the prompt is built from
# --------------------------------------------------------------------------- #
class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return Tensor(self.value.astype(np.float32))

    def numpy(self):
        return self.value


@dataclass
class Prompt:
    ref_code: object
    ref_spk_embedding: object
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str


DESIGN_F0_HZ = 150.0
TARGET_F0_HZ = 205.0


@pytest.fixture
def worker(tmp_path, monkeypatch):
    """A runtime whose design model emits a tone at a known pitch."""
    project = tmp_path / "project"
    project.mkdir()
    prefixes = []

    def design_tone(**kwargs):
        frames = 24000 * 3
        tone = 0.25 * np.sin(2 * np.pi * DESIGN_F0_HZ * np.arange(frames) / 24000.0)
        return [tone.astype(np.float32)], 24000

    class Model:
        def generate_voice_design(self, **kwargs):
            return design_tone(**kwargs)

        def create_voice_clone_prompt(self, *, ref_audio, ref_text, x_vector_only_mode):
            prefixes.append(Path(ref_audio))
            return [Prompt(Tensor(np.ones((12, 16), dtype=np.int64)),
                           Tensor(np.arange(128, dtype=np.float32)), x_vector_only_mode, True, ref_text)]

    monkeypatch.setattr(voice, "safe_project_root", lambda _: project)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        Tensor=Tensor, from_numpy=Tensor,
        cuda=SimpleNamespace(is_available=lambda: False, current_device=lambda: 0),
        manual_seed=lambda _: None,
        random=SimpleNamespace(fork_rng=lambda **_: contextlib.nullcontext())))
    monkeypatch.setitem(sys.modules, "qwen_tts.inference.qwen3_tts_model",
                        SimpleNamespace(VoiceClonePromptItem=Prompt))
    runtime = voice.CharacterVoiceRuntime(project, lambda kind: Model())
    monkeypatch.setattr(runtime, "_model_metadata",
                        lambda kind: {"model": "fixture-base", "model_revision": "revision-1",
                                      "runtime_version": "0.1.1"})
    anchors = casting_anchors.anchored_plan(None, None, count=1)
    conditioning = performer_reference.conditioning_payload(
        performer_reference.resolve(anchors[0], anchors[0]["concept"]))
    return SimpleNamespace(runtime=runtime, project=project, prefixes=prefixes,
                           conditioning=conditioning, anchor=anchors[0])


def test_the_base_prompt_is_built_from_the_conditioned_reference(worker):
    identity_id = "voice-acoustics-fixture"
    worker.runtime.design("project", identity_id, "Create an original synthetic voice.",
                          "A short spoken fixture line for conditioning.", seed=7,
                          conditioning=worker.conditioning)
    acoustics = worker.runtime.acoustics("project", identity_id)
    worker.runtime.prepare("project", identity_id)

    # The Base prompt encodes the *shaped* reference, not the design output.
    conditioned = (worker.project / acoustics["reference_audio"]).resolve()
    assert worker.prefixes == [conditioned]
    original = (worker.project / acoustics["source_reference_audio"]).resolve()
    assert worker.prefixes[0] != original
    assert file_sha256(original) == acoustics["source_reference_audio_sha256"]
    assert file_sha256(conditioned) == acoustics["reference_audio_sha256"]


def test_the_shaped_reference_actually_moves_toward_the_acoustic_target(worker):
    worker.conditioning["target_profile"]["f0_median_hz"] = TARGET_F0_HZ
    identity_id = "voice-acoustics-move"
    worker.runtime.design("project", identity_id, "Create an original synthetic voice.",
                          "A short spoken fixture line for conditioning.", seed=7,
                          conditioning=worker.conditioning)
    acoustics = worker.runtime.acoustics("project", identity_id)

    before = acoustics["profile_before"]["f0_median_hz"]
    after = acoustics["profile_after"]["f0_median_hz"]
    assert abs(before - DESIGN_F0_HZ) < 10.0, "the measurement must track the design output"
    assert after > before, "conditioning must raise the measured pitch toward the target"
    assert abs(after - TARGET_F0_HZ) < abs(before - TARGET_F0_HZ), \
        "the conditioned reference must be closer to the target than the design output"
    # Pitch is moved by bounded resynthesis, so the change is real audio, not a hint.
    assert acoustics["shaping_plan"]["semitones"] != 0.0
    shaped = worker.project / acoustics["reference_audio"]
    unshaped = worker.project / acoustics["source_reference_audio"]
    assert file_sha256(shaped) != file_sha256(unshaped)
    shaped_samples, rate = sf.read(str(shaped), dtype="float32")
    assert rate > 0 and shaped_samples.size > 0


def test_a_conditioned_identity_cannot_be_prepared_before_its_conditioning_pass(worker):
    identity_id = "voice-acoustics-required"
    worker.runtime.design("project", identity_id, "Create an original synthetic voice.",
                          "A short spoken fixture line for conditioning.", seed=7,
                          conditioning=worker.conditioning)
    with pytest.raises(ValueError, match="must apply its acoustic conditioning"):
        worker.runtime.prepare("project", identity_id)


def test_conditioning_receipts_are_immutable_and_bound_to_one_target(worker):
    identity_id = "voice-acoustics-immutable"
    worker.runtime.design("project", identity_id, "Create an original synthetic voice.",
                          "A short spoken fixture line for conditioning.", seed=7,
                          conditioning=worker.conditioning)
    first = worker.runtime.acoustics("project", identity_id)
    assert worker.runtime.acoustics("project", identity_id) == first

    other = copy.deepcopy(worker.conditioning)
    other["target_profile"]["f0_median_hz"] = 88.0
    with pytest.raises(ValueError, match="already conditioned with a different target"):
        worker.runtime.acoustics("project", identity_id, other)


def test_width_and_acoustics_never_change_the_design_receipt(worker):
    """The design output stays the pre-conditioning record it always was."""
    identity_id = "voice-acoustics-provenance"
    design = worker.runtime.design("project", identity_id, "Create an original synthetic voice.",
                                  "A short spoken fixture line for conditioning.", seed=7,
                                  conditioning=worker.conditioning)
    worker.runtime.acoustics("project", identity_id)
    stored = json.loads((worker.project / "casting/voices" / identity_id / "reference.json")
                        .read_text(encoding="utf-8"))
    assert stored == design
    assert stored["provenance"]["origin"] == "generated_original_voice"
    assert stored["provenance"]["actor_audio_used"] is False


def test_the_conditioning_receipt_states_what_was_and_was_not_reached(worker):
    """The receipt separates reach, shortfall, and keys outside any actuator's scope."""
    identity_id = "voice-acoustics-reach"
    worker.conditioning["target_profile"]["f0_median_hz"] = TARGET_F0_HZ
    worker.runtime.design("project", identity_id, "Create an original synthetic voice.",
                          "A short spoken fixture line for conditioning.", seed=7,
                          conditioning=worker.conditioning)
    acoustics = worker.runtime.acoustics("project", identity_id)
    reach = acoustics["reach"]

    assert reach["verdict"] in {"reached", "shortfall", "could_not_measure"}
    assert reach["measurable"] is True
    assert "f0_median_hz" in reach["reached"] + reach["shortfall"]
    # The documented prior carries keys no pitch/rate transform moves; they are
    # reported as out of scope rather than counted as failures.
    assert reach["out_of_scope"], "a descriptor prior carries keys outside the actuator's reach"
    assert not set(reach["shortfall"]) & set(reach["out_of_scope"])
    assert acoustics["achieved"] == reach["reached_target"]
    assert acoustics["achieved_all_keys"] is False
    assert acoustics["profile_after_measured"], "the raw re-measurement is retained"
    assert acoustics["reconciliation"]["rate_agrees"] is True


def test_a_reference_with_no_measurable_target_key_reports_could_not_measure(worker):
    """An unmeasurable comparison is never reported as reached."""
    from service import voice_acoustics

    assert voice_acoustics.achievement(
        {"exceeded": ["peak_dbfs"], "compared": ["peak_dbfs"]})["verdict"] == "could_not_measure"


# --------------------------------------------------------------------------- #
# production layer: the candidate and the generation request
# --------------------------------------------------------------------------- #
ELDER_FACTS = (
    CastingDecision(dimension="presentation", value="masculine", reason="established gender",
                    level="SUPPORTED", evidence_ids=["fact-gender"]),
    CastingDecision(dimension="perceived_vocal_age", value="older adult", reason="apparent age",
                    level="INFERRED", evidence_ids=["fact-age"]),
    CastingDecision(dimension="accent", value="British", reason="accent evidence", level="SUPPORTED",
                    evidence_ids=["fact-accent"]),
)
SPARSE_FACTS = (
    CastingDecision(dimension="presentation", value="masculine", reason="established gender",
                    level="SUPPORTED", evidence_ids=["fact-gender"]),
)


class RecordingProvider:
    """Records exactly what the generation request received."""

    def __init__(self):
        self.requests = []

    def design_character_voice(self, project_id, identity_id, instruction, text, seed, conditioning=None):
        self.requests.append({"identity_id": identity_id, "instruction": instruction, "text": text,
                              "seed": seed, "conditioning": copy.deepcopy(conditioning)})
        root = safe_project_root(project_id)
        reference = make_tone_wav(root / f"voices/{identity_id}-reference.wav", seconds=1.0,
                                  frequency=200.0)
        prompt = root / f"voices/{identity_id}-prompt.npz"
        prompt.write_bytes(b"LOCAL-TEST-SIGNAL-NOT-A-MODEL-PROMPT")
        return {"mechanism": "qwen_reference", "backend": "qwen3tts",
                "model": "LOCAL-TEST-SIGNAL-NOT-TTS", "runtime_version": "test-only",
                "reference_audio": reference.relative_to(root).as_posix(),
                "reference_audio_sha256": file_sha256(reference), "reference_text": text,
                "prompt_artifact": prompt.relative_to(root).as_posix(),
                "prompt_sha256": file_sha256(prompt),
                "provenance": {"origin": "generated_original_voice", "fixture_only": True}}

    def synthesize_segment(self, project_id, role, segment):
        path = make_tone_wav(safe_project_root(project_id) / f"signals/{segment.id}.wav", seconds=8)
        return TTSResult(success=True, audio_path=str(path), raw={
            "identity_id": role.voice_identity.id,
            "prompt_sha256": role.voice_identity.prompt_sha256,
            "execution_profile": {"model_id": "LOCAL-TEST-SIGNAL-NOT-TTS", "fixture_only": True}})


def build_studio(monkeypatch, tmp_path, *, decisions=ELDER_FACTS, name="Acoustic conditioning"):
    root = use_projects_root(monkeypatch, tmp_path)
    project = create_project(name)
    role = Role(id="role-lead", name="Lead", role_type="character", tts_backend="qwen3tts_cv",
                qwen3_speaker="ryan", qwen3_language="English")
    project.roles = [role]
    project.character_cast.profiles[role.id] = CastingProfile(
        role_id=role.id, revision="fixture-profile", dossier_revision="fixture-dossier",
        design_instruction="Create an original, consistent character voice.",
        decisions=[copy.deepcopy(row) for row in decisions])
    save_project(project)
    save_segments(project.id, [Segment(id="line-1", index=0, text="A fixture line.",
                                       role_id=role.id)])
    return root, project, role


def prepared_round(project_id, role_id, *, request_id="acoustic-prepared"):
    casting.prepare(project_id, request_id=request_id)
    state = casting._state(load_project(project_id))
    return state["roles"][role_id]["round"]


def generate(project_id, round_, provider, *, candidate_index=0, request_id="acoustic-design"):
    project = load_project(project_id)
    role = next(item for item in project.roles if item.id == "role-lead")
    expected = role.voice_identity.id if role.voice_identity else voice_revision_for_role(role)
    candidate = round_["candidates"][candidate_index]
    return lab.design(project_id, "role-lead", candidate["candidate_id"], expected, request_id,
                      provider=provider)


def test_every_fresh_candidate_has_a_performer_a_performance_and_reference_evidence(monkeypatch, tmp_path):
    _root, project, _role = build_studio(monkeypatch, tmp_path)
    round_ = prepared_round(project.id, "role-lead")
    assert round_["candidates"], "production must propose candidates"
    for candidate in round_["candidates"]:
        if candidate["prior_match"]:
            continue
        assert candidate["provenance"] == "generated"
        assert candidate["anchor"]["anchored"] is True
        assert candidate["anchor"]["performer"]
        assert candidate["anchor"]["performance"]
        reference = candidate["reference"]
        assert reference["performer"] == candidate["anchor"]["performer"]
        assert reference["performance"] == candidate["anchor"]["performance"]
        # Source evidence and the state of the acquisition attempt are both stored.
        assert reference["source_type"]
        assert "retrieval_state" in reference
        assert reference["rationale"]


def test_a_sparse_character_is_cast_not_left_generic_and_never_shows_inference_as_book_fact(
        monkeypatch, tmp_path):
    _root, project, _role = build_studio(monkeypatch, tmp_path, decisions=SPARSE_FACTS,
                                         name="Sparse acoustic conditioning")
    round_ = prepared_round(project.id, "role-lead")
    supported = [row["dimension"] for candidate in round_["candidates"]
                 for row in candidate["supported_traits"]]
    inferred = [row["dimension"] for candidate in round_["candidates"]
                for row in candidate["inferred_traits"]]
    assert set(supported) == {"presentation"}
    assert inferred and not set(inferred) & set(supported)
    for candidate in round_["candidates"]:
        assert candidate["anchor"]["anchored"] is True
        assert candidate["conditioning"], "a sparse character is still acoustically conditioned"
        assert all(row["evidence_ids"] == [] for row in candidate["inferred_traits"])
    assert len({candidate["concept"]["key"] for candidate in round_["candidates"]}) == \
        len(round_["candidates"])


def test_the_generation_request_receives_acoustic_values_not_only_a_performer_name(monkeypatch, tmp_path):
    _root, project, _role = build_studio(monkeypatch, tmp_path)
    round_ = prepared_round(project.id, "role-lead")
    provider = RecordingProvider()
    generate(project.id, round_, provider)

    assert len(provider.requests) == 1
    request = provider.requests[0]
    conditioning = request["conditioning"]
    assert conditioning, "the request must carry conditioning"
    parameters = conditioning["parameters"]
    numbers = [value for value in parameters.values() if isinstance(value, (int, float))]
    assert numbers, "conditioning must be numeric, not prose"
    assert parameters["pitch_target_hz"] and parameters["rate_words_per_minute"]
    assert conditioning["target_profile"]["f0_median_hz"]
    assert conditioning["characterization"] in {"measured", "inferred"}
    # The performer's name is provenance, and the numbers are the instruction.
    assert conditioning["reference"]["performer"]
    assert parameters["pitch_target_hz"] is not None
    assert "Hz" in request["instruction"] and "words per minute" in request["instruction"]


def test_a_blocked_candidate_is_a_failure_state_and_cannot_be_generated(monkeypatch, tmp_path):
    """Production never falls back to an unanchored generic voice."""
    _root, project, _role = build_studio(monkeypatch, tmp_path)
    round_ = prepared_round(project.id, "role-lead")
    blocked = [dict(candidate, blocked=True, anchor={"anchored": False,
               "unanchored_reason": casting_anchors.UNANCHORED_REASON},
               conditioning=None, reference=None) for candidate in round_["candidates"]]
    with monkeypatch.context() as patch:
        patch.setattr(casting, "_proposal_plan", lambda *a, **k: [{
            "kind": "generated", "rationale": casting_anchors.UNANCHORED_REASON,
            "design_instruction": casting_anchors.UNANCHORED_REASON,
            "anchor": blocked[0]["anchor"], "inspiration": None, "tags": [], "source_urls": [],
            "concept": None, "reference": None, "conditioning": None,
            "supported": [], "inferred": [], "blocked": True,
            "blocked_reason": casting_anchors.UNANCHORED_REASON}])
        patch.setattr(casting, "MAX_CANDIDATES", 1)
        fresh = casting._new_round(load_project(project.id), "role-lead")
    candidate = fresh["candidates"][0]
    assert candidate["blocked"] is True
    assert candidate["blocked_reason"] == casting_anchors.UNANCHORED_REASON

    project = load_project(project.id)
    project.voice_lab.shortlists["role-lead"] = {"matches": [{
        "candidate_id": candidate["candidate_id"], "anchor": candidate["anchor"]}]}
    save_project(project)
    role = next(item for item in project.roles if item.id == "role-lead")
    expected = role.voice_identity.id if role.voice_identity else voice_revision_for_role(role)
    with pytest.raises(Exception, match="no performer reference|unanchored"):
        lab.design(project.id, "role-lead", candidate["candidate_id"], expected, "blocked-design",
                   provider=RecordingProvider())


def test_three_candidates_for_one_character_differ_in_casting_dimension(monkeypatch, tmp_path):
    _root, project, _role = build_studio(monkeypatch, tmp_path)
    round_ = prepared_round(project.id, "role-lead")
    candidates = [candidate for candidate in round_["candidates"] if not candidate["prior_match"]]
    assert len(candidates) >= 3
    assert len({candidate["concept"]["key"] for candidate in candidates}) == len(candidates)
    assert len({candidate["anchor"]["performer"] for candidate in candidates}) == len(candidates)
    # The differences are the dimensions a human rating can actually speak to.
    emphases = {candidate["concept"]["emphasis"] for candidate in candidates}
    assert len(emphases) == len(candidates)
    measured = {tuple(sorted((row["dimension"], row["value"]) for row in candidate["inferred_traits"]))
                for candidate in candidates}
    assert len(measured) == len(candidates)


def test_the_stored_candidate_keeps_the_characterization_derived_from_its_reference(monkeypatch, tmp_path):
    _root, project, _role = build_studio(monkeypatch, tmp_path)
    round_ = prepared_round(project.id, "role-lead")
    for candidate in round_["candidates"]:
        if candidate["prior_match"]:
            continue
        reference = candidate["reference"]
        assert reference["characterization"] in {"measured", "inferred"}
        assert reference["characterization_method"]
        profile = candidate["conditioning"]["target_profile"]
        keys = set(reference["measured_properties"]) | set(reference["inferred_properties"])
        assert keys <= set(profile), "every characterization property is present on the target profile"
        # Measured and inferred properties are never conflated.
        assert not set(reference["measured_properties"]) & set(reference["inferred_properties"])


def test_a_measured_reference_is_labelled_measured(monkeypatch, tmp_path):
    _root, project, _role = build_studio(monkeypatch, tmp_path)
    anchors = casting_anchors.anchored_plan(
        project.character_cast.profiles["role-lead"], None, count=1)
    anchor = anchors[0]
    audio = make_tone_wav(tmp_path / "permitted-reference.wav", seconds=2.0, frequency=118.0)
    evidence = performer_reference.resolve(anchor, anchor["concept"], audio_path=audio,
                                           retrieval_state={"state": "measured",
                                                            "source_type": "eligible acquisition"})
    assert evidence["characterization"]["state"] == "measured"
    assert "f0_median_hz" in evidence["characterization"]["measured"]
    assert evidence["conditioning"]["parameters"]["pitch_target_hz"] > 0


def test_a_fresh_round_in_a_real_shaped_project_carries_the_whole_chain(monkeypatch, tmp_path):
    """The persisted candidate, not just the in-memory proposal, carries the chain.

    A project whose profile uses the shapes real dossiers produce (an `EXPLICIT`
    cadence, an `INFERRED` presentation, an unsupported pitch placeholder) must
    still yield candidates that each record a performer, a reference, a labelled
    characterization and the numeric conditioning the design request will receive.
    """
    decisions = (
        CastingDecision(dimension="cadence", value="deliberate", reason="text evidence",
                        level="EXPLICIT", evidence_ids=["fact-cadence"]),
        CastingDecision(dimension="presentation", value="unspecified", reason="not established",
                        level="UNKNOWN"),
        CastingDecision(dimension="pitch", value="mid", reason="placeholder", level="INFERRED"),
    )
    _root, project, _role = build_studio(monkeypatch, tmp_path, decisions=decisions,
                                         name="Real-shaped dossier")
    round_ = prepared_round(project.id, "role-lead")
    assert len(round_["candidates"]) == 3
    performers = set()
    for candidate in round_["candidates"]:
        assert candidate["provenance"] == "generated"
        assert candidate["blocked"] is False and candidate["blocked_reason"] is None
        anchor = candidate["anchor"]
        assert anchor["anchored"] is True and anchor["performer"] and anchor["performance"]
        performers.add(anchor["performer"])
        reference = candidate["reference"]
        assert reference["characterization"] in {"measured", "inferred"}
        assert reference["characterization_method"]
        assert reference["rationale"]
        parameters = candidate["conditioning"]["parameters"]
        numeric = [v for v in parameters.values() if isinstance(v, (int, float)) and v is not None]
        assert len(numeric) >= 8, "the conditioning must be numeric, not prose"
        # The book's own two dimensions are reported as book-supported, the
        # placeholder is not, and nothing inferred claims an evidence id.
        supported = {row["dimension"]: row for row in candidate["supported_traits"]}
        assert "cadence" in supported and "pitch" not in supported
        assert all(row["evidence_ids"] == [] and row["level"] == casting_inference.INFERENCE_LEVEL
                   for row in candidate["inferred_traits"])
        assert "presentation" in {row["dimension"] for row in candidate["inferred_traits"]}, \
            "the dimension the book left unknown must be auditioned as inference"
    assert len(performers) == 3, "one round must not offer the same performer three times"


def test_fresh_candidates_complement_a_prior_voice_instead_of_duplicating_it():
    """A historical voice already answers its own dimensions, so order around them."""
    from dubbing_service.app.audiobook import casting_inference

    priors = [{"kind": "prior_voice", "performer": None,
               "prior_evidence": {"match_reasons": ["accent: British",
                                                    "perceived vocal age: older adult"]}}]
    proposals = casting._generated_proposals(None, None, None, None, "Create an original voice.", "",
                                             count=3, priors=priors)
    assert len(proposals) == 3
    overlaps = [len(set(row["concept"]["inferred"]) & {"accent", "perceived_vocal_age"})
                for row in proposals]
    assert overlaps == sorted(overlaps), "the least overlapping concept must be offered first"
    assert len({row["concept"]["key"] for row in proposals}) == 3
    assert len({row["anchor"]["performer"] for row in proposals}) == 3
    # The prior voice's own performer is never reused for a fresh candidate.
    assert all(row["anchor"]["anchored"] for row in proposals)
    assert all(casting_inference.INFERENCE_LEVEL in
               {trait["level"] for trait in row["inferred"]} for row in proposals)
