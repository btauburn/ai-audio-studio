"""M4 direction resolution: precedence, dispositions, bounds and safeguards.

The capability records come from the real M1 owner, read from a synthetic
repository tree under ``tmp_path`` so that the default and experimental profiles
can both be exercised without a model, a worker or a GPU.  The resolver itself is
pure and reads nothing.
"""

from __future__ import annotations

import hashlib
import socket
import sys
from pathlib import Path

import pytest

from dubbing_service.app.audiobook import performance_director as director
from dubbing_service.app.audiobook.fingerprint import text_sha256
from dubbing_service.app.audiobook.provider_capabilities import performance_capabilities
from dubbing_service.app.models.project import SourceAnchor
from dubbing_service.app.storage.segments import save_segments
from tests.audiobook_helpers import build_project, build_segments, use_projects_root

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_REVISION = "c1" * 32

DEFAULT_PROFILE = "faster-base-032"
EXPERIMENTAL_PROFILE = "faster-base-032-direction-x1"

SITE_PACKAGES = Path("environments/faster_qwen_base032/.venv/Lib/site-packages")
MODEL_PY = SITE_PACKAGES / "faster_qwen3_tts" / "model.py"
GENERATE_PY = SITE_PACKAGES / "faster_qwen3_tts" / "generate.py"
DIST_INFO = SITE_PACKAGES / "faster_qwen3_tts-0.3.2.dist-info" / "METADATA"
BASE_ADAPTER = Path("service/faster_qwen3tts_base_adapter.py")
DIRECTION_ADAPTER = Path("service/faster_qwen3tts_base_direction_adapter.py")

SIGNATURE_WITH_INSTRUCT = (
    "class Model:\n"
    "    def _prepare_generation(self, text: str, ref_audio=None,\n"
    "                            instruct: Optional[str] = None):\n"
    "        return text, instruct\n"
)

DELIVERY_PHRASE = "Measured and conversational"


def _write(root: Path, relative: Path, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def capability_tree(tmp_path: Path, *, direction_adapter: bool) -> Path:
    """A synthetic repository tree the real capability owner can read."""

    root = tmp_path / "capability-repo"
    _write(root, MODEL_PY, SIGNATURE_WITH_INSTRUCT)
    _write(root, GENERATE_PY, "# generation entry point\n")
    _write(root, DIST_INFO, "Metadata-Version: 2.1\nName: faster-qwen3-tts\nVersion: 0.3.2\n")
    _write(root, BASE_ADAPTER, "base adapter without the directive path\n")
    if direction_adapter:
        _write(root, DIRECTION_ADAPTER, "# experimental direction adapter\n")
    return root


def reference_identity() -> dict:
    return {"mechanism": "qwen_reference", "clone_mode": "icl"}


def synthetic_capability(*, seed: str = "unsupported", native: str = "unsupported") -> dict:
    """A clearly synthetic record, used only to reach states no real profile reports."""

    controls = {
        name: {"state": "unsupported", "stage": "native", "review_copy_eligible": False,
               "reason": "synthetic fixture"}
        for name in ("native_instruction", "emotion_label", "intensity", "emphasis",
                     "role_delivery", "pace", "volume", "numeric_rate", "seed")
    }
    controls["native_instruction"]["state"] = native
    controls["seed"]["state"] = seed
    record = {"schema_version": "performance-capability-v1",
              "runtime_profile": "faster-base-032-direction-x1", "backend": "qwen3tts",
              "identity_mechanism": "qwen_reference", "conditioning_mode": "icl",
              "evidence": {"fixture": "synthetic"}, "controls": controls}
    record["sha256"] = hashlib.sha256(
        director.canonical_json_bytes(record)).hexdigest()
    return record


@pytest.fixture()
def project(tmp_path, monkeypatch):
    use_projects_root(monkeypatch, tmp_path)
    built = build_project(name="resolver-book")
    segments = build_segments(built, ("First line.", "Second line.", "Third line."),
                              role_id="narrator")
    for segment in segments:
        segment.source_anchor = SourceAnchor(source_revision_id=SOURCE_REVISION)
    save_segments(built.id, segments)
    from dubbing_service.app.storage.projects import load_project
    from dubbing_service.app.storage.segments import load_segments

    return {"project": load_project(built.id), "segments": load_segments(built.id)}


def model_provenance(mode: str) -> dict:
    """A model-mode interpretation must carry its full provenance."""

    if mode != "model":
        return {}
    return {"model": "synthetic-planner", "prompt_sha256": "d1" * 32,
            "response_sha256": "d2" * 32, "budget_reservation": "reservation-1"}


def scene(project, segments, *, turns=(), scene_id="scene-1", planner="pd-planner-1",
          mode="manual", chapter_id=None) -> director.SceneIntention:
    studio = project
    project = studio["project"] if isinstance(studio, dict) else studio
    utterances = [director.SceneUtterance(id=segment.id, text_sha256=text_sha256(segment.text),
                                          role_id=segment.role_id) for segment in segments]
    proofs = director.scene_proofs(project, [segment.id for segment in segments])
    interpretation = director.SceneInterpretation(mode=mode, planner_version=planner,
                                                  **model_provenance(mode))
    return director.SceneIntention(
        scene_id=scene_id,
        chapter_id=chapter_id or (segments[0].chapter_id or "chap0"),
        source_revision_sha256=proofs["source_revision_sha256"],
        utterances=utterances,
        attribution_digest=proofs["attribution_digest"],
        alias_digest=proofs["alias_digest"],
        turns=list(turns),
        interpretation=interpretation,
        cache_key=director.scene_cache_key(
            source_revision_sha256=proofs["source_revision_sha256"], utterances=utterances,
            attribution_digest=proofs["attribution_digest"], alias_digest=proofs["alias_digest"],
            interpretation=interpretation),
        author=director.RecordAuthor(origin="manual", actor="synthetic", event_id="evt-scene"),
    )


def profile_with(*, narrator=False, values=("unhurried, complete clauses",)) -> director.DeliveryProfile:
    return director.compile_profile("narrator", narrator=narrator).model_copy(
        update={"habits": [director.DeliveryHabit(dimension="cadence", value=value,
                                                  level="SUPPORTED", origin="dossier")
                           for value in values]})


def default_capability(tmp_path: Path) -> dict:
    return performance_capabilities(DEFAULT_PROFILE, reference_identity(),
                                    repo_root=capability_tree(tmp_path, direction_adapter=False))


def experimental_capability(tmp_path: Path) -> dict:
    return performance_capabilities(EXPERIMENTAL_PROFILE, reference_identity(),
                                    repo_root=capability_tree(tmp_path, direction_adapter=True))


def _guard_against_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the resolver must not open a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


# --------------------------------------------------------------------- capability


def test_the_default_profile_resolves_every_native_control_to_advisory(project, tmp_path) -> None:
    capability = default_capability(tmp_path)
    assert capability["controls"]["native_instruction"]["state"] == "available_not_forwarded"
    plan = scene(project, project["segments"])
    rows = director.resolve_directions(profile_with(), plan, capability,
                                       policy=director.DirectionPolicy())
    assert len(rows) == 3
    for row in rows:
        states = {disposition.control: disposition.state for disposition in row.dispositions}
        assert states["native_instruction"] == "advisory"
        assert row.resolved.native_instruction is None
        assert row.resolved.seed_salt == 0
        assert director.direction_component(row) is None, (
            "the default profile must put nothing into a fingerprint")


def test_the_experimental_profile_forwards_a_bounded_instruction(project, tmp_path) -> None:
    capability = experimental_capability(tmp_path)
    assert capability["controls"]["native_instruction"]["state"] == "forwarded_experimental"
    plan = scene(project, project["segments"][:1])
    rows = director.resolve_directions(profile_with(), plan, capability,
                                       policy=director.DirectionPolicy())
    row = rows[0]
    states = {disposition.control: disposition.state for disposition in row.dispositions}
    assert states["native_instruction"] == "forward_native"
    assert row.resolved.native_instruction == "unhurried, complete clauses."
    assert director.direction_component(row) is not None


def test_narrator_preservation_prohibits_forwarding(project, tmp_path) -> None:
    capability = experimental_capability(tmp_path)
    plan = scene(project, project["segments"][:1])
    rows = director.resolve_directions(profile_with(narrator=True), plan, capability,
                                       policy=director.DirectionPolicy())
    row = rows[0]
    assert row.resolved.native_instruction is None
    assert director.direction_component(row) is None
    states = {disposition.control: disposition.state for disposition in row.dispositions}
    assert states["native_instruction"] == "advisory"
    assert "narrator preservation" in row.dispositions[0].reason


def test_the_policy_cannot_permit_narrator_forwarding() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        director.DirectionPolicy(narrator_forwarding=True)
    with pytest.raises(director.PerformanceDirectorError):
        director.resolve_directions(None, None, synthetic_capability(),
                                    policy={"schema_version": "direction-policy-v1",
                                            "narrator_forwarding": True})
    assert director.DirectionPolicy().narrator_forwarding is False


def test_an_unknown_capability_schema_is_refused(project) -> None:
    plan = scene(project, project["segments"][:1])
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.resolve_directions(None, plan, {"schema_version": "mystery", "controls": {}},
                                    policy=director.DirectionPolicy())
    assert refused.value.code == "schema_unknown"


def test_a_missing_control_is_never_optimistic(project) -> None:
    plan = scene(project, project["segments"][:1])
    capability = synthetic_capability()
    capability["controls"].pop("native_instruction")
    capability["sha256"] = director.sha256_bytes(director.canonical_json_bytes({k: v for k, v in capability.items() if k != "sha256"}))
    rows = director.resolve_directions(profile_with(), plan, capability,
                                       policy=director.DirectionPolicy())
    states = {disposition.control: disposition.state for disposition in rows[0].dispositions}
    assert states["native_instruction"] == "unsupported"
    assert director.direction_component(rows[0]) is None


# -------------------------------------------------------------------- precedence


def test_exactly_one_disposition_per_requested_control(project, tmp_path) -> None:
    capability = experimental_capability(tmp_path)
    plan = scene(project, project["segments"][:1],
                 turns=[director.SceneTurn(utterance_id=project["segments"][0].id,
                                           intention="reassure", confidence="manual",
                                           boundary_hint_after="quick")])
    policy = director.DirectionPolicy(
        line_overrides={project["segments"][0].id: director.DirectionPolicyOverride(
            delivery=[DELIVERY_PHRASE], emotion_label="sad", seed_salt=17)})
    row = director.resolve_directions(profile_with(), plan, capability, policy=policy)[0]
    controls = [disposition.control for disposition in row.dispositions]
    assert len(controls) == len(set(controls))
    assert set(controls) == {"native_instruction", "pace_hint", "emotion_label", "seed_salt"}
    assert set(row.requested) <= set(controls) | {"intention", "delivery"}


def test_manual_line_correction_outranks_the_plan_and_the_profile(project, tmp_path) -> None:
    capability = experimental_capability(tmp_path)
    segment = project["segments"][0]
    plan = scene(project, [segment],
                 turns=[director.SceneTurn(utterance_id=segment.id, intention="deflect",
                                           confidence="manual")])
    policy = director.DirectionPolicy(line_overrides={segment.id:
        director.DirectionPolicyOverride(delivery=["Clipped and quick"], intention="concede")})
    row = director.resolve_directions(profile_with(), plan, capability, policy=policy)[0]
    assert row.requested["intention"] == "concede"
    assert row.resolved.native_instruction == "Clipped and quick."
    assert "manual_line" in row.precedence_trace
    assert row.precedence_trace.index("manual_line") < len(row.precedence_trace)


def test_human_scene_correction_outranks_an_inferred_intention(project, tmp_path) -> None:
    capability = experimental_capability(tmp_path)
    segment = project["segments"][0]
    manual = scene(project, [segment],
                   turns=[director.SceneTurn(utterance_id=segment.id, intention="deflect",
                                             confidence="manual")])
    inferred = scene(project, [segment],
                     turns=[director.SceneTurn(utterance_id=segment.id, intention="concede",
                                               confidence="model")], mode="model")
    manual_row = director.resolve_directions(None, manual, capability,
                                             policy=director.DirectionPolicy())[0]
    inferred_row = director.resolve_directions(None, inferred, capability,
                                               policy=director.DirectionPolicy())[0]
    assert manual_row.requested["intention"] == "deflect"
    assert "human_scene" in manual_row.precedence_trace
    assert inferred_row.requested["intention"] == "concede"
    assert "inferred_intention" in inferred_row.precedence_trace
    assert "human_scene" not in inferred_row.precedence_trace


def test_an_inferred_intention_never_becomes_explicit_or_approved(project, tmp_path) -> None:
    capability = experimental_capability(tmp_path)
    segment = project["segments"][0]
    plan = scene(project, [segment],
                 turns=[director.SceneTurn(utterance_id=segment.id, intention="conceal",
                                           confidence="model")], mode="model")
    row = director.resolve_directions(profile_with(), plan, capability,
                                      policy=director.DirectionPolicy())[0]
    assert "inferred_intention" in row.precedence_trace
    assert "manual_line" not in row.precedence_trace
    assert "human_scene" not in row.precedence_trace
    # An inferred intention alone never produces delivery priorities: the profile
    # does, and it is traced as the approved profile.
    assert "approved_profile" in row.precedence_trace


def test_an_unknown_intention_resolves_to_baseline(project, tmp_path) -> None:
    capability = experimental_capability(tmp_path)
    row = director.resolve_directions(None, scene(project, project["segments"][:1]),
                                      capability, policy=director.DirectionPolicy())[0]
    assert row.requested["intention"] == "unknown"
    assert "baseline" in row.precedence_trace
    assert row.resolved.native_instruction is None
    assert row.dispositions == []


# ------------------------------------------------------------------------- holds


def test_a_contradiction_is_held_and_never_blended(project, tmp_path) -> None:
    capability = experimental_capability(tmp_path)
    plan = scene(project, project["segments"][:1])
    rows = director.resolve_directions(profile_with(values=("slightly faster", "slower")),
                                       plan, capability, policy=director.DirectionPolicy())
    row = rows[0]
    assert row.resolved.native_instruction is None, "a contradiction is never blended"
    states = {disposition.control: disposition.state for disposition in row.dispositions}
    assert states["native_instruction"] == "hold"
    assert "contradictory" in row.dispositions[0].reason


def test_a_required_unsupported_control_is_held(project, tmp_path) -> None:
    capability = default_capability(tmp_path)
    segment = project["segments"][0]
    plan = scene(project, [segment])
    policy = director.DirectionPolicy(line_overrides={segment.id:
        director.DirectionPolicyOverride(delivery=[DELIVERY_PHRASE], require_forward=True)})
    row = director.resolve_directions(profile_with(), plan, capability, policy=policy)[0]
    states = {disposition.control: disposition.state for disposition in row.dispositions}
    assert states["native_instruction"] == "hold"
    assert row.dispositions[0].reason.startswith("capability_unsupported")
    assert row.resolved.native_instruction is None
    assert director.direction_component(row) is None


def test_an_overlong_instruction_is_held_and_never_truncated(project, tmp_path) -> None:
    capability = experimental_capability(tmp_path)
    segment = project["segments"][0]
    plan = scene(project, [segment])
    # Two bounded priorities whose composition exceeds the 240-character policy
    # limit: 120 + "; " + 120 + "." is 243 characters.
    policy = director.DirectionPolicy(line_overrides={segment.id:
        director.DirectionPolicyOverride(delivery=["a" * 120, "b" * 120])})
    row = director.resolve_directions(None, plan, capability, policy=policy)[0]
    states = {disposition.control: disposition.state for disposition in row.dispositions}
    assert states["native_instruction"] == "hold"
    assert "never truncated" in row.dispositions[0].reason
    assert row.resolved.native_instruction is None


def test_the_instruction_bound_is_exactly_two_hundred_and_forty_characters(project, tmp_path) -> None:
    capability = experimental_capability(tmp_path)
    segment = project["segments"][0]
    plan = scene(project, [segment])

    def resolve(phrases):
        policy = director.DirectionPolicy(line_overrides={segment.id:
            director.DirectionPolicyOverride(delivery=list(phrases))})
        return director.resolve_directions(None, plan, capability, policy=policy)[0]

    # 119 + "; " + 118 + "." is exactly 240 characters and is accepted.
    accepted = resolve(["a" * 119, "b" * 118])
    assert accepted.resolved.native_instruction == "a" * 119 + "; " + "b" * 118 + "."
    assert len(accepted.resolved.native_instruction) == 240
    # One character more is held, not truncated.
    held = resolve(["a" * 119, "b" * 119])
    assert {d.control: d.state for d in held.dispositions}["native_instruction"] == "hold"
    assert "over the policy limit of 240" in held.dispositions[0].reason
    assert held.resolved.native_instruction is None

    # A policy may be stricter, never wider.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        director.DirectionPolicy(max_instruction_chars=241)
    stricter = director.DirectionPolicy(max_instruction_chars=10)
    policy = director.DirectionPolicy(max_instruction_chars=10,
                                      line_overrides={segment.id:
                                          director.DirectionPolicyOverride(
                                              delivery=[DELIVERY_PHRASE])})
    row = director.resolve_directions(None, plan, capability, policy=policy)[0]
    assert {d.control: d.state for d in row.dispositions}["native_instruction"] == "hold"
    assert stricter.max_instruction_chars == 10


def test_more_than_three_priorities_are_held(project, tmp_path) -> None:
    capability = experimental_capability(tmp_path)
    plan = scene(project, project["segments"][:1])
    four = ("measured", "warm", "articulate", "restrained")
    row = director.resolve_directions(profile_with(values=four), plan, capability,
                                      policy=director.DirectionPolicy())[0]
    assert {d.control: d.state for d in row.dispositions}["native_instruction"] == "hold"
    assert "exceed the policy limit" in row.dispositions[0].reason

    three = four[:3]
    row = director.resolve_directions(profile_with(values=three), plan, capability,
                                      policy=director.DirectionPolicy())[0]
    assert {d.control: d.state for d in row.dispositions}["native_instruction"] == "forward_native"


def test_a_performer_reference_voice_change_or_numeric_rate_is_refused(project, tmp_path) -> None:
    capability = experimental_capability(tmp_path)
    segment = project["segments"][0]
    plan = scene(project, [segment])
    for text in ("sound like Morgan Freeman", "in the style of a movie trailer voice",
                 "speak 20% faster", "read this at 180 wpm", "clone the original narrator"):
        policy = director.DirectionPolicy(line_overrides={segment.id:
            director.DirectionPolicyOverride(delivery=[text])})
        with pytest.raises(director.PerformanceDirectorError) as refused:
            director.resolve_directions(None, plan, capability, policy=policy)
        assert refused.value.code == "direction_conflict", text


def test_a_forbidden_term_from_the_policy_is_refused(project, tmp_path) -> None:
    capability = experimental_capability(tmp_path)
    segment = project["segments"][0]
    plan = scene(project, [segment])
    policy = director.DirectionPolicy(forbidden_terms=["narrator"],
                                      line_overrides={segment.id:
                                          director.DirectionPolicyOverride(
                                              delivery=["Measured narrator warmth"])})
    with pytest.raises(director.PerformanceDirectorError) as refused:
        director.resolve_directions(None, plan, capability, policy=policy)
    assert refused.value.code == "direction_conflict"


def test_legacy_labels_never_become_a_native_instruction(project, tmp_path) -> None:
    capability = experimental_capability(tmp_path)
    segment = project["segments"][0]
    plan = scene(project, [segment])
    policy = director.DirectionPolicy(line_overrides={segment.id:
        director.DirectionPolicyOverride(delivery=[DELIVERY_PHRASE], emotion_label="angry",
                                         intensity="1.0", emphasis="strong",
                                         role_delivery="shouting", numeric_rate="1.5x")})
    row = director.resolve_directions(None, plan, capability, policy=policy)[0]
    states = {disposition.control: disposition.state for disposition in row.dispositions}
    for control in ("emotion_label", "intensity", "emphasis", "role_delivery", "numeric_rate"):
        assert states[control] in ("unsupported", "hold", "advisory"), control
    # The legacy values are recorded as requests and never enter the instruction.
    for legacy in ("angry", "1.0", "strong", "shouting", "1.5x"):
        assert legacy not in (row.resolved.native_instruction or "")
    assert row.resolved.native_instruction == DELIVERY_PHRASE + "."

    # And Segment.emotion_text is not an input at all.
    segment.emotion_text = "whispering"
    again = director.resolve_directions(None, plan, capability, policy=policy)[0]
    assert again.resolved.native_instruction == row.resolved.native_instruction


def test_the_seed_salt_is_explicit_and_never_a_hidden_default(project) -> None:
    segment = project["segments"][0]
    plan = scene(project, [segment])
    # Every real profile reports the seed as unsupported, so a requested salt is
    # not forwarded and cannot enter a fingerprint by accident.
    for capability in (synthetic_capability(), synthetic_capability(native="available")):
        policy = director.DirectionPolicy(line_overrides={segment.id:
            director.DirectionPolicyOverride(seed_salt=12345)})
        row = director.resolve_directions(None, plan, capability, policy=policy)[0]
        assert row.resolved.seed_salt == 0
        assert director.direction_component(row) is None
        assert {d.control: d.state for d in row.dispositions}["seed_salt"] == "unsupported"

    # An explicit salt is forwarded only where the runtime really accepts one.
    supported = synthetic_capability(seed="forwarded_experimental")
    policy = director.DirectionPolicy(line_overrides={segment.id:
        director.DirectionPolicyOverride(seed_salt=12345)})
    row = director.resolve_directions(None, plan, supported, policy=policy)[0]
    assert row.resolved.seed_salt == 12345
    assert row.resolved.native_instruction is None
    assert director.direction_component(row) is not None

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        director.DirectionPolicyOverride(seed_salt=2 ** 31)


def test_every_row_records_its_own_sources(project, tmp_path) -> None:
    capability = experimental_capability(tmp_path)
    profile = profile_with()
    plan = scene(project, project["segments"])
    rows = director.resolve_directions(profile, plan, capability,
                                       policy=director.DirectionPolicy())
    for row in rows:
        assert row.sources.capability_sha256 == capability["sha256"]
        assert row.sources.profile_sha256 == director.content_sha256(profile)
        assert row.sources.scene_plan_sha256 == director.content_sha256(plan)
        assert row.sources.resolver_version == director.RESOLVER_VERSION
        assert len(row.precedence_trace) <= 10


def test_a_set_built_from_a_scene_binds_the_project_and_chapter(project, tmp_path) -> None:
    project["project"].roles[0].role_type = "character"
    capability = experimental_capability(tmp_path)
    plan = scene(project, project["segments"])
    built = director.build_direction_set(project["project"], profile_with(), plan, capability,
                                         policy=director.DirectionPolicy())
    assert built.project_id == project["project"].id
    assert built.chapter_id == plan.chapter_id
    assert len(built.directions) == 3


def test_resolving_without_a_plan_or_a_profile_yields_nothing() -> None:
    assert director.resolve_directions(None, None, synthetic_capability(),
                                       policy=director.DirectionPolicy()) == []
    assert director.resolve_directions(profile_with(), None, synthetic_capability(),
                                       policy=director.DirectionPolicy()) == []


def _model_modules_imported_by(action) -> list[str]:
    """Modules a call pulled in that load a model, a provider or the renderer.

    Comparing before and after keeps the check about this call rather than about
    whatever else the pytest process already imported.
    """

    watched = ("torch", "dubbing_service.app.audiobook.renderer",
               "dubbing_service.app.providers.tts")
    before = {name for name in sys.modules if name.startswith(watched)}
    action()
    after = {name for name in sys.modules if name.startswith(watched)}
    return sorted(after - before)


def test_the_resolver_constructs_no_http_client_or_model(project, tmp_path, monkeypatch) -> None:
    _guard_against_network(monkeypatch)
    capability = experimental_capability(tmp_path)
    plan = scene(project, project["segments"])

    def resolve():
        return director.resolve_directions(profile_with(), plan, capability,
                                           policy=director.DirectionPolicy())

    rows = resolve()
    assert len(rows) == 3
    assert _model_modules_imported_by(resolve) == [], (
        "resolving a direction must not import torch, the renderer or the provider package")
