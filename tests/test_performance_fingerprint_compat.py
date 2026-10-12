"""M4 fingerprint compatibility: the golden maps, exact omission and rollback.

The golden values are the ones Codex captured before M4 from real stored Take
component maps (``m4/fingerprint-baseline.json``) and from one synthetic
runtime-bound vector (``m4/synthetic-runtime-baseline.json``).  They are read
from the evidence folder, never regenerated here, and the preserved original
``fingerprint.py``/``pipeline.py`` bytes in ``m4/baseline-source/`` are loaded as
a second module so the "before" value comes from the original implementation
rather than from the changed one.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from dubbing_service.app.audiobook import performance_director as director
from dubbing_service.app.audiobook import fingerprint as fingerprint_module
from dubbing_service.app.audiobook import pipeline
from dubbing_service.app.audiobook.fingerprint import (
    canonical_json_bytes,
    sha256_bytes,
    text_sha256,
    voice_revision_for_role,
)
from dubbing_service.app.models.project import SourceAnchor, Role
from dubbing_service.app.storage.projects import load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audiobook_helpers import build_project, build_segments, use_projects_root

REPO_ROOT = Path(__file__).resolve().parents[1]
#: The pre-M4 captures live in the evidence folder beside the repository.  They are
#: read, never regenerated; a checkout without them reports the skip explicitly.
EVIDENCE_ROOT = REPO_ROOT.parent / "runtime_evidence" / "2026-10-06-performance-director-spec"
M4 = EVIDENCE_ROOT / "m4"
BASELINE_SOURCE = M4 / "baseline-source"
SOURCE_REVISION = "c1" * 32


def _require_evidence() -> None:
    if not (M4 / "fingerprint-baseline.json").is_file() or not BASELINE_SOURCE.is_dir():
        pytest.skip(f"the pre-M4 baseline captures are absent from {M4}")


def _load_baseline_module(name: str):
    """Import the preserved pre-M4 source as its own module.

    The module is registered in ``sys.modules`` before execution: its
    annotations are postponed (``from __future__ import annotations``) and
    ``models/project`` imports a class that needs to resolve it by name.
    """

    path = BASELINE_SOURCE / f"{name}.py"
    assert path.is_file(), f"the preserved baseline source is missing: {path}"
    module_name = f"m4_baseline_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


@pytest.fixture()
def studio(tmp_path, monkeypatch):
    use_projects_root(monkeypatch, tmp_path)
    built = build_project(name="fingerprint-book")
    built.roles[0].role_type = "character"
    save_project(built)
    # CPU-only positive seam; unlike production this fixture has no real worker.
    capability = {"schema_version": "performance-capability-v1", "runtime_profile": "faster-base-032-direction-x1",
                  "identity_mechanism": "qwen_reference", "conditioning_mode": "icl", "backend": "qwen3tts",
                  "evidence": {"fixture": "fingerprint seam only"},
                  "controls": {"seed": {"state": "forwarded_experimental"}, "native_instruction": {"state": "forwarded_experimental"}}}
    capability["sha256"] = sha256_bytes(canonical_json_bytes(capability))
    monkeypatch.setattr(director, "_live_capability", lambda project, segment: dict(capability))
    segments = build_segments(built, ("First line.", "Second line.", "Third line."),
                              role_id="narrator")
    for index, segment in enumerate(segments):
        segment.source_anchor = SourceAnchor(source_revision_id=SOURCE_REVISION)
        segment.index = index
    save_segments(built.id, segments)
    director.clear_direction_cache()
    yield {"project": load_project(built.id), "segments": load_segments(built.id),
           "project_id": built.id}
    director.clear_direction_cache()


def _row(segment, *, salt, capability_sha):
    return director.UtteranceDirection(
        utterance_id=segment.id, text_sha256=text_sha256(segment.text),
        role_id=segment.role_id or "",
        voice_revision=voice_revision_for_role(Role(id=segment.role_id, name='Narrator')),
        resolved=director.ResolvedControls(seed_salt=salt),
        sources=director.DirectionSources(capability_sha256=capability_sha,
                                          resolver_version=director.RESOLVER_VERSION))


def _live_sha(studio, segment) -> str:
    return director._live_capability(studio["project"], segment)["sha256"]


def _activate(studio, rows, *, expected_revision=0, event_id="evt-directions"):
    stored = director.write_direction_set(
        studio["project_id"],
        director.DirectionSet(project_id=studio["project_id"], directions=list(rows)))

    def change(index: dict) -> None:
        index["enabled"] = True
        index["active"]["direction_set"] = stored["sha256"]

    director.commit_index(studio["project_id"], expected_revision, change, actor="synthetic",
                          event_id=event_id)
    studio["project"] = load_project(studio["project_id"])
    director.clear_direction_cache()
    return stored


# --------------------------------------------------------------- the golden maps


def test_the_preserved_baseline_sources_are_the_ones_the_goldens_came_from() -> None:
    _require_evidence()
    for name in ("fingerprint", "pipeline"):
        source = (BASELINE_SOURCE / f"{name}.py").read_bytes()
        current = (REPO_ROOT / "dubbing_service" / "app" / "audiobook" / f"{name}.py").read_bytes()
        assert hashlib.sha256(source).hexdigest() != hashlib.sha256(current).hexdigest(), (
            f"{name}.py should have changed in M4, otherwise this comparison proves nothing")
    capture = json.loads((M4 / "fingerprint-baseline.json").read_text(encoding="utf-8"))
    assert set(capture["baselineSourceHashes"]) == {
        "dubbing_service/app/audiobook/fingerprint.py",
        "dubbing_service/app/audiobook/pipeline.py"}
    for relative, digest in capture["baselineSourceHashes"].items():
        preserved = BASELINE_SOURCE / Path(relative).name
        assert hashlib.sha256(preserved.read_bytes()).hexdigest() == digest


def test_every_supplied_baseline_component_map_still_hashes_to_its_stored_fingerprint() -> None:
    _require_evidence()
    capture = json.loads((M4 / "fingerprint-baseline.json").read_text(encoding="utf-8"))
    assert capture["cases"], "the capture must carry at least one case"
    for name, case in capture["cases"].items():
        assert fingerprint_module.fingerprint_of_components(
            case["components"]) == case["storedFingerprint"], name
        assert "performance_direction" not in case["components"]
    synthetic = json.loads((M4 / "synthetic-runtime-baseline.json").read_text(encoding="utf-8"))
    assert fingerprint_module.fingerprint_of_components(
        synthetic["components"]) == synthetic["fingerprint"]
    assert synthetic["runtimeBinding"]["sha256"] in synthetic["components"]["runtime_version"]
    assert capture["counts"]["fingerprintMismatches"] == 0


def test_the_capture_records_which_source_bytes_produced_it() -> None:
    if not (M4 / "fingerprint-baseline.json").is_file():
        pytest.skip("Private pre-M4 capture is not part of the source checkout")
    capture = json.loads((M4 / "fingerprint-baseline.json").read_text(encoding="utf-8"))
    synthetic = json.loads((M4 / "synthetic-runtime-baseline.json").read_text(encoding="utf-8"))
    assert synthetic["baselineFingerprintSourceSha256"] == \
        capture["baselineSourceHashes"]["dubbing_service/app/audiobook/fingerprint.py"]
    assert "not a real worker" in synthetic["provenance"]


# ------------------------------------------------------------- exact omission


def test_the_component_is_absent_unless_something_is_forwarded() -> None:
    plain = fingerprint_module.SynthesisRequirement(utterance_id="seg-1")
    assert "performance_direction" not in plain.components()

    forwarded = fingerprint_module.SynthesisRequirement(utterance_id="seg-1",
                                                        performance_direction="a" * 64)
    assert forwarded.components()["performance_direction"] == "a" * 64
    assert forwarded.fingerprint() != plain.fingerprint()

    # An explicit None is the same as omitting it.
    explicit = fingerprint_module.build_requirement(utterance_id="seg-1", engine_id="qwen3tts",
                                                    spoken_text_hash="x",
                                                    performance_direction=None)
    assert "performance_direction" not in explicit.components()
    assert fingerprint_module.SynthesisRequirement(
        utterance_id="seg-1").components() == fingerprint_module.SynthesisRequirement(
            utterance_id="seg-1", performance_direction=None).components()


def test_the_component_is_audio_affecting_and_never_part_of_metadata() -> None:
    assert "performance_direction" in fingerprint_module.AUDIO_AFFECTING_COMPONENTS
    components = set(fingerprint_module.SynthesisRequirement(
        utterance_id="seg-1", performance_direction="a" * 64).components())
    assert components.isdisjoint(fingerprint_module.NON_AUDIO_METADATA_FIELDS)
    assert "performance_direction" in fingerprint_module.iter_requirement_fields()


# ------------------------------------------------- before and after, per line


def test_a_project_without_an_index_matches_the_preserved_original_implementation(studio) -> None:
    _require_evidence()
    """The strongest compatibility claim: the original bytes compute the same thing."""

    baseline = _load_baseline_module("fingerprint")
    segment = studio["segments"][0]
    role = studio["project"].roles[0]
    arguments = dict(project_id=studio["project_id"], engine_id="qwen3tts",
                     renderer_id="qwen3tts", role=role, spoken_text_hash=text_sha256(segment.text))

    before = baseline.requirement_for_segment(segment, **arguments)
    after = fingerprint_module.requirement_for_segment(segment, **arguments)
    assert before.components() == after.components()
    assert before.fingerprint() == after.fingerprint()
    assert "performance_direction" not in after.components()

    # And through the canonical funnel, with no index present.
    assert pipeline.requirement_for(studio["project"], segment, role=role,
                                    engine_id="qwen3tts",
                                    renderer_id="qwen3tts").fingerprint() == before.fingerprint()


def test_the_hook_is_the_only_place_the_component_is_attached(studio) -> None:
    segment = studio["segments"][0]
    role = studio["project"].roles[0]
    without = pipeline.requirement_for(studio["project"], segment, role=role)
    assert "performance_direction" not in without.components()

    _activate(studio, [_row(segment, salt=31, capability_sha=_live_sha(studio, segment))])
    with_index = pipeline.requirement_for(studio["project"], segment, role=role)
    assert with_index.components()["performance_direction"] == \
        director.requirement_direction(studio["project"], segment)
    assert with_index.fingerprint() != without.fingerprint()

    # Caller values are refused by the funnel; the lower-level fingerprint module
    # itself never reads a project, an index or a file.
    import inspect

    source = inspect.getsource(fingerprint_module.requirement_for_segment)
    assert "performance_direction" in source
    assert "metadata" not in source


def test_one_line_of_invalidation_and_exact_rollback(studio) -> None:
    segments = studio["segments"]
    role = studio["project"].roles[0]

    def fingerprints():
        return {segment.id: pipeline.requirement_for(studio["project"], segment,
                                                     role=role).fingerprint()
                for segment in segments}

    baseline = fingerprints()
    first = segments[0]
    _activate(studio, [_row(first, salt=5, capability_sha=_live_sha(studio, first))])
    after = fingerprints()
    assert after[first.id] != baseline[first.id], "the forwarded line must change"
    for segment in segments[1:]:
        assert after[segment.id] == baseline[segment.id], "no other line may change"

    # A different instruction for the same line is a different fingerprint.
    stored = _activate(studio, [_row(first, salt=6, capability_sha=_live_sha(studio, first))],
                       expected_revision=1, event_id="evt-second")
    changed = fingerprints()
    assert changed[first.id] != after[first.id]

    # Clearing the pointer restores every previous fingerprint exactly.
    director.commit_index(studio["project_id"], 2,
                          lambda index: index["active"].update({"direction_set": None}),
                          actor="synthetic", event_id="evt-rollback")
    studio["project"] = load_project(studio["project_id"])
    director.clear_direction_cache()
    restored = fingerprints()
    assert restored == baseline
    assert Path(stored["path"]).is_file(), "the directed set remains as history"


def test_advisory_only_directions_never_change_a_fingerprint(studio) -> None:
    """The default profile resolves to advisory, so nothing enters a requirement."""

    segments = studio["segments"]
    role = studio["project"].roles[0]
    baseline = pipeline.requirement_for(studio["project"], segments[0], role=role).fingerprint()
    _activate(studio, [_row(segments[0], salt=0, capability_sha=_live_sha(studio, segments[0]))])
    assert pipeline.requirement_for(studio["project"], segments[0],
                                    role=role).fingerprint() == baseline


# ------------------------------------------------- the canonical callers


def test_a_frozen_manifest_line_funnels_through_the_hook(studio) -> None:
    from dubbing_service.app.audiobook import casting_review

    segment = studio["segments"][0]
    without = casting_review._line_requirement(studio["project"], segment)
    _activate(studio, [_row(segment, salt=13, capability_sha=_live_sha(studio, segment))])
    with_index = casting_review._line_requirement(load_project(studio["project_id"]), segment)
    assert with_index != without, "a frozen manifest must see the forwarded direction"
    assert with_index == pipeline.requirement_for(
        load_project(studio["project_id"]), segment,
        role=load_project(studio["project_id"]).roles[0],
        engine_id=load_project(studio["project_id"]).roles[0].tts_backend,
        renderer_id=load_project(studio["project_id"]).roles[0].tts_backend).fingerprint()


def test_a_book_creator_binding_funnels_through_the_hook(studio) -> None:
    from dubbing_service.app.audiobook import book_creator

    segments = studio["segments"]
    without = book_creator._binding(studio["project"], segments)
    _activate(studio, [_row(segments[0], salt=17, capability_sha=_live_sha(studio, segments[0]))])
    project = load_project(studio["project_id"])
    with_index = book_creator._binding(project, load_segments(studio["project_id"]))
    assert with_index[segments[0].id] != without[segments[0].id]
    for segment in segments[1:]:
        assert with_index[segment.id] == without[segment.id]


def test_a_chapter_fork_binding_funnels_through_the_hook(studio) -> None:
    """The chapter owner calls the same funnel with its own renderer identity."""

    import inspect

    from dubbing_service.app.audiobook import voice_lab_chapter

    source = inspect.getsource(voice_lab_chapter)
    assert "pipeline.requirement_for(project, segment" in source, (
        "the chapter fork must keep using the one canonical funnel")

    segment = studio["segments"][0]
    role = studio["project"].roles[0]
    without = pipeline.requirement_for(studio["project"], segment, role=role,
                                       renderer_id="qwen3tts").fingerprint()
    _activate(studio, [_row(segment, salt=19, capability_sha=_live_sha(studio, segment))])
    with_index = pipeline.requirement_for(load_project(studio["project_id"]), segment, role=role,
                                          renderer_id="qwen3tts").fingerprint()
    assert with_index != without


def test_no_module_attaches_the_component_outside_the_funnel() -> None:
    """A static scan for a second writer of the new component."""

    offenders = []
    root = REPO_ROOT / "dubbing_service"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "performance_direction=" not in text:
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative.endswith(("audiobook/performance_director.py", "audiobook/fingerprint.py",
                              "audiobook/pipeline.py")):
            continue
        offenders.append(relative)
    assert offenders == [], f"unexpected writers of the component: {offenders}"


def test_the_metadata_exception_is_documented_where_the_pointer_is_read() -> None:
    """The one documented exception to the metadata rule, and its two references."""

    from dubbing_service.app.models.project import Project

    description = Project.model_fields["metadata"].description
    assert "performance_director" in description
    assert "唯一例外" in description, "the note must state that this key is the exception"
    docstring = pipeline.requirement_for.__doc__
    assert "single funnel" in docstring
    assert "metadata" in docstring


def test_the_direction_policy_and_index_schemas_are_pinned() -> None:
    """M4 adds schemas; it must not silently rename an accepted one."""

    assert director.DIRECTION_POLICY_SCHEMA == "direction-policy-v1"
    assert director.RESOLVER_VERSION == "pd-resolver-1"
    assert director.default_index().schema_version == "performance-director-index-v1"
    assert director.DirectionPolicy().schema_version == "direction-policy-v1"
    assert director.MAX_DELIVERY_PRIORITIES == 3
    assert director.MAX_INSTRUCTION_CHARS == 240


def test_the_canonical_json_and_hash_helpers_are_unchanged() -> None:
    """M4 reuses the accepted M1 rule instead of regenerating hashes."""

    assert fingerprint_module.canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    assert sha256_bytes(b"") == hashlib.sha256(b"").hexdigest()
    assert fingerprint_module.FINGERPRINT_SCHEMA == "audiobook-synthesis-requirement-v1"
    assert canonical_json_bytes({"a": 2, "b": 1}) == b'{"a":2,"b":1}'
