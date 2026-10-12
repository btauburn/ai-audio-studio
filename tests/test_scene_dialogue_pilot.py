"""CPU tests for the scene dialogue pilot runner. No model, GPU, network or live project state."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "scene_dialogue_pilot.py"
spec = importlib.util.spec_from_file_location("scene_dialogue_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

REAL_SOURCE = Path(r"C:\Users\brian\omni-audiobook\runtime_evidence\2026-09-28-autocast-elf-acceptance\projects")
REAL_PROJECTS = ("9c7257c4", "b3b6b320", "c59f2ad6")


def sha(data):
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ synthetic canonical source

BLOCKS = {
    "fear": [("n", "The engine died and the road went black around them."),
             ("a", "\u201cI am scared, Dad, and I do not know why.\u201d"),
             ("b", "\u201cWe are not going to die out here.\u201d"),
             ("n", "She gripped the seat, terrified of the dark outside."),
             ("a", "\u201cDon\u2019t look at it.\u201d")],
    "worry": [("n", "Jane looked at her phone and frowned at the empty screen."),
              ("a", "\u201cI\u2019m worried about the signal, Dad.\u201d"),
              ("b", "\u201cThat is odd, and it is not right.\u201d"),
              ("n", "He kept driving, nervous and quiet."),
              ("a", "\u201cWhat if no one comes?\u201d")],
    "reassurance": [("n", "Erick reached across and took her hand."),
                    ("a", "\u201cDon\u2019t worry. I love you, and of course they will want you.\u201d"),
                    ("b", "\u201cI know. I\u2019m sorry I pushed.\u201d"),
                    ("n", "She nodded slowly and let out a breath."),
                    ("a", "\u201cYou be good now.\u201d")],
    "dry_humor": [("n", "Erick cracked another joke, smiling at the dark."),
                  ("a", "\u201cAre you getting Jurassic Park vibes yet?\u201d"),
                  ("b", "\u201cHello darkness, my old friend.\u201d"),
                  ("n", "Jane groaned and shook her head."),
                  ("a", "\u201cDad. Please stop talking to the darkness.\u201d")],
}
PLAIN = [("n", "Trees whizzed by on both sides of the highway."), ("a", "\u201cOkay then.\u201d"),
         ("b", "\u201cSure.\u201d"), ("n", "A mile passed in silence.")]


def make_identity(root: Path, voice_id: str, seed: int):
    folder = root / "casting" / "voices" / voice_id
    folder.mkdir(parents=True)
    reference = bytes([(seed * 7 + i) % 251 for i in range(300)])
    prompt = bytes([(seed * 11 + i) % 241 for i in range(400)])
    (folder / "reference.wav").write_bytes(reference)
    (folder / "conditioning.npz").write_bytes(prompt)
    return {"id": voice_id, "status": "LOCKED", "mechanism": "qwen_reference", "backend": "qwen3tts",
            "reference_audio": f"casting/voices/{voice_id}/reference.wav", "reference_audio_sha256": sha(reference),
            "prompt_artifact": f"casting/voices/{voice_id}/conditioning.npz", "prompt_sha256": sha(prompt),
            "provenance": {"origin": "eligible_reference_clone", "actor_audio_used": True}}


def make_project(root: Path, project_id: str, theme_blocks, *, chapter="chap_1", long_line=False):
    folder = root / project_id
    folder.mkdir(parents=True)
    roles = [{"id": "role_n", "name": "narrator", "role_type": "narrator", "voice_identity": make_identity(folder, "voice_n", 1)},
             {"id": "role_a", "name": "Erick", "role_type": "character", "voice_identity": make_identity(folder, "voice_a", 2)},
             {"id": "role_b", "name": "Jane", "role_type": "character", "voice_identity": make_identity(folder, "voice_b", 3)},
             {"id": "role_x", "name": "Nobody", "role_type": "character", "voice_identity": None}]
    who = {"n": "role_n", "a": "role_a", "b": "role_b"}
    segments, records, offset = [], [], 0

    def add(kind, text):
        nonlocal offset
        index = len(segments)
        anchor = {"source_revision_id": "src_1", "chapter_id": chapter, "start_offset": offset,
                  "end_offset": offset + len(text), "occurrence_index": index}
        offset += len(text) + 2
        segment = {"id": f"{project_id}-{index:03}", "chapter_id": chapter, "role_id": who[kind], "index": index,
                   "text": text, "source_anchor": anchor, "lineage": None}
        segments.append(segment)
        records.append({"utterance_id": segment["id"], "source_text_sha256": sha(text.encode()), "source_anchor": anchor})

    add("n", "001\n\nThe chapter heading is attached to this narration.")
    for name in theme_blocks:
        for kind, text in PLAIN + BLOCKS[name] + PLAIN:
            add(kind, text)
    if long_line:
        add("n", "A very long narration line. " * 12)
    project = {"id": project_id, "name": project_id, "roles": roles, "chapters": [{"id": chapter, "title": chapter, "index": 1}],
               "source_revisions": [{"id": "src_1", "sha256": sha(b"epub")}],
               "metadata": {"scene_pilot_fixture": True, "chapter_rerun": {"authorized": True, "chapter_id": chapter, "chapter_title": chapter,
                                              "source_sha256": sha(b"epub"),
                                              "source_utterance_ids": [s["id"] for s in segments],
                                              "source_utterances": records}}}
    (folder / "project.json").write_text(json.dumps(project), encoding="utf-8")
    (folder / "segments.json").write_text(json.dumps(segments), encoding="utf-8")
    (folder / "audio").mkdir()
    (folder / "audio" / "big.wav").write_bytes(b"not needed by the worker")
    discovery = folder / "voice_lab" / "discovery" / "acquisitions" / "a1"
    discovery.mkdir(parents=True)
    (discovery / "acquisition.json").write_text("{}", encoding="utf-8")
    return folder


@pytest.fixture()
def source(tmp_path):
    root = tmp_path / "source"
    make_project(root, "p1", ["fear", "worry"], chapter="c1")
    make_project(root, "p2", ["reassurance", "dry_humor"], chapter="c2")
    make_project(root, "p3", ["fear", "dry_humor"], chapter="c3", long_line=True)
    return root


def tree_hashes(root: Path):
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in sorted(root.rglob("*")) if p.is_file()}


def args_for(tmp_path, source, **overrides):
    values = dict(source_root=source, projects=["p1", "p2", "p3"], output_dir=tmp_path / "out", generate=False,
                  plan_builder=None, min_directed_dialogue=pilot.MIN_DIRECTED_DIALOGUE,
                  min_directed_scenes=pilot.MIN_DIRECTED_SCENES)
    values.update(overrides)
    return SimpleNamespace(**values)


def loader(source_row):
    return source_row["project"], source_row["segments"]


def fake_plan_builder(project, segments, *, cache_root=None, selected_ids=None, mutate=None):
    """Mimics scene-dialogue-plan-v1: dialogue gets a recommendation on every other line."""
    chosen = set(selected_ids or [])
    roles = {r["id"]: r for r in project["roles"]}
    rows = []
    for seg in segments:
        if seg["id"] not in chosen:
            continue
        narrator = roles[seg["role_id"]]["role_type"] == "narrator"
        directed = not narrator and seg["index"] % 2 == 0
        rows.append({"utterance_id": seg["id"], "role_id": seg["role_id"], "text_sha256": sha(seg["text"].encode()),
                     "voice_revision": "rev-1", "narrator": narrator,
                     "native_instruction": "Speak quietly, with held-back worry." if directed else None,
                     "basis": "inferred" if directed else "none", "evidence": [], "holds": [],
                     "recommendation_only": True})
    plan = {"schema_version": "scene-dialogue-plan-v1", "project_id": project["id"], "binding_sha256": sha(project["id"].encode()),
            "recommendation_only": True, "directions": rows, "scenes": [], "cache": {}}
    if mutate:
        mutate(plan)
    return plan


# ------------------------------------------------------------------ source integrity


def test_registry_fails_closed_on_missing_duplicate_and_malformed():
    good = {"source_revisions": [{"id": "src_1", "sha256": "a" * 64}]}
    assert pilot.source_registry(good) == {"src_1": "a" * 64}
    for bad in ({}, {"source_revisions": []},
                {"source_revisions": [{"id": "s", "sha256": "a" * 64}, {"id": "s", "sha256": "b" * 64}]},
                {"source_revisions": [{"id": "s", "sha256": "short"}]},
                {"source_revisions": [{"sha256": "a" * 64}]},
                {"source_revisions": ["nope"]}):
        with pytest.raises(pilot.PilotError):
            pilot.source_registry(bad)


def test_canonical_chapter_must_match_recorded_source(source):
    row = pilot.read_source(source, "p1")
    pilot.verify_canonical_chapter(row)
    changed = copy.deepcopy(row)
    changed["segments"][3]["text"] += " altered"
    with pytest.raises(pilot.PilotError, match="differs"):
        pilot.verify_canonical_chapter(changed)
    dropped = copy.deepcopy(row)
    del dropped["segments"][4]
    with pytest.raises(pilot.PilotError, match="cover"):
        pilot.verify_canonical_chapter(dropped)
    unauthorized = copy.deepcopy(row)
    unauthorized["project"]["metadata"]["chapter_rerun"]["authorized"] = False
    with pytest.raises(pilot.PilotError, match="authorized"):
        pilot.verify_canonical_chapter(unauthorized)


def test_invalid_project_id_is_refused(source):
    with pytest.raises(pilot.PilotError):
        pilot.read_source(source, "../p1")


# ------------------------------------------------------------------ selection


def selected(source):
    ids = ["p1", "p2", "p3"]
    rows, roles, registries = {}, {}, {}
    for pid in ids:
        row = pilot.read_source(source, pid)
        live, _ = pilot.verify_canonical_chapter(row)
        registries[pid] = pilot.source_registry(row["project"])
        roles[pid] = pilot.role_table(row["project"])
        rows[pid] = pilot.line_rows(row, live, roles[pid], registries[pid])
    chosen, search = pilot.select_scenes(ids, rows)
    return ids, rows, roles, registries, chosen, search


def test_selection_is_twenty_complete_contiguous_lines_with_real_cues(source):
    ids, rows, roles, registries, chosen, search = selected(source)
    assert chosen and search["candidates"] > 0
    scenes, lines = pilot.freeze_selection(ids, rows, chosen, roles, registries)
    assert len(lines) == 20
    narrator = sum(l["narrator"] for l in lines)
    assert narrator >= 6 and len(lines) - narrator >= 14
    assert len({l["project_id"] for l in lines}) >= 2
    for scene in scenes:
        indexes = [l["index"] for l in lines if l["scene_id"] == scene["scene_id"]]
        assert indexes == list(range(indexes[0], indexes[0] + len(indexes)))   # no line dropped inside a scene
        assert 4 <= len(indexes) <= 7
        assert any(not l["narrator"] for l in lines if l["scene_id"] == scene["scene_id"])
        assert any(l["narrator"] for l in lines if l["scene_id"] == scene["scene_id"])
        assert all(not c["synthesized"] for c in scene["context_before"] + scene["context_after"])
    by_id = {r["utterance_id"]: r for p in rows.values() for r in p}
    for line in lines:
        assert line["text"] == by_id[line["utterance_id"]]["text"] and line["chars"] <= pilot.MAX_CHARS
        assert line["voice"]["id"] in {"voice_n", "voice_a", "voice_b"}
        assert line["source"]["source_revision_sha256"] == sha(b"epub")
    for scene in scenes:  # cues are verbatim source slices at the recorded offsets
        for hits in scene["source_cues"] if isinstance(scene["source_cues"], list) else []:
            assert by_id[hits["utterance_id"]]["text"][hits["start"]:hits["end"]] == hits["cue"]
        assert scene["source_cues"], "a scene needs real cues"
    assert len({s["theme"] for s in scenes}) == len(scenes)


def test_selection_excludes_headings_overlong_and_unvoiced_lines(source):
    ids, rows, roles, registries, chosen, _ = selected(source)
    ineligible = {r["utterance_id"]: r["ineligible"] for p in rows.values() for r in p if not r["eligible"]}
    assert "chapter_heading" in ineligible["p1-000"]
    assert any("over_codec_budget" in v for v in ineligible.values())
    scenes, lines = pilot.freeze_selection(ids, rows, chosen, roles, registries)
    assert not {l["utterance_id"] for l in lines} & set(ineligible)


def test_selection_is_deterministic(source):
    first = selected(source)[4]
    second = selected(source)[4]
    assert [(w["project_id"], w["start"], w["length"]) for w in first] == [(w["project_id"], w["start"], w["length"]) for w in second]


def test_selection_without_cues_reports_infeasible(source):
    ids, rows, *_ = selected(source)
    for project in rows.values():
        for row in project:
            row["cues"] = {}
    chosen, search = pilot.select_scenes(ids, rows)
    assert chosen is None and search["candidates"] == 0


@pytest.mark.skipif(not REAL_SOURCE.is_dir(), reason="real canonical projects not present")
def test_real_chapters_supply_a_valid_twenty_line_selection():
    rows, roles, registries = {}, {}, {}
    for pid in REAL_PROJECTS:
        row = pilot.read_source(REAL_SOURCE, pid)
        live, _ = pilot.verify_canonical_chapter(row)
        registries[pid] = pilot.source_registry(row["project"])
        roles[pid] = pilot.role_table(row["project"])
        rows[pid] = pilot.line_rows(row, live, roles[pid], registries[pid])
    chosen, _ = pilot.select_scenes(list(REAL_PROJECTS), rows)
    assert chosen
    scenes, lines = pilot.freeze_selection(list(REAL_PROJECTS), rows, chosen, roles, registries)
    narrator = sum(l["narrator"] for l in lines)
    assert len(lines) == 20 and narrator >= 6 and len(lines) - narrator >= 14
    assert all(l["chars"] <= 250 and l["voice"]["provenance_origin"] == "eligible_reference_clone" for l in lines)


# ------------------------------------------------------------------ prepare / plans


def test_prepare_with_plan_freezes_state_and_leaves_originals_untouched(tmp_path, source):
    before = tree_hashes(source)
    state = pilot.prepare(args_for(tmp_path, source), plan_builder=fake_plan_builder, model_loader=loader)
    assert state["status"] == "ready", state["holds"]
    assert tree_hashes(source) == before and state["source_unchanged"] is True
    assert state["human_review"] == "pending" and state["production_ready"] is False
    assert state["default_runtime_changed"] is False and state["approvals_created"] is False
    for line in state["lines"]:
        if line["narrator"]:
            assert line["direction"]["native_instruction"] is None
        assert line["direction"]["recommendation_only"] is True and line["direction"]["human_review"] == "pending"
    assert state["adequacy"]["directed_dialogue_lines"] >= pilot.MIN_DIRECTED_DIALOGUE
    assert state["clone_coverage"]["covered_clones"] == 3
    assert (tmp_path / "out" / "selection.json").is_file() and (tmp_path / "out" / "state.json").is_file()
    copied = tree_hashes(tmp_path / "out" / "projects")
    assert not any("/audio/" in name for name in copied)          # only the state a worker needs
    assert any(name.endswith("reference.wav") for name in copied)
    assert any("acquisitions" in name for name in copied)
    again = pilot.prepare(args_for(tmp_path, source), plan_builder=fake_plan_builder, model_loader=loader)
    assert again["frozen_sha256"] == state["frozen_sha256"]       # idempotent re-run


def test_prepare_without_plan_owner_holds_and_invents_nothing(tmp_path, source, monkeypatch):
    monkeypatch.setattr(pilot, "discover_plan_module", lambda: None)
    monkeypatch.delenv("SCENE_DIALOGUE_PLAN_BUILDER", raising=False)
    state = pilot.prepare(args_for(tmp_path, source), model_loader=loader)
    assert state["status"] == "held"
    assert [h["code"] for h in state["holds"]] == ["plan_api_unavailable"]
    assert all(l["direction"]["native_instruction"] is None and l["direction"]["basis"] == "unavailable" for l in state["lines"])
    assert len(state["lines"]) == 20
    with pytest.raises(pilot.PilotError, match="held"):
        pilot.load_state(tmp_path / "out")


@pytest.mark.parametrize("mutate,code", [
    (lambda plan: plan["directions"][0].update(native_instruction="x", narrator=True), "plan_line_refused"),
    (lambda plan: plan.update(recommendation_only=False), "plan_not_recommendation_only"),
    (lambda plan: plan.update(schema_version="other"), "plan_schema"),
    (lambda plan: plan["directions"].pop(), "plan_line_missing"),
    (lambda plan: plan["directions"][1].update(text_sha256="0" * 64), "plan_line_refused"),
    (lambda plan: [d.update(basis="explicit_human") for d in plan["directions"] if d["native_instruction"]], "plan_line_refused"),
    (lambda plan: [d.update(native_instruction="Use a deeper pitch and a Scottish accent.") for d in plan["directions"]
                   if d["native_instruction"]], "plan_line_refused"),
    (lambda plan: [d.update(native_instruction="Sound like a famous actor.") for d in plan["directions"]
                   if d["native_instruction"]], "plan_line_refused"),
    (lambda plan: [d.update(native_instruction=None, basis="none") for d in plan["directions"]], "insufficient_automatic_directions"),
])
def test_bad_or_inadequate_plans_hold(tmp_path, source, mutate, code):
    def builder(project, segments, **kw):
        plan = fake_plan_builder(project, segments, **kw)
        if project["id"] == "p1":
            if code == "plan_line_refused" and "narrator" in repr(mutate.__code__.co_names) and False:
                pass
            # mutate dialogue-bearing rows so narrator row 0 exists for narrator-injection case
            if code == "plan_line_refused" and mutate.__code__.co_consts and "narrator" in mutate.__code__.co_names:
                pass
            mutate(plan)
        elif code == "insufficient_automatic_directions":
            mutate(plan)
        return plan
    state = pilot.prepare(args_for(tmp_path, source), plan_builder=builder, model_loader=loader)
    assert state["status"] == "held"
    assert code in {h["code"] for h in state["holds"]}


def test_narrator_instruction_in_plan_is_refused(tmp_path, source):
    def builder(project, segments, **kw):
        plan = fake_plan_builder(project, segments, **kw)
        for row in plan["directions"]:
            if row["narrator"]:
                row["native_instruction"] = "Read with a warm smile."
        return plan
    state = pilot.prepare(args_for(tmp_path, source), plan_builder=builder, model_loader=loader)
    assert state["status"] == "held"
    assert any("narrator carries a native instruction" in h["detail"] for h in state["holds"])
    assert all(l["direction"]["native_instruction"] is None for l in state["lines"] if l["narrator"])


def test_output_inside_source_store_is_refused(tmp_path, source, capsys):
    code = pilot.main(["--source-root", str(source), "--projects", "p1", "--output-dir", str(source / "out")])
    assert code == 2 and "outside the source project store" in capsys.readouterr().out


def test_copy_conflict_is_preserved(tmp_path, source):
    pilot.prepare(args_for(tmp_path, source), plan_builder=fake_plan_builder, model_loader=loader)
    target = tmp_path / "out" / "projects" / "p1" / "casting" / "voices" / "voice_n" / "reference.wav"
    target.write_bytes(b"tampered")
    with pytest.raises(pilot.PilotError):
        pilot.prepare(args_for(tmp_path, source), plan_builder=fake_plan_builder, model_loader=loader)
    assert target.read_bytes() == b"tampered"


# ------------------------------------------------------------------ generation with the real render()


class FakeResponse:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def json(self):
        return self.body


class FakeClient:
    def __init__(self, workdir, runtime, *, forward=True):
        self.workdir, self.runtime, self.forward, self.calls = workdir, runtime, forward, []

    def post(self, url, json):  # noqa: A002 - mirrors httpx
        self.calls.append(json)
        seed = int(hashlib.sha256((json["identity_id"] + json["text"]).encode()).hexdigest()[:8], 16)
        variation = 20 if json["performance"].get("native_instruction") else 0
        rate = 24000
        body = np.concatenate([np.zeros(int(0.05 * rate)),
                               0.2 * np.sin(2 * np.pi * (180 + seed % 90 + variation) * np.arange(rate // 2) / rate),
                               np.zeros(int(0.04 * rate))]).astype("float32")
        path = self.workdir / f"{len(self.calls):04}.wav"
        sf.write(path, body, rate, subtype="PCM_16")
        performance = json["performance"]
        direction = None
        if performance.get("native_instruction") is not None:
            text = performance["native_instruction"]
            direction = {"policy": pilot.DIRECTION_POLICY, "forwarded": self.forward,
                         "instruct_sha256": hashlib.sha256(text.encode()).hexdigest(), "instruct_chars": len(text),
                         "conditioning_mode": "icl", "seed_salt": 0, "seed_effective": seed % 2**32,
                         "direction_sha256": performance["direction_sha256"]}
        return FakeResponse({"success": True, "audio_file": str(path), "duration_seconds": len(body) / rate,
                             "worker_runtime_identity": {"sha256": self.runtime}, "fallback": None,
                             "identity_id": json["identity_id"], "prompt_sha256": json["prompt_sha256"],
                             "execution_profile": {"runtime_profile": pilot.DIRECTED_PROFILE}, "performance": {},
                             "generation": {"direction": direction},
                             "generation_settings": {"seed": seed, "temperature": 0.9}})


class FakeWorker:
    instances = []
    runtime = "7" * 64
    profile = pilot.DIRECTED_PROFILE
    forward = True

    def __init__(self, output, directed=False):
        assert directed is True
        scratch = Path(output) / "worker-scratch"
        scratch.mkdir(exist_ok=True)
        self.client = FakeClient(scratch, self.runtime, forward=self.forward)
        self.identity = {"sha256": self.runtime, "contract": {"runtime_profile": self.profile,
                                                              "checkpoint_manifest_sha256": "c" * 64, "model_id": "m"}}
        self.entered = self.exited = False
        FakeWorker.instances.append(self)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *a):
        self.exited = True


@pytest.fixture()
def prepared(tmp_path, source, monkeypatch):
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(tmp_path / "unrelated"))
    FakeWorker.instances, FakeWorker.runtime, FakeWorker.profile, FakeWorker.forward = [], "7" * 64, pilot.DIRECTED_PROFILE, True
    args = args_for(tmp_path, source)
    state = pilot.prepare(args, plan_builder=fake_plan_builder, model_loader=loader)
    assert state["status"] == "ready", state["holds"]
    _, render = pilot.default_render_tools()
    return args, pilot.load_state(args.output_dir), render


def test_generation_forwards_only_directed_dialogue_and_builds_manifest(prepared):
    args, state, render = prepared
    result = pilot.generate(args, state, FakeWorker, render)
    audit = result["audit"]
    directed = [l for l in state["lines"] if not l["narrator"] and l["direction"]["native_instruction"]]
    assert audit["directed_forwarded"] == len(directed) >= pilot.MIN_DIRECTED_DIALOGUE
    assert audit["narrator_instructions"] == audit["neutral_instructions"] == audit["baseline_instructions"] == 0
    worker = FakeWorker.instances[0]
    assert worker.entered and worker.exited and len(FakeWorker.instances) == 1     # one worker, released by the runner
    sent = [c["performance"] for c in worker.client.calls]
    assert sum(1 for p in sent if p) == len(directed) and len(sent) == 40
    manifest = json.loads((args.output_dir / "scene-comparison.json").read_bytes())
    assert manifest["schema_version"] == "scene-dialogue-comparison-v1" and manifest["production_ready"] is False
    assert set(manifest["arms"]) == {"baseline", "directed"} and len(manifest["lines"]) == 20
    base, dire = manifest["arms"]["baseline"], manifest["arms"]["directed"]
    assert set(base["samples"]) == set(dire["samples"]) == {l["utterance_id"] for l in manifest["lines"]}
    assert base["engine"] == dire["engine"] and base["boundary_policy_sha256"] == dire["boundary_policy_sha256"]
    for line in manifest["lines"]:
        a, b = base["samples"][line["utterance_id"]], dire["samples"][line["utterance_id"]]
        assert a["instruction"]["native_instruction"] is None
        assert (b["instruction"]["native_instruction"] is not None) == b["instruction_forwarded"]
        if line["narrator"]:
            assert b["instruction"]["native_instruction"] is None
        for key in ("text_sha256", "voice_id", "reference_id", "prompt_id", "runtime_sha256", "generation_settings"):
            assert a[key] == b[key]
        assert not Path(a["audio_file"]).is_absolute() and ".." not in Path(a["audio_file"]).parts
        assert sha((args.output_dir / a["audio_file"]).read_bytes()) == a["audio_sha256"]
    assert set(base["scenes"]) == set(dire["scenes"]) == {s["scene_id"] for s in state["scenes"]}
    for scene_id, row in dire["scenes"].items():
        timing = json.loads((args.output_dir / row["timing_file"]).read_bytes())
        assert timing["policy_sha256"] == row["policy_sha256"] == base["scenes"][scene_id]["policy_sha256"]
        assert all(t["boundary"] is None or t["boundary"]["trimming"] is False for t in timing["timing"])


def test_changed_cast_refused_before_worker_dispatch(prepared):
    args, state, render = prepared
    path = args.source_root / "p1" / "project.json"
    raw = json.loads(path.read_bytes())
    raw["roles"][1]["voice_identity"]["id"] = "changed-voice"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(pilot.PilotError, match="project contract"):
        pilot.generate(args, state, FakeWorker, render)
    assert not FakeWorker.instances


def test_notification_bookkeeping_is_separate_from_cast_and_approval():
    original = {"roles": [{"id": "a"}], "voice_lab": {"revision": 1, "casting": {
        "revision": 2, "notifications": {"complete": {"next_retry_at": "old"}},
        "roles": {"a": {"assignment": {"approved": True}}}}}}
    live = copy.deepcopy(original)
    live["voice_lab"]["revision"] += 1
    live["voice_lab"]["casting"]["revision"] += 1
    live["voice_lab"]["casting"]["notifications"]["complete"]["next_retry_at"] = "new"
    assert pilot.source_contract(original) == pilot.source_contract(live)
    live["voice_lab"]["casting"]["roles"]["a"]["assignment"]["approved"] = False
    assert pilot.source_contract(original) != pilot.source_contract(live)


def test_cli_preparation_retry_reuses_frozen_plan(tmp_path, source, monkeypatch):
    actual_prepare = pilot.prepare
    calls = []
    def prepare_once(args):
        calls.append(True)
        return actual_prepare(args, plan_builder=fake_plan_builder, model_loader=loader)
    monkeypatch.setattr(pilot, "prepare", prepare_once)
    argv = ["--source-root", str(source), "--projects", "p1", "p2", "p3", "--output-dir", str(tmp_path / "out")]
    assert pilot.main(argv) == 0
    before = tree_hashes(tmp_path / "out")
    assert pilot.main(argv) == 0
    assert calls == [True] and tree_hashes(tmp_path / "out") == before


def test_retry_resumes_verified_files_without_starting_a_worker(prepared):
    args, state, render = prepared
    pilot.generate(args, state, FakeWorker, render)
    manifest_bytes = (args.output_dir / "scene-comparison.json").read_bytes()
    FakeWorker.instances = []
    pilot.generate(args, state, FakeWorker, render)
    assert FakeWorker.instances == []                      # nothing left to synthesize; no GPU lane touched
    assert (args.output_dir / "scene-comparison.json").read_bytes() == manifest_bytes


def test_partial_run_resumes_only_the_missing_samples(prepared):
    args, state, render = prepared
    first = state["lines"][0]["utterance_id"]
    calls = {"n": 0}

    def flaky(worker, *a, **k):
        calls["n"] += 1
        if calls["n"] == 5:
            raise RuntimeError("worker dropped")
        return render(worker, *a, **k)
    with pytest.raises(RuntimeError):
        pilot.generate(args, state, FakeWorker, flaky)
    assert FakeWorker.instances[0].exited                  # runner always releases what it started
    FakeWorker.instances = []
    pilot.generate(args, state, FakeWorker, render)
    assert len(FakeWorker.instances[0].client.calls) == 40 - 4
    assert (args.output_dir / "receipts" / "baseline" / f"{first}.json").is_file()


def test_ambiguous_dispatch_holds_instead_of_resending(prepared):
    args, state, render = prepared
    line = state["lines"][0]
    sample = args.output_dir / "samples" / "baseline" / f"{line['utterance_id']}.wav"
    request_sha = pilot.expected_request_sha(line["project_id"], line["voice"], line["text"], None)
    pilot.publish(sample.with_suffix(f".pending-{request_sha[:16]}.json"), b"{}")
    with pytest.raises(RuntimeError, match="Unknown previous synthesis outcome"):
        pilot.generate(args, state, FakeWorker, render)
    assert FakeWorker.instances[0].client.calls == []       # no synthesis after an unresolved dispatch


def test_runtime_pin_blocks_a_different_worker(prepared):
    args, state, render = prepared
    pilot.generate(args, state, FakeWorker, render)
    for path in (args.output_dir / "samples" / "directed").glob(f"{state['lines'][-1]['utterance_id']}*"):
        pass
    FakeWorker.runtime = "8" * 64
    other = tmp = args.output_dir.parent / "out2"
    assert other != args.output_dir
    with pytest.raises(pilot.PilotError, match="differs from the pinned"):
        pilot.pin_runtime(args.output_dir, "8" * 64, {"contract": {"runtime_profile": pilot.DIRECTED_PROFILE,
                                                                   "checkpoint_manifest_sha256": "c" * 64, "model_id": "m"}})


def test_wrong_runtime_profile_is_refused(prepared):
    args, state, render = prepared
    FakeWorker.profile = "faster-base-032"
    with pytest.raises(pilot.PilotError, match="pinned"):
        pilot.generate(args, state, FakeWorker, render)
    assert FakeWorker.instances[0].client.calls == []


def test_unconfirmed_forwarding_is_refused(prepared):
    args, state, render = prepared
    FakeWorker.forward = False
    with pytest.raises(pilot.PilotError, match="did not confirm"):
        pilot.generate(args, state, FakeWorker, render)


def test_verify_sample_rejects_instruction_on_narrator_and_baseline(prepared):
    args, state, render = prepared
    narrator = next(l for l in state["lines"] if l["narrator"])
    dialogue = next(l for l in state["lines"] if not l["narrator"] and l["direction"]["native_instruction"])
    neutral = next(l for l in state["lines"] if not l["narrator"] and not l["direction"]["native_instruction"])
    worker = FakeWorker(args.output_dir, directed=True)
    pilot.generate  # noqa: B018 - generation is exercised elsewhere; here receipts are crafted directly
    good = render(worker, dialogue["project_id"], dialogue["voice"], dialogue, args.output_dir / "t" / "d.wav",
                  dialogue["direction"]["native_instruction"])
    assert pilot.verify_sample("directed", dialogue, good, "7" * 64)["forwarded"] is True
    with pytest.raises(pilot.PilotError, match="arm rule"):
        pilot.verify_sample("baseline", dialogue, good, "7" * 64)       # baseline must never carry one
    leaked = render(worker, narrator["project_id"], narrator["voice"], narrator, args.output_dir / "t" / "n.wav", "Whisper it.")
    with pytest.raises(pilot.PilotError, match="arm rule"):
        pilot.verify_sample("directed", narrator, leaked, "7" * 64)     # narrator must never carry one
    leaked_neutral = render(worker, neutral["project_id"], neutral["voice"], neutral, args.output_dir / "t" / "u.wav", "Sound tense.")
    with pytest.raises(pilot.PilotError, match="arm rule"):
        pilot.verify_sample("directed", neutral, leaked_neutral, "7" * 64)
    with pytest.raises(pilot.PilotError, match="pinned"):
        pilot.verify_sample("directed", dialogue, good, "9" * 64)


def test_audit_counts_are_exact():
    line_d = {"utterance_id": "d", "narrator": False, "direction": {"native_instruction": "Speak softly."}}
    line_n = {"utterance_id": "n", "narrator": True, "direction": {"native_instruction": None}}
    ok = {arm: {"d": {"instruction_forwarded": arm == "directed"}, "n": {"instruction_forwarded": False}} for arm in pilot.ARMS}
    assert pilot.audit_forwarding([line_d, line_n], ok)["directed_forwarded"] == 1
    bad = copy.deepcopy(ok)
    bad["directed"]["n"]["instruction_forwarded"] = True
    with pytest.raises(pilot.PilotError):
        pilot.audit_forwarding([line_d, line_n], bad)


def test_state_tamper_after_preparation_is_detected(prepared):
    args, state, render = prepared
    path = args.output_dir / "state.json"
    raw = json.loads(path.read_bytes())
    raw["lines"][0]["text"] += " tampered"
    path.write_bytes(pilot.encoded(raw))
    with pytest.raises(pilot.PilotError, match="changed since preparation"):
        pilot.load_state(args.output_dir)


def test_direction_digest_matches_the_worker_adapter():
    from dubbing_service.app.audiobook.performance_provider import component_sha256
    assert pilot.direction_digest("Speak gently.", 0) == component_sha256("Speak gently.", 0)
