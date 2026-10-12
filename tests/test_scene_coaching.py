"""Source-bound coaching tests with invented prose; no model, network or GPU."""
import copy
import pytest
from dubbing_service.app.audiobook import scene_coaching as coach
from dubbing_service.app.audiobook.scene_dialogue_plan import _sha


def fixture():
    context = [{"utterance_id": "n", "text": "Mara suddenly shouted,", "role_id": "narr", "narrator": True},
               {"utterance_id": "a", "text": "What was that?", "role_id": "mara", "narrator": False},
               {"utterance_id": "b", "text": "I panicked.", "role_id": "tom", "narrator": False}]
    request = {"instruction": "Coach temporary delivery", "constraints": ["No identity change"],
               "response_contract": {"schema_version": coach.RESPONSE_SCHEMA},
               "scenes": [{"scene_id": "s", "project_id": "p", "context": context, "selected_dialogue_ids": ["a", "b"]}]}
    directions = [{"utterance_id": uid, "objective": "Demand clarity", "emotion": "fear", "intensity": "strong",
                   "cadence": "A sudden start", "emphasis": [text], "ending": "Land firmly",
                   "native_instruction": "Deliver a forceful demand; keep every word clear.", "confidence": "moderate",
                   "evidence": [{"utterance_id": uid, "start": 0, "end": len(text), "text": text}]}
                  for uid, text in [("a", "What was that?"), ("b", "I panicked.")]]
    response = {"schema_version": coach.RESPONSE_SCHEMA, "scenes": [{"scene_id": "s", "directions": directions}]}
    bundle = {"schema_version": coach.BUNDLE_SCHEMA, "request": request, "response": response,
              "request_sha256": _sha(request), "response_sha256": _sha(response),
              "provenance": {"origin": "model", "model_requested": "fixture", "receipt_sha256": "a"*64}}
    plan = {"directions": [{"utterance_id": r["utterance_id"], "narrator": r["narrator"],
                           "native_instruction": None, "source": {"source_revision_sha256": "b"*64},
                           "voice_revision": "v1", "basis": "none", "status": "held", "holds": [],
                           "approved": False, "recommendation_only": True} for r in context],
            "source": {"proven": True}, "recommendation_only": True, "approved": False,
            "binding_sha256": "c"*64, "holds": [{"code": "no_scene_evidence"}], "coverage": {}, "limitations": []}
    return bundle, plan, {"id": "p"}, [{"id": r["utterance_id"], **r} for r in context]


def rehash(bundle):
    bundle["request_sha256"] = _sha(bundle["request"])
    bundle["response_sha256"] = _sha(bundle["response"])


def test_applies_only_dialogue_preserves_inputs_and_keeps_human_gate(tmp_path):
    bundle, plan, project, segments = fixture()
    before = copy.deepcopy((bundle, plan, project, segments))
    result = coach.apply_coaching(plan, bundle, project, segments, cache_root=tmp_path)
    assert (bundle, plan, project, segments) == before
    assert result["directions"][0]["native_instruction"] is None
    assert all(r["basis"] == "model" and r["approved"] is False for r in result["directions"][1:])
    assert result["coaching"]["human_review"] == "pending" and result["approved"] is False
    assert result["coverage"]["model_coached"] == 2 and result["coverage"]["explicit_source"] == 0
    assert result["directions"][1]["coaching"]["objective"] == "Demand clarity"
    assert result["plan_sha256"] == _sha({k: v for k, v in result.items() if k != "plan_sha256"})


def test_voice_independent_cache_reuses_interpretation_but_preserves_new_binding(tmp_path):
    bundle, plan, project, segments = fixture()
    one = coach.apply_coaching(plan, bundle, project, segments, cache_root=tmp_path)
    cache = next(tmp_path.rglob('*.json')); before = (cache.read_bytes(), cache.stat().st_mtime_ns)
    plan["binding_sha256"] = "d"*64
    for row in plan["directions"]: row["voice_revision"] = "v2"
    two = coach.apply_coaching(plan, bundle, project, segments, cache_root=tmp_path)
    assert one["coaching"]["cache"][0]["hit"] is False and two["coaching"]["cache"][0]["hit"] is True
    assert one["coaching"]["cache"][0]["key"] == two["coaching"]["cache"][0]["key"]
    assert one["binding_sha256"] != two["binding_sha256"]
    assert two["directions"][1]["voice_revision"] == "v2"
    assert (cache.read_bytes(), cache.stat().st_mtime_ns) == before


@pytest.mark.parametrize('mutate', [
    lambda b: b['response']['scenes'][0]['directions'][0].update(native_instruction='Use a Scottish accent.'),
    lambda b: b['response']['scenes'][0]['directions'][0].update(utterance_id='n'),
    lambda b: b['response']['scenes'][0]['directions'].pop(),
    lambda b: b['response']['scenes'][0]['directions'].append(copy.deepcopy(b['response']['scenes'][0]['directions'][0])),
    lambda b: b['response']['scenes'][0]['directions'][0]['evidence'][0].update(start=1),
    lambda b: b['response']['scenes'][0]['directions'][0].update(emphasis=['invented wording']),
    lambda b: b['response']['scenes'][0]['directions'][0].update(native_instruction='x'*221),
    lambda b: b['provenance'].update(origin='human'),
])
def test_bad_coaching_never_writes_cache(tmp_path, mutate):
    bundle, plan, project, segments = fixture(); mutate(bundle); rehash(bundle)
    with pytest.raises(coach.CoachingError): coach.apply_coaching(plan, bundle, project, segments, cache_root=tmp_path)
    assert not (tmp_path / 'scene-coaching').exists()


def test_hash_tamper_is_refused():
    bundle, _, _, _ = fixture(); bundle['response']['scenes'][0]['directions'][0]['emotion'] = 'joy'
    with pytest.raises(coach.CoachingError, match='hash'): coach.validate_bundle(bundle)


@pytest.mark.parametrize('change', ['prose', 'attribution', 'source', 'voice', 'safety', 'selected'])
def test_changed_live_contract_or_unresolved_safety_is_not_cleared(tmp_path, change):
    bundle, plan, project, segments = fixture()
    if change == 'prose': segments[0]['text'] += ' changed'
    if change == 'attribution': segments[1]['role_id'] = 'tom'
    if change == 'source': plan['source']['proven'] = False
    if change == 'voice': plan['directions'][1]['voice_revision'] = None
    if change == 'safety': plan['holds'].append({'code': 'voice_unbound'})
    if change == 'selected': plan['directions'].pop()
    with pytest.raises(coach.CoachingError): coach.apply_coaching(plan, bundle, project, segments, cache_root=tmp_path)
    assert not (tmp_path / 'scene-coaching').exists()


def test_existing_cache_conflict_is_retained(tmp_path):
    bundle, plan, project, segments = fixture()
    coach.apply_coaching(plan, bundle, project, segments, cache_root=tmp_path)
    path = next(tmp_path.rglob('*.json')); path.write_bytes(b'old conflicting bytes')
    with pytest.raises(coach.CoachingError, match='conflict'): coach.apply_coaching(plan, bundle, project, segments, cache_root=tmp_path)
    assert path.read_bytes() == b'old conflicting bytes'
