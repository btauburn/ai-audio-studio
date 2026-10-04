"""Migration uses disposable library/project stores and synthetic fixture evidence only."""
# ruff: noqa: F811
import copy
import json

import pytest

from dubbing_service.app.audiobook import preferred_voices as preferred
from dubbing_service.app.audiobook import preferred_voice_migration as migration
from dubbing_service.app.audiobook import voice_lab as lab
from dubbing_service.app.audiobook import voice_lab_library as shelf
from dubbing_service.app.storage import voices as library
from dubbing_service.app.storage.projects import load_project, save_project
from tests.test_voice_lab_library import source, add_reference_identity


def historical(source):
    root, _, project, _, _ = source
    record = add_reference_identity(source)
    intent = "a" * 64
    record["key"] = "v_" + lab.digest([record["identity"]["id"], record["version"], intent])[:28]
    record.pop("generation_language", None)
    record.update(audition_project_id="fixture-child", take_ids=["fixture-1", "fixture-2"])
    takes = [{"id": tid, "audio_sha256": str(index) * 64, "language": "English",
              "fingerprint_components": {"voice_revision": record["version"], "language": "English"},
              "generation_metadata": {"character_voice_identity": {"id": record["identity"]["id"]}},
              "qa_report": {"take_id": tid, "output_audio_hash": str(index) * 64, "passed": True}}
             for index, tid in enumerate(record["take_ids"], 1)]
    proof = {"schema": "voice-lab-audition-takes-v2", "project_id": "fixture-child", "takes": takes,
             "assembled_audio_sha256": record["audio_sha256"], "assembled_qa": {"passed": True}}
    path = root/project.id/f"voice_lab/auditions/{intent}/take.json"
    path.parent.mkdir(parents=True)
    content = json.dumps(proof).encode()
    path.write_bytes(content)
    record["details"] = {"take_proof_file": str(path.relative_to(root/project.id)).replace("\\", "/"),
                         "take_proof_sha256": migration._bytes_hash(content)}
    project = load_project(project.id)
    project.voice_lab.scope = "production"  # This remains inside the disposable fixture root.
    project.voice_lab.voices[record["key"]] = record
    event = {"event_id": "fixture-rated", "role_id": "narrator", "candidate_id": "fixture-candidate",
             "key": record["key"], "audio_sha256": record["audio_sha256"], "stars": 5,
             "voice_id": record["voice_id"], "voice_version": record["version"], "actor": "fixture-operator",
             "origin": "human", "created_at": "2026-01-01T00:00:00Z", "action": "save"}
    candidate = {"candidate_id": event["candidate_id"], "key": record["key"], "audio_sha256": record["audio_sha256"],
                 "inferred_traits": [{"dimension": "presentation", "value": "masculine", "level": "CASTING_INFERENCE"}]}
    project.voice_lab.casting = {"schema":"casting-review-v1", "revision":0,
                               "events": {event["event_id"]: event}, "roles": {"narrator": {"round": {"candidates": [candidate]}}}}
    save_project(project)
    return project, record, event, proof


def test_actual_audition_formula_migrates_exact_approval_once_and_preserves_source(source):
    root, _, _, _, _ = source
    project, record, event, _ = historical(source)
    before = (root/project.id/'project.json').read_bytes()
    dry = migration.plan(project.id)
    assert dry['eligible_events'] == 1 and not dry['held']
    assert not library._voices_file_path().exists()
    result = migration.migrate(project.id, dry_run=False, expected_plan_sha256=dry['plan_sha256'])
    assert result['status'] == 'complete'
    entry = preferred.preferred_entries()[0]
    rating = entry['ratings'][0]
    assert rating['source_event'] == event and rating['rating'] == 5
    assert rating['audio_sha256'] == record['audio_sha256'] and rating['key'] == record['key']
    _, saved = preferred._record(entry['library_voice_id'], entry['version'])
    assert saved['generation_language'] == 'English'
    assert saved['key'] == record['key'] and saved['identity'] == record['identity']
    assert (root/project.id/'project.json').read_bytes() == before
    library_before = library._voices_file_path().read_bytes()
    assert migration.migrate(project.id, dry_run=False, expected_plan_sha256=dry['plan_sha256'])['already_applied']
    assert library._voices_file_path().read_bytes() == library_before
    match = preferred.match_preferred({'presentation':'masculine','language':'en'})
    assert match and match[0]['sample_audio_sha256'] == record['audio_sha256']


def test_unrated_selection_and_favorite_never_become_star_ratings(source):
    project, record, event, _ = historical(source)
    project = load_project(project.id)
    event.pop('stars')
    project.voice_lab.casting['events'][event['event_id']] = event
    project.voice_lab.preferences[record['key']] = {'favorite':True, 'audio_sha256':record['audio_sha256']}
    save_project(project)
    dry = migration.plan(project.id)
    assert not dry['rows']
    result = migration.migrate(project.id, dry_run=False, expected_plan_sha256=dry['plan_sha256'])
    assert result['status'] == 'no_op' and not library._voices_file_path().exists()


@pytest.mark.parametrize('defect', ['identity','qa'])
def test_unknown_first_language_does_not_hide_invalid_later_take(source, defect):
    _, record, _, proof = historical(source)
    proof['takes'][0].pop('language')
    if defect == 'identity':
        proof['takes'][1]['generation_metadata']['character_voice_identity']['id'] = 'wrong-identity'
    else:
        proof['takes'][1]['qa_report']['passed'] = False
    with pytest.raises(shelf.VoiceLibraryError, match='unbound identity or Take'):
        shelf._audition_proof_binding(record, json.dumps(proof).encode())


def test_audition_formula_without_immutable_proof_cannot_publish(source):
    project, record, _, _ = historical(source)
    record['audition_intent_hash'] = 'a'*64
    record['details'] = {}
    with pytest.raises(shelf.VoiceLibraryError, match='immutable Take proof'):
        shelf._publication_record(project.id, record)


def test_self_consistent_event_and_arguments_for_wrong_audio_are_rejected(source):
    project, record, event, _ = historical(source)
    published = shelf.publish_version(project.id, record['key'])
    event['audio_sha256'] = 'f'*64
    with pytest.raises(preferred.PreferredVoiceError, match='exact heard voice'):
        preferred.record_rating(library_voice_id=published['library_voice_id'], version=published['version'],
            rating=5, project_id=project.id, role_id='narrator', key=event['key'], audio_sha256=event['audio_sha256'],
            evidence_id=event['event_id'], source_event=event)


def test_rollback_preserves_unrelated_write_interleaved_during_migration(source, monkeypatch):
    project, _, _, _ = historical(source)
    dry = migration.plan(project.id)
    original = shelf.publish_version
    def interleave(*args, **kwargs):
        result = original(*args, **kwargs)
        with library.voices_lock():
            rows = library.load_voices()
            unrelated = rows[0].model_copy(deep=True)
            unrelated.id = 'unrelated-owner-write'
            unrelated.name = 'New unrelated work'
            unrelated.voice_lab_versions = {}
            unrelated.voice_lab_preferred = {}
            rows.append(unrelated)
            library.save_voices(rows)
        return result
    monkeypatch.setattr(shelf, 'publish_version', interleave)
    result = migration.migrate(project.id, dry_run=False, expected_plan_sha256=dry['plan_sha256'])
    migration.rollback(project.id, result['migration_id'])
    assert any(row.id == 'unrelated-owner-write' for row in library.load_voices())
    assert not preferred.preferred_entries()


def test_corrupted_backup_cannot_mutate_current_library(source):
    project, _, _, _ = historical(source)
    dry = migration.plan(project.id)
    result = migration.migrate(project.id, dry_run=False, expected_plan_sha256=dry['plan_sha256'])
    directory = library.voices_audio_dir()/'voice_lab/migrations'/result['migration_id']
    (directory/'voices.before.json').write_text('[] wrong bytes', 'utf-8')
    before = library._voices_file_path().read_bytes()
    with pytest.raises(shelf.VoiceLibraryError, match='backup integrity'):
        migration.rollback(project.id, result['migration_id'])
    assert library._voices_file_path().read_bytes() == before


def test_rollback_refuses_new_rating_on_an_owned_version(source):
    project, record, _, _ = historical(source)
    dry = migration.plan(project.id)
    result = migration.migrate(project.id, dry_run=False, expected_plan_sha256=dry['plan_sha256'])
    entry = preferred.preferred_entries()[0]
    preferred.record_rating(library_voice_id=entry['library_voice_id'],version=entry['version'], rating=4,
        project_id='subsequent-project',role_id='other-role', key=record['key'],audio_sha256=record['audio_sha256'],evidence_id='later-event')
    before = library._voices_file_path().read_bytes()
    with pytest.raises(shelf.VoiceLibraryError, match='newer work'):
        migration.rollback(project.id, result['migration_id'])
    assert library._voices_file_path().read_bytes() == before


@pytest.mark.parametrize('direction',['Not British','Avoid a male voice',"Don't use a British male voice"])
def test_negated_historical_directions_are_not_positive_compatibility_traits(direction):
    traits, _ = migration._traits({}, {'_historical_round_direction':{'retained':[direction]}}, {})
    assert 'accent' not in traits and 'presentation' not in traits


def test_later_requested_changes_are_not_traits_of_the_heard_sample():
    traits, _ = migration._traits({}, {}, {'interpretation':{'retained':['Make it British male']}})
    assert 'accent' not in traits and 'presentation' not in traits


def test_stale_plan_cannot_write_the_library(source):
    project, _, _, _ = historical(source)
    with pytest.raises(shelf.VoiceLibraryError, match='plan changed'):
        migration.migrate(project.id, dry_run=False, expected_plan_sha256='0'*64)
    assert not library._voices_file_path().exists()


def test_qualitative_rejection_preserves_stars_identity_and_never_dispatches(source):
    from dubbing_service.app.audiobook import casting_review as casting
    root, _, _, _, _ = source
    project, record, _, _ = historical(source)
    original = copy.deepcopy(load_project(project.id).model_dump(mode='json'))
    args = {'event_id':'fixture-quality-rejection', 'role_id':'narrator', 'candidate_id':'fixture-candidate',
            'key':record['key'], 'audio_sha256':record['audio_sha256'],
            'raw_feedback':'Delivery too loud for narration; rejected for this role.', 'requested_performer':'Richard Harris'}
    result = casting.qualitative_rejection(project.id, **args)
    assert result['qualitatively_rejected'] and not result['rating_invented'] and not result['tts_queued']
    after = load_project(project.id)
    assert after.roles == load_project(project.id).roles
    assert after.model_dump(mode='json')['roles'] == original['roles']
    assert after.voice_lab.voices == original['voice_lab']['voices']
    assert after.voice_lab.preferences == original['voice_lab']['preferences']
    assert after.voice_lab.casting['roles']['narrator']['round']['candidates'][0]['human_fit']['stars'] is None
    snapshot = (root/project.id/'project.json').read_bytes()
    assert casting.qualitative_rejection(project.id, **args) == result
    assert (root/project.id/'project.json').read_bytes() == snapshot
    assert not (root/project.id/'jobs.json').exists()


@pytest.mark.parametrize('after_write', [False, True])
def test_interrupted_rating_keeps_published_scope_in_rollback(source, monkeypatch, after_write):
    project, _, _, _ = historical(source)
    dry = migration.plan(project.id)
    original = preferred.record_rating
    def interrupted(*args, **kwargs):
        if after_write:
            original(*args, **kwargs)
        raise OSError('Injected rating boundary failure in disposable store')
    monkeypatch.setattr(preferred, 'record_rating', interrupted)
    with pytest.raises(OSError, match='Injected rating'):
        migration.migrate(project.id, dry_run=False, expected_plan_sha256=dry['plan_sha256'])
    migration_id = 'migration_' + dry['plan_sha256'][:32]
    receipt = json.loads((library.voices_audio_dir()/'voice_lab/migrations'/migration_id/'receipt.json').read_text('utf-8'))
    assert receipt['status'] == 'incomplete' and not receipt['completed']
    assert len(receipt['published_scopes']) == len(receipt['owned_versions']) == 1
    result = migration.rollback(project.id, migration_id)
    assert result['status'] == 'rolled_back'
    assert not preferred.preferred_entries()
    assert all(not row.voice_lab_versions for row in library.load_voices())


def test_unfinalized_empty_journal_cannot_claim_rollback_success(source):
    project, _, _, _ = historical(source)
    dry = migration.plan(project.id)
    result = migration.migrate(project.id, dry_run=False, expected_plan_sha256=dry['plan_sha256'])
    path = library.voices_audio_dir()/'voice_lab/migrations'/result['migration_id']/'receipt.json'
    receipt = json.loads(path.read_text('utf-8'))
    receipt.update(status='applying', published_scopes=[])
    receipt.pop('owned_versions')
    receipt.pop('ownership_finalized')
    path.write_text(json.dumps(receipt),'utf-8')
    before = library._voices_file_path().read_bytes()
    with pytest.raises(shelf.VoiceLibraryError,match='ownership journal needs recovery'):
        migration.rollback(project.id,result['migration_id'])
    assert library._voices_file_path().read_bytes() == before


def test_two_verified_intent_keys_keep_original_ratings_and_one_immutable_version(source):
    root, _, _, _, _ = source
    project, record, event, proof = historical(source)
    project = load_project(project.id)
    original = project.voice_lab.casting['events'][event['event_id']]
    original['stars'] = 4
    alternate = copy.deepcopy(record)
    intent = 'b' * 64
    alternate['key'] = 'v_' + lab.digest([record['identity']['id'], record['version'], intent])[:28]
    path = root/project.id/f'voice_lab/auditions/{intent}/take.json'
    path.parent.mkdir(parents=True)
    raw = json.dumps(proof).encode()
    path.write_bytes(raw)
    alternate['details'] = {'take_proof_file':str(path.relative_to(root/project.id)).replace('\\','/'),
                            'take_proof_sha256':migration._bytes_hash(raw)}
    project.voice_lab.voices[alternate['key']] = alternate
    alias_event = {**event, 'event_id':'fixture-rated-z-alias', 'candidate_id':'fixture-candidate-alias', 'key':alternate['key']}
    project.voice_lab.casting['events'][alias_event['event_id']] = alias_event
    project.voice_lab.casting['roles']['narrator']['round']['candidates'].append({
        'candidate_id':alias_event['candidate_id'],'key':alternate['key'],'audio_sha256':record['audio_sha256'],
        'inferred_traits':[{'dimension':'presentation','value':'masculine','level':'CASTING_INFERENCE'}]})
    save_project(project)
    dry = migration.plan(project.id)
    assert dry['eligible_events'] == 2 and not dry['held']
    result = migration.migrate(project.id, dry_run=False, expected_plan_sha256=dry['plan_sha256'])
    assert result['status'] == 'complete' and len(result['owned_versions']) == 1
    entry = preferred.preferred_entries()[0]
    assert [row['key'] for row in entry['ratings']] == [record['key'],alternate['key']]
    saved = preferred._record(entry['library_voice_id'],entry['version'])[1]
    assert saved['key'] == record['key']
    assert preferred._rating_key_bound(entry['ratings'][1],saved)
    match = preferred.match_preferred({'presentation':'masculine','language':'en'})
    assert match[0]['signal'] == 5 and match[0]['promoting_rating']['key'] == alternate['key']
    for original_row in entry['ratings']:
        for field, value in [('rating', 5 if original_row['rating'] == 4 else 4),
                             ('role_id','unheard-role'),('event_id','invented-event'),
                             ('project_id','other-project'),('actor','different-human'),
                             ('origin','machine'),('original_recorded_at','invented-date')]:
            tampered = copy.deepcopy(original_row)
            tampered[field] = value
            assert not preferred._rating_key_bound(tampered,saved), field
    corrupted = copy.deepcopy(entry['ratings'][1])
    corrupted['heard_key_binding']['audio_sha256'] = 'f'*64
    assert not preferred._rating_key_bound(corrupted,saved)


def test_rolled_back_plan_gets_preserved_deterministic_recovery_attempt(source):
    project, _, _, _ = historical(source)
    dry = migration.plan(project.id)
    first = migration.migrate(project.id,dry_run=False,expected_plan_sha256=dry['plan_sha256'])
    migration.rollback(project.id,first['migration_id'])
    recovered = migration.migrate(project.id,dry_run=False,expected_plan_sha256=dry['plan_sha256'])
    assert recovered['status'] == 'complete' and recovered['migration_id'] != first['migration_id']
    assert migration.migrate(project.id,dry_run=False,expected_plan_sha256=dry['plan_sha256'])['already_applied']
    directory = library.voices_audio_dir()/'voice_lab/migrations'/first['migration_id']
    assert json.loads((directory/'receipt.json').read_text())['status'] == 'rolled_back'
