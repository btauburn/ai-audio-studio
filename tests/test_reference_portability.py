"""External keeper portability in disposable stores; no real model or network."""
import copy
import hashlib
import json
import shutil

import pytest

from dubbing_service.app.audiobook import reference_portability as portability
from dubbing_service.app.audiobook import voice_lab as lab, voice_lab_library as shelf
from dubbing_service.app.storage.projects import create_project, load_project, save_project
from dubbing_service.app.storage import voices as library
from service import qwen_character_voice as voice
from tests.test_voice_lab_library import source, add_reference_identity
from tests.test_voice_lab_reference_clone import fixture as acquisition_fixture, acquire, Prompt, Tensor


@pytest.fixture
def external(source,acquisition_fixture,request,monkeypatch):
    root,_,project,target,_ = source
    f = acquisition_fixture
    acquired = acquire(f)
    if getattr(request,'param',None) == 'isolated':
        from dubbing_service.app.audiobook import reference_separation
        from dubbing_service.app.storage.projects import _get_project_meta_lock
        recipe = {'fixture_cpu_separator_replay':True}
        path = f.directory/f"acquisitions/{acquired['acquisition_id']}/acquisition.json"
        receipt = json.loads(path.read_text())
        receipt['source_processing'] = recipe
        receipt['audio_review']['source_processing'] = recipe
        receipt['preprocessing'].append({'operation':'reference_vocals_isolation','recipe':recipe})
        path.write_text(json.dumps(receipt))
        acquired = {**receipt,'acquisition_sha256':voice.sha256(path)}
        f.preprocessing_replays = []
        def replay(wave,rate):
            assert not _get_project_meta_lock(target.id).locked(), 'CPU replay must not block project metadata'
            f.preprocessing_replays.append(rate)
            return wave,rate,recipe
        monkeypatch.setattr(reference_separation,'isolate_speech',replay)
    record = add_reference_identity(source)
    for relative in (acquired['original_path'],acquired['reference_path'],
                     f"acquisitions/{acquired['acquisition_id']}/acquisition.json"):
        destination = root/project.id/'voice_lab/discovery'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(f.directory/relative,destination)
    identity = record['identity']
    reference = root/project.id/identity['reference_audio']
    shutil.copyfile(f.directory/acquired['reference_path'],reference)
    identity.update(reference_audio_sha256=acquired['reference_audio_sha256'],reference_text=acquired['reference_text'],
        provenance={'origin':'eligible_reference_clone','source':acquired['source'],'eligibility':acquired['eligibility'],
            'acquisition_id':acquired['acquisition_id'],'acquisition_sha256':acquired['acquisition_sha256']})
    prompt = root/project.id/identity['prompt_artifact']
    prompt.unlink()
    import numpy as np
    item = Prompt(Tensor(np.ones((12,16),dtype=np.int64)),Tensor(np.arange(128,dtype=np.float32)),False,True,identity['reference_text'])
    voice.save_prompt(prompt,item,{'schema':'qwen-original-prompt-v1','identity_id':identity['id'],
        'reference_audio_sha256':identity['reference_audio_sha256'],'origin':'eligible_reference_clone',
        'acquisition_id':acquired['acquisition_id'],'acquisition_sha256':acquired['acquisition_sha256'],
        'model':'fixture-base','model_revision':'revision-1','runtime_version':'0.1.1'})
    identity['prompt_sha256'] = voice.sha256(prompt)
    record['generation_language'] = 'English'
    record['version'] = lab.digest(identity)[:32]
    record['key'] = 'v_'+lab.digest([record['voice_id'],record['version']])[:28]
    loaded = load_project(project.id)
    loaded.voice_lab.voices = {record['key']:record}
    save_project(loaded)
    published = shelf.publish_version(project.id,record['key'])
    return source,f,acquired,record,published


def test_exact_archive_import_preserves_source_and_library_metadata(external):
    source,f,acquired,record,published = external
    root,_,project,target,_ = source
    before = (root/project.id/'project.json').read_bytes()
    library_before = library._voices_file_path().read_bytes()
    result = shelf.import_version(target.id,published['library_voice_id'],published['version'])
    assert result['reference_portability']['cached_prompt_reused']
    stored = load_project(target.id)
    imported = stored.voice_lab.voices[record['key']]
    assert not stored.voice_lab.preferences and not stored.voice_lab.preference_history
    assert imported['identity']['id'] == record['identity']['id'] and imported['version'] == record['version']
    assert imported['identity']['provenance'] == record['identity']['provenance']
    assert shelf._profile_fingerprint(imported) == shelf._profile_fingerprint(record)
    assert voice.sha256(root/target.id/imported['identity']['prompt_artifact']) == record['identity']['prompt_sha256']
    checked = voice._eligible_acquisition(root/target.id,acquired['acquisition_id'],acquired['acquisition_sha256'])[0]
    assert checked['reference_audio_sha256'] == record['identity']['reference_audio_sha256']
    assert (root/project.id/'project.json').read_bytes() == before
    assert library._voices_file_path().read_bytes() == library_before
    target_before = (root/target.id/'project.json').read_bytes()
    assert not shelf.import_version(target.id,published['library_voice_id'],published['version'])['imported']
    assert (root/target.id/'project.json').read_bytes() == target_before
    assert not f.calls


def test_cached_prompt_reaches_preinference_model_boundary_without_recloning(external,monkeypatch):
    source,f,_,record,published = external
    root,_,_,target,_ = source
    shelf.import_version(target.id,published['library_voice_id'],published['version'])
    imported = load_project(target.id).voice_lab.voices[record['key']]
    identity = imported['identity']
    monkeypatch.setattr(voice,'safe_project_root',lambda pid:root/pid)
    class BeforeModel(RuntimeError): pass
    reached = []
    def trap(kind):
        reached.append(kind)
        raise BeforeModel('All identity/acquisition/prompt/model-provenance checks completed')
    runtime = voice.CharacterVoiceRuntime(root,trap)
    monkeypatch.setattr(runtime,'_model_metadata',lambda kind:{'model':'fixture-base','model_revision':'revision-1','runtime_version':'0.1.1'})
    original = voice.load_prompt
    loads = []
    def load(*args,**kwargs):
        loads.append(args[0])
        return original(*args,**kwargs)
    monkeypatch.setattr(voice,'load_prompt',load)
    for _ in range(2):
        with pytest.raises(BeforeModel):
            runtime.synthesize(target.id,identity['id'],text='Fixture.',language='English',
                **{key:identity[key] for key in ('prompt_artifact','prompt_sha256','reference_audio','reference_audio_sha256')})
    assert reached == ['base','base'] and len(loads) == 1 and len(runtime._prompt_cache) == 1
    assert not f.calls


@pytest.mark.parametrize('asset',['receipt','reference','original'])
def test_missing_or_tampered_source_fails_before_target_metadata(external,asset):
    source,_,acquired,_,published = external
    root,_,project,target,_ = source
    relative = (f"acquisitions/{acquired['acquisition_id']}/acquisition.json" if asset == 'receipt'
                else acquired['reference_path' if asset == 'reference' else 'original_path'])
    path = root/project.id/'voice_lab/discovery'/relative
    if asset == 'receipt': path.unlink()
    else: path.write_bytes(b'changed fixture bytes')
    before = (root/target.id/'project.json').read_bytes()
    with pytest.raises((ValueError,OSError)):
        shelf.import_version(target.id,published['library_voice_id'],published['version'])
    assert (root/target.id/'project.json').read_bytes() == before


def test_verified_archive_is_self_contained_when_source_project_is_unavailable(external):
    source,_,acquired,record,published = external
    root,_,project,target,_ = source
    shelf.import_version(target.id,published['library_voice_id'],published['version'])
    moved = root/(project.id+'-fixture-unavailable')
    (root/project.id).rename(moved)
    second = create_project('Second isolated portability target')
    result = shelf.import_version(second.id,published['library_voice_id'],published['version'])
    assert result['imported']
    assert voice._eligible_acquisition(root/second.id,acquired['acquisition_id'],acquired['acquisition_sha256'])[0]['clone_ready']
    assert load_project(second.id).voice_lab.voices[record['key']]['identity']['provenance'] == record['identity']['provenance']


@pytest.mark.parametrize('asset',['receipt','reference','original'])
def test_target_tamper_is_not_silently_repaired_or_recloned(external,asset):
    source,f,acquired,_,published = external
    root,_,_,target,_ = source
    shelf.import_version(target.id,published['library_voice_id'],published['version'])
    relative = (f"acquisitions/{acquired['acquisition_id']}/acquisition.json" if asset == 'receipt'
                else acquired['reference_path' if asset == 'reference' else 'original_path'])
    path = root/target.id/'voice_lab/discovery'/relative
    path.write_bytes(b'tampered target fixture')
    before = (root/target.id/'project.json').read_bytes()
    with pytest.raises((ValueError,OSError)):
        shelf.import_version(target.id,published['library_voice_id'],published['version'])
    assert path.read_bytes() == b'tampered target fixture' and (root/target.id/'project.json').read_bytes() == before
    assert not f.calls


def test_revoked_use_evidence_cannot_be_waived_by_clone_ready_flags(external):
    source,f,acquired,_,_ = external
    root,_,project,_,_ = source
    path = root/project.id/f"voice_lab/discovery/acquisitions/{acquired['acquisition_id']}/acquisition.json"
    receipt = json.loads(path.read_text())
    receipt['eligibility']['intended_use_evidence']['status'] = 'revoked'
    receipt['clone_ready'] = True
    receipt['technical_qa']['status'] = 'pass'
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='eligible use'):
        voice._eligible_acquisition(root/project.id,acquired['acquisition_id'],hashlib.sha256(path.read_bytes()).hexdigest())
    assert not f.calls


def test_prompt_cannot_claim_another_acquisition_with_valid_zip(external):
    source,_,_,_,published = external
    _,_,_,target,_ = source
    entry = copy.deepcopy(library.load_voices()[0].voice_lab_versions[published['version']])
    entry['record']['identity']['provenance']['acquisition_id'] = 'b'*32
    asset = entry['assets']['prompt_artifact']
    raw = (library.voices_audio_dir()/asset['path']).read_bytes()
    with pytest.raises(shelf.VoiceLibraryError,match='Cached external prompt'):
        portability.prepare(entry,published['library_voice_id'],raw,library.voices_audio_dir(),shelf.safe_project_root(target.id))



@pytest.mark.parametrize('external',['isolated'],indirect=True)
def test_isolated_source_replay_occurs_outside_target_metadata_lock(external):
    source,f,_,_,published = external
    _,_,_,target,_ = source
    result = shelf.import_version(target.id,published['library_voice_id'],published['version'])
    assert result['imported'] and len(f.preprocessing_replays) >= 3
    assert not f.calls



def test_reimport_into_original_project_does_not_edit_original_files_or_metadata(external):
    source,_,_,_,published = external
    root,_,project,_,_ = source
    def snapshot():
        return {str(path.relative_to(root/project.id)):voice.sha256(path)
                for path in (root/project.id).rglob('*') if path.is_file()}
    before = snapshot()
    result = shelf.import_version(project.id,published['library_voice_id'],published['version'])
    assert not result['imported'] and snapshot() == before


def test_retired_reference_is_removed_from_future_keeper_proposals(external):
    _,_,_,record,published = external
    from dubbing_service.app.audiobook import preferred_voices as preferred
    preferred.record_rating(library_voice_id=published['library_voice_id'],version=published['version'],
        rating=5,project_id='disposable-fixture-source',role_id='fixture-narrator',key=record['key'],
        audio_sha256=record['audio_sha256'],traits={'presentation':'masculine'},evidence_id='fixture-approval')
    assert preferred.match_preferred({'presentation':'masculine','language':'en'})
    preferred.retire(library_voice_id=published['library_voice_id'],version=published['version'],
        reason='Fixture source-use approval revoked; exclude future proposals')
    assert not preferred.match_preferred({'presentation':'masculine','language':'en'})
