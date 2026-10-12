"""CPU-only source preservation and immutable chapter review regressions."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('cloned_chapter_script',Path(__file__).parents[1]/'scripts/render_cloned_chapters.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def mapped(text):
    return {'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'quotes':[{'quote_index':i,'start_offset':m.start(),'end_offset':m.end(),'speaker':'Jane',
                       'kind':'dialogue','confidence':0.95,'reason':'Source tag'} for i,m in enumerate(module.quote_spans(text))]}


def test_lossless_partition_with_mixed_and_open_source_quotes():
    text='001\n\nJane said, “We should go." Erick nodded.\n\n“Come on.\n\nHe followed.'
    mapping=mapped(text);rows=module.partition_chapter({'text':text},mapping)
    assert len(mapping['quotes'])==2
    assert ''.join(''.join(r['text'].split()) for r in rows)==''.join(text.split())
    assert [r['speaker'] for r in rows if r['kind']=='dialogue']==['Jane','Jane']
    assert all(text[r['start_offset']:r['end_offset']]==r['text'] for r in rows)


@pytest.mark.parametrize('change,error',[
    (lambda m:m.update(text_sha256=hashlib.sha256(b'other book').hexdigest()),'different source'),
    (lambda m:m['quotes'].pop(),'Every quoted'),
    (lambda m:m['quotes'][0].update(speaker=None),'Unresolved'),
    (lambda m:m['quotes'][0].update(start_offset=0),'offsets'),
    (lambda m:m['quotes'][0].update(kind='narrative_quote'),'narrator'),
])
def test_source_and_attribution_holds_are_not_narrator_fallbacks(change,error):
    text='Jane replied, “We should go.”';mapping=mapped(text);change(mapping)
    with pytest.raises(ValueError,match=error):module.partition_chapter({'text':text},mapping)


def test_immutable_publication_retry_and_conflict(tmp_path):
    p=tmp_path/'receipt.json';module.publish(p,b'first');module.publish(p,b'first')
    with pytest.raises(ValueError,match='conflict'):module.publish(p,b'different')
    assert p.read_bytes()==b'first'


def test_rebuild_rejects_wrong_epub_before_any_audio_assembly(monkeypatch,tmp_path):
    from dubbing_service.app.audiobook import voice_lab_chapter_export as exporter,epub_chapter_selection
    from dubbing_service.app.storage import projects
    project=SimpleNamespace(chapters=[SimpleNamespace(id='c',title='001')],source_revisions=[])
    root=tmp_path/'approved';root.mkdir();(root/'project.json').write_bytes(b'unchanged');(root/'segments.json').write_bytes(b'unchanged')
    monkeypatch.setattr(projects,'load_project',lambda p:project)
    from dubbing_service.app.core import config
    monkeypatch.setattr(config,'safe_project_root',lambda p:root)
    monkeypatch.setattr(epub_chapter_selection,'select_epub_chapters',lambda *a,**k:{'source_sha256':hashlib.sha256(b'actual epub').hexdigest(),
        'chapters':[{'chapter_number':1,'title':'001','text':'Full source.'}]})
    monkeypatch.setattr(exporter,'_source_snapshot',lambda *a:(project,[],{}))
    monkeypatch.setattr(exporter,'_take_snapshot',lambda *a:pytest.fail('Must not inspect audio for a different source'))
    epub=tmp_path/'source.epub';epub.write_bytes(b'actual epub')
    args=SimpleNamespace(epub=epub,reuse_projects=['a','b','c'],output_dir=tmp_path/'exports')
    with pytest.raises(ValueError,match='exact EPUB'):module.rebuild_approved(args)
    assert (root/'project.json').read_bytes()==b'unchanged'


def test_cloned_pipeline_allows_valid_clone_to_reach_render_entrypoint(monkeypatch):
    from tests.audiobook_helpers import build_project
    from dubbing_service.app.audiobook import pipeline
    from dubbing_service.app.models.project import Role,Segment
    from dubbing_service.app.models.character import CharacterVoiceIdentity
    from dubbing_service.app.storage.projects import save_project
    from dubbing_service.app.storage.segments import save_segments,load_segments
    p=build_project();p.metadata['director_voice_policy']={'mode':'cloned_only'}
    p.character_cast.policy.enabled=False
    identity=CharacterVoiceIdentity(id='voice',role_id='role',mechanism='qwen_reference',model='fixture',
        dossier_revision='fixture',casting_profile_revision='fixture',design_instruction='fixture',
        reference_audio='ref.wav',reference_audio_sha256=hashlib.sha256(b'fixture reference').hexdigest(),
        prompt_artifact='prompt.bin',prompt_sha256=hashlib.sha256(b'fixture prompt').hexdigest(),
        provenance={'origin':'eligible_reference_clone','actor_audio_used':True})
    p.roles=[Role(id='role',name='Erick',tts_backend='qwen3tts',voice_identity=identity)];save_project(p)
    save_segments(p.id,[Segment(id='line',index=0,role_id='role',text='A signal fixture for the cloned-only synthesis guard.')])
    class FixtureRenderer:
        renderer_id='fixture';engine_id='fixture';backend='qwen3tts'
    # The real render entrypoint must reach render_segment for a valid clone.
    reached=[]
    def stop(p,s,**kwargs):
        reached.append(s.id)
        raise RuntimeError('reached renderer boundary')
    monkeypatch.setattr(pipeline,'render_segment',stop)
    with pytest.raises(RuntimeError,match='reached renderer'):pipeline.render_utterances(p.id,['line'],renderer=FixtureRenderer(),promote=False)
    assert reached==['line']
    assert load_segments(p.id)[0].selected_take_id is None
