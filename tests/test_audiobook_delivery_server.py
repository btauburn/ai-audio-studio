import copy
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dubbing_service.app.audiobook.delivery import store
from dubbing_service.app.audiobook.delivery.server import create_app
from dubbing_service.app.audiobook.delivery.config import load_config


@pytest.fixture
def listening(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'project.json').write_text('{"id":"project"}')
    (project / 'segments.json').write_text('[]')
    exports = tmp_path / 'exports'
    exports.mkdir()
    audio = exports / 'chapter.mp3'
    audio.write_bytes(bytes(range(256)) * 32)
    config = {'project_dir':str(project),'state_dir':str(project/'delivery'),'port':8778,
              'listening_base_url':'https://desktop.example.ts.net:8443',
              'allowed_audio_roots':[str(exports)],
              'recipients':{'brian':{'id':'brian','enabled':True,'tailscale_login':'brian@example.test','allow_download':True,'allow_feedback':True},
                            'mk':{'id':'mk','enabled':False}}}
    projection = {'schema_version':1,'project_id':'project','book_title':'Super Supportive',
                  'chapters':[{'chapter_id':'chapter256','chapter_number':256,'title':'Chapter 256','duration_seconds':60.0,
                               'audio_path':str(audio),'audio_sha256':hashlib.sha256(audio.read_bytes()).hexdigest(),
                               'revision':'revision1','export_manifest_sha256':'exporthash','selected_takes':[{'segment_id':'s1','take_id':'t1','audio_sha256':'takehash'}]}]}
    batch = store.publish_batch(Path(config['state_dir']),projection,'brian')
    client = TestClient(create_app(config),base_url='http://127.0.0.1:8778',client=('127.0.0.1',54321),
                        headers={'Tailscale-User-Login':'brian@example.test'})
    yield config,batch,client,audio
    client.close()


def post(client,batch,body,headers=None):
    return client.post(f"/api/batches/{batch['batch_id']}/feedback",json=body,
                       headers={'X-Review-Request':'1',**(headers or {})})


def test_manifest_has_no_local_paths_or_source_text(listening):
    _,batch,client,_=listening
    response=client.get('/api/batches/'+batch['batch_id'])
    assert response.status_code==200
    public=response.json()
    assert len(public['chapters'])==1
    assert 'audio_path' not in response.text and 'selected_takes' not in response.text
    assert public['chapters'][0]['audio_sha256']==batch['chapters'][0]['audio_sha256']


@pytest.mark.parametrize('range_value,status,length', [('bytes=0-1023',206,1024),('bytes=1024-',206,7168),('bytes=-100',206,100),('bytes=8192-',416,None)])
def test_audio_ranges(listening,range_value,status,length):
    _,batch,client,audio=listening
    response=client.get(f"/audio/{batch['batch_id']}/chapter256",headers={'Range':range_value})
    assert response.status_code==status
    if length is not None:
        assert len(response.content)==length
        assert response.headers['accept-ranges']=='bytes'
        assert response.headers['content-type']=='audio/mpeg'
        assert response.content in audio.read_bytes()


def test_head_and_download(listening):
    _,batch,client,_=listening
    url=f"/audio/{batch['batch_id']}/chapter256"
    response=client.head(url)
    assert response.status_code==200 and not response.content
    assert response.headers['content-length']=='8192'
    assert client.get(url+'?download=true').headers['content-disposition'].startswith('attachment;')


@pytest.mark.parametrize('path',['/audio/BAD/../project.json','/audio/BAD/%2e%2e%2fproject.json','/audio/BAD/%5c..%5cproject.json','/assets/../config.json','/assets/config.py','/api/batches/%2e%2e%2fproject'])
def test_arbitrary_files_are_never_served(listening,path):
    _,_,client,_=listening
    assert client.get(path).status_code in (404,307)


def test_audio_bytes_changed_fail_closed(listening):
    _,batch,client,audio=listening
    audio.write_bytes(b'new-version')
    assert client.get(f"/audio/{batch['batch_id']}/chapter256").status_code==409


def test_manifest_audio_path_outside_export_allowlist(listening,tmp_path):
    config,batch,client,_=listening
    external=tmp_path/'private.mp3';external.write_bytes(b'private')
    path=Path(config['state_dir'])/'manifests'/(batch['batch_id']+'.json')
    altered=copy.deepcopy(batch);altered['chapters'][0]['audio_path']=str(external)
    path.write_text(json.dumps(altered))
    assert client.get(f"/audio/{batch['batch_id']}/chapter256").status_code==409


@pytest.mark.parametrize('headers', [{'Tailscale-User-Login':''},{'Tailscale-User-Login':'mk@example.test'},{'Host':'evil.example'}])
def test_private_identity_and_host_required(listening,headers):
    _,_,client,_=listening
    assert client.get('/health',headers=headers).status_code==403


def test_network_peer_cannot_forge_proxy_identity(listening):
    config,_,_,_=listening
    with TestClient(create_app(config),base_url='http://127.0.0.1:8778',client=('192.168.1.25',8888),headers={'Tailscale-User-Login':'brian@example.test','X-Forwarded-For':'127.0.0.1'}) as client:
        assert client.get('/health').status_code==403


def test_disabled_brian_denied(listening):
    config,_,client,_=listening
    config['recipients']['brian']['enabled']=False
    assert client.get('/health').status_code==403


def test_feedback_exact_binding_and_no_canonical_mutation(listening):
    config,batch,client,_=listening
    project=Path(config['project_dir'])
    before={name:(project/name).read_bytes() for name in ('project.json','segments.json')}
    response=post(client,batch,{'chapter_id':'chapter256','playback_seconds':12.75,'text':'Test feedback'})
    assert response.status_code==200 and response.json()['saved']
    records=store.list_feedback(Path(config['state_dir']))
    assert len(records)==1
    record=records[0]
    assert record['project_id']=='project' and record['batch_id']==batch['batch_id']
    assert record['playback_seconds']==12.75 and record['text']=='Test feedback'
    assert record['artifact_bindings'][0]['revision']=='revision1'
    assert record['artifact_bindings'][0]['audio_sha256']==batch['chapters'][0]['audio_sha256']
    assert record['created_at']
    assert before=={name:(project/name).read_bytes() for name in before}
    assert client.get('/api/batches/'+batch['batch_id']+'/feedback').json()['feedback']==records


def test_old_feedback_keeps_its_version_when_new_batch_exists(listening):
    config,batch,client,_=listening
    post(client,batch,{'chapter_id':'chapter256','playback_seconds':12,'text':'Old version'})
    changed=copy.deepcopy(batch);changed['chapters'][0]['revision']='revision2';changed['chapters'][0]['audio_sha256']='newhash'
    new=store.publish_batch(Path(config['state_dir']),changed,'brian')
    assert new['batch_id']!=batch['batch_id']
    assert client.get('/api/batches/'+new['batch_id']+'/feedback').json()=={'feedback':[]}
    assert store.list_feedback(Path(config['state_dir']))[0]['artifact_bindings'][0]['revision']=='revision1'


@pytest.mark.parametrize('position',[-0.01,60.01,float('inf'),float('nan')])
def test_invalid_timestamp_refused(listening,position):
    _,batch,client,_=listening
    response=client.post('/api/batches/'+batch['batch_id']+'/feedback',content=json.dumps({'chapter_id':'chapter256','playback_seconds':position,'text':'note'}),headers={'Content-Type':'application/json','X-Review-Request':'1'})
    assert response.status_code==422


@pytest.mark.parametrize('position',[0,60])
def test_timestamp_boundaries(listening,position):
    _,batch,client,_=listening
    assert post(client,batch,{'chapter_id':'chapter256','playback_seconds':position,'text':'note'}).status_code==200


def test_overall_binds_whole_batch(listening):
    config,batch,client,_=listening
    assert post(client,batch,{'text':'Overall'}).status_code==200
    record=store.list_feedback(Path(config['state_dir']))[0]
    assert record['kind']=='overall' and record['playback_seconds'] is None
    assert len(record['artifact_bindings'])==len(batch['chapters'])


@pytest.mark.parametrize('body',[{'text':' '},{'text':'a'*4001},{'text':'note','chapter_id':'../../private'},{'text':'note','audio_path':'C:/evil'},{'text':'note','playback_seconds':1}])
def test_feedback_input_validation(listening,body):
    config,batch,client,_=listening
    assert post(client,batch,body).status_code in (404,422)
    assert store.list_feedback(Path(config['state_dir']))==[]


def test_csrf_and_size_limits(listening):
    _,batch,client,_=listening
    endpoint='/api/batches/'+batch['batch_id']+'/feedback'
    assert client.post(endpoint,json={'text':'bad'}).status_code==403
    assert post(client,batch,{'text':'bad'},headers={'Origin':'https://evil.example'}).status_code==403
    assert client.post(endpoint,content='x'*16385,headers={'X-Review-Request':'1','Content-Type':'application/json'}).status_code==413


def test_same_batch_publication_is_immutable(listening):
    config,batch,_,_=listening
    path=Path(config['state_dir'])/'manifests'/(batch['batch_id']+'.json')
    before=path.read_bytes()
    assert store.publish_batch(Path(config['state_dir']),batch,'brian')==batch
    assert path.read_bytes()==before


def test_mk_cannot_be_enabled_in_v1(listening,tmp_path):
    config,_,_,_=listening
    config['recipients']['mk']['enabled']=True
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='disabled MK'):
        load_config(path)


def test_mobile_page_security_headers_and_assets(listening):
    _,batch,client,_=listening
    response=client.get('/review/'+batch['batch_id'])
    assert response.status_code==200 and 'width=device-width' in response.text
    assert response.headers['x-frame-options']=='DENY'
    assert "frame-ancestors 'none'" in response.headers['content-security-policy']
    assert client.get('/assets/player.js').status_code==200
    assert client.get('/assets/player.css').status_code==200
    assert client.get('/docs').status_code==404
