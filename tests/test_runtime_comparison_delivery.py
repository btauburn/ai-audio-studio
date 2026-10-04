import pytest
from tests.test_voice_lab_delivery import private_lab,Upstream

def test_private_runtime_pairs_rewrite_urls_and_load_external_asset(private_lab):
    _,client,state=private_lab
    state.upstream=Upstream(body={'pairs':[{'baseline':{'audio_url':'/api/v1/projects/project/voice-lab/runtime-comparison/audio/short-narrator/baseline'}}]})
    response=client.get('/api/voice-lab/project/runtime-comparison')
    assert response.status_code==200
    assert response.json()['pairs'][0]['baseline']['audio_url']=='/api/voice-lab/project/runtime-comparison/audio/short-narrator/baseline'
    assert state.calls[-1].method=='GET'
    assert client.get('/assets/runtime-comparison.js').status_code==200
    page=client.get('/voice-lab?project=project')
    assert 'src="/assets/runtime-comparison.js"' in page.text

@pytest.mark.parametrize('method',['GET','HEAD'])
def test_private_runtime_audio_preserves_ranged_read(method,private_lab):
    _,client,state=private_lab
    state.upstream=Upstream(chunks=(b'abc',),headers={'content-type':'audio/wav','content-range':'bytes 0-2/9','accept-ranges':'bytes','etag':'pinned'},status=206)
    response=client.request(method,'/api/voice-lab/project/runtime-comparison/audio/short-narrator/candidate',headers={'Range':'bytes=0-2'})
    assert response.status_code==206
    assert state.calls[-1].method==method and state.calls[-1].headers['range']=='bytes=0-2'
    assert response.headers['content-range']=='bytes 0-2/9' and state.upstream.closed

@pytest.mark.parametrize('path',['/api/voice-lab/project/runtime-comparison','/api/voice-lab/project/runtime-comparison/audio/short/candidate'])
def test_private_runtime_comparison_refuses_post(path,private_lab):
    _,client,state=private_lab
    assert client.post(path,json={},headers={'X-Review-Request':'1'}).status_code==404 and not state.calls

def test_private_runtime_comparison_refuses_unknown_lane_or_project(private_lab):
    _,client,state=private_lab
    assert client.get('/api/voice-lab/project/runtime-comparison/audio/short/evil').status_code==404
    assert client.get('/api/voice-lab/secret/runtime-comparison').status_code==404
    assert not state.calls
