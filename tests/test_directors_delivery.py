"""Private projection keeps existing auth while adding only director operations."""
from tests.test_voice_lab_delivery import private_lab


def test_director_assets_are_served_and_page_paths_rewritten(private_lab):
    _, client, _ = private_lab
    page = client.get('/voice-lab')
    assert '/assets/casting-director.js' in page.text
    assert '/assets/casting-director.css' in page.text
    assert client.get('/assets/casting-director.js').status_code == 200
    assert client.get('/assets/casting-director.css').status_code == 200


def test_director_read_proxy_and_nested_audio_urls(private_lab):
    _, client, state = private_lab
    state.upstream.body = {'roles': [{'candidates': [{'audio_url': '/api/v1/projects/project/voice-lab/audio/v_' + 'a' * 28}]}]}
    response = client.get('/api/voice-lab/project/directors')
    assert response.status_code == 200
    assert state.calls[0].url.endswith('/voice-lab/directors')
    assert response.json()['roles'][0]['candidates'][0]['audio_url'].startswith('/api/voice-lab/project/audio/')


def test_director_mutations_require_authority_and_forward_only_review_header(private_lab):
    config, client, state = private_lab
    endpoint = '/api/voice-lab/project/directors/analyze'
    assert client.post(endpoint).status_code == 403
    config['recipients']['brian']['allow_feedback'] = False
    assert client.post(endpoint, headers={'X-Review-Request': '1'}).status_code == 403
    assert state.calls == []
    config['recipients']['brian']['allow_feedback'] = True
    result = client.post(endpoint, headers={'X-Review-Request': '1', 'Authorization': 'never-forward'})
    assert result.status_code == 200
    assert state.calls[-1].headers == {'x-review-request': '1'}
