"""Real Windows reader locks must not break an otherwise valid metadata import."""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dubbing_service.app.main import create_app
from dubbing_service.app.storage import atomic
from tests.test_voice_data_consistency import _configure_isolated_storage


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows file-sharing semantics')
@pytest.mark.parametrize('target_name', ['project.json', 'segments.json'])
def test_import_retries_same_stage_after_a_real_short_reader(tmp_path, monkeypatch, target_name):
    projects_root, _ = _configure_isolated_storage(tmp_path, monkeypatch)
    with TestClient(create_app()) as client:
        response = client.post('/api/v1/projects', json={'name': 'reader-fixture'})
        assert response.status_code == 200
        project_id = response.json()['id']
        root = projects_root / project_id
        target = root / target_name
        if target_name == 'segments.json':
            target.write_text('[]', encoding='utf-8')
        real_replace = atomic.os.replace
        attempted_stages = []

        def replace_with_short_reader(source, destination):
            if Path(destination) == target:
                attempted_stages.append(str(source))
                if len(attempted_stages) == 1:
                    # This invokes the real filesystem error while the ordinary
                    # reader is open; the handle closes before the next attempt.
                    with target.open('rb'):
                        return real_replace(source, destination)
            return real_replace(source, destination)

        monkeypatch.setattr(atomic.os, 'replace', replace_with_short_reader)
        response = client.post(f'/api/v1/projects/{project_id}/segments/import-json',
            json={'json_text': json.dumps([{'speaker_name': 'Fixture', 'text': 'Fixture line.'}])})
        assert response.status_code == 200, response.text
        assert len(attempted_stages) == 2
        assert attempted_stages[0] == attempted_stages[1]
        project = json.loads((root / 'project.json').read_text('utf-8'))
        segments = json.loads((root / 'segments.json').read_text('utf-8'))
        role = next(role for role in project['roles'] if role['name'] == 'Fixture')
        assert segments[0]['role_id'] == role['id']
        assert not list(root.glob('*.stage'))
        assert not list(root.glob('*.backup'))
