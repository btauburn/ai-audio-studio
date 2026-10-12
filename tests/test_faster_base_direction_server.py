"""CPU contract tests for the staged experimental Faster Base direction server.

The pinned baseline server and its native owner are replaced by explicit fakes, so
no real server, model, native import, port, uvicorn or health probe is initialized.
These prove the wrapper contract only; they do not prove a launched worker.
"""
import argparse
import hashlib
import importlib
import json
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from service import faster_qwen3tts_base_adapter as base_adapter  # noqa: E402
from service import faster_qwen3tts_base_direction_adapter as direction  # noqa: E402
from service.qwen_runtime_identity import source_snapshot  # noqa: E402
from tests.test_faster_base_direction_adapter import CONTROLS, ICL, FakeFast, FakeTorch  # noqa: E402

WORKSPACE = Path(__file__).resolve().parents[1]
BASELINE_PROFILE = 'faster-base-032'
PROFILE = direction.PROFILE
NEW_SOURCES = ('service/faster_qwen3tts_base_direction_server.py', 'service/faster_qwen3tts_base_direction_adapter.py')
BASELINE_SOURCES = ('service/faster_qwen3tts_base_server.py', 'service/faster_qwen3tts_base_adapter.py')
NATIVE_SOURCES = ('service/qwen3tts_server.py', 'service/qwen_character_voice.py', 'service/qwen_runtime_identity.py')
CHANGED_ONLY_BY_DIRECTION = 'faster_qwen3tts_base_direction_server.py'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


class FakeNative:
    """Explicit stand-in for service.qwen3tts_server; records identity construction."""

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--host', default='127.0.0.1')
        self.parser.add_argument('--port', type=int, default=8003)
        self.parser.add_argument('--base-model-dir', default='models/qwen3_tts/Qwen3-TTS-Base')
        self.parser.add_argument('--dtype', default='bfloat16')
        self.parser.add_argument('--device', default='cuda:0')
        self.source_snapshot = source_snapshot
        self.snapshots = []
        self._runtime_sources = source_snapshot(WORKSPACE, NATIVE_SOURCES)
        self._runtime_profile = 'native-base'
        self._runtime_snapshot = self.WorkerIdentitySnapshot(WORKSPACE, self._parse_server_args(),
                                                             self._runtime_sources, 'native-base')
        self.qwen3tts_model = None
        self.config = None
        self._model_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self.app = FastAPI()
        self.HTTPException = HTTPException
        self.logger = mock.MagicMock()

        @self.app.post('/api/v1/tts/character/prepare')
        async def prepare_echo(request: Request):
            return await request.json()

    def _parse_server_args(self):
        args, _unknown = self.parser.parse_known_args([])
        return args

    class WorkerIdentitySnapshot:
        """Records identity inputs; the real freezing and checkpoint checks need model files absent here."""

        def __init__(self, root, args, sources, profile):
            self.root, self.profile, self.sources = root, profile, dict(sources)
            self.args = args

        def read(self, args):
            contract = {'schema': 'qwen-base-worker-identity-v1', 'runtime_profile': self.profile,
                        'implementation_sha256': dict(self.sources), 'fallback': 'disabled'}
            return {'contract': contract, 'sha256': _digest(contract)}

    def base_runtime_identity(self):
        return self._runtime_snapshot.read(None)

    def _load_qwen3tts_model_if_needed(self, kind='voice_design', attention='auto'):
        return None


def build_fake_baseline(native):
    """Mirrors the baseline's rebinding surface: profile, adapter class, sources, character call, health."""
    module = SimpleNamespace()
    module.native = native
    module.PROFILE = BASELINE_PROFILE
    module.ROOT = WORKSPACE
    module.FasterBaseAdapter = base_adapter.FasterBaseAdapter
    module.app = native.app
    native._runtime_profile = BASELINE_PROFILE
    native._runtime_sources.update(native.source_snapshot(WORKSPACE, BASELINE_SOURCES))
    native.parser.add_argument('--execution-profile', choices=[BASELINE_PROFILE], default=BASELINE_PROFILE)
    native._runtime_snapshot = native.WorkerIdentitySnapshot(WORKSPACE, native._parse_server_args(),
                                                             native._runtime_sources, BASELINE_PROFILE)

    def character_call(method, **kwargs):
        with native._infer_lock:
            try:
                expected = kwargs.pop('expected_runtime_sha256', None)
                binding = native.base_runtime_identity() if method.__name__ == 'synthesize' else None
                if expected is not None and expected != binding['sha256']:
                    raise native.HTTPException(status_code=409, detail='Worker runtime changed before synthesis')
                result = method(**kwargs)
                if binding is not None:
                    result['worker_runtime_identity'] = binding
                if method.__name__ == 'synthesize' and isinstance(native.qwen3tts_model, module.FasterBaseAdapter):
                    result['execution_profile'].update(runtime_profile=module.PROFILE, faster_version='0.3.2', experimental=True)
                    result['generation'] = dict(native.qwen3tts_model.last_metadata)
                    result['fallback'] = None
                return result
            except (ValueError, FileNotFoundError) as exc:
                raise native.HTTPException(status_code=400, detail=str(exc)) from exc
            except native.HTTPException:
                raise
            except Exception as exc:
                raise native.HTTPException(status_code=500, detail=str(exc)) from exc

    @native.app.get('/health')
    async def experimental_health():
        return {'status': 'healthy', 'runtime_profile': module.PROFILE, 'faster_version': '0.3.2', 'experimental': True}

    native._character_call = character_call
    module.character_call = character_call
    return module


class ServerCase(unittest.TestCase):
    def setUp(self):
        self.native = FakeNative()
        self.baseline = build_fake_baseline(self.native)
        self.torch = FakeTorch()
        self.fast = FakeFast()
        generate_module = SimpleNamespace(fast_generate=lambda **kw: self.fast.generate(**kw))
        patcher = mock.patch.dict(sys.modules, {
            'service.qwen3tts_server': self.native,
            'service.faster_qwen3tts_base_server': self.baseline,
            'torch': self.torch,
            'faster_qwen3_tts': SimpleNamespace(),
            'faster_qwen3_tts.generate': generate_module,
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        sys.modules.pop('service.faster_qwen3tts_base_direction_server', None)
        self.server = importlib.import_module('service.faster_qwen3tts_base_direction_server')
        self.addCleanup(sys.modules.pop, 'service.faster_qwen3tts_base_direction_server', None)
        self.native.qwen3tts_model = self.server.DirectedFasterBaseAdapter(self.fast)
        self.calls = []

    def synthesize_method(self, *, fail=None, skip_generation=False):
        test = self

        def synthesize(project_id, identity_id='id-1', text='hello there', performance=None, **_):
            test.calls.append({'performance': performance, 'active': direction.active_direction()})
            if fail is not None:
                raise fail
            test.torch.manual_seed(int(hashlib.sha256((identity_id + text).encode('utf-8')).hexdigest()[:8], 16))
            if not skip_generation:
                test.native.qwen3tts_model.generate_voice_clone(
                    text=text, language='English', voice_clone_prompt=[ICL], max_new_tokens=40, **CONTROLS)
            return {'success': True, 'audio_file': None, 'identity_id': identity_id, 'prompt_sha256': 'p' * 64,
                    'performance': performance or {}, 'execution_profile': {}, 'fallback': None}
        return synthesize

    def request_performance(self, instruction, salt):
        digest = direction.component_sha256(instruction, salt)
        return {'native_instruction': instruction, 'seed_salt': salt, 'direction_sha256': digest,
                'emotion': 'legacy field stays where it was'}


class IdentityAndProfileTests(ServerCase):
    def test_identity_contains_profile_and_both_new_source_hashes_from_real_bytes(self):
        self.assertEqual(self.native._runtime_profile, PROFILE)
        snapshot = self.native._runtime_snapshot
        self.assertEqual(snapshot.profile, PROFILE)
        for name in NEW_SOURCES:
            expected = hashlib.sha256((WORKSPACE / name).read_bytes()).hexdigest()
            self.assertEqual(snapshot.sources[name], expected)
        for name in BASELINE_SOURCES + NATIVE_SOURCES:
            self.assertIn(name, snapshot.sources)
        identity = self.native.base_runtime_identity()
        self.assertEqual(identity['contract']['runtime_profile'], PROFILE)
        self.assertEqual(identity['sha256'], _digest(identity['contract']))

    def test_identity_changes_when_new_server_source_changes(self):
        before = self.native.base_runtime_identity()['sha256']
        sources = dict(self.native._runtime_snapshot.sources)
        sources[NEW_SOURCES[0]] = 'f' * 64
        changed = self.native.WorkerIdentitySnapshot(WORKSPACE, None, sources, PROFILE).read(None)
        self.assertNotEqual(changed['sha256'], before)
        self.assertTrue(NEW_SOURCES[0].endswith(CHANGED_ONLY_BY_DIRECTION))

    def test_single_execution_profile_option_accepts_only_the_direction_profile(self):
        options = [a for a in self.native.parser._actions if '--execution-profile' in a.option_strings]
        self.assertEqual(len(options), 1)
        self.assertEqual(options[0].default, PROFILE)
        accepted, _unknown = self.native.parser.parse_known_args(['--execution-profile', PROFILE])
        self.assertEqual(accepted.execution_profile, PROFILE)
        with mock.patch('sys.stderr'), self.assertRaises(SystemExit):
            self.native.parser.parse_known_args(['--execution-profile', BASELINE_PROFILE])

    def test_base_module_and_health_report_the_experimental_profile(self):
        self.assertEqual(self.baseline.PROFILE, PROFILE)
        self.assertIs(self.baseline.FasterBaseAdapter, self.server.DirectedFasterBaseAdapter)
        response = TestClient(self.native.app).get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['runtime_profile'], PROFILE)
        self.assertIs(response.json()['experimental'], True)


class SynthesizeWrapperTests(ServerCase):
    def test_positive_control_reaches_adapter_with_keys_stripped_and_receipt_attached(self):
        performance = self.request_performance('Calm and warm.', 5)
        self.native._character_call(self.synthesize_method(), project_id='p', identity_id='id-1',
                                    text='hello there', performance=performance)
        self.assertEqual(len(self.calls), 1)
        forwarded = self.calls[0]['performance']
        self.assertEqual(forwarded, {'emotion': 'legacy field stays where it was'})
        self.assertIsNotNone(self.calls[0]['active'])
        self.assertEqual(self.fast.prepare_calls[0]['instruct'], 'Calm and warm.')
        seed = int(hashlib.sha256(b'id-1hello there').hexdigest()[:8], 16)
        self.assertEqual(self.torch.manual_seed_calls, [seed, seed + 5])

    def test_result_receipt_is_the_adapter_receipt_and_matches_request(self):
        result = self.native._character_call(self.synthesize_method(), project_id='p', identity_id='id-1',
                                             text='hello there', performance=self.request_performance('Calm.', 5))
        receipt = result['generation']['direction']
        self.assertEqual(receipt['direction_sha256'], direction.component_sha256('Calm.', 5))
        self.assertEqual(receipt['seed_effective'], int(hashlib.sha256(b'id-1hello there').hexdigest()[:8], 16) + 5)
        self.assertIs(receipt['forwarded'], True)
        self.assertIn('worker_runtime_identity', result)
        self.assertEqual(result['execution_profile']['runtime_profile'], PROFILE)

    def test_no_direction_keys_is_parent_path_with_no_receipt(self):
        self.native._character_call(self.synthesize_method(), project_id='p', identity_id='id-1',
                                    text='hello there', performance={'emotion': 'x'})
        self.assertIsNone(self.calls[0]['active'])
        self.assertNotIn('instruct', self.fast.prepare_calls[0])

    def test_malformed_or_mismatched_digest_refused_before_synthesis(self):
        bad = self.request_performance('Calm.', 5)
        bad['direction_sha256'] = 'a' * 64
        method = self.synthesize_method()
        with self.assertRaises(HTTPException) as caught:
            self.native._character_call(method, project_id='p', identity_id='id-1', text='t', performance=bad)
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.fast.prepare_calls, [])

    def test_partial_direction_keys_refused(self):
        partial = {'native_instruction': 'Calm.'}
        with self.assertRaises(HTTPException):
            self.native._character_call(self.synthesize_method(), project_id='p', identity_id='id-1',
                                        text='t', performance=partial)
        self.assertEqual(self.calls, [])

    def test_missing_receipt_fails_closed_without_success(self):
        method = self.synthesize_method(skip_generation=True)
        with self.assertRaises(HTTPException) as caught:
            self.native._character_call(method, project_id='p', identity_id='id-1', text='t',
                                        performance=self.request_performance('Calm.', 5))
        self.assertEqual(caught.exception.status_code, 500)
        self.assertIsNone(direction.active_direction())

    def test_context_resets_after_success_error_and_cancellation(self):
        self.native._character_call(self.synthesize_method(), project_id='p', identity_id='id-1', text='hello there',
                                    performance=self.request_performance('Calm.', 5))
        self.assertIsNone(direction.active_direction())
        with self.assertRaises(HTTPException):
            self.native._character_call(self.synthesize_method(fail=ValueError('bad')), project_id='p',
                                        identity_id='id-1', text='t', performance=self.request_performance('Calm.', 5))
        self.assertIsNone(direction.active_direction())
        with self.assertRaises(HTTPException) as caught:
            self.native._character_call(self.synthesize_method(fail=RuntimeError('cancelled')), project_id='p',
                                        identity_id='id-1', text='t', performance=self.request_performance('Calm.', 5))
        self.assertEqual(caught.exception.status_code, 500)
        self.assertIsNone(direction.active_direction())

    def test_direction_context_is_isolated_per_thread(self):
        started = threading.Event()
        release = threading.Event()
        self.fast.hooks['talker'] = lambda: (started.set(), release.wait(5)) and None
        seen = {}

        def owner():
            self.native._character_call(self.synthesize_method(), project_id='p', identity_id='id-1', text='t',
                                        performance=self.request_performance('Calm.', 5))
        thread = threading.Thread(target=owner)
        thread.start()
        self.assertTrue(started.wait(5))
        seen['other_thread_active'] = direction.active_direction()
        release.set()
        thread.join(5)
        self.assertIsNotNone(self.calls[0]['active'])
        self.assertIsNone(seen['other_thread_active'])
        self.assertIsNone(direction.active_direction())


class OperationScopeTests(ServerCase):
    def test_direction_on_non_synthesize_method_refused_before_call(self):
        called = []

        def prepare(**kwargs):
            called.append(kwargs)
            return {'success': True}

        with self.assertRaises(HTTPException) as caught:
            self.native._character_call(prepare, project_id='p', identity_id='id-1',
                                        performance=self.request_performance('Calm.', 5))
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(called, [])

    def test_http_prepare_with_direction_keys_refused_by_route_guard(self):
        body = {'project_id': 'p', 'identity_id': 'id-1', 'performance': self.request_performance('Calm.', 5)}
        for path in ('/api/v1/tts/character/prepare', '/api/v1/tts/character/design',
                     '/api/v1/tts/character/acoustics', '/api/v1/tts/character/import-reference'):
            with self.subTest(path=path):
                response = TestClient(self.native.app).post(path, json=body)
                self.assertEqual(response.status_code, 400)
                self.assertIn('synthesize', response.json()['detail'])

    def test_http_route_without_direction_keys_passes_body_through_unchanged(self):
        body = {'project_id': 'p', 'identity_id': 'id-1', 'performance': {'emotion': 'x'}}
        response = TestClient(self.native.app).post('/api/v1/tts/character/prepare', json=body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), body)

    def test_malformed_body_mentioning_a_direction_key_is_refused(self):
        raw = b'{"performance": {"native_instruction": "Calm.", '
        response = TestClient(self.native.app).post('/api/v1/tts/character/prepare', content=raw,
                                                    headers={'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
