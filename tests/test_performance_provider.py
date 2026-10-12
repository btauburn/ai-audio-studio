"""CPU contract tests for the staged native-instruction provider.

HTTP is an explicit in-memory fake at urllib; no worker, model, port or network is
touched. Roles and segments are duck-typed stand-ins with the attributes the parent
provider reads. These prove staged provider behaviour, not a registered runtime.
"""
import json
import hashlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubbing_service.app.audiobook import fingerprint, performance_provider as pp  # noqa: E402
from dubbing_service.app.audiobook import runtime_binding  # noqa: E402
from dubbing_service.app.providers import tts  # noqa: E402
from service import faster_qwen3tts_base_direction_adapter as service_direction  # noqa: E402

PROFILE = pp.PROFILE
BASELINE = 'faster-base-032'
PROJECT = 'proj-1'


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self.payload).encode('utf-8')


def identity_for(profile, **overrides):
    contract = {'schema': 'qwen-base-worker-identity-v1', 'runtime_profile': profile,
                'model_id': 'Qwen3-TTS-Base', 'checkpoint_manifest_sha256': 'c' * 64,
                'implementation_sha256': {name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                                          for name in pp.WORKER_SOURCES},
                'fallback': 'disabled'}
    contract.update(overrides)
    return {'contract': contract, 'sha256': pp.sha256_bytes(pp.canonical_json_bytes(contract))}


def make_role(identity_id='id-1', role_id='role-1', backend='qwen3tts', mechanism='qwen_reference', status='LOCKED'):
    identity = SimpleNamespace(id=identity_id, status=status, backend=backend, mechanism=mechanism,
                               prompt_artifact=f'casting/voices/{identity_id}/prompt.npz', prompt_sha256='p' * 64,
                               reference_audio=f'casting/voices/{identity_id}/reference.wav',
                               reference_audio_sha256='r' * 64)
    return SimpleNamespace(id=role_id, voice_identity=identity, style_instruction='legacy delivery',
                           qwen3_language=None, voice_description='', description='')


def make_segment(text='Hello there, friend.'):
    return SimpleNamespace(text=text, emotion_text='angry legacy', emotion_alpha=0.5, playback_speed=1.0,
                           volume_gain=0.0, qwen3_language=None)


def seed_base(identity_id, text):
    return int(pp.sha256_bytes((identity_id + text).encode('utf-8'))[:8], 16)


def receipt_for(instruction, salt, identity_id, text, digest=None):
    instruction, salt = pp.resolve_controls(instruction, salt)
    base = seed_base(identity_id, text)
    forwarded = instruction is not None
    return {'policy': 'native-instruct-icl-x1', 'forwarded': forwarded,
            'instruct_sha256': pp.sha256_bytes(instruction.encode('utf-8')) if forwarded else None,
            'instruct_chars': len(instruction) if forwarded else 0, 'conditioning_mode': 'icl',
            'seed_salt': salt, 'seed_effective': base if salt == 0 else (base + salt) % 4294967296,
            'direction_sha256': digest or pp.component_sha256(instruction, salt)}


class FakeWorker:
    """Records every HTTP request. Responses are built from the request, then optionally mutated."""

    def __init__(self, identity, mutate=None, status=None):
        self.identity = identity
        self.mutate = mutate
        self.status = status
        self.requests = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        body = json.loads(request.data.decode('utf-8')) if request.data else None
        self.requests.append((request.get_method(), url, body))
        if url.endswith('/api/v1/tts/runtime-identity'):
            return FakeResponse(self.identity)
        if url.endswith('/api/v1/tts/character/synthesize'):
            if self.status is not None:
                raise RuntimeError(f'character voice synthesize failed: {self.status}')
            result = {'success': True, 'audio_file': 'outputs/qwen_identity_fake.wav', 'identity_id': body['identity_id'],
                      'prompt_sha256': body['prompt_sha256'], 'performance': body['performance'],
                      'worker_runtime_identity': self.identity, 'fallback': None,
                      'execution_profile': {'runtime_profile': self.identity['contract']['runtime_profile']},
                      'generation': {}}
            perf = body['performance']
            if 'direction_sha256' in perf:
                result['generation'] = {'direction': receipt_for(perf['native_instruction'], perf['seed_salt'],
                                                                 body['identity_id'], body['text'])}
                result['execution_profile']['runtime_profile'] = PROFILE
            if self.mutate is not None:
                self.mutate(result)
            return FakeResponse(result)
        raise AssertionError(f'unexpected request {request.get_method()} {url}')

    def synth_posts(self):
        return [r for r in self.requests if r[1].endswith('/api/v1/tts/character/synthesize')]


class ProviderCase(unittest.TestCase):
    def setUp(self):
        patches = [
            mock.patch('dubbing_service.app.providers.tts.get_config', return_value=SimpleNamespace(tts_api_key=None)),
            mock.patch('dubbing_service.app.providers.tts.get_qwen3tts_base_url', return_value='http://worker.invalid'),
            mock.patch('dubbing_service.app.audiobook.project_qwen_runtime.project_profile',
                       side_effect=lambda *_: PROFILE if pp._CALL.get() is not None else BASELINE),
            mock.patch.object(pp, 'canonical_direction_lookup', return_value=None),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def use_worker(self, worker):
        patcher = mock.patch('urllib.request.urlopen', side_effect=worker)
        patcher.start()
        self.addCleanup(patcher.stop)
        return worker

    def lookup(self, role, segment, *, instruction='Calm and warm.', salt=0, project=PROJECT, role_id=None,
               identity_id='id-1', text=None, digest=None):
        found = pp.ResolvedDirection(
            project_id=project, role_id=role_id or role.id, voice_identity_id=identity_id,
            utterance_text_sha256=fingerprint.text_sha256(text if text is not None else segment.text),
            native_instruction=instruction, seed_salt=salt,
            direction_sha256=digest or pp.component_sha256(instruction, salt))
        return lambda project_id, role_obj, seg: found

    def provider(self, lookup=None):
        return pp.DirectedQwen3TTSProvider(direction_lookup=lookup)

    def run_provider(self, provider, role, segment):
        return provider.synthesize_segment(PROJECT, role, segment)


class NoDirectionParityTests(ProviderCase):
    def test_no_lookup_sends_the_same_body_as_the_parent_provider(self):
        role, segment = make_role(), make_segment()
        parent_worker = self.use_worker(FakeWorker(identity_for(BASELINE)))
        parent_result = self.run_provider(tts.Qwen3TTSProvider(), role, segment)
        directed_worker = self.use_worker(FakeWorker(identity_for(BASELINE)))
        directed_result = self.run_provider(self.provider(), role, segment)
        self.assertTrue(parent_result.success and directed_result.success)
        self.assertEqual(parent_worker.synth_posts(), directed_worker.synth_posts())
        body = directed_worker.synth_posts()[0][2]
        self.assertNotIn('native_instruction', body['performance'])
        self.assertNotIn('direction_sha256', body['performance'])

    def test_lookup_returning_none_is_the_parent_path(self):
        role, segment = make_role(), make_segment()
        worker = self.use_worker(FakeWorker(identity_for(BASELINE)))
        result = self.run_provider(self.provider(lambda *_: None), role, segment)
        self.assertTrue(result.success)
        self.assertNotIn('direction_sha256', worker.synth_posts()[0][2]['performance'])

    def test_null_and_zero_resolved_direction_is_the_parent_path(self):
        role, segment = make_role(), make_segment()
        worker = self.use_worker(FakeWorker(identity_for(BASELINE)))
        result = self.run_provider(self.provider(self.lookup(role, segment, instruction='   ', salt=0)), role, segment)
        self.assertTrue(result.success)
        self.assertNotIn('native_instruction', worker.synth_posts()[0][2]['performance'])

    def test_legacy_fields_are_never_promoted_to_an_instruction(self):
        role, segment = make_role(), make_segment()
        worker = self.use_worker(FakeWorker(identity_for(BASELINE)))
        self.run_provider(self.provider(), role, segment)
        performance = worker.synth_posts()[0][2]['performance']
        self.assertEqual(performance['emotion'], 'angry legacy')
        self.assertNotIn('native_instruction', performance)

    def test_owner_instruction_is_used_and_legacy_emotion_is_not(self):
        role, segment = make_role(), make_segment()
        worker = self.use_worker(FakeWorker(identity_for(PROFILE)))
        result = self.run_provider(self.provider(self.lookup(role, segment, instruction='Soft.', salt=0)), role, segment)
        self.assertTrue(result.success)
        self.assertEqual(worker.synth_posts()[0][2]['performance']['native_instruction'], 'Soft.')


class BindingTests(ProviderCase):
    def test_positive_control_reaches_the_experimental_worker(self):
        role, segment = make_role(), make_segment()
        worker = self.use_worker(FakeWorker(identity_for(PROFILE)))
        result = self.run_provider(self.provider(self.lookup(role, segment)), role, segment)
        self.assertTrue(result.success)
        self.assertEqual(len(worker.synth_posts()), 1)

    def test_every_binding_mismatch_refuses_before_any_synthesis_request(self):
        role, segment = make_role(), make_segment()
        cases = {
            'other project': self.lookup(role, segment, project='proj-2'),
            'other role': self.lookup(role, segment, role_id='role-9'),
            'other voice identity': self.lookup(role, segment, identity_id='id-9'),
            'other utterance text': self.lookup(role, segment, text='Different text.'),
            'digest for other controls': self.lookup(role, segment, digest='e' * 64),
            'unsupported lookup value': lambda *_: True,
        }
        for name, lookup in cases.items():
            with self.subTest(name):
                worker = self.use_worker(FakeWorker(identity_for(PROFILE)))
                result = self.run_provider(self.provider(lookup), role, segment)
                self.assertFalse(result.success)
                self.assertEqual(worker.synth_posts(), [])

    def test_directed_non_character_role_is_refused_not_blended_into_voicedesign(self):
        role = make_role()
        role.voice_identity = None
        segment = make_segment()
        worker = self.use_worker(FakeWorker(identity_for(PROFILE)))
        result = self.run_provider(self.provider(self.lookup(role, segment, identity_id=None)), role, segment)
        self.assertFalse(result.success)
        self.assertIn('VoiceDesign is refused', result.raw['error'])
        self.assertEqual(worker.requests, [])

    def test_directed_indextts_or_unlocked_identity_is_refused(self):
        segment = make_segment()
        for kwargs in ({'backend': 'indextts'}, {'mechanism': 'vd'}, {'status': 'DRAFT'}):
            with self.subTest(kwargs):
                role = make_role(**kwargs)
                worker = self.use_worker(FakeWorker(identity_for(PROFILE)))
                result = self.run_provider(self.provider(self.lookup(role, segment)), role, segment)
                self.assertFalse(result.success)
                self.assertEqual(worker.synth_posts(), [])


class WorkerIdentityGateTests(ProviderCase):
    def test_default_worker_is_never_sent_a_direction(self):
        role, segment = make_role(), make_segment()
        worker = self.use_worker(FakeWorker(identity_for(BASELINE)))
        result = self.run_provider(self.provider(self.lookup(role, segment)), role, segment)
        self.assertFalse(result.success)
        self.assertIn('experimental direction profile', result.raw['error'])
        self.assertEqual(worker.synth_posts(), [])

    def test_native_worker_and_bad_identity_hash_are_refused(self):
        role, segment = make_role(), make_segment()
        bad_hash = {**identity_for(PROFILE), 'sha256': 'f' * 64}
        for name, identity in {'native': identity_for('native-base'), 'bad hash': bad_hash,
                               'fallback enabled': identity_for(PROFILE, fallback='cloud')}.items():
            with self.subTest(name):
                worker = self.use_worker(FakeWorker(identity))
                result = self.run_provider(self.provider(self.lookup(role, segment)), role, segment)
                self.assertFalse(result.success)
                self.assertEqual(worker.synth_posts(), [])

    def test_only_the_character_synthesize_endpoint_is_posted(self):
        role, segment = make_role(), make_segment()
        worker = self.use_worker(FakeWorker(identity_for(PROFILE)))
        self.run_provider(self.provider(self.lookup(role, segment)), role, segment)
        posted = [url for method, url, _ in worker.requests if method == 'POST']
        self.assertEqual(posted, ['http://worker.invalid/api/v1/tts/character/synthesize'])


class ReceiptVerificationTests(ProviderCase):
    def directed(self, mutate, instruction='Calm and warm.', salt=0):
        role, segment = make_role(), make_segment()
        worker = self.use_worker(FakeWorker(identity_for(PROFILE), mutate=mutate))
        result = self.run_provider(self.provider(self.lookup(role, segment, instruction=instruction, salt=salt)),
                                   role, segment)
        return result, worker

    def test_valid_instruction_receipt_is_success_with_exact_forwarding(self):
        result, worker = self.directed(None)
        self.assertTrue(result.success)
        self.assertEqual(result.audio_path, 'outputs/qwen_identity_fake.wav')
        self.assertEqual(worker.synth_posts()[0][2]['performance']['native_instruction'], 'Calm and warm.')

    def test_valid_seed_only_receipt_is_success_without_forwarding(self):
        result, worker = self.directed(None, instruction=None, salt=9)
        self.assertTrue(result.success)
        receipt = result.raw['generation']['direction']
        self.assertIs(receipt['forwarded'], False)
        self.assertEqual(receipt['seed_effective'], (seed_base('id-1', 'Hello there, friend.') + 9) % 4294967296)
        self.assertIsNone(worker.synth_posts()[0][2]['performance']['native_instruction'])
        self.assertEqual(worker.synth_posts()[0][2]['performance']['seed_salt'], 9)

    def test_missing_receipt_is_a_failure(self):
        def drop(result):
            result['generation'] = {}
        result, _ = self.directed(drop)
        self.assertFalse(result.success)

    def test_every_receipt_field_mismatch_is_a_failure(self):
        cases = {
            'policy': ('policy', 'other-policy'),
            'forwarded flag': ('forwarded', False),
            'instruct sha': ('instruct_sha256', 'a' * 64),
            'instruct chars': ('instruct_chars', 3),
            'conditioning mode': ('conditioning_mode', 'x_vector'),
            'salt': ('seed_salt', 4),
            'effective seed': ('seed_effective', 1),
            'digest': ('direction_sha256', 'b' * 64),
        }
        for name, (field, value) in cases.items():
            with self.subTest(name):
                def tamper(result, field=field, value=value):
                    result['generation']['direction'][field] = value
                result, _ = self.directed(tamper)
                self.assertFalse(result.success)

    def test_incomplete_receipt_is_a_failure(self):
        def drop_key(result):
            del result['generation']['direction']['seed_effective']
        result, _ = self.directed(drop_key)
        self.assertFalse(result.success)

    def test_wrong_runtime_fallback_voice_or_prompt_in_the_response_is_a_failure(self):
        mutations = {
            'runtime identity': lambda r: r.update(worker_runtime_identity=identity_for(PROFILE, runtime_profile=PROFILE, extra=1)),
            'fallback audio': lambda r: r.update(fallback='native-base'),
            'worker profile': lambda r: r['execution_profile'].update(runtime_profile=BASELINE),
            'voice identity': lambda r: r.update(identity_id='id-other'),
            'prompt': lambda r: r.update(prompt_sha256='0' * 64),
        }
        for name, mutate in mutations.items():
            with self.subTest(name):
                result, _ = self.directed(mutate)
                self.assertFalse(result.success)

    def test_worker_error_is_a_failure_not_a_candidate(self):
        role, segment = make_role(), make_segment()
        self.use_worker(FakeWorker(identity_for(PROFILE), status='cancelled before response'))
        result = self.run_provider(self.provider(self.lookup(role, segment)), role, segment)
        self.assertFalse(result.success)
        self.assertIn('cancelled before response', result.raw['error'])

    def test_receipt_seed_check_uses_the_runtime_formula(self):
        self.assertEqual(seed_base('id-1', 'Hello there, friend.'),
                         int(pp.sha256_bytes(('id-1' + 'Hello there, friend.').encode())[:8], 16))


class CrossModuleAgreementTests(unittest.TestCase):
    """The service side and the provider must agree byte-for-byte on the component digest and limits."""

    def test_component_digest_matches_across_packages(self):
        for instruction, salt in ((None, 0), ('Calm.', 0), ('Calm.', 7), (None, 2147483647), ('x' * 240, 1)):
            with self.subTest(instruction=instruction, salt=salt):
                self.assertEqual(pp.component_sha256(instruction, salt),
                                 service_direction.component_sha256(instruction, salt))

    def test_canonical_bytes_match_the_snapshot_fingerprint(self):
        value = {'control_policy': 'native-instruct-icl-x1', 'native_instruction': 'Calm, 中文', 'seed_salt': 0}
        self.assertEqual(pp.canonical_json_bytes(value), fingerprint.canonical_json_bytes(value))
        self.assertEqual(pp.canonical_json_bytes(value), service_direction.canonical_json_bytes(value))

    def test_limits_and_policy_are_the_frozen_contract(self):
        self.assertEqual(pp.MAX_INSTRUCTION_CHARS, 240)
        self.assertEqual(pp.SEED_SALT_MAX, 2147483647)
        self.assertEqual(pp.POLICY, 'native-instruct-icl-x1')
        self.assertEqual(pp.PROFILE, 'faster-base-032-direction-x1')

    def test_provider_loads_from_the_staged_workspace_not_the_live_checkout(self):
        workspace = str(Path(__file__).resolve().parents[1]).lower()
        self.assertTrue(pp.__file__.lower().startswith(workspace))
        self.assertTrue(runtime_binding.__file__.lower().startswith(workspace))


if __name__ == '__main__':
    unittest.main()
