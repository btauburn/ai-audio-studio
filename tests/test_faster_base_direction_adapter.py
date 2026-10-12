"""CPU contract tests for the staged native-instruction Faster Base adapter.

Explicit fakes replace torch, the Faster library, token decode and waveform
output. These prove the adapter's wrapper contract, not GPU or model behaviour.
"""
import contextlib
import hashlib
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from service.faster_qwen3tts_base_adapter import FasterBaseAdapter, GenerationCancelled  # noqa: E402
from service import faster_qwen3tts_base_direction_adapter as direction_module  # noqa: E402
from service.faster_qwen3tts_base_direction_adapter import (  # noqa: E402
    DirectedFasterBaseAdapter, Direction, DirectionError, component_sha256, direction_scope,
    parse_request, resolve_controls,
)

CONTROLS = dict(do_sample=True, top_k=50, top_p=1., temperature=.9, repetition_penalty=1.05,
                subtalker_dosample=True, subtalker_top_k=50, subtalker_top_p=1., subtalker_temperature=.9,
                non_streaming_mode=True)
ICL = SimpleNamespace(icl_mode=True, x_vector_only_mode=False, ref_code=[1, 2, 3])
XVEC = SimpleNamespace(icl_mode=False, x_vector_only_mode=True, ref_code=None)
TOKENS_PER_WORD = 3


class FakeTorch:
    """Explicit RNG and inference-mode fake; records every seed mutation."""

    def __init__(self):
        self.seed = 0
        self.manual_seed_calls = []

    def manual_seed(self, value):
        self.manual_seed_calls.append(value)
        self.seed = value

    def initial_seed(self):
        return self.seed

    def inference_mode(self):
        return contextlib.nullcontext()


class FakeTensor:
    def __init__(self, *shape):
        self.shape = shape

    def unsqueeze(self, dim):
        return self


class FakeGraph:
    def run(self, *args, **kwargs):
        return 'ok'


class FakeTokenizer:
    def decode(self, payload):
        return [np.full(240, 0.1, dtype=np.float32)], 24000


class FakeFast:
    """Stands in for the installed Faster library at the seams the adapter touches."""

    def __init__(self, base_prefill=100, steps=20):
        self.base_prefill = base_prefill
        self.steps = steps
        self.talker_graph = FakeGraph()
        self.predictor_graph = FakeGraph()
        self.speech_tokenizer = FakeTokenizer()
        self.prepare_calls = []
        self.generate_calls = 0
        self.hooks = {}
        self.prepare_error = None

    def _prepare_generation(self, **kwargs):
        self.prepare_calls.append(dict(kwargs))
        if self.prepare_error is not None:
            raise self.prepare_error
        instruct = kwargs.get('instruct')
        prefill = self.base_prefill + (len(instruct.split()) * TOKENS_PER_WORD if instruct else 0)
        model = SimpleNamespace(speech_tokenizer=self.speech_tokenizer)
        return model, None, None, FakeTensor(1, prefill), None, None, None, None

    def generate(self, *, talker_graph, predictor_graph, **kwargs):
        self.generate_calls += 1
        for _ in range(self.steps):
            self._hook('talker')
            talker_graph.run()
            self._hook('predictor')
            predictor_graph.run()
        self._hook('before_waveform')
        return FakeTensor(self.steps, 16), {}

    def _hook(self, phase):
        action = self.hooks.get(phase)
        if action is not None:
            action()


class AdapterCase(unittest.TestCase):
    def setUp(self):
        self.torch = FakeTorch()
        self.fast = FakeFast()
        generate_module = SimpleNamespace(fast_generate=lambda **kw: self.fast.generate(**kw))
        packages = {'torch': self.torch, 'faster_qwen3_tts': SimpleNamespace(),
                    'faster_qwen3_tts.generate': generate_module}
        patcher = mock.patch.dict(sys.modules, packages)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.adapter = DirectedFasterBaseAdapter(self.fast)

    def run_once(self, direction=None, prompt=None, budget=40, text='hello there', adapter=None):
        self.torch.seed = 1000
        target = adapter or self.adapter
        with direction_scope(direction):
            return target.generate_voice_clone(text=text, language='English',
                                               voice_clone_prompt=[ICL] if prompt is None else prompt,
                                               max_new_tokens=budget, **CONTROLS)

    def assert_restored(self, fast=None):
        fast = fast or self.fast
        self.assertNotIn('_prepare_generation', vars(fast))
        self.assertFalse(self.adapter._direction_guard.locked())
        self.assertFalse(self.adapter.lock.locked())


class ParentParityTests(AdapterCase):
    def test_no_direction_passes_identical_keywords_to_prepare(self):
        parent_fast = FakeFast()
        parent = FasterBaseAdapter(parent_fast)
        self.torch.seed = 1000
        parent.generate_voice_clone(text='hello there', language='English', voice_clone_prompt=[ICL],
                                    max_new_tokens=40, **CONTROLS)
        self.run_once()
        self.assertEqual(self.fast.prepare_calls, parent_fast.prepare_calls)
        self.assertNotIn('instruct', self.fast.prepare_calls[0])
        self.assertEqual(self.torch.manual_seed_calls, [])
        self.assertNotIn('direction', self.adapter.last_metadata)

    def test_null_instruction_and_zero_salt_take_parent_path(self):
        digest = component_sha256(None, 0)
        self.run_once(Direction(None, 0, digest))
        self.assertNotIn('instruct', self.fast.prepare_calls[0])
        self.assertEqual(self.torch.manual_seed_calls, [])
        self.assertNotIn('direction', self.adapter.last_metadata)


class DirectionForwardingTests(AdapterCase):
    def test_instruction_reaches_prepare_exactly_and_receipt_is_strict(self):
        text = 'Calm, warm and unhurried.'
        direction = Direction(text, 0, component_sha256(text, 0))
        self.run_once(direction)
        self.assertEqual(self.fast.prepare_calls[0]['instruct'], text)
        receipt = self.adapter.last_metadata['direction']
        self.assertEqual(set(receipt), {'policy', 'forwarded', 'instruct_sha256', 'instruct_chars',
                                        'conditioning_mode', 'seed_salt', 'seed_effective', 'direction_sha256'})
        self.assertEqual(receipt['policy'], 'native-instruct-icl-x1')
        self.assertIs(receipt['forwarded'], True)
        self.assertEqual(receipt['instruct_sha256'], hashlib.sha256(text.encode('utf-8')).hexdigest())
        self.assertEqual(receipt['instruct_chars'], len(text))
        self.assertEqual(receipt['conditioning_mode'], 'icl')
        self.assertEqual(receipt['direction_sha256'], direction.direction_sha256)
        self.assertEqual(self.adapter.last_metadata['prefill_tokens'], 100 + len(text.split()) * TOKENS_PER_WORD)
        self.assert_restored()

    def test_seed_only_adds_no_instruct_argument_and_reseeds_after_runtime(self):
        direction = Direction(None, 7, component_sha256(None, 7))
        self.run_once(direction)
        self.assertNotIn('instruct', self.fast.prepare_calls[0])
        self.assertEqual(self.torch.manual_seed_calls, [1007])
        receipt = self.adapter.last_metadata['direction']
        self.assertIs(receipt['forwarded'], False)
        self.assertIsNone(receipt['instruct_sha256'])
        self.assertEqual(receipt['instruct_chars'], 0)
        self.assertEqual(receipt['seed_salt'], 7)
        self.assertEqual(receipt['seed_effective'], 1007)

    def test_zero_salt_makes_no_additional_seed_change(self):
        text = 'Bright and steady.'
        self.run_once(Direction(text, 0, component_sha256(text, 0)))
        self.assertEqual(self.torch.manual_seed_calls, [])
        self.assertEqual(self.adapter.last_metadata['direction']['seed_effective'], 1000)

    def test_seed_wraps_modulo_two_to_the_thirty_two(self):
        self.torch.seed = 4294967295
        direction = Direction(None, 2, component_sha256(None, 2))
        with direction_scope(direction):
            self.adapter.generate_voice_clone(text='hello there', language='English', voice_clone_prompt=[ICL],
                                              max_new_tokens=40, **CONTROLS)
        self.assertEqual(self.adapter.last_metadata['direction']['seed_effective'], 1)
        self.assertEqual(self.torch.manual_seed_calls, [1])


class RefusalTests(AdapterCase):
    def test_x_vector_prompt_refused_before_any_generation(self):
        direction = Direction('calm', 0, component_sha256('calm', 0))
        with self.assertRaisesRegex(DirectionError, 'icl'):
            self.run_once(direction, prompt=[XVEC])
        self.assertEqual(self.fast.prepare_calls, [])
        self.assertEqual(self.fast.generate_calls, 0)
        self.assert_restored()

    def test_unknown_conditioning_mode_refused(self):
        direction = Direction('calm', 0, component_sha256('calm', 0))
        unknown = SimpleNamespace(icl_mode=None, x_vector_only_mode=False, ref_code=[1])
        with self.assertRaises(DirectionError):
            self.run_once(direction, prompt=[unknown])
        self.assertEqual(self.fast.prepare_calls, [])

    def test_missing_reference_codes_refused(self):
        direction = Direction('calm', 0, component_sha256('calm', 0))
        empty = SimpleNamespace(icl_mode=True, x_vector_only_mode=False, ref_code=None)
        with self.assertRaises(DirectionError):
            self.run_once(direction, prompt=[empty])
        self.assertEqual(self.fast.generate_calls, 0)

    def test_instruction_tokens_count_toward_the_parent_capacity_refusal(self):
        self.fast.base_prefill = 2000
        self.run_once(None, budget=40)
        self.assertEqual(self.adapter.last_metadata['prefill_tokens'], 2000)
        short = Direction('two words', 0, component_sha256('two words', 0))
        self.run_once(short, budget=40)
        self.assertEqual(self.adapter.last_metadata['prefill_tokens'], 2000 + 2 * TOKENS_PER_WORD)
        long_text = ' '.join(['word'] * 14)
        long = Direction(long_text, 0, component_sha256(long_text, 0))
        with self.assertRaisesRegex(ValueError, 'static cache capacity'):
            self.run_once(long, budget=40)
        self.assert_restored()
        self.assertNotIn('direction', self.adapter.last_metadata)

    def test_boundary_240_accepted_241_refused(self):
        self.assertEqual(resolve_controls('x' * 240, 0)[0], 'x' * 240)
        with self.assertRaisesRegex(DirectionError, '240'):
            resolve_controls('x' * 241, 0)

    def test_malicious_and_malformed_directives_refused(self):
        bad_instructions = ['line\nbreak', 'nul\x00byte', 'lone\ud800surrogate', 5, b'bytes']
        for value in bad_instructions:
            with self.subTest(value=repr(value)):
                with self.assertRaises(DirectionError):
                    resolve_controls(value, 0)
        for salt in (True, False, -1, 2147483648, 1.0, '1', None):
            with self.subTest(salt=repr(salt)):
                with self.assertRaises(DirectionError):
                    resolve_controls(None, salt)
        good = {'native_instruction': 'calm', 'seed_salt': 0, 'direction_sha256': component_sha256('calm', 0)}
        self.assertIsNotNone(parse_request(good))
        for name, bad in {
            'wrong digest': {**good, 'direction_sha256': 'a' * 64},
            'uppercase digest': {**good, 'direction_sha256': good['direction_sha256'].upper()},
            'short digest': {**good, 'direction_sha256': 'ab'},
            'missing digest': {k: v for k, v in good.items() if k != 'direction_sha256'},
            'missing salt': {k: v for k, v in good.items() if k != 'seed_salt'},
            'digest for other controls': {**good, 'seed_salt': 1},
        }.items():
            with self.subTest(name):
                with self.assertRaises(DirectionError):
                    parse_request(bad)

    def test_only_null_and_zero_request_is_none(self):
        self.assertIsNone(parse_request(None))
        self.assertIsNone(parse_request({'emotion': 'angry'}))
        self.assertIsNone(parse_request({'native_instruction': '   ', 'seed_salt': 0,
                                         'direction_sha256': component_sha256(None, 0)}))


class ScopeAndRestorationTests(AdapterCase):
    def test_success_restores_seam_and_guard(self):
        self.run_once(Direction('calm', 0, component_sha256('calm', 0)))
        self.assert_restored()

    def test_prepare_error_restores_seam_and_publishes_no_receipt(self):
        self.fast.prepare_error = RuntimeError('prepare failed')
        with self.assertRaisesRegex(RuntimeError, 'prepare failed'):
            self.run_once(Direction('calm', 0, component_sha256('calm', 0)))
        self.assert_restored()
        self.assertNotIn('direction', self.adapter.last_metadata)

    def test_stale_receipt_cleared_by_a_later_refusal(self):
        self.run_once(Direction('calm', 0, component_sha256('calm', 0)))
        self.assertIn('direction', self.adapter.last_metadata)
        with self.assertRaises(DirectionError):
            self.run_once(Direction('calm', 0, component_sha256('calm', 0)), prompt=[XVEC])
        self.assertNotIn('direction', self.adapter.last_metadata)

    def test_stale_receipt_cleared_by_a_later_parent_path_failure(self):
        self.run_once(Direction('calm', 0, component_sha256('calm', 0)))
        self.fast.prepare_error = RuntimeError('boom')
        with self.assertRaises(RuntimeError):
            self.run_once(None)
        self.assertNotIn('direction', self.adapter.last_metadata)

    def test_nested_scopes_restore_the_outer_direction(self):
        outer = Direction('outer', 0, component_sha256('outer', 0))
        inner = Direction('inner', 0, component_sha256('inner', 0))
        with direction_scope(outer):
            with direction_scope(inner):
                self.assertIs(direction_module.active_direction(), inner)
            self.assertIs(direction_module.active_direction(), outer)
        self.assertIsNone(direction_module.active_direction())

    def test_concurrent_directed_request_is_refused_and_owner_is_undisturbed(self):
        hold = threading.Event()
        release = threading.Event()
        self.fast.hooks['talker'] = lambda: (hold.set(), release.wait(5)) and None
        results = {}

        def owner():
            try:
                results['owner'] = self.run_once(Direction('owner', 0, component_sha256('owner', 0)))
            except Exception as exc:  # pragma: no cover - reported by assertion below
                results['owner_error'] = exc

        thread = threading.Thread(target=owner)
        thread.start()
        self.assertTrue(hold.wait(5))
        self.assertIn('_prepare_generation', vars(self.fast))
        with self.assertRaisesRegex(RuntimeError, 'one serial'):
            self.run_once(Direction('intruder', 0, component_sha256('intruder', 0)))
        with self.assertRaisesRegex(RuntimeError, 'one serial'):
            self.run_once(None)
        self.assertEqual(len(self.fast.prepare_calls), 1)
        self.assertEqual(self.fast.prepare_calls[0]['instruct'], 'owner')
        self.assertIn('_prepare_generation', vars(self.fast))
        self.fast.hooks.pop('talker')
        release.set()
        thread.join(10)
        self.assertNotIn('owner_error', results)
        self.assertIn('owner', results)
        self.assertEqual(self.adapter.last_metadata['direction']['direction_sha256'],
                         component_sha256('owner', 0))
        self.assert_restored()


class CancellationTests(AdapterCase):
    PHASES = ('before_preparation', 'talker', 'predictor', 'before_waveform', 'after_waveform')

    def test_cancellation_at_each_phase_publishes_nothing(self):
        for phase in self.PHASES:
            with self.subTest(phase=phase):
                fast = FakeFast()
                self.fast = fast
                adapter = DirectedFasterBaseAdapter(fast)
                torch_seed_before = self.torch.seed
                if phase == 'before_preparation':
                    adapter.cancel_event.set()
                elif phase == 'talker':
                    fast.hooks['talker'] = adapter.cancel_event.set
                elif phase == 'predictor':
                    fast.hooks['predictor'] = adapter.cancel_event.set
                elif phase == 'before_waveform':
                    fast.hooks['before_waveform'] = adapter.cancel_event.set
                else:
                    fast.speech_tokenizer = SimpleNamespace(decode=lambda payload, ev=adapter.cancel_event: (
                        ev.set(), ([np.full(240, 0.1, dtype=np.float32)], 24000))[1])
                self.torch.seed = torch_seed_before
                with direction_scope(Direction('calm', 3, component_sha256('calm', 3))):
                    with self.assertRaises(GenerationCancelled):
                        adapter.generate_voice_clone(text='hello there', language='English', voice_clone_prompt=[ICL],
                                                     max_new_tokens=40, **CONTROLS)
                self.assertNotIn('direction', adapter.last_metadata)
                self.assertNotIn('_prepare_generation', vars(fast))
                self.assertFalse(adapter.lock.locked())
                self.assertFalse(adapter._direction_guard.locked())


if __name__ == '__main__':
    unittest.main()
