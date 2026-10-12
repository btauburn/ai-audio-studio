"""Staged native-instruction provider for the experimental Faster Base direction profile.

Not registered: project profile, worker allowlist and production exclusions are
integration work. The direction comes only from an injected trusted lookup owned
by the M4 owner; legacy performance fields are never promoted to an instruction.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from pathlib import Path

from dubbing_service.app.audiobook.fingerprint import canonical_json_bytes, sha256_bytes, text_sha256
from dubbing_service.app.providers.tts import Qwen3TTSProvider, TTSResult
from dubbing_service.app.audiobook.project_qwen_runtime import project_profile

PROFILE = 'faster-base-032-direction-x1'
POLICY = 'native-instruct-icl-x1'
MAX_INSTRUCTION_CHARS = 240
SEED_SALT_MAX = 2147483647
SEED_MODULUS = 4294967296
RECEIPT_KEYS = frozenset({'policy', 'forwarded', 'instruct_sha256', 'instruct_chars', 'conditioning_mode',
                          'seed_salt', 'seed_effective', 'direction_sha256'})
_WORKER_SCHEMA = 'qwen-base-worker-identity-v1'
WORKER_SOURCES = ('service/qwen3tts_server.py', 'service/qwen_character_voice.py', 'service/qwen_runtime_identity.py',
                  'service/faster_qwen3tts_base_server.py', 'service/faster_qwen3tts_base_adapter.py',
                  'service/faster_qwen3tts_base_direction_server.py', 'service/faster_qwen3tts_base_direction_adapter.py')


class DirectionRefused(RuntimeError):
    pass


def resolve_controls(native_instruction: object, seed_salt: object = 0) -> tuple[Optional[str], int]:
    if native_instruction is not None:
        if not isinstance(native_instruction, str):
            raise DirectionRefused('native_instruction must be a string or null')
        if native_instruction.strip() == '':
            native_instruction = None
    if native_instruction is not None:
        if len(native_instruction) > MAX_INSTRUCTION_CHARS:
            raise DirectionRefused(f'native_instruction exceeds {MAX_INSTRUCTION_CHARS} characters')
        if any(unicodedata.category(ch) in {'Cc', 'Cs'} for ch in native_instruction):
            raise DirectionRefused('native_instruction contains control or surrogate characters')
        from .performance_director import validate_native_instruction
        validate_native_instruction(native_instruction)
    if type(seed_salt) is not int or not 0 <= seed_salt <= SEED_SALT_MAX:
        raise DirectionRefused('seed_salt must be an integer from 0 to 2147483647')
    return native_instruction, seed_salt


def component_sha256(native_instruction: object, seed_salt: object = 0) -> str:
    instruction, salt = resolve_controls(native_instruction, seed_salt)
    return sha256_bytes(canonical_json_bytes({'control_policy': POLICY, 'native_instruction': instruction,
                                              'seed_salt': salt}))


@dataclass(frozen=True)
class ResolvedDirection:
    project_id: str
    role_id: str
    voice_identity_id: Optional[str]
    utterance_text_sha256: str
    native_instruction: Optional[str]
    seed_salt: int
    direction_sha256: str


DirectionLookup = Callable[[str, Any, Any], Optional[ResolvedDirection]]


def canonical_direction_lookup(project_id, role, segment):
    from dubbing_service.app.storage.projects import load_project
    from .performance_director import active_direction_for, direction_component
    project = load_project(project_id)
    row = active_direction_for(project, segment)
    if row is None or direction_component(row) is None:
        return None
    identity = getattr(role, 'voice_identity', None)
    return ResolvedDirection(project_id, row.role_id, getattr(identity, 'id', None), row.text_sha256,
                             row.resolved.native_instruction, row.resolved.seed_salt, direction_component(row))


def require_direction_project(project_id):
    from .project_qwen_runtime import project_profile
    if project_profile(project_id) != PROFILE:
        raise DirectionRefused('Direction forwarding requires the experimental profile on a review-only performance pilot')


@dataclass(frozen=True)
class _CallDirection:
    instruction: Optional[str]
    seed_salt: int
    digest: str
    identity_id: str
    text: str


_CALL: contextvars.ContextVar[Optional[_CallDirection]] = contextvars.ContextVar('performance_direction', default=None)


def validate_experimental_identity(value: Any) -> Dict[str, Any]:
    from .runtime_binding import validate_worker
    validate_worker(value)
    if not isinstance(value, dict) or not isinstance(value.get('contract'), dict):
        raise DirectionRefused('worker runtime identity is missing; no direction is sent')
    contract = value['contract']
    if (value.get('sha256') != sha256_bytes(canonical_json_bytes(contract))
            or contract.get('schema') != _WORKER_SCHEMA
            or contract.get('runtime_profile') != PROFILE
            or contract.get('fallback') != 'disabled'):
        raise DirectionRefused('worker is not the experimental direction profile; no direction is sent')
    sources = contract.get('implementation_sha256') or {}
    root = Path(__file__).resolve().parents[3]
    if (not contract.get('model_id') or not isinstance(contract.get('checkpoint_manifest_sha256'), str)
            or len(contract['checkpoint_manifest_sha256']) != 64
            or any(sources.get(name) != hashlib.sha256((root / name).read_bytes()).hexdigest() for name in WORKER_SOURCES)):
        raise DirectionRefused('experimental worker model or implementation provenance differs from the registered source')
    return value


def _verify_result(result: Dict[str, Any], identity: Dict[str, Any], call: _CallDirection,
                   prompt_sha256: str) -> None:
    if result.get('worker_runtime_identity') != identity or result.get('fallback') is not None:
        raise DirectionRefused('worker returned mismatched runtime or fallback audio; refusing result')
    if result.get('identity_id') != call.identity_id or result.get('prompt_sha256') != prompt_sha256:
        raise DirectionRefused('worker returned another voice or prompt; refusing result')
    if (result.get('execution_profile') or {}).get('runtime_profile') != PROFILE:
        raise DirectionRefused('worker did not report the experimental profile; refusing result')
    generation = result.get('generation')
    receipt = generation.get('direction') if isinstance(generation, dict) else None
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_KEYS:
        raise DirectionRefused('direction receipt is missing or incomplete; no candidate')
    forwarded = call.instruction is not None
    base_seed = int(sha256_bytes((call.identity_id + call.text).encode('utf-8'))[:8], 16)
    effective = base_seed if call.seed_salt == 0 else (base_seed + call.seed_salt) % SEED_MODULUS
    checks = (
        receipt['policy'] == POLICY,
        receipt['forwarded'] is forwarded,
        receipt['instruct_sha256'] == (sha256_bytes(call.instruction.encode('utf-8')) if forwarded else None),
        receipt['instruct_chars'] == (len(call.instruction) if forwarded else 0),
        receipt['conditioning_mode'] == 'icl',
        receipt['seed_salt'] == call.seed_salt,
        type(receipt['seed_effective']) is int and receipt['seed_effective'] == effective,
        receipt['direction_sha256'] == call.digest,
    )
    if not all(checks):
        raise DirectionRefused('direction receipt does not match the request; no candidate')


class DirectedQwen3TTSProvider(Qwen3TTSProvider):
    def __init__(self, direction_lookup: Optional[DirectionLookup] = None) -> None:
        super().__init__()
        self._direction_lookup = direction_lookup or canonical_direction_lookup

    def runtime_identity(self):
        identity = super().runtime_identity()
        if identity['contract']['runtime_profile'] == PROFILE or _CALL.get() is not None:
            validate_experimental_identity(identity)
        return identity

    def synthesize_segment(self, project_id: str, role: Any, segment: Any, previous_text: Optional[str] = None) -> TTSResult:
        identity = getattr(role, 'voice_identity', None)
        try:
            call = self._checked_direction(project_id, role, segment)
        except (RuntimeError, ValueError) as exc:
            return TTSResult(False, {'error': str(exc), 'identity_id': getattr(identity, 'id', None)})
        if call is None:
            return super().synthesize_segment(project_id, role, segment, previous_text)
        if identity is None or identity.backend != 'qwen3tts' or identity.mechanism != 'qwen_reference':
            return TTSResult(False, {'error': 'native direction requires a locked Qwen character identity; VoiceDesign is refused',
                                     'identity_id': getattr(identity, 'id', None)})
        token = _CALL.set(call)
        try:
            return self._synthesize_character_identity(project_id, role, segment)
        finally:
            _CALL.reset(token)

    def _checked_direction(self, project_id: str, role: Any, segment: Any) -> Optional[_CallDirection]:
        if self._direction_lookup is None:
            return None
        found = self._direction_lookup(project_id, role, segment)
        if found is None:
            return None
        if not isinstance(found, ResolvedDirection):
            raise DirectionRefused('direction lookup returned an unsupported value')
        identity = getattr(role, 'voice_identity', None)
        identity_id = getattr(identity, 'id', None)
        if found.project_id != project_id or found.role_id != role.id or found.voice_identity_id != identity_id:
            raise DirectionRefused('direction is bound to another project, role or voice')
        if found.utterance_text_sha256 != text_sha256(segment.text):
            raise DirectionRefused('direction is bound to different utterance text')
        instruction, salt = resolve_controls(found.native_instruction, found.seed_salt)
        digest = component_sha256(instruction, salt)
        if found.direction_sha256 != digest:
            raise DirectionRefused('direction component digest does not match the resolved controls')
        if instruction is None and salt == 0:
            return None
        return _CallDirection(instruction, salt, digest, identity_id, segment.text)

    def _character_request(self, operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        call = _CALL.get()
        if call is None or operation != 'synthesize':
            return super()._character_request(operation, payload)
        require_direction_project(payload['project_id'])
        self._bind_project_runtime(payload['project_id'])
        identity = validate_experimental_identity(self._expected_runtime_identity)
        performance = {**(payload.get('performance') or {}), 'native_instruction': call.instruction,
                       'seed_salt': call.seed_salt, 'direction_sha256': call.digest}
        request = {**payload, 'performance': performance, 'expected_runtime_sha256': identity['sha256']}
        result = self._post_synthesize(request)
        _verify_result(result, identity, call, payload['prompt_sha256'])
        return result

    def _experimental_identity(self) -> Dict[str, Any]:
        return validate_experimental_identity(self.runtime_identity())

    def _post_synthesize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/api/v1/tts/character/synthesize",
            data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST",
        )
        self._authorize(request)
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"character voice synthesize failed: {detail}") from exc
        if not isinstance(result, dict) or result.get("success") is False:
            raise RuntimeError("character voice synthesize returned no successful result")
        return result

    def _authorize(self, request: urllib.request.Request) -> None:
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
