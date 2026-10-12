"""Request-scoped native-instruction direction for the pinned Faster Base adapter.

Staged only for profile ``faster-base-032-direction-x1``; no existing owner imports it.
Without an active direction, ``generate_voice_clone`` forwards the same keyword
arguments to the parent. With one, the library's ``_prepare_generation`` seam is
wrapped for this call only: the instruction is passed there and the seed salt is
applied after the runtime's own seeding. Every parent guard still runs.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import re
import threading
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from service.faster_qwen3tts_base_adapter import FasterBaseAdapter

PROFILE = 'faster-base-032-direction-x1'
POLICY = 'native-instruct-icl-x1'
CONDITIONING_MODE = 'icl'
MAX_INSTRUCTION_CHARS = 240
SEED_SALT_MAX = 2147483647
SEED_MODULUS = 4294967296
DIRECTION_KEYS = ('native_instruction', 'seed_salt', 'direction_sha256')
_DIGEST = re.compile(r'[0-9a-f]{64}')
_NUMERIC_RATE = re.compile(r'(?i)\b\d+(?:\.\d+)?\s*(?:%|percent\b|wpm\b|words per minute\b|x\b|times\b)')
_BORROWED_VOICE = re.compile(r'(?i)\b(?:sounds? like|sounding like|in the (?:voice|style) of|impersonat\w*|imitat\w*|clone)\b')
_MISSING = object()


class DirectionError(ValueError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    # Same bytes as dubbing_service.app.audiobook.fingerprint.canonical_json_bytes.
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def resolve_controls(native_instruction: object, seed_salt: object = 0) -> tuple[Optional[str], int]:
    if native_instruction is not None:
        if not isinstance(native_instruction, str):
            raise DirectionError('native_instruction must be a string or null')
        if native_instruction.strip() == '':
            native_instruction = None
    if native_instruction is not None:
        if len(native_instruction) > MAX_INSTRUCTION_CHARS:
            raise DirectionError(f'native_instruction exceeds {MAX_INSTRUCTION_CHARS} characters')
        if any(unicodedata.category(ch) in {'Cc', 'Cs'} for ch in native_instruction):
            raise DirectionError('native_instruction contains control or surrogate characters')
        if _NUMERIC_RATE.search(native_instruction) or _BORROWED_VOICE.search(native_instruction):
            raise DirectionError('native_instruction must contain delivery wording, without a numeric rate or another voice')
    if type(seed_salt) is not int or not 0 <= seed_salt <= SEED_SALT_MAX:
        raise DirectionError('seed_salt must be an integer from 0 to 2147483647')
    return native_instruction, seed_salt


def component_sha256(native_instruction: object, seed_salt: object = 0) -> str:
    instruction, salt = resolve_controls(native_instruction, seed_salt)
    payload = {'control_policy': POLICY, 'native_instruction': instruction, 'seed_salt': salt}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class Direction:
    native_instruction: Optional[str]
    seed_salt: int
    direction_sha256: str

    @property
    def requested(self) -> bool:
        return self.native_instruction is not None or self.seed_salt != 0


def parse_request(performance: Optional[dict]) -> Optional[Direction]:
    """Validate the three request keys. None means the unchanged parent path."""
    if not performance:
        return None
    present = [key for key in DIRECTION_KEYS if key in performance]
    if not present:
        return None
    if len(present) != len(DIRECTION_KEYS):
        raise DirectionError('native direction requires native_instruction, seed_salt and direction_sha256 together')
    digest = performance['direction_sha256']
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise DirectionError('direction_sha256 must be 64 lowercase hexadecimal characters')
    instruction, salt = resolve_controls(performance['native_instruction'], performance['seed_salt'])
    if component_sha256(instruction, salt) != digest:
        raise DirectionError('direction_sha256 does not match the resolved controls')
    if instruction is None and salt == 0:
        return None
    return Direction(instruction, salt, digest)


def strip_request_keys(performance: Optional[dict]) -> dict:
    return {key: value for key, value in (performance or {}).items() if key not in DIRECTION_KEYS}


@dataclass
class _PrepareCall:
    direction: Direction
    prepared: bool = False
    seed_effective: Optional[int] = None


_ACTIVE: contextvars.ContextVar[Optional[Direction]] = contextvars.ContextVar('faster_base_direction', default=None)
_OWNED: contextvars.ContextVar[Optional[_PrepareCall]] = contextvars.ContextVar('faster_base_direction_owned', default=None)


@contextlib.contextmanager
def direction_scope(direction: Optional[Direction]) -> Iterator[None]:
    token = _ACTIVE.set(direction)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active_direction() -> Optional[Direction]:
    return _ACTIVE.get()


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def _require_full_reference(voice_clone_prompt: Any) -> None:
    if not isinstance(voice_clone_prompt, list) or len(voice_clone_prompt) != 1:
        raise DirectionError('one persisted identity is required')
    item = voice_clone_prompt[0]
    icl = _field(item, 'icl_mode')
    xvec = _field(item, 'x_vector_only_mode')
    if icl is None or xvec is None or bool(xvec) or not bool(icl):
        raise DirectionError('native direction requires full persisted reference conditioning (icl)')
    if _field(item, 'ref_code') is None:
        raise DirectionError('native direction requires full reference codes')


def _apply_seed(direction: Direction) -> int:
    import torch
    base = int(torch.initial_seed())
    if direction.seed_salt == 0:
        return base
    effective = (base + direction.seed_salt) % SEED_MODULUS
    torch.manual_seed(effective)
    return effective


def _receipt(call: _PrepareCall) -> dict:
    direction = call.direction
    instruction = direction.native_instruction
    forwarded = instruction is not None
    return {
        'policy': POLICY,
        'forwarded': forwarded,
        'instruct_sha256': hashlib.sha256(instruction.encode('utf-8')).hexdigest() if forwarded else None,
        'instruct_chars': len(instruction) if forwarded else 0,
        'conditioning_mode': CONDITIONING_MODE,
        'seed_salt': direction.seed_salt,
        'seed_effective': call.seed_effective,
        'direction_sha256': direction.direction_sha256,
    }


class DirectedFasterBaseAdapter(FasterBaseAdapter):
    def __init__(self, fast: Any):
        super().__init__(fast)
        self._direction_guard = threading.Lock()

    def generate_voice_clone(self, *, text, language, voice_clone_prompt, **controls):
        direction = active_direction()
        # Admission spans metadata publication as well as the parent's GPU lease.
        # A rejected contender owns neither the seam nor the prior owner's receipt.
        if not self._direction_guard.acquire(blocking=False):
            raise RuntimeError('Faster supports one serial GPU request')
        try:
            self._drop_receipt()
            if direction is None or not direction.requested:
                result = super().generate_voice_clone(text=text, language=language,
                                                      voice_clone_prompt=voice_clone_prompt, **controls)
                self.last_metadata = {**self.last_metadata, 'runtime_profile': PROFILE}
                return result
            return self._generate_directed(direction, text=text, language=language,
                                          voice_clone_prompt=voice_clone_prompt, **controls)
        except BaseException:
            self._drop_receipt()
            raise
        finally:
            self._direction_guard.release()

    def _drop_receipt(self) -> None:
        self.last_metadata = {key: value for key, value in self.last_metadata.items() if key != 'direction'}

    def _generate_directed(self, direction: Direction, *, text, language, voice_clone_prompt, **controls):
        _require_full_reference(voice_clone_prompt)
        instruction, salt = resolve_controls(direction.native_instruction, direction.seed_salt)
        if component_sha256(instruction, salt) != direction.direction_sha256:
            raise DirectionError('direction digest differs from the resolved controls')
        call = _PrepareCall(direction)
        fast = self.fast
        previous = fast.__dict__.get('_prepare_generation', _MISSING)
        original = fast._prepare_generation

        def scoped_prepare(*args, **kwargs):
            if _OWNED.get() is not call:
                return original(*args, **kwargs)
            if direction.native_instruction is not None:
                kwargs['instruct'] = direction.native_instruction
            result = original(*args, **kwargs)
            call.prepared = True
            call.seed_effective = _apply_seed(direction)
            return result

        parent_lock = self.lock
        def restore_prepare():
            if previous is _MISSING:
                fast.__dict__.pop('_prepare_generation', None)
            else:
                fast.__dict__['_prepare_generation'] = previous

        class OwnedPrepareLock:
            def acquire(inner, *args, **kwargs):
                acquired = parent_lock.acquire(*args, **kwargs)
                if acquired:
                    try:
                        fast._prepare_generation = scoped_prepare
                    except BaseException:
                        parent_lock.release()
                        raise
                return acquired
            def release(inner):
                try:
                    restore_prepare()
                finally:
                    parent_lock.release()
            def locked(inner):
                return parent_lock.locked()

        # The parent acquires its real lock exactly once. The wrapper becomes
        # visible only after that acquisition and disappears before release.
        self.lock = OwnedPrepareLock()
        owned_token = _OWNED.set(call)
        try:
            arrays, rate = super().generate_voice_clone(text=text, language=language,
                                                        voice_clone_prompt=voice_clone_prompt, **controls)
            if not call.prepared:
                raise DirectionError('direction was not confirmed at token preparation')
            self.last_metadata = {**self.last_metadata, 'runtime_profile': PROFILE, 'direction': _receipt(call)}
            return arrays, rate
        finally:
            _OWNED.reset(owned_token)
            self.lock = parent_lock
