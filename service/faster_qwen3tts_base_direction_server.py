"""Experimental Faster Base direction profile (staged; additive to the baseline worker).

Imports the pinned Faster Base server and rebinds only this process's profile,
adapter class, source identity and character-call wrapper. No baseline file is
edited. Launch is a human-reviewed step and is not performed by staging.
"""
from __future__ import annotations

import json
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import service.faster_qwen3tts_base_server as base  # noqa: E402
from service import faster_qwen3tts_base_direction_adapter as direction  # noqa: E402
from service.faster_qwen3tts_base_direction_adapter import DirectedFasterBaseAdapter  # noqa: E402

native = base.native
PROFILE = direction.PROFILE
SERVER_SOURCE = 'service/faster_qwen3tts_base_direction_server.py'
ADAPTER_SOURCE = 'service/faster_qwen3tts_base_direction_adapter.py'
CHARACTER_PREFIX = '/api/v1/tts/character/'
RECEIPT_KEYS = frozenset({'policy', 'forwarded', 'instruct_sha256', 'instruct_chars',
                          'conditioning_mode', 'seed_salt', 'seed_effective', 'direction_sha256'})

base.PROFILE = PROFILE
base.FasterBaseAdapter = DirectedFasterBaseAdapter
native._runtime_profile = PROFILE
native._runtime_sources.update(native.source_snapshot(ROOT, (SERVER_SOURCE, ADAPTER_SOURCE)))
for _action in native.parser._actions:
    if '--execution-profile' in _action.option_strings:
        _action.choices = [PROFILE]
        _action.default = PROFILE
        break
else:
    native.parser.add_argument('--execution-profile', choices=[PROFILE], default=PROFILE)
native._runtime_snapshot = native.WorkerIdentitySnapshot(ROOT, native._parse_server_args(),
                                                         native._runtime_sources, PROFILE)


def _contains_direction_key(value) -> bool:
    if isinstance(value, dict):
        return any(key in direction.DIRECTION_KEYS or _contains_direction_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_direction_key(item) for item in value)
    return False


def _receipt_matches(receipt, requested, *, identity_id=None, text=None) -> bool:
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_KEYS:
        return False
    return (receipt['policy'] == direction.POLICY
            and receipt['forwarded'] is (requested.native_instruction is not None)
            and receipt['conditioning_mode'] == direction.CONDITIONING_MODE
            and receipt['seed_salt'] == requested.seed_salt
            and type(receipt['seed_salt']) is int
            and receipt['instruct_sha256'] == (hashlib.sha256(requested.native_instruction.encode('utf-8')).hexdigest() if requested.native_instruction is not None else None)
            and type(receipt['instruct_chars']) is int
            and receipt['instruct_chars'] == (len(requested.native_instruction) if requested.native_instruction is not None else 0)
            and type(receipt['seed_effective']) is int and 0 <= receipt['seed_effective'] < 2**32
            and (identity_id is None or receipt['seed_effective'] == (int(hashlib.sha256((identity_id + text).encode('utf-8')).hexdigest()[:8], 16) + requested.seed_salt) % 2**32)
            and receipt['direction_sha256'] == requested.direction_sha256)


def _discard_output(result) -> None:
    output = result.get('audio_file') if isinstance(result, dict) else None
    if output:
        candidate = Path(output).resolve()
        if candidate.parent == (ROOT / 'outputs').resolve() and candidate.name.startswith('qwen_identity_'):
            candidate.unlink(missing_ok=True)


_base_character_call = native._character_call


def directed_character_call(method, **kwargs):
    if getattr(method, '__name__', '') != 'synthesize':
        if _contains_direction_key(kwargs):
            raise native.HTTPException(status_code=400, detail='native direction is accepted only on character synthesize')
        return _base_character_call(method, **kwargs)
    try:
        requested = direction.parse_request(kwargs.get('performance'))
    except direction.DirectionError as exc:
        raise native.HTTPException(status_code=400, detail=str(exc)) from exc
    if kwargs.get('performance') is not None:
        kwargs['performance'] = direction.strip_request_keys(kwargs['performance'])
    if requested is None:
        return _base_character_call(method, **kwargs)
    with direction.direction_scope(requested):
        result = _base_character_call(method, **kwargs)
    if not _receipt_matches((result.get('generation') or {}).get('direction'), requested,
                            identity_id=kwargs.get('identity_id'), text=kwargs.get('text')):
        _discard_output(result)
        raise native.HTTPException(status_code=500, detail='direction receipt missing or mismatched; no audio returned')
    return result


native._character_call = directed_character_call


class _RefuseDirectionOutsideSynthesis:
    def __init__(self, asgi_app):
        self.app = asgi_app

    async def __call__(self, scope, receive, send):
        path = scope.get('path') or ''
        if (scope['type'] != 'http' or scope.get('method') != 'POST'
                or not path.startswith(CHARACTER_PREFIX) or path == CHARACTER_PREFIX + 'synthesize'):
            await self.app(scope, receive, send)
            return
        body = b''
        more = True
        while more:
            message = await receive()
            body += message.get('body', b'')
            more = message.get('more_body', False)
        if _body_has_direction(body):
            from starlette.responses import JSONResponse
            response = JSONResponse({'detail': 'native direction is accepted only on character synthesize'}, status_code=400)
            await response(scope, receive, send)
            return
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {'type': 'http.request', 'body': body, 'more_body': False}
            return await receive()

        await self.app(scope, replay, send)


def _body_has_direction(body: bytes) -> bool:
    try:
        return _contains_direction_key(json.loads(body.decode('utf-8')))
    except (UnicodeDecodeError, ValueError):
        return any(key.encode() in body for key in direction.DIRECTION_KEYS)


app = base.app
app.add_middleware(_RefuseDirectionOutsideSynthesis)


if __name__ == '__main__':
    args = native._parse_server_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, reload=False, log_level='info')
