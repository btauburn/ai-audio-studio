"""Subprocess isolation preserves the native module's route registry in pytest."""
import subprocess
import sys


def _run(code):
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_graph_worker_rejects_unknown_before_load_and_requires_explicit_speaker():
    _run('''
from fastapi import BackgroundTasks, HTTPException
from pydantic import ValidationError
import service.faster_qwen3tts_customvoice_server as server
server._load_model_if_needed = lambda: (_ for _ in ()).throw(AssertionError("must not load"))
try:
    server.native.TTSRequest(text="A missing voice must fail")
except ValidationError:
    pass
else:
    raise AssertionError("missing speaker accepted")
try:
    server.synthesize_tts(server.native.TTSRequest(text="hello", speaker="unknown"), BackgroundTasks())
except HTTPException as exc:
    assert exc.status_code == 422
else:
    raise AssertionError("unknown speaker accepted")
''')


def test_graph_worker_rejects_multiple_outputs_for_one_explicit_voice():
    _run('''
import sys
from types import SimpleNamespace
from fastapi import BackgroundTasks, HTTPException
import service.faster_qwen3tts_customvoice_server as server
sys.modules["torch"] = SimpleNamespace(cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None))
sys.modules["soundfile"] = SimpleNamespace(write=lambda *a: (_ for _ in ()).throw(AssertionError("must not save")))
server._load_model_if_needed = lambda: None
server.native._infer_lock = None
server._generation_config = dict(max_new_tokens=100, temperature=.9, top_k=50, top_p=1., do_sample=True, repetition_penalty=1.05)
def generate(**kwargs):
    assert kwargs["speaker"] == "Ryan"
    server._generation_local.metadata = {}
    return [[0.1] * 16, [0.2] * 16], 24000
server.native.qwen3tts_model = SimpleNamespace(generate_custom_voice=generate)
try:
    server.synthesize_tts(server.native.TTSRequest(text="hello", speaker="Ryan"), BackgroundTasks())
except HTTPException as exc:
    assert exc.status_code == 500
    assert "waveform output" in str(exc.detail)
else:
    raise AssertionError("multiple outputs accepted")
''')
