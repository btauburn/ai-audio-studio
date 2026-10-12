"""Single source of truth for local TTS backend processes.

The platform intentionally manages one GPU worker at a time.  Each worker has
an explicit interpreter, model directory, loopback port and capability list so
the router does not need backend-specific subprocess branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Brian approved this implemented Base cloning profile as the default on 2026-10-04.
# CustomVoice has a separate profile contract and retains its existing default.
DEFAULT_QWEN_BASE_PROFILE = "faster-base-032"


@dataclass(frozen=True, slots=True)
class BackendSpec:
    backend_id: str
    display_name: str
    description: str
    port: int
    server_script: str
    model_directory: str
    environment: str
    capabilities: tuple[str, ...]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


BACKEND_SPECS: dict[str, BackendSpec] = {
    "indextts": BackendSpec(
        backend_id="indextts",
        display_name="IndexTTS2",
        description="音色克隆 + 8维情感向量控制",
        port=8001,
        server_script="service/server.py",
        model_directory="models/index_tts",
        environment="platform",
        capabilities=("reference_audio", "emotion_vector", "emotion_text"),
    ),
    "cosyvoice3": BackendSpec(
        backend_id="cosyvoice3",
        display_name="CosyVoice3",
        description="音色克隆 + 自然语言指令控制",
        port=8002,
        server_script="service/cosyvoice_server.py",
        model_directory="models/cosyvoice3/Fun-CosyVoice3-0.5B",
        environment="cosyvoice",
        capabilities=("reference_audio", "instruction", "speed"),
    ),
    "qwen3tts": BackendSpec(
        backend_id="qwen3tts",
        display_name="Qwen3-TTS VoiceDesign",
        description="文本描述生成音色，无需参考音频",
        port=8003,
        server_script="service/qwen3tts_server.py",
        model_directory="models/qwen3_tts/Qwen3-TTS-VoiceDesign",
        environment="qwen3tts",
        capabilities=("voice_design", "instruction", "language"),
    ),
    "qwen3tts_cv": BackendSpec(
        backend_id="qwen3tts_cv",
        display_name="Qwen3-TTS CustomVoice",
        description="内置音色 + 文本指令控制情感",
        port=8004,
        server_script="service/qwen3tts_customvoice_server.py",
        model_directory="models/qwen3_tts/Qwen3-TTS-CustomVoice",
        environment="qwen3tts",
        capabilities=("builtin_speaker", "instruction", "language"),
    ),
}


def get_backend_spec(backend_id: str) -> BackendSpec:
    try:
        return BACKEND_SPECS[backend_id]
    except KeyError as exc:
        raise ValueError(f"unsupported TTS backend: {backend_id}") from exc


def public_backend_definitions() -> list[dict[str, object]]:
    return [
        {
            "id": spec.backend_id,
            "name": spec.display_name,
            "description": spec.description,
            "capabilities": list(spec.capabilities),
            "port": spec.port,
        }
        for spec in BACKEND_SPECS.values()
    ]


def _python_candidates(project_root: Path, environment: str, platform_python: Path) -> Iterable[Path]:
    if environment == "platform":
        yield platform_python
        yield project_root / ".venv" / "Scripts" / "python.exe"
        yield project_root / ".venv" / "bin" / "python"
        return

    yield project_root / "environments" / environment / ".venv" / "Scripts" / "python.exe"
    yield project_root / "environments" / environment / ".venv" / "bin" / "python"


def resolve_backend_python(project_root: Path, spec: BackendSpec, platform_python: Path) -> Path:
    for candidate in _python_candidates(project_root, spec.environment, platform_python):
        if candidate.is_file():
            return candidate.resolve()
    expected = project_root / "environments" / spec.environment / ".venv"
    raise FileNotFoundError(f"{spec.display_name} environment is missing: {expected}")


def build_backend_command(
    project_root: Path,
    backend_id: str,
    platform_python: Path,
    *,
    fp32: bool = False,
    qwen_attention: str = "auto",
    qwen_execution_profile: str | None = None,
    preserve_outputs: bool = False,
) -> list[str]:
    spec = get_backend_spec(backend_id)
    if qwen_execution_profile is None:
        qwen_execution_profile = DEFAULT_QWEN_BASE_PROFILE if backend_id == "qwen3tts" else "native-current"
    environment = spec.environment
    server_script = spec.server_script
    attention = qwen_attention
    if backend_id == "qwen3tts" and qwen_execution_profile in {"faster-base-032", "faster-base-032-direction-x1"}:
        environment = "faster_qwen_base032"
        server_script = ("service/faster_qwen3tts_base_direction_server.py" if qwen_execution_profile == "faster-base-032-direction-x1"
                         else "service/faster_qwen3tts_base_server.py")
    elif backend_id == "qwen3tts" and qwen_execution_profile not in {"native-current", "native-sdpa", "qwen-cudagraph"}:
        raise ValueError(f"unsupported Qwen Base execution profile: {qwen_execution_profile}")
    if backend_id == "qwen3tts_cv":
        if qwen_execution_profile not in {"native-current", "native-sdpa", "qwen-cudagraph"}:
            raise ValueError(f"unsupported Qwen execution profile: {qwen_execution_profile}")
        if qwen_execution_profile == "qwen-cudagraph":
            environment = "faster_qwen3tts"
            server_script = "service/faster_qwen3tts_customvoice_server.py"
            attention = "sdpa"
        elif qwen_execution_profile == "native-sdpa":
            attention = "sdpa"
    runtime_spec = BackendSpec(
        backend_id=spec.backend_id,
        display_name=spec.display_name,
        description=spec.description,
        port=spec.port,
        server_script=server_script,
        model_directory=spec.model_directory,
        environment=environment,
        capabilities=spec.capabilities,
    )
    python = resolve_backend_python(project_root, runtime_spec, platform_python)
    server = (project_root / server_script).resolve()
    model_dir = (project_root / spec.model_directory).resolve()

    missing = [str(path) for path in (server, model_dir) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing TTS runtime path(s): " + ", ".join(missing))

    command = [
        str(python),
        "-u",
        str(server),
        "--host",
        "127.0.0.1",
        "--port",
        str(spec.port),
        "--model_dir",
        str(model_dir),
    ]
    if backend_id == "indextts":
        command.extend(["--no-fp16" if fp32 else "--fp16", "--no-cuda-kernel"])
    elif backend_id.startswith("qwen3tts"):
        command.extend(["--device", "cuda:0", "--dtype", "bfloat16"])
    if backend_id.startswith("qwen3tts") and preserve_outputs:
        command.extend(["--outputs-keep-days", "0", "--outputs-max-files", "0"])
    if backend_id == "qwen3tts_cv":
        if attention not in {"auto", "eager", "sdpa", "flash_attention_2"}:
            raise ValueError(f"unsupported Qwen attention mode: {attention}")
        command.extend(["--attention", attention])
        command.extend(["--execution-profile", qwen_execution_profile])
    return command
