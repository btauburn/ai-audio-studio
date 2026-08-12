<p align="center">
  <img src="dubbing_service/webui/brand/wordmark.svg" width="560" alt="AI Audio Studio" />
</p>

<h1 align="center">AI Audio Studio</h1>

<p align="center">
  A Windows-first, local-first production workspace for stories, news, knowledge content, and podcasts
</p>

<p align="center">
  <a href="https://github.com/tadashijlu/ai-audio-studio/actions/workflows/windows-no-model-ci.yml"><img alt="Windows no-model CI" src="https://github.com/tadashijlu/ai-audio-studio/actions/workflows/windows-no-model-ci.yml/badge.svg" /></a>
  <img alt="Platform layer license: Apache-2.0" src="https://img.shields.io/badge/platform%20layer-Apache--2.0-22c28f" />
  <img alt="Windows 10 and 11" src="https://img.shields.io/badge/Windows-10%20%7C%2011-2563eb" />
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776ab" />
</p>

<p align="center">
  <a href="README.md">简体中文</a> ·
  <a href="docs/installation.md">Installation</a> ·
  <a href="docs/user_guide.md">User guide (Chinese)</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

> [!IMPORTANT]
> `0.2.0-beta.1` targets Windows with an NVIDIA GPU. The TadashiJLU-authored platform layer is open-sourced under Apache-2.0. IndexTTS2, CosyVoice, Matcha-TTS, browser vendors, and separately downloaded models retain their own licenses; they are not relicensed as one Apache-2.0 work. Review the [path-level license map](LICENSES/README.md) before redistribution or commercial use.

AI Audio Studio turns isolated TTS calls into a traceable production pipeline:

**Script → Voice → A/B Takes → Non-destructive editing → Sound design → Master → Release package**

![AI Audio Studio voice review workspace](docs/assets/screenshots/studio-overview.png)

## Highlights

- **Independent A/B and immutable Takes** — generating a new version never silently destroys the other slot; history remains until explicitly removed.
- **Non-destructive editing** — trims, silence, gain, speed, fades, and padding are stored as recipes; the source Take is preserved.
- **Human review** — per-segment review states, issue notes, final Take selection, and resumable synthesis jobs.
- **Four-track sound design** — Dialogue, BGM, Ambience, and SFX with Stable Audio 3 sentence-level SFX planning/generation, licensing metadata, gain, looping, fades, crossfades, and music ducking.
- **Release-grade output** — loudness and peak QC, immutable Masters, WAV/MP3, SRT/VTT, chapters, metadata, rights manifests, checksums, and ZIP packages.
- **Local-first boundaries** — projects, voices, secrets, models, and generated media stay outside Git by default.

## TTS backends

| Backend | Best for | Voice control | Licensing note |
| --- | --- | --- | --- |
| IndexTTS2 | Narration, reference-voice cloning, detailed emotion | Reference audio plus vector/text/audio emotion control | Custom upstream license; the current auxiliary set includes CC-BY-NC-4.0 MaskGCT material, so this path is not commercially cleared by default |
| Qwen3-TTS CustomVoice | Fast character assignment without reference audio | Built-in voices plus style instructions | Weights are downloaded separately; review the model page terms |
| Qwen3-TTS VoiceDesign | Designing a voice from natural language | Text voice descriptions | Weights are downloaded separately; review the model page terms |
| CosyVoice3 | Natural-language style and delivery control | Reference voice and instruction control | Vendored code is Apache-2.0; model weights retain their model-page terms |

Model weights, API keys, reference voices, and user projects are never shipped in this repository. Provisioning uses manifest-pinned sources and verifies revisions, file metadata, size, and SHA256 before promotion into the project model directory.

## Full installation

### Requirements

- Windows 10/11, an NVIDIA GPU, and a compatible driver;
- verified baseline: RTX 4070 Ti SUPER with 16 GiB VRAM; lower-memory configurations are not yet part of a supported matrix;
- at least **80 GiB** free disk space for approximately 35.2 GiB of models, download cache, and four uv environments;
- internet access for first-time provisioning; routine inference uses project-local offline model paths.

Install the system tools in PowerShell:

```powershell
winget install --id Git.Git -e
winget install --id GitHub.GitLFS -e
winget install --id astral-sh.uv -e
winget install --id Gyan.FFmpeg -e
```

Open a new PowerShell window, then clone and bootstrap:

```powershell
git lfs install
git clone https://github.com/tadashijlu/ai-audio-studio.git
Set-Location .\ai-audio-studio
git lfs pull

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Runtime All
Copy-Item .\dubbing_service\config.example.json .\dubbing_service\config.json
```

The four Python environments are intentionally isolated: the root `.venv` hosts the platform, IndexTTS2, and subtitles; the Qwen, CosyVoice, and Stable Audio runtimes live under `environments/`.

Review [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), then preflight all nine manifests. `Full` includes gated Stable Audio 3 Small-SFX: accept the Stability AI and Gemma terms on the official model page, complete the applicable Stability registration, and expose a read token only in the current PowerShell process:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_models.ps1 -Profile Full -DryRun

$env:HF_TOKEN = '<Hugging Face read token>'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_models.ps1 `
  -Profile Full `
  -AcceptThirdPartyLicenses `
  -StableAudioLicenseTier community
Remove-Item Env:HF_TOKEN
```

The acceptance switch records acknowledgement only; it grants no additional commercial or redistribution rights. Interrupted downloads can be retried safely.

With the platform and workers stopped, validate and launch:

```powershell
.\.venv\Scripts\python.exe .\scripts\platform_doctor.py --startup
.\start_platform.bat
```

The doctor should finish with `0 failed`. Open `http://127.0.0.1:9000/web/`, choose a TTS backend in Platform Settings, then start the worker and load its model.

> [!WARNING]
> The default launcher binds only to `127.0.0.1`. This beta has no accounts, TLS, or authorization. Never expose it directly to the public internet. `start_platform.bat --lan` is only for a trusted, firewall-protected LAN.

The detailed installation guide, partial developer profiles, health checks, and first-sample procedure are in [docs/installation.md](docs/installation.md).

## Optional DeepSeek segmentation

Manual segmentation and local JSON import work without an LLM. For automatic segmentation, start the platform, open **Platform Settings → LLM Providers**, and add DeepSeek:

- Base URL: `https://api.deepseek.com`
- primary model: `deepseek-v4-flash`
- fallback model: `deepseek-v4-pro`
- API key: stored in the operating-system credential vault, never committed

Original text is sent to DeepSeek. Use manual segmentation or locally generated JSON for confidential or restricted material.

## Runtime topology

The platform manages only one GPU TTS worker at a time, releasing the previous process when switching backends.

| Service | Port | Interpreter | Local model directory |
| --- | ---: | --- | --- |
| Web/API platform | 9000 | `.venv` | — |
| IndexTTS2 | 8001 | `.venv` | `models/index_tts` |
| CosyVoice3 | 8002 | `environments/cosyvoice/.venv` | `models/cosyvoice3/Fun-CosyVoice3-0.5B` |
| Qwen VoiceDesign | 8003 | `environments/qwen3tts/.venv` | `models/qwen3_tts/Qwen3-TTS-VoiceDesign` |
| Qwen CustomVoice | 8004 | `environments/qwen3tts/.venv` | `models/qwen3_tts/Qwen3-TTS-CustomVoice` |
| Stable Audio 3 Small-SFX | 8005 | `environments/stableaudio/.venv` | `models/stable_audio_3/small-sfx` |

## Privacy and source boundaries

Git ignores secrets, model weights and caches, virtual environments, `dubbing_projects/`, voice libraries, reference audio, BGM/SFX, output, temporary files, and logs. Never force-add them.

Before publishing source, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\export_source.ps1 -ValidateOnly
```

## Documentation

- [Installation, models, and first acceptance test](docs/installation.md)
- [End-to-end production guide](docs/user_guide.md)
- [Architecture](docs/architecture.md)
- [Operations and troubleshooting](docs/operations.md)
- [Model supply-chain rules](models/README.md)
- [License scope map](LICENSES/README.md)
- [Third-party and model notices](THIRD_PARTY_NOTICES.md)
- [Security policy](SECURITY.md)
- [Dependency security and compatibility risks](docs/dependency_security.md)
- [Contributing](CONTRIBUTING.md) · [Support](SUPPORT.md) · [Changelog](CHANGELOG.md)

## Current scope

- Single-user local production is the priority; multi-user accounts, cloud sync, publishing-platform APIs, analytics, and a full DAW are out of scope.
- BGM remains asset-import only. SFX can optionally use local Stable Audio 3 Small-SFX after current license-receipt and real 12-category/30-run quality gates pass; no music-generation model is deployed.
- GitHub CI is model-free and GPU-free. Real TTS, subtitle, and release-package E2E must be accepted on a local GPU machine.
- Only manifest-pinned, SHA256-verified models stored inside the project are supported. Do not load untrusted checkpoints, Transformers repositories, or training configurations.
- The IndexTTS2 path is not commercially cleared while its current MaskGCT dependency remains non-commercial without separate permission.
- Stable Audio 3 weights retain the Stability AI Community/Enterprise License and bundled T5Gemma retains the Gemma Terms; the platform Apache-2.0 license does not cover those weights.

## Contributing and security

Reproducible bugs, documentation improvements, and focused audio-production changes are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) first. Report vulnerabilities privately according to [SECURITY.md](SECURITY.md); never paste credentials, private voices, or project content into a public issue.

## License and attribution

Copyright 2026 **TadashiJLU**.

The original platform layer is licensed under the [Apache License 2.0](LICENSE). This is a multi-license distribution: upstream IndexTTS2 and derivative modifications, vendored CosyVoice/Matcha-TTS, browser vendors, model weights, and example media retain their own terms. See [LICENSES/README.md](LICENSES/README.md), adjacent license files, and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

This project is based on a pinned commit of [index-tts/index-tts](https://github.com/index-tts/index-tts). Modifications in this project are not endorsed, warranted, or guaranteed by the original upstream right-holder.
