# Local model store

模型权重只保存在本项目内，但不进入 Git 或源码导出。Git 跟踪本 README 和 `models/manifests/*.json`；manifest 为每个运行文件固定来源、revision、大小和 SHA256。ModelScope 条目把可下载的仓库 branch/tag 记为 `source_revision`，把 HubApi 返回的逐文件 Revision 记为 `source_file_revision`；供应器先用前者查询并核对后者、大小和 SHA256，再执行下载，二者不可混用。schema v2 的 Hugging Face 条目固定 40 位 commit、逐文件 blob id、大小和 SHA256，并把门控下载、许可确认与运行时离线加载分开。

| Manifest | 目标 |
| --- | --- |
| `index_tts2.json` | `models/index_tts/`（core + auxiliary） |
| `cosyvoice3.json` | `models/cosyvoice3/Fun-CosyVoice3-0.5B/` 与项目内 WeText |
| `qwen3_tts_voicedesign.json` | `models/qwen3_tts/Qwen3-TTS-VoiceDesign/` |
| `qwen3_tts_customvoice.json` | `models/qwen3_tts/Qwen3-TTS-CustomVoice/` |
| `whisper_large_v3_turbo.json` | `models/whisper/faster-whisper-large-v3-turbo/`（字幕默认） |
| `whisper_large_v3.json` | `models/whisper/faster-whisper-large-v3/` |
| `whisper_distil_large_v3.json` | `models/whisper/faster-distil-whisper-large-v3/`（仅英语） |
| `whisperx_zh_alignment.json` | `models/whisper/whisperx-wav2vec2-zh/`（中文词级对齐） |
| `stable_audio_3_small_sfx.json` | `models/stable_audio_3/small-sfx/`（可选音效生成） |

运行时必须传入明确的本地路径，并设置 Hugging Face/Transformers 离线模式，禁止启动时下载。

快速/完整校验：

```powershell
.\.venv\Scripts\python.exe .\scripts\model_preflight.py --manifest .\models\manifests\index_tts2.json --quick
.\.venv\Scripts\python.exe .\scripts\model_preflight.py --manifest .\models\manifests\index_tts2.json
```

缺失时先 dry-run；只有显式授权后才使用 manifest 固定的 ModelScope 文件：

```powershell
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\index_tts2.json --dry-run --allow-modelscope-download
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\index_tts2.json --allow-modelscope-download
```

## Stable Audio 3 Small-SFX 门控供应

Stable Audio 3 Small-SFX 来自官方门控仓库 `stabilityai/stable-audio-3-small-sfx`，固定到 commit `ae12755283df9d62ca39a9b050a39a0b607b8c20`，完整运行文件约 3.49 GB（3.25 GiB）。供应前必须在官方模型页取得访问权，逐项审阅 Stability AI 与 Gemma 条款并完成适用的 Stability 注册。令牌只允许从当前进程的 `HF_TOKEN` 环境变量读取；命令行、manifest、许可回执和日志都不得保存令牌或 Hugging Face 账号。

推荐通过安装脚本供应；`Full` 包含该模型，也可单独选择 `Sfx`：

```powershell
$env:HF_TOKEN = '<仅当前 PowerShell 进程可见的 Hugging Face read token>'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_models.ps1 `
  -Profile Sfx `
  -AcceptThirdPartyLicenses `
  -StableAudioLicenseTier community
Remove-Item Env:HF_TOKEN
```

底层等价命令要求允许门控下载，以及与当前条款确认文本精确匹配的哈希。哈希变化代表条款或确认文本变化，旧回执不会被接受：

```powershell
.\.venv\Scripts\python.exe .\scripts\model_provision.py `
  --manifest .\models\manifests\stable_audio_3_small_sfx.json `
  --allow-huggingface-gated-download `
  --accept-license-hash 4ffeca25eadd41b113dc2893f3526c917413b827a845579530de023d1a37a451 `
  --license-tier community
```

供应器先在 `models/.download_cache/huggingface/` staging，核对固定 commit、逐文件 blob id、大小与 SHA256，再复制到目标同目录临时文件并原子提升。运行时不读取 `HF_TOKEN`，不访问 Hugging Face，也不会隐式补文件。许可回执只记录 tier、三个布尔确认、随机 operator confirmation UUID、model revision 和确认哈希，不记录身份或令牌。

## Small-SFX 真实质量门

有权重和当前许可回执仍不足以启用生产音效生成。必须启动端口 8005 的真实本地 worker，然后完成 12 类、每类 2 个固定 seed 的人工质量评审，以及 5 轮共 30 次技术稳定性测试：

```powershell
$env:DUBBING_WORKER_INSTANCE_TOKEN = '<与本次 worker 一致的随机实例令牌>'
.\.venv\Scripts\python.exe .\scripts\benchmark_stable_audio_sfx.py `
  --generate `
  --tts-unload-endpoint http://127.0.0.1:8001/api/v1/tts/unload-model `
  --tts-load-endpoint http://127.0.0.1:8001/api/v1/tts/load-model `
  --results-dir .\models\stable_audio_3\small-sfx\benchmark_runs\acceptance-01

# 人工试听 24 个 quality/*.wav，并填写 manual_scores.csv 后：
.\.venv\Scripts\python.exe .\scripts\benchmark_stable_audio_sfx.py `
  --evaluate `
  --results-dir .\models\stable_audio_3\small-sfx\benchmark_runs\acceptance-01
```

每个类别生成两个固定 seed 候选；至少一个候选必须同时达到语义匹配、事件时序、伪影控制、背景洁净度和可用性阈值，该类别才通过；总计至少 10/12 类通过。技术门必须 30/30 通过，且五轮都真实执行 TTS unload → SFX load/6 runs → SFX unload → TTS load；每轮记录显存基线、SFX 峰值、释放后值和 TTS 恢复值，释放后不得超过基线 512 MiB（可显式调严）。warm runs 的 p95 latency 必须不超过 `max(5 秒, 请求时长)`。另检查 HTTP/模型/seed 响应契约、44.1 kHz 双声道 PCM16、时长、静音与削波，并记录 latency/RTF。只有真实生成证据和完整人工评分均通过时，脚本才会原子写入 `models/stable_audio_3/small-sfx/quality_gate.json`；缺权重、mock、空评分、仅连续 SFX 请求或旧 revision 都不会生成通过标记。

字幕模型同样只允许通过 manifest 固定的 ModelScope revision 恢复。默认使用 Turbo；中文词级对齐还需要第四个 alignment manifest。三个 ASR 变体与中文对齐模型全部安装约占 6.99 GiB，其中 Distil 只能用于纯英语。逐个执行：

```powershell
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\whisper_large_v3_turbo.json --allow-modelscope-download
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\whisper_large_v3.json --allow-modelscope-download
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\whisper_distil_large_v3.json --allow-modelscope-download
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\whisperx_zh_alignment.json --allow-modelscope-download
```

预演时在对应命令末尾同时添加 `--dry-run`；预演只报告计划下载内容，不产生模型文件。下载完成后运行 `platform_doctor.py --full`，以 manifest 中的大小和 SHA256 做发布前验收。

供应器可以先尝试 manifest 中列出的项目内恢复 blob，再访问显式允许的固定 ModelScope 或门控 Hugging Face 来源；下载/复制后再次校验，并通过同目录临时文件原子提升。无效旧目标进入 `.quarantine/`。`provenance_only` 条目不能联网补取，必须从可信备份恢复。

IndexTTS2 的 BigVGAN 已固定到 `nv-community/bigvgan_v2_22khz_80band_256x` 的 ModelScope 文件 revision，文件大小和 SHA256 与既有验证副本一致。运行时仍严格离线，只有显式供应命令可以联网。`recovery_blobs/` 仅作为可选本地恢复缓存，不是 GitHub 新机部署的必需条件。

默认冷备份 `scripts/backup_project.ps1` 会保留全部 manifest，并选择性保留当前存在且被引用的 `recovery_blobs/`，但不复制其余数十 GiB 的运行模型；`-ValidateOnly` 会校验被纳入备份的本地恢复缓存。普通模型在恢复机上通过固定来源 revision 重新供应。默认备份和源码导出不会携带 `HF_TOKEN`、许可回执、质量门或 benchmark WAV；恢复 Stable Audio 后必须由操作者重新确认许可并在目标 GPU 上重跑质量门。

manifest 中的许可证字段是供应链提示，不是法律意见。特别是 IndexTTS2 自定义许可和 MaskGCT `CC-BY-NC-4.0` 辅助权重，在商业生产/再分发前必须单独审查。Stable Audio 3 的 Community License 不是 Apache/MIT 式无限制开源许可；商业使用必须满足其收入门槛、注册等条件，超出适用范围时需 Enterprise License，并同时遵守 Gemma 条款。
