# Local model store

模型权重只保存在本项目内，但不进入 Git 或源码导出。Git 跟踪本 README 和 `models/manifests/*.json`；manifest 为每个运行文件固定来源、revision、大小和 SHA256。ModelScope 条目把可下载的仓库 branch/tag 记为 `source_revision`，把 HubApi 返回的逐文件 Revision 记为 `source_file_revision`；供应器先用前者查询并核对后者、大小和 SHA256，再执行下载，二者不可混用。

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

字幕模型同样只允许通过 manifest 固定的 ModelScope revision 恢复。默认使用 Turbo；中文词级对齐还需要第四个 alignment manifest。三个 ASR 变体与中文对齐模型全部安装约占 6.99 GiB，其中 Distil 只能用于纯英语。逐个执行：

```powershell
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\whisper_large_v3_turbo.json --allow-modelscope-download
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\whisper_large_v3.json --allow-modelscope-download
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\whisper_distil_large_v3.json --allow-modelscope-download
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\whisperx_zh_alignment.json --allow-modelscope-download
```

预演时在对应命令末尾同时添加 `--dry-run`；预演只报告计划下载内容，不产生模型文件。下载完成后运行 `platform_doctor.py --full`，以 manifest 中的大小和 SHA256 做发布前验收。

供应器可以先尝试 manifest 中列出的项目内恢复 blob，再访问固定 ModelScope 来源；下载/复制后再次校验，并通过同目录临时文件原子提升。无效旧目标进入 `.quarantine/`。`provenance_only` 条目不能联网补取，必须从可信备份恢复。

IndexTTS2 的 BigVGAN 已固定到 `nv-community/bigvgan_v2_22khz_80band_256x` 的 ModelScope 文件 revision，文件大小和 SHA256 与既有验证副本一致。运行时仍严格离线，只有显式供应命令可以联网。`recovery_blobs/` 仅作为可选本地恢复缓存，不是 GitHub 新机部署的必需条件。

默认冷备份 `scripts/backup_project.ps1` 会保留全部 manifest，并选择性保留当前存在且被引用的 `recovery_blobs/`，但不复制其余数十 GiB 的运行模型；`-ValidateOnly` 会校验被纳入备份的本地恢复缓存。普通模型在恢复机上通过固定 ModelScope revision 重新供应。

manifest 中的许可证字段是供应链提示，不是法律意见。特别是 IndexTTS2 自定义许可和 MaskGCT `CC-BY-NC-4.0` 辅助权重，在商业生产/再分发前必须单独审查。
