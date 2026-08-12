# 配音平台生产运维手册

本文是 Windows 本机/单机生产部署的权威操作入口。命令均从仓库根目录执行，且显式调用目标虚拟环境，避免重装系统后依赖全局 `python` 或误用环境。

## 1. 基线与安全原则

- 支持的环境管理工具是 `uv`；不要用全局 pip、Conda 或手工复制 site-packages 修复环境。
- 模型只从项目内已验证 blob 或 manifest 固定、显式授权的 ModelScope/Hugging Face 源恢复。不要把 Hugging Face 缓存当成生产依赖。
- 平台和本地 TTS 默认是单机信任边界，没有业务用户认证。除非已经部署反向代理鉴权和防火墙规则，否则只绑定 `127.0.0.1`。
- 不要运行两个平台进程写同一个项目数据目录。
- `config.json`、LLM Provider、声音和项目目录含密钥或个人数据，必须按私密生产数据处理。
- 任何升级、环境重建或模型恢复前先做冷备份。

## 2. 端口与进程

| 端口 | 服务 | 默认解释器 |
| ---: | --- | --- |
| 9000 | 配音平台与 WebUI | `.venv/Scripts/python.exe` |
| 8001 | IndexTTS2 | `.venv/Scripts/python.exe` |
| 8002 | CosyVoice3 | `environments/cosyvoice/.venv/Scripts/python.exe` |
| 8003 | Qwen3 VoiceDesign | `environments/qwen3tts/.venv/Scripts/python.exe` |
| 8004 | Qwen3 CustomVoice | `environments/qwen3tts/.venv/Scripts/python.exe` |
| 8005 | Stable Audio 3 Small-SFX | `environments/stableaudio/.venv/Scripts/python.exe` |

平台正常情况下通过同一个 GPU 租约只运行一个 8001～8005 生成工作进程。配置页所显示的“配置后端”和实际“运行后端”可能暂时不同；保存后端配置不会热切换已运行进程，必须停止/重启 TTS。

## 3. 新机器恢复

### 3.1 系统前置

确认以下命令可用：

```powershell
uv --version
ffmpeg -version
ffprobe -version
nvidia-smi
```

本机无需全局 `python`。bootstrap 会通过 uv 安装并固定 Python 3.12.13、3.11.13 与 3.10.19。FFmpeg/ffprobe 必须在 PATH 上；GPU 驱动版本必须支持四个环境锁定的 CUDA PyTorch wheel。

### 3.2 重建四个环境

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Runtime All
```

只修复某一边界时可使用：

```powershell
.\scripts\bootstrap.ps1 -Runtime Platform
.\scripts\bootstrap.ps1 -Runtime Qwen
.\scripts\bootstrap.ps1 -Runtime CosyVoice
.\scripts\bootstrap.ps1 -Runtime StableAudio
```

脚本使用四个独立 lockfile 并在每个环境后运行 `uv pip check`。完成后应存在：

```text
.venv/Scripts/python.exe
environments/qwen3tts/.venv/Scripts/python.exe
environments/cosyvoice/.venv/Scripts/python.exe
environments/stableaudio/.venv/Scripts/python.exe
```

旧 `.venv_qwen3tts` 已删除，后端注册表也没有兼容回退；不要重新创建或向同名目录安装包。

### 3.3 配置

若本机配置不存在，先复制示例，再只在本机填写：

```powershell
Copy-Item .\dubbing_service\config.example.json .\dubbing_service\config.json
```

默认地址：

```text
IndexTTS2             http://127.0.0.1:8001
CosyVoice3            http://127.0.0.1:8002
Qwen VoiceDesign      http://127.0.0.1:8003
Qwen CustomVoice      http://127.0.0.1:8004
DeepSeek API          https://api.deepseek.com
```

环境变量优先于文件。平台 UI 新建或更新 LLM 提供商时，API Key 会写入操作系统凭据库（Windows Credential Locker / macOS Keychain），JSON 只保留 `credential_ref`；平台级 LLM/TTS Key 同样只保留 `*_api_key_ref`。旧版本留下的明文配置必须迁移：

```powershell
.\.venv\Scripts\python.exe .\scripts\migrate_llm_secrets.py
.\.venv\Scripts\python.exe .\scripts\migrate_llm_secrets.py --check
```

`--check` 必须报告 `plaintext_remaining=0`（或 `0 remaining`），startup doctor 也会拒绝含明文 Key 的配置。系统凭据不会进入项目备份；重装系统或换机器后应在 UI 重新填写，或由进程环境传递 `DUBBING_LLM_API_KEY`。将 `DUBBING_PROJECTS_ROOT` 指向独立数据盘。不要提交以下文件：

- `.env`
- `dubbing_service/config.json`
- `dubbing_service/llm_providers.json`
- `dubbing_service/voices.json` 和 `dubbing_service/voices/`
- `dubbing_projects/`

### 3.4 校验或恢复模型

快速检查只验证普通文件、路径和大小，不读取所有权重：

```powershell
.\.venv\Scripts\python.exe .\scripts\platform_doctor.py --startup
```

全量发布校验会读取 manifest 中每个文件并计算 SHA256，耗时明显更长：

```powershell
.\.venv\Scripts\python.exe .\scripts\platform_doctor.py --full
```

单独检查某个 manifest：

```powershell
.\.venv\Scripts\python.exe .\scripts\model_preflight.py --manifest .\models\manifests\index_tts2.json
.\.venv\Scripts\python.exe .\scripts\model_preflight.py --manifest .\models\manifests\cosyvoice3.json
.\.venv\Scripts\python.exe .\scripts\model_preflight.py --manifest .\models\manifests\qwen3_tts_voicedesign.json
.\.venv\Scripts\python.exe .\scripts\model_preflight.py --manifest .\models\manifests\qwen3_tts_customvoice.json
.\.venv\Scripts\python.exe .\scripts\model_preflight.py --manifest .\models\manifests\whisper_large_v3_turbo.json
.\.venv\Scripts\python.exe .\scripts\model_preflight.py --manifest .\models\manifests\whisper_large_v3.json
.\.venv\Scripts\python.exe .\scripts\model_preflight.py --manifest .\models\manifests\whisper_distil_large_v3.json
.\.venv\Scripts\python.exe .\scripts\model_preflight.py --manifest .\models\manifests\whisperx_zh_alignment.json
.\.venv\Scripts\python.exe .\scripts\model_preflight.py --manifest .\models\manifests\stable_audio_3_small_sfx.json --require-license-acceptance --require-quality-gate
```

缺失或损坏时，先预演，再显式允许 ModelScope 下载。以下以 IndexTTS2 为例，其他后端替换 manifest 路径：

```powershell
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\index_tts2.json --dry-run --allow-modelscope-download
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\index_tts2.json --allow-modelscope-download
```

供应器会先检查可选项目内恢复 blob，再访问 manifest 固定的 ModelScope revision/file。下载内容只有在大小与 SHA256 都匹配后才原子提升；原损坏目标保留在 `models/.quarantine/`。BigVGAN 也已固定到经过核验的 ModelScope 文件 revision，不使用 Hugging Face 运行时补取。

### 3.5 恢复字幕模型

字幕运行时不接受 Hugging Face 仓库名或任意磁盘路径，只接受三个 UI 模型 ID，并映射到项目内目录。默认 `large-v3-turbo` 兼顾中英文质量与速度；`large-v3` 更慢但质量优先；`distil-large-v3` 仅限纯英语。中文词级时间轴需要项目内 WhisperX alignment 模型，缺失时字幕仍可生成，但只使用确定性的 ASR 时间戳降级结果。

先预演四个 ModelScope 下载（不会写模型）：

```powershell
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\whisper_large_v3_turbo.json --dry-run --allow-modelscope-download
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\whisper_large_v3.json --dry-run --allow-modelscope-download
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\whisper_distil_large_v3.json --dry-run --allow-modelscope-download
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\whisperx_zh_alignment.json --dry-run --allow-modelscope-download
```

确认项目所在磁盘至少有 15 GiB 余量后，去掉 `--dry-run` 执行同一组命令。最终运行文件约 6.99 GiB；供应器还会保留项目内下载缓存并用临时副本做原子提升，因此安装过程需要额外空间。每个文件都会在进入运行目录前核对固定大小与 SHA256。若近期只需要中文配音字幕，可先安装 Turbo 与中文 alignment 两个 manifest；但完整发布验收要求四个 manifest 都通过：

```powershell
.\.venv\Scripts\python.exe .\scripts\platform_doctor.py --full
```

### 3.6 供应并验收 Stable Audio 3 Small-SFX

该可选模型使用官方门控 Hugging Face 来源，不走 ModelScope。先在模型页接受 Stability AI 与 Gemma 条款、完成适用的 Stability 注册，再把 read token 仅放入当前 PowerShell 进程。`Full` profile 会包含 Small-SFX；只安装音效模型可用 `Sfx`：

```powershell
$env:HF_TOKEN = '<Hugging Face read token>'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_models.ps1 `
  -Profile Sfx `
  -AcceptThirdPartyLicenses `
  -StableAudioLicenseTier community
Remove-Item Env:HF_TOKEN
```

供应固定 commit、逐文件 blob id、大小和 SHA256，目标是 `models/stable_audio_3/small-sfx/`。token 不写入命令参数或磁盘，运行时严格离线。下载成功只代表文件与许可回执就绪；生产启用还要求真实质量门。启动 8005 worker 后运行 12 类 × 2 seed 质量样本和 5 轮 × 6 次技术测试，人工填写评分模板，再执行评估：

```powershell
$env:DUBBING_WORKER_INSTANCE_TOKEN = '<本次 worker 的随机实例令牌>'
.\.venv\Scripts\python.exe .\scripts\benchmark_stable_audio_sfx.py --generate `
  --tts-unload-endpoint http://127.0.0.1:8001/api/v1/tts/unload-model `
  --tts-load-endpoint http://127.0.0.1:8001/api/v1/tts/load-model `
  --results-dir .\models\stable_audio_3\small-sfx\benchmark_runs\acceptance-01
.\.venv\Scripts\python.exe .\scripts\benchmark_stable_audio_sfx.py --evaluate `
  --results-dir .\models\stable_audio_3\small-sfx\benchmark_runs\acceptance-01
```

至少 10/12 类质量通过，技术测试 30/30 通过，并且五轮 TTS unload → SFX load/6 runs → SFX unload → TTS load 的显存释放与恢复证据完整时，才原子写 `quality_gate.json`。warm p95 还必须不超过 `max(5 秒, 请求时长)`。示例使用已运行的 IndexTTS2 worker；可把两个 TTS endpoint 换成当前后端或平台代理协调端点。权重、许可回执或 revision 不匹配会在请求 worker 前失败；mock/CI 和仅连续 SFX 请求不生成通过标记。

## 4. 日常启动与停止

### 4.1 推荐 Windows 入口

在所有 9000、8001～8005 服务停止时运行：

```bat
start_platform.bat
```

只解析同一个批处理入口并执行完整 startup doctor、不打开启动器时：

```bat
start_platform.bat --check
```

启动脚本会：

1. 设置项目内缓存、ModelScope 缓存、WeText 路径和离线变量；
2. 运行只读 startup doctor；
3. 启动桌面启动器与 9000 平台；
4. 等待 `/api/v1/ready` 后打开 `http://127.0.0.1:9000/web/`。

启动器默认让平台监听 `127.0.0.1`。确实需要局域网访问时运行 `start_platform.bat --lan`；必须限制 Windows 防火墙来源，并在跨不受信任网络前增加认证和 TLS。

### 4.2 手工 loopback 启动

```powershell
.\.venv\Scripts\python.exe -u -m uvicorn dubbing_service.app.main:app --host 127.0.0.1 --port 9000
```

生产模式不要加 `--reload`。浏览器访问：

```text
http://127.0.0.1:9000/web/
```

WebUI 不会自动启动 TTS。在“平台配置”中选择后端并点击启动/加载模型。平台退出时会清理其托管的 TTS 子进程。

### 4.3 必要时手工启动工作进程

通常应由平台管理；故障隔离时才直接运行：

```powershell
$env:HF_HOME=(Resolve-Path .\models\.runtime_cache)
$env:HF_HUB_CACHE="$env:HF_HOME\hub"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
$env:COSYVOICE_WETEXT_DIR=(Resolve-Path .\models\cosyvoice3\wetext)

.\.venv\Scripts\python.exe -u .\service\server.py --host 127.0.0.1 --port 8001 --model_dir .\models\index_tts --fp16 --no-cuda-kernel

.\environments\cosyvoice\.venv\Scripts\python.exe -u .\service\cosyvoice_server.py --host 127.0.0.1 --port 8002 --model_dir .\models\cosyvoice3\Fun-CosyVoice3-0.5B

.\environments\qwen3tts\.venv\Scripts\python.exe -u .\service\qwen3tts_server.py --host 127.0.0.1 --port 8003 --model_dir .\models\qwen3_tts\Qwen3-TTS-VoiceDesign --device cuda:0 --dtype bfloat16

.\environments\qwen3tts\.venv\Scripts\python.exe -u .\service\qwen3tts_customvoice_server.py --host 127.0.0.1 --port 8004 --model_dir .\models\qwen3_tts\Qwen3-TTS-CustomVoice --device cuda:0 --dtype bfloat16

$env:DUBBING_WORKER_INSTANCE_TOKEN = '<随机实例令牌>'
.\environments\stableaudio\.venv\Scripts\python.exe -u .\service\stableaudio_sfx_server.py --host 127.0.0.1 --port 8005 --model-dir .\models\stable_audio_3\small-sfx --device cuda:0 --dtype float16
```

不要同时启动多个大模型，除非已经测量显存并明确接受资源竞争。手工工作进程不受平台进程生命周期管理，必须自行停止。

根 `webui.py` 是上游 IndexTTS2 单模型 Gradio demo，旧 MCP 脚本也是独立兼容工具；二者都会绕过平台 GPU 租约，不能与 9000 平台并行。MCP 尚未完成现代 Claude Desktop 客户端验收，不属于当前生产入口。

## 5. 健康检查

### 5.1 平台

```powershell
Invoke-RestMethod http://127.0.0.1:9000/api/v1/ready
Invoke-RestMethod http://127.0.0.1:9000/api/v1/health
Invoke-RestMethod http://127.0.0.1:9000/api/v1/settings/tts-service/status
```

- `ready` 只证明平台 HTTP 已可用，不加载 GPU。
- `health` 聚合配置、TTS、Whisper 与 GPU 信息；TTS 未启动时部分字段为 null/false 是正常状态。
- `tts-service/status` 区分 `runtime_backend` 和 `config_backend`。

### 5.2 TTS 工作进程

根据当前后端检查一个端口：

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod http://127.0.0.1:8002/health
Invoke-RestMethod http://127.0.0.1:8003/health
Invoke-RestMethod http://127.0.0.1:8004/health
```

通过标准不是只有 HTTP 200，还应确认 `backend`、`model_loaded` 和 `device` 与预期一致。平台托管启动还会用随机 `instance_token` 校验本次新进程；若目标端口由旧/orphan 服务占用，平台拒绝启动，不会把旧健康响应误认成新进程。

Stable Audio SFX worker 使用 8005，`/health` 及所有控制/生成接口均要求本次实例 token，避免同机其他进程取得管理权限。通常由平台按生成批次托管；隔离诊断时：

```powershell
$headers = @{ 'X-Dubbing-Worker-Token' = $env:DUBBING_WORKER_INSTANCE_TOKEN }
Invoke-RestMethod http://127.0.0.1:8005/health -Headers $headers
```

### 5.3 端口与进程定位

```powershell
Get-NetTCPConnection -State Listen -LocalPort 9000,8001,8002,8003,8004,8005 |
    Select-Object LocalAddress,LocalPort,OwningProcess

Get-CimInstance Win32_Process |
    Where-Object ProcessId -in (Get-NetTCPConnection -State Listen -LocalPort 9000,8001,8002,8003,8004,8005).OwningProcess |
    Select-Object ProcessId,Name,CommandLine
```

doctor 把端口占用视为启动前失败。如果服务是为了验收而有意运行，应先记录健康响应，再停止服务并重跑 doctor；不要把“端口已占用”误判为依赖损坏。

## 6. DeepSeek 分段

当前默认模型是 `deepseek-v4-flash`：JSON 模式、thinking disabled、`max_tokens=64000`。配置允许时，Flash 在重试后仍遇到网络/429/5xx，或输出截断、空、非法 JSON、结构错误、遗漏/重复/改写原文时，只回退一次 `deepseek-v4-pro`。认证和参数类 4xx 不自动回退，避免掩盖错误配置。

仅测试 Flash 连接：

```powershell
.\.venv\Scripts\python.exe .\scripts\test_llm_connection.py
```

同时显式测试 Pro：

```powershell
.\.venv\Scripts\python.exe .\scripts\test_llm_connection.py --include-pro
```

测试真实分段链路：

```powershell
.\.venv\Scripts\python.exe .\scripts\smoke_llm_segmentation.py
```

这些命令会产生真实外部 API 请求，可能计费。输出不会打印 API Key；分段 smoke 使用临时项目根目录，不写用户项目，但仍会把测试文本发送给 DeepSeek。

## 7. 数据完整性与备份

### 7.1 只读数据审计

```powershell
.\.venv\Scripts\python.exe .\scripts\audit_project_data.py

.\.venv\Scripts\python.exe .\scripts\audit_project_data.py --projects-root E:\secure-backups\index-tts\dubbing_projects
```

不传参数时检查默认 `dubbing_projects/`；生产数据盘或恢复副本必须通过 `--projects-root` 显式指定。脚本检查 JSON、角色引用、音频路径越界和缺失文件，但不会修改数据。审计实时目录和备份目录都通过，才有资格继续做逐文件哈希或恢复启动验证。

### 7.2 冷备份

先停止平台和所有 TTS 工作进程。建议先做只读清单与恢复材料校验（不会创建备份目录）：

```powershell
.\scripts\backup_project.ps1 -ValidateOnly
```

校验通过后再运行：

```powershell
.\scripts\backup_project.ps1
```

默认备份到仓库同级的私密备份目录，排除 Git、虚拟环境、缓存、日志、输出/temp、API Key、本机 Provider/配置和普通模型权重；同时保留源码、默认项目数据、声音、所有模型 manifest，以及当前存在且被 manifest 引用的可选恢复缓存。脚本会校验恢复缓存的大小与 SHA256，再逐文件复制并校验备份副本，最终生成 `backup-manifest.json`、`backup.json` 和 `BACKUP_RESTORE.md`。任何不完整执行只保留带 `.partial` 后缀的目录，不应视为有效备份。

默认备份不含 API Key；恢复时必须从独立密钥存储重新注入，并从 `dubbing_service/config.example.json` 重建本机配置。备份仍可能包含私密声音、prompt 和作品文本，必须加密、限制访问且不能上传公共仓库。

Stable Audio 许可回执有意不进入备份。恢复后必须重新确认当前条款，系统会生成新的 `operator_confirmation_id`；项目内生成素材保留原始 provenance ID，不会被改写。只要新回执仍有效，且 `confirmation_sha256` 与生成时许可文本哈希一致、`license_tier` 仍为同一 Community/Enterprise 层级，旧素材可继续采用并进入 Master/Release；许可文本哈希或层级任一变化都会继续阻断正式输出。

需要连同模型一起制作离线灾备时：

```powershell
.\scripts\backup_project.ps1 -IncludeModels -DestinationRoot E:\secure-backups\index-tts
```

`-IncludeModels` 会额外纳入运行模型，但仍排除下载缓存、运行缓存与 quarantine。备份脚本使用 `uv pip list --format freeze` 生成不带安装 URL 的四个环境版本快照；恢复时仍以 lockfile + bootstrap 为准，快照只用于审计，不用于重新混装环境。

如果生产数据根在仓库外，`backup_project.ps1` 不会自动包含它，必须对 `DUBBING_PROJECTS_ROOT` 做独立的卷快照/冷备份，并在恢复演练中同时验证。

### 7.3 源码白名单导出

先只验证选择集和常见密钥扫描：

```powershell
.\scripts\export_source.ps1 -ValidateOnly
```

实际导出：

```powershell
.\scripts\export_source.ps1 -DestinationRoot E:\source-exports\index-tts
```

导出脚本使用目录、扩展名和根文件三重白名单，拒绝 reparse point，并排除模型、虚拟环境、密钥配置、项目、声音、prompt、日志、生成物和媒体/权重格式。导出目录含 SHA256 manifest。它是源码交付，不是完整可运行备份，也不改变上游许可证。

## 8. 发布检查

发布前依次执行：

```powershell
uv pip check --python .\.venv\Scripts\python.exe
uv pip check --python .\environments\qwen3tts\.venv\Scripts\python.exe
uv pip check --python .\environments\cosyvoice\.venv\Scripts\python.exe
uv pip check --python .\environments\stableaudio\.venv\Scripts\python.exe

.\.venv\Scripts\python.exe .\scripts\platform_doctor.py --full
.\.venv\Scripts\python.exe -m pytest -q
.\scripts\export_source.ps1 -ValidateOnly
```

另外必须人工完成：

- 从空闲状态分别启动四个 TTS 后端，核对 `/health`，每个后端至少合成一条可听音频；启用 SFX 时另做 Stable Audio 生成与 TTS 显存交接验收；
- 验证创建项目 → 自动分段 → 角色配置 → 单段/批量合成 → 波形编辑 → 整书导出 → 字幕；
- 用非生产测试项目验证 A/B 切换、失败重试和停止批处理；
- 恢复一份冷备份到隔离目录并重新运行数据审计；
- 确认防火墙、鉴权、日志留存、磁盘容量和许可证批准。

## 9. 故障处理

### 批处理出现乱码或碎片命令

项目维护的 `.bat/.cmd` 必须是 UTF-8 无 BOM、CRLF 换行；`.gitattributes` 已固定该规则，并有回归测试。先运行 `start_platform.bat --check`。若仍出现 `'xxx' is not recognized`，检查文件是否被外部编辑器重新保存为混合换行，不要据此重装模型或虚拟环境。

### 平台启动前 doctor 报端口占用

先查进程命令行和健康接口。若是旧平台/工作进程，正常停止其进程树后重试；不要直接删除环境或模型。

### 模型目录存在但加载失败

目录存在不等于完整。先对对应 manifest 运行完整 `model_preflight.py`，再决定是否 provision；不要手工把未知缓存文件覆盖到目标目录。

### 切换后端后请求仍发往旧模型

比较 `/settings/tts-service/status` 的 `runtime_backend` 与 `config_backend`。停止旧 TTS，再从平台启动目标后端；仅保存配置不足以热切换进程。

### 启动/停止 TTS 返回 409

全局 TTS/GPU 租约正在保护一个 REST 或 WebSocket 批处理。等待当前批次完成或从任务界面正常停止；不要直接结束工作进程，否则会把正在写入的分段留在恢复状态。跨项目第二个批处理也会立即返回 409，这是防止相互杀模型和重复合成的预期行为。

### DeepSeek Flash 失败且未回退

检查 HTTP 状态。401/403/400 属于凭据或请求错误，设计上不回退 Pro。修复 Base URL、API Key 或模型权限后重试。

### 项目 JSON 异常

立即停止所有写入，复制整个项目目录作为取证副本，再从最近冷备份恢复。不要在原目录同时运行修复脚本和平台。自动分段事务遗留的隐藏 `.stage`/`.backup` 文件应保留到原因确认后再处理。

### GPU 显存不足

确认只有一个 8001～8005 生成工作进程，停止孤儿进程，重新启动所需后端。不要通过把四套依赖合并到同一环境来解决显存问题。

## 10. 恢复顺序

发生系统重装或磁盘迁移时按以下顺序：

1. 恢复代码、lockfile、manifest 和私密冷备份；
2. 恢复外部项目数据根；
3. 安装 uv、FFmpeg、GPU 驱动；
4. 用 bootstrap 重建四个环境；
5. 用 manifest 校验/恢复模型，并重新确认未进入备份的 Stable Audio 许可回执；
6. 只读审计 JSON 与音频引用；
7. 在端口空闲时运行 startup doctor；
8. 启动 9000，不启动 TTS，先验证 WebUI 与项目读取；
9. 按 IndexTTS2 → CosyVoice3 → Qwen VD → Qwen CV 逐个做健康与样音；启用 SFX 时再做 Stable Audio 质量门；
10. 最后恢复 DeepSeek 连接与真实分段 smoke。

不要恢复旧虚拟环境目录，也不要把模型、项目数据或 API Key 放入源码导出包。
