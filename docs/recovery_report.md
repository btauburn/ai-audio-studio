# 本机恢复与生产化审查报告

报告日期：2026-07-11

工作目录：本地仓库根目录

状态定义：本报告记录本次系统重装后恢复验证的事实，不替代许可证审批、安全评审、容量压测或正式发布签字。

## 1. 结论摘要

本项目已经从“复制旧目录后依赖历史机器状态”整理为可重建、可校验的本地平台：

- 平台/IndexTTS2、Qwen3-TTS、CosyVoice3 分为三个独立 uv 环境，各自有 pyproject 与 lockfile。
- 四个 TTS 后端统一登记在后端注册表中，分别使用 8001～8004 和项目内模型目录；平台同一时刻只托管一个 GPU 工作进程，REST 与 WebSocket 批处理共享同一把全局 TTS/GPU 租约。
- TTS、ASR 与中文对齐权重由八份 manifest 固定来源、branch/tag、逐文件 revision、大小和 SHA256；运行时禁止隐式下载，修复只允许可选项目内验证 blob 或显式 ModelScope 下载。BigVGAN 已补充固定 ModelScope 来源，本地恢复副本不再是新机前置条件。
- DeepSeek 默认分段模型已切换为 V4 Flash，并在允许的失败类型下最多回退一次 V4 Pro。
- 自动分段、手工 JSON 导入和角色删除的关联数据写入已具备多文件 staging/backup/rollback 事务；常规 JSON 写入采用锁、原子替换、flush 与 fsync。音频替换遵守“先提交元数据，再回收旧文件”。
- WebUI 浏览器依赖已本地化，打开页面不再自动启动 TTS；角色级后端/音色配置、波形保存和批处理清理均已补强。
- 已提供环境 bootstrap、只读 doctor、模型 preflight/provision、项目数据审计、私密全量备份和白名单源码导出工具。

四个后端、DeepSeek 和本机冷备份副本校验已经完成。就当前目标而言，项目已可用于**私人本机或受信任 LAN 内的非商业使用**；这不代表公网生产、多人服务或商业发布已经获准。公开或商业使用前仍必须完成鉴权与网络边界、许可证/声音授权、异机灾备和长稳容量验证。

## 2. 本机环境恢复

### 2.1 系统工具

| 项目 | 当前结果 |
| --- | --- |
| FFmpeg | `8.1.2-full_build-www.gyan.dev`，PATH 可用 |
| ffprobe | `8.1.2-full_build-www.gyan.dev`，PATH 可用 |
| uv | bootstrap 与三个环境的 `uv pip check` 可用 |
| 全局 `python` | PATH 中不存在；这是可接受状态，所有命令已改为显式环境解释器 |

### 2.2 三个环境

| 边界 | Python | 锁定核心 | `uv pip check` |
| --- | --- | --- | --- |
| 平台 + IndexTTS2 `.venv` | 3.12.13 | PyTorch 2.8/CUDA 12.9 系、Transformers 4.52.4 | 205 packages，兼容 |
| Qwen `environments/qwen3tts/.venv` | 3.12.13 | PyTorch 2.10/CUDA 13.0 系、`qwen-tts==0.1.1` | 89 packages，兼容 |
| CosyVoice `environments/cosyvoice/.venv` | 3.10.19 | PyTorch 2.3.1/CUDA 12.1 系、Windows WeText 路径 | 123 packages，兼容 |

结论：三个环境都能由各自 lockfile 独立重建，不能再合并。旧 `.venv_qwen3tts` 已删除，后端注册表也不再提供兼容回退；任一隔离环境缺失时会明确失败。

## 3. 模型与后端验证

八份 manifest 共记录 88 个运行文件，总大小 34,204,630,277 bytes：

| Manifest | 文件数 | manifest 记录总大小 | 供应来源 |
| --- | ---: | ---: | --- |
| `index_tts2.json` | 25 | 8,872,311,679 bytes | 固定 ModelScope 来源；可选本地恢复缓存 |
| `cosyvoice3.json` | 19 | 8,786,676,188 bytes | ModelScope（模型 + WeText） |
| `qwen3_tts_voicedesign.json` | 12 | 4,520,159,146 bytes | ModelScope |
| `qwen3_tts_customvoice.json` | 12 | 4,520,159,633 bytes | ModelScope |
| `whisper_large_v3_turbo.json` | 5 | 1,621,665,983 bytes | ModelScope |
| `whisper_large_v3.json` | 5 | 3,090,835,702 bytes | ModelScope |
| `whisper_distil_large_v3.json` | 5 | 1,516,479,628 bytes | ModelScope |
| `whisperx_zh_alignment.json` | 5 | 1,276,342,318 bytes | ModelScope |

只读 startup doctor 已确认三个解释器、八套本地模型、FFmpeg/ffprobe、私密配置 JSON、现有项目 JSON、五个端口以及 manifest 所列文件的存在和大小。ModelScope 供应链现包含 13 个固定 source、88 个可下载 artifact；branch/tag、逐文件 revision、大小与 SHA256 均由 manifest 约束。最新 doctor 数量以本报告末尾的发布验证记录和实际命令输出为准。

四个后端均通过同一套真实平台全链路 E2E：隔离临时项目创建 → 文本导入 → 角色配置 → 工作进程启动/身份校验 → 模型加载 → 批量合成 → ffprobe → 停止/卸载 → 项目删除。隔离项目已清理，后端配置已恢复，端口与 GPU 均释放。

| 后端 | 平台 E2E 输出 | 结论 |
| --- | --- | --- |
| IndexTTS2 | pcm_s16le、22050 Hz、mono、3.552653 秒、156,716 bytes | 通过 |
| Qwen CustomVoice | pcm_s16le、24000 Hz、mono、3.44 秒、165,164 bytes | 通过 |
| CosyVoice3 | pcm_f32le、24000 Hz、mono、3.64 秒、349,520 bytes | 通过 |
| Qwen VoiceDesign | pcm_s16le、24000 Hz、mono、3.20 秒、153,644 bytes | 通过 |

此外，四个模型均做过严格离线直接推理；IndexTTS2 的冷加载、8 维情感合成和 RTF，CosyVoice3 的项目内 WeText 与 RTF，以及两个 Qwen 模型的独立 load/synthesize/unload 均通过。CosyVoice3 的断网验证证明其当前启动链路不依赖 Hugging Face 或运行时 ModelScope 下载。Qwen 与 Cosy 的运行环境、服务脚本和模型目录都与后端注册表一致。

字幕链路同样做了严格离线真实推理：Turbo、Large-v3 与 Distil-large-v3 均在 CUDA 上生成非空段落和词级时间戳；Turbo 的中文结果随后实际加载项目内 WhisperX 中文对齐模型，输出有效且时间单调的词级结果，统一卸载后缓存归零。Distil 在后端强制只接受英文；所有模型 ID 只能解析到项目白名单目录，不能被仓库 ID 或任意路径替换。

IndexTTS2 首轮质检发现旧输出触及 0 dBFS；本轮新增统一音频安全处理，在峰值超过 0.95 时整体缩放并保留 0.95 headroom。修复后样音最大 -0.4 dB、平均 -8.6 dB，无满刻度硬削波，并新增 4 项回归测试。该处理只做整体增益，不改变波形相对动态。

## 4. DeepSeek 恢复

当前配置和代码事实：

- 默认模型：`deepseek-v4-flash`；JSON response format、thinking disabled、`max_tokens=64000`。
- 备用模型：`deepseek-v4-pro`；由 Flash 配置显式启用，最多回退一次。
- Provider URL：OpenAI-compatible `{base_url}/chat/completions`；示例 base URL 为 `https://api.deepseek.com`，不再依赖历史自定义端口。
- 重试：网络错误、HTTP 429 和 5xx，有界次数与延迟；认证/参数 4xx 不重试、不切 Pro。
- 回退还覆盖输出截断、空输出、非法 JSON、错误结构和分段未完整覆盖原文。
- Flash 和 Pro 的实时连接均返回 HTTP 200；使用隔离临时项目执行的真实自动分段成功。
- 测试输出只记录 request id 是否存在、finish reason 和 usage 等，不打印凭据。
- 五个 Provider 凭据和一个平台凭据已从 JSON 明文迁移到 Windows Credential Locker；配置文件只保存不可逆引用，API/health 只返回掩码，doctor 会把任何残留明文视为失败。

结论：默认 Flash → 单次 Pro 的链路已按预期工作。DeepSeek 仍是外部依赖，生产必须监控延迟、费用、配额和数据合规，并保留手工分段路径。旧版明文配置已迁移到操作系统凭据库；任何曾经过不安全日志或共享终端处理的旧凭据都应在对应控制台轮换。

## 5. 数据完整性与事务审查

已经确认：

- 项目 ID 通过 `safe_project_root` 限制在数据根内，拒绝路径越界。
- `project.json` 与 `segments.json` 的读-改-写分别使用项目级进程内锁。
- 配置、LLM 列表、声音列表、字幕和项目 JSON 采用同目录临时文件后 `os.replace`。
- 自动分段固定按“项目元数据锁 → 分段锁”取锁，同时 staging `project.json`、`segments.json`、`text/raw.txt`，保留备份；任一提升失败会回滚已替换路径。
- 手工 JSON 导入和角色删除也通过同一事务助手提交关联的 `project.json` 与 `segments.json`，不会留下半更新状态。
- 自动分段、分段更新/合并/删除、角色提示音和情感参考音都先完成元数据提交，再回收旧音频；上传使用 staging/backup，元数据失败会回滚新文件。清理失败最多留下可审计的孤儿，不会产生指向已删除音频的新引用。
- REST 与 WebSocket 批量合成共享全局 TTS/GPU 租约。第二个跨项目批次会立即拒绝；批次期间启动、停止或卸载工作进程返回 409，防止相互杀模型。
- 同一项目的长批量、A/B 合成、编辑、角色/分段配置、导入、导出与删除具有原子冲突保留；冲突返回 409，不会在旧快照上写回。不同项目的普通编辑不互相阻塞，租约由不可伪造 token 持有，异常路径与陈旧 token 均不能泄漏或释放新 owner。
- 单段生成状态按 `GENERATING → DONE/FAILED` 持久化；音频或元数据落盘失败不会被误报为成功。
- 活动项目删除返回 409；删除操作本身先取得原子删除保留，避免“通过检查后又开始合成”的竞态。

仍需接受的边界：

- 这些锁不是跨进程文件锁；不能水平启动多个平台实例共享同一项目目录。
- 文件系统多文件事务不是数据库 WAL，掉电、磁盘故障与误删除仍要靠冷备份。
- `audit_project_data.py` 支持 `--projects-root` 显式审计生产数据盘或恢复副本；它不负责调度备份，也不替代存储层快照。
- 项目删除仍是物理递归删除；虽然活动任务和删除竞态已被拒绝，仍必须在 UI/运维权限层控制并先备份。

本轮冷备份副本验证：

- 实时项目目录与私密恢复副本均已使用只读审计脚本检查，未发现结构或引用错误；私人项目数量和内容不进入源码仓库。
- 对两边全部 472 个项目数据文件逐个比较相对路径、长度与 SHA256；总计 198,877,645 bytes，差异 0。
- 该结果证明本轮数据副本字节一致且内部引用有效。它不等同于在另一台灾备机从零重建三个环境和四个模型，异机演练仍列为后续运维项目。

## 6. WebUI 与平台验证

已完成：

- Windows 主入口已用真实 `cmd.exe` 从项目内和外部工作目录执行；`start_platform.bat --check` 返回 doctor 258/0，正常启动模式到 `/api/v1/ready` 返回 `status=ok`，关闭启动器后平台子进程和 9000 端口全部释放。
- 项目自有 `.bat/.cmd` 已统一为 UTF-8 无 BOM + CRLF，并由 `.gitattributes` 和自动发现入口的回归测试固定，防止混合换行再次被 `cmd.exe` 拆成碎片命令。
- 前端契约、运行时 smoke、并发按钮与角色语音配置等专项测试共 48 项通过。
- 本地 Vue、Tailwind、WaveSurfer 文件可从 `/web/vendor/` 加载，Tailwind CSS 构建可复现。
- 真实浏览器打开页面和平台配置页通过，控制台 0 错误日志。
- 页面挂载不启动 TTS；服务状态保持 `running=false` 直到用户明确操作。
- WebSocket 超时/关闭清理、单段编辑防重复、波形草稿防抖/切换前 flush、beforeunload 防丢已覆盖。
- 角色编辑支持后端、Qwen speaker/language、VoiceDesign 描述和风格指令，并通过 voice-profile API 持久化。

## 7. 自动化质量结果

| 检查 | 结果 |
| --- | --- |
| 全套 pytest | 233 passed，10 个第三方依赖弃用 warning，0 failed |
| Ruff（`app/scripts/service`） | 通过 |
| 前端专项 | 48 passed |
| 三环境 `uv pip check` | 全部通过 |
| startup doctor quick | 发布提交前重新验证，结果见本报告末尾 |
| startup doctor full | 发布提交前重新验证；八份 manifest 的 88 个运行文件均需通过 SHA256 |
| 前端 npm audit | runtime/full 均为 0 vulnerability |
| Python pip-audit | 平台 17 条/3 包；Qwen 3 条/1 包；CosyVoice 38 条/8 包，均为上游 ML 兼容性固定项 |
| PowerShell 源码导出 parser | 通过 |
| 源码导出 allow-list/密钥/二进制验证 | 全部白名单文本/源码通过；真实临时导出与 manifest 数量一致，反向扫描 0 个禁入文件 |

warnings 需要在后续依赖维护中跟踪，但本轮没有把 warning 等同于失败，也没有通过随意升级底层依赖来消除 warning。

可安全升级的 FastAPI、Starlette、python-multipart、Gradio、NLTK、pytest 等已更新并完成真实回归。剩余 Python 告警集中在 `transformers`、`protobuf`、Lightning/ONNX/ModelScope 等模型运行栈；它们受 IndexTTS、Qwen-TTS 或 CosyVoice 的精确兼容约束。三个 worker 仅监听 loopback、运行时严格使用已校验本地权重，降低了暴露面，但不等于漏洞为零；后续应按每个隔离环境单独做升级分支和四后端真实样音回归。

## 8. 目录和供应链整理结果

权威结构：

```text
index-tts/
├── .venv/                         # 平台 + IndexTTS2
├── environments/
│   ├── qwen3tts/                  # 独立 pyproject/lock/.venv
│   └── cosyvoice/                 # 独立 pyproject/lock/.venv
├── models/
│   ├── manifests/                 # 可跟踪的供应链事实
│   ├── recovery_blobs/            # 可选的已验证本地恢复缓存
│   ├── index_tts/                 # 本地权重，不进 Git/源码包
│   ├── qwen3_tts/
│   ├── cosyvoice3/
│   └── whisper/                   # 三套 ASR + 中文对齐模型
├── dubbing_service/               # 平台业务层
├── service/                       # 四个推理工作进程
├── indextts/、cosyvoice/          # 引擎源码
├── scripts/                       # bootstrap/doctor/provision/audit/backup/export
└── docs/                          # 架构、运维、恢复报告
```

明确废弃的认知：

- `checkpoints/` 不是生产模型目录。
- `.venv_qwen3tts` 不是当前 Qwen 环境。
- CosyVoice 不再复用平台环境。
- 启动时不再自动联网补模型。
- `uv run` 不应被无差别用于所有后端；必须进入正确 uv 项目或直接调用正确解释器。
- 打开 WebUI 不代表应立即占用 GPU。

本轮已删除旧 `.venv_qwen3tts`、历史 Hugging Face/ModelScope 下载缓存、临时模型下载目录、测试输出和无引用杂项；当前模型权威目录保留四个 TTS 模型、四套字幕模型、manifests 与可选本地恢复缓存。普通权重均可按固定 ModelScope 来源重建。这些整理不涉及项目文本、角色配置、参考音频或生成结果。

`cosyvoice/` 原先是根仓库未登记的嵌套 Git 仓库，普通根级提交会漏掉其本地 Windows/WeText 适配。本轮已按 [third_party_sources.md](third_party_sources.md) 整理为 vendored source；CosyVoice、Matcha-TTS 的完整 Git bundle、本地 patch 和未跟踪适配文件副本均先保存到恢复备份，再移除嵌套元数据。实际运行源码未删除。

## 9. 源码交付与许可证风险

新增 `scripts/export_source.ps1` 使用显式目录、根文件和文本扩展名白名单，拒绝 reparse point，并在复制前扫描常见私钥/云密钥/token 格式。默认不包含：

- `models/` 权重与所有模型缓存（只包含 README 和 manifests）；
- `.venv*` 与 dependency cache；
- `.env`、`config.json`、LLM Provider、voices 配置；
- 项目文本、参考声音、生成音频、字幕、日志和临时文件；
- ONNX/PT/PTH/Safetensors/FST/音视频/压缩包等二进制。

实际临时导出已确认 `dubbing_service/llm_models.json` 与八份 `models/manifests/*.json` 全部进入，`config.json`、`llm_providers.json`、`voices.json`、模型权重、项目和音频均未进入；校验后临时导出已清理。

但是“源码包不含隐私”不等于“源码可任意再许可”。上线/分发前的主要许可证风险：

1. 根 IndexTTS2 自定义许可包含规模阈值、下游传递、用途和模型改进限制；中英文阈值描述存在差异，而协议声明中文优先。
2. IndexTTS2 辅助 MaskGCT 权重在 manifest 中为 CC-BY-NC-4.0，商业生产需要单独授权、替换或法律结论。
3. CosyVoice 主源码、Qwen/Cosy 模型、WebUI vendor 与 nested third-party 各有自己的许可证和 notice 保留义务。
4. 当前自研平台代码没有一个可以覆盖所有上游组件的统一宽松许可证。

因此，本报告只能确认技术可运行性，不能确认商业使用权。许可证审核是生产 Go/No-Go 的阻塞项。

## 10. 公开或商业使用前必须跟进

| 优先级 | 项目 | 完成标准 |
| --- | --- | --- |
| P0 | 许可证与训练/参考音频授权 | 法律书面结论；MaskGCT 商业权利明确；角色声音取得授权 |
| P0 | 9000 暴露安全 | 默认已改为 `127.0.0.1:9000`；只有显式 `--lan` 才绑定 `0.0.0.0`，且必须由 Windows 防火墙限制来源。公网/多人服务前仍需 TLS、鉴权、最小权限与审计日志 |
| P1 | 异机全量灾备演练 | 在另一台/隔离系统从源码、私密备份和 manifests 重建三环境与四模型；doctor、数据审计、四后端样音全通过 |
| P1 | 外部数据盘审计 | `DUBBING_PROJECTS_ROOT` 的备份、容量、完整性与访问控制自动化 |
| P1 | 长跑和容量测试 | 大项目、多小时批量、停止/恢复、磁盘满、API 限流和 GPU 峰值验证 |

## 11. 按使用范围的准入判断

当前判断分为两层：

- **私人本机 / 受信任 LAN / 非商业使用：GO。** 三环境、四模型、DeepSeek、平台真实 E2E、数据副本和恢复工具均已验证，可恢复近期实际使用。LAN 模式没有内置账户认证，必须维持可信网段防火墙限制，严禁直接暴露公网。
- **公网、多用户、开源正式发布或商业使用：Conditional No-Go。** 阻塞项是鉴权与网络边界、上游许可证和参考声音授权，以及异机灾备与多小时容量/故障测试，不是四个后端当前不可用。

上述 GO 只覆盖当前配音平台及四个受管 TTS 后端。根 `webui.py` 是独立的上游 Gradio demo；旧 MCP 工具尚未完成现代 Claude Desktop 客户端验收，二者均绕过平台 GPU 租约，不包含在当前生产准入中。

上述 P0 关闭并完成 P1 验证后，再按 [operations.md](operations.md) 的发布检查重新签发更高范围的 Go/No-Go。

## 12. 2026-08-10 GitHub 交付前复验

本次私有 GitHub 交付前，在当前 Windows 主机重新执行了以下只读或可逆检查：

| 检查 | 最终结果 |
| --- | --- |
| `python -m pytest` | 233 passed，0 failed；10 个 warning 均来自第三方依赖弃用提示 |
| `platform_doctor.py --full --verbose` | 256 passed，0 failed；八份 manifest 的全部本地文件通过大小与 SHA256 校验 |
| `start_platform.bat --check` | 启动预检通过，中文批处理入口可正常执行 |
| 目标 Python 静态检查 | Ruff `E4/E7/E9/F/I` 通过 |
| 三个 uv 环境依赖一致性 | 平台、Qwen3-TTS、CosyVoice 环境均通过 `uv pip check` |
| 源码交付边界 | allow-list 验证通过；模型权重、API Key、个人工程、音色与输出不进入 Git |

交付采用独立私有仓库：官方 `index-tts/index-tts` 仅作为 `upstream`，个人仓库作为 `origin`。这次复验确认的是当前机器上的技术恢复和可重复安装资料；不改变第 9 至 11 节所述许可证与商业使用边界。
