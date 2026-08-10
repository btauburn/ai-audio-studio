# 配音平台服务（dubbing_service）

`dubbing_service` 是当前生产业务入口：FastAPI 后端、原生 Vue WebUI、项目文件存储、DeepSeek 自动分段、四后端 TTS 编排、音频编辑/导出和字幕。它不是早期 MVP，也不要求用户提前手工启动一个固定的 IndexTTS 服务。

完整架构见 [`../docs/architecture.md`](../docs/architecture.md)，生产命令见 [`../docs/operations.md`](../docs/operations.md)。

## 启动

从仓库根目录运行。默认仅本机访问：

```bat
start_platform.bat
```

受信任局域网/Mac 访问使用：

```bat
start_platform.bat --lan
```

只有 `--lan` 入口监听 `0.0.0.0:9000`。平台无内置账户认证，只能用于受信任 LAN，并应通过 Windows 防火墙限制来源。

仅本机使用时：

```powershell
.\.venv\Scripts\python.exe -u -m uvicorn dubbing_service.app.main:app --host 127.0.0.1 --port 9000
```

生产不使用 `--reload`。入口：

- WebUI：`http://127.0.0.1:9000/web/`
- 就绪：`GET /api/v1/ready`
- 聚合健康：`GET /api/v1/health`
- OpenAPI：`http://127.0.0.1:9000/docs`

WebUI 挂载时不会自动启动 TTS。用户在平台配置中选择后端并明确启动；平台退出会清理其托管的 TTS 子进程。

## 运行架构

平台使用根 `.venv`。TTS 工作进程按注册表选择不同环境：

| ID | 端口 | 环境 | 服务脚本 |
| --- | ---: | --- | --- |
| `indextts` | 8001 | 根 `.venv` | `service/server.py` |
| `cosyvoice3` | 8002 | `environments/cosyvoice/.venv` | `service/cosyvoice_server.py` |
| `qwen3tts` | 8003 | `environments/qwen3tts/.venv` | `service/qwen3tts_server.py` |
| `qwen3tts_cv` | 8004 | `environments/qwen3tts/.venv` | `service/qwen3tts_customvoice_server.py` |

单一事实来源是 `app/core/tts_backends.py`。新增后端不要在前端、路由和启动脚本各自维护一套映射。

## 目录

```text
dubbing_service/
├── app/
│   ├── main.py                 # FastAPI、路由挂载、健康、生命周期
│   ├── core/                   # 配置、日志、错误、后端注册表
│   ├── models/                 # Project/Role/Segment/Subtitle
│   ├── providers/              # TTS、LLM、Whisper 适配
│   ├── routers/                # 业务 API 与 WebSocket
│   └── storage/                # 文件存储、锁、原子写入、事务
├── webui/                      # Vue 页面、JS、本地 vendor
├── config.example.json         # 无密钥模板
├── config.json                 # 本机私密配置，Git 忽略
├── llm_models.json             # Flash/Pro 等默认模型预设；源码导出包含
├── llm_providers.json          # Base URL/API Key，本机私密，源码导出排除
├── voices.json / voices/       # 本机音色库数据，源码导出排除
└── logs/                       # 运行日志，源码导出排除
```

## 配置

优先级：环境变量 > `config.json` > 默认值。

| 环境变量 | 含义 | 默认/回退 |
| --- | --- | --- |
| `DUBBING_TTS_BASE_URL` | IndexTTS2 | `http://127.0.0.1:8001` |
| `DUBBING_COSYVOICE_BASE_URL` | CosyVoice3 | 8002 |
| `DUBBING_QWEN3_TTS_BASE_URL` | Qwen VoiceDesign | 8003 |
| `DUBBING_QWEN3_TTS_CV_BASE_URL` | Qwen CustomVoice | 8004 |
| `DUBBING_TTS_API_KEY` | 可选 TTS 网关密钥 | 空 |
| `DUBBING_LLM_BASE_URL` | OpenAI-compatible LLM | 本机配置 |
| `DUBBING_LLM_API_KEY` | LLM 密钥 | 本机配置 |
| `DUBBING_PROJECTS_ROOT` | 用户项目根 | 仓库 `dubbing_projects/` |

不要在日志、issue、源码导出或健康检查截图中暴露密钥。

## 业务流程

### 项目和角色

- 一个项目对应一个文件目录和 `project.json`。
- 角色可以绑定参考音频/音色库，也可以保存角色级 `tts_backend`、Qwen speaker/language、VoiceDesign 描述和 style instruction。
- 后端优先级：分段 > 角色 > 全局 > `indextts`。
- IndexTTS2/CosyVoice3 需要角色参考音频；Qwen VD/CV 不需要。

### DeepSeek 自动分段

- 默认 `deepseek-v4-flash`、JSON 模式、thinking disabled。
- 网络错误、429、5xx 有界重试；认证/参数 4xx 立即失败。
- 校验 JSON、段落非空以及对原文的完整覆盖。
- 配置允许时，Flash 的可回退失败最多调用一次 `deepseek-v4-pro`。
- 验证通过后，`project.json`、`segments.json`、`text/raw.txt` 作为一个带回滚的应用层事务提交。

### 合成和导出

- HTTP 单段和 WebSocket 批量使用相同 Provider/后端规则与全局 TTS/GPU 租约；跨项目并发批次立即拒绝。
- 批处理期间不允许切换/停止工作进程或删除活动项目；租约与活动分段状态均采用所有权 token，异常路径安全释放。
- 每段按 `GENERATING → DONE/FAILED` 立即持久化；持久化失败返回 durability failure，不向前端误报成功。
- 音频先写临时/服务输出，再复制或原子提升到项目 `audio/segments/`；编辑、导入、角色删除和参考音频替换均先提交元数据，再回收旧文件。
- 波形草稿在切换项目和导出前 flush；整书导出由 FFmpeg 处理。
- 字幕由 faster-whisper 生成并原子保存 JSON/SRT。

## 数据一致性边界

- JSON 写入采用临时文件 + `os.replace`；重要路径会 flush/fsync。
- 项目元数据和分段使用进程内锁；跨两类数据固定锁顺序。
- 自动分段 staging 三个文件、保留备份并在任一步失败时回滚。
- 手工 JSON 导入和角色删除同时提交 project/segments；项目和分段关键写入 flush/fsync。
- 锁不是跨进程锁：禁止两个平台进程共享同一 `DUBBING_PROJECTS_ROOT`。
- 原子写入不是备份。项目删除是物理删除，上线前必须配置冷备份和恢复演练。

## 维护与测试

环境预检：

```powershell
.\.venv\Scripts\python.exe .\scripts\platform_doctor.py --startup
```

全量测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

DeepSeek 实时检查（会调用并可能计费）：

```powershell
.\.venv\Scripts\python.exe .\scripts\test_llm_connection.py
.\.venv\Scripts\python.exe .\scripts\smoke_llm_segmentation.py
```

只读项目审计：

```powershell
.\.venv\Scripts\python.exe .\scripts\audit_project_data.py
.\.venv\Scripts\python.exe .\scripts\audit_project_data.py --projects-root E:\path\to\recovered\dubbing_projects
```

不传参数时检查默认仓库数据根；外部 `DUBBING_PROJECTS_ROOT` 或恢复副本用 `--projects-root` 显式指定。

## 安全与许可证

- 9000 没有内置用户认证。`start_platform.bat` 的 LAN 监听模式必须受防火墙保护；公网需要 TLS、反向代理和鉴权。
- LLM 会把输入文本发送给 DeepSeek，使用前确认内容与地区合规。
- 参考声音、作品文本、字幕和输出都是私密业务数据。
- IndexTTS2 自定义许可、MaskGCT 非商业条款、CosyVoice/Qwen 和 nested third-party 许可证需要独立审查。技术测试通过不代表已经获准商业生产。
