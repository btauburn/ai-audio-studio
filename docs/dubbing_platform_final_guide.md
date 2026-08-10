# IndexTTS 配音平台（Dubbing Service）最终项目文档

> [!WARNING]
> 此文件是早期阶段归档，包含单一环境、旧启动命令和旧前端依赖描述，不再作为生产事实来源。当前说明请阅读 [架构说明](architecture.md)、[生产运维手册](operations.md) 和 [恢复报告](recovery_report.md)。

> 适用范围：`index-tts` 仓库中的 `dubbing_service` 平台
> 
> 目标：给生产使用提供统一说明，覆盖功能、架构、启动方式、配置项与运维要点。

---

## 1. 项目定位

`dubbing_service` 是基于 FastAPI 的配音平台后端 + 原生 WebUI，提供从「项目创建」到「整书导出 + 字幕生成」的一体化流程。

核心能力：

1. 项目管理（创建/删除/打开目录）
2. 角色管理（角色创建、参考音频上传、绑定音色库）
3. 文本导入与分段管理（手工导入 JSON / LLM 自动分段）
4. 多后端 TTS 合成（IndexTTS2 / CosyVoice3 / Qwen3 VoiceDesign / Qwen3 CustomVoice）
5. 分段编辑与预览（裁剪、静音、变速、音量、情感控制）
6. A/B 版本对比（B 版本合成、切换、清理）
7. 导出整书音频（含编辑态导出）
8. 字幕生成与编辑（Whisper）

---

## 2. 技术与架构

## 2.1 技术栈

- 后端：FastAPI + Pydantic
- 存储：文件系统 JSON（无数据库）
- 前端：`dubbing_service/webui/index.html + app.js`（CDN 加载 Vue）
- TTS Provider：
  - `indextts`
  - `cosyvoice3`
  - `qwen3tts`
  - `qwen3tts_cv`
- 字幕：faster-whisper

## 2.2 关键模块

- 路由：`dubbing_service/app/routers/*`
- Provider：`dubbing_service/app/providers/*`
- 存储：`dubbing_service/app/storage/*`
- 配置：`dubbing_service/app/core/config.py`
- 日志：`dubbing_service/app/core/logging_config.py`

API 前缀统一为：`/api/v1`

---

## 3. 生产推荐启动方式

## 3.1 Windows（推荐）

直接使用：

```bat
start_platform.bat
```

说明：

- 启动器默认以**非 reload 模式**运行，避免 Windows + Python 3.12 的 asyncio 噪声日志问题。
- 启动后自动打开：`http://127.0.0.1:9000/web/`

## 3.2 手工启动（可用于服务化）

在仓库根目录：

```powershell
uv run python -m uvicorn dubbing_service.app.main:app --host 127.0.0.1 --port 9000
```

生产模式请勿加 `--reload`。

---

## 4. 配置项（生产重点）

可通过环境变量 + `dubbing_service/config.json` 协同配置。

常用环境变量：

- `DUBBING_PROJECTS_ROOT`：项目数据目录
- `DUBBING_LOG_LEVEL`：日志级别（建议 `INFO`）
- `DUBBING_TTS_BASE_URL`：基础 TTS 服务地址（IndexTTS2/CosyVoice3）
- `DUBBING_LLM_BASE_URL` / `DUBBING_LLM_API_KEY`：LLM 配置

平台内还可通过「平台配置」页面设置：

- TTS/LLM 服务地址与密钥
- TTS 默认后端
- TTS 合成参数（温度、top_p、top_k、语速等）

---

## 5. 关键业务规则（上线必读）

1. 后端 ID 白名单：
   - `indextts` / `cosyvoice3` / `qwen3tts` / `qwen3tts_cv`
2. 分段后端回退逻辑：
   - 分段显式后端 > 全局后端 > `indextts`
3. A/B 音频命名：
   - A：`audio/segments/{id}.wav`
   - B：`audio/segments/{id}_alt.wav`
4. 分段状态一致性：
   - 如果当前活跃版本没有音频，不会保持 `done`
5. restore 安全校验：
   - 路径越界拒绝
   - 非法 `tts_backend`/`alt_tts_backend` 拒绝

---

## 6. 运行与监控

## 6.1 健康检查

```text
GET /api/v1/health
```

可观察：

- 服务状态
- TTS 健康与模型加载信息
- Whisper 模型加载信息
- 基础配置状态

## 6.2 日志

日志目录：`dubbing_service/logs/`

- 轮转策略：10MB × 5
- 已增加对 Windows 下特定 asyncio 噪声的精确过滤，降低日志污染

---

## 7. 数据与目录建议

生产建议：

1. 将 `DUBBING_PROJECTS_ROOT` 指向独立数据盘目录
2. 定期备份：
   - `dubbing_projects/*/project.json`
   - `segments.json`
   - 导出音频与字幕
3. 保留周期清理：
   - 使用平台内 cleanup 接口清理 tmp/debug/orphan 文件

---

## 8. 已验证状态（当前版本）

- WebUI 挂载流程已加状态标记与错误透传
- 启动器默认生产模式（非 reload）
- 核心路由/前端按钮契约/运行冒烟/链路回归均已通过
- 全量单测通过（当前：46/46）

---

## 9. 常见故障与处理

1. **页面提示初始化失败/超时**
   - 打开浏览器控制台查看首条红色错误
   - 强刷（Ctrl+F5）
   - 检查 `/web/app.js` 是否 200

2. **日志出现 asyncio `_write_send` 噪声**
   - 确认使用非 reload 模式启动
   - 启动器默认已规避；如手工运行请移除 `--reload`

3. **批量合成报角色缺少参考音频**
   - 为角色上传 `speaker_audio` 或绑定音色库 `speaker_voice_id`

4. **Qwen3 后端不可用**
   - 检查 `qwen3tts_base_url` / `qwen3tts_cv_base_url`
   - 检查对应服务是否已启动且健康

---

## 10. 版本管理建议

生产发布建议采用：

1. 固定 tag（如 `release/YYYYMMDD-x`）
2. 发布前执行：
   - 全量测试
   - 启动巡检清单
3. 变更记录至少包含：
   - API 变更
   - 配置变更
   - 数据结构变更
   - 回滚方案
