# 生产模式启动与巡检清单（IndexTTS 配音平台）

> [!WARNING]
> 此清单保留为历史参考，其中 `.venv` 单环境、`checkpoints/` 和通用 `uv run` 等内容已经过时。生产发布请使用 [operations.md](operations.md) 的“发布检查”，当前恢复事实见 [recovery_report.md](recovery_report.md)。

> 用法：发布前/重启后按顺序勾选。
> 
> 适用入口：`start_platform.bat` 或手工 `uvicorn` 启动。

---

## A. 启动前检查（Pre-flight）

- [ ] 代码版本已固定（tag 或明确 commit）
- [ ] `.venv` 依赖完整（`uv sync --all-extras` 已成功）
- [ ] 模型目录存在且可读（`checkpoints/`）
- [ ] `DUBBING_PROJECTS_ROOT` 已指向生产数据目录
- [ ] 磁盘可用空间充足（建议 > 20GB）
- [ ] GPU 可用（若使用 GPU 推理）
- [ ] 关键配置已确认：
  - [ ] `tts_base_url`
  - [ ] `qwen3tts_base_url`
  - [ ] `qwen3tts_cv_base_url`
  - [ ] `llm_base_url` / `llm_api_key`

---

## B. 生产模式启动（Start）

## B1. 推荐方式（Windows）

```bat
start_platform.bat
```

检查点：

- [ ] 启动器状态显示“运行中”
- [ ] 自动打开 `http://127.0.0.1:9000/web/`
- [ ] 未勾选开发重载（生产默认应关闭）

## B2. 手工方式（服务化）

```powershell
uv run python -m uvicorn dubbing_service.app.main:app --host 127.0.0.1 --port 9000
```

- [ ] 生产环境不使用 `--reload`

---

## C. 启动后 10 分钟巡检（Smoke）

## C1. 基础健康

- [ ] `GET /api/v1/health` 返回 200
- [ ] WebUI 首页可打开，无“初始化超时/失败”提示
- [ ] 日志无连续 ERROR 刷屏

## C2. 项目工作流

- [ ] 新建项目成功
- [ ] 导入文本成功
- [ ] 新建角色成功
- [ ] 上传角色参考音频成功（或绑定音色库成功）
- [ ] 导入 JSON 分段成功

## C3. 合成链路

- [ ] 分段后端下拉可用（indextts/cosyvoice3/qwen3tts/qwen3tts_cv）
- [ ] 单段合成成功（至少 1 段）
- [ ] 批量合成任务可启动并结束
- [ ] WebSocket 合成过程无异常中断

## C4. A/B 与编辑

- [ ] synthesize-alt 可生成 B 版本
- [ ] swap-alt 切换后播放正常
- [ ] clear-alt 后状态与音频一致（无 done+空音频）
- [ ] 音频编辑（裁剪/静音/变速/音量）至少验证 1 段

## C5. 导出与字幕

- [ ] 整书导出成功（full 或 full-with-edits）
- [ ] 字幕生成成功（Whisper）
- [ ] 字幕编辑保存成功

---

## D. 日志与资源巡检（Runbook）

- [ ] `dubbing_service/logs/` 轮转正常
- [ ] 无 asyncio `_SelectorSocketTransport._write_send` 噪声刷屏
- [ ] GPU 显存占用稳定，无持续上涨
- [ ] `dubbing_projects` 无异常激增临时文件

建议巡检频率：

- 首次上线：每 30 分钟一次（前 2 小时）
- 稳定后：每日 1 次

---

## E. 清理与维护

可在平台页面触发：

- [ ] 清理项目 tmp 文件
- [ ] 清理 audio 临时文件
- [ ] 清理分段 debug 文件
- [ ] 清理角色 prompts 冗余文件

全局维护建议：

- [ ] 每周检查 `dubbing_projects` 总容量
- [ ] 每周抽样检查导出音频与字幕可回放性

---

## F. 回滚与应急

- [ ] 保留上一版本代码与依赖锁文件
- [ ] 保留最近一次配置备份（`dubbing_service/config.json`）
- [ ] 保留业务数据备份（`dubbing_projects`）
- [ ] 回滚后优先验证：health / WebUI 打开 / 单段合成

---

## G. 发布准入标准（Go / No-Go）

满足以下全部条件才建议 Go：

- [ ] 全量自动化测试通过
- [ ] 本清单 A~F 无阻塞项
- [ ] 关键链路（创建项目→分段→合成→导出→字幕）全通
- [ ] 日志无持续 ERROR 刷屏

任一阻塞项未满足：**No-Go**。
