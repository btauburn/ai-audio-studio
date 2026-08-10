# IndexTTS2 配音平台 · 字幕层设计与音效层对接说明

> 版本：v1 · 适配当前仓库实现（Whisper 字幕层 + 导出整书音频）

---

## 1. 背景与总体目标

### 1.1 总体目标

本说明文档的目标是：

- **沉淀字幕层设计与实现细节**，便于后续维护和扩展；
- **明确字幕 JSON / SRT 等文件格式与约定**，为后续工具链（如外部编辑器、脚本）提供稳定接口；
- **为下一阶段的音效规划与生成层提供对接规范**，无论该层是：
  - 集成在当前配音平台项目中；
  - 还是拆分成一个单独的服务 / 仓库。

### 1.2 当前功能边界

当前「字幕层」的主要职责：

- 接收 **整书导出音频**，调用 **faster-whisper** 生成：
  - 含段落级时间戳的字幕；
  - 含词级时间戳的细粒度对齐信息；
- 将字幕以 **JSON + SRT** 的形式存储到项目目录；
- 提供基础的：
  - 字幕列表 / 获取；
  - 下载（JSON / SRT）；
  - 文本编辑（不改变时间轴与词级时间戳）。

**不做的事情**（留给后续音效/动画层或上层业务）：

- 不做音效规划（如 BGM、环境音、拟声、转场等）；
- 不做动画 / 字幕样式规划（如卡拉 OK 高亮、弹幕、镜头剪辑）；
- 不做语言学后处理（如自动纠错、术语统一等）。

字幕层的定位是：**「干净的时间轴 + 文本」基础设施**，为后续任何「基于时间线的内容规划」提供输入。

---

## 2. 系统分层概览

从上到下，当前系统与规划中的层级大致如下：

1. **文本 & 分段 & 情感层（已实现）**
   - 粘贴原文 → LLM 自动分段 → 每段生成情感描述等。
   - 输出结构：`segments` 列表（含角色、文本、情感、音频路径等）。

2. **TTS 合成层（已实现）**
   - 基于分段文本和角色信息，调用 IndexTTS2 TTS 服务生成海量分段音频；
   - 支持单段试听 / 批量合成 / 带编辑的导出整书音频；
   - 输出：
     - 分段音频：`audio/segments/...`；
     - 整书导出音频：`audio/final/...`。

3. **字幕层（已实现）**
   - 基于「整书导出音频」，调用 faster-whisper 生成字幕；
   - 生成 **段落级 + 词级时间戳**；
   - 存储为：
     - JSON（结构化 + 词级对齐信息）；
     - SRT（通用字幕格式，方便导入到常见视频编辑工具）；
   - 支持前端在 Modal 中编辑文本并保存。

4. **音效规划 & 生成层（规划中）**
   - 目标：基于「文本 + 字幕时间轴」，规划：
     - BGM 段落、进出点；
     - 环境音 / Foley / 特效音；
     - 可能的「节奏线」（如章节间空拍、呼吸位，配合剪辑）。
   - 可以是：
     - **同仓库内的一个模块**；
     - **独立项目**，与当前项目通过 API + 文件系统 / HTTP 接口通信。

字幕层位于 3，向上为编辑 UI 和导出逻辑，向下为 Whisper 模型与文件存储，是比较稳定的中间层。

---

## 3. 字幕层设计与实现

### 3.1 数据模型（Pydantic）

代码位置：`dubbing_service/app/models/subtitles.py`

#### 3.1.1 `SubtitleWord`

- 表示 **单词/词语** 的时间戳信息：
  - `index: int`：在整本字幕中的全局词序，从 0 开始；
  - `text: str`：词文本内容；
  - `start: float`：词起始时间（秒）；
  - `end: float`：词结束时间（秒）。

#### 3.1.2 `SubtitleSegment`

- 表示 **句子/分句级字幕段落**：
  - `index: int`：在整本字幕中的段落索引，从 0 开始；
  - `start: float`：段落起始时间（秒）；
  - `end: float`：段落结束时间（秒）；
  - `text: str`：该段落最终显示的完整文本；
  - `words: List[SubtitleWord]`：该段落内的词级时间戳列表。

> 约定：
> - 段落 `segments` 已按时间顺序排序；
> - `words` 内部也按时间顺序排序；
> - `start/end` 单位为秒，浮点数，允许小数（毫秒级）。

#### 3.1.3 `SubtitleFile`

- 表示 **完整的字幕文件**，基于一条整书音频生成：

核心字段：

- 标识与所属项目：
  - `id: str`：字幕 ID（短 UUID），同时用于 JSON 文件名；
  - `project_id: str`：所属项目 ID；

- 源音频信息：
  - `audio_rel_path: str`：源音频相对项目目录的路径，如 `audio/final/full_2024xxxx.wav`；
  - `audio_filename: str`：源音频文件名（方便前端展示与选择）。

- 存储位置（相对项目根目录）：
  - `json_rel_path: Optional[str]`：字幕 JSON 相对路径，一般为 `subtitles/<id>.json`；
  - `srt_rel_path: Optional[str]`：字幕 SRT 相对路径，一般为 `subtitles/<id>.srt`；

- 元信息：
  - `created_at: datetime`：创建时间（UTC）；
  - `updated_at: datetime`：最近更新时间（UTC）；
  - `language: Optional[str]`：识别出的语言代码（如 `zh` / `en` / `ja`），可为空；
  - `duration: Optional[float]`：整条音频总时长（秒）；

- 引擎信息：
  - `provider: Optional[str]`：引擎名称，如 `faster-whisper`；
  - `model_name: Optional[str]`：模型名，如 `large-v3-turbo`、`distil-large-v3` 等；

- 内容：
  - `segments: List[SubtitleSegment]`：按时间排序的全部字幕段落列表。

> 对音效层来说：`audio_rel_path` + `duration` + `segments[*].(start,end)` + `words[*]` 是最关键的信息。

#### 3.1.4 `SubtitleSummary`

- 用于列表展示的精简版本，字段包括：
  - `id`
  - `audio_rel_path`
  - `audio_filename`
  - `created_at`
  - `updated_at`
  - `language`
  - `duration`
  - `json_rel_path`
  - `srt_rel_path`

前端「导出 & 字幕」页面中「字幕文件」表格使用该模型。

#### 3.1.5 生成字幕的请求与响应

- `GenerateSubtitleRequest`
  - `audio_filename: Optional[str]`
    - 为空表示使用该项目 **最新一条整书导出**；
    - 否则根据 `audio/final/<audio_filename>` 精确匹配；
  - `language: Optional[str]`
    - 为空 → Whisper 自动检测；
    - 非空 → 传给 `WhisperModel.transcribe(language=...)`；
  - `task: Optional[Literal["transcribe", "translate"]]`
    - 转写原文 / 翻译成英文；
  - `model_size: Optional[str]`
    - 可覆写本次请求的 Whisper 模型，如 `large-v3-turbo` / `large-v3` / `distil-large-v3`；

- `GenerateSubtitleResponse`
  - `success: bool`
  - `subtitle_id: Optional[str]`
  - `json_rel_path: Optional[str]`
  - `srt_rel_path: Optional[str]`
  - `message: Optional[str]`

---

### 3.2 Whisper Provider 设计

代码位置：`dubbing_service/app/providers/subtitles.py`

#### 3.2.1 模型加载与缓存

- 使用 `faster_whisper.WhisperModel`：
  - 通过 `_detect_device_and_compute_type()` 自动选择：
    - 如果 `torch.cuda.is_available()`：
      - `device = "cuda"`；
      - `compute_type = "float16"`（即 fp16）；
    - 否则：
      - `device = "cpu"`；
      - `compute_type = "int8"`；

- 使用 `@lru_cache(maxsize=4)` 缓存 `_load_whisper_model(model_size, device, compute_type)`：
  - 同一进程内相同配置只加载一次，后续复用；
  - 上层通过 `clear_whisper_model_cache()` + `torch.cuda.empty_cache()` 手动释放显存；
  - `is_whisper_model_loaded()` 通过缓存 `currsize` 判断是否已加载。

#### 3.2.2 `FastWhisperSubtitleProvider`

构造函数：

- `__init__(model_size: str = "medium")`：
  - 当前业务逻辑会在路由层把默认改为 `large-v3-turbo`；
  - 内部会调用 `_detect_device_and_compute_type()` 决定 device / compute_type；
  - `_load_whisper_model()` 返回的模型实例被存到 `self._model`。

`generate_subtitles(...)` 核心步骤：

1. 校验音频文件存在；
2. 构建 `transcribe_kwargs`：
   - `beam_size = 5`；
   - `word_timestamps = True`；
   - 如果 `language` 非空 → `transcribe_kwargs["language"] = language`；
   - 如果 `task` 非空 → `transcribe_kwargs["task"] = task`；
3. 调用 `self._model.transcribe(str(audio_path), **transcribe_kwargs)`：
   - 返回 `segments_iter`（生成器）和 `info` 元数据；
4. 遍历 `segments_iter` 构建 `SubtitleSegment`：
   - 段落级：
     - `start`, `end`, `text`；
   - 词级：
     - 遍历 `seg.words`，为每个词构造 `SubtitleWord`，并维护全局 `index` 计数；
   - 对时间戳做基础兜底修正（避免全 0 / 倒序）；
5. 汇总：
   - `detected_lang = info.language`（可能为空）；
   - `total_duration = info.duration`，若为空/0 且有分段，则使用最后一个段落的 `end` 兜底；
6. 构建 `SubtitleFile` 返回。

> 这保证了：即使部分音频元数据不完整，`SubtitleFile.duration` 也尽量有有效值，便于后续音效层使用整体时长。

---

### 3.3 路由与存储

#### 3.3.1 存储布局

代码位置：`dubbing_service/app/storage/subtitles.py`

- 工程根配置：`AppConfig.projects_root_abs`（见 `core/config.py`）；
- 项目根目录：`<projects_root_abs>/<project_id>`；
- 字幕目录：`<project_root>/subtitles/`；
- 字幕 JSON 文件：`<project_root>/subtitles/<subtitle_id>.json`；
- SRT 文件路径：由 `SubtitleFile.srt_rel_path` 决定，默认 `subtitles/<subtitle_id>.srt`。

> `SubtitleFile.json_rel_path` / `srt_rel_path` 是**相对项目根目录**的路径字符串，不含绝对路径，便于跨机器迁移项目。

辅助函数：

- `generate_subtitle_id()`：短 UUID（8 字节 hex），用于文件名与 API 标识；
- `save_subtitle(project_id, subtitle)`：
  - 将 `SubtitleFile` 序列化为 JSON 写入对应文件；
- `load_subtitle(project_id, subtitle_id)`：
  - 从对应 JSON 读入并构造 `SubtitleFile`；
- `list_subtitles(project_id)`：
  - 遍历 `subtitles/*.json`；
  - **忽略损坏文件**，保证整体列表尽可能可用。

#### 3.3.2 API 路由

代码位置：`dubbing_service/app/routers/subtitles.py`

- `POST /api/projects/{project_id}/subtitles/generate-from-final`
  - 请求体：`GenerateSubtitleRequest`；
  - 流程：
    1. `_pick_final_audio_path(project_id, audio_filename)` 选择整书音频：
       - 若显式 `audio_filename`，则在 `audio/final/` 下精确查找；
       - 否则取该目录下**按修改时间排序的最新一个**；
    2. 用 `generate_subtitle_id()` 生成字幕 ID；
    3. 构造 `FastWhisperSubtitleProvider`，`model_size = payload.model_size or "large-v3-turbo"`；
    4. 调用 `generate_subtitles(...)` 生成 `SubtitleFile`；
    5. `_ensure_subtitle_paths()` 填充缺失的 `json_rel_path` / `srt_rel_path`；
    6. `save_subtitle(...)` 保存 JSON；
    7. 调用 `subtitle_file_to_srt()` 生成 SRT 并写入文件；
    8. 返回 `GenerateSubtitleResponse`。

- `GET /api/projects/{project_id}/subtitles`
  - 返回 `List[SubtitleSummary]`；
  - 内部调用 `list_subtitles(...)`，并按 `created_at` 逆序返回（最新的在前）。

- `GET /api/projects/{project_id}/subtitles/{subtitle_id}`
  - 返回完整 `SubtitleFile` JSON；
  - 供前端编辑 Modal 使用。

- `PUT /api/projects/{project_id}/subtitles/{subtitle_id}`
  - 请求体：`SubtitleFile`（整份）；
  - 行为：
    - 用传入的 `payload` 覆盖更新；
    - 强制 `subtitle.id = subtitle_id`，`subtitle.project_id = project_id`；
    - `_ensure_subtitle_paths(...)` 补完路径；
    - `subtitle.updated_at = datetime.utcnow()`；
    - `save_subtitle(...)` 写回 JSON；
    - 重新生成并写入 SRT 文件；
    - 返回更新后的 `SubtitleFile`。

- `GET /api/projects/{project_id}/subtitles/{subtitle_id}/download?format=srt|json`
  - `format=json`：
    - 根据 `json_rel_path` 定位 JSON；若文件缺失则尝试重新 `save_subtitle(...)`；
  - `format=srt`（默认）：
    - 根据 `srt_rel_path` 定位 SRT；若文件缺失则调用 `subtitle_file_to_srt(...)` 重建。

---

### 3.4 前端交互与状态

代码位置：`dubbing_service/webui/app.js` + `index.html`

#### 3.4.1 「导出 & 字幕」Tab

- 子功能：
  - 导出整书音频（带/不带编辑）；
  - 针对**最新导出音频**一键生成字幕；
  - 针对**任意一条导出记录**单独生成字幕；
  - 列出所有字幕文件，支持下载 JSON / SRT 与在 Modal 中编辑。

- 关键状态：
  - `subtitleTask`：`"transcribe" | "translate"`；
  - `subtitleLanguage`：识别语言下拉框（默认 `"zh"`，自动检测在最后）；
  - `subtitleModel`：Whisper 模型选择（`large-v3-turbo` 默认）；
  - `subtitles: SubtitleSummary[]`：字幕列表；
  - `subtitleEditing: SubtitleFile | null`：当前 Modal 中编辑的字幕；
  - `modals.subtitleEdit: boolean`：是否显示编辑 Modal。

#### 3.4.2 生成字幕

- `generateSubtitleForAudio(fa)`：
  - 前置：如检测到 TTS 模型已加载，会弹窗提示是否先卸载，以缓解显存压力；
  - 构造 payload：
    - `audio_filename = fa.filename`；
    - `task = subtitleTask`；
    - `language = subtitleLanguage`（为 `"auto"` 时传 `null`）；
    - `model_size = subtitleModel`；
  - 调用 `POST /subtitles/generate-from-final`；
  - 成功后：
    - Toast 提示；
    - 调用 `loadSubtitles()` 刷新字幕列表；
    - 调用 `refreshHealth()` 更新 GPU 灯状态。

- `generateSubtitleForLatestFull()`：
  - 从 `finalAudios` 中取最新一条，复用 `generateSubtitleForAudio(target)` 逻辑。

#### 3.4.3 字幕编辑 Modal

- 打开：
  - 列表行按钮 `@click="openSubtitleEditor(sub)"`；
  - 调用 `GET /subtitles/{id}` 拉取完整 `SubtitleFile`；
  - 将数据赋给 `subtitleEditing`。

- 编辑：
  - 只修改 `subtitleEditing.segments[*].text`；
  - 段落 `index`、时间戳 `start/end`、`words` 列表不会在 UI 中被改动。

- 保存：
  - `saveSubtitleEdits()` 调用 `PUT /subtitles/{id}`；
  - 成功后：
    - 以服务端返回的 `SubtitleFile` 覆盖本地 `subtitleEditing`；
    - Toast 提示；
    - 调用 `loadSubtitles()` 刷新列表（更新时间等）。

> 设计上故意保持「文本可编辑，时间轴不可更改」，让字幕层在时间线上对 TTS 和未来的 SFX 始终保持一致。

---

## 4. JSON 与 SRT 格式说明

### 4.1 JSON（`SubtitleFile`）

- **文件路径**：
  - 绝对路径：`<projects_root_abs>/<project_id>/subtitles/<subtitle_id>.json`；
  - 对外约定：以 `json_rel_path` 为准，始终为相对路径，便于在其他系统中识别和拼接。

- **结构**：
  - 直接序列化自 `SubtitleFile` Pydantic 模型；
  - `datetime` 字段采用 ISO8601 字符串格式（由 Pydantic 默认实现）。

> 为保持兼容性，建议任何外部工具在读写 JSON 时：
> - 不删除/重命名现有字段；
> - 新增字段时使用小写 + 下划线风格；
> - 不依赖字段顺序（JSON 无序）。

### 4.2 SRT

- **生成函数**：`subtitle_file_to_srt(subtitle: SubtitleFile) -> str`；
- **时间格式**：`HH:MM:SS,mmm`：
  - 小时：2 位；
  - 分钟：2 位；
  - 秒：2 位；
  - 毫秒：3 位；
- 每段结构：
  1. 行 1：序号（从 1 开始）；
  2. 行 2：`start --> end`；
  3. 行 3：文本；
  4. 空行分隔；
- 文本来源：`SubtitleSegment.text`，空则为 `""`。

> 时间边界处理：
> - 负数时间会被强制成 0；
> - 起止时间相等时表示零长度段，会出现在 SRT 中，但通常不会影响播放器。

---

## 5. 与音效规划/生成层的对接建议

你正在考虑是否将「音效规划 & 生成层」单独做成一个项目。下面从**数据接口**和**系统边界**两个维度给出建议。

### 5.1 关键对接信息

无论音效层在同一项目还是独立项目，对字幕层最重要的输出是：

1. **音频定位**
   - `project_id`
   - `audio_rel_path`（如 `audio/final/xxx.wav`）
   - 可由此推导出整书输出文件的绝对路径。

2. **整体时长**
   - `SubtitleFile.duration`（秒）
   - 若 Whisper 未提供 `info.duration`，会退化为 **最后一个字幕段 `end`**。

3. **时间轴**
   - 段落级：`segments[*].start / end / text`
   - 词级：`segments[*].words[*].start / end / text`

4. **语种与模型信息（可选）**
   - `language`：主要语言；
   - `provider` / `model_name`：有助于之后诊断「音效节拍与 ASR 误差」的关系。

音效层只要记住：

- **一个 `SubtitleFile` 对应一条「整书音频」**；
- SFX 规划可以以 `subtitle_id` 或 `(project_id, audio_rel_path)` 作主键来关联。

### 5.2 方案 A：在当前项目中增加音效层

特点：

- 与现有数据结构保持同仓；
- 可以共用 `projects_root_abs` + 现有的项目目录结构；
- API 风格可完全复用现有 FastAPI 结构。

可行的方式：

- 在 `dubbing_service/app/models/` 下新增 `sfx.py`：
  - 定义 `SfxPlanFile`：
    - `id`
    - `project_id`
    - `subtitle_id` 或 `audio_rel_path`
    - `events: List[SfxEvent]`
  - `SfxEvent`：
    - `id`
    - `time: float`（开始时间秒）
    - `duration: Optional[float]`
    - `category: Literal["bgm", "foley", "fx", ...]`
    - `payload: dict`（音源路径 / 预设名称 / 音量等）；

- 在 `dubbing_service/app/routers/` 下增加 `sfx.py`：
  - `POST /projects/{project_id}/sfx-plans`：创建/更新计划；
  - `GET /projects/{project_id}/sfx-plans?subtitle_id=...`：获取某字幕对应的音效计划；

- 在项目目录结构中新增：
  - `<project_root>/sfx/`：存放 `SfxPlanFile` JSON；
  - `<project_root>/sfx/renders/`：若需要缓存渲染后的「带音效的整书音频」；

优点：

- 共享部署与配置，更容易做到端到端的「一键配音 + 一键加音效」；
- IDE 和代码结构对你保持在一个上下文中，开发成本低；

缺点：

- 项目体积和职责继续膨胀；
- 若将来要单独下游（如视频编辑器）使用音效层 API，需要额外抽象。

### 5.3 方案 B：音效层作为独立项目

特点：

- 与当前配音平台解耦，作为「后处理」管线；
- 可被多个上游系统复用（不仅是当前 IndexTTS2 平台）。

对接方式建议：

1. **以文件系统为桥梁**
   - 约定一个共享的 `projects_root_abs`：
     - 音效项目通过 `project_id`、`subtitle_id`、`audio_rel_path` 直接读取字幕 JSON & 整书音频；
   - 不要求当前项目暴露字幕 JSON 的 HTTP 下载（虽然已经有），只需要路径约定；

2. **以轻量 API 为桥梁**
   - 当前项目：提供「枚举可用字幕」的 API（已存在 `GET /subtitles`）；
   - 音效项目：提供：
     - `POST /sfx-plans/from-subtitle`：输入 `project_id + subtitle_id`，输出 `SfxPlanFile`；
     - `POST /render`：输入 `project_id + subtitle_id + sfx_plan_id`，输出带音效的整书音频；

3. **版本与兼容性**
   - 需要将当前文档中的字段约定视为 **v1 协议**；
   - 如果将来要做「大改」（例如换掉 Whisper 或字幕结构），建议：
     - 为新结构增加 `version` 字段；
     - 或在路径上区分，如 `subtitles_v2/` 目录；
   - 保证音效项目可以根据版本选择不同解析逻辑。

优点：

- 清晰的模块边界，有利于单独演进音效算法；
- 可单独部署 / 扩缩容；

缺点：

- 本地开发时需要同时跑两个服务；
- 需要更多「协议文档」和测试来保证两边兼容。

### 5.4 推荐实践

结合你当前的目标和开发节奏，建议：

1. **先在当前项目中实现最小可用的音效层（方案 A）**：
   - 重用项目目录结构和配置；
   - 直接依赖 `SubtitleFile` JSON + `audio/final` 音频；
   - 先实现：
     - 一个简单的 `SfxPlanFile`（比如只挂 BGM 和章节间空拍）；
     - 一个简单的「音效渲染」脚本/接口；

2. 稳定后，如果发现有明确需求（如复用到其他产品），
   - 再考虑以当前数据结构为基础，对音效层做「抽取成独立项目」。

---

## 6. 后续开发建议

为提高后续开发顺畅度，可以考虑：

1. **为字幕与音效规划建立统一「时间单位与精度约定」**
   - 当前字幕以秒(float) 为单位，推荐 SFX 层也使用秒(float)，避免毫秒/帧制切换问题；

2. **不要直接修改字幕 JSON 结构**，而是：
   - 如需新增字段，保持兼容；
   - 对「重大变更」使用版本号或新目录；

3. **调试工具**
   - 后续可以考虑写一个小脚本/Notebook：
     - 给定 `project_id + subtitle_id`，画出字幕时间轴；
     - 叠加一个假想的 SFX 时间轴，验证对齐是否合理；

4. **日志与可观测性**
   - 重要地方（生成字幕 / 渲染带音效音频）建议打印：
     - 使用的模型名；
     - 语言参数；
     - 总时长；

本说明文档即为后续所有与字幕层 / 音效层交互的「v1 规范」，后续如果有大的结构演进，可以在本文件基础上新增「变更记录」章节。
