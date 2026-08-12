const { createApp, ref, computed, onMounted, onBeforeUnmount } = Vue;

if (typeof window !== "undefined") {
  window.__WEBUI_MOUNTED__ = false;
  window.__WEBUI_INIT_ERROR__ = null;
  window.addEventListener("error", (event) => {
    const msg = event && event.message ? String(event.message) : "未知错误";
    window.__WEBUI_INIT_ERROR__ = msg;
  });
}

createApp({
  setup() {
    const apiBase = ""; // same origin

    const health = ref(null);
    let healthRefreshPromise = null;
    let lastHealthRefreshAt = 0;
    const projects = ref([]);
    const llmConfigs = ref([]);
    const llmProviders = ref([]);
    const voices = ref([]);
    const selectedProjectId = ref(null);
    const showProjectList = ref(false);
    const roles = ref([]);
    const segments = ref([]);
    const emotionWarnings = ref({});
    const activePlaySegId = ref(null);

    const activeTab = ref("text");
    const tabs = [
      { id: "project", label: "脚本 & 音色" },
      { id: "text", label: "配音审听" },
      { id: "sound", label: "声音设计" },
      { id: "synthesis", label: "导出发布" },
    ];

    const leftTab = ref("projects");
    const taskListTab = ref("synth");

    const modals = ref({
      createProject: false,
      createRole: false,
      uploadAudio: false,
      importJson: false,
      selectVoice: false,
      editSegPrompt: false,
      createLlmProvider: false,
      createLlmConfig: false,
      editSegEmotion: false,
      subtitleEdit: false,
      taskList: false,
      takeHistory: false,
    });
    const takeHistoryState = ref({ segmentId: null, items: [], loading: false });

    function createRoleFormState(role = null, defaultLanguage = "Chinese") {
      const source = role && typeof role === "object" ? role : {};
      return {
        name: source.name || "",
        role_type: source.role_type || "narrator",
        description: source.description || "",
        tts_backend: source.tts_backend || "",
        qwen3_speaker: source.qwen3_speaker || "vivian",
        qwen3_language: source.qwen3_language || defaultLanguage || "Chinese",
        voice_description: source.voice_description || "",
        style_instruction: source.style_instruction || "",
        preferred_speed_min: Number(source.preferred_speed_min || 0.9),
        preferred_speed_max: Number(source.preferred_speed_max || 1.1),
        voice_source: source.voice_source || "",
        voice_license_name: source.voice_license_name || "",
        voice_commercial_use: source.voice_commercial_use === true,
        voice_rights_note: source.voice_rights_note || "",
      };
    }

    const forms = ref({
      project: { name: "", description: "" },
      role: createRoleFormState(),
    });
    const roleSaving = ref(false);

    const platformSettings = ref({
      tts_service_name: "",
      tts_base_url: "",
      cosyvoice_base_url: "",
      qwen3tts_base_url: "",
      qwen3tts_cv_base_url: "",
      tts_api_key: "",
      llm_service_name: "",
      llm_base_url: "",
      llm_api_key: "",
    });

    // TTS 后端配置
    const ttsBackend = ref({
      current: "indextts",
      available: [],
    });

    // TTS 语音合成参数配置
    const ttsParams = ref({
      max_text_tokens_per_segment: 120,
      max_mel_tokens: 1500,
      interval_silence: 200,
      temperature: 0.8,
      top_p: 0.8,
      top_k: 30,
      num_beams: 3,
      repetition_penalty: 1.8,
      length_penalty: 0.0,
      emotion_alpha: 0.65,
      context_injection_length: 50,
      cosyvoice_speed: 1.0,
      qwen3_default_language: "Chinese",
      // 默认值，用于显示"默认: xxx"
      defaults: {
        max_text_tokens_per_segment: 120,
        max_mel_tokens: 1500,
        interval_silence: 200,
        temperature: 0.8,
        top_p: 0.8,
        top_k: 30,
        num_beams: 3,
        repetition_penalty: 1.8,
        length_penalty: 0.0,
        emotion_alpha: 0.65,
        context_injection_length: 50,
        cosyvoice_speed: 1.0,
        qwen3_default_language: "Chinese",
      },
    });

    const TTS_PARAM_DEFAULTS = Object.freeze({
      max_text_tokens_per_segment: 120,
      max_mel_tokens: 1500,
      interval_silence: 200,
      temperature: 0.8,
      top_p: 0.8,
      top_k: 30,
      num_beams: 3,
      repetition_penalty: 1.8,
      length_penalty: 0.0,
      emotion_alpha: 0.65,
      context_injection_length: 50,
      cosyvoice_speed: 1.0,
      qwen3_default_language: "Chinese",
    });
    const TTS_NUMERIC_PARAM_KEYS = [
      "max_text_tokens_per_segment",
      "max_mel_tokens",
      "interval_silence",
      "temperature",
      "top_p",
      "top_k",
      "num_beams",
      "repetition_penalty",
      "length_penalty",
      "emotion_alpha",
      "context_injection_length",
      "cosyvoice_speed",
    ];
    const QWEN3_LANGUAGE_IDS = new Set([
      "Chinese",
      "English",
      "Japanese",
      "Korean",
      "German",
      "French",
      "Russian",
      "Portuguese",
      "Spanish",
      "Italian",
      "Auto",
    ]);

    function normalizeTtsParamsResponse(data, fallback, resetToDefaults = false) {
      const response = data && typeof data === "object" ? data : {};
      const current = fallback && typeof fallback === "object" ? fallback : {};
      const responseDefaults =
        response.defaults && typeof response.defaults === "object" ? response.defaults : {};
      const currentDefaults =
        current.defaults && typeof current.defaults === "object" ? current.defaults : {};
      const mergedDefaults = {
        ...TTS_PARAM_DEFAULTS,
        ...currentDefaults,
        ...responseDefaults,
      };
      const defaults = {};
      TTS_NUMERIC_PARAM_KEYS.forEach((key) => {
        const raw = mergedDefaults[key];
        const parsed = Number(raw);
        defaults[key] = raw !== null && raw !== "" && Number.isFinite(parsed)
          ? parsed
          : TTS_PARAM_DEFAULTS[key];
      });
      const defaultLanguage =
        typeof mergedDefaults.qwen3_default_language === "string"
          ? mergedDefaults.qwen3_default_language.trim()
          : "";
      defaults.qwen3_default_language = QWEN3_LANGUAGE_IDS.has(defaultLanguage)
        ? defaultLanguage
        : TTS_PARAM_DEFAULTS.qwen3_default_language;
      const valueFallback = resetToDefaults ? defaults : { ...defaults, ...current };
      const normalized = {};

      TTS_NUMERIC_PARAM_KEYS.forEach((key) => {
        const responseRaw = response[key];
        const fallbackRaw = valueFallback[key];
        const responseValue = Number(responseRaw);
        const fallbackValue = Number(fallbackRaw);
        normalized[key] = responseRaw !== null && responseRaw !== "" && Number.isFinite(responseValue)
          ? responseValue
          : fallbackRaw !== null && fallbackRaw !== "" && Number.isFinite(fallbackValue)
            ? fallbackValue
            : TTS_PARAM_DEFAULTS[key];
      });
      normalized.cosyvoice_speed = Math.min(
        2.0,
        Math.max(0.5, normalized.cosyvoice_speed),
      );

      const responseLanguage =
        typeof response.qwen3_default_language === "string"
          ? response.qwen3_default_language.trim()
          : "";
      const fallbackLanguage =
        typeof valueFallback.qwen3_default_language === "string"
          ? valueFallback.qwen3_default_language.trim()
          : "";
      normalized.qwen3_default_language = QWEN3_LANGUAGE_IDS.has(responseLanguage)
        ? responseLanguage
        : QWEN3_LANGUAGE_IDS.has(fallbackLanguage)
          ? fallbackLanguage
          : TTS_PARAM_DEFAULTS.qwen3_default_language;
      normalized.defaults = defaults;
      return normalized;
    }

    const getQwen3DefaultLanguage = () =>
      (ttsParams.value && ttsParams.value.qwen3_default_language) || "Chinese";

    const normalizeBaseUrl = (value, fallback) => {
      const raw = typeof value === "string" ? value.trim() : "";
      const base = raw || fallback || "";
      return base.replace(/\/+$/, "");
    };

    const replacePort = (baseUrl, fromPorts, targetPort) => {
      for (const port of fromPorts) {
        const token = `:${port}`;
        if (baseUrl.includes(token)) {
          return baseUrl.replace(token, `:${targetPort}`);
        }
      }
      return baseUrl;
    };

    const resolveTtsBaseUrl = (backend) => {
      const base = normalizeBaseUrl(platformSettings.value?.tts_base_url, "http://127.0.0.1:8001");
      if (backend === "cosyvoice3") {
        const override = normalizeBaseUrl(platformSettings.value?.cosyvoice_base_url, "");
        return override || replacePort(base, ["8001", "8003", "8004"], "8002");
      }
      if (backend === "qwen3tts") {
        const override = normalizeBaseUrl(platformSettings.value?.qwen3tts_base_url, "");
        return override || replacePort(base, ["8001", "8002"], "8003");
      }
      if (backend === "qwen3tts_cv") {
        const override = normalizeBaseUrl(platformSettings.value?.qwen3tts_cv_base_url, "");
        return override || replacePort(base, ["8001", "8002", "8003"], "8004");
      }
      return base;
    };

    const platformLoading = ref({
      load: false,
      save: false,
      test: false,
      unloadTts: false,
      unloadWhisper: false,
      ttsParams: false,
      globalSegPrompt: false,
    });

    // 全局 LLM 分段提示词模板（平台级配置）
    const globalSegPromptTemplate = ref({
      template: "",
      isDefault: true,
    });

    const llmTestResult = ref(null);
    const llmConfigTestStatus = ref({});

    const llm = ref({
      text: "",
      model: "deepseek-v4-flash",
      // 为空表示使用模型自身的默认温度，用户填写后才显式传递给后端
      temperature: null,
      // 为空表示使用模型自身的默认 top_p
      top_p: null,
      // 选中的 LLM 配置 ID，用于在自动分段时携带给后端
      configId: null,
    });

    const loading = ref({
      importText: false,
      autoSegment: false,
      batchSynthesize: false,
      exportFullWithEdits: false,
      // 整书字幕生成
      generateSubtitle: false,
      // 删除导出音频
      deleteFinalAudio: false,
    });

    const shouldStopBatchSynthesize = ref(false);
    const showStopBatchButton = ref(false);
    const stoppingBatchSynthesize = ref(false);
    const synthQueue = ref([]);
    let synthQueueRunning = false;

    const synthTasks = ref([]);
    const llmTasks = ref([]);

    // 断点续传：记录项目加载时检测到的中断合成信息
    const synthesisResumable = ref(null); // { resetCount, pendingCount, failedCount, doneCount, totalCount }

    // ── 撤销/重做 (Undo/Redo) ──────────────────────────────────────
    const UNDO_MAX_HISTORY = 50;
    const segmentUndoStack = [];   // 快照数组（每个元素为 segments JSON 字符串）
    const segmentRedoStack = [];
    const canUndo = ref(false);
    const canRedo = ref(false);
    let _undoRestoring = false;    // 防止恢复操作本身触发新快照

    function _updateUndoRedoFlags() {
      canUndo.value = segmentUndoStack.length > 0;
      canRedo.value = segmentRedoStack.length > 0;
    }

    function pushSegmentSnapshot() {
      if (_undoRestoring) return;
      const segs = segments.value;
      if (!segs || !segs.length) return;
      try {
        const snapshot = JSON.stringify(segs.map((s) => ({
          ...s,
          // 排除前端临时字段
          _synthPending: undefined,
          _emotionWarning: undefined,
          _lastErrorHint: undefined,
        })));
        segmentUndoStack.push(snapshot);
        if (segmentUndoStack.length > UNDO_MAX_HISTORY) {
          segmentUndoStack.shift();
        }
        // 新操作清空重做栈
        segmentRedoStack.length = 0;
        _updateUndoRedoFlags();
      } catch (e) {
        console.warn("保存撤销快照失败:", e);
      }
    }

    function clearUndoHistory() {
      segmentUndoStack.length = 0;
      segmentRedoStack.length = 0;
      _updateUndoRedoFlags();
    }

    async function _restoreSegmentSnapshot(snapshot) {
      if (!currentProject.value) return;
      const projectId = currentProject.value.id;
      _undoRestoring = true;
      try {
        const parsed = JSON.parse(snapshot);
        const data = await fetchJson(
          `/api/v1/projects/${projectId}/segments/restore`,
          {
            method: "PUT",
            body: JSON.stringify({ segments: parsed }),
          },
        );
        if (Array.isArray(data)) {
          segments.value = data;
          applyEmotionWarningsToSegments();
          reinitWaveformsForSegments();
        }
      } catch (e) {
        console.error("恢复段落快照失败:", e);
        showToast("撤销/重做失败: " + e.message);
      } finally {
        _undoRestoring = false;
        _updateUndoRedoFlags();
      }
    }

    async function undoSegments() {
      if (!segmentUndoStack.length || !currentProject.value) return;
      // 先保存当前状态到重做栈
      const segs = segments.value;
      if (segs && segs.length) {
        try {
          segmentRedoStack.push(JSON.stringify(segs));
        } catch {}
      }
      const snapshot = segmentUndoStack.pop();
      await _restoreSegmentSnapshot(snapshot);
    }

    async function redoSegments() {
      if (!segmentRedoStack.length || !currentProject.value) return;
      // 先保存当前状态到撤销栈
      const segs = segments.value;
      if (segs && segs.length) {
        try {
          segmentUndoStack.push(JSON.stringify(segs));
        } catch {}
      }
      const snapshot = segmentRedoStack.pop();
      await _restoreSegmentSnapshot(snapshot);
    }

    const hasRunningSynthTasks = Vue.computed(() =>
      synthTasks.value.some((t) => t && t.status === "running"),
    );
    const hasRunningLlmTasks = Vue.computed(() =>
      llmTasks.value.some((t) => t && t.status === "running"),
    );
    const hasRunningAnyTasks = Vue.computed(
      () => hasRunningSynthTasks.value || hasRunningLlmTasks.value,
    );

    function createSynthTask(config) {
      if (!config || !config.projectId) return null;
      const now = new Date();
      const pad = (n) => String(n).padStart(2, "0");
      const createdAtDisplay =
        now.getFullYear().toString() +
        "-" +
        pad(now.getMonth() + 1) +
        "-" +
        pad(now.getDate()) +
        " " +
        pad(now.getHours()) +
        ":" +
        pad(now.getMinutes()) +
        ":" +
        pad(now.getSeconds());

      const taskId =
        (config.type === "batch_synthesize" ? "batch_" : "single_") +
        now.getTime().toString(36) +
        "_" +
        Math.random().toString(36).slice(2, 6);

      const segmentIds = Array.isArray(config.segmentIds) ? config.segmentIds : [];

      const task = {
        id: taskId,
        type: config.type || "single_synthesize",
        projectId: config.projectId,
        projectName: config.projectName || config.projectId,
        createdAt: now.toISOString(),
        createdAtDisplay,
        status: "waiting",
        total: segmentIds.length || null,
        done: 0,
        failed: 0,
        message: "",
        stopRequested: false,
        segmentIds,
        segmentBackends: config.segmentBackends || {},
        remainingSegmentIds: [...segmentIds],
        _persistent: false,
      };

      synthTasks.value.unshift(task);
      if (synthTasks.value.length > 50) {
        synthTasks.value = synthTasks.value.slice(0, 50);
      }
      return task;
    }

    function persistentStatusToUi(status) {
      const mapping = {
        queued: "waiting",
        running: "running",
        succeeded: "completed",
        succeeded_with_errors: "completed_with_errors",
        failed: "failed",
        cancelled: "stopped",
        interrupted: "interrupted",
      };
      return mapping[status] || status || "waiting";
    }

    function taskFromPersistentJob(job) {
      const createdAt = job.created_at || new Date().toISOString();
      return {
        id: job.id,
        type: job.job_type,
        projectId: job.project_id,
        projectName: job.project_name || job.project_id,
        createdAt,
        createdAtDisplay: formatDateTime(createdAt),
        status: persistentStatusToUi(job.status),
        total: Number(job.total || (job.segment_ids || []).length || 0),
        done: Number(job.done || 0),
        failed: Number(job.failed || 0),
        message: job.message || "",
        stopRequested: !!job.stop_requested,
        segmentIds: Array.isArray(job.segment_ids) ? job.segment_ids : [],
        segmentBackends: job.segment_backends || {},
        remainingSegmentIds: Array.isArray(job.remaining_segment_ids)
          ? job.remaining_segment_ids
          : [],
        _persistent: true,
      };
    }

    async function persistSynthTask(task, segmentBackends = {}) {
      if (!task || !task.projectId) return null;
      task.segmentBackends = { ...(task.segmentBackends || {}), ...segmentBackends };
      const data = await fetchJson(
        `/api/v1/projects/${task.projectId}/jobs/${task.id}`,
        {
          method: "PUT",
          body: JSON.stringify({
            id: task.id,
            job_type: task.type,
            project_name: task.projectName,
            segment_ids: task.segmentIds || [],
            segment_backends: task.segmentBackends || {},
          }),
        },
      );
      task._persistent = true;
      return data;
    }

    async function recordPersistentJobSegment(task, segmentId, success, error = null) {
      if (!task || !task._persistent || !segmentId) return null;
      return fetchJson(
        `/api/v1/projects/${task.projectId}/jobs/${task.id}/segments/${segmentId}/result`,
        {
          method: "POST",
          body: JSON.stringify({ success: !!success, error }),
        },
      );
    }

    async function loadPersistentSynthTasks(projectId) {
      if (!projectId) return [];
      try {
        const jobs = await fetchJson(`/api/v1/projects/${projectId}/jobs`);
        const restored = (Array.isArray(jobs) ? jobs : []).map(taskFromPersistentJob);
        const otherProjects = (synthTasks.value || []).filter(
          (task) => task && task.projectId !== projectId,
        );
        synthTasks.value = [...restored, ...otherProjects].slice(0, 100);
        refreshBatchUiFlagsForCurrentProject();
        return restored;
      } catch (e) {
        console.error("加载持久合成任务失败:", e);
        return [];
      }
    }

    async function resumePersistentSynthTask(task, { alreadyQueued = false } = {}) {
      if (!task || !task._persistent) return;
      try {
        let job = null;
        if (task.status === "interrupted" && !alreadyQueued) {
          job = await fetchJson(
            `/api/v1/projects/${task.projectId}/jobs/${task.id}/resume`,
            { method: "POST" },
          );
        }
        const remaining = Array.isArray(job?.remaining_segment_ids)
          ? job.remaining_segment_ids
          : (task.remainingSegmentIds || task.segmentIds || []);
        if (!remaining.length) {
          showToast("该任务没有可继续的分段");
          return;
        }
        task.status = "waiting";
        task.stopRequested = false;
        task.remainingSegmentIds = [...remaining];
        await runSynthBatchWS(task, remaining, task.segmentBackends || {});
      } catch (e) {
        console.error("继续持久任务失败:", e);
        showToast("继续任务失败: " + e.message);
      }
    }

    async function autoResumeQueuedSynthTasks(projectId, tasks) {
      const queued = (tasks || []).filter(
        (task) => task.projectId === projectId && task.status === "waiting" && task._persistent,
      );
      for (const task of queued) {
        await resumePersistentSynthTask(task, { alreadyQueued: true });
      }
    }

    function createLlmTask(config) {
      if (!config || !config.projectId) return null;
      const now = new Date();
      const pad = (n) => String(n).padStart(2, "0");
      const createdAtDisplay =
        now.getFullYear().toString() +
        "-" +
        pad(now.getMonth() + 1) +
        "-" +
        pad(now.getDate()) +
        " " +
        pad(now.getHours()) +
        ":" +
        pad(now.getMinutes()) +
        ":" +
        pad(now.getSeconds());

      const taskId =
        "llm_" +
        now.getTime().toString(36) +
        "_" +
        Math.random().toString(36).slice(2, 6);

      const textLength =
        typeof config.textLength === "number" && Number.isFinite(config.textLength)
          ? config.textLength
          : null;

      const task = {
        id: taskId,
        type: "llm_segment",
        projectId: config.projectId,
        projectName: config.projectName || config.projectId,
        createdAt: now.toISOString(),
        createdAtDisplay,
        status: "running",
        total: 1,
        done: 0,
        failed: 0,
        message: "",
        textLength,
        totalSegments: null,
        newRolesCount: null,
      };

      llmTasks.value.unshift(task);
      if (llmTasks.value.length > 50) {
        llmTasks.value = llmTasks.value.slice(0, 50);
      }
      return task;
    }

    function enqueueSynthJobs(task, segmentIds, segmentBackends = {}) {
      if (!task) return;
      if (!Array.isArray(segmentIds) || !segmentIds.length) return;
      const projectId = task.projectId;
      const projectName = task.projectName;
      const fallbackBackend = (ttsBackend.value && ttsBackend.value.current) || "indextts";
      segmentIds.forEach((segId) => {
        if (!segId) return;
        synthQueue.value.push({
          id: `${task.id}:${segId}:${Math.random().toString(36).slice(2, 6)}`,
          taskId: task.id,
          projectId,
          projectName,
          segmentId: segId,
          ttsBackend: segmentBackends[segId] || fallbackBackend,
        });
      });
    }

    function findLatestBatchTaskForProject(projectId) {
      if (!projectId) return null;
      const list = synthTasks.value || [];
      return (
        list.find(
          (t) =>
            t &&
            t.type === "batch_synthesize" &&
            t.projectId === projectId &&
            (t.status === "waiting" || t.status === "running"),
        ) || null
      );
    }

    async function removeSynthTask(taskId) {
      if (!taskId) return;
      const list = synthTasks.value || [];
      const idx = list.findIndex((t) => t && t.id === taskId);
      if (idx === -1) return;
      const task = list[idx];
      if (task.status && task.status !== "waiting") {
        showToast("任务已开始执行，无法删除");
        return;
      }
      if (task._persistent) {
        try {
          await fetchJson(
            `/api/v1/projects/${task.projectId}/jobs/${task.id}`,
            { method: "DELETE" },
          );
        } catch (e) {
          showToast("删除任务失败: " + e.message);
          return;
        }
      }
      list.splice(idx, 1);
      synthTasks.value = list;
      synthQueue.value = synthQueue.value.filter((job) => job && job.taskId !== taskId);
    }

    async function stopSynthTask(taskId) {
      if (!taskId) return;
      const task = (synthTasks.value || []).find((t) => t && t.id === taskId);
      if (!task) return;
      if (
        task.status === "completed" ||
        task.status === "completed_with_errors" ||
        task.status === "failed" ||
        task.status === "stopped"
      ) {
        return;
      }
      task._stopping = true;
      task.stopRequested = true;
      if (task._persistent) {
        try {
          await fetchJson(
            `/api/v1/projects/${task.projectId}/jobs/${task.id}/cancel`,
            { method: "POST" },
          );
        } catch (e) {
          console.error("持久任务停止状态保存失败:", e);
        }
      }
      synthQueue.value = synthQueue.value.filter((job) => job && job.taskId !== taskId);
      if (task.status === "waiting") {
        task.status = "stopped";
      }
      if (task.type === "batch_synthesize" && task.status === "running") {
        shouldStopBatchSynthesize.value = true;
        stoppingBatchSynthesize.value = true;
      }
      refreshBatchUiFlagsForCurrentProject();
    }

    function refreshBatchUiFlagsForCurrentProject() {
      if (!currentProject.value) {
        loading.value.batchSynthesize = false;
        showStopBatchButton.value = false;
        stoppingBatchSynthesize.value = false;
        return;
      }
      const projectId = currentProject.value.id;
      const task = findLatestBatchTaskForProject(projectId);
      if (!task) {
        loading.value.batchSynthesize = false;
        showStopBatchButton.value = false;
        stoppingBatchSynthesize.value = false;
        return;
      }
      if (task.status === "waiting" || task.status === "running") {
        loading.value.batchSynthesize = true;
        showStopBatchButton.value = true;
      } else {
        loading.value.batchSynthesize = false;
        showStopBatchButton.value = false;
        stoppingBatchSynthesize.value = false;
      }
    }

    async function cleanupProjectTmpFiles() {
      if (!currentProject.value) {
        showToast("请先选择项目");
        return;
      }

      if (!window.confirm("将清理项目目录中原子写入残留的临时文件（*.tmp / *.tmp-<uuid>）。确认继续？")) {
        return;
      }

      try {
        const data = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/cleanup-tmp-files`,
          {
            method: "POST",
            body: JSON.stringify({}),
          },
        );
        const deleted = data && typeof data.deleted_files === "number" ? data.deleted_files : 0;
        showToast(`清理完成：删除临时文件 ${deleted} 个`);
      } catch (e) {
        console.error(e);
        showToast("清理失败: " + e.message);
      }
    }

    async function cleanupAudioTemp() {
      if (!currentProject.value) {
        showToast("请先选择项目");
        return;
      }

      if (!window.confirm("将清理 audio 下导出/编辑产生的临时目录与预览/临时文件（主要用于崩溃残留）。确认继续？")) {
        return;
      }

      try {
        const data = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/audio/cleanup-temp`,
          {
            method: "POST",
            body: JSON.stringify({}),
          },
        );
        const deletedDirs = data && typeof data.deleted_dirs === "number" ? data.deleted_dirs : 0;
        const deletedFiles = data && typeof data.deleted_files === "number" ? data.deleted_files : 0;
        showToast(`清理完成：删除目录 ${deletedDirs} 个，删除文件 ${deletedFiles} 个`);
      } catch (e) {
        console.error(e);
        showToast("清理失败: " + e.message);
      }
    }

    async function cleanupSegDebugFiles() {
      if (!currentProject.value) {
        showToast("请先选择项目");
        return;
      }

      if (!window.confirm("将清理 text/seg_prompt_used_* 与 logs/seg_llm_* 等分段调试文件，仅保留最近若干份。确认继续？")) {
        return;
      }

      try {
        const data = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/segments/cleanup-debug-files`,
          {
            method: "POST",
            body: JSON.stringify({}),
          },
        );
        const deletedText = data && typeof data.deleted_text_files === "number" ? data.deleted_text_files : 0;
        const deletedLogs = data && typeof data.deleted_log_files === "number" ? data.deleted_log_files : 0;
        showToast(`清理完成：text 删除 ${deletedText} 个，logs 删除 ${deletedLogs} 个`);
      } catch (e) {
        console.error(e);
        showToast("清理失败: " + e.message);
      }
    }

    async function cleanupRolePrompts() {
      if (!currentProject.value) {
        showToast("请先选择项目");
        return;
      }

      if (!window.confirm("将清理 roles/prompts 下重复/无用的角色参考音频文件。建议在不进行合成时执行。确认继续？")) {
        return;
      }

      try {
        const data = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/roles/cleanup-prompts`,
          {
            method: "POST",
            body: JSON.stringify({}),
          },
        );
        await loadRolesAndSegments();
        const deleted = data && typeof data.deleted_files === "number" ? data.deleted_files : 0;
        const updated = data && typeof data.updated_roles === "number" ? data.updated_roles : 0;
        showToast(`清理完成：删除文件 ${deleted} 个，更新角色 ${updated} 个`);
      } catch (e) {
        console.error(e);
        showToast("清理失败: " + e.message);
      }
    }

    function taskSegmentIndex(task) {
      if (!task || task.type !== "single_synthesize") return "-";
      if (!currentProject.value || currentProject.value.id !== task.projectId) return "-";
      const segIds = Array.isArray(task.segmentIds) ? task.segmentIds : [];
      if (!segIds.length) return "-";
      const segId = segIds[0];
      const list = segments.value || [];
      const idx = list.findIndex((s) => s && s.id === segId);
      if (idx === -1) return "-";
      return idx + 1;
    }

    const llmConfigForm = ref({
      id: null,
      name: "",
      provider_id: null,
      model: "deepseek-v4-flash",
      max_tokens_k: null,
      description: "",
      response_format_json: false,
    });

    const llmConfigSaving = ref(false);

    const llmProviderForm = ref({
      id: null,
      name: "",
      base_url: "",
      api_key: "",
      description: "",
    });

    const llmProviderSaving = ref(false);

    const selectedLlmProviderId = ref(null);

    const voiceForm = ref({
      id: null,
      name: "",
      tags: [],
      tagsInput: "",
      description: "",
      audio_path: null,
      created_at: null,
      updated_at: null,
    });

    const voiceSaving = ref(false);
    const voiceFileInput = ref(null);
    const voiceLocalPreviewUrl = ref(null);
    const voiceFormCard = ref(null);
    const selectedVoiceId = ref(null);
    const currentRoleForVoice = ref(null);
    const showVoiceForm = ref(false);

    // AI 生成音色状态
    const voiceAiGenerate = ref({
      instruct: "",
      previewText: "大家好，我是你的配音小助手。",
      language: "Chinese",
      generating: false,
      previewUrl: null,
      previewFile: null,
      error: null,
    });

    // AI 生成音色快捷标签（含常用对立标签）
    const voiceAiTags = [
      // 性别
      "男性", "女性", "中性",
      // 年龄
      "儿童", "青年", "中年", "老年",
      // 情绪对立
      "温柔", "强势", "沉稳", "激昂", "平静", "激动",
      "开心", "悲伤", "严肃", "活泼",
      // 音色特点对立
      "甜美", "沙哑", "清脆", "低沉", "磁性", "浑厚",
      // 语速对立
      "语速快", "语速慢",
      // 其他常用
      "成熟", "可爱", "知性", "阳光", "冷酷", "热情",
    ];

    // 特殊音色类型标签（用于区分语音和情感音色）
    const voiceTypeLabels = {
      speaker: "语音",
      emotion: "情感",
    };

    const voiceGenderTags = ["男", "女", "其他"];
    const voiceAgeTags = ["童年", "少年", "青年", "中年", "老年"];

    // 其他标签预设（风格/情绪/特点，含常用对立标签）
    const voiceStyleTags = [
      // 情绪对立
      "温柔", "强势",
      "沉稳", "激昂",
      "平静", "激动",
      "开心", "悲伤",
      "严肃", "活泼",
      // 音色特点对立
      "甜美", "沙哑",
      "清脆", "低沉",
      "磁性", "尖锐",
      "浑厚", "轻柔",
      // 语速对立
      "语速快", "语速慢",
      // 其他常用
      "成熟", "可爱", "知性", "阳光", "冷酷", "热情",
    ];

    const voiceTagPresets = [
      voiceTypeLabels.speaker,
      voiceTypeLabels.emotion,
      ...voiceGenderTags,
      ...voiceAgeTags,
      ...voiceStyleTags,
    ];

    // 音色库筛选状态
    const voiceFilterTags = ref([]);
    const voiceOtherSearch = ref("");

    // 计算过滤后的音色列表
    const filteredVoices = computed(() => {
      const list = Array.isArray(voices.value) ? voices.value : [];
      const tagFilters = Array.isArray(voiceFilterTags.value) ? voiceFilterTags.value : [];
      const q = (voiceOtherSearch.value || "").trim().toLowerCase();

      if (!tagFilters.length && !q) return list;

      return list.filter((v) => {
        if (!v) return false;
        const tags = Array.isArray(v.tags) ? v.tags : [];

        if (tagFilters.length) {
          const ok = tagFilters.every((tag) => tags.includes(tag));
          if (!ok) return false;
        }

        if (q) {
          const name = (v.name || "").toString().toLowerCase();
          const tagHit = tags.some((t) => (t || "").toString().toLowerCase().includes(q));
          const nameHit = name.includes(q);
          if (!nameHit && !tagHit) return false;
        }

        return true;
      });
    });

    function isLegacyUntypedVoice(voice) {
      const tags = voice && Array.isArray(voice.tags) ? voice.tags : [];
      return !tags.includes(voiceTypeLabels.speaker) && !tags.includes(voiceTypeLabels.emotion);
    }

    // 角色主音色不能绑定明确标记为“情感”的条目。为兼容旧项目，尚未
    // 采用类型标签的历史音色仍可见、可绑定，并在弹窗中明确提示补标。
    const speakerVoices = computed(() => {
      const list = Array.isArray(voices.value) ? voices.value : [];
      return list.filter((voice) => {
        if (!voice) return false;
        const tags = Array.isArray(voice.tags) ? voice.tags : [];
        return tags.includes(voiceTypeLabels.speaker) || isLegacyUntypedVoice(voice);
      });
    });

    // 切换筛选标签
    // 规则：
    // - 普通标签：可多选，使用 AND 关系；
    // - 特殊类型标签“语音音色/情感音色”：在筛选时互斥，不能同时选中两者。
    function toggleVoiceFilterTag(tag) {
      const idx = voiceFilterTags.value.indexOf(tag);

      // 已选中则取消
      if (idx >= 0) {
        voiceFilterTags.value.splice(idx, 1);
        return;
      }

      // 类型标签互斥
      if (isVoiceTypeTag(tag)) {
        const typeTags = [voiceTypeLabels.speaker, voiceTypeLabels.emotion];
        voiceFilterTags.value = voiceFilterTags.value.filter(
          (t) => !typeTags.includes(t),
        );
      } else if (isGenderTag(tag)) {
        // 性别标签互斥
        voiceFilterTags.value = voiceFilterTags.value.filter(
          (t) => !voiceGenderTags.includes(t),
        );
      } else if (isAgeTag(tag)) {
        // 年龄段标签互斥
        voiceFilterTags.value = voiceFilterTags.value.filter(
          (t) => !voiceAgeTags.includes(t),
        );
      }

      voiceFilterTags.value.push(tag);
    }

    // 清除所有筛选
    function clearVoiceFilter() {
      voiceFilterTags.value = [];
      voiceOtherSearch.value = "";
    }

    // 判断标签是否为特殊音色类型标签
    function isVoiceTypeTag(tag) {
      return tag === voiceTypeLabels.speaker || tag === voiceTypeLabels.emotion;
    }

    // 获取音色类型标签的样式类（鲜艳的相反色：语音=青色，情感=橙色）
    function getVoiceTypeTagClass(tag) {
      if (tag === voiceTypeLabels.speaker) {
        return "bg-teal-500 text-white border-teal-400";
      }
      if (tag === voiceTypeLabels.emotion) {
        return "bg-orange-500 text-white border-orange-400";
      }
      return "bg-slate-800 text-slate-200";
    }

    // 获取音色类型标签在 UI 中展示的文案（缩写为“语音/情感”）
    function getVoiceTypeDisplayText(tag) {
      if (tag === voiceTypeLabels.speaker) return "语音";
      if (tag === voiceTypeLabels.emotion) return "情感";
      return tag;
    }

    function isGenderTag(tag) {
      return voiceGenderTags.includes(tag);
    }

    function isAgeTag(tag) {
      return voiceAgeTags.includes(tag);
    }

    function getVoiceGenderTagClass(tag) {
      // 性别标签高亮色：男/女是一对相反色，其他保持醒目的粉色
      if (tag === "男") {
        return "bg-blue-500 text-white border-blue-400";
      }
      if (tag === "女") {
        return "bg-rose-500 text-white border-rose-400";
      }
      // 其他
      return "bg-fuchsia-500 text-white border-fuchsia-400";
    }

    function getVoiceAgeTagClass(tag) {
      // 年龄标签：按从年轻到年老，蓝色由浅到深
      switch (tag) {
        case "童年":
          return "bg-sky-300 text-white border-sky-200";
        case "少年":
          return "bg-sky-400 text-white border-sky-300";
        case "青年":
          return "bg-sky-500 text-white border-sky-400";
        case "中年":
          return "bg-sky-600 text-white border-sky-500";
        case "老年":
          return "bg-sky-700 text-white border-sky-600";
        default:
          return "bg-sky-500 text-white border-sky-400";
      }
    }

    function getVoiceFilterTagClass(tag, isActive) {
      if (isVoiceTypeTag(tag)) {
        if (isActive) {
          return getVoiceTypeTagClass(tag);
        }
        const base = getVoiceTypeTagClass(tag);
        return "border-slate-600 " + base.replace("bg-", "bg-opacity-30 bg-");
      }
      if (isGenderTag(tag)) {
        if (isActive) {
          return getVoiceGenderTagClass(tag);
        }
        const base = getVoiceGenderTagClass(tag);
        return "border-slate-600 " + base.replace("bg-", "bg-opacity-30 bg-");
      }
      if (isAgeTag(tag)) {
        if (isActive) {
          return getVoiceAgeTagClass(tag);
        }
        const base = getVoiceAgeTagClass(tag);
        return "border-slate-600 " + base.replace("bg-", "bg-opacity-30 bg-");
      }
      if (isActive) {
        return "border-primary-500 bg-primary-500/20 text-primary-200";
      }
      return "border-slate-700 text-slate-400 hover:border-slate-500";
    }

    function sortVoiceTags(tags) {
      if (!Array.isArray(tags)) {
        return [];
      }
      const result = [];
      const seen = new Set();

      // 1. 音色类型：语音音色 / 情感音色
      const typeOrder = [voiceTypeLabels.speaker, voiceTypeLabels.emotion];
      for (const t of typeOrder) {
        if (tags.includes(t)) {
          result.push(t);
          seen.add(t);
        }
      }

      // 2. 性别：按预设顺序
      for (const g of voiceGenderTags) {
        if (tags.includes(g)) {
          result.push(g);
          seen.add(g);
        }
      }

      // 3. 年龄：按预设顺序
      for (const a of voiceAgeTags) {
        if (tags.includes(a)) {
          result.push(a);
          seen.add(a);
        }
      }

      // 4. 其他标签：保持原有顺序，排在最后
      for (const tag of tags) {
        if (!seen.has(tag)) {
          result.push(tag);
          seen.add(tag);
        }
      }

      return result;
    }

    const lastExport = ref("");
    const finalAudios = ref([]);
    const masters = ref([]);
    const creatingMaster = ref(false);
    const productionSettings = ref({
      pronunciation_dictionary: [],
      sound_design: { version: 1, clips: [], mix: {} },
      release: {},
    });
    const audioAssets = ref([]);
    const qualityReport = ref(null);
    const releasePackages = ref([]);
    const productionBusy = ref(false);
    const pronunciationDraft = ref({ source: "", pronunciation: "", note: "" });
    const assetUploadForm = ref({
      track: "bgm", name: "", tags: "", source: "", author: "",
      license_name: "", commercial_use: true, proof_note: "",
    });
    const assetFileInput = ref(null);
    const coverFileInput = ref(null);
    const timelineDraft = ref({ asset_id: "", start: 0, duration: null, gain_db: 0, fade_in: 0, fade_out: 0, loop: false });
    const soundModelStatus = ref(null);
    const soundPlans = ref([]);
    const activeSoundPlanId = ref("");
    const soundJobs = ref([]);
    const soundGenerationBusy = ref(false);
    const soundPlanDensity = ref("balanced");
    const soundLicenseForm = ref({
      license_tier: "community",
      stability_registered: false,
      stability_terms_accepted: false,
      gemma_terms_accepted: false,
    });
    const soundPreview = ref(null);
    const soundPreviewAudioUrl = ref("");
    let soundJobPollTimer = null;
    const activeSoundPlan = computed(() => {
      const selected = soundPlans.value.find((item) => item.id === activeSoundPlanId.value);
      return selected || soundPlans.value[0] || null;
    });
    const subtitleTask = ref("transcribe");
    const subtitleLanguage = ref("zh");
    const subtitleModel = ref("large-v3-turbo");
    // 整书字幕列表
    const subtitles = ref([]);
    const subtitleEditing = ref(null);
    const subtitleEditingLoading = ref(false);
    const subtitleEditingSaving = ref(false);
    const subtitleEditMode = ref("table"); // 'table' or 'json'
    const subtitleJsonText = ref("");
    const toast = ref("");

    const currentRoleForUpload = ref(null);
    const audioInput = ref(null);

    // 从 JSON 导入分段相关状态
    const jsonImportText = ref("");
    const jsonFileInput = ref(null);

    // LLM 提示词查看/保存相关状态
    const llmOriginalText = ref("");
    const llmShowingPrompt = ref(false);
    const llmTextboxCollapsed = ref(false);  // 控制文本框折叠
    const segPromptTemplate = ref("");
    const segPromptTemplateSaving = ref(false);
    const segPromptTemplateOriginal = ref("");

    const editingProjectId = ref(null);
    const editingRoleId = ref(null);

    const segmentInlineEdit = ref({ id: null, field: null, value: "" });

    const segEmotionModal = ref({
      id: null,
      mode: "text", // text | audio | vector | none
      text: "",
      vector: [0, 0, 0, 0, 0, 0, 0, 0],
      // 音色库当前已应用的音色ID
      audioVoiceId: null,
      // 下拉框中临时选中的音色ID（点击"使用该音色"后才生效）
      voiceSelectId: null,
      // 已保存到分段的音频路径
      savedAudioPath: null,
      // 已保存到分段的音频来源：'recording' | 'upload' | null（仅用于前端展示）
      savedSource: null,
      // 当前音频来源：'voice' | 'upload' | 'recording' | 'saved' | null
      audioSource: null,
      // 上传文件 & 录制相关（仅前端状态）
      selectedFile: null,
      uploadPreviewUrl: null,
      recordedFile: null,
      recordedUrl: null,
      audioVersion: 0,
      // 分段的 TTS 后端
      ttsBackend: 'indextts',
      // 情感权重（仅 IndexTTS2）
      emotionAlpha: 0.65,
      // Qwen3-TTS CustomVoice 音色
      qwen3Speaker: 'vivian',
      // Qwen3-TTS 语言
      qwen3Language: getQwen3DefaultLanguage(),
    });
    const segEmotionUploadInput = ref(null);

    // 分段情感 AI 生成状态
    const segEmotionAiGenerate = ref({
      instruct: "",
      previewText: "",
      language: getQwen3DefaultLanguage(),
      generating: false,
      previewUrl: null,
      previewFile: null,
      error: null,
    });

    const segEmotionVoiceSearch = ref("");
    const segEmotionVoiceTypeFilter = ref("");

    const filteredEmotionVoices = computed(() => {
      const list = Array.isArray(voices.value) ? voices.value : [];
      const q = (segEmotionVoiceSearch.value || "").trim().toLowerCase();
      const typeTag = (segEmotionVoiceTypeFilter.value || "").trim();

      if (!q && !typeTag) return list;

      return list.filter((v) => {
        if (!v) return false;
        const tags = Array.isArray(v.tags) ? v.tags : [];

        if (typeTag) {
          if (!tags.includes(typeTag)) return false;
        }

        if (q) {
          const name = (v.name || "").toString().toLowerCase();
          const tagHit = tags.some((t) => (t || "").toString().toLowerCase().includes(q));
          const nameHit = name.includes(q);
          if (!nameHit && !tagHit) return false;
        }

        return true;
      });
    });

    function clearSegEmotionVoiceFilter() {
      segEmotionVoiceSearch.value = "";
      segEmotionVoiceTypeFilter.value = "";
    }

    // ===== 麦克风录制相关状态 =====
    const micRecording = ref({
      // 录制场景：'voice' = 音色表单, 'emotion' = 情感描述模态框
      context: null,
      isRecording: false,
      isPaused: false,
      mediaRecorder: null,
      audioChunks: [],
      recordedBlob: null,
      recordedUrl: null,
      duration: 0,
      timer: null,
    });

    // 开始麦克风录制
    async function startMicRecording(context) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        
        micRecording.value = {
          context,
          isRecording: true,
          isPaused: false,
          mediaRecorder,
          audioChunks: [],
          recordedBlob: null,
          recordedUrl: null,
          duration: 0,
          timer: null,
        };

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            micRecording.value.audioChunks.push(e.data);
          }
        };

        mediaRecorder.onstop = () => {
          const blob = new Blob(micRecording.value.audioChunks, { type: "audio/webm" });
          micRecording.value.recordedBlob = blob;
          if (micRecording.value.recordedUrl) {
            URL.revokeObjectURL(micRecording.value.recordedUrl);
          }
          micRecording.value.recordedUrl = URL.createObjectURL(blob);

          // 停止所有音轨
          stream.getTracks().forEach((track) => track.stop());
        };

        mediaRecorder.start();

        // 计时器
        micRecording.value.timer = setInterval(() => {
          if (micRecording.value.isRecording && !micRecording.value.isPaused) {
            micRecording.value.duration++;
          }
        }, 1000);
      } catch (err) {
        console.error("麦克风访问失败:", err);
        showToast("无法访问麦克风，请检查浏览器权限设置");
      }
    }

    // 停止麦克风录制
    function stopMicRecording() {
      if (micRecording.value.mediaRecorder && micRecording.value.isRecording) {
        micRecording.value.mediaRecorder.stop();
        micRecording.value.isRecording = false;
        if (micRecording.value.timer) {
          clearInterval(micRecording.value.timer);
          micRecording.value.timer = null;
        }
      }
    }

    // 重新录制
    function restartMicRecording() {
      const ctx = micRecording.value.context;
      cancelMicRecording();
      startMicRecording(ctx);
    }

    // 取消录制
    function cancelMicRecording() {
      if (micRecording.value.mediaRecorder && micRecording.value.isRecording) {
        micRecording.value.mediaRecorder.stop();
      }
      if (micRecording.value.timer) {
        clearInterval(micRecording.value.timer);
      }
      if (micRecording.value.recordedUrl) {
        URL.revokeObjectURL(micRecording.value.recordedUrl);
      }
      micRecording.value = {
        context: null,
        isRecording: false,
        isPaused: false,
        mediaRecorder: null,
        audioChunks: [],
        recordedBlob: null,
        recordedUrl: null,
        duration: 0,
        timer: null,
      };
    }

    // 格式化录制时长
    function formatRecordingDuration(seconds) {
      const m = Math.floor(seconds / 60);
      const s = seconds % 60;
      return `${m}:${s.toString().padStart(2, "0")}`;
    }

    // 使用录制的音频作为音色参考音频
    function useMicRecordingForVoice() {
      if (!micRecording.value.recordedBlob) return;
      // 创建 File 对象以便复用现有上传逻辑
      const file = new File([micRecording.value.recordedBlob], "recorded_audio.webm", {
        type: "audio/webm",
      });
      voiceForm.value.audioFile = file;
      voiceLocalPreviewUrl.value = micRecording.value.recordedUrl;
      // 不清除录制状态，让用户可以继续试听
    }

    // 使用录制的音频作为情感参考音频
    function useMicRecordingForEmotion() {
      if (!micRecording.value.recordedBlob) return;
      // 创建 File 对象
      const file = new File([micRecording.value.recordedBlob], "emotion_audio.webm", {
        type: "audio/webm",
      });
      segEmotionModal.value.recordedFile = file;
      segEmotionModal.value.recordedUrl = micRecording.value.recordedUrl;
      // 设置音频来源为录制
      segEmotionModal.value.audioSource = 'recording';
    }

    // 将录制的音频直接上传到分段（不添加到音色库）
    async function uploadRecordingToSegment() {
      if (!micRecording.value.recordedBlob) {
        showToast("没有可上传的录音");
        return;
      }
      if (!currentProject.value || !segEmotionModal.value.id) {
        showToast("当前项目或分段信息缺失，无法上传");
        return;
      }
      try {
        const file = new File([micRecording.value.recordedBlob], "recorded_audio.webm", {
          type: "audio/webm",
        });
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch(
          `/api/v1/projects/${currentProject.value.id}/segments/${segEmotionModal.value.id}/emotion-audio`,
          { method: "POST", body: formData }
        );
        if (!res.ok) {
          let detail = "";
          try {
            const cloned = res.clone();
            try {
              const data = await cloned.json();
              detail = data.detail || JSON.stringify(data);
            } catch {
              detail = await res.text();
            }
          } catch {
            // 响应体不可读时忽略
          }
          throw new Error(detail || `请求失败 (${res.status})`);
        }

        const updatedSeg = await res.json();

        // 更新 segments 数组中的对应分段
        const idx = segments.value.findIndex((s) => s.id === updatedSeg.id);
        if (idx !== -1) {
          Object.assign(segments.value[idx], updatedSeg);
        }

        // 更新模态框状态
        segEmotionModal.value.mode = "audio";
        segEmotionModal.value.savedAudioPath = updatedSeg.emotion_audio || null;
        segEmotionModal.value.audioSource = 'saved';
        segEmotionModal.value.audioVersion = (segEmotionModal.value.audioVersion || 0) + 1;
        // 清除录制状态（已上传保存）
        segEmotionModal.value.recordedFile = null;
        segEmotionModal.value.recordedUrl = null;

        showToast("录制的音频已上传到当前分段");
      } catch (e) {
        console.error("上传录音到分段失败:", e);
        showToast("上传失败: " + e.message);
      }
    }

    // 将录制的音频保存到音色库（名为"临时情感音频-时间"，标签为"情感"），
    // 同时同步一份到项目目录下的 emotion-recording 备份，便于刷新后恢复最近一次录制预览。
    async function saveRecordingToVoiceLibrary() {
      if (!micRecording.value.recordedBlob) {
        showToast("没有可保存的录音");
        return;
      }
      if (!currentProject.value || !segEmotionModal.value.id) {
        showToast("当前项目或分段信息缺失，无法保存录音到音色库");
        return;
      }
      try {
        // 生成时间戳名称
        const now = new Date();
        const timeStr = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}-${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`;
        const voiceName = `临时情感音频-${timeStr}`;
        
        // 1. 创建音色
        const payload = {
          name: voiceName,
          tags: [voiceTypeLabels.emotion],
          description: "麦克风录制的临时情感参考音频",
        };
        const voiceData = await fetchJson("/api/v1/voices", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        
        // 2. 上传录音文件到音色库
        const file = new File([micRecording.value.recordedBlob], "recorded_audio.webm", {
          type: "audio/webm",
        });
        const formData = new FormData();
        formData.append("file", file);
        const res = await fetch(`/api/v1/voices/${voiceData.id}/audio`, {
          method: "POST",
          body: formData,
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.detail || res.statusText);
        }
        console.debug("[Emotion] saveRecordingToVoiceLibrary: created voice in library", {
          voiceId: voiceData.id,
          voiceName,
        });

        // 3. 将本次录音同步到项目目录下的 emotion-recording 备份文件（segment_id.webm），用于刷新后恢复
        try {
          const backupForm = new FormData();
          backupForm.append(
            "file",
            new File([micRecording.value.recordedBlob], "recorded_backup.webm", { type: "audio/webm" }),
          );
          const ts = Date.now();
          const backupRes = await fetch(
            `/api/v1/projects/${currentProject.value.id}/segments/${segEmotionModal.value.id}/emotion-recording?t=${ts}`,
            { method: "POST", body: backupForm },
          );
          if (!backupRes.ok) {
            const data = await backupRes.json().catch(() => ({}));
            console.warn("[Emotion] 保存录音备份到 emotion-recording 失败", data);
          } else {
            console.debug("[Emotion] emotion-recording backup updated", {
              segmentId: segEmotionModal.value.id,
            });
          }
        } catch (e) {
          console.warn("[Emotion] 同步录音到 emotion-recording 备份失败", e);
        }

        // 4. 刷新音色列表并选中新创建的音色
        await loadVoices();
        segEmotionModal.value.audioVoiceId = voiceData.id;
        segEmotionModal.value.voiceSelectId = voiceData.id;
        segEmotionModal.value.audioSource = 'voice';
        
        showToast(`已保存到音色库：${voiceName}`);
      } catch (e) {
        console.error("保存录音到音色库失败:", e);
        showToast("保存失败: " + e.message);
      }
    }

    // 显式应用在下拉框中选择的音色
    function applySelectedEmotionVoice() {
      const vid = segEmotionModal.value.voiceSelectId;
      if (!vid) {
        segEmotionModal.value.audioVoiceId = null;
        if (segEmotionModal.value.savedAudioPath) {
          segEmotionModal.value.audioSource = 'saved';
        } else if (segEmotionModal.value.recordedUrl) {
          segEmotionModal.value.audioSource = 'recording';
        } else {
          segEmotionModal.value.audioSource = null;
        }
        return;
      }
      segEmotionModal.value.audioVoiceId = vid;
      segEmotionModal.value.audioSource = 'voice';
    }

    // 选择上传文件时，创建本地预览URL并保存文件引用
    function onEmotionFileSelected() {
      const input = segEmotionUploadInput.value;
      if (input && input.files && input.files[0]) {
        // 清除之前的预览URL
        if (segEmotionModal.value.uploadPreviewUrl) {
          URL.revokeObjectURL(segEmotionModal.value.uploadPreviewUrl);
        }
        // 保存文件引用，以便后续上传时使用
        segEmotionModal.value.selectedFile = input.files[0];
        segEmotionModal.value.uploadPreviewUrl = URL.createObjectURL(input.files[0]);
      } else {
        segEmotionModal.value.selectedFile = null;
        segEmotionModal.value.uploadPreviewUrl = null;
      }
    }

    // 获取情感音频的URL（用于试听）
    function getEmotionAudioUrl(audioPath) {
      if (!audioPath) return '';
      // 使用分段的情感音频API，添加时间戳避免缓存
      if (currentProject.value && segEmotionModal.value.id) {
        const ts = Date.now();
        return `/api/v1/projects/${currentProject.value.id}/segments/${segEmotionModal.value.id}/emotion-audio?t=${ts}`;
      }
      return '';
    }

    // 列宽配置：支持在分段列表和完整音频列表中拖动调整列宽
    const columnWidths = ref({
      segments: {
        text: null,
        emotion: null,
        audio: null,
        actions: null,
      },
      final: {
        filename: null,
        path: null,
        audio: null,
      },
    });

    // 分段音频波形编辑相关状态：按 segmentId 存储速度、音量、多段删除区间、当前播放位置、预设静音等
    const waveformStates = ref({});
    const WAVEFORM_STORAGE_KEY = "tts_waveform_states_v1";
    const LAST_PROJECT_KEY = "tts_last_project_id_v1";
    const wavesurferMap = {};
    let waveformPersistTimer = null;
    let waveformPersistProjectId = null;
    let waveformPersistDirty = false;
    let waveformPersistInFlight = null;
    let waveformPersistRevision = 0;
    let waveformPersistLastError = null;
    const segmentAudioEditPending = ref({});
    // 分段级试听所用的独立 Audio 对象，与主波形播放解耦，避免移动波形上的进度线
    let segmentPreviewAudio = null;
    let segmentPreviewObjectUrl = null;
    // 单段试听状态
    const singlePreviewSegmentId = ref(null);
    const singlePreviewPlaying = ref(false);
    // 平滑进度更新用的 requestAnimationFrame ID
    let previewProgressRAF = null;

    /**
     * 将处理后音频的播放时间映射到原始音频位置
     * 
     * 服务端处理顺序：
     * 1. 删除区间：原始音频 → 删除后音频
     * 2. 插入静音：在删除后音频上插入静音
     * 3. 变速：改变播放速度
     * 
     * 逆向映射：播放时间 → 内容时间 → 删除后位置 → 原始位置
     */
    function mapPreviewTimeToOriginal(playedTime, segId, processedDuration) {
      const st = getWaveformState(segId);
      const ws = wavesurferMap[segId];
      
      // 获取原始音频时长（从 WaveSurfer）
      let originalDuration = 0;
      if (ws && typeof ws.getDuration === "function") {
        originalDuration = ws.getDuration() || 0;
      }
      if (!originalDuration || originalDuration <= 0) {
        originalDuration = st.duration || processedDuration || 0;
      }
      
      const speed = st.speed || 1;
      const regions = st.regions || [];  // 删除区间（原始音频时间）
      const silences = st.silences || [];  // 静音插入（at 是原始音频时间）
      
      // 如果没有任何编辑，直接按比例映射
      if (speed === 1 && regions.length === 0 && silences.length === 0) {
        return { time: playedTime, originalDuration };
      }
      
      // 排序删除区间
      const sortedRegions = [...regions]
        .filter(r => r && typeof r.start === 'number' && typeof r.end === 'number' && r.end > r.start)
        .sort((a, b) => a.start - b.start);
      
      // 排序静音插入，并计算它们在删除后音频中的实际位置
      const sortedSilences = [...silences]
        .filter(s => s && typeof s.at === 'number' && typeof s.duration === 'number' && s.duration > 0)
        .sort((a, b) => a.at - b.at)
        .map(s => {
          // 计算静音点在删除后音频中的位置
          let deletedBefore = 0;
          for (const r of sortedRegions) {
            if (r.end <= s.at) {
              // 整个删除区间在静音点之前
              deletedBefore += r.end - r.start;
            } else if (r.start < s.at) {
              // 静音点在删除区间内，移到区间开始处
              deletedBefore += s.at - r.start;
            }
          }
          return { 
            ...s, 
            atInDeleted: s.at - deletedBefore  // 在删除后音频中的位置
          };
        });
      
      // 1. 变速：播放时间转换为内容时间
      let contentTime = playedTime * speed;
      
      // 2. 处理静音插入（逆向：从内容时间找到删除后位置）
      // 静音是在删除后音频上插入的
      let afterDeleteTime = contentTime;
      let originalTimeAtSilence = null;  // 如果在静音期间，记录原始位置
      
      if (sortedSilences.length > 0) {
        let processedOffset = 0;  // 处理后音频的偏移
        let deletedOffset = 0;    // 删除后音频的偏移
        
        for (const silence of sortedSilences) {
          const silencePos = silence.atInDeleted;  // 在删除后音频中的位置
          const silenceDur = silence.duration;
          const segmentLength = silencePos - deletedOffset;
          
          if (contentTime < processedOffset + segmentLength) {
            // 在这个静音插入点之前
            afterDeleteTime = deletedOffset + (contentTime - processedOffset);
            break;
          }
          
          processedOffset += segmentLength;
          
          // 检查是否在静音期间
          if (contentTime < processedOffset + silenceDur) {
            // 在静音期间，返回原始音频中的静音插入点
            originalTimeAtSilence = silence.at;
            break;
          }
          
          processedOffset += silenceDur;
          deletedOffset = silencePos;
          afterDeleteTime = deletedOffset + (contentTime - processedOffset);
        }
      }
      
      // 如果在静音期间，直接返回原始位置
      if (originalTimeAtSilence !== null) {
        return { 
          time: Math.max(0, Math.min(originalDuration, originalTimeAtSilence)), 
          originalDuration 
        };
      }
      
      // 3. 处理删除区间（逆向：从删除后位置找到原始位置）
      let originalTime = afterDeleteTime;
      if (sortedRegions.length > 0) {
        let deletedOffset = 0;    // 删除后音频的偏移
        let originalOffset = 0;   // 原始音频的偏移
        
        for (const region of sortedRegions) {
          const segmentLength = region.start - originalOffset;
          
          if (afterDeleteTime < deletedOffset + segmentLength) {
            // 在这个删除区间之前
            originalTime = originalOffset + (afterDeleteTime - deletedOffset);
            break;
          }
          
          deletedOffset += segmentLength;
          originalOffset = region.end;
          originalTime = originalOffset + (afterDeleteTime - deletedOffset);
        }
      }
      
      return { 
        time: Math.max(0, Math.min(originalDuration, originalTime)), 
        originalDuration 
      };
    }

    // 平滑更新预览进度（使用 requestAnimationFrame 实现流畅动画）
    function updatePreviewProgress(segId, audio, isSequential = false) {
      if (!audio) return;
      
      const update = () => {
        // 检查是否应该继续更新
        const shouldContinue = isSequential 
          ? sequentialPreviewRunning.value 
          : (singlePreviewPlaying.value && singlePreviewSegmentId.value === segId);
        
        if (!shouldContinue || audio.paused || audio.ended) {
          previewProgressRAF = null;
          return;
        }

        const playedTime = audio.currentTime || 0;
        const processedDuration = audio.duration || 0;
        
        if (processedDuration > 0) {
          const st = getWaveformState(segId);
          
          // 映射到原始音频位置
          const { time: originalTime, originalDuration } = mapPreviewTimeToOriginal(
            playedTime, segId, processedDuration
          );
          
          // 更新时间显示（显示原始音频位置，不修改 duration 以免影响标记位置）
          st.currentTime = originalTime;
          // 注意：不要修改 st.duration，它应该保持为原始音频时长
          
          // 同步 WaveSurfer 进度指示器到原始音频位置
          const ws = wavesurferMap[segId];
          if (ws && typeof ws.seekTo === "function" && originalDuration > 0) {
            const progress = originalTime / originalDuration;
            ws.seekTo(Math.min(1, Math.max(0, progress)));
          }
        }

        // 继续下一帧更新
        previewProgressRAF = requestAnimationFrame(update);
      };

      // 取消之前的动画帧
      if (previewProgressRAF) {
        cancelAnimationFrame(previewProgressRAF);
      }
      previewProgressRAF = requestAnimationFrame(update);
    }

    // 停止进度更新动画
    function stopPreviewProgressUpdate() {
      if (previewProgressRAF) {
        cancelAnimationFrame(previewProgressRAF);
        previewProgressRAF = null;
      }
    }

    function getWaveformState(segId) {
      const base = {
        speed: 1,
        volume: 1,
        // 当前选中的删除区间（便于 UI 文本展示），真实的多段列表存放在 regions 中
        region: null,
        // 是否处于“正在画新删除区间”的模式
        selecting: false,
        currentTime: 0,
        duration: 0,
        silenceSec: 0,
        // 预设静音插入参数：用户点击“添加间隔”时记录，用于后续试听和应用处理
        pendingSilenceAt: null,
        pendingSilenceDuration: null,
        // 当前时间是否处于“静音插入点”高亮状态（用于把时间文字渲染为黄色）
        highlightSilenceTime: false,
        // 多段静音插入
        silences: [],
        selectedSilenceId: null,
        // 多段删除区间
        regions: [],
        selectedRegionId: null,
        fadeIn: 0,
        fadeOut: 0,
        leadingSilence: 0,
        trailingSilence: 0,
        processingPreset: "none",
      };
      if (!segId) {
        return { ...base };
      }
      let state = waveformStates.value[segId];
      if (!state) {
        state = { ...base };
        waveformStates.value[segId] = state;
      } else {
        // 为老的持久化数据补充缺失字段，避免 undefined
        if (!Array.isArray(state.silences)) state.silences = [];
        if (!Array.isArray(state.regions)) state.regions = [];
        if (!Object.prototype.hasOwnProperty.call(state, "selectedSilenceId")) {
          state.selectedSilenceId = null;
        }
        if (!Object.prototype.hasOwnProperty.call(state, "selectedRegionId")) {
          state.selectedRegionId = null;
        }
        if (!Object.prototype.hasOwnProperty.call(state, "fadeIn")) state.fadeIn = 0;
        if (!Object.prototype.hasOwnProperty.call(state, "fadeOut")) state.fadeOut = 0;
        if (!Object.prototype.hasOwnProperty.call(state, "leadingSilence")) state.leadingSilence = 0;
        if (!Object.prototype.hasOwnProperty.call(state, "trailingSilence")) state.trailingSilence = 0;
        if (!Object.prototype.hasOwnProperty.call(state, "processingPreset")) state.processingPreset = "none";
      }
      return state;
    }

    function getSegmentBaseSpeed(seg) {
      const raw = Number(seg && seg.playback_speed);
      if (!Number.isFinite(raw)) return 1.0;
      return Math.min(2.0, Math.max(0.5, raw));
    }

    function getSegmentBaseVolume(seg) {
      const raw = Number(seg && seg.volume_gain);
      if (!Number.isFinite(raw)) return 1.0;
      return Math.min(2.0, Math.max(0.0, raw));
    }

    function getSegmentPreviewSpeed(seg) {
      if (!seg) return 1.0;
      const st = getWaveformState(seg.id);
      const desired =
        typeof st.speed === "number" && !Number.isNaN(st.speed)
          ? Math.min(2.0, Math.max(0.5, st.speed))
          : getSegmentBaseSpeed(seg);
      const base = getSegmentBaseSpeed(seg);
      return base > 1e-6 ? desired / base : desired;
    }

    function getSegmentPreviewVolume(seg) {
      if (!seg) return 1.0;
      const st = getWaveformState(seg.id);
      const desired =
        typeof st.volume === "number" && !Number.isNaN(st.volume)
          ? Math.min(2.0, Math.max(0.0, st.volume))
          : getSegmentBaseVolume(seg);
      const base = getSegmentBaseVolume(seg);
      const ratio = base > 1e-6 ? desired / base : desired;
      return Math.min(1.0, Math.max(0.0, ratio));
    }

    function hasEffectiveWaveformEditDraft(seg) {
      if (!seg) return false;
      const st = getWaveformState(seg.id);
      const regions = Array.isArray(st.regions) ? st.regions : [];
      const silences = Array.isArray(st.silences) ? st.silences : [];
      const hasRegion =
        regions.some(
          (r) => r && typeof r.start === "number" && typeof r.end === "number" && r.end > r.start,
        ) ||
        !!(st.region && typeof st.region.start === "number" && typeof st.region.end === "number" && st.region.end > st.region.start);
      const hasSilence = silences.some(
        (s) => s && typeof s.duration === "number" && s.duration > 0,
      );
      const desiredSpeed =
        typeof st.speed === "number" && !Number.isNaN(st.speed)
          ? Math.min(2.0, Math.max(0.5, st.speed))
          : getSegmentBaseSpeed(seg);
      const desiredVolume =
        typeof st.volume === "number" && !Number.isNaN(st.volume)
          ? Math.min(2.0, Math.max(0.0, st.volume))
          : getSegmentBaseVolume(seg);
      const hasSpeed = Math.abs(desiredSpeed - getSegmentBaseSpeed(seg)) > 1e-3;
      const hasVolume = Math.abs(desiredVolume - getSegmentBaseVolume(seg)) > 1e-3;
      const hasFinishing = Number(st.fadeIn || 0) > 0 || Number(st.fadeOut || 0) > 0
        || Number(st.leadingSilence || 0) > 0 || Number(st.trailingSilence || 0) > 0
        || (st.processingPreset && st.processingPreset !== "none");
      return hasRegion || hasSilence || hasSpeed || hasVolume || hasFinishing;
    }

    function loadWaveformStatesFromStorage(projectId) {
      // 为保证跨浏览器统一，从 现在 起不再使用 localStorage 恢复波形状态，全部依赖后端 edit_state
      return {};
    }

    function saveLastProjectId(projectId) {
      try {
        if (!projectId) {
          window.localStorage.removeItem(LAST_PROJECT_KEY);
        } else {
          window.localStorage.setItem(LAST_PROJECT_KEY, projectId);
        }
      } catch (e) {
        console.error(e);
      }
    }

    function formatTaskStatus(task) {
      if (!task) return "";
      const status = task.status || "";
      if (status === "waiting") return "等待中";
      if (status === "running") return "进行中";
      if (status === "completed") return "已完成";
      if (status === "completed_with_errors") return "已完成（部分失败）";
      if (status === "stopped") return "已停止";
      if (status === "interrupted") return "已中断，可继续";
      if (status === "failed") return "失败";
      return status || "";
    }

    function restoreLastProjectId() {
      try {
        const v = window.localStorage.getItem(LAST_PROJECT_KEY);
        return v || null;
      } catch (e) {
        console.error(e);
        return null;
      }
    }

    async function persistWaveformStatesForProject(projectId) {
      if (!projectId) return [];
      try {
        const itemsForServer = [];
        const current = waveformStates.value || {};

        Object.keys(current).forEach((segId) => {
          const st = current[segId];
          if (!st) return;
          const silencesArr = Array.isArray(st.silences) ? st.silences : [];
          const regionsArr = Array.isArray(st.regions) ? st.regions : [];
          const silenceSecValue =
            typeof st.silenceSec === "number" && !Number.isNaN(st.silenceSec)
              ? st.silenceSec
              : 0;

          const validRegions = regionsArr.filter((r) => {
            return (
              r &&
              typeof r.start === "number" &&
              typeof r.end === "number" &&
              r.end > r.start
            );
          });
          const orderedRegions = [...validRegions].sort((a, b) => {
            const aStart = typeof a.start === "number" ? a.start : 0;
            const bStart = typeof b.start === "number" ? b.start : 0;
            return aStart - bStart;
          });

          let selectedRegion = null;
          if (orderedRegions.length === 1 && !st.selectedRegionId) {
            selectedRegion = orderedRegions[0];
          } else if (st.selectedRegionId) {
            selectedRegion =
              orderedRegions.find((r) => r.id === st.selectedRegionId) || null;
          }

          const regionsForServer = orderedRegions.map((r) => ({
            id: r.id || null,
            start: typeof r.start === "number" ? r.start : 0,
            end: typeof r.end === "number" ? r.end : 0,
          }));

          const regionValue = selectedRegion
            ? {
                start: typeof selectedRegion.start === "number"
                  ? selectedRegion.start
                  : 0,
                end: typeof selectedRegion.end === "number"
                  ? selectedRegion.end
                  : 0,
              }
            : null;

          // 构造后端 SegmentEditState 结构
          itemsForServer.push({
            segment_id: segId,
            state: {
              speed:
                typeof st.speed === "number" && !Number.isNaN(st.speed)
                  ? st.speed
                  : null,
              volume:
                typeof st.volume === "number" && !Number.isNaN(st.volume)
                  ? st.volume
                  : null,
              // 兼容旧字段：保留一个“当前选中的区间”供老版本 UI 使用
              region: regionValue,
              // 新字段：完整的多段删除区间列表
              regions: regionsForServer,
              selected_region_id:
                st.selectedRegionId || (selectedRegion && selectedRegion.id) || null,
              silences: silencesArr.map((s) => ({
                id: s.id || null,
                at: typeof s.at === "number" ? s.at : 0,
                duration: typeof s.duration === "number" ? s.duration : 0,
              })),
              selected_silence_id: st.selectedSilenceId || null,
              silence_sec:
                typeof silenceSecValue === "number" && !Number.isNaN(silenceSecValue)
                  ? silenceSecValue
                  : null,
              fade_in: Number(st.fadeIn || 0),
              fade_out: Number(st.fadeOut || 0),
              leading_silence: Number(st.leadingSilence || 0),
              trailing_silence: Number(st.trailingSilence || 0),
              processing_preset: st.processingPreset || "none",
            },
          });
        });
        // 同步到后端，持久化到 segments.json，便于跨浏览器恢复
        if (itemsForServer.length) {
          try {
            const updatedSegments = await fetchJson(`/api/v1/projects/${projectId}/segments/edit-state`, {
              method: "POST",
              body: JSON.stringify({ items: itemsForServer }),
            });
            if (
              currentProject.value &&
              currentProject.value.id === projectId &&
              Array.isArray(updatedSegments) &&
              Array.isArray(segments.value)
            ) {
              const editStateById = new Map(
                updatedSegments.map((seg) => [seg.id, seg.edit_state || null]),
              );
              segments.value = segments.value.map((seg) => (
                editStateById.has(seg.id)
                  ? { ...seg, edit_state: editStateById.get(seg.id) }
                  : seg
              ));
            }
            return updatedSegments;
          } catch (e) {
            throw e;
          }
        }
      } catch (e) {
        console.error("保存波形编辑草稿失败:", e);
        throw e;
      }
      return [];
    }

    function reportWaveformPersistError(error) {
      waveformPersistLastError = error || new Error("未知错误");
      console.error("波形编辑草稿自动保存失败:", waveformPersistLastError);
      showToast(`波形编辑草稿保存失败: ${waveformPersistLastError.message || waveformPersistLastError}`);
    }

    function scheduleWaveformStatePersist(projectId, delayMs = 400) {
      if (!projectId) return;
      waveformPersistProjectId = projectId;
      waveformPersistDirty = true;
      waveformPersistRevision += 1;
      if (waveformPersistTimer) {
        clearTimeout(waveformPersistTimer);
      }
      waveformPersistTimer = setTimeout(() => {
        waveformPersistTimer = null;
        flushWaveformStatePersist(projectId).catch(reportWaveformPersistError);
      }, Math.max(0, delayMs));
    }

    async function flushWaveformStatePersist(projectId) {
      const targetProjectId = projectId || waveformPersistProjectId;
      if (!targetProjectId) return [];
      if (waveformPersistTimer) {
        clearTimeout(waveformPersistTimer);
        waveformPersistTimer = null;
      }
      if (waveformPersistInFlight) {
        try {
          await waveformPersistInFlight;
        } catch (_error) {
          // 当前 flush 会立即重试最新草稿，错误由本轮结果统一上抛。
        }
      }
      if (!waveformPersistDirty || waveformPersistProjectId !== targetProjectId) {
        if (waveformPersistLastError) {
          throw waveformPersistLastError;
        }
        return [];
      }

      const revisionAtStart = waveformPersistRevision;
      waveformPersistDirty = false;
      const request = persistWaveformStatesForProject(targetProjectId);
      waveformPersistInFlight = request;
      let requestFailed = false;
      try {
        const result = await request;
        waveformPersistLastError = null;
        if (waveformPersistRevision !== revisionAtStart) {
          waveformPersistDirty = true;
        }
        return result;
      } catch (error) {
        requestFailed = true;
        waveformPersistDirty = true;
        waveformPersistLastError = error;
        throw error;
      } finally {
        if (waveformPersistInFlight === request) {
          waveformPersistInFlight = null;
        }
        if (!requestFailed && waveformPersistDirty && waveformPersistProjectId === targetProjectId && !waveformPersistTimer) {
          scheduleWaveformStatePersist(targetProjectId);
        }
      }
    }

    function clearWaveformPersistTimer() {
      if (waveformPersistTimer) {
        clearTimeout(waveformPersistTimer);
        waveformPersistTimer = null;
      }
    }

    function onBeforeUnloadWithPendingWaveformEdits(event) {
      if (!waveformPersistDirty && !waveformPersistInFlight) return;
      event.preventDefault();
      event.returnValue = "";
    }

    function onGlobalKeydown(e) {
      const code = e.code || "";
      const key = e.key || "";

      // Ctrl+Z / Ctrl+Shift+Z：撤销/重做
      if ((e.ctrlKey || e.metaKey) && (key === "z" || key === "Z")) {
        // 如果焦点在输入框内，不拦截浏览器原生撤销
        const target = e.target;
        if (target) {
          const tag = target.tagName;
          if (target.isContentEditable || tag === "INPUT" || tag === "TEXTAREA") {
            return;
          }
        }
        e.preventDefault();
        if (e.shiftKey) {
          redoSegments();
        } else {
          undoSegments();
        }
        return;
      }

      // 空格键：播放/暂停当前选中段落
      if (code !== "Space" && key !== " " && key !== "Spacebar") {
        return;
      }
      const target = e.target;
      if (target) {
        const tag = target.tagName;
        const editable =
          target.isContentEditable ||
          tag === "INPUT" ||
          tag === "TEXTAREA" ||
          tag === "SELECT";
        if (editable) {
          return;
        }
      }
      e.preventDefault();
      const segId = activePlaySegId.value;
      if (!segId) return;
      const seg = (segments.value || []).find((s) => s && s.id === segId);
      if (!seg) return;
      toggleSegmentPlay(seg);
    }

    function destroyWaveSurferInstances() {
      Object.keys(wavesurferMap).forEach((id) => {
        const ws = wavesurferMap[id];
        if (ws && typeof ws.destroy === "function") {
          try {
            ws.destroy();
          } catch (e) {
            console.error(e);
          }
        }
        delete wavesurferMap[id];
      });
      activePlaySegId.value = null;
    }

    // 停止所有播放（波形播放、单段试听、连续试听）
    function stopAllPlayback() {
      // 停止所有 WaveSurfer 实例的播放
      Object.keys(wavesurferMap).forEach((id) => {
        const ws = wavesurferMap[id];
        if (ws && ws.isPlaying && ws.isPlaying()) {
          try {
            ws.pause();
          } catch (e) {
            console.error(e);
          }
        }
      });
      activePlaySegId.value = null;
      // 停止单段试听
      stopPreviewProgressUpdate();
      if (segmentPreviewAudio) {
        try {
          segmentPreviewAudio.pause();
          segmentPreviewAudio.onended = null;
        } catch (e) {
          console.error(e);
        }
      }
      singlePreviewSegmentId.value = null;
      singlePreviewPlaying.value = false;
      // 停止连续试听
      if (sequentialPreviewRunning.value) {
        sequentialPreviewStopRequested.value = true;
        sequentialPreviewRunning.value = false;
        sequentialPreviewCurrentSegmentId.value = null;
      }
    }

    function destroyWaveSurferInstance(segId) {
      if (!segId) return;
      if (activePlaySegId.value === segId) {
        activePlaySegId.value = null;
      }
      const ws = wavesurferMap[segId];
      if (ws && typeof ws.destroy === "function") {
        try {
          ws.destroy();
        } catch (e) {
          console.error(e);
        }
      }
      delete wavesurferMap[segId];
    }

    function resetWaveforms() {
      destroyWaveSurferInstances();
      waveformStates.value = {};
    }

    function reinitWaveformsForSegments() {
      destroyWaveSurferInstances();
      Vue.nextTick(() => {
        (segments.value || []).forEach((seg) => {
          initWaveformForSegment(seg);
        });
      });
    }

    const currentProject = Vue.computed(() =>
      projects.value.find((p) => p.id === selectedProjectId.value) || null,
    );

    const currentLlmConfig = Vue.computed(() => {
      if (!llm.value.configId) return null;
      return llmConfigs.value.find((c) => c.id === llm.value.configId) || null;
    });

    const filteredLlmConfigs = Vue.computed(() => {
      const providerId = selectedLlmProviderId.value;
      const configs = llmConfigs.value || [];
      if (!providerId) return configs;

      const providers = llmProviders.value || [];
      const currentProvider = providers.find((p) => p.id === providerId) || null;
      const isDefault = !!(currentProvider && currentProvider.is_default);

      return configs.filter((cfg) => {
        if (cfg.provider_id === providerId) return true;
        // 兼容旧数据：provider_id 为空表示“使用默认提供商”
        if (!cfg.provider_id && isDefault) return true;
        return false;
      });
    });

    const hasPendingSynthesis = Vue.computed(() => {
      if (loading.value.batchSynthesize) return true;
      const list = segments.value || [];
      return list.some((s) => s && s._synthPending);
    });

    function ensureProjectMutationAllowed(action = "修改项目") {
      if (!hasPendingSynthesis.value) return true;
      showToast(`当前项目正在合成，暂时不能${action}。请等待合成完成或先停止任务。`);
      return false;
    }

    // ── 项目统计面板 ──────────────────────────────────────────────
    const projectStats = Vue.computed(() => {
      const segs = segments.value || [];
      const total = segs.length;
      if (!total) return null;

      let pending = 0, generating = 0, done = 0, failed = 0;
      const roleSegCounts = {}; // roleId -> count

      segs.forEach((s) => {
        if (s.status === "done") done++;
        else if (s.status === "failed") failed++;
        else if (s.status === "generating") generating++;
        else pending++;

        const rid = s.role_id || "__unassigned__";
        roleSegCounts[rid] = (roleSegCounts[rid] || 0) + 1;
      });

      // 角色分布（按段落数降序，最多显示 8 个）
      const roleDistribution = Object.entries(roleSegCounts)
        .map(([roleId, count]) => {
          const role = (roles.value || []).find((r) => r.id === roleId);
          return {
            roleId,
            name: role ? role.name : (roleId === "__unassigned__" ? "未分配" : roleId),
            count,
            percent: Math.round((count / total) * 100),
          };
        })
        .sort((a, b) => b.count - a.count)
        .slice(0, 8);

      return {
        total,
        pending,
        generating,
        done,
        failed,
        donePercent: total ? Math.round((done / total) * 100) : 0,
        roleCount: (roles.value || []).length,
        roleDistribution,
      };
    });

    function initWaveformForSegment(seg) {
      if (!seg || !seg.audio_file) return;
      const containerId = `waveform-${seg.id}`;
      const selector = `#${containerId}`;
      if (wavesurferMap[seg.id]) return;
      const el = document.getElementById(containerId);
      if (!el) return;
      if (typeof WaveSurfer === "undefined") {
        console.warn("WaveSurfer 未加载");
        return;
      }
      const st = getWaveformState(seg.id);
      const projectId = selectedProjectId.value;
      // B 模式用紫色波形作视觉区分
      const inBMode = seg.active_version === "b";
      const waveColor = inBMode ? "#a78bfa" : "#38bdf8";
      const progressColor = inBMode ? "#8b5cf6" : "#0ea5e9";
      const ws = WaveSurfer.create({
        container: selector,
        waveColor,
        progressColor,
        height: 64,
        barWidth: 2,
        cursorColor: "#f97316",
      });
      // swap 后 audio_file 始终是当前版本的音频，直接用 /audio 端点
      ws.load(
        `/api/v1/projects/${projectId}/segments/${seg.id}/audio?ts=${encodeURIComponent(
          seg.updated_at || "",
        )}`,
      );
      ws.setPlaybackRate(getSegmentPreviewSpeed(seg));
      ws.setVolume(getSegmentPreviewVolume(seg));
      ws.on("ready", () => {
        const st2 = getWaveformState(seg.id);
        const dur = typeof ws.getDuration === "function" ? ws.getDuration() : 0;
        st2.duration = dur || 0;
        st2.currentTime = 0;
        // 如果存在已持久化的删除区间或静音间隔，重新赋值以触发样式计算刷新
        if (
          st2.region &&
          typeof st2.region.start === "number" &&
          typeof st2.region.end === "number" &&
          st2.region.end > st2.region.start
        ) {
          st2.region = { start: st2.region.start, end: st2.region.end };
        }
        if (Array.isArray(st2.silences) && st2.silences.length) {
          st2.silences = [...st2.silences];
        }
        if (seg && seg._synthPending) {
          seg._synthPending = false;
        }
      });
      ws.on("audioprocess", (time) => {
        const st2 = getWaveformState(seg.id);
        if (typeof time === "number" && Number.isFinite(time)) {
          st2.currentTime = time;
        } else if (typeof ws.getCurrentTime === "function") {
          st2.currentTime = ws.getCurrentTime();
        }
      });
      ws.on("seek", (progress) => {
        const st2 = getWaveformState(seg.id);
        const dur = st2.duration || (typeof ws.getDuration === "function" ? ws.getDuration() : 0);
        if (dur && typeof progress === "number") {
          st2.currentTime = Math.max(0, Math.min(dur, progress * dur));
        }
      });
      ws.on("finish", () => {
        const st2 = getWaveformState(seg.id);
        const dur = typeof ws.getDuration === "function" ? ws.getDuration() : st2.duration || 0;
        st2.currentTime = dur || 0;
        // 清除播放标记，触发 Vue 响应式更新（按钮从暂停恢复为播放）
        if (activePlaySegId.value === seg.id) {
          activePlaySegId.value = null;
        }
      });
      ws.on("pause", () => {
        if (activePlaySegId.value === seg.id) {
          activePlaySegId.value = null;
        }
      });
      wavesurferMap[seg.id] = ws;
    }

    function toggleSegmentPlay(seg) {
      if (!seg) return;
      const ws = wavesurferMap[seg.id];
      if (!ws) return;
      // 如果当前波形正在播放，则暂停
      if (ws.isPlaying && ws.isPlaying()) {
        ws.pause();
        activePlaySegId.value = null;
        return;
      }
      // 如果当前段落正在被试听（单段或连续），则停止试听而不是开始波形播放
      if (singlePreviewPlaying.value && singlePreviewSegmentId.value === seg.id) {
        stopSinglePreview();
        return;
      }
      if (sequentialPreviewRunning.value && sequentialPreviewCurrentSegmentId.value === seg.id) {
        stopSequentialPreviewSilent();
        showToast("已停止连续试听");
        return;
      }
      // 开始播放前，先停止所有其他播放
      stopAllPlayback();
      activePlaySegId.value = seg.id;
      ws.play();
    }

    function onWaveSpeedChange(seg, value) {
      if (!seg) return;
      const st = getWaveformState(seg.id);
      let v = parseFloat(value);
      if (!Number.isFinite(v)) return;
      v = Math.min(2.0, Math.max(0.5, v));
      st.speed = v;
      const ws = wavesurferMap[seg.id];
      if (ws) {
        ws.setPlaybackRate(getSegmentPreviewSpeed(seg));
      }
      const pid = selectedProjectId.value;
      if (pid) {
        scheduleWaveformStatePersist(pid);
      }
    }

    function onWaveVolumeChange(seg, value) {
      if (!seg) return;
      const st = getWaveformState(seg.id);
      let v = parseFloat(value);
      if (!Number.isFinite(v)) return;
      v = Math.min(2.0, Math.max(0.0, v));
      st.volume = v;
      const ws = wavesurferMap[seg.id];
      if (ws) {
        ws.setVolume(getSegmentPreviewVolume(seg));
      }
      const pid = selectedProjectId.value;
      if (pid) {
        scheduleWaveformStatePersist(pid);
      }
    }

    function startDeleteRegionSelection(seg) {
      if (!seg) return;
      const st = getWaveformState(seg.id);
      st.selecting = true;
      st.region = null;
      st.selectedRegionId = null;
    }

    function clearWaveformRegion(seg) {
      if (!seg) return;
      const st = getWaveformState(seg.id);
      const regions = Array.isArray(st.regions) ? st.regions : [];
      if (!regions.length) {
        st.region = null;
        st.selectedRegionId = null;
      } else {
        const targetId = st.selectedRegionId;
        if (!targetId) {
          showToast("请先点击要清除的删除区间");
          return;
        }
        const filtered = regions.filter((r) => r.id !== targetId);
        st.regions = filtered;
        if (!filtered.length) {
          st.region = null;
          st.selectedRegionId = null;
        } else {
          const ordered = [...filtered].sort((a, b) => {
            const aStart = typeof a.start === "number" ? a.start : 0;
            const bStart = typeof b.start === "number" ? b.start : 0;
            return aStart - bStart;
          });
          const first = ordered[0];
          st.selectedRegionId = first.id || null;
          st.region = { start: first.start, end: first.end };
        }
      }
      st.selecting = false;
      const pid = selectedProjectId.value;
      if (pid) {
        scheduleWaveformStatePersist(pid);
      }
    }

    function waveformRegionStyle(seg, region) {
      if (!seg || !region) return {};
      const st = getWaveformState(seg.id);
      const container = document.getElementById(`waveform-container-${seg.id}`);
      if (!container) return {};
      const duration =
        typeof st.duration === "number" && Number.isFinite(st.duration)
          ? st.duration
          : 0;
      const width = container.clientWidth || 1;
      if (!duration || !width) return {};
      const startPx = (region.start / duration) * width;
      const endPx = (region.end / duration) * width;
      const left = Math.max(0, Math.min(startPx, endPx));
      const right = Math.max(left, Math.max(startPx, endPx));
      return {
        left: `${left}px`,
        width: `${Math.max(1, right - left)}px`,
      };
    }

    function waveformSilenceRegionStyle(seg, silence) {
      if (!seg || !silence) return {};
      const container = document.getElementById(`waveform-container-${seg.id}`);
      if (!container) return {};
      const st = getWaveformState(seg.id);
      const total =
        typeof st.duration === "number" && Number.isFinite(st.duration)
          ? st.duration
          : 0;
      const width = container.clientWidth || 1;
      if (!total || !width) return {};
      const at = typeof silence.at === "number" ? silence.at : 0;
      const clamped = Math.max(0, Math.min(at, total));
      const centerPx = (clamped / total) * width;
      const lineHeight = 2; // 固定 2 像素高的水平线，位于波形中部
      const halfWidth = lineHeight / 2;
      const left = Math.max(0, Math.min(width - lineHeight, centerPx - halfWidth));
      return {
        left: `${left}px`,
        width: `${lineHeight}px`,
        top: "0px",
        bottom: "0px",
      };
    }

    function getWaveformTimeInfo(seg) {
      if (!seg) {
        return {
          current: 0,
          total: 0,
          currentAffected: false,
          hasSilence: false,
        };
      }
      const st = getWaveformState(seg.id);
      const baseCurrent =
        typeof st.currentTime === "number" && Number.isFinite(st.currentTime)
          ? st.currentTime
          : 0;
      const baseDuration =
        typeof st.duration === "number" && Number.isFinite(st.duration) ? st.duration : 0;
      const silences = Array.isArray(st.silences) ? st.silences : [];
      if (!silences.length) {
        return {
          current: baseCurrent,
          total: baseDuration,
          currentAffected: false,
          hasSilence: false,
        };
      }
      const sorted = [...silences].sort((a, b) => {
        const aAt = typeof a.at === "number" ? a.at : 0;
        const bAt = typeof b.at === "number" ? b.at : 0;
        return aAt - bAt;
      });
      let addedBeforeCurrent = 0;
      let totalSilence = 0;
      for (const s of sorted) {
        const at = typeof s.at === "number" ? s.at : 0;
        const dur = typeof s.duration === "number" ? s.duration : 0;
        if (dur <= 0) continue;
        totalSilence += dur;
        if (baseCurrent >= at) {
          addedBeforeCurrent += dur;
        }
      }
      return {
        current: baseCurrent + addedBeforeCurrent,
        total: baseDuration + totalSilence,
        currentAffected: addedBeforeCurrent > 0,
        hasSilence: totalSilence > 0,
      };
    }

    function getSelectedSilence(seg) {
      if (!seg) return null;
      const st = getWaveformState(seg.id);
      const silences = Array.isArray(st.silences) ? st.silences : [];
      if (!silences.length || !st.selectedSilenceId) return null;
      return silences.find((s) => s.id === st.selectedSilenceId) || null;
    }

    function onWaveformMouseDown(seg, event) {
      if (!seg) return;
      const st = getWaveformState(seg.id);
      activePlaySegId.value = seg.id;
      const ws = wavesurferMap[seg.id];
      if (!ws || typeof ws.getDuration !== "function") return;
      const container = event.currentTarget;
      const rect = container.getBoundingClientRect();
      const duration = ws.getDuration() || 0;
      if (!duration || !rect.width) return;

      const clampX = (rawX) => Math.min(Math.max(rawX - rect.left, 0), rect.width);
      const x0 = clampX(event.clientX);
      const t0 = (x0 / rect.width) * duration;
      const regions = Array.isArray(st.regions) ? st.regions : [];
      let existing = null;
      let activeRegionId = null;
      let mode = "select"; // select | move | resize-start | resize-end
      const wasSelecting = !!st.selecting;

      if (st.selecting) {
        mode = "select";
      } else if (regions.length) {
        const thresholdPx = Math.max(4, rect.width * 0.01);
        let hit = null;
        let hitMode = null;
        for (const r of regions) {
          if (
            !r ||
            typeof r.start !== "number" ||
            typeof r.end !== "number" ||
            r.end <= r.start
          ) {
            continue;
          }
          const startPx = (r.start / duration) * rect.width;
          const endPx = (r.end / duration) * rect.width;
          if (Math.abs(x0 - startPx) <= thresholdPx) {
            hit = r;
            hitMode = "resize-start";
            break;
          }
          if (Math.abs(x0 - endPx) <= thresholdPx) {
            hit = r;
            hitMode = "resize-end";
            break;
          }
          if (!hit && x0 > startPx && x0 < endPx) {
            hit = r;
            hitMode = "move";
          }
        }

        if (hit) {
          activeRegionId = hit.id || `${seg.id}-region-${hit.start}-${hit.end}`;
          st.selectedRegionId = activeRegionId;
          existing = { start: hit.start, end: hit.end };
          st.region = { ...existing };
          mode = hitMode || "move";
        } else {
          if (typeof ws.seekTo === "function") {
            const progress = Math.max(0, Math.min(1, t0 / duration));
            ws.seekTo(progress);
            const t = progress * duration;
            st.currentTime = t;
          }
          return;
        }
      } else {
        if (typeof ws.seekTo === "function") {
          const progress = Math.max(0, Math.min(1, t0 / duration));
          ws.seekTo(progress);
          const t = progress * duration;
          st.currentTime = t;
        }
        return;
      }

      const baseRegion = existing || { start: t0, end: t0 };
      const minLen = Math.max(0.05, duration * 0.02); // 选区最短时长，避免被压扁到 0

      function onMove(e) {
        const x = clampX(e.clientX);
        const t = (x / rect.width) * duration;

        if (mode === "select") {
          const t1 = Math.max(0, Math.min(t0, t));
          const t2 = Math.max(t1 + 1e-3, Math.max(t0, t));
          st.region = { start: t1, end: t2 };
        } else if (mode === "move") {
          const length = baseRegion.end - baseRegion.start;
          if (length <= 0) return;
          const delta = t - t0;
          let newStart = baseRegion.start + delta;
          newStart = Math.max(0, Math.min(newStart, duration - length));
          const newEnd = newStart + length;
          st.region = { start: newStart, end: newEnd };
        } else if (mode === "resize-start") {
          let newStart = Math.min(t, baseRegion.end - minLen);
          newStart = Math.max(0, newStart);
          st.region = { start: newStart, end: baseRegion.end };
        } else if (mode === "resize-end") {
          let newEnd = Math.max(t, baseRegion.start + minLen);
          newEnd = Math.min(duration, newEnd);
          st.region = { start: baseRegion.start, end: newEnd };
        }
      }

      function onUp() {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        if (st.region && st.region.end > st.region.start) {
          const current = { start: st.region.start, end: st.region.end };
          const all = Array.isArray(st.regions) ? st.regions : [];
          let id = activeRegionId;
          if (!id) {
            id =
              st.selectedRegionId ||
              `${seg.id}-region-${Date.now()}-${Math.random()
                .toString(36)
                .slice(2, 8)}`;
          }
          st.selectedRegionId = id;
          const others = all.filter((r) => r.id && r.id !== id);
          const updatedRegion = { id, start: current.start, end: current.end };
          st.regions = [...others, updatedRegion].sort((a, b) => {
            const aStart = typeof a.start === "number" ? a.start : 0;
            const bStart = typeof b.start === "number" ? b.start : 0;
            return aStart - bStart;
          });
        }
        st.selecting = false;
        const pid = selectedProjectId.value;
        if (pid) {
          scheduleWaveformStatePersist(pid);
        }
      }

      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    }

    function isSegmentAudioEditPending(seg) {
      return !!(seg && seg.id && segmentAudioEditPending.value[seg.id]);
    }

    function setSegmentAudioEditPending(segId, pending) {
      if (!segId) return;
      segmentAudioEditPending.value = {
        ...segmentAudioEditPending.value,
        [segId]: !!pending,
      };
    }

    async function applySegmentAudioEdit(seg, withDeleteRegion) {
      if (!currentProject.value || !seg) return;
      if (isSegmentAudioEditPending(seg)) return;
      const st = getWaveformState(seg.id);
      const payload = {
        speed: st.speed || 1.0,
        volume: typeof st.volume === "number" ? st.volume : 1.0,
        fade_in: Number(st.fadeIn || 0),
        fade_out: Number(st.fadeOut || 0),
        leading_silence: Number(st.leadingSilence || 0),
        trailing_silence: Number(st.trailingSilence || 0),
        processing_preset: st.processingPreset || "none",
      };
      if (withDeleteRegion) {
        let regionForDelete = null;
        const regions = Array.isArray(st.regions) ? st.regions : [];
        if (regions.length) {
          if (st.selectedRegionId) {
            const found = regions.find((r) => r.id === st.selectedRegionId);
            if (found && found.end > found.start) {
              regionForDelete = { start: found.start, end: found.end };
            }
          }
          if (!regionForDelete) {
            const ordered = [...regions].sort((a, b) => {
              const aStart = typeof a.start === "number" ? a.start : 0;
              const bStart = typeof b.start === "number" ? b.start : 0;
              return aStart - bStart;
            });
            const first = ordered[0];
            if (first && first.end > first.start) {
              regionForDelete = { start: first.start, end: first.end };
            }
          }
        } else if (st.region && st.region.end > st.region.start) {
          regionForDelete = { start: st.region.start, end: st.region.end };
        }
        if (regionForDelete) {
          payload.delete_region = {
            start: regionForDelete.start,
            end: regionForDelete.end,
          };
        }
      } else {
        // 未显式点击“删除区间”时，如果当前仍存在删除区间草稿，
        // 应用处理也应一并提交这些删除区间，避免只更新速度/音量而不真正裁剪。
        const previewPayload = buildPreviewPayloadForSegment(seg);
        if (previewPayload) {
          if (
            Array.isArray(previewPayload.delete_regions) &&
            previewPayload.delete_regions.length
          ) {
            payload.delete_regions = previewPayload.delete_regions.map((r) => ({
              start: r.start,
              end: r.end,
            }));
          } else if (previewPayload.delete_region) {
            payload.delete_region = {
              start: previewPayload.delete_region.start,
              end: previewPayload.delete_region.end,
            };
          }
        }
      }
      // 如果用户预设了多个静音插入参数，则在正式应用处理时一并提交给后端
      const silences = Array.isArray(st.silences)
        ? st.silences.filter(
            (s) =>
              typeof s.at === "number" &&
              typeof s.duration === "number" &&
              s.duration > 0,
          )
        : [];
      if (silences.length) {
        const ordered = [...silences].sort((a, b) => {
          const aAt = typeof a.at === "number" ? a.at : 0;
          const bAt = typeof b.at === "number" ? b.at : 0;
          return aAt - bAt;
        });
        payload.silence_intervals = ordered.map((s) => ({
          at: s.at,
          duration: s.duration,
        }));
        // 为保持与旧字段向后兼容，同时传递第一段静音
        payload.insert_silence_at = ordered[0].at;
        payload.silence_duration = ordered[0].duration;
      }
      setSegmentAudioEditPending(seg.id, true);
      try {
        const updated = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/segments/${seg.id}/edit-audio`,
          {
            method: "POST",
            body: JSON.stringify(payload),
          },
        );
        const idx = segments.value.findIndex((s) => s.id === updated.id);
        if (idx !== -1) {
          segments.value[idx] = updated;
        }
        st.speed =
          typeof updated.playback_speed === "number"
            ? updated.playback_speed
            : payload.speed;
        st.volume =
          typeof updated.volume_gain === "number"
            ? updated.volume_gain
            : payload.volume;
        st.region = null;
        const savedEditState = updated.edit_state || {};
        st.regions = Array.isArray(savedEditState.regions)
          ? savedEditState.regions.map((region) => ({ ...region }))
          : [];
        st.selectedRegionId = null;
        st.pendingSilenceAt = null;
        st.pendingSilenceDuration = null;
        st.silences = Array.isArray(savedEditState.silences)
          ? savedEditState.silences.map((silence) => ({ ...silence }))
          : [];
        st.selectedSilenceId = null;
        st.fadeIn = Number(savedEditState.fade_in || 0);
        st.fadeOut = Number(savedEditState.fade_out || 0);
        st.leadingSilence = Number(savedEditState.leading_silence || 0);
        st.trailingSilence = Number(savedEditState.trailing_silence || 0);
        st.processingPreset = savedEditState.processing_preset || "none";
        const ws = wavesurferMap[seg.id];
        if (ws) {
          ws.load(
            `/api/v1/projects/${currentProject.value.id}/segments/${seg.id}/audio?ts=${encodeURIComponent(
              updated.updated_at || "",
            )}`,
          );
          ws.setPlaybackRate(getSegmentPreviewSpeed(updated));
          ws.setVolume(getSegmentPreviewVolume(updated));
        }
        showToast(
          withDeleteRegion
            ? "删除区间已保存（原始音频未修改）"
            : "编辑方案已保存（原始音频未修改）",
        );
      } catch (e) {
        console.error(e);
        showToast("更新音频失败: " + (e.message || e));
      } finally {
        setSegmentAudioEditPending(seg.id, false);
      }
    }

    function applySegmentAudioProcessing(seg) {
      return applySegmentAudioEdit(seg, false);
    }

    async function renderSegmentAsNewTake(seg) {
      if (!currentProject.value || !seg || isSegmentAudioEditPending(seg)) return;
      if (!seg.edit_state) {
        showToast("请先保存编辑方案");
        return;
      }
      setSegmentAudioEditPending(seg.id, true);
      try {
        const updated = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/segments/${seg.id}/render-as-new-take`,
          { method: "POST" },
        );
        const idx = segments.value.findIndex((item) => item.id === updated.id);
        if (idx !== -1) Object.assign(segments.value[idx], updated);
        const realSeg = idx !== -1 ? segments.value[idx] : updated;
        Vue.nextTick(() => {
          destroyWaveSurferInstance(realSeg.id);
          initWaveformForSegment(realSeg);
        });
        showToast("已渲染为新 Take，原始版本已保留在历史记录中");
      } catch (e) {
        console.error("渲染新 Take 失败:", e);
        showToast("渲染新 Take 失败: " + (e.message || e));
      } finally {
        setSegmentAudioEditPending(seg.id, false);
      }
    }

    function deleteWaveformRegion(seg) {
      if (!seg) return;
      const st = getWaveformState(seg.id);
      if (!st.region || st.region.end <= st.region.start) {
        showToast("请先选择要删除的区间");
        return;
      }
      return applySegmentAudioEdit(seg, true);
    }

    function addSilenceAtCursor(seg) {
      if (!seg) return;
      const st = getWaveformState(seg.id);
      const ws = wavesurferMap[seg.id];
      const sec = Number(st.silenceSec || 0);
      if (!Number.isFinite(sec) || sec <= 0) {
        showToast("请先输入大于 0 的间隔时长（秒）");
        return;
      }
      let pos = 0;
      if (ws && typeof ws.getCurrentTime === "function") {
        pos = ws.getCurrentTime();
      } else if (typeof st.currentTime === "number") {
        pos = st.currentTime;
      }

      // 将插入时间与时长都量化到 0.01 秒，便于与前端展示精度一致
      const roundedPos = Math.max(0, Math.round(pos * 100) / 100);
      const roundedSec = Math.max(0, Math.round(sec * 100) / 100);

      // 不立即调用后端修改原始音频，只在前端记录一次“预插入静音”的设置（可多段）
      if (!Array.isArray(st.silences)) {
        st.silences = [];
      }
      const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      st.silences.push({
        id,
        at: roundedPos,
        duration: roundedSec,
      });
      st.silences.sort((a, b) => {
        const aAt = typeof a.at === "number" ? a.at : 0;
        const bAt = typeof b.at === "number" ? b.at : 0;
        return aAt - bAt;
      });
      st.selectedSilenceId = id;

      const pid = selectedProjectId.value;
      if (pid) {
        scheduleWaveformStatePersist(pid);
      }

      showToast("已预添加间隔：仅用于本段后续的试听和应用处理，可重复添加多段");
    }

    function clearSilenceAt(seg) {
      if (!seg) return;
      const st = getWaveformState(seg.id);
      const silences = Array.isArray(st.silences) ? st.silences : [];
      if (!silences.length) {
        st.selectedSilenceId = null;
        return;
      }
      const targetId = st.selectedSilenceId;
      if (!targetId) {
        showToast("请先点击要清除的间隔");
        return;
      }
      st.silences = silences.filter((s) => s.id !== targetId);
      if (!st.silences.length) {
        st.selectedSilenceId = null;
      } else {
        const ordered = [...st.silences].sort((a, b) => {
          const aAt = typeof a.at === "number" ? a.at : 0;
          const bAt = typeof b.at === "number" ? b.at : 0;
          return aAt - bAt;
        });
        st.selectedSilenceId = ordered[0].id;
      }

      const pid = selectedProjectId.value;
      if (pid) {
        scheduleWaveformStatePersist(pid);
      }
    }

    function onSilenceMarkerClick(seg, silence) {
      if (!seg || !silence) return;
      const st = getWaveformState(seg.id);
      const at = typeof silence.at === "number" ? silence.at : 0;
      if (!Array.isArray(st.silences)) {
        st.silences = [];
      }
      st.selectedSilenceId = silence.id || null;
      const ws = wavesurferMap[seg.id];
      let dur = st.duration;
      if ((!dur || !Number.isFinite(dur)) && ws && typeof ws.getDuration === "function") {
        dur = ws.getDuration() || 0;
      }
      if (ws && typeof ws.seekTo === "function" && dur && dur > 0) {
        const progress = Math.max(0, Math.min(1, at / dur));
        ws.seekTo(progress);
      }
      st.currentTime = at;
    }

    function buildPreviewPayloadForSegment(seg) {
      if (!seg) return null;
      const st = getWaveformState(seg.id);

      const payload = {
        speed: st.speed || 1.0,
        volume: typeof st.volume === "number" ? st.volume : 1.0,
        fade_in: Number(st.fadeIn || 0),
        fade_out: Number(st.fadeOut || 0),
        leading_silence: Number(st.leadingSilence || 0),
        trailing_silence: Number(st.trailingSilence || 0),
        processing_preset: st.processingPreset || "none",
      };

      const regions = Array.isArray(st.regions)
        ? st.regions.filter(
            (r) =>
              r &&
              typeof r.start === "number" &&
              typeof r.end === "number" &&
              r.end > r.start,
          )
        : [];
      if (regions.length) {
        const orderedRegions = [...regions].sort((a, b) => {
          const aStart = typeof a.start === "number" ? a.start : 0;
          const bStart = typeof b.start === "number" ? b.start : 0;
          return aStart - bStart;
        });
        payload.delete_regions = orderedRegions.map((r) => ({
          start: r.start,
          end: r.end,
        }));
        // 为保持与旧字段向后兼容，同时传递第一段删除区间
        const first = orderedRegions[0];
        payload.delete_region = {
          start: first.start,
          end: first.end,
        };
      } else if (st.region && st.region.end > st.region.start) {
        // 兼容旧逻辑：当还没有写入 regions 列表但存在单一区间时
        payload.delete_region = {
          start: st.region.start,
          end: st.region.end,
        };
      }

      const silences = Array.isArray(st.silences)
        ? st.silences.filter(
            (s) =>
              typeof s.at === "number" &&
              typeof s.duration === "number" &&
              s.duration > 0,
          )
        : [];
      if (silences.length) {
        const ordered = [...silences].sort((a, b) => {
          const aAt = typeof a.at === "number" ? a.at : 0;
          const bAt = typeof b.at === "number" ? b.at : 0;
          return aAt - bAt;
        });
        payload.silence_intervals = ordered.map((s) => ({
          at: s.at,
          duration: s.duration,
        }));
        // 为保持与旧字段向后兼容，同时传递第一段静音
        payload.insert_silence_at = ordered[0].at;
        payload.silence_duration = ordered[0].duration;
      } else if (
        typeof st.pendingSilenceAt === "number" &&
        typeof st.pendingSilenceDuration === "number" &&
        st.pendingSilenceDuration > 0
      ) {
        // 兼容旧的单一静音字段
        payload.insert_silence_at = st.pendingSilenceAt;
        payload.silence_duration = st.pendingSilenceDuration;
      }

      return payload;
    }

    // 停止单段试听
    function stopSinglePreview() {
      stopPreviewProgressUpdate();
      if (segmentPreviewAudio) {
        try {
          segmentPreviewAudio.pause();
          segmentPreviewAudio.onended = null;
        } catch (e) {
          console.error(e);
        }
      }
      singlePreviewSegmentId.value = null;
      singlePreviewPlaying.value = false;
    }

    async function previewSegment(seg) {
      if (!currentProject.value || !seg) return;

      // 如果正在试听同一个段落，则停止
      if (singlePreviewPlaying.value && singlePreviewSegmentId.value === seg.id) {
        stopSinglePreview();
        return;
      }

      // 开始播放前，先停止所有其他播放
      stopAllPlayback();

      const payload = buildPreviewPayloadForSegment(seg);
      if (!payload) return;

      try {
        if (!segmentPreviewAudio) {
          segmentPreviewAudio = new Audio();
        } else {
          try {
            segmentPreviewAudio.pause();
            segmentPreviewAudio.ontimeupdate = null;
            segmentPreviewAudio.onended = null;
          } catch (e) {
            console.error(e);
          }
        }

        const res = await fetch(
          `/api/v1/projects/${currentProject.value.id}/segments/${seg.id}/preview-audio`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
          },
        );

        if (!res.ok) {
          let detail = "";
          try {
            const cloned = res.clone();
            try {
              const data = await cloned.json();
              detail = data.detail || JSON.stringify(data);
            } catch {
              detail = await res.text();
            }
          } catch {
            // 响应体不可读时忽略
          }
          throw new Error(detail || `请求失败 (${res.status})`);
        }

        const blob = await res.blob();
        if (segmentPreviewObjectUrl) {
          try {
            URL.revokeObjectURL(segmentPreviewObjectUrl);
          } catch (e) {
            console.error(e);
          }
        }
        segmentPreviewObjectUrl = URL.createObjectURL(blob);
        segmentPreviewAudio.src = segmentPreviewObjectUrl;

        // 设置试听状态
        singlePreviewSegmentId.value = seg.id;
        singlePreviewPlaying.value = true;

        const ws = wavesurferMap[seg.id];

        // 播放结束时清理状态
        segmentPreviewAudio.onended = () => {
          if (singlePreviewSegmentId.value === seg.id) {
            stopPreviewProgressUpdate();
            singlePreviewSegmentId.value = null;
            singlePreviewPlaying.value = false;
            // 播放结束后将进度条移到末尾
            const st2 = getWaveformState(seg.id);
            st2.currentTime = st2.duration || 0;
            // 同步 WaveSurfer 进度指示器到末尾
            if (ws && typeof ws.seekTo === "function") {
              ws.seekTo(1);
            }
          }
        };

        await segmentPreviewAudio.play();
        // 启动平滑进度更新
        updatePreviewProgress(seg.id, segmentPreviewAudio, false);
      } catch (e) {
        console.error(e);
        singlePreviewSegmentId.value = null;
        singlePreviewPlaying.value = false;
        showToast("预览音频失败: " + e.message);
      }
    }

    async function previewSegmentEmotionAudio(seg) {
      if (!seg || seg.emotion_mode !== "audio" || !seg.emotion_audio) {
        showToast("当前分段未绑定情感参考音频");
        return;
      }

      // 如有正在进行的顺序试听（逐个或连续），先停止
      if (sequentialPreviewRunning.value) {
        stopSequentialPreviewSilent();
      }

      try {
        if (!segmentPreviewAudio) {
          segmentPreviewAudio = new Audio();
        } else {
          try {
            segmentPreviewAudio.pause();
            segmentPreviewAudio.onended = null;
          } catch (e) {
            console.error(e);
          }
        }

        let res = null;

        // 优先使用分段专属情感音频接口（支持“只上传到该分段”的场景），加时间戳避免缓存
        try {
          if (currentProject.value && seg.id) {
            const ts = Date.now();
            res = await fetch(
              `/api/v1/projects/${currentProject.value.id}/segments/${seg.id}/emotion-audio?t=${ts}`,
            );
            if (!res.ok) {
              res = null;
            }
          }
        } catch (e) {
          console.error(e);
          res = null;
        }

        // 如果项目内情感音频不可用，再尝试从音色库匹配
        if (!res) {
          try {
            if (!voices.value || !voices.value.length) {
              await loadVoices();
            }
          } catch (e) {
            console.error(e);
          }

          const voice = findVoiceByAudioPath(seg.emotion_audio);
          if (!voice) {
            showToast("找不到对应的情感参考音频");
            return;
          }

          const ts = Date.now();
          res = await fetch(`/api/v1/voices/${voice.id}/audio?t=${ts}`);
        }

        if (!res || !res.ok) {
          let detail = "";
          if (res) {
            try {
              const cloned = res.clone();
              try {
                const data = await cloned.json();
                detail = data.detail || JSON.stringify(data);
              } catch {
                detail = await res.text();
              }
            } catch {
              // 响应体不可读时忽略
            }
          }
          throw new Error(detail || `请求失败 (${res ? res.status : "无响应"})`);
        }

        const blob = await res.blob();
        if (segmentPreviewObjectUrl) {
          try {
            URL.revokeObjectURL(segmentPreviewObjectUrl);
          } catch (e) {
            console.error(e);
          }
        }
        segmentPreviewObjectUrl = URL.createObjectURL(blob);
        segmentPreviewAudio.src = segmentPreviewObjectUrl;
        await segmentPreviewAudio.play();
      } catch (e) {
        console.error(e);
        showToast("试听情感参考音频失败: " + e.message);
      }
    }

    const sequentialPreviewRunning = ref(false);
    const sequentialPreviewStopRequested = ref(false);
    const sequentialPreviewStartSegmentId = ref(null);
    const sequentialPreviewCurrentSegmentId = ref(null);  // 当前正在播放的分段 ID

    function stopSequentialPreviewSilent() {
      if (!sequentialPreviewRunning.value && !sequentialPreviewStopRequested.value) return;
      sequentialPreviewStopRequested.value = true;
      stopPreviewProgressUpdate();
      try {
        if (segmentPreviewAudio) {
          segmentPreviewAudio.pause();
          segmentPreviewAudio.onended = null;
        }
      } catch (e) {
        console.error(e);
      }
      sequentialPreviewRunning.value = false;
      sequentialPreviewStartSegmentId.value = null;
      sequentialPreviewCurrentSegmentId.value = null;
    }

    async function sequentialPreviewAllSegments() {
      if (!currentProject.value || !segments.value.length) return;

      // 如果当前就是“逐个试听所有分段”，再次点击视为停止
      if (sequentialPreviewRunning.value && sequentialPreviewStartSegmentId.value === null) {
        stopSequentialPreviewSilent();
        showToast("已停止逐个试听所有分段");
        return;
      }

      // 开始播放前，先停止所有其他播放
      stopAllPlayback();

      sequentialPreviewRunning.value = true;
      sequentialPreviewStopRequested.value = false;
      sequentialPreviewStartSegmentId.value = null;

      // 只播放已经有 audio_file 的段落
      const playable = segments.value.filter((s) => !!s.audio_file);
      if (!playable.length) {
        sequentialPreviewRunning.value = false;
        sequentialPreviewStartSegmentId.value = null;
        showToast("当前没有可播放的分段音频");
        return;
      }

      // 确保有独立的预览 Audio 对象
      if (!segmentPreviewAudio) {
        segmentPreviewAudio = new Audio();
      } else {
        try {
          segmentPreviewAudio.pause();
        } catch (e) {
          console.error(e);
        }
      }

      let index = 0;

      const playNext = async () => {
        if (sequentialPreviewStopRequested.value) {
          sequentialPreviewRunning.value = false;
          sequentialPreviewStopRequested.value = false;
          sequentialPreviewStartSegmentId.value = null;
          sequentialPreviewCurrentSegmentId.value = null;
          return;
        }
        if (index >= playable.length) {
          sequentialPreviewRunning.value = false;
          sequentialPreviewStopRequested.value = false;
          sequentialPreviewStartSegmentId.value = null;
          sequentialPreviewCurrentSegmentId.value = null;
          showToast("已播放完所有可用分段");
          return;
        }

        const seg = playable[index];
        index += 1;
        sequentialPreviewCurrentSegmentId.value = seg.id;

        const payload = buildPreviewPayloadForSegment(seg);
        if (!payload) {
          await playNext();
          return;
        }

        try {
          const res = await fetch(
            `/api/v1/projects/${currentProject.value.id}/segments/${seg.id}/preview-audio`,
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify(payload),
            },
          );

          if (!res.ok) {
            let detail = "";
            try {
              const cloned = res.clone();
              try {
                const data = await cloned.json();
                detail = data.detail || JSON.stringify(data);
              } catch {
                detail = await res.text();
              }
            } catch {
              // 响应体不可读时忽略
            }
            console.error("逐个播放预览失败:", detail || `请求失败 (${res.status})`);
            await playNext();
            return;
          }

          const blob = await res.blob();
          if (segmentPreviewObjectUrl) {
            try {
              URL.revokeObjectURL(segmentPreviewObjectUrl);
            } catch (e) {
              console.error(e);
            }
          }
          segmentPreviewObjectUrl = URL.createObjectURL(blob);
          segmentPreviewAudio.src = segmentPreviewObjectUrl;

          // 同步进度到当前播放段落的波形图（包括进度指示器）
          const st = getWaveformState(seg.id);
          const ws = wavesurferMap[seg.id];

          // 播放当前分段，结束后自动播下一个
          await segmentPreviewAudio.play();
          // 启动平滑进度更新
          updatePreviewProgress(seg.id, segmentPreviewAudio, true);

          segmentPreviewAudio.onended = () => {
            // 停止进度更新
            stopPreviewProgressUpdate();
            // 播放结束后将进度条移到末尾
            st.currentTime = st.duration || 0;
            if (ws && typeof ws.seekTo === "function") {
              ws.seekTo(1);
            }
            // 使用微任务避免递归过深
            Promise.resolve().then(() => {
              playNext();
            });
          };
        } catch (e) {
          console.error(e);
          await playNext();
        }
      };

      showToast("开始逐个试听所有分段");
      await playNext();
    }

    async function sequentialPreviewFromSegment(seg) {
      if (!currentProject.value || !segments.value.length || !seg) return;

      // 如果当前正从这一分段开始连续试听，再次点击视为停止
      if (sequentialPreviewRunning.value && sequentialPreviewStartSegmentId.value === seg.id) {
        stopSequentialPreviewSilent();
        showToast("已停止连续试听");
        return;
      }

      // 开始播放前，先停止所有其他播放
      stopAllPlayback();

      const all = segments.value;
      const startIndex = all.findIndex((s) => s.id === seg.id);
      if (startIndex === -1) return;

      const playable = all.slice(startIndex).filter((s) => !!s.audio_file);
      if (!playable.length) {
        showToast("从当前分段开始没有可播放的分段音频");
        return;
      }

      sequentialPreviewRunning.value = true;
      sequentialPreviewStopRequested.value = false;
      sequentialPreviewStartSegmentId.value = seg.id;

      // 确保有独立的预览 Audio 对象
      if (!segmentPreviewAudio) {
        segmentPreviewAudio = new Audio();
      } else {
        try {
          segmentPreviewAudio.pause();
        } catch (e) {
          console.error(e);
        }
      }

      let index = 0;

      const playNext = async () => {
        if (sequentialPreviewStopRequested.value) {
          sequentialPreviewRunning.value = false;
          sequentialPreviewStopRequested.value = false;
          sequentialPreviewStartSegmentId.value = null;
          sequentialPreviewCurrentSegmentId.value = null;
          return;
        }
        if (index >= playable.length) {
          sequentialPreviewRunning.value = false;
          sequentialPreviewStopRequested.value = false;
          sequentialPreviewStartSegmentId.value = null;
          sequentialPreviewCurrentSegmentId.value = null;
          showToast("已播放完所有可用分段");
          return;
        }

        const s = playable[index];
        index += 1;
        sequentialPreviewCurrentSegmentId.value = s.id;

        const payload = buildPreviewPayloadForSegment(s);
        if (!payload) {
          await playNext();
          return;
        }

        try {
          const res = await fetch(
            `/api/v1/projects/${currentProject.value.id}/segments/${s.id}/preview-audio`,
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify(payload),
            },
          );

          if (!res.ok) {
            let detail = "";
            try {
              const cloned = res.clone();
              try {
                const data = await cloned.json();
                detail = data.detail || JSON.stringify(data);
              } catch {
                detail = await res.text();
              }
            } catch {
              // 响应体不可读时忽略
            }
            console.error("逐个播放预览失败:", detail || `请求失败 (${res.status})`);
            await playNext();
            return;
          }

          const blob = await res.blob();
          if (segmentPreviewObjectUrl) {
            try {
              URL.revokeObjectURL(segmentPreviewObjectUrl);
            } catch (e) {
              console.error(e);
            }
          }
          segmentPreviewObjectUrl = URL.createObjectURL(blob);
          segmentPreviewAudio.src = segmentPreviewObjectUrl;

          // 同步进度到当前播放段落的波形图（包括进度指示器）
          const st = getWaveformState(s.id);
          const ws = wavesurferMap[s.id];

          // 播放当前分段，结束后自动播下一个
          await segmentPreviewAudio.play();
          // 启动平滑进度更新
          updatePreviewProgress(s.id, segmentPreviewAudio, true);

          segmentPreviewAudio.onended = () => {
            // 停止进度更新
            stopPreviewProgressUpdate();
            // 播放结束后将进度条移到末尾
            st.currentTime = st.duration || 0;
            if (ws && typeof ws.seekTo === "function") {
              ws.seekTo(1);
            }
            // 使用微任务避免递归过深
            Promise.resolve().then(() => {
              playNext();
            });
          };
        } catch (e) {
          console.error(e);
          await playNext();
        }
      };

      showToast("开始从当前分段连续试听");
      await playNext();
    }

    function showToast(msg) {
      toast.value = msg;
      setTimeout(() => {
        if (toast.value === msg) toast.value = "";
      }, 3000);
    }

    const EMOTION_ERROR_HINT = "提示：TTS 在处理该段情感设置时出现问题，请检查情感文本/向量/参考音频后重试。";

    function normalizeErrorMessage(err) {
      if (!err) return "";
      if (typeof err === "string") return err;
      if (typeof err === "object") {
        if (err.error) return String(err.error);
        if (err.detail) return String(err.detail);
        if (err.message) return String(err.message);
      }
      try {
        return JSON.stringify(err);
      } catch (e) {
        return String(err);
      }
    }

    function isEmotionRelatedError(message) {
      if (!message) return false;
      const text = String(message);
      return /情感|emotion|emo_|emotion_mode|emotion_audio/i.test(text);
    }

    function withEmotionHint(message) {
      if (!message) return EMOTION_ERROR_HINT;
      if (message.includes(EMOTION_ERROR_HINT)) return message;
      return `${message}。${EMOTION_ERROR_HINT}`;
    }

    async function fetchJson(path, options = {}) {
      const res = await fetch(apiBase + path, {
        headers: {
          "Content-Type": "application/json",
          ...(options.headers || {}),
        },
        ...options,
      });

      if (!res.ok) {
        let detail = "";
        try {
          const cloned = res.clone();
          try {
            const data = await cloned.json();
            // 处理结构化错误信息（如 {detail: {message: "xxx", ...}}）
            if (data.detail) {
              if (typeof data.detail === "string") {
                detail = data.detail;
              } else if (typeof data.detail === "object" && data.detail.message) {
                detail = data.detail.message;
              } else {
                detail = JSON.stringify(data.detail);
              }
            } else {
              detail = JSON.stringify(data);
            }
          } catch {
            detail = await res.text();
          }
        } catch {
          // 响应体不可读时忽略
        }
        throw new Error(`请求失败 (${res.status}): ${detail}`);
      }

      if (res.status === 204) {
        // 无内容响应（如 DELETE 成功），不再尝试解析 JSON
        return null;
      }

      return res.json();
    }

    async function refreshHealth(options = {}) {
      const force = options.force !== false;
      const minIntervalMs = typeof options.minIntervalMs === "number" ? options.minIntervalMs : 0;
      const now = Date.now();
      if (!force && health.value && now - lastHealthRefreshAt < minIntervalMs) {
        return health.value;
      }
      if (healthRefreshPromise) {
        return healthRefreshPromise;
      }

      healthRefreshPromise = (async () => {
        try {
          const data = await fetchJson("/api/v1/health", { headers: { "Content-Type": undefined } });
          health.value = data;
          lastHealthRefreshAt = Date.now();
          return data;
        } catch (e) {
          console.error(e);
          return health.value;
        } finally {
          healthRefreshPromise = null;
        }
      })();

      try {
        return await healthRefreshPromise;
      } finally {
      }
    }

    const roleColorPalette = [
      {
        id: "emerald",
        badge: "bg-emerald-900/70 text-emerald-200 border border-emerald-500/70",
        row: "bg-emerald-500/5",
        text: "text-emerald-200",
      },
      {
        id: "sky",
        badge: "bg-sky-900/70 text-sky-200 border border-sky-500/70",
        row: "bg-sky-500/5",
        text: "text-sky-200",
      },
      {
        id: "amber",
        badge: "bg-amber-900/70 text-amber-100 border border-amber-500/70",
        row: "bg-amber-500/10",
        text: "text-amber-100",
      },
      {
        id: "violet",
        badge: "bg-violet-900/70 text-violet-200 border border-violet-500/70",
        row: "bg-violet-500/5",
        text: "text-violet-200",
      },
      {
        id: "rose",
        badge: "bg-rose-900/70 text-rose-200 border border-rose-500/70",
        row: "bg-rose-500/5",
        text: "text-rose-200",
      },
      {
        id: "cyan",
        badge: "bg-cyan-900/70 text-cyan-200 border border-cyan-500/70",
        row: "bg-cyan-500/5",
        text: "text-cyan-200",
      },
      {
        id: "lime",
        badge: "bg-lime-900/70 text-lime-200 border border-lime-500/70",
        row: "bg-lime-500/5",
        text: "text-lime-200",
      },
    ];

    function hashRoleId(roleId) {
      let hash = 0;
      const s = String(roleId || "");
      for (let i = 0; i < s.length; i++) {
        hash = (hash * 31 + s.charCodeAt(i)) >>> 0;
      }
      return hash >>> 0;
    }

    function getRoleColor(roleId) {
      if (!roleId) return null;
      if (!roleColorPalette.length) {
        return {
          badge: "bg-slate-800 text-slate-100 border border-slate-600",
          row: "",
          text: "text-slate-100",
        };
      }

      const hash = hashRoleId(roleId);
      const idx = hash % roleColorPalette.length;
      const tpl = roleColorPalette[idx];

      // 为了保持与原有配色体系的风格大致一致，这里仍然复用原来的 tailwind 类，
      // 只是通过 idx 在多角色之间打散不同色系。
      return tpl;
    }

    function roleBadgeClass(roleId) {
      const color = getRoleColor(roleId);
      return color ? color.badge : "bg-slate-800 text-slate-100 border border-slate-600";
    }

    function segmentRowClass(seg) {
      const color = getRoleColor(seg.role_id);
      // 上半行：作为卡片头部，仅使用统一背景和 hover，不再使用下边框
      const base =
        "bg-slate-950/70 hover:bg-slate-900/90 border-t border-slate-900/80 transition-colors";
      return color && color.row ? `${base} ${color.row}` : base;
    }

    function segmentWaveformRowClass(seg) {
      const color = getRoleColor(seg.role_id);
      // 下半行：作为卡片底部，继承相同背景，并加一条更深的底部边框
      const base =
        "bg-slate-950/70 hover:bg-slate-900/90 border-b border-slate-800/80 shadow-sm transition-colors";
      return color && color.row ? `${base} ${color.row}` : base;
    }

    function segmentTextClass(seg) {
      if (!seg) return "";
      const color = getRoleColor(seg.role_id);
      if (color && color.text) return color.text;
      // 默认与原来的文本颜色保持一致
      return "text-slate-100";
    }

    function segmentEmotionTextClass(seg) {
      if (!seg) return "";
      const color = getRoleColor(seg.role_id);
      if (color && color.text) return color.text;
      // 默认与原来的情感描述颜色保持一致
      return "text-slate-300";
    }

    function segmentStatusBadgeClass(seg) {
      if (!seg || !seg.status) {
        return "bg-slate-800 text-slate-300";
      }
      // 情感相关“降级成功”：status 为 done，但存在情感解析 warning，用黄色提示
      if (seg.status === "done" && seg._emotionWarning) {
        return "bg-amber-700/80 text-amber-100";
      }
      if (seg.status === "pending") {
        return "bg-slate-800 text-slate-300";
      }
      if (seg.status === "generating") {
        return "bg-sky-800/70 text-sky-200";
      }
      if (seg.status === "done") {
        return "bg-emerald-800/70 text-emerald-200";
      }
      if (seg.status === "failed") {
        return "bg-rose-900/70 text-rose-200";
      }
      return "bg-slate-800 text-slate-300";
    }

    function segmentReviewLabel(status) {
      const labels = {
        draft: "未生成",
        to_review: "待审听",
        needs_revision: "需修改",
        approved: "已通过",
      };
      return labels[status] || "待审听";
    }

    async function onSegmentReviewStatusChange(seg, nextStatus) {
      if (!seg || !currentProject.value || !nextStatus) return;
      try {
        let updated;
        if (nextStatus === "needs_revision") {
          const note = prompt("请简要记录问题（发音、情绪、节奏、文本或音质等）", seg.review_note || "");
          if (note === null) return;
          updated = await fetchJson(
            `/api/v1/projects/${currentProject.value.id}/segments/${seg.id}/review-issues`,
            {
              method: "POST",
              body: JSON.stringify({ category: "other", note: note || null }),
            },
          );
        } else {
          updated = await fetchJson(
            `/api/v1/projects/${currentProject.value.id}/segments/${seg.id}/review`,
            {
              method: "PATCH",
              body: JSON.stringify({ review_status: nextStatus, note: seg.review_note || null }),
            },
          );
        }
        const idx = segments.value.findIndex((item) => item.id === updated.id);
        if (idx !== -1) Object.assign(segments.value[idx], updated);
        showToast(`审听状态已更新为：${segmentReviewLabel(updated.review_status)}`);
      } catch (e) {
        showToast("更新审听状态失败: " + e.message);
      }
    }

    async function flagSegmentIssue(seg) {
      if (!seg || !currentProject.value) return;
      const category = prompt("问题类型：pronunciation / emotion / pacing / text / audio / other", "pronunciation");
      if (category === null) return;
      const normalized = String(category).trim().toLowerCase();
      if (!["pronunciation", "emotion", "pacing", "text", "audio", "other"].includes(normalized)) {
        showToast("问题类型无效");
        return;
      }
      const note = prompt("问题说明", "");
      if (note === null) return;
      try {
        const updated = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/segments/${seg.id}/review-issues`,
          { method: "POST", body: JSON.stringify({ category: normalized, note: note || null }) },
        );
        Object.assign(seg, updated);
        showToast("审听问题已记录");
      } catch (e) {
        showToast("记录问题失败: " + e.message);
      }
    }

    async function resolveAllReviewIssues(seg) {
      const unresolved = (seg.review_issues || []).filter((item) => !item.resolved);
      if (!currentProject.value || !unresolved.length) return;
      try {
        let updated = seg;
        for (const issue of unresolved) {
          updated = await fetchJson(
            `/api/v1/projects/${currentProject.value.id}/segments/${seg.id}/review-issues/${issue.id}`,
            { method: "PATCH", body: JSON.stringify({ resolved: true }) },
          );
        }
        Object.assign(seg, updated);
        showToast("问题已标记解决，请重新审听后通过");
      } catch (e) {
        showToast("更新问题失败: " + e.message);
      }
    }

    async function batchApproveReviewable() {
      if (!currentProject.value) return;
      const ids = segments.value
        .filter((seg) => seg.status === "done" && seg.review_status !== "approved" && !(seg.review_issues || []).some((item) => !item.resolved) && getCurrentTakeId(seg))
        .map((seg) => seg.id);
      if (!ids.length) {
        showToast("没有可批量通过的分段");
        return;
      }
      try {
        segments.value = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/segments/review/batch-approve`,
          { method: "POST", body: JSON.stringify({ segment_ids: ids }) },
        );
        showToast(`已批量通过 ${ids.length} 个分段`);
      } catch (e) {
        showToast("批量通过失败: " + e.message);
      }
    }

    function jumpToNextReview() {
      const target = segments.value.find((seg) => seg.review_status === "to_review" || seg.review_status === "needs_revision");
      if (!target) {
        showToast("没有待审听或需修改的分段");
        return;
      }
      const row = document.getElementById(`segment-row-${target.id}`);
      if (row) row.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    function setEmotionWarning(segId, hint) {
      if (!segId) return;
      const map = emotionWarnings.value;
      if (hint && typeof hint === "string") {
        map[segId] = hint;
      }
    }

    function applyEmotionWarningsToSegments() {
      if (!Array.isArray(segments.value)) return;
      const map = emotionWarnings.value || {};
      segments.value.forEach((seg) => {
        if (!seg || !seg.id) return;
        const hint = map[seg.id];
        if (hint) {
          seg._emotionWarning = true;
          seg._lastErrorHint = hint;
        } else if (seg._emotionWarning) {
          seg._emotionWarning = false;
        }
      });
    }

    function loadColumnWidthsFromStorage() {
      try {
        const raw = window.localStorage.getItem("tts_column_widths");
        if (!raw) return;
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object") return;
        for (const table of ["segments", "final"]) {
          if (parsed[table] && typeof parsed[table] === "object") {
            columnWidths.value[table] = {
              ...columnWidths.value[table],
              ...parsed[table],
            };
          }
        }
      } catch (e) {
        console.error(e);
      }
    }

    function persistColumnWidths() {
      try {
        window.localStorage.setItem("tts_column_widths", JSON.stringify(columnWidths.value));
      } catch (e) {
        console.error(e);
      }
    }

    function startColumnResize(table, key, event) {
      const tableMap = columnWidths.value[table];
      if (!tableMap) return;
      const thEl = event.currentTarget && event.currentTarget.parentElement;
      if (!thEl) return;
      const startX = event.clientX;
      const startWidth = thEl.offsetWidth || 0;
      const minWidth = 60;

      function onMouseMove(e) {
        const delta = e.clientX - startX;
        const next = Math.max(minWidth, startWidth + delta);
        tableMap[key] = `${next}px`;
      }

      function onMouseUp() {
        window.removeEventListener("mousemove", onMouseMove);
        window.removeEventListener("mouseup", onMouseUp);
        persistColumnWidths();
      }

      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", onMouseUp);
    }

    function startEditRole(role) {
      if (!currentProject.value || !role) return;
      editingRoleId.value = role.id;
      forms.value.role = createRoleFormState(role, getQwen3DefaultLanguage());
      modals.value.createRole = true;
    }

    async function deleteRole(roleId) {
      if (!currentProject.value || !roleId) return;
      if (!ensureProjectMutationAllowed("删除角色")) return;
      const target = roles.value.find((r) => r.id === roleId);
      const name = target && target.name ? target.name : roleId;
      if (
        !window.confirm(
          `将删除角色「${name}」，并清空所有分段中对该角色的绑定（role_id 置为空），此操作不可恢复。确认删除？`,
        )
      ) {
        return;
      }

      try {
        await fetchJson(`/api/v1/projects/${currentProject.value.id}/roles/${roleId}`, {
          method: "DELETE",
          body: JSON.stringify({}),
        });
        await loadRolesAndSegments();
        if (editingRoleId.value === roleId) {
          editingRoleId.value = null;
          forms.value.role = createRoleFormState(null, getQwen3DefaultLanguage());
        }
        showToast("角色已删除");
      } catch (e) {
        console.error(e);
        showToast("角色删除失败: " + e.message);
      }
    }

    async function loadLlmProviders() {
      try {
        const data = await fetchJson("/api/v1/llm-providers", { headers: { "Content-Type": undefined } });
        llmProviders.value = data;
        if (!selectedLlmProviderId.value && Array.isArray(data) && data.length) {
          const defaultProvider = data.find((p) => p.is_default) || data[0];
          selectedLlmProviderId.value = defaultProvider.id;
        } else if (selectedLlmProviderId.value) {
          const stillExists = data.some((p) => p.id === selectedLlmProviderId.value);
          if (!stillExists) {
            const fallback = data.find((p) => p.is_default) || data[0] || null;
            selectedLlmProviderId.value = fallback ? fallback.id : null;
          }
        }
      } catch (e) {
        console.error(e);
      }
    }

    async function loadPlatformSettings() {
      platformLoading.value.load = true;
      try {
        const data = await fetchJson("/api/v1/settings/platform", { headers: { "Content-Type": undefined } });
        platformSettings.value = {
          tts_service_name: data.tts_service_name || "",
          tts_base_url: data.tts_base_url || "",
          cosyvoice_base_url: data.cosyvoice_base_url || "",
          qwen3tts_base_url: data.qwen3tts_base_url || "",
          qwen3tts_cv_base_url: data.qwen3tts_cv_base_url || "",
          tts_api_key: data.tts_api_key || "",
          llm_service_name: data.llm_service_name || "",
          llm_base_url: data.llm_base_url || "",
          llm_api_key: data.llm_api_key || "",
        };
      } catch (e) {
        console.error(e);
        showToast("加载平台配置失败");
      } finally {
        platformLoading.value.load = false;
      }
    }

    async function savePlatformSettings() {
      platformLoading.value.save = true;
      try {
        const body = JSON.stringify(platformSettings.value);
        const data = await fetchJson("/api/v1/settings/platform", { method: "PUT", body });
        platformSettings.value = {
          tts_service_name: data.tts_service_name || "",
          tts_base_url: data.tts_base_url || "",
          cosyvoice_base_url: data.cosyvoice_base_url || "",
          qwen3tts_base_url: data.qwen3tts_base_url || "",
          qwen3tts_cv_base_url: data.qwen3tts_cv_base_url || "",
          tts_api_key: data.tts_api_key || "",
          llm_service_name: data.llm_service_name || "",
          llm_base_url: data.llm_base_url || "",
          llm_api_key: data.llm_api_key || "",
        };
        showToast("平台配置已保存");
        await loadLlmConfigs();
        modals.value.createLlmConfig = false;
      } catch (e) {
        console.error(e);
        showToast("保存平台配置失败: " + e.message);
      } finally {
        platformLoading.value.save = false;
      }
    }

    async function loadTtsParams() {
      platformLoading.value.ttsParams = true;
      try {
        const data = await fetchJson("/api/v1/settings/tts-params", { headers: { "Content-Type": undefined } });
        ttsParams.value = normalizeTtsParamsResponse(data, ttsParams.value, false);
      } catch (e) {
        // 静默失败，使用默认值（可能是服务未重启）
        console.error("加载 TTS 参数失败:", e);
      } finally {
        platformLoading.value.ttsParams = false;
      }
    }

    async function loadTtsBackend() {
      try {
        const data = await fetchJson("/api/v1/settings/tts-backend", { headers: { "Content-Type": undefined } });
        ttsBackend.value = {
          current: data.backend || "indextts",
          available: data.available_backends || [],
        };
      } catch (e) {
        console.error("加载 TTS 后端配置失败:", e);
      }
    }

    async function switchTtsBackend(backendId) {
      if (!backendId || (ttsBackend.value && ttsBackend.value.current === backendId)) {
        return;
      }
      try {
        const data = await fetchJson("/api/v1/settings/tts-backend", {
          method: "PUT",
          body: JSON.stringify({ backend: backendId }),
        });
        ttsBackend.value = {
          current: data.backend || backendId,
          available: data.available_backends || ttsBackend.value.available || [],
        };
        const backendNames = {
          indextts: "IndexTTS2",
          cosyvoice3: "CosyVoice3",
          qwen3tts: "Qwen3-TTS VoiceDesign",
          qwen3tts_cv: "Qwen3-TTS CustomVoice",
        };
        ttsService.value.configBackend = data.backend || backendId;
        if (ttsService.value.running && ttsService.value.backend && ttsService.value.backend !== backendId) {
          showToast(`已将默认后端切换为 ${backendNames[backendId] || backendId}，重启 TTS 服务后生效`);
        } else {
          showToast(`已将默认后端设置为 ${backendNames[backendId] || backendId}`);
        }
      } catch (e) {
        console.error("切换 TTS 后端失败:", e);
        showToast("切换 TTS 后端失败: " + e.message);
      }
    }

    // 判断当前是否为 CosyVoice3 后端（全局）
    const isCosyVoice3 = computed(() => ttsBackend.value && ttsBackend.value.current === "cosyvoice3");

    function getSegmentBackend(seg) {
      const validBackends = new Set(["indextts", "cosyvoice3", "qwen3tts", "qwen3tts_cv"]);
      const segmentBackend = seg && typeof seg.tts_backend === "string" ? seg.tts_backend.trim() : "";
      if (validBackends.has(segmentBackend)) {
        return segmentBackend;
      }
      const role = seg && seg.role_id
        ? (roles.value || []).find((item) => item && item.id === seg.role_id)
        : null;
      const roleBackend = role && typeof role.tts_backend === "string" ? role.tts_backend.trim() : "";
      if (validBackends.has(roleBackend)) {
        return roleBackend;
      }
      const globalBackend = (ttsBackend.value && ttsBackend.value.current) || "indextts";
      return validBackends.has(globalBackend) ? globalBackend : "indextts";
    }

    // 判断分段是否使用 CosyVoice3
    function isSegmentCosyVoice3(seg) {
      if (!seg) return false;
      return getSegmentBackend(seg) === "cosyvoice3";
    }

    // 判断分段是否使用 Qwen3-TTS VoiceDesign
    function isSegmentQwen3TTS(seg) {
      if (!seg) return false;
      return getSegmentBackend(seg) === "qwen3tts";
    }

    // 判断分段是否使用 Qwen3-TTS CustomVoice
    function isSegmentQwen3TTSCustomVoice(seg) {
      if (!seg) return false;
      return getSegmentBackend(seg) === "qwen3tts_cv";
    }

    // 判断分段是否使用任一 Qwen3-TTS 后端
    function isSegmentQwen3TTSAny(seg) {
      if (!seg) return false;
      const b = getSegmentBackend(seg);
      return b === 'qwen3tts' || b === 'qwen3tts_cv';
    }

    // CustomVoice 内置音色列表 (官方支持: aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian)
    const qwen3CustomVoiceSpeakers = ref([
      { id: 'vivian', name: 'Vivian', gender: 'female', language: 'Chinese', description: '明亮、略带锐气的年轻女声' },
      { id: 'serena', name: 'Serena', gender: 'female', language: 'Chinese', description: '温暖柔和的年轻女声' },
      { id: 'uncle_fu', name: 'Uncle Fu', gender: 'male', language: 'Chinese', description: '音色低沉醇厚的成熟男声' },
      { id: 'dylan', name: 'Dylan', gender: 'male', language: 'Chinese', description: '清晰自然的北京青年男声（北京方言）' },
      { id: 'eric', name: 'Eric', gender: 'male', language: 'Chinese', description: '活泼、略带沙哑的成都男声（四川方言）' },
      { id: 'ryan', name: 'Ryan', gender: 'male', language: 'English', description: '富有节奏感的动态男声' },
      { id: 'aiden', name: 'Aiden', gender: 'male', language: 'English', description: '清晰中频的阳光美式男声' },
      { id: 'ono_anna', name: 'Ono Anna', gender: 'female', language: 'Japanese', description: '轻快灵活的俏皮日文女声' },
      { id: 'sohee', name: 'Sohee', gender: 'female', language: 'Korean', description: '富含情感的温暖韩文女声' },
    ]);

    // Qwen3-TTS 语言选项 (支持10种语言)
    const qwen3Languages = ref([
      { id: 'Chinese', name: '中文' },
      { id: 'English', name: '英文' },
      { id: 'Japanese', name: '日文' },
      { id: 'Korean', name: '韩文' },
      { id: 'German', name: '德文' },
      { id: 'French', name: '法文' },
      { id: 'Russian', name: '俄文' },
      { id: 'Portuguese', name: '葡萄牙文' },
      { id: 'Spanish', name: '西班牙文' },
      { id: 'Italian', name: '意大利文' },
      { id: 'Auto', name: '自动' },
    ]);

    // 根据音色自动获取对应语言
    function getLanguageForSpeaker(speakerId) {
      const speaker = qwen3CustomVoiceSpeakers.value.find(s => s.id === speakerId);
      return speaker ? speaker.language : 'Chinese';
    }

    // TTS 服务状态
    const ttsService = ref({
      running: false,
      backend: null,
      configBackend: null,
      pid: null,
      logs: [],
      lastError: null,
      loading: false,
    });

    function getTtsBackendName(backendId) {
      const backendNames = {
        indextts: "IndexTTS2",
        cosyvoice3: "CosyVoice3",
        qwen3tts: "Qwen3 VoiceDesign",
        qwen3tts_cv: "Qwen3 CustomVoice",
      };
      return backendNames[backendId] || backendId || "IndexTTS2";
    }

    const displayedTtsBackend = computed(() => {
      if (ttsService.value.running && ttsService.value.backend) {
        return ttsService.value.backend;
      }
      return ttsService.value.configBackend || (ttsBackend.value && ttsBackend.value.current) || "indextts";
    });

    const ttsServiceNeedsRestart = computed(() => {
      const configured = ttsService.value.configBackend || (ttsBackend.value && ttsBackend.value.current);
      return !!(
        ttsService.value.running
        && configured
        && ttsService.value.backend
        && configured !== ttsService.value.backend
      );
    });

    function isTtsServiceErrorLog(line) {
      if (!line) return false;
      return /error|exception|traceback|fatal|fail|failed|错误|异常|失败/i.test(String(line));
    }

    async function loadTtsServiceStatus() {
      try {
        const data = await fetchJson("/api/v1/settings/tts-service/status", { headers: { "Content-Type": undefined } });
        const runtimeBackend = data.runtime_backend || data.backend || null;
        const configBackend = data.config_backend || (ttsBackend.value && ttsBackend.value.current) || "indextts";
        ttsService.value.running = data.running;
        ttsService.value.backend = runtimeBackend;
        ttsService.value.configBackend = configBackend;
        ttsService.value.pid = data.pid;
        if (ttsBackend.value) {
          ttsBackend.value.current = configBackend;
        }
        const newLogs = Array.isArray(data.logs) ? data.logs : [];
        if (newLogs.length > 0) {
          ttsService.value.logs = [...ttsService.value.logs, ...newLogs].slice(-100);
          const lastError = [...newLogs].reverse().find(isTtsServiceErrorLog);
          if (lastError) {
            ttsService.value.lastError = String(lastError).trim();
          }
        }
      } catch (e) {
        console.error("获取 TTS 服务状态失败:", e);
        ttsService.value.lastError = `获取 TTS 服务状态失败: ${e.message || e}`;
      }
    }

    async function startTtsService(backend, options = {}) {
      const persistDefault = options.persistDefault !== false;
      ttsService.value.loading = true;
      try {
        boostTtsServicePolling(60000);
        const data = await fetchJson("/api/v1/settings/tts-service/start", {
          method: "POST",
          body: JSON.stringify({
            backend: backend || ttsBackend.value.current || "indextts",
            persist_default: persistDefault,
          }),
        });
        const runtimeBackend = data.runtime_backend || data.backend || backend;
        const configBackend =
          data.config_backend
          || ttsService.value.configBackend
          || (ttsBackend.value && ttsBackend.value.current)
          || "indextts";
        ttsService.value.running = data.running;
        ttsService.value.backend = runtimeBackend;
        ttsService.value.configBackend = configBackend;
        ttsService.value.pid = data.pid;
        ttsService.value.logs = [];
        ttsService.value.lastError = null;
        const backendNames = { indextts: 'IndexTTS2', cosyvoice3: 'CosyVoice3', qwen3tts: 'Qwen3-TTS VoiceDesign', qwen3tts_cv: 'Qwen3-TTS CustomVoice' };
        showToast(`${backendNames[runtimeBackend] || runtimeBackend} 服务启动中...`);
        // 更新后端配置
        ttsBackend.value.current = configBackend;
        // 延迟刷新健康状态
        setTimeout(() => refreshHealth(), 3000);
      } catch (e) {
        console.error("启动 TTS 服务失败:", e);
        showToast("启动 TTS 服务失败: " + e.message);
      } finally {
        ttsService.value.loading = false;
      }
    }

    async function stopTtsService() {
      ttsService.value.loading = true;
      try {
        boostTtsServicePolling(30000);
        const data = await fetchJson("/api/v1/settings/tts-service/stop", { method: "POST" });
        const configBackend = data.config_backend || (ttsBackend.value && ttsBackend.value.current) || "indextts";
        ttsService.value.running = data.running;
        ttsService.value.backend = data.runtime_backend || data.backend || null;
        ttsService.value.configBackend = configBackend;
        ttsService.value.pid = null;
        ttsService.value.lastError = null;
        if (ttsBackend.value) {
          ttsBackend.value.current = configBackend;
        }
        showToast("TTS 服务已停止");
        // 刷新健康状态
        setTimeout(() => refreshHealth(), 500);
      } catch (e) {
        console.error("停止 TTS 服务失败:", e);
        showToast("停止 TTS 服务失败: " + e.message);
      } finally {
        ttsService.value.loading = false;
      }
    }

    // 等待 TTS 服务就绪（通过健康检查，并确保模型已加载）
    async function waitForTtsServiceReady(backend, maxWaitMs = 15000) {
      const startTime = Date.now();
      
      let modelLoadTriggered = false;
      
      while (Date.now() - startTime < maxWaitMs) {
        try {
          const data = await fetchJson("/api/v1/health", { headers: { "Content-Type": undefined } });
          const runtimeBackend = data.tts_runtime_backend || data.tts_backend || null;
          const backendMatches = !backend || runtimeBackend === backend;
          if (data.tts_health_ok && backendMatches) {
            // 如果响应中没有 model_loaded 字段（如 Qwen3 TTS），
            // 服务能正常响应即视为就绪
            const hasModelLoadedField = ("tts_model_loaded" in data);
            if (!hasModelLoadedField || data.tts_model_loaded === true) {
              // 服务就绪（且模型已加载，或服务不报告此字段）
              ttsService.value.running = true;
              ttsService.value.backend = runtimeBackend || backend;
              ttsService.value.configBackend = data.tts_config_backend || backend;
              // 立即刷新健康指示灯
              refreshHealth().catch(() => {});
              return true;
            } else if (!modelLoadTriggered) {
              // 服务就绪但模型未加载，触发加载
              console.debug(`[TTS] 服务已启动但模型未加载，触发加载...`);
              showToast("正在加载 TTS 模型...");
              modelLoadTriggered = true;
              try {
                await fetchJson("/api/v1/settings/tts-service/load-model", {
                  method: "POST",
                  body: JSON.stringify({ backend }),
                });
              } catch (loadErr) {
                console.error("触发模型加载失败:", loadErr);
              }
            }
            // 模型正在加载中，继续等待
          }
        } catch (e) {
          // 服务尚未就绪，继续等待
        }
        await new Promise(resolve => setTimeout(resolve, 500));
      }
      return false;
    }

    async function saveTtsParams() {
      platformLoading.value.ttsParams = true;
      try {
        const { defaults, ...params } = ttsParams.value;
        const body = JSON.stringify(params);
        const data = await fetchJson("/api/v1/settings/tts-params", { method: "PUT", body });
        ttsParams.value = normalizeTtsParamsResponse(data, ttsParams.value, false);
        showToast("TTS 参数已保存");
      } catch (e) {
        console.error(e);
        showToast("保存 TTS 参数失败: " + e.message);
      } finally {
        platformLoading.value.ttsParams = false;
      }
    }

    async function resetTtsParams() {
      platformLoading.value.ttsParams = true;
      try {
        const data = await fetchJson("/api/v1/settings/tts-params/reset", { method: "POST", body: JSON.stringify({}) });
        ttsParams.value = normalizeTtsParamsResponse(data, ttsParams.value, true);
        showToast("TTS 参数已恢复默认值");
      } catch (e) {
        console.error(e);
        showToast("重置 TTS 参数失败: " + e.message);
      } finally {
        platformLoading.value.ttsParams = false;
      }
    }

    async function loadGlobalSegPromptTemplate() {
      platformLoading.value.globalSegPrompt = true;
      try {
        const data = await fetchJson("/api/v1/settings/seg-prompt-template", { headers: { "Content-Type": undefined } });
        globalSegPromptTemplate.value = {
          template: data.template || "",
          isDefault: data.is_default !== false,
        };
      } catch (e) {
        console.error("加载全局分段模板失败:", e);
      } finally {
        platformLoading.value.globalSegPrompt = false;
      }
    }

    async function saveGlobalSegPromptTemplate() {
      platformLoading.value.globalSegPrompt = true;
      try {
        const body = JSON.stringify({ template: globalSegPromptTemplate.value.template });
        const data = await fetchJson("/api/v1/settings/seg-prompt-template", { method: "PUT", body });
        globalSegPromptTemplate.value = {
          template: data.template || "",
          isDefault: data.is_default !== false,
        };
        showToast("全局分段提示词模板已保存");
      } catch (e) {
        console.error(e);
        showToast("保存全局分段模板失败: " + e.message);
      } finally {
        platformLoading.value.globalSegPrompt = false;
      }
    }

    async function resetGlobalSegPromptTemplate() {
      platformLoading.value.globalSegPrompt = true;
      try {
        const data = await fetchJson("/api/v1/settings/seg-prompt-template/reset", { method: "POST", body: JSON.stringify({}) });
        globalSegPromptTemplate.value = {
          template: data.template || "",
          isDefault: true, // 重置后一定是使用内置默认
        };
        showToast("全局分段提示词模板已恢复为内置默认值（已自动保存）");
      } catch (e) {
        console.error(e);
        showToast("重置全局分段模板失败: " + e.message);
      } finally {
        platformLoading.value.globalSegPrompt = false;
      }
    }

    async function unloadTtsModel() {
      platformLoading.value.unloadTts = true;
      try {
        await fetchJson("/api/v1/settings/gpu/unload-tts", {
          method: "POST",
          body: JSON.stringify({}),
        });
        showToast("已请求从显存中卸载 TTS 模型");
        try {
          await refreshHealth();
        } catch (e) {
          console.error(e);
        }
      } catch (e) {
        console.error(e);
        showToast("卸载 TTS 模型失败: " + e.message);
      } finally {
        platformLoading.value.unloadTts = false;
      }
    }

    async function unloadWhisperModel() {
      platformLoading.value.unloadWhisper = true;
      try {
        await fetchJson("/api/v1/settings/gpu/unload-whisper", {
          method: "POST",
          body: JSON.stringify({}),
        });
        showToast("已请求从显存中卸载 Whisper 模型");
        try {
          await refreshHealth();
        } catch (e) {
          console.error(e);
        }
      } catch (e) {
        console.error(e);
        showToast("卸载 Whisper 模型失败: " + e.message);
      } finally {
        platformLoading.value.unloadWhisper = false;
      }
    }

    async function testLlm() {
      platformLoading.value.test = true;
      llmTestResult.value = null;
      try {
        const data = await fetchJson("/api/v1/settings/test-llm", {
          method: "POST",
          body: JSON.stringify({}),
        });
        llmTestResult.value = data;
        showToast(data.success ? "LLM 测试成功" : "LLM 测试失败");
      } catch (e) {
        console.error(e);
        llmTestResult.value = { success: false, message: e.message };
        showToast("LLM 测试失败: " + e.message);
      } finally {
        platformLoading.value.test = false;
      }
    }

    function normalizeVoiceTags(rawInput, extraList) {
      const fromInput = (rawInput || "")
        .split(/[\s,，；;]+/)
        .map((t) => t.trim())
        .filter((t) => t.length);
      const merged = [...(extraList || []), ...fromInput];
      const set = new Set(merged);
      const unique = Array.from(set);
      return enforceExclusiveTags(unique);
    }

    function enforceExclusiveTags(tags) {
      if (!Array.isArray(tags)) {
        return [];
      }
      const result = [];
      let typeTag = null;
      let genderTag = null;
      let ageTag = null;
      for (const tag of tags) {
        if (isVoiceTypeTag(tag)) {
          typeTag = tag;
        } else if (isGenderTag(tag)) {
          genderTag = tag;
        } else if (isAgeTag(tag)) {
          ageTag = tag;
        } else if (!result.includes(tag)) {
          result.push(tag);
        }
      }
      if (typeTag && !result.includes(typeTag)) {
        result.unshift(typeTag);
      }
      if (genderTag && !result.includes(genderTag)) {
        result.push(genderTag);
      }
      if (ageTag && !result.includes(ageTag)) {
        result.push(ageTag);
      }
      return result;
    }

    // 保证标签输入框与上方标签列表双向同步：
    // - 用户在输入框中输入/删除标签时，实时更新 voiceForm.tags；
    // - 点击预设标签按钮会更新 tags 并回写到 tagsInput。
    Vue.watch(
      () => voiceForm.value.tagsInput,
      (val) => {
        const tags = normalizeVoiceTags(val, []);
        voiceForm.value.tags = tags;
      },
    );

    async function loadVoices() {
      try {
        const data = await fetchJson("/api/v1/voices", { headers: { "Content-Type": undefined } });
        voices.value = Array.isArray(data) ? data : [];
      } catch (e) {
        console.error(e);
        showToast("加载音色库失败: " + e.message);
      }
    }

    function clearVoiceLocalPreview() {
      if (voiceLocalPreviewUrl.value) {
        URL.revokeObjectURL(voiceLocalPreviewUrl.value);
        voiceLocalPreviewUrl.value = null;
      }
    }

    function scrollToVoiceForm() {
      if (!showVoiceForm.value) return;
      Vue.nextTick(() => {
        const el = voiceFormCard.value;
        if (el && typeof el.scrollIntoView === "function") {
          try {
            el.scrollIntoView({ behavior: "smooth", block: "start" });
          } catch {
            el.scrollIntoView();
          }
        }
      });
    }

    function collapseVoiceForm() {
      showVoiceForm.value = false;
      if (voiceFileInput.value) {
        voiceFileInput.value.value = "";
      }
      clearVoiceLocalPreview();
    }

    // 切换音色类型标签（语音/情感），互斥选择
    function toggleVoiceTypeTag(tag) {
      const speakerTag = voiceTypeLabels.speaker;
      const emotionTag = voiceTypeLabels.emotion;
      const tags = voiceForm.value.tags;
      
      if (tag === speakerTag) {
        // 如果已选中语音音色，则取消
        if (tags.includes(speakerTag)) {
          voiceForm.value.tags = tags.filter(t => t !== speakerTag);
        } else {
          // 否则添加语音音色，移除情感音色
          voiceForm.value.tags = [speakerTag, ...tags.filter(t => t !== emotionTag)];
        }
      } else if (tag === emotionTag) {
        // 如果已选中情感音色，则取消
        if (tags.includes(emotionTag)) {
          voiceForm.value.tags = tags.filter(t => t !== emotionTag);
        } else {
          // 否则添加情感音色，移除语音音色
          voiceForm.value.tags = [emotionTag, ...tags.filter(t => t !== speakerTag)];
        }
      }
    }

    function startCreateVoice() {
      showVoiceForm.value = true;
      voiceForm.value = {
        id: null,
        name: "",
        tags: [voiceTypeLabels.speaker],
        tagsInput: "",
        description: "",
        audio_path: null,
        created_at: null,
        updated_at: null,
      };
      if (voiceFileInput.value) {
        voiceFileInput.value.value = "";
      }
      clearVoiceLocalPreview();
      resetVoiceAiGenerate();
      scrollToVoiceForm();
    }

    // 重置 AI 生成状态
    function resetVoiceAiGenerate() {
      voiceAiGenerate.value = {
        instruct: "",
        previewText: "大家好，我是你的配音小助手。",
        language: "Chinese",
        generating: false,
        previewUrl: null,
        previewFile: null,
        error: null,
      };
    }

    // 切换 AI 生成音色快捷标签
    function toggleVoiceAiTag(tag) {
      const current = voiceAiGenerate.value.instruct || "";
      if (current.includes(tag)) {
        voiceAiGenerate.value.instruct = current.replace(tag, "").replace(/，+/g, "，").replace(/^，|，$/g, "").trim();
      } else {
        voiceAiGenerate.value.instruct = current ? `${current}，${tag}` : tag;
      }
    }

    // 使用 AI 生成音色
    async function generateVoiceWithAi() {
      const state = voiceAiGenerate.value;
      if (!state.instruct.trim()) {
        state.error = "请输入音色描述";
        return;
      }

      state.generating = true;
      state.error = null;
      state.previewUrl = null;
      state.previewFile = null;

      try {
        // 自动启动 Qwen3-TTS 服务（如果未运行或运行的是其他后端）
        if (!ttsService.value.running || ttsService.value.backend !== 'qwen3tts') {
          showToast("正在切换到 Qwen3-TTS...");
          // 后端会自动停止旧服务，直接启动新服务即可
          await startTtsService('qwen3tts', { persistDefault: false });
          // 等待服务就绪（包括模型加载）
          const ready = await waitForTtsServiceReady('qwen3tts', 120000);
          if (!ready) {
            state.error = "Qwen3-TTS 服务或模型加载超时，请稍后重试";
            state.generating = false;
            return;
          }
        }

        // 调用 Qwen3-TTS VoiceDesign API
        const resp = await fetchJson("/api/v1/qwen3tts/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: state.previewText || "大家好，我是你的配音小助手。",
            instruct: state.instruct,
            language: state.language,
          }),
        });

        if (resp.success && resp.audio_file) {
          // 创建预览 URL
          state.previewFile = resp.audio_file;
          state.previewUrl = `/api/v1/qwen3tts/audio?path=${encodeURIComponent(resp.audio_file)}`;
          showToast("AI 音色生成成功");
        } else {
          state.error = resp.message || resp.error || "生成失败";
        }
      } catch (err) {
        state.error = err.message || "生成请求失败";
      } finally {
        state.generating = false;
      }
    }

    // 使用 AI 生成的音频作为音色参考
    async function useAiGeneratedVoice() {
      const state = voiceAiGenerate.value;
      if (!state.previewFile) {
        showToast("请先生成音频", "error");
        return;
      }

      // 将生成的音频文件路径设置为待上传文件
      // 由于是服务器端文件，需要通过特殊方式处理
      voiceLocalPreviewUrl.value = state.previewUrl;
      
      // 在 voiceForm 中标记使用 AI 生成的音频
      voiceForm.value._aiGeneratedAudioPath = state.previewFile;
      
      // 如果名称为空，使用音色描述的前几个字作为名称
      if (!voiceForm.value.name && state.instruct) {
        voiceForm.value.name = state.instruct.slice(0, 20).replace(/[，,。.]/g, " ").trim();
      }
      
      // 将音色描述添加到 description
      if (!voiceForm.value.description && state.instruct) {
        voiceForm.value.description = `AI 生成音色：${state.instruct}`;
      }

      showToast("已选择 AI 生成的音频");
    }

    function startEditVoice(v) {
      showVoiceForm.value = true;
      voiceForm.value = {
        id: v.id,
        name: v.name || "",
        tags: Array.isArray(v.tags) ? v.tags.slice() : [],
        tagsInput: Array.isArray(v.tags) ? v.tags.join(" ") : "",
        description: v.description || "",
        audio_path: v.audio_path || null,
        created_at: v.created_at || null,
        updated_at: v.updated_at || null,
      };
      if (voiceFileInput.value) {
        voiceFileInput.value.value = "";
      }
      clearVoiceLocalPreview();
      scrollToVoiceForm();
    }

    function onVoiceFileChange(event) {
      const file = event && event.target && event.target.files && event.target.files[0];
      clearVoiceLocalPreview();
      if (!file) return;
      voiceLocalPreviewUrl.value = URL.createObjectURL(file);

      // 新建音色时，选择参考音频后，始终用文件名填充名称（去掉扩展名）
      if (!voiceForm.value.id && file.name) {
        const rawName = file.name;
        const dotIndex = rawName.lastIndexOf(".");
        const baseName = dotIndex > 0 ? rawName.slice(0, dotIndex) : rawName;
        voiceForm.value.name = baseName;
      }
    }

    function toggleVoiceTag(tag) {
      const current = Array.isArray(voiceForm.value.tags)
        ? voiceForm.value.tags.slice()
        : [];
      let tags = current;
      const wasSelected = tags.includes(tag);
      
      if (wasSelected) {
        tags = tags.filter((t) => t !== tag);
      } else {
        if (isGenderTag(tag)) {
          tags = tags.filter((t) => !isGenderTag(t));
        } else if (isAgeTag(tag)) {
          tags = tags.filter((t) => !isAgeTag(t));
        }
        tags.push(tag);
      }
      tags = enforceExclusiveTags(tags);
      voiceForm.value.tags = tags;
      voiceForm.value.tagsInput = tags.join(" ");
      
      // 同步更新 AI 生成音色的描述（排除类型标签：语音/情感）
      // 顺序：性别 → 年龄段 → 其他标签
      const genderTags = tags.filter(t => isGenderTag(t));
      const ageTags = tags.filter(t => isAgeTag(t));
      const otherTags = tags.filter(t => !isVoiceTypeTag(t) && !isGenderTag(t) && !isAgeTag(t));
      const orderedTags = [...genderTags, ...ageTags, ...otherTags];
      
      if (orderedTags.length > 0) {
        // 同步到 AI 音色描述：用逗号连接
        voiceAiGenerate.value.instruct = orderedTags.join("，");
      } else if (wasSelected) {
        // 如果取消选择且没有剩余相关标签，清空描述
        voiceAiGenerate.value.instruct = "";
      }
    }

    async function submitVoice() {
      // 兜底：如果名称仍为空且选择了参考音频，则在提交前用文件名自动填充一次
      if (!voiceForm.value.name.trim()) {
        const file = voiceFileInput.value && voiceFileInput.value.files ? voiceFileInput.value.files[0] : null;
        if (file && file.name) {
          const rawName = file.name;
          const dotIndex = rawName.lastIndexOf(".");
          const baseName = dotIndex > 0 ? rawName.slice(0, dotIndex) : rawName;
          voiceForm.value.name = baseName;
        }
      }

      if (!voiceForm.value.name.trim()) {
        showToast("音色名称不能为空");
        return;
      }

      // 检查名称唯一性（排除当前编辑的音色）
      const voiceName = voiceForm.value.name.trim();
      const existingByName = (voices.value || []).find(v => 
        v && v.name === voiceName && (!voiceForm.value.id || v.id !== voiceForm.value.id)
      );
      if (existingByName) {
        showToast(`已存在同名音色：${existingByName.name}，请修改名称`);
        return;
      }

      // 获取要上传的文件
      let file = voiceForm.value.audioFile 
        || (voiceFileInput.value && voiceFileInput.value.files ? voiceFileInput.value.files[0] : null);

      if (!file && voiceForm.value._aiGeneratedAudioPath) {
        try {
          const aiPath = String(voiceForm.value._aiGeneratedAudioPath || "");
          const aiResp = await fetch(`/api/v1/qwen3tts/audio?path=${encodeURIComponent(aiPath)}`);
          if (!aiResp.ok) {
            const errText = await aiResp.text().catch(() => "");
            throw new Error(errText || `请求失败 (${aiResp.status})`);
          }
          const aiBlob = await aiResp.blob();
          const rawName = aiPath.split("/").pop()?.split("\\").pop() || "";
          const fileName = rawName && rawName.trim() ? rawName : `ai_voice_${Date.now()}.wav`;
          file = new File([aiBlob], fileName, { type: aiBlob.type || "audio/wav" });
        } catch (e) {
          console.error(e);
          showToast("读取 AI 生成音频失败: " + e.message);
          return;
        }
      }

      // 如果有文件，先检查是否已存在相同音频的音色
      if (file) {
        try {
          const newFileHash = await computeFileHash(file);
          
          // 检查所有音色（排除当前编辑的音色）
          for (const v of (voices.value || [])) {
            if (!v || !v.audio_path) continue;
            if (voiceForm.value.id && v.id === voiceForm.value.id) continue; // 跳过自己
            try {
              const existingRes = await fetch(`/api/v1/voices/${v.id}/audio`);
              if (existingRes.ok) {
                const existingBlob = await existingRes.blob();
                const existingHash = await computeFileHash(existingBlob);
                if (newFileHash === existingHash) {
                  showToast(`已存在相同音频的音色：${v.name}，请直接使用该音色`);
                  return;
                }
              }
            } catch (e) {
              console.warn("无法比对音频:", e);
            }
          }
        } catch (e) {
          console.warn("计算文件哈希失败:", e);
        }
      }

      voiceSaving.value = true;
      try {
        const tags = normalizeVoiceTags(voiceForm.value.tagsInput, voiceForm.value.tags);
        const payload = {
          name: voiceForm.value.name.trim(),
          tags,
          description: voiceForm.value.description || null,
        };

        let voiceData;
        if (voiceForm.value.id) {
          voiceData = await fetchJson(`/api/v1/voices/${voiceForm.value.id}`, {
            method: "PUT",
            body: JSON.stringify(payload),
          });
          showToast("音色已更新");
        } else {
          voiceData = await fetchJson("/api/v1/voices", {
            method: "POST",
            body: JSON.stringify(payload),
          });
          showToast("音色已创建");
        }

        // 如果用户选择了文件或通过麦克风录制，则上传或替换参考音频
        if (file) {
          const formData = new FormData();
          formData.append("file", file);
          const res = await fetch(`/api/v1/voices/${voiceData.id}/audio`, {
            method: "POST",
            body: formData,
          });
          if (!res.ok) {
            const data = await res.json().catch(() => ({}));
            throw new Error(data.detail || res.statusText);
          }
          voiceData = await res.json();
          if (voiceFileInput.value) {
            voiceFileInput.value.value = "";
          }
          clearVoiceLocalPreview();
          voiceForm.value._aiGeneratedAudioPath = null;
          // 清除麦克风录制的文件状态
          voiceForm.value.audioFile = null;
        }

        startEditVoice(voiceData);
        await loadVoices();
      } catch (e) {
        console.error(e);
        showToast("保存音色失败: " + e.message);
      } finally {
        voiceSaving.value = false;
      }
    }

    async function deleteVoice(id) {
      if (!id) return;
      if (!window.confirm("确认删除该音色？")) return;
      try {
        await fetchJson(`/api/v1/voices/${id}`, { method: "DELETE", body: JSON.stringify({}) });
        showToast("音色已删除");
        await loadVoices();
        if (currentProject.value && currentProject.value.id) {
          await loadRolesAndSegments();
        }
        if (voiceForm.value.id === id) {
          startCreateVoice();
        }
      } catch (e) {
        console.error(e);
        showToast("删除音色失败: " + e.message);
      }
    }

    async function openSelectVoiceModal(role) {
      if (!currentProject.value) {
        showToast("请先选择项目");
        return;
      }
      currentRoleForVoice.value = role;
      selectedVoiceId.value = null;

      // 先立刻打开弹窗，避免在等待音色列表加载或其它长请求（如 LLM 分段）时看起来“没反应”
      modals.value.selectVoice = true;

      // 如果当前还没有加载过音色列表，则在后台异步加载
      if (!voices.value.length) {
        try {
          await loadVoices();
        } catch (e) {
          // 加载失败时 loadVoices 内部会处理并提示，这里不再额外中断弹窗
          console.error(e);
        }
      }
    }

    function closeSelectVoiceModal() {
      modals.value.selectVoice = false;
      currentRoleForVoice.value = null;
      selectedVoiceId.value = null;
    }

    async function bindVoiceToRole() {
      if (!currentProject.value || !currentRoleForVoice.value || !selectedVoiceId.value) {
        showToast("请选择要绑定的音色");
        return;
      }
      if (!speakerVoices.value.some((voice) => voice && voice.id === selectedVoiceId.value)) {
        selectedVoiceId.value = null;
        showToast("情感音色不能绑定为角色主音色，请选择“语音”类型的音色");
        return;
      }
      if (!ensureProjectMutationAllowed("绑定角色音色")) return;
      try {
        await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/roles/${currentRoleForVoice.value.id}/bind-voice`,
          {
            method: "POST",
            body: JSON.stringify({ voice_id: selectedVoiceId.value }),
          },
        );
        await loadRolesAndSegments();
        showToast("已从音色库绑定音色");
        closeSelectVoiceModal();
      } catch (e) {
        console.error(e);
        showToast("绑定音色失败: " + e.message);
      }
    }

    function getRoleBoundVoice(role) {
      if (!role || !role.speaker_voice_id) return null;
      const vs = voices.value || [];
      return vs.find((v) => v && v.id === role.speaker_voice_id) || null;
    }

    async function addRoleSpeakerToVoiceLibrary(role) {
      if (!currentProject.value || !role || !role.speaker_audio) {
        showToast("当前角色或项目信息缺失，无法添加到音色库");
        return;
      }
      try {
        const res = await fetch(
          `/api/v1/projects/${currentProject.value.id}/roles/${role.id}/speaker-audio`,
        );
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.detail || res.statusText);
        }
        const blob = await res.blob();
        const file = new File([blob], "role_speaker_audio.webm", {
          type: blob.type || "audio/webm",
        });

        // 先按音频内容比对音色库，避免重复创建
        let matchedVoice = null;
        try {
          if (!voices.value.length) {
            await loadVoices();
          }
          const newHash = await computeFileHash(file);
          for (const v of voices.value || []) {
            if (!v || !v.audio_path) continue;
            try {
              const existingRes = await fetch(`/api/v1/voices/${v.id}/audio`);
              if (!existingRes.ok) continue;
              const existingBlob = await existingRes.blob();
              const existingHash = await computeFileHash(existingBlob);
              if (newHash === existingHash) {
                matchedVoice = v;
                break;
              }
            } catch (e) {
              console.warn("无法比对音频:", e);
            }
          }
        } catch (e) {
          console.warn("计算文件哈希失败:", e);
        }

        if (matchedVoice) {
          await fetchJson(
            `/api/v1/projects/${currentProject.value.id}/roles/${role.id}/bind-voice`,
            {
              method: "POST",
              body: JSON.stringify({ voice_id: matchedVoice.id }),
            },
          );
          await loadRolesAndSegments();
          showToast(`已存在相同音色：${matchedVoice.name}，已为该角色绑定此音色`);
          return;
        }

        const now = new Date();
        const timeStr = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(
          2,
          "0",
        )}${String(now.getDate()).padStart(2, "0")}-${String(now.getHours()).padStart(
          2,
          "0",
        )}${String(now.getMinutes()).padStart(2, "0")}${String(now.getSeconds()).padStart(
          2,
          "0",
        )}`;
        const voiceName = `临时语音音频+${timeStr}`;

        const payload = {
          name: voiceName,
          tags: [voiceTypeLabels.speaker],
          description: `从角色「${role.name}」创建的临时语音音频`,
        };
        let voiceData = await fetchJson("/api/v1/voices", {
          method: "POST",
          body: JSON.stringify(payload),
        });

        const formData = new FormData();
        formData.append("file", file);
        const upRes = await fetch(`/api/v1/voices/${voiceData.id}/audio`, {
          method: "POST",
          body: formData,
        });
        if (!upRes.ok) {
          const data = await upRes.json().catch(() => ({}));
          throw new Error(data.detail || upRes.statusText);
        }
        voiceData = await upRes.json();

        await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/roles/${role.id}/bind-voice`,
          {
            method: "POST",
            body: JSON.stringify({ voice_id: voiceData.id }),
          },
        );

        await Promise.all([loadVoices(), loadRolesAndSegments()]);
        showToast(`已创建临时音色并绑定到角色：${voiceName}`);
      } catch (e) {
        console.error("添加到音色库失败:", e);
        showToast("添加到音色库失败: " + e.message);
      }
    }

    function formatDateTime(value) {
      if (!value) return "";
      try {
        let v = value;
        if (typeof v === "string") {
          if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(v) && !(/[zZ]|[+-]\d{2}:\d{2}$/.test(v))) {
            v = v + "Z";
          }
        }
        const d = new Date(v);
        if (Number.isNaN(d.getTime())) return String(value);
        const pad = (n) => (n < 10 ? "0" + n : String(n));
        const y = d.getFullYear();
        const m = pad(d.getMonth() + 1);
        const day = pad(d.getDate());
        const h = pad(d.getHours());
        const mi = pad(d.getMinutes());
        const s = pad(d.getSeconds());
        return `${y}-${m}-${day} ${h}:${mi}:${s}`;
      } catch {
        return String(value);
      }
    }

    function formatDurationSeconds(value) {
      if (value == null || Number.isNaN(value)) return "——";
      const totalCenti = Math.max(0, Math.round(Number(value) * 100));
      const totalSeconds = totalCenti / 100;
      const h = Math.floor(totalSeconds / 3600);
      const m = Math.floor((totalSeconds % 3600) / 60);
      const s = Math.floor(totalSeconds % 60);
      const cs = totalCenti % 100;
      const pad2 = (n) => (n < 10 ? "0" + n : String(n));
      if (h > 0) {
        return `${h}:${pad2(m)}:${pad2(s)}.${pad2(cs)}`;
      }
      return `${m}:${pad2(s)}.${pad2(cs)}`;
    }

    async function loadFinalAudios() {
      const projectId = currentProject.value && currentProject.value.id;
      if (!projectId) {
        finalAudios.value = [];
        return;
      }
      try {
        const data = await fetchJson(`/api/v1/projects/${projectId}/exports/final`, {
          headers: { "Content-Type": undefined },
        });
        if (!currentProject.value || currentProject.value.id !== projectId) return;
        finalAudios.value = Array.isArray(data) ? data : [];
      } catch (e) {
        console.error(e);
        showToast("加载导出音频列表失败: " + e.message);
      }
    }

    async function loadSubtitles() {
      const projectId = currentProject.value && currentProject.value.id;
      if (!projectId) {
        subtitles.value = [];
        return;
      }
      try {
        const data = await fetchJson(`/api/v1/projects/${projectId}/subtitles`, {
          headers: { "Content-Type": undefined },
        });
        if (!currentProject.value || currentProject.value.id !== projectId) return;
        subtitles.value = Array.isArray(data) ? data : [];
      } catch (e) {
        console.error(e);
        showToast("加载字幕列表失败: " + e.message);
      }
    }

    async function generateSubtitleForLatestFull() {
      if (!currentProject.value) {
        showToast("请先选择项目");
        return;
      }
      if (!finalAudios.value || !finalAudios.value.length) {
        showToast("当前没有整书导出记录，请先导出整书音频。");
        return;
      }

      const target = finalAudios.value[0];
      await generateSubtitleForAudio(target);
    }

    async function generateSubtitleForAudio(fa) {
      if (!currentProject.value) {
        showToast("请先选择项目");
        return;
      }
      if (!fa || !fa.filename) {
        showToast("无效的导出音频记录");
        return;
      }

      // 轻量提示：如果当前检测到 TTS 模型已加载，提醒用户可以先卸载 TTS 释放显存
      try {
        await refreshHealth();
      } catch (e) {
        console.error(e);
      }
      const ttsLoaded = !!(health.value && health.value.tts_model_loaded);
      if (ttsLoaded) {
        const ok = window.confirm(
          "检测到 TTS 模型当前已加载，生成整书字幕可能会与 TTS 争抢显存。\n\n" +
            "是否先从显存卸载 TTS 模型，然后再生成字幕？\n\n" +
            "选择“取消”将不卸载 TTS，直接生成字幕。",
        );
        if (ok) {
          try {
            await unloadTtsModel();
          } catch (e) {
            console.error(e);
          }
        }
      }

      loading.value.generateSubtitle = true;
      try {
        const lang = subtitleLanguage.value;
        const payload = {
          audio_filename: fa.filename,
          task: subtitleTask.value === "translate" ? "translate" : "transcribe",
          language: lang && lang !== "auto" ? lang : null,
          model_size: subtitleModel.value || null,
        };
        await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/subtitles/generate-from-final`,
          {
            method: "POST",
            body: JSON.stringify(payload),
          },
        );
        showToast("字幕生成完成");
        await loadSubtitles();
        try {
          await refreshHealth();
        } catch (e) {
          console.error(e);
        }
      } catch (e) {
        console.error(e);
        showToast("生成字幕失败: " + e.message);
      } finally {
        loading.value.generateSubtitle = false;
      }
    }

    async function deleteFinalAudio(fa) {
      if (!currentProject.value) {
        showToast("请先选择项目");
        return;
      }
      if (!fa || !fa.filename) {
        showToast("无效的导出音频");
        return;
      }
      if (!confirm(`确定要删除导出音频 "${fa.filename}" 及其相关文件吗？此操作不可恢复。`)) {
        return;
      }
      loading.value.deleteFinalAudio = true;
      try {
        await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/exports/final/${encodeURIComponent(fa.filename)}`,
          { method: "DELETE" }
        );
        if (subtitleEditing.value && subtitleEditing.value.audio_filename === fa.filename) {
          closeSubtitleEditor();
        }
        showToast("删除成功");
        await loadFinalAudios();
        await loadSubtitles();
      } catch (e) {
        console.error(e);
        showToast("删除失败: " + e.message);
      } finally {
        loading.value.deleteFinalAudio = false;
      }
    }

    async function openSubtitleEditor(sub) {
      if (!currentProject.value) {
        showToast("请先选择项目");
        return;
      }
      if (!sub || !sub.id) {
        showToast("无效的字幕记录");
        return;
      }
      subtitleEditing.value = null;
      subtitleEditingLoading.value = true;
      modals.value.subtitleEdit = true;
      try {
        const data = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/subtitles/${sub.id}`,
          {
            headers: { "Content-Type": undefined },
          },
        );
        if (!modals.value.subtitleEdit) {
          return;
        }
        if (!data || !Array.isArray(data.segments)) {
          data.segments = Array.isArray(data.segments) ? data.segments : [];
        }
        subtitleEditing.value = data;
      } catch (e) {
        console.error(e);
        showToast("加载字幕详情失败: " + e.message);
        modals.value.subtitleEdit = false;
      } finally {
        subtitleEditingLoading.value = false;
      }
    }

    function closeSubtitleEditor() {
      modals.value.subtitleEdit = false;
      subtitleEditing.value = null;
      subtitleEditingLoading.value = false;
      subtitleEditingSaving.value = false;
      subtitleEditMode.value = "table";
      subtitleJsonText.value = "";
    }

    function switchToJsonEditMode() {
      if (subtitleEditing.value) {
        subtitleJsonText.value = JSON.stringify(subtitleEditing.value, null, 2);
      }
      subtitleEditMode.value = "json";
    }

    function validateSubtitleJsonPayload(payload, base, projectId) {
      if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
        return "JSON 根节点必须是对象";
      }

      const requiredStringFields = ["id", "project_id", "audio_rel_path", "audio_filename"];
      const missing = requiredStringFields.filter((key) => {
        const value = payload[key];
        return typeof value !== "string" || !value.trim();
      });
      if (missing.length) {
        return `JSON 缺少字段: ${missing.join(", ")}`;
      }

      if (base && payload.id !== base.id) {
        return "JSON 中的 id 与当前字幕不一致";
      }

      if (projectId && payload.project_id !== projectId) {
        return "JSON 中的 project_id 与当前项目不一致";
      }

      if (!Array.isArray(payload.segments)) {
        return "JSON 中的 segments 必须为数组";
      }

      for (let i = 0; i < payload.segments.length; i += 1) {
        const seg = payload.segments[i];
        if (!seg || typeof seg !== "object" || Array.isArray(seg)) {
          return `segments[${i}] 必须为对象`;
        }
        if (!Number.isFinite(seg.index) || seg.index < 0) {
          return `segments[${i}].index 无效`;
        }
        if (!Number.isFinite(seg.start) || seg.start < 0) {
          return `segments[${i}].start 无效`;
        }
        if (!Number.isFinite(seg.end) || seg.end < 0) {
          return `segments[${i}].end 无效`;
        }
        if (seg.end < seg.start) {
          return `segments[${i}].end 不能小于 start`;
        }
        if (typeof seg.text !== "string") {
          return `segments[${i}].text 必须为字符串`;
        }
        if (seg.words != null && !Array.isArray(seg.words)) {
          return `segments[${i}].words 必须为数组`;
        }
      }

      return "";
    }

    async function saveSubtitleEdits() {
      if (!currentProject.value) {
        showToast("请先选择项目");
        return;
      }
      if (!subtitleEditing.value || !subtitleEditing.value.id) {
        showToast("当前没有可保存的字幕编辑");
        return;
      }

      const base = subtitleEditing.value;
      let payload = base;

      // 如果是 JSON 模式，先解析 JSON 文本并校验
      if (subtitleEditMode.value === "json") {
        let parsed;
        try {
          parsed = JSON.parse(subtitleJsonText.value);
        } catch (e) {
          showToast("JSON 格式错误: " + e.message);
          return;
        }
        const error = validateSubtitleJsonPayload(parsed, base, currentProject.value && currentProject.value.id);
        if (error) {
          showToast(error);
          return;
        }
        payload = parsed;
      }

      subtitleEditingSaving.value = true;
      try {
        const data = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/subtitles/${base.id}`,
          {
            method: "PUT",
            body: JSON.stringify(payload),
          },
        );
        subtitleEditing.value = data;
        if (subtitleEditMode.value === "json") {
          subtitleJsonText.value = JSON.stringify(data, null, 2);
        }
        showToast("字幕已保存");
        await loadSubtitles();
      } catch (e) {
        console.error(e);
        showToast("保存字幕失败: " + e.message);
      } finally {
        subtitleEditingSaving.value = false;
      }
    }

    async function deleteSubtitle(sub) {
      if (!currentProject.value || !sub || !sub.id) return;
      if (!window.confirm(`确认删除该字幕？\n\n源音频：${sub.audio_filename || "(未知)"}`)) {
        return;
      }
      try {
        await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/subtitles/${sub.id}`,
          {
            method: "DELETE",
            body: JSON.stringify({}),
          },
        );
        showToast("字幕已删除");
        await loadSubtitles();
      } catch (e) {
        console.error(e);
        showToast("删除字幕失败: " + e.message);
      }
    }

    function openImportJsonModal() {
      if (!currentProject.value) {
        showToast("请先选择项目");
        return;
      }
      jsonImportText.value = "";
      modals.value.importJson = true;
      Vue.nextTick(() => {
        if (jsonFileInput.value) {
          jsonFileInput.value.value = "";
        }
      });
    }

    function closeImportJsonModal() {
      modals.value.importJson = false;
      jsonImportText.value = "";
    }

    function handleJsonFileChange(event) {
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target && typeof e.target.result === "string" ? e.target.result : "";
        jsonImportText.value = text || "";
      };
      reader.readAsText(file, "utf-8");
    }

    async function loadMasters() {
      const projectId = currentProject.value && currentProject.value.id;
      if (!projectId) {
        masters.value = [];
        return;
      }
      try {
        const data = await fetchJson(`/api/v1/projects/${projectId}/masters`);
        masters.value = Array.isArray(data) ? data : [];
      } catch (e) {
        console.error("加载 Master 失败:", e);
        masters.value = [];
      }
    }

    function productionDefaults(data = null) {
      const source = data && typeof data === "object" ? data : {};
      return {
        ...source,
        pronunciation_dictionary: Array.isArray(source.pronunciation_dictionary) ? source.pronunciation_dictionary : [],
        sound_design: {
          version: 1,
          clips: [],
          mix: {},
          ...(source.sound_design || {}),
        },
        release: { ...(source.release || {}) },
      };
    }

    async function loadProductionWorkspace() {
      const projectId = currentProject.value && currentProject.value.id;
      if (!projectId) return;
      try {
        const [settings, assets, releases] = await Promise.all([
          fetchJson(`/api/v1/projects/${projectId}/production`),
          fetchJson(`/api/v1/projects/${projectId}/production/assets`),
          fetchJson(`/api/v1/projects/${projectId}/release-packages`),
        ]);
        productionSettings.value = productionDefaults(settings);
        audioAssets.value = Array.isArray(assets) ? assets : [];
        releasePackages.value = Array.isArray(releases) ? releases : [];
        await loadSoundGenerationWorkspace();
      } catch (e) {
        console.error("加载声音制作工作区失败:", e);
        showToast("加载声音制作工作区失败: " + e.message);
      }
    }

    async function saveContentMode() {
      if (!currentProject.value) return;
      try {
        const settings = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/production/profile`,
          { method: "PUT", body: JSON.stringify({ content_mode: productionSettings.value.content_mode || "story" }) },
        );
        productionSettings.value = productionDefaults(settings);
        showToast("作品类型已保存");
      } catch (e) {
        showToast("保存作品类型失败: " + e.message);
      }
    }

    async function createChapter() {
      if (!currentProject.value) return;
      const title = prompt("章节或场景名称");
      if (!title || !title.trim()) return;
      try {
        await fetchJson(`/api/v1/projects/${currentProject.value.id}/chapters`, {
          method: "POST", body: JSON.stringify({ title: title.trim() }),
        });
        await loadProjects();
        showToast("章节/场景已创建");
      } catch (e) {
        showToast("创建章节/场景失败: " + e.message);
      }
    }

    async function deleteChapter(chapter) {
      if (!currentProject.value || !chapter || !confirm(`删除“${chapter.title}”？`)) return;
      try {
        await fetchJson(`/api/v1/projects/${currentProject.value.id}/chapters/${chapter.id}`, { method: "DELETE" });
        await loadProjects();
        showToast("章节/场景已删除");
      } catch (e) {
        showToast("无法删除章节/场景: " + e.message);
      }
    }

    async function onSegmentChapterChange(seg, chapterId) {
      try {
        const updated = await updateSegment(seg.id, { chapter_id: chapterId || null });
        Object.assign(seg, updated);
      } catch (e) {
        showToast("更新章节归属失败: " + e.message);
        await loadRolesAndSegments();
      }
    }

    function newLocalId(prefix) {
      if (window.crypto && typeof window.crypto.randomUUID === "function") {
        return `${prefix}_${window.crypto.randomUUID().replaceAll("-", "").slice(0, 12)}`;
      }
      return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
    }

    async function savePronunciations() {
      if (!currentProject.value) return;
      productionBusy.value = true;
      try {
        const settings = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/production/pronunciations`,
          { method: "PUT", body: JSON.stringify({ entries: productionSettings.value.pronunciation_dictionary }) },
        );
        productionSettings.value = productionDefaults(settings);
        showToast("发音词典已保存；现有 Take 不会被覆盖");
      } catch (e) {
        showToast("保存发音词典失败: " + e.message);
      } finally {
        productionBusy.value = false;
      }
    }

    async function addPronunciation() {
      const draft = pronunciationDraft.value;
      if (!draft.source.trim() || !draft.pronunciation.trim()) {
        showToast("请填写原词和目标读法");
        return;
      }
      productionSettings.value.pronunciation_dictionary.push({
        id: newLocalId("term"), source: draft.source.trim(), pronunciation: draft.pronunciation.trim(),
        note: draft.note.trim() || null, enabled: true,
      });
      pronunciationDraft.value = { source: "", pronunciation: "", note: "" };
      await savePronunciations();
    }

    async function removePronunciation(id) {
      productionSettings.value.pronunciation_dictionary = productionSettings.value.pronunciation_dictionary.filter((item) => item.id !== id);
      await savePronunciations();
    }

    async function uploadSoundAsset() {
      const file = assetFileInput.value && assetFileInput.value.files && assetFileInput.value.files[0];
      if (!currentProject.value || !file) {
        showToast("请选择本地音频素材");
        return;
      }
      productionBusy.value = true;
      try {
        const form = new FormData();
        form.append("file", file);
        Object.entries(assetUploadForm.value).forEach(([key, value]) => {
          if (value !== null && value !== undefined) form.append(key, String(value));
        });
        const response = await fetch(`/api/v1/projects/${currentProject.value.id}/production/assets`, { method: "POST", body: form });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || response.statusText);
        audioAssets.value.push(data);
        assetUploadForm.value.name = "";
        if (assetFileInput.value) assetFileInput.value.value = "";
        showToast("声音素材已导入并记录校验值");
      } catch (e) {
        showToast("导入素材失败: " + e.message);
      } finally {
        productionBusy.value = false;
      }
    }

    function assetAudioUrl(asset) {
      if (!currentProject.value || !asset) return "";
      return `/api/v1/projects/${currentProject.value.id}/production/assets/${asset.id}/audio`;
    }

    function soundTrackLabel(track) {
      return ({ bgm: "BGM", ambience: "Ambience", sfx: "SFX" })[track] || track;
    }

    function replaceSoundPlan(plan) {
      if (!plan || !plan.id) return;
      const index = soundPlans.value.findIndex((item) => item.id === plan.id);
      if (index >= 0) soundPlans.value.splice(index, 1, plan);
      else soundPlans.value.unshift(plan);
      activeSoundPlanId.value = plan.id;
    }

    function clearSoundJobPolling() {
      if (soundJobPollTimer) {
        clearInterval(soundJobPollTimer);
        soundJobPollTimer = null;
      }
    }

    function hasActiveSoundJob() {
      return soundJobs.value.some((job) => ["queued", "running", "cancelling"].includes(job.status));
    }

    async function refreshSoundJobs(options = {}) {
      const projectId = currentProject.value && currentProject.value.id;
      if (!projectId) return;
      try {
        const jobs = await fetchJson(`/api/v1/projects/${projectId}/sound-jobs`);
        if (!currentProject.value || currentProject.value.id !== projectId) return;
        const wasActive = hasActiveSoundJob();
        soundJobs.value = Array.isArray(jobs) ? jobs : [];
        const isActive = hasActiveSoundJob();
        soundGenerationBusy.value = isActive;
        if (!isActive) {
          clearSoundJobPolling();
          if (wasActive || options.reloadPlan) {
            const plans = await fetchJson(`/api/v1/projects/${projectId}/sound-plans`);
            soundPlans.value = Array.isArray(plans) ? plans : [];
            if (!soundPlans.value.some((item) => item.id === activeSoundPlanId.value)) {
              activeSoundPlanId.value = soundPlans.value[0] ? soundPlans.value[0].id : "";
            }
          }
        } else if (!soundJobPollTimer) {
          soundJobPollTimer = setInterval(() => void refreshSoundJobs(), 1500);
        }
      } catch (e) {
        console.error("刷新音效生成任务失败:", e);
      }
    }

    async function loadSoundGenerationWorkspace() {
      const projectId = currentProject.value && currentProject.value.id;
      if (!projectId) return;
      try {
        const [model, plans, jobs] = await Promise.all([
          fetchJson("/api/v1/sound-generation/model"),
          fetchJson(`/api/v1/projects/${projectId}/sound-plans`),
          fetchJson(`/api/v1/projects/${projectId}/sound-jobs`),
        ]);
        if (!currentProject.value || currentProject.value.id !== projectId) return;
        soundModelStatus.value = model;
        soundPlans.value = Array.isArray(plans) ? plans : [];
        soundJobs.value = Array.isArray(jobs) ? jobs : [];
        if (!soundPlans.value.some((item) => item.id === activeSoundPlanId.value)) {
          activeSoundPlanId.value = soundPlans.value[0] ? soundPlans.value[0].id : "";
        }
        if (activeSoundPlan.value && activeSoundPlan.value.density) {
          soundPlanDensity.value = activeSoundPlan.value.density;
        } else {
          soundPlanDensity.value = productionSettings.value.content_mode === "story" ? "balanced" : "conservative";
        }
        soundGenerationBusy.value = hasActiveSoundJob();
        if (soundGenerationBusy.value && !soundJobPollTimer) {
          soundJobPollTimer = setInterval(() => void refreshSoundJobs(), 1500);
        }
      } catch (e) {
        console.error("加载 Stable Audio 工作区失败:", e);
        showToast("加载音效生成工作区失败: " + e.message);
      }
    }

    async function confirmStableAudioLicense() {
      const form = soundLicenseForm.value;
      if (!(form.stability_registered && form.stability_terms_accepted && form.gemma_terms_accepted)) {
        showToast("请逐项确认 Stability 注册、Stability 条款和 Gemma 条款");
        return;
      }
      try {
        await fetchJson("/api/v1/sound-generation/license-confirmation", {
          method: "POST",
          body: JSON.stringify(form),
        });
        soundModelStatus.value = await fetchJson("/api/v1/sound-generation/model");
        showToast("Stable Audio 许可确认已绑定当前许可哈希");
      } catch (e) {
        showToast("保存许可确认失败: " + e.message);
      }
    }

    async function draftSoundPlan() {
      if (!currentProject.value || soundGenerationBusy.value) return;
      soundGenerationBusy.value = true;
      try {
        const plan = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/sound-plans`,
          {
            method: "POST",
            body: JSON.stringify({ density: soundPlanDensity.value }),
          },
        );
        replaceSoundPlan(plan);
        showToast(`SoundPlan 已起草：${plan.cues.length} 个 cue，需逐条人工批准`);
      } catch (e) {
        showToast("起草 SoundPlan 失败: " + e.message);
      } finally {
        soundGenerationBusy.value = false;
      }
    }

    async function saveSoundCue(cue, reviewStatus = null) {
      if (!currentProject.value || !activeSoundPlan.value || !cue) return;
      const payload = {
        track: cue.track,
        description_cn: cue.description_cn || null,
        prompt: cue.prompt,
        placement: cue.placement,
        requested_duration_seconds: Number(cue.requested_duration_seconds),
        playback_duration_seconds: cue.playback_duration_seconds ? Number(cue.playback_duration_seconds) : null,
        gain_db: Number(cue.gain_db || 0),
        fade_in: Number(cue.fade_in || 0),
        fade_out: Number(cue.fade_out || 0),
        loop: Boolean(cue.loop),
        enabled: Boolean(cue.enabled),
      };
      if (reviewStatus) payload.review_status = reviewStatus;
      try {
        const plan = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/sound-plans/${activeSoundPlan.value.id}/cues/${cue.id}`,
          { method: "PATCH", body: JSON.stringify(payload) },
        );
        replaceSoundPlan(plan);
        if (cue.adopted_asset_id) {
          const settings = await fetchJson(
            `/api/v1/projects/${currentProject.value.id}/production`,
          );
          productionSettings.value = productionDefaults(settings);
        }
        showToast(reviewStatus === "approved" ? "cue 已批准" : reviewStatus === "rejected" ? "cue 已拒绝" : "cue 已保存；生成参数变化会要求重新批准");
      } catch (e) {
        showToast("保存 cue 失败: " + e.message);
      }
    }

    function normalizeSoundCueTrack(cue) {
      if (!cue || !cue.placement) return;
      if (cue.track === "ambience") {
        if (!cue.placement.end) {
          const fallbackId = cue.placement.start.segment_id || (segments.value[0] && segments.value[0].id);
          cue.placement.end = { segment_id: fallbackId, edge: "end", offset_seconds: 0 };
        }
        cue.requested_duration_seconds = Math.min(60, Math.max(20, Number(cue.requested_duration_seconds || 20)));
        cue.loop = true;
      } else {
        cue.placement.end = null;
        cue.requested_duration_seconds = Math.min(15, Math.max(1, Number(cue.requested_duration_seconds || 4)));
        cue.loop = false;
      }
    }

    async function generateCueCandidates(cue, count = 2) {
      if (!currentProject.value || !activeSoundPlan.value || !cue || soundGenerationBusy.value) return;
      if (!soundModelStatus.value || !soundModelStatus.value.enabled) {
        showToast("Stable Audio 模型、许可或真实 CUDA 质量门尚未就绪");
        return;
      }
      soundGenerationBusy.value = true;
      try {
        const job = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/sound-plans/${activeSoundPlan.value.id}/jobs`,
          { method: "POST", body: JSON.stringify({ cue_ids: [cue.id], candidate_count: count }) },
        );
        soundJobs.value.unshift(job);
        if (!soundJobPollTimer) soundJobPollTimer = setInterval(() => void refreshSoundJobs(), 1500);
        showToast("候选生成已启动；关闭浏览器不会中止任务");
      } catch (e) {
        soundGenerationBusy.value = false;
        showToast("启动候选生成失败: " + e.message);
      }
    }

    function soundCandidateAudioUrl(cue, candidate) {
      if (!currentProject.value || !activeSoundPlan.value || !cue || !candidate) return "";
      return `/api/v1/projects/${currentProject.value.id}/sound-plans/${activeSoundPlan.value.id}/cues/${cue.id}/candidates/${candidate.id}/audio`;
    }

    async function drawCandidateWaveform(canvas, cue, candidate) {
      if (!canvas || !candidate || !["ready", "adopted"].includes(candidate.status)) return;
      const url = soundCandidateAudioUrl(cue, candidate);
      if (!url || canvas.dataset.waveUrl === url) return;
      canvas.dataset.waveUrl = url;
      const context = canvas.getContext("2d");
      const width = Math.max(240, canvas.clientWidth || 240);
      const height = 38;
      canvas.width = width;
      canvas.height = height;
      context.clearRect(0, 0, width, height);
      context.strokeStyle = "#334155";
      context.beginPath(); context.moveTo(0, height / 2); context.lineTo(width, height / 2); context.stroke();
      try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(response.statusText);
        const bytes = await response.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const buffer = await audioContext.decodeAudioData(bytes.slice(0));
        const samples = buffer.getChannelData(0);
        const block = Math.max(1, Math.floor(samples.length / width));
        context.strokeStyle = "#38bdf8";
        context.beginPath();
        for (let x = 0; x < width; x += 1) {
          let peak = 0;
          const start = x * block;
          const end = Math.min(samples.length, start + block);
          for (let i = start; i < end; i += 1) peak = Math.max(peak, Math.abs(samples[i]));
          const amplitude = Math.max(1, peak * (height / 2));
          context.moveTo(x, height / 2 - amplitude);
          context.lineTo(x, height / 2 + amplitude);
        }
        context.stroke();
        await audioContext.close();
      } catch (e) {
        canvas.dataset.waveUrl = "";
        console.debug("候选波形解码失败:", e);
      }
    }

    async function adoptSoundCandidate(cue, candidate) {
      if (!currentProject.value || !activeSoundPlan.value) return;
      try {
        const plan = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/sound-plans/${activeSoundPlan.value.id}/cues/${cue.id}/candidates/${candidate.id}/adopt`,
          { method: "POST" },
        );
        replaceSoundPlan(plan);
        await loadProductionWorkspace();
        showToast("候选已原子注册为正式素材，并按语义锚点加入时间线");
      } catch (e) {
        showToast("采用候选失败: " + e.message);
      }
    }

    async function unadoptSoundCandidate(cue) {
      if (!currentProject.value || !activeSoundPlan.value || !cue || !cue.adopted_asset_id) return;
      if (!confirm("解除采用会从当前时间线和素材库移除正式副本，但保留候选 WAV，是否继续？")) return;
      try {
        const plan = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/sound-plans/${activeSoundPlan.value.id}/cues/${cue.id}/adoption`,
          { method: "DELETE" },
        );
        replaceSoundPlan(plan);
        await loadProductionWorkspace();
        showToast("已解除采用；候选仍可重新采用或清理");
      } catch (e) {
        showToast("解除采用失败: " + e.message);
      }
    }

    async function deleteSoundCandidate(cue, candidate) {
      if (!currentProject.value || !activeSoundPlan.value || !confirm("清理这个未采用候选及其 WAV？")) return;
      try {
        await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/sound-plans/${activeSoundPlan.value.id}/cues/${cue.id}/candidates/${candidate.id}`,
          { method: "DELETE" },
        );
        const plan = await fetchJson(`/api/v1/projects/${currentProject.value.id}/sound-plans/${activeSoundPlan.value.id}`);
        replaceSoundPlan(plan);
        showToast("未采用候选已清理");
      } catch (e) {
        showToast("清理候选失败: " + e.message);
      }
    }

    async function createSoundPreview() {
      if (!currentProject.value || productionBusy.value) return;
      productionBusy.value = true;
      try {
        if (!(await saveSoundDesign())) return;
        soundPreview.value = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/sound-preview`,
          { method: "POST" },
        );
        soundPreviewAudioUrl.value = `${soundPreview.value.audio_url}?sha=${soundPreview.value.audio_sha256}`;
        showToast(`四轨 Preview 已生成，含 ${soundPreview.value.warnings.length} 项非阻断提示`);
      } catch (e) {
        showToast("生成四轨 Preview 失败: " + e.message);
      } finally {
        productionBusy.value = false;
      }
    }

    function cueAnchorLabel(cue) {
      if (!cue || !cue.placement || !cue.placement.start) return "未设置";
      const start = cue.placement.start;
      const end = cue.placement.end;
      const offset = Number(start.offset_seconds || 0);
      const startText = `${start.segment_id}:${start.edge}${offset >= 0 ? "+" : ""}${offset.toFixed(2)}s`;
      if (!end) return startText;
      const endOffset = Number(end.offset_seconds || 0);
      return `${startText} → ${end.segment_id}:${end.edge}${endOffset >= 0 ? "+" : ""}${endOffset.toFixed(2)}s`;
    }

    async function deleteSoundAsset(asset) {
      if (!currentProject.value || !asset || !confirm(`删除素材“${asset.name}”？`)) return;
      try {
        await fetchJson(`/api/v1/projects/${currentProject.value.id}/production/assets/${asset.id}`, { method: "DELETE" });
        audioAssets.value = audioAssets.value.filter((item) => item.id !== asset.id);
        showToast("素材已删除");
      } catch (e) {
        showToast("无法删除素材: " + e.message);
      }
    }

    async function saveSoundDesign() {
      if (!currentProject.value) return false;
      productionBusy.value = true;
      try {
        const settings = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/production/sound-design`,
          { method: "PUT", body: JSON.stringify(productionSettings.value.sound_design) },
        );
        productionSettings.value = productionDefaults(settings);
        showToast("四轨时间线与混音设置已保存");
        return true;
      } catch (e) {
        showToast("保存声音设计失败: " + e.message);
        return false;
      } finally {
        productionBusy.value = false;
      }
    }

    async function addTimelineClip() {
      const asset = audioAssets.value.find((item) => item.id === timelineDraft.value.asset_id);
      if (!asset) {
        showToast("请先选择素材");
        return;
      }
      const draft = timelineDraft.value;
      productionSettings.value.sound_design.clips.push({
        id: newLocalId("clip"), track: asset.track, asset_id: asset.id,
        start: Number(draft.start || 0), source_offset: 0,
        duration: draft.duration ? Number(draft.duration) : null,
        gain_db: Number(draft.gain_db || 0), fade_in: Number(draft.fade_in || 0),
        fade_out: Number(draft.fade_out || 0), loop: Boolean(draft.loop),
      });
      await saveSoundDesign();
    }

    async function removeTimelineClip(id) {
      productionSettings.value.sound_design.clips = productionSettings.value.sound_design.clips.filter((item) => item.id !== id);
      await saveSoundDesign();
    }

    function isAdoptedSoundClip(id) {
      return soundPlans.value.some((plan) =>
        (plan.cues || []).some((cue) => cue.adopted_clip_id === id),
      );
    }

    async function runQualityCheck() {
      if (!currentProject.value) return;
      productionBusy.value = true;
      try {
        qualityReport.value = await fetchJson(`/api/v1/projects/${currentProject.value.id}/quality-check`);
        showToast(qualityReport.value.passed ? "质量检查通过" : "质量检查发现阻断问题");
      } catch (e) {
        showToast("质量检查失败: " + e.message);
      } finally {
        productionBusy.value = false;
      }
    }

    async function saveReleaseMetadata() {
      if (!currentProject.value) return;
      try {
        const settings = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/production/release-metadata`,
          { method: "PUT", body: JSON.stringify(productionSettings.value.release) },
        );
        productionSettings.value = productionDefaults(settings);
        showToast("发布信息已保存");
      } catch (e) {
        showToast("保存发布信息失败: " + e.message);
      }
    }

    async function uploadReleaseCover() {
      const file = coverFileInput.value && coverFileInput.value.files && coverFileInput.value.files[0];
      if (!currentProject.value || !file) return;
      const form = new FormData();
      form.append("file", file);
      try {
        const response = await fetch(`/api/v1/projects/${currentProject.value.id}/production/cover`, { method: "POST", body: form });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || response.statusText);
        productionSettings.value = productionDefaults(data);
        showToast("发布封面已保存");
      } catch (e) {
        showToast("上传封面失败: " + e.message);
      }
    }

    async function createReleasePackage(master) {
      if (!currentProject.value || !master || productionBusy.value) return;
      productionBusy.value = true;
      try {
        await saveReleaseMetadata();
        const data = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/release-packages/${master.id}`,
          { method: "POST" },
        );
        releasePackages.value.unshift(data);
        showToast("通用发布包已生成");
      } catch (e) {
        showToast("生成发布包失败: " + e.message);
      } finally {
        productionBusy.value = false;
      }
    }

    function masterAudioUrl(master) {
      if (!currentProject.value || !master) return "";
      return `/api/v1/projects/${currentProject.value.id}/masters/${master.id}/audio`;
    }

    async function createReviewedMaster() {
      if (!currentProject.value || creatingMaster.value) return;
      const name = prompt("Master 名称", `正式版 ${new Date().toLocaleString()}`);
      if (name === null) return;
      creatingMaster.value = true;
      try {
        await flushWaveformStatePersist(currentProject.value.id);
        await fetchJson(`/api/v1/projects/${currentProject.value.id}/masters`, {
          method: "POST",
          body: JSON.stringify({ name: name || null }),
        });
        await loadMasters();
        showToast("正式 Master 已生成并冻结");
      } catch (e) {
        showToast("生成 Master 失败: " + e.message);
      } finally {
        creatingMaster.value = false;
      }
    }

    function confirmReplacingExistingSegments(actionLabel) {
      const existing = Array.isArray(segments.value) ? segments.value : [];
      if (!existing.length) return true;
      const audioCount = existing.filter(
        (seg) => seg && (seg.audio_file || seg.alt_audio_file || seg.emotion_audio),
      ).length;
      const audioWarning = audioCount > 0
        ? `，其中 ${audioCount} 段包含已生成配音或情感参考音频`
        : "";
      return window.confirm(
        `当前项目已有 ${existing.length} 个分段${audioWarning}。\n\n`
        + `${actionLabel}会覆盖现有分段列表，原分段及其关联状态可能无法继续使用。确认继续吗？`,
      );
    }

    async function loadLlmConfigs() {
      try {
        const data = await fetchJson("/api/v1/llm-configs", { headers: { "Content-Type": undefined } });
        const normalized = Array.isArray(data)
          ? data.map((c) => {
              const name =
                c && typeof c.name === "string"
                  ? c.name.replace(/（默认）|\(默认\)/g, "").trim()
                  : c.name;
              return { ...c, name };
            })
          : [];
        llmConfigs.value = normalized;
        const defaultCfg = normalized.find((c) => c.is_default) || normalized[0];
        if (defaultCfg) {
          llm.value.model = defaultCfg.model;
          llm.value.configId = defaultCfg.id;
        }

        // 如果尚未选择特定提供商，则基于配置推断一个常用提供商，便于右侧列表默认过滤
        if (!selectedLlmProviderId.value && normalized.length) {
          const withProvider = normalized.find((c) => c.provider_id) || null;
          if (withProvider && withProvider.provider_id) {
            selectedLlmProviderId.value = withProvider.provider_id;
          }
        }
      } catch (e) {
        console.error(e);
      }
    }

    function onLlmConfigChange() {
      if (!llm.value.configId) return;
      const cfg = llmConfigs.value.find((c) => c.id === llm.value.configId);
      if (cfg) {
        llm.value.model = cfg.model;
      }
    }

    function startCreateLlmConfig() {
      llmConfigForm.value = {
        id: null,
        name: "",
        provider_id: null,
        model: "deepseek-v4-flash",
        max_tokens_k: null,
        description: "",
        response_format_json: false,
      };
      modals.value.createLlmConfig = true;
    }

    function startEditLlmConfig(cfg) {
      llmConfigForm.value = {
        id: cfg.id,
        name: cfg.name || "",
        provider_id: cfg.provider_id ?? null,
        model: cfg.model || "",
        max_tokens_k: cfg.max_tokens ? Math.round(cfg.max_tokens / 1000) : null,
        description: cfg.description || "",
        response_format_json: !!cfg.response_format_json,
      };
      modals.value.createLlmConfig = true;
    }

    function closeLlmConfigModal() {
      modals.value.createLlmConfig = false;
    }

    async function submitLlmConfig() {
      if (!llmConfigForm.value.name.trim() || !llmConfigForm.value.model.trim()) {
        showToast("LLM 配置名称和模型不能为空");
        return;
      }
      llmConfigSaving.value = true;
      try {
        const payload = {
          name: llmConfigForm.value.name.trim(),
          provider_id: llmConfigForm.value.provider_id || null,
          model: llmConfigForm.value.model.trim(),
          max_tokens:
            typeof llmConfigForm.value.max_tokens_k === "number" && !Number.isNaN(llmConfigForm.value.max_tokens_k) && llmConfigForm.value.max_tokens_k > 0
              ? llmConfigForm.value.max_tokens_k * 1000
              : null,
          description: llmConfigForm.value.description || null,
          response_format_json: !!llmConfigForm.value.response_format_json,
        };

        if (llmConfigForm.value.id) {
          await fetchJson(`/api/v1/llm-configs/${llmConfigForm.value.id}`, {
            method: "PUT",
            body: JSON.stringify(payload),
          });
          showToast("LLM 配置已更新");
        } else {
          await fetchJson("/api/v1/llm-configs", {
            method: "POST",
            body: JSON.stringify(payload),
          });
          showToast("LLM 配置已创建");
        }

        await loadLlmConfigs();
        closeLlmConfigModal();
      } catch (e) {
        console.error(e);
        showToast("保存 LLM 配置失败: " + e.message);
      } finally {
        llmConfigSaving.value = false;
      }
    }

    async function deleteLlmConfig(id) {
      if (!id) return;
      if (!window.confirm("确认删除该 LLM 配置？")) return;
      try {
        await fetchJson(`/api/v1/llm-configs/${id}`, { method: "DELETE", body: JSON.stringify({}) });
        showToast("LLM 配置已删除");
        await loadLlmConfigs();
        if (llmConfigForm.value.id === id) {
          startCreateLlmConfig();
        }
      } catch (e) {
        console.error(e);
        showToast("删除 LLM 配置失败: " + e.message);
      }
    }

    async function setDefaultLlmConfig(id) {
      if (!id) return;
      try {
        const cfg = llmConfigs.value.find((c) => c.id === id);
        const providerId = cfg ? cfg.provider_id || null : null;

        await fetchJson(`/api/v1/llm-configs/${id}/set-default`, {
          method: "POST",
          body: JSON.stringify({}),
        });
        showToast("默认 LLM 配置已更新");
        await loadLlmConfigs();
        await loadLlmProviders();
        if (providerId) {
          selectedLlmProviderId.value = providerId;
        }
      } catch (e) {
        console.error(e);
        showToast("设置默认 LLM 配置失败: " + e.message);
      }
    }

    async function testLlmConfig(cfg) {
      const id = cfg && cfg.id ? cfg.id : cfg;
      if (!id) return;

      if (!llmConfigTestStatus.value) {
        llmConfigTestStatus.value = {};
      }

      llmConfigTestStatus.value[id] = {
        ...(llmConfigTestStatus.value[id] || {}),
        loading: true,
      };

      try {
        const data = await fetchJson(`/api/v1/llm-configs/${id}/test`, {
          method: "POST",
          body: JSON.stringify({}),
        });
        llmConfigTestStatus.value[id] = {
          loading: false,
          success: !!data.success,
          message: data.message || "",
        };
        showToast(data.success ? "LLM 配置测试成功" : "LLM 配置测试失败");
      } catch (e) {
        console.error(e);
        llmConfigTestStatus.value[id] = {
          loading: false,
          success: false,
          message: e.message || "测试失败",
        };
        showToast("LLM 配置测试失败: " + e.message);
      }
    }

    function startCreateLlmProvider() {
      llmProviderForm.value = {
        id: null,
        name: "",
        base_url: "",
        api_key: "",
        description: "",
      };
      modals.value.createLlmProvider = true;
    }

    function startEditLlmProvider(p) {
      llmProviderForm.value = {
        id: p.id,
        name: p.name || "",
        base_url: p.base_url || "",
        api_key: p.api_key || "",
        description: p.description || "",
      };
      modals.value.createLlmProvider = true;
    }

    function closeLlmProviderModal() {
      modals.value.createLlmProvider = false;
    }

    async function submitLlmProvider() {
      if (!llmProviderForm.value.name.trim() || !llmProviderForm.value.base_url.trim()) {
        showToast("LLM 提供商名称和地址不能为空");
        return;
      }
      llmProviderSaving.value = true;
      try {
        const payload = {
          name: llmProviderForm.value.name.trim(),
          base_url: llmProviderForm.value.base_url.trim(),
          api_key: llmProviderForm.value.api_key ? llmProviderForm.value.api_key : null,
          description: llmProviderForm.value.description || null,
        };

        if (llmProviderForm.value.id) {
          await fetchJson(`/api/v1/llm-providers/${llmProviderForm.value.id}`, {
            method: "PUT",
            body: JSON.stringify(payload),
          });
          showToast("LLM 提供商已更新");
        } else {
          await fetchJson("/api/v1/llm-providers", {
            method: "POST",
            body: JSON.stringify(payload),
          });
          showToast("LLM 提供商已创建");
        }

        await loadLlmProviders();
        modals.value.createLlmProvider = false;
      } catch (e) {
        console.error(e);
        showToast("保存 LLM 提供商失败: " + e.message);
      } finally {
        llmProviderSaving.value = false;
      }
    }

    async function deleteLlmProvider(id) {
      if (!id) return;
      const usedCount = llmConfigs.value.filter((c) => c.provider_id === id).length;
      const message = usedCount
        ? `该 LLM 提供商下仍有关联的 ${usedCount} 个 LLM 配置，删除后这些配置将无法使用。确认删除？`
        : "确认删除该 LLM 提供商？";
      if (!window.confirm(message)) return;
      try {
        await fetchJson(`/api/v1/llm-providers/${id}`, { method: "DELETE", body: JSON.stringify({}) });
        showToast("LLM 提供商已删除");
        await loadLlmProviders();
      } catch (e) {
        console.error(e);
        showToast("删除 LLM 提供商失败: " + e.message);
      }
    }

    async function setDefaultLlmProvider(id) {
      if (!id) return;
      try {
        const data = await fetchJson(`/api/v1/llm-providers/${id}/set-default`, {
          method: "POST",
          body: JSON.stringify({}),
        });
        llmProviders.value = data;
        const defaultProvider = data.find((p) => p.is_default) || data[0] || null;
        selectedLlmProviderId.value = defaultProvider ? defaultProvider.id : null;
        showToast("默认 LLM 提供商已更新");
      } catch (e) {
        console.error(e);
        showToast("设置默认 LLM 提供商失败: " + e.message);
      }
    }

    function findProviderName(providerId) {
      if (!providerId) return "默认提供商";
      const p = llmProviders.value.find((x) => x.id === providerId);
      return p ? p.name : "未知提供商";
    }

    function selectLlmProvider(id) {
      selectedLlmProviderId.value = id || null;
    }

    function resetProjectWorkspaceState() {
      roles.value = [];
      segments.value = [];
      finalAudios.value = [];
      masters.value = [];
      productionSettings.value = productionDefaults();
      audioAssets.value = [];
      qualityReport.value = null;
      releasePackages.value = [];
      soundModelStatus.value = null;
      soundPlans.value = [];
      activeSoundPlanId.value = "";
      soundJobs.value = [];
      soundGenerationBusy.value = false;
      soundPreview.value = null;
      soundPreviewAudioUrl.value = "";
      clearSoundJobPolling();
      subtitles.value = [];
      llm.value.text = "";
      resetWaveforms();
      llmShowingPrompt.value = false;
      llmOriginalText.value = "";
      segPromptTemplate.value = "";
      segPromptTemplateOriginal.value = "";
      synthesisResumable.value = null;
      clearUndoHistory();
    }

    async function activateProject(id, options = {}) {
      showProjectList.value = false;
      if (!id) {
        resetProjectWorkspaceState();
        selectedProjectId.value = null;
        saveLastProjectId(null);
        refreshBatchUiFlagsForCurrentProject();
        return;
      }

      const persistCurrent = options.persistCurrent !== false;
      const previousProjectId = selectedProjectId.value;
      if (persistCurrent && previousProjectId && previousProjectId !== id) {
        await flushWaveformStatePersist(previousProjectId);
      }

      resetProjectWorkspaceState();
      selectedProjectId.value = id;
      saveLastProjectId(id);

      await Promise.all([
        loadRolesAndSegments(),
        loadProjectText(),
        loadFinalAudios(),
        loadMasters(),
        loadProductionWorkspace(),
        loadSubtitles(),
      ]);
      const restoredSynthTasks = await loadPersistentSynthTasks(id);
      await refreshAutoSegmentStatusForProject(id);
      refreshBatchUiFlagsForCurrentProject();
      await detectResumableSynthesis(id);
      if (restoredSynthTasks.some((task) => task.status === "waiting")) {
        void autoResumeQueuedSynthTasks(id, restoredSynthTasks);
      }
    }

    async function loadProjects() {
      try {
        const data = await fetchJson("/api/v1/projects");
        projects.value = data;
        if (!projects.value.length) {
          await activateProject(null, { persistCurrent: false });
          return;
        }

        // 优先使用当前 selectedProjectId（可能来自本地存储），如果不存在或无效则回退到第一个项目
        let targetId = selectedProjectId.value;
        if (!targetId || !projects.value.some((p) => p.id === targetId)) {
          targetId = projects.value[0].id;
        }

        await activateProject(targetId, { persistCurrent: false });
      } catch (e) {
        console.error(e);
        showToast("加载项目失败");
      }
    }

    async function selectProject(id) {
      if (selectedProjectId.value === id) {
        // 即便是当前项目，再次点击也顺便收起项目列表
        showProjectList.value = false;
        return;
      }
      await activateProject(id);
    }

    let autoSegmentPollTimer = null;

    async function refreshAutoSegmentStatusForProject(projectId) {
      if (!projectId) return;
      try {
        const data = await fetchJson(
          `/api/v1/projects/${projectId}/segments/auto-generate/status`,
          { headers: { "Content-Type": undefined } },
        );
        if (selectedProjectId.value !== projectId) return;
        const wasRunning = loading.value.autoSegment;
        const isRunning = !!(data && data.running);
        loading.value.autoSegment = isRunning;

        // 如果正在运行且尚未启动轮询，则启动轮询
        if (isRunning && !autoSegmentPollTimer) {
          autoSegmentPollTimer = setInterval(async () => {
            const currentProjectId = selectedProjectId.value;
            if (!currentProjectId) {
              clearInterval(autoSegmentPollTimer);
              autoSegmentPollTimer = null;
              return;
            }
            try {
              const statusData = await fetchJson(
                `/api/v1/projects/${currentProjectId}/segments/auto-generate/status`,
                { headers: { "Content-Type": undefined } },
              );
              if (selectedProjectId.value !== currentProjectId) return;
              const stillRunning = !!(statusData && statusData.running);
              loading.value.autoSegment = stillRunning;

              // 任务完成，停止轮询并刷新数据
              if (!stillRunning) {
                clearInterval(autoSegmentPollTimer);
                autoSegmentPollTimer = null;
                // 重新加载分段数据
                await loadRolesAndSegments();
                showToast("LLM 分段已完成");
              }
            } catch (e) {
              console.error("轮询 LLM 分段状态失败:", e);
            }
          }, 2000); // 每 2 秒检查一次
        }

        // 如果任务已完成且之前在运行，停止轮询
        if (!isRunning && autoSegmentPollTimer) {
          clearInterval(autoSegmentPollTimer);
          autoSegmentPollTimer = null;
        }
      } catch (e) {
        console.error(e);
      }
    }

    async function loadRolesAndSegments() {
      const projectId = selectedProjectId.value;
      if (!projectId) return;
      try {
        const [r, s] = await Promise.all([
          fetchJson(`/api/v1/projects/${projectId}/roles`),
          fetchJson(`/api/v1/projects/${projectId}/segments`, {
            headers: { "Content-Type": undefined },
          }),
        ]);
        if (selectedProjectId.value !== projectId) return;
        resetWaveforms();
        roles.value = r;

        // 若角色存在 speaker_voice_id，则确保音色库已加载，方便在角色卡片中显示“音色名+标签”
        try {
          const needVoices = Array.isArray(r) && r.some((it) => it && it.speaker_voice_id);
          if (needVoices && (!voices.value || !voices.value.length)) {
            await loadVoices();
          }
        } catch (e) {
          console.warn("加载音色库失败(角色展示用):", e);
        }

        segments.value = s;
        applyEmotionWarningsToSegments();
        segments.value.forEach((seg) => {
          const st = getWaveformState(seg.id);
          const backendState = seg.edit_state || null;
          const baseSpeed =
            typeof seg.playback_speed === "number" ? seg.playback_speed : 1.0;
          const baseVolume =
            typeof seg.volume_gain === "number" ? seg.volume_gain : 1.0;

          st.speed =
            typeof backendState?.speed === "number" &&
            !Number.isNaN(backendState.speed)
              ? backendState.speed
              : baseSpeed;
          st.volume =
            typeof backendState?.volume === "number" &&
            !Number.isNaN(backendState.volume)
              ? backendState.volume
              : baseVolume;

          // 恢复删除区间（多段 + 当前选中，用于前端 UI 展示）
          let backendRegions = Array.isArray(backendState?.regions)
            ? backendState.regions
            : [];
          // 向后兼容：旧数据只有单个 region 字段时，自动转成一条 regions 记录
          if ((!backendRegions || !backendRegions.length) && backendState?.region) {
            backendRegions = [
              {
                id: backendState.region.id || null,
                start: backendState.region.start,
                end: backendState.region.end,
              },
            ];
          }
          const normalizedRegions = backendRegions
            .map((r) => ({
              id: r.id || `${seg.id}-region-${r.start}-${r.end}`,
              start: r.start,
              end: r.end,
            }))
            .filter((r) =>
              typeof r.start === "number" &&
              typeof r.end === "number" &&
              r.end > r.start,
            );
          normalizedRegions.sort((a, b) => {
            const aStart = typeof a.start === "number" ? a.start : 0;
            const bStart = typeof b.start === "number" ? b.start : 0;
            return aStart - bStart;
          });

          st.regions = normalizedRegions;
          st.selectedRegionId =
            backendState?.selected_region_id ||
            (normalizedRegions.length ? normalizedRegions[0].id : null);

          if (st.selectedRegionId) {
            const current =
              normalizedRegions.find((r) => r.id === st.selectedRegionId) ||
              normalizedRegions[0];
            st.region = { start: current.start, end: current.end };
          } else {
            st.region = null;
          }

          st.selecting = false;
          // 从持久化状态恢复多段静音及选中状态
          const backendSilences = Array.isArray(backendState?.silences)
            ? backendState.silences
            : [];
          st.silences = backendSilences;
          st.selectedSilenceId =
            backendState?.selected_silence_id || null;
          st.pendingSilenceAt = null;
          st.pendingSilenceDuration = null;
          st.highlightSilenceTime = false;
          st.silenceSec =
            typeof backendState?.silence_sec === "number" &&
            !Number.isNaN(backendState.silence_sec)
              ? backendState.silence_sec
              : 0;
          st.fadeIn = Number(backendState?.fade_in || 0);
          st.fadeOut = Number(backendState?.fade_out || 0);
          st.leadingSilence = Number(backendState?.leading_silence || 0);
          st.trailingSilence = Number(backendState?.trailing_silence || 0);
          st.processingPreset = backendState?.processing_preset || "none";
        });
        reinitWaveformsForSegments();
      } catch (e) {
        console.error(e);
        showToast("加载角色或段落失败");
      }
    }

    async function detectResumableSynthesis(projectId) {
      if (!projectId || selectedProjectId.value !== projectId) return;
      // 如果当前项目已有正在运行的合成任务，不触发断点续传提示
      const existingTask = findLatestBatchTaskForProject(projectId);
      if (existingTask) return;

      try {
        const data = await fetchJson(
          `/api/v1/projects/${projectId}/segments/reset-stale`,
          { method: "POST" },
        );
        if (selectedProjectId.value !== projectId) return;

        const activeGenerating = Number(data.active_generating_count || 0);
        if (Number.isFinite(activeGenerating) && activeGenerating > 0) {
          synthesisResumable.value = null;
          return;
        }

        const resumable = data.pending_count + data.failed_count;
        if (resumable > 0 && data.done_count > 0) {
          // 有已完成的段落，同时还有未完成的 → 说明之前的合成被中断
          synthesisResumable.value = {
            resetCount: data.reset_count,
            pendingCount: data.pending_count,
            failedCount: data.failed_count,
            doneCount: data.done_count,
            totalCount: data.total_count,
          };
          // 如果有 generating 被重置的段落，刷新前端段落状态
          if (data.reset_count > 0) {
            (segments.value || []).forEach((seg) => {
              if (seg.status === "generating") {
                seg.status = "pending";
              }
            });
          }
        } else {
          synthesisResumable.value = null;
        }
      } catch (e) {
        console.warn("检测断点续传状态失败:", e);
      }
    }

    function dismissResumableSynthesis() {
      synthesisResumable.value = null;
    }

    async function resumeSynthesis() {
      if (!currentProject.value || !synthesisResumable.value) return;
      synthesisResumable.value = null;
      // 复用现有的批量合成逻辑（自动拾取 pending + failed 段落）
      await batchSynthesize();
    }

    async function loadProjectText() {
      const projectId = selectedProjectId.value;
      if (!projectId) return;
      try {
        const data = await fetchJson(`/api/v1/projects/${projectId}/text`, {
          headers: { "Content-Type": undefined },
        });
        if (selectedProjectId.value !== projectId) return;
        if (!llmShowingPrompt.value) {
          llm.value.text = data.text || "";
        }
      } catch (e) {
        console.error(e);
      }
    }

    function openCreateProjectModal() {
      editingProjectId.value = null;
      forms.value.project = { name: "", description: "" };
      modals.value.createProject = true;
    }

    function closeProjectModal() {
      modals.value.createProject = false;
      forms.value.project = { name: "", description: "" };
      editingProjectId.value = null;
    }

    async function submitProject() {
      if (!forms.value.project.name.trim()) {
        showToast("项目名称不能为空");
        return;
      }
      try {
        const body = JSON.stringify({
          name: forms.value.project.name.trim(),
          description: forms.value.project.description || null,
        });

        if (editingProjectId.value) {
          const p = await fetchJson(`/api/v1/projects/${editingProjectId.value}`, {
            method: "PUT",
            body,
          });
          const idx = projects.value.findIndex((x) => x.id === p.id);
          if (idx !== -1) {
            projects.value[idx] = p;
          } else {
            projects.value.push(p);
          }
          selectedProjectId.value = p.id;
          showToast("项目更新成功");
        } else {
          const p = await fetchJson("/api/v1/projects", { method: "POST", body });
          projects.value.push(p);
          await activateProject(p.id);
          showToast("项目创建成功");
        }

        closeProjectModal();
      } catch (e) {
        console.error(e);
        showToast("项目保存失败");
      }
    }

    function startEditProject() {
      if (!currentProject.value) return;
      editingProjectId.value = currentProject.value.id;
      forms.value.project = {
        name: currentProject.value.name || "",
        description: currentProject.value.description || "",
      };
      modals.value.createProject = true;
    }

    async function deleteProject(id) {
      if (!id) return;
      const target = projects.value.find((p) => p.id === id);
      const name = target && target.name ? target.name : id;
      if (
        !window.confirm(
          `将彻底删除项目「${name}」及其所有文件（文本、分段、音频等），此操作不可恢复。确认删除？`,
        )
      ) {
        return;
      }

      try {
        await fetchJson(`/api/v1/projects/${id}`, { method: "DELETE", body: JSON.stringify({}) });
        projects.value = projects.value.filter((p) => p.id !== id);

        if (selectedProjectId.value === id) {
          if (projects.value.length) {
            await activateProject(projects.value[0].id, { persistCurrent: false });
          } else {
            await activateProject(null, { persistCurrent: false });
          }
        }

        showToast("项目已删除");
      } catch (e) {
        console.error(e);
        showToast("项目删除失败: " + e.message);
      }
    }

    function closeRoleModal() {
      modals.value.createRole = false;
      forms.value.role = createRoleFormState(null, getQwen3DefaultLanguage());
      editingRoleId.value = null;
    }

    function openCreateRoleModal() {
      if (!currentProject.value) {
        showToast("请先选择项目");
        return;
      }
      editingRoleId.value = null;
      forms.value.role = createRoleFormState(null, getQwen3DefaultLanguage());
      modals.value.createRole = true;
    }

    async function submitRole() {
      if (!currentProject.value) return;
      if (!ensureProjectMutationAllowed("保存角色")) return;
      if (roleSaving.value) return;
      if (!forms.value.role.name.trim()) {
        showToast("角色名称不能为空");
        return;
      }
      roleSaving.value = true;
      try {
        const body = JSON.stringify({
          name: forms.value.role.name.trim(),
          role_type: forms.value.role.role_type,
          description: forms.value.role.description || null,
        });
        let savedRoleId = editingRoleId.value;
        let createdRole = null;
        if (editingRoleId.value) {
          await fetchJson(`/api/v1/projects/${currentProject.value.id}/roles/${editingRoleId.value}`, {
            method: "PUT",
            body,
          });
        } else {
          createdRole = await fetchJson(`/api/v1/projects/${currentProject.value.id}/roles`, {
            method: "POST",
            body,
          });
          savedRoleId = createdRole && createdRole.id;
        }

        if (!savedRoleId) {
          throw new Error("角色已保存，但响应中缺少角色 ID");
        }

        const profileBody = JSON.stringify({
          tts_backend: forms.value.role.tts_backend || null,
          qwen3_speaker: forms.value.role.qwen3_speaker || null,
          qwen3_language: forms.value.role.qwen3_language || getQwen3DefaultLanguage(),
          voice_description: (forms.value.role.voice_description || "").trim() || null,
          style_instruction: (forms.value.role.style_instruction || "").trim() || null,
          preferred_speed_min: Number(forms.value.role.preferred_speed_min || 0.9),
          preferred_speed_max: Number(forms.value.role.preferred_speed_max || 1.1),
          voice_source: (forms.value.role.voice_source || "").trim() || null,
          voice_license_name: (forms.value.role.voice_license_name || "").trim() || null,
          voice_commercial_use: Boolean(forms.value.role.voice_commercial_use),
          voice_rights_note: (forms.value.role.voice_rights_note || "").trim() || null,
        });
        try {
          await fetchJson(
            `/api/v1/projects/${currentProject.value.id}/roles/${savedRoleId}/voice-profile`,
            {
              method: "PATCH",
              body: profileBody,
            },
          );
        } catch (profileError) {
          await loadRolesAndSegments();
          if (createdRole) {
            editingRoleId.value = savedRoleId;
          }
          showToast(`角色基础信息已保存，但后端音色配置保存失败: ${profileError.message || profileError}`);
          return;
        }

        await loadRolesAndSegments();
        showToast(editingRoleId.value ? "角色更新成功" : "角色创建成功");
        closeRoleModal();
      } catch (e) {
        console.error(e);
        showToast("角色保存失败: " + e.message);
      } finally {
        roleSaving.value = false;
      }
    }

    function openUploadAudio(role) {
      currentRoleForUpload.value = role;
      modals.value.uploadAudio = true;
      Vue.nextTick(() => {
        if (audioInput.value) {
          audioInput.value.value = "";
        }
      });
    }

    function closeUploadAudioModal() {
      modals.value.uploadAudio = false;
      currentRoleForUpload.value = null;
    }

    async function submitUploadAudio() {
      if (!currentProject.value || !currentRoleForUpload.value || !audioInput.value) return;
      if (!ensureProjectMutationAllowed("更换角色参考音频")) return;
      const file = audioInput.value.files[0];
      if (!file) {
        showToast("请选择要上传的音频文件");
        return;
      }
      try {
        // 优先判断该音频是否已存在于音色库：按音频内容比对
        let matchedVoice = null;
        try {
          if (!voices.value.length) {
            await loadVoices();
          }
          const newFileHash = await computeFileHash(file);
          for (const v of voices.value || []) {
            if (!v || !v.audio_path) continue;
            try {
              const existingRes = await fetch(`/api/v1/voices/${v.id}/audio`);
              if (existingRes.ok) {
                const existingBlob = await existingRes.blob();
                const existingHash = await computeFileHash(existingBlob);
                if (newFileHash === existingHash) {
                  matchedVoice = v;
                  break;
                }
              }
            } catch (e) {
              console.warn("无法比对音频:", e);
            }
          }
        } catch (e) {
          console.warn("计算文件哈希失败:", e);
        }

        if (matchedVoice) {
          // 已存在相同音色：直接绑定该音色，而不是上传独立文件
          await fetchJson(
            `/api/v1/projects/${currentProject.value.id}/roles/${currentRoleForUpload.value.id}/bind-voice`,
            {
              method: "POST",
              body: JSON.stringify({ voice_id: matchedVoice.id }),
            },
          );
          await loadRolesAndSegments();
          showToast(`已存在相同音色：${matchedVoice.name}，已为该角色绑定此音色`);
          closeUploadAudioModal();
          return;
        }

        // 否则按原逻辑上传为角色独立参考音频
        const formData = new FormData();
        formData.append("file", file);
        const res = await fetch(
          `/api/v1/projects/${currentProject.value.id}/roles/${currentRoleForUpload.value.id}/speaker-audio`,
          {
            method: "POST",
            body: formData,
          },
        );
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.detail || res.statusText);
        }
        const data = await res.json();
        const idx = roles.value.findIndex((r) => r.id === data.id);
        if (idx !== -1) {
          roles.value[idx] = data;
        }
        await loadRolesAndSegments();
        showToast("参考音频上传成功");
        closeUploadAudioModal();
      } catch (e) {
        console.error(e);
        showToast("上传失败: " + e.message);
      }
    }

    async function importText() {
      if (!currentProject.value || !llm.value.text.trim()) return;
      if (!ensureProjectMutationAllowed("导入文本")) return;
      loading.value.importText = true;
      try {
        await fetchJson(`/api/v1/projects/${currentProject.value.id}/text/import`, {
          method: "POST",
          body: JSON.stringify({ text: llm.value.text }),
        });
        showToast("文本保存成功");
      } catch (e) {
        console.error(e);
        showToast("文本保存失败");
      } finally {
        loading.value.importText = false;
      }
    }

    async function openEditSegPromptTemplateModal() {
      if (!currentProject.value) {
        showToast("请先选择项目");
        return;
      }
      await loadSegPromptTemplate();
      // 记录打开时的原始模板内容，供“取消”还原
      segPromptTemplateOriginal.value = segPromptTemplate.value || "";
      modals.value.editSegPrompt = true;
    }

    function closeEditSegPromptTemplateModal(isCancel = false) {
      if (isCancel) {
        segPromptTemplate.value = segPromptTemplateOriginal.value || "";
      }
      modals.value.editSegPrompt = false;
    }

    async function saveSegPromptTemplate(useDefault) {
      if (!currentProject.value) {
        showToast("请先选择项目");
        return;
      }
      segPromptTemplateSaving.value = true;
      try {
        const payload = {
          content: useDefault ? null : segPromptTemplate.value,
          use_default: !!useDefault,
        };
        const data = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/segments/prompt-template`,
          {
            method: "PUT",
            body: JSON.stringify(payload),
          },
        );
        segPromptTemplate.value = data.content || "";
        // 保存成功后更新原始快照，后续“取消”不会退回到更旧版本
        segPromptTemplateOriginal.value = segPromptTemplate.value;
        // 如果当前章节文本框正在显示提示词，则用最新模板重新构建完整提示词
        if (llmShowingPrompt.value) {
          llm.value.text = buildSegmentationPromptText();
        }
        // 仅在“保存模板”（useDefault=false）时自动关闭编辑窗口
        if (!useDefault) {
          closeEditSegPromptTemplateModal();
        }
        showToast(useDefault ? "已恢复默认分段提示词模板" : "分段提示词模板已保存");
      } catch (e) {
        console.error(e);
        showToast("保存分段提示词模板失败: " + e.message);
      } finally {
        segPromptTemplateSaving.value = false;
      }
    }

    async function importSegmentsFromJson() {
      if (!currentProject.value) {
        showToast("请先选择项目");
        return;
      }
      if (!jsonImportText.value.trim()) {
        showToast("JSON 文本为空，请先粘贴或上传 JSON 脚本。");
        return;
      }
      if (!ensureProjectMutationAllowed("导入分段")) return;
      if (!confirmReplacingExistingSegments("JSON 导入")) {
        return;
      }
      pushSegmentSnapshot();
      loading.value.autoSegment = true;
      try {
        const body = { json_text: jsonImportText.value };
        const data = await fetchJson(`/api/v1/projects/${currentProject.value.id}/segments/import-json`, {
          method: "POST",
          body: JSON.stringify(body),
        });
        await loadRolesAndSegments();
        const count = (data.segments || []).length;
        const newRolesCount = typeof data.new_roles_count === "number" ? data.new_roles_count : 0;
        let msg = `JSON 导入分段完成，共 ${count} 段`;
        if (newRolesCount > 0) {
          msg += `，自动创建 ${newRolesCount} 个新角色，请在左侧为这些角色上传参考音频后再批量合成。`;
        }
        showToast(msg);
        closeImportJsonModal();
      } catch (e) {
        console.error(e);
        showToast("JSON 导入分段失败: " + e.message);
      } finally {
        loading.value.autoSegment = false;
      }
    }

    async function autoSegment() {
      const payloadText = llmShowingPrompt.value
        ? (llmOriginalText.value || llm.value.text || "")
        : (llm.value.text || "");
      if (!currentProject.value || !payloadText.trim()) return;
      if (!ensureProjectMutationAllowed("重新分段")) return;
      if (!confirmReplacingExistingSegments("LLM 重新分段")) {
        return;
      }
      pushSegmentSnapshot();

      const projectId = currentProject.value.id;
      const projectName = currentProject.value.name || projectId;
      const textLength = payloadText.length;

      const task = createLlmTask({
        projectId,
        projectName,
        textLength,
      });

      if (selectedProjectId.value === projectId) {
        loading.value.autoSegment = true;
      }

      try {
        const body = {
          text: payloadText,
          model: llm.value.model || null,
          temperature:
            typeof llm.value.temperature === "number" ? llm.value.temperature : null,
          top_p: typeof llm.value.top_p === "number" ? llm.value.top_p : null,
          llm_config_id: llm.value.configId || null,
        };
        const data = await fetchJson(
          `/api/v1/projects/${projectId}/segments/auto-generate`,
          {
            method: "POST",
            body: JSON.stringify(body),
          },
        );
        if (selectedProjectId.value === projectId) {
          await loadRolesAndSegments();
        }
        const count = (data.segments || []).length;
        const newRolesCount =
          typeof data.new_roles_count === "number" ? data.new_roles_count : 0;

        if (task) {
          task.status = "completed";
          task.total = 1;
          task.done = 1;
          task.failed = 0;
          task.totalSegments = count || 0;
          task.newRolesCount = newRolesCount;
          if (count || newRolesCount) {
            let summary = "";
            if (count) summary += `生成 ${count} 段`;
            if (newRolesCount) {
              summary += (summary ? "，" : "") + `新角色 ${newRolesCount} 个`;
            }
            task.message = summary;
          }
          // 强制触发 llmTasks 的响应式更新，避免在跨项目并发时任务状态停留在“进行中”
          llmTasks.value = [...llmTasks.value];
        }

        let msg = `LLM 分段完成，共 ${count} 段`;
        if (newRolesCount > 0) {
          msg += `，自动创建 ${newRolesCount} 个新角色，请在左侧为这些角色上传参考音频后再批量合成。`;
        }
        showToast(msg);
      } catch (e) {
        console.error(e);
        if (task) {
          task.status = "failed";
          task.failed = 1;
          task.message = (e && e.message) || "LLM 分段失败";
          // 同样在失败时刷新 llmTasks，保证任务列表和顶部状态及时反映
          llmTasks.value = [...llmTasks.value];
        }
        showToast("LLM 分段失败: " + e.message);
        // 失败时也刷新分段数据，确保前端状态与后端一致
        if (selectedProjectId.value === projectId) {
          try {
            await loadRolesAndSegments();
          } catch (refreshErr) {
            console.error("刷新分段数据失败:", refreshErr);
          }
        }
      } finally {
        if (selectedProjectId.value === projectId) {
          loading.value.autoSegment = false;
        }
      }
    }

    async function loadSegPromptTemplate() {
      if (!currentProject.value) return;
      const projectId = currentProject.value.id;
      try {
        const data = await fetchJson(
          `/api/v1/projects/${projectId}/segments/prompt-template`,
          {
            headers: { "Content-Type": undefined },
          },
        );
        if (!currentProject.value || currentProject.value.id !== projectId) return;
        segPromptTemplate.value = data.content || "";
      } catch (e) {
        console.error(e);
        showToast("加载分段提示词模板失败: " + e.message);
      }
    }

    function buildSegmentationPromptText() {
      const roleNames = roles.value.map((r) => r.name);
      const roleHint = roleNames.length
        ? roleNames.join("，")
        : "（当前项目暂未显式创建角色）";

      const userText = llmOriginalText.value || llm.value.text || "";
      let tpl = segPromptTemplate.value || "";
      let rendered = tpl.replace("<ROLE_HINT>", roleHint);

      if (rendered.includes("{{STORY_TEXT}}")) {
        rendered = rendered.replace("{{STORY_TEXT}}", userText);
      } else {
        rendered = `${rendered}\n\n<STORY_TEXT>\n${userText}\n</STORY_TEXT>\n`;
      }

      return rendered;
    }

    async function togglePromptView() {
      if (!currentProject.value) {
        showToast("请先选择项目");
        return;
      }
      if (!llmShowingPrompt.value) {
        if (!llm.value.text.trim()) {
          showToast("请输入要分段的文本后再查看提示词");
          return;
        }
        llmOriginalText.value = llm.value.text;
        await loadSegPromptTemplate();
        llm.value.text = buildSegmentationPromptText();
        llmShowingPrompt.value = true;
      } else {
        llm.value.text = llmOriginalText.value;
        llmShowingPrompt.value = false;
      }
    }

    async function savePrompt() {
      if (!currentProject.value) return;
      if (!llm.value.text.trim()) {
        showToast("当前提示词为空，无法保存");
        return;
      }
      loading.value.importText = true;
      try {
        const now = new Date();
        const pad = (n) => String(n).padStart(2, "0");
        const ts =
          now.getFullYear().toString() +
          pad(now.getMonth() + 1) +
          pad(now.getDate()) +
          "_" +
          pad(now.getHours()) +
          pad(now.getMinutes()) +
          pad(now.getSeconds());
        const filename = `seg_prompt_${ts}.txt`;
        await fetchJson(`/api/v1/projects/${currentProject.value.id}/text/import`, {
          method: "POST",
          body: JSON.stringify({ text: llm.value.text, filename }),
        });
        showToast(`提示词保存成功（保存在项目 text/${filename}）`);
      } catch (e) {
        console.error(e);
        showToast("提示词保存失败: " + e.message);
      } finally {
        loading.value.importText = false;
      }
    }

    async function batchSynthesize() {
      if (!currentProject.value || !segments.value.length) return;

      loading.value.batchSynthesize = true;
      synthesisResumable.value = null;
      const projectId = currentProject.value.id;
      const projectName = currentProject.value.name || projectId;

      try {
        const existing = findLatestBatchTaskForProject(projectId);
        if (existing) {
          showToast("当前项目已有一个批量合成任务正在队列中或执行中，请先等待其完成或在任务列表中停止。");
          return;
        }

        const plan = await fetchJson(
          `/api/v1/projects/${projectId}/segments/synthesize/plan`,
          {
            method: "POST",
            body: JSON.stringify({ retry_failed: true }),
          },
        );

        const ids = Array.isArray(plan.segment_ids) ? plan.segment_ids : [];
        if (!ids.length) {
          showToast("没有需要合成的段落");
          return;
        }

        const task = createSynthTask({
          type: "batch_synthesize",
          projectId,
          projectName,
          segmentIds: ids,
        });
        if (!task) {
          showToast("创建批量任务失败");
          return;
        }

        // 收集每个分段的 tts_backend 信息
        const segmentBackends = {};
        ids.forEach((segId) => {
          const seg = (segments.value || []).find((s) => s.id === segId);
          if (seg) {
            segmentBackends[segId] = getSegmentBackend(seg);
          }
        });
        await persistSynthTask(task, segmentBackends);
        refreshBatchUiFlagsForCurrentProject();
        showToast(`已添加批量合成任务，共 ${ids.length} 段，开始执行`);
        // 使用 WebSocket 批量合成（不再逐段 HTTP 调用）
        await runSynthBatchWS(task, ids, segmentBackends);
      } catch (e) {
        console.error(e);
        showToast("批量合成失败: " + e.message);
      } finally {
        loading.value.batchSynthesize = false;
        refreshBatchUiFlagsForCurrentProject();
      }
    }

    // ── WebSocket 批量合成 ──────────────────────────────────────────
    const activeSynthesisSockets = new Set();
    const synthesisSocketTimeouts = new Set();

    function cleanupSynthesisSockets() {
      synthesisSocketTimeouts.forEach((timeoutId) => clearTimeout(timeoutId));
      synthesisSocketTimeouts.clear();
      activeSynthesisSockets.forEach((socket) => {
        try {
          if (socket && socket.readyState < 2) socket.close();
        } catch (_error) {
          // 页面卸载期间只做尽力清理。
        }
      });
      activeSynthesisSockets.clear();
    }

    async function runSynthBatchWS(task, segmentIds, segmentBackends) {
      const projectId = task.projectId;
      const isCurrentProject = currentProject.value && currentProject.value.id === projectId;

      // 按后端分组
      const backendGroups = {};
      segmentIds.forEach((segId) => {
        const backend = segmentBackends[segId] || "indextts";
        if (!backendGroups[backend]) backendGroups[backend] = [];
        backendGroups[backend].push(segId);
      });

      task.status = "running";
      refreshBatchUiFlagsForCurrentProject();
      let hadTransportError = false;

      // 标记所有待合成段落（保留旧 audio_file 以便失败时恢复）
      if (isCurrentProject) {
        segmentIds.forEach((segId) => {
          const seg = (segments.value || []).find((s) => s.id === segId);
          if (seg) {
            seg._synthPending = true;
            seg._prevAudioFile = seg.audio_file || null;
          }
        });
      }

      for (const [backend, groupSegIds] of Object.entries(backendGroups)) {
        if (task.stopRequested) break;

        // 确保 TTS 服务运行正确的后端
        const backendNames = { indextts: "IndexTTS2", cosyvoice3: "CosyVoice3", qwen3tts: "Qwen3-TTS VoiceDesign", qwen3tts_cv: "Qwen3-TTS CustomVoice" };
        if (!ttsService.value.running || ttsService.value.backend !== backend) {
          showToast(`正在切换到 ${backendNames[backend] || backend}...`);
          await startTtsService(backend, { persistDefault: false });
          const ready = await waitForTtsServiceReady(backend, 120000);
          if (!ready) {
            const startupError = `${backendNames[backend] || backend} 服务或模型加载超时`;
            showToast(startupError);
            groupSegIds.forEach((segId) => {
              task.failed += 1;
              if (isCurrentProject) {
                const seg = (segments.value || []).find((s) => s.id === segId);
                if (seg) { seg.status = "failed"; seg._synthPending = false; }
              }
              void recordPersistentJobSegment(task, segId, false, startupError);
            });
            refreshBatchUiFlagsForCurrentProject();
            continue;
          }
          showToast(`${backendNames[backend] || backend} 已就绪`);
        }

        // 保存分段最新设置
        if (isCurrentProject) {
          for (const segId of groupSegIds) {
            const seg = (segments.value || []).find((s) => s.id === segId);
            if (seg) {
              try {
                await updateSegmentForQueue(projectId, segId, {
                  text: seg.text,
                  emotion_text: seg.emotion_text,
                  role_id: seg.role_id || null,
                  emotion_alpha: typeof seg.emotion_alpha === "number" && Number.isFinite(seg.emotion_alpha) ? seg.emotion_alpha : null,
                });
              } catch (e) {
                console.error("保存分段设置失败:", e);
              }
            }
          }
        }

        // 连接 WebSocket
        const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${wsProtocol}//${location.host}/api/v1/ws/synthesis/${projectId}`;

        try {
          let needsStatusSync = false;
          let sawBatchComplete = false;
          const completedSegIds = new Set();
          let nextSegIndex = 0;
          let inFlightSegId = null;

          const markNextSegmentGenerating = () => {
            if (!isCurrentProject || task.stopRequested) return;
            const nextSegId = groupSegIds[nextSegIndex];
            if (!nextSegId) return;
            nextSegIndex += 1;
            inFlightSegId = nextSegId;
            const seg = (segments.value || []).find((s) => s.id === nextSegId);
            if (!seg) return;
            seg.status = "generating";
            destroyWaveSurferInstance(seg.id);
            seg.audio_file = null;
          };

          await new Promise((resolve) => {
            const ws = new WebSocket(wsUrl);
            activeSynthesisSockets.add(ws);
            let resolved = false;
            let timeoutId = null;
            const finish = (closeSocket = false) => {
              if (resolved) return;
              resolved = true;
              if (timeoutId) {
                clearTimeout(timeoutId);
                synthesisSocketTimeouts.delete(timeoutId);
                timeoutId = null;
              }
              activeSynthesisSockets.delete(ws);
              if (closeSocket) {
                try {
                  if (ws.readyState < 2) ws.close();
                } catch (_error) {
                  // 连接可能已被浏览器关闭。
                }
              }
              resolve();
            };

            ws.onopen = () => {
              ws.send(JSON.stringify({
                action: "start",
                job_id: task._persistent ? task.id : null,
                segment_ids: groupSegIds,
                segment_backends: Object.fromEntries(groupSegIds.map((id) => [id, backend])),
              }));
            };

            ws.onmessage = (event) => {
              let msg;
              try { msg = JSON.parse(event.data); } catch { return; }

              if (msg.type === "batch_start") {
                markNextSegmentGenerating();
              } else if (msg.type === "segment_result") {
                const segId = msg.segment_id;
                completedSegIds.add(segId);
                if (inFlightSegId === segId) {
                  inFlightSegId = null;
                }
                if (msg.success) {
                  task.done += 1;
                  if (isCurrentProject) {
                    const seg = (segments.value || []).find((s) => s.id === segId);
                    if (seg) {
                      seg.status = "done";
                      if (msg.audio_file) seg.audio_file = msg.audio_file;
                      seg._synthPending = false;
                      if (msg.warning) {
                        seg._emotionWarning = true;
                        seg._lastErrorHint = msg.warning;
                        setEmotionWarning(segId, msg.warning);
                      } else {
                        seg._emotionWarning = false;
                        setEmotionWarning(segId, null);
                      }
                      if (seg.audio_file) {
                        Vue.nextTick(() => {
                          destroyWaveSurferInstance(seg.id);
                          initWaveformForSegment(seg);
                        });
                      }
                    }
                  }
                } else {
                  task.failed += 1;
                  if (isCurrentProject) {
                    const seg = (segments.value || []).find((s) => s.id === segId);
                    if (seg) {
                      seg.status = "failed";
                      seg._synthPending = false;
                      // 恢复合成前的音频引用
                      if (!seg.audio_file && seg._prevAudioFile) {
                        seg.audio_file = seg._prevAudioFile;
                        Vue.nextTick(() => initWaveformForSegment(seg));
                      }
                      const errMsg = msg.error || "合成失败";
                      seg._lastErrorHint = errMsg;
                      if (isEmotionRelatedError(errMsg)) {
                        seg._emotionWarning = true;
                        setEmotionWarning(segId, errMsg);
                      }
                    }
                  }
                  showToast("分段合成失败: " + (msg.error || ""));
                }
                if (isCurrentProject) applyEmotionWarningsToSegments();
                refreshBatchUiFlagsForCurrentProject();

                // 检查是否需要停止
                if (task.stopRequested) {
                  try { ws.send(JSON.stringify({ action: "stop" })); } catch {}
                } else {
                  markNextSegmentGenerating();
                }
              } else if (msg.type === "batch_complete") {
                sawBatchComplete = true;
                finish(true);
              } else if (msg.type === "error") {
                needsStatusSync = true;
                showToast("WebSocket 合成错误: " + (msg.message || ""));
                finish(true);
              }
            };

            ws.onerror = () => {
              console.error("WebSocket synthesis connection error");
              needsStatusSync = true;
              finish(true);
            };
            ws.onclose = () => {
              if (!sawBatchComplete) {
                needsStatusSync = true;
              }
              finish();
            };

            // 安全超时：5分钟 × 段落数，最少 60 秒
            const timeoutMs = Math.max(60000, groupSegIds.length * 300000);
            timeoutId = setTimeout(() => {
              if (!resolved) {
                needsStatusSync = true;
                finish(true);
              }
            }, timeoutMs);
            synthesisSocketTimeouts.add(timeoutId);
          });

          if (!sawBatchComplete && needsStatusSync) {
            hadTransportError = true;
          }

          if (isCurrentProject) {
            // Always refresh canonical Take/slot metadata after a group.  The
            // WebSocket result only contains a compact progress payload.
            if (needsStatusSync || completedSegIds.size > 0) {
              try {
                await refreshSegmentsStatus(true);
              } catch (e) {
                console.error("刷新分段状态失败:", e);
              }
            }

            groupSegIds.forEach((segId) => {
              if (completedSegIds.has(segId)) return;
              const seg = (segments.value || []).find((s) => s.id === segId);
              if (!seg) return;
              seg._synthPending = false;
              if (seg.status === "generating" && (!needsStatusSync || segId !== inFlightSegId)) {
                seg.status = "pending";
              }
              if (!seg.audio_file && seg._prevAudioFile && seg.status !== "done" && seg.status !== "generating") {
                seg.audio_file = seg._prevAudioFile;
                Vue.nextTick(() => initWaveformForSegment(seg));
              }
            });
          }
        } catch (e) {
          console.error("WebSocket batch synthesis error:", e);
          hadTransportError = true;
          // 将未完成的段落标记为失败并恢复音频引用
          groupSegIds.forEach((segId) => {
            if (isCurrentProject) {
              const seg = (segments.value || []).find((s) => s.id === segId);
              if (seg && seg.status === "generating") {
                seg.status = "failed";
                seg._synthPending = false;
                if (!seg.audio_file && seg._prevAudioFile) {
                  seg.audio_file = seg._prevAudioFile;
                  Vue.nextTick(() => initWaveformForSegment(seg));
                }
                task.failed += 1;
              }
            }
          });
        }

        try { await refreshHealth(); } catch {}
      }

      // 更新任务最终状态
      if (task.stopRequested) {
        task.status = "stopped";
      } else if (task.failed > 0 || hadTransportError) {
        task.status = "completed_with_errors";
      } else {
        task.status = "completed";
      }
      refreshBatchUiFlagsForCurrentProject();
    }

    async function runSynthQueue() {
      if (synthQueueRunning) return;
      synthQueueRunning = true;
      try {
        while (synthQueue.value.length) {
          const job = synthQueue.value.shift();
          if (!job || !job.segmentId || !job.projectId) {
            continue;
          }

          const task = (synthTasks.value || []).find((t) => t && t.id === job.taskId) || null;
          if (task && task.stopRequested) {
            const hasMoreForTask = synthQueue.value.some((j) => j && j.taskId === job.taskId);
            if (!hasMoreForTask && task.status === "waiting") {
              task.status = "stopped";
            }
            continue;
          }

          const isCurrentProject =
            currentProject.value && currentProject.value.id === job.projectId;
          let seg = null;
          if (isCurrentProject) {
            seg = (segments.value || []).find((s) => s.id === job.segmentId) || null;
            if (seg) {
              seg._synthPending = true;
              seg._prevAudioFile = seg.audio_file || null;
              seg.status = 'generating';  // 立即更新状态列显示
              // 立即清理旧波形和音频路径
              destroyWaveSurferInstance(seg.id);
              seg.audio_file = null;
            }
          }

          if (task && task.status === "waiting") {
            task.status = "running";
            refreshBatchUiFlagsForCurrentProject();
          }

          // 根据分段配置自动启动对应的 TTS 后端
          const segBackend = job.ttsBackend || (seg ? getSegmentBackend(seg) : null) || 'indextts';
          if (!ttsService.value.running || ttsService.value.backend !== segBackend) {
            const backendNames = { indextts: 'IndexTTS2', cosyvoice3: 'CosyVoice3', qwen3tts: 'Qwen3-TTS VoiceDesign', qwen3tts_cv: 'Qwen3-TTS CustomVoice' };
            showToast(`正在切换到 ${backendNames[segBackend] || segBackend}...`);
            // 后端会自动停止旧服务，直接启动新服务即可
            await startTtsService(segBackend, { persistDefault: false });
            // 等待服务就绪（包括模型加载）
            const ready = await waitForTtsServiceReady(segBackend, 120000);
            if (!ready) {
              showToast(`${backendNames[segBackend] || segBackend} 服务或模型加载超时`);
              if (seg) {
                seg._synthPending = false;
                seg.status = 'failed';
              }
              if (task) {
                task.failed += 1;
                try {
                  await recordPersistentJobSegment(
                    task,
                    job.segmentId,
                    false,
                    `${backendNames[segBackend] || segBackend} 服务或模型加载超时`,
                  );
                } catch (e) {
                  console.error("保存任务失败结果失败:", e);
                }
              }
              continue;
            }
            showToast(`${backendNames[segBackend] || segBackend} 已就绪`);
          }

          if (isCurrentProject && seg) {
            try {
              const savePayload = {
                text: seg.text,
                emotion_text: seg.emotion_text,
                role_id: seg.role_id || null,
                emotion_alpha:
                  typeof seg.emotion_alpha === "number" &&
                  Number.isFinite(seg.emotion_alpha)
                    ? seg.emotion_alpha
                    : null,
              };
              await updateSegmentForQueue(job.projectId, job.segmentId, savePayload);
            } catch (e) {
              console.error(e);
              seg._synthPending = false;
              seg.status = 'failed';
              showToast("保存分段最新设置失败: " + e.message);
              if (task) {
                task.failed += 1;
                try {
                  await recordPersistentJobSegment(task, job.segmentId, false, e.message || "保存分段设置失败");
                } catch (persistError) {
                  console.error("保存任务失败结果失败:", persistError);
                }
              }
              const hasMoreForTask = synthQueue.value.some((j) => j && j.taskId === job.taskId);
              if (task && !hasMoreForTask) {
                if (task.stopRequested) {
                  task.status = "stopped";
                } else if (task.failed > 0) {
                  task.status = "completed_with_errors";
                } else {
                  task.status = "completed";
                }
              }
              refreshBatchUiFlagsForCurrentProject();
              continue;
            }
          }

          let success = false;

          try {
            let emotionWarningHint = null;
            if (seg) {
              seg.status = "generating";
            }
            const data = await fetchJson(
              `/api/v1/projects/${job.projectId}/segments/synthesize`,
              {
                method: "POST",
                body: JSON.stringify({
                  segment_ids: [job.segmentId],
                  retry_failed: true,
                  job_id: task && task._persistent ? task.id : null,
                }),
              },
            );
            const results = data.results || [];
            const r = results[0];
            const targetSegId = r && r.segment_id ? r.segment_id : job.segmentId;
            let target = null;
            if (isCurrentProject) {
              target = (segments.value || []).find((s) => s.id === targetSegId) || null;
            }

            if (r && r.success) {
              success = true;
              if (target) {
                target.status = "done";
                if (r.audio_file) {
                  target.audio_file = r.audio_file;
                }
                if (r.warning) {
                  emotionWarningHint = r.warning;
                  target._emotionWarning = true;
                  target._lastErrorHint = r.warning;
                  setEmotionWarning(targetSegId, r.warning);
                } else {
                  target._emotionWarning = false;
                  setEmotionWarning(targetSegId, null);
                }
              } else if (isCurrentProject && r && r.warning) {
                setEmotionWarning(targetSegId, r.warning);
              } else if (isCurrentProject && r) {
                setEmotionWarning(targetSegId, null);
              }

              if (isCurrentProject) {
                applyEmotionWarningsToSegments();
              }

              if (target && target.audio_file) {
                Vue.nextTick(() => {
                  destroyWaveSurferInstance(target.id);
                  initWaveformForSegment(target);
                });
              }

              showToast("分段合成完成");
            } else {
              if (target) {
                target.status = "failed";
                // 恢复合成前的音频引用
                if (!target.audio_file && target._prevAudioFile) {
                  target.audio_file = target._prevAudioFile;
                  Vue.nextTick(() => initWaveformForSegment(target));
                }
              }
              const baseErr = normalizeErrorMessage((r && (r.error || r.detail)) || "合成失败");
              let uiErr = baseErr;
              const isEmotionError = isEmotionRelatedError(baseErr);
              if (target) {
                if (isEmotionError) {
                  target._emotionWarning = true;
                  target._lastErrorHint = baseErr;
                  setEmotionWarning(target.id, baseErr);
                } else {
                  target._emotionWarning = false;
                  target._lastErrorHint = baseErr || "";
                  setEmotionWarning(target.id, null);
                }
              } else if (isCurrentProject && isEmotionError && r && r.segment_id) {
                setEmotionWarning(r.segment_id, baseErr);
              }
              if (isCurrentProject) {
                if (isEmotionError) {
                  uiErr = withEmotionHint(baseErr);
                }

                applyEmotionWarningsToSegments();
              }
              showToast("分段合成失败: " + uiErr);
            }
          } catch (e) {
            console.error(e);
            let target = null;
            if (isCurrentProject) {
              target = (segments.value || []).find((s) => s.id === job.segmentId) || null;
            }
            let msg = normalizeErrorMessage(e);
            const isEmotionError = isEmotionRelatedError(msg);
            if (target) {
              if (isEmotionError) {
                target._emotionWarning = true;
                target._lastErrorHint = msg;
                setEmotionWarning(target.id, msg);
              } else {
                target._emotionWarning = false;
                target._lastErrorHint = msg || "";
                setEmotionWarning(target.id, null);
              }
            } else if (isCurrentProject && isEmotionError) {
              setEmotionWarning(job.segmentId, msg);
            }
            if (isEmotionError) {
              msg = withEmotionHint(msg);
            }

            if (isCurrentProject) {
              applyEmotionWarningsToSegments();
            }
            showToast("分段合成失败: " + msg);
          } finally {
            try {
              await refreshHealth();
            } catch (e) {
              console.error(e);
            }
            // 无论成功还是失败，合成完成后都要清除 _synthPending 标志
            if (seg) {
              seg._synthPending = false;
            }
          }

          if (task) {
            if (success) {
              task.done += 1;
            } else {
              task.failed += 1;
            }
            const hasMoreForTask = synthQueue.value.some((j) => j && j.taskId === job.taskId);
            if (!hasMoreForTask) {
              if (task.stopRequested) {
                task.status = "stopped";
              } else if (task.failed > 0) {
                task.status = "completed_with_errors";
              } else {
                task.status = "completed";
              }
            }
          }

          refreshBatchUiFlagsForCurrentProject();
        }
      } finally {
        synthQueueRunning = false;
        refreshBatchUiFlagsForCurrentProject();
      }
    }

    function isSegmentSynthRunning(seg) {
      return !!(seg && seg._synthPending);
    }

    // 检查某个分段是否正在播放（波形、单独试听或连续试听）
    function isSegmentPlaying(seg) {
      if (!seg) return false;
      const segId = seg.id;
      // 引用 activePlaySegId 建立 Vue 响应式依赖，
      // 使得播放/暂停/结束时 Vue 能感知变化并重新渲染按钮
      const _playing = activePlaySegId.value;
      // 检查波形播放
      const ws = wavesurferMap[segId];
      if (ws && ws.isPlaying && ws.isPlaying()) return true;
      // 检查单独试听
      if (singlePreviewPlaying.value && singlePreviewSegmentId.value === segId) return true;
      // 检查连续试听
      if (sequentialPreviewRunning.value && sequentialPreviewCurrentSegmentId.value === segId) return true;
      return false;
    }

    function isSegmentInQueueWaiting(seg) {
      if (!seg || !currentProject.value) return false;
      const projectId = currentProject.value.id;
      const segId = seg.id;
      const tasksById = new Map();
      (synthTasks.value || []).forEach((t) => {
        if (t && t.id) {
          tasksById.set(t.id, t);
        }
      });
      return (synthQueue.value || []).some((job) => {
        if (!job || job.projectId !== projectId || job.segmentId !== segId) return false;
        const task = tasksById.get(job.taskId);
        if (!task) return false;
        return task.type === "single_synthesize" && task.status === "waiting";
      });
    }

    function cancelSegmentSynthesis(seg) {
      if (!seg || !currentProject.value) return;
      const projectId = currentProject.value.id;
      const segId = seg.id;
      if (seg._synthPending) {
        // 正在合成中的分段不支持取消
        return;
      }
      const tasksById = new Map();
      (synthTasks.value || []).forEach((t) => {
        if (t && t.id) {
          tasksById.set(t.id, t);
        }
      });
      const originalQueue = synthQueue.value || [];
      const remainingJobs = [];
      const affectedTaskIds = new Set();
      originalQueue.forEach((job) => {
        if (!job || job.projectId !== projectId || job.segmentId !== segId) {
          remainingJobs.push(job);
          return;
        }
        const task = tasksById.get(job.taskId);
        if (!task || task.type !== "single_synthesize" || task.status !== "waiting") {
          remainingJobs.push(job);
          return;
        }
        affectedTaskIds.add(task.id);
      });
      if (!affectedTaskIds.size) {
        showToast("没有可取消的单段合成任务");
        return;
      }
      synthQueue.value = remainingJobs;
      const tasks = synthTasks.value || [];
      const filteredTasks = tasks.filter((task) => {
        if (!task || !affectedTaskIds.has(task.id)) return true;
        const stillHasJobs = synthQueue.value.some((job) => job && job.taskId === task.id);
        return stillHasJobs;
      });
      synthTasks.value = filteredTasks;
      showToast("已取消该段的合成任务");
    }

    async function onClickSegmentSynthesize(seg) {
      if (!seg) return;
      if (isSegmentSynthRunning(seg)) {
        return;
      }
      if (isSegmentInQueueWaiting(seg)) {
        cancelSegmentSynthesis(seg);
        return;
      }
      synthesizeSegment(seg);
    }

    async function synthesizeSegment(seg) {
      if (!currentProject.value || !seg) return;
      if (seg._synthPending) return;
      // 如果当前分段已存在音频，则在重新生成前给出确认提示
      if (seg.audio_file) {
        const ok = window.confirm("当前分段已存在配音，确认要重新生成此分段的配音吗？");
        if (!ok) return;
      }
      
      const projectId = currentProject.value.id;
      const projectName = currentProject.value.name || projectId;
      const segId = seg.id;

      const task = createSynthTask({
        type: "single_synthesize",
        projectId,
        projectName,
        segmentIds: [segId],
      });
      if (!task) {
        showToast("创建合成任务失败");
        return;
      }
      const segmentBackends = { [segId]: getSegmentBackend(seg) };
      try {
        await persistSynthTask(task, segmentBackends);
      } catch (e) {
        synthTasks.value = (synthTasks.value || []).filter((item) => item.id !== task.id);
        showToast("创建持久合成任务失败: " + e.message);
        return;
      }
      enqueueSynthJobs(task, [segId], segmentBackends);
      refreshBatchUiFlagsForCurrentProject();
      runSynthQueue();
    }

    // ── 段落 A/B 对比（内联切换）──────────────────────────────────
    // 每个段落有独立 A/B 槽位。切换只改变当前视图，底层 Take 不交换。

    // 纯 UI 指示：当前是否为 B 版本（通过后端 active_version 判断）
    function isAbMode(seg) {
      return seg && (seg.active_version === "b");
    }

    async function toggleAbView(seg) {
      if (!seg || !currentProject.value) return;
      if (!ensureProjectMutationAllowed("切换 A/B 版本")) return;
      pushSegmentSnapshot();
      try {
        const nextSlot = isAbMode(seg) ? "a" : "b";
        const updated = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/segments/${seg.id}/active-version/${nextSlot}`,
          { method: "PUT" },
        );
        const idx = segments.value.findIndex((s) => s.id === updated.id);
        if (idx !== -1) {
          Object.assign(segments.value[idx], updated);
        }
        // audio_file 是当前槽位的兼容投影，底层 Take 保持不变。
        const realSeg = idx !== -1 ? segments.value[idx] : seg;
        Vue.nextTick(() => {
          destroyWaveSurferInstance(realSeg.id);
          if (realSeg.audio_file) {
            initWaveformForSegment(realSeg);
          }
        });
      } catch (e) {
        console.error("切换 A/B 失败:", e);
        showToast("切换失败: " + e.message);
      }
    }

    function getCurrentTakeId(seg) {
      if (!seg || !seg.take_slots) return null;
      const slot = isAbMode(seg) ? "b" : "a";
      return seg.take_slots[slot] ? seg.take_slots[slot].take_id : null;
    }

    function isCurrentTakeSelected(seg) {
      const takeId = getCurrentTakeId(seg);
      return !!takeId && seg.selected_take_id === takeId;
    }

    function archivedTakeCount(seg) {
      if (!seg || !Array.isArray(seg.takes)) return 0;
      const slotIds = new Set(
        Object.values(seg.take_slots || {}).map((slot) => slot && slot.take_id).filter(Boolean),
      );
      return seg.takes.filter((take) => take && !slotIds.has(take.id)).length;
    }

    async function selectCurrentTake(seg) {
      if (!currentProject.value || !seg) return;
      const takeId = getCurrentTakeId(seg);
      if (!takeId) {
        showToast("当前版本还没有可采用的音频");
        return;
      }
      try {
        const updated = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/segments/${seg.id}/selected-take`,
          { method: "PUT", body: JSON.stringify({ take_id: takeId }) },
        );
        const idx = segments.value.findIndex((item) => item.id === updated.id);
        if (idx !== -1) Object.assign(segments.value[idx], updated);
        showToast(`已采用 ${isAbMode(updated) ? "B" : "A"} 版本用于最终合成`);
      } catch (e) {
        console.error("选择最终 Take 失败:", e);
        showToast("选择版本失败: " + e.message);
      }
    }

    function takeLocationLabel(take) {
      const seg = (segments.value || []).find((item) => item.id === takeHistoryState.value.segmentId);
      if (!seg || !take) return "历史";
      if (seg.take_slots?.a?.take_id === take.id) return "A";
      if (seg.take_slots?.b?.take_id === take.id) return "B";
      return "历史";
    }

    function takeAudioUrl(take) {
      if (!currentProject.value || !takeHistoryState.value.segmentId || !take) return "";
      return `/api/v1/projects/${currentProject.value.id}/segments/${takeHistoryState.value.segmentId}/takes/${take.id}/audio`;
    }

    async function openTakeHistory(seg) {
      if (!currentProject.value || !seg) return;
      takeHistoryState.value = { segmentId: seg.id, items: [], loading: true };
      modals.value.takeHistory = true;
      try {
        const items = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/segments/${seg.id}/takes`,
        );
        takeHistoryState.value.items = Array.isArray(items) ? items : [];
      } catch (e) {
        showToast("加载 Take 历史失败: " + e.message);
      } finally {
        takeHistoryState.value.loading = false;
      }
    }

    function closeTakeHistory() {
      modals.value.takeHistory = false;
      takeHistoryState.value = { segmentId: null, items: [], loading: false };
    }

    async function restoreHistoryTake(take, slot) {
      if (!currentProject.value || !takeHistoryState.value.segmentId || !take) return;
      try {
        const segmentId = takeHistoryState.value.segmentId;
        const updated = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/segments/${segmentId}/takes/${take.id}/restore`,
          { method: "POST", body: JSON.stringify({ slot }) },
        );
        const idx = segments.value.findIndex((item) => item.id === updated.id);
        if (idx !== -1) Object.assign(segments.value[idx], updated);
        await openTakeHistory(updated);
        showToast(`已将历史 Take 恢复到 ${slot.toUpperCase()}`);
      } catch (e) {
        showToast("恢复 Take 失败: " + e.message);
      }
    }

    async function toggleTakePin(take) {
      if (!currentProject.value || !takeHistoryState.value.segmentId || !take) return;
      try {
        const segmentId = takeHistoryState.value.segmentId;
        const updated = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/segments/${segmentId}/takes/${take.id}/pin`,
          { method: "PUT", body: JSON.stringify({ pinned: !take.pinned }) },
        );
        const idx = segments.value.findIndex((item) => item.id === updated.id);
        if (idx !== -1) Object.assign(segments.value[idx], updated);
        await openTakeHistory(updated);
      } catch (e) {
        showToast("更新 Take 保留状态失败: " + e.message);
      }
    }

    async function clearAlt(seg) {
      if (!currentProject.value || !seg) return;
      if (!ensureProjectMutationAllowed("清除 B 版本")) return;
      pushSegmentSnapshot();
      try {
        const updated = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/segments/${seg.id}/alt`,
          { method: "DELETE" },
        );
        const idx = segments.value.findIndex((s) => s.id === updated.id);
        if (idx !== -1) {
          Object.assign(segments.value[idx], updated);
        }
        showToast("B 版本已移入历史记录");
        const realSeg = idx !== -1 ? segments.value[idx] : seg;
        Vue.nextTick(() => {
          destroyWaveSurferInstance(realSeg.id);
          if (realSeg.audio_file) initWaveformForSegment(realSeg);
        });
      } catch (e) {
        console.error("清除 B 版本失败:", e);
        showToast("清除失败: " + e.message);
      }
    }

    async function refreshSegmentsStatus(silent = false) {
      if (!currentProject.value) return;
      try {
        const data = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/segments/refresh-status`,
          {
            method: "POST",
            headers: { "Content-Type": undefined },
          },
        );
        if (!Array.isArray(data)) {
          showToast("更新状态失败: 响应格式不正确");
          return;
        }

        // 仅根据后端最新结果同步 status/audio_file 等字段，保留本地波形编辑状态
        const map = new Map();
        data.forEach((seg) => {
          if (seg && seg.id) {
            map.set(seg.id, seg);
          }
        });

        segments.value = (segments.value || []).map((oldSeg) => {
          const updated = map.get(oldSeg.id);
          if (!updated) return oldSeg;
          return {
            ...oldSeg,
            ...updated,
          };
        });

        if (!silent) showToast("段落状态已更新");
      } catch (e) {
        console.error(e);
        showToast("更新状态失败: " + e.message);
      }
    }

    async function batchUpdateEmotionAlpha() {
      if (!currentProject.value || !segments.value.length) return;
      if (!ensureProjectMutationAllowed("批量修改情感权重")) return;
      
      // 获取当前默认值作为提示
      const currentDefault = ttsParams.value.emotion_alpha || 0.65;
      const input = prompt(`请输入新的情感权重（0-1 之间）：\n\n当前配置默认值: ${currentDefault}\n将应用到所有 ${segments.value.length} 个分段`, String(currentDefault));
      
      if (input === null) return; // 用户取消
      
      const newAlpha = parseFloat(input);
      if (isNaN(newAlpha) || newAlpha < 0 || newAlpha > 1) {
        showToast("情感权重必须是 0-1 之间的数字");
        return;
      }
      pushSegmentSnapshot();
      
      try {
        const data = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/segments/batch-update-emotion-alpha`,
          {
            method: "POST",
            body: JSON.stringify({ emotion_alpha: newAlpha }),
          },
        );
        
        if (Array.isArray(data)) {
          segments.value = data;
          showToast(`已将所有分段的情感权重修改为 ${newAlpha}`);
        } else {
          showToast("批量修改失败: 响应格式不正确");
        }
      } catch (e) {
        console.error(e);
        showToast("批量修改情感权重失败: " + e.message);
      }
    }

    function stopBatchSynthesize() {
      if (!loading.value.batchSynthesize || !showStopBatchButton.value || stoppingBatchSynthesize.value) return;
      const projectId = currentProject.value && currentProject.value.id;
      if (projectId) {
        const task = findLatestBatchTaskForProject(projectId);
        if (task) {
          stopSynthTask(task.id);
          showToast("已请求停止批量合成，当前段落完成后停止");
          return;
        }
      }
      // 兜底逻辑：未找到对应任务时，回退到旧的停止方式
      shouldStopBatchSynthesize.value = true;
      stoppingBatchSynthesize.value = true;
      showToast("已请求停止批量合成，当前段落完成后停止");
    }

    async function updateSegment(segId, payload) {
      if (!currentProject.value || !segId) return null;
      if (!ensureProjectMutationAllowed("修改分段")) return null;
      pushSegmentSnapshot();
      const body = JSON.stringify(payload || {});
      const updated = await fetchJson(`/api/v1/projects/${currentProject.value.id}/segments/${segId}`, {
        method: "PATCH",
        body,
      });
      const idx = segments.value.findIndex((s) => s.id === updated.id);
      if (idx !== -1) {
        const oldSeg = segments.value[idx];
        if (oldSeg && typeof oldSeg === "object") {
          Object.assign(oldSeg, updated);
        } else {
          segments.value[idx] = updated;
        }
      }
      return updated;
    }

    async function updateSegmentForQueue(projectId, segId, payload) {
      if (!projectId || !segId) return null;
      const body = JSON.stringify(payload || {});
      const updated = await fetchJson(`/api/v1/projects/${projectId}/segments/${segId}`, {
        method: "PATCH",
        body,
      });
      if (currentProject.value && currentProject.value.id === projectId) {
        const idx = segments.value.findIndex((s) => s.id === updated.id);
        if (idx !== -1) {
          const oldSeg = segments.value[idx];
          if (oldSeg && typeof oldSeg === "object") {
            Object.assign(oldSeg, updated);
          } else {
            segments.value[idx] = updated;
          }
        }
      }
      return updated;
    }

    function formatEmotionAlpha(value) {
      let v = null;
      if (typeof value === "number" && Number.isFinite(value)) {
        v = value;
      } else {
        const parsed = parseFloat(value);
        if (Number.isFinite(parsed)) {
          v = parsed;
        }
      }
      if (v === null) {
        v = 0.65;
      }
      if (v < 0) v = 0;
      if (v > 1) v = 1;
      return v.toFixed(2);
    }

    function formatEmotionVector(vec) {
      if (!Array.isArray(vec) || vec.length !== 8) return "";
      const parts = [];
      for (let i = 0; i < 8; i += 1) {
        const raw = vec[i];
        let v = null;
        if (typeof raw === "number" && Number.isFinite(raw)) {
          v = raw;
        } else if (raw !== null && raw !== undefined && String(raw).trim()) {
          const parsed = parseFloat(String(raw).trim());
          if (Number.isFinite(parsed)) v = parsed;
        }
        if (v === null) v = 0;
        if (v < 0) v = 0;
        if (v > 1) v = 1;
        parts.push(v.toFixed(2));
      }
      return `[${parts.join(", ")}]`;
    }

    function findVoiceByAudioPath(audioPath) {
      if (!audioPath) return null;
      const audioPathStr = String(audioPath);
      const audioBase = audioPathStr.split(/[\\/]/).pop();
      const vs = voices.value || [];
      return (
        vs.find((v) => {
          if (!v || !v.audio_path) return false;
          const voicePath = String(v.audio_path);
          const voiceBase = voicePath.split(/[\\/]/).pop();
          return (
            voicePath === audioPathStr ||
            voicePath.endsWith(audioPathStr) ||
            audioPathStr.endsWith(voicePath) ||
            (audioBase && voiceBase && audioBase === voiceBase)
          );
        }) || null
      );
    }

    function getEmotionAudioLabel(seg) {
      if (!seg || !seg.emotion_audio) return "——";
      const audioPath = seg.emotion_audio;
      const match = findVoiceByAudioPath(audioPath);
      if (match) {
        return match.name || match.audio_path || "——";
      }
      return audioPath;
    }

    function getVoiceNameById(id) {
      if (!id) return "";
      const vs = voices.value || [];
      const v = vs.find((x) => x && x.id === id);
      if (!v) return "";
      return v.name || v.audio_path || id;
    }

    // 计算文件的简单哈希（用于音频比对）
    async function computeFileHash(file) {
      const buffer = await file.arrayBuffer();
      const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    }

    async function uploadSegmentEmotionAudio() {
      // 只使用上传选择的文件（区域2的按钮只处理区域2的文件）
      const file = segEmotionModal.value.selectedFile || 
        (segEmotionUploadInput.value && segEmotionUploadInput.value.files ? segEmotionUploadInput.value.files[0] : null);
      if (!file) {
        showToast("请先选择要上传的音频文件");
        return;
      }
      try {
        const rawName = file.name || "情感参考音频";
        const dot = rawName.lastIndexOf(".");
        const baseName = dot > 0 ? rawName.slice(0, dot) : rawName;
        let voiceName = baseName || "情感参考音频";

        // 计算当前文件的哈希
        const newFileHash = await computeFileHash(file);
        
        // 检查所有音色，查找是否有相同音频
        let matchedVoice = null;
        for (const v of (voices.value || [])) {
          if (!v || !v.audio_path) continue;
          try {
            const existingRes = await fetch(`/api/v1/voices/${v.id}/audio`);
            if (existingRes.ok) {
              const existingBlob = await existingRes.blob();
              const existingHash = await computeFileHash(existingBlob);
              if (newFileHash === existingHash) {
                matchedVoice = v;
                break;
              }
            }
          } catch (e) {
            console.warn("无法比对音频:", e);
          }
        }

        if (matchedVoice) {
          // 找到相同音频，直接使用已有音色
          segEmotionModal.value.audioVoiceId = matchedVoice.id;
          segEmotionModal.value.voiceSelectId = matchedVoice.id;
          segEmotionModal.value.audioSource = 'voice';
          showToast(`已存在相同音频的音色：${matchedVoice.name}`);
          return;
        }

        // 检查是否已存在同名音色，如果有则添加时间戳
        const existingVoice = (voices.value || []).find(v => v && v.name === voiceName);
        if (existingVoice) {
          const now = new Date();
          const timeStr = `${now.getHours().toString().padStart(2,'0')}${now.getMinutes().toString().padStart(2,'0')}${now.getSeconds().toString().padStart(2,'0')}`;
          voiceName = `${baseName}-${timeStr}`;
        }

        const payload = {
          name: voiceName,
          tags: [voiceTypeLabels.emotion],
          description: "上传的情感参考音频",
        };

        let voiceData = await fetchJson("/api/v1/voices", {
          method: "POST",
          body: JSON.stringify(payload),
        });

        const formData = new FormData();
        formData.append("file", file);
        const res = await fetch(`/api/v1/voices/${voiceData.id}/audio`, {
          method: "POST",
          body: formData,
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.detail || res.statusText || "上传失败");
        }
        voiceData = await res.json();

        try {
          await loadVoices();
        } catch (e) {
          console.error(e);
        }

        segEmotionModal.value.audioVoiceId = voiceData.id;
        segEmotionModal.value.voiceSelectId = voiceData.id;
        segEmotionModal.value.audioSource = 'voice';
        showToast(`情感参考音频已上传到音色库：${voiceName}`);
      } catch (e) {
        console.error(e);
        showToast("上传情感参考音频失败: " + e.message);
      }
    }

    async function uploadSegmentEmotionAudioToSegment() {
      // 只使用上传选择的文件（区域2的按钮只处理区域2的文件）
      const file = segEmotionModal.value.selectedFile || 
        (segEmotionUploadInput.value && segEmotionUploadInput.value.files ? segEmotionUploadInput.value.files[0] : null);
      if (!file) {
        showToast("请先选择要上传的音频文件");
        return;
      }

      if (!currentProject.value || !segEmotionModal.value.id) {
        showToast("当前项目或分段信息缺失，无法上传");
        return;
      }

      try {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch(
          `/api/v1/projects/${currentProject.value.id}/segments/${segEmotionModal.value.id}/emotion-audio`,
          {
            method: "POST",
            body: formData,
          },
        );
        if (!res.ok) {
          let detail = "";
          try {
            const cloned = res.clone();
            try {
              const data = await cloned.json();
              detail = data.detail || JSON.stringify(data);
            } catch {
              detail = await res.text();
            }
          } catch {
            // 响应体不可读时忽略
          }
          throw new Error(detail || `请求失败 (${res.status})`);
        }

        const updatedSeg = await res.json();

        // 更新 segments 数组中的对应分段
        const idx = segments.value.findIndex((s) => s.id === updatedSeg.id);
        if (idx !== -1) {
          Object.assign(segments.value[idx], updatedSeg);
        }

        // 更新模态框状态
        segEmotionModal.value.mode = "audio";
        segEmotionModal.value.savedAudioPath = updatedSeg.emotion_audio || null;
        segEmotionModal.value.audioSource = 'saved';
        segEmotionModal.value.audioVersion = (segEmotionModal.value.audioVersion || 0) + 1;

        showToast("情感参考音频已上传到当前分段");
      } catch (e) {
        console.error(e);
        showToast("上传情感参考音频失败: " + e.message);
      }
    }

    async function onSegmentEmotionAlphaChange(seg, raw) {
      if (!seg) return;
      let v = null;
      if (typeof raw === "number" && Number.isFinite(raw)) {
        v = raw;
      } else {
        const parsed = parseFloat(raw);
        if (Number.isFinite(parsed)) {
          v = parsed;
        }
      }
      if (v === null) {
        showToast("请输入 0-1 之间的数字作为情感权重");
        return;
      }
      if (v < 0) v = 0;
      if (v > 1) v = 1;
      try {
        const updated = await updateSegment(seg.id, { emotion_alpha: v });
        if (updated) {
          showToast(updated.audio_file ? "情感权重已更新，重新合成后生效" : "情感权重已更新");
        }
      } catch (e) {
        console.error(e);
        showToast("更新情感权重失败: " + e.message);
      }
    }

    async function onSegmentTtsBackendChange(seg, value) {
      if (!seg) return;
      const newBackend = value || 'indextts';
      try {
        const updated = await updateSegment(seg.id, { tts_backend: newBackend });
        if (updated) {
          const backendNames = { indextts: 'IndexTTS2', cosyvoice3: 'CosyVoice3', qwen3tts: 'Qwen3-TTS VoiceDesign', qwen3tts_cv: 'Qwen3-TTS CustomVoice' };
          const suffix = updated.audio_file ? '，重新合成后生效' : '';
          showToast(`TTS 后端已更新为: ${backendNames[newBackend] || newBackend}${suffix}`);
        }
      } catch (e) {
        console.error(e);
        showToast("更新 TTS 后端失败: " + e.message);
      }
    }

    async function onSegmentRoleChange(seg, value) {
      if (!seg) return;
      let newRoleId = value;
      if (!newRoleId) {
        newRoleId = null;
      }
      try {
        const updated = await updateSegment(seg.id, { role_id: newRoleId });
        if (!updated) return;
        if (!updated.role_id) {
          showToast("已将该分段设置为未绑定角色，请在上方创建并绑定角色后再合成。");
        } else {
          showToast(updated.audio_file ? "分段角色已更新，重新合成后生效" : "分段角色已更新");
        }
      } catch (e) {
        console.error(e);
        showToast("更新分段角色失败: " + e.message);
      }
    }

    // 检查未使用的角色（角色列表中有，但分段中未使用）
    function checkUnusedRoles() {
      if (!roles.value.length) {
        showToast("当前项目没有角色");
        return;
      }
      if (!segments.value.length) {
        showToast("当前项目没有分段，所有角色都未使用");
        return;
      }
      
      // 收集分段中使用的角色 ID
      const usedRoleIds = new Set();
      segments.value.forEach((seg) => {
        if (seg.role_id) {
          usedRoleIds.add(seg.role_id);
        }
      });
      
      // 找出未使用的角色
      const unusedRoles = roles.value.filter((r) => !usedRoleIds.has(r.id));
      
      if (unusedRoles.length === 0) {
        showToast("所有角色都已在分段中使用");
      } else {
        const names = unusedRoles.map((r) => r.name).join("、");
        showToast(`未使用的角色（${unusedRoles.length}个）：${names}`);
      }
    }

    // 检查某个角色是否未被任何分段使用
    function isRoleUnused(roleId) {
      if (!roleId || !segments.value.length) return false;
      return !segments.value.some((seg) => seg.role_id === roleId);
    }

    // 处理分段角色下拉选择变化（支持自定义角色）
    async function onSegmentRoleSelectChange(seg, value) {
      if (!seg) return;
      
      // 自定义角色选项
      if (value === "__custom__") {
        const customName = prompt("请输入新角色名称：");
        if (!customName || !customName.trim()) {
          return;
        }
        const trimmed = customName.trim();
        
        // 检查是否已存在同名角色
        const existingRole = roles.value.find(
          (r) => r.name.toLowerCase() === trimmed.toLowerCase()
        );
        
        if (existingRole) {
          // 已有角色，直接绑定
          await onSegmentRoleChange(seg, existingRole.id);
          showToast(`已绑定到现有角色「${existingRole.name}」`);
        } else {
          // 创建新角色
          try {
            const res = await fetch(
              `/api/v1/projects/${currentProject.value.id}/roles`,
              {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  name: trimmed,
                  role_type: "character",
                  description: "",
                }),
              }
            );
            if (!res.ok) {
              const err = await res.json().catch(() => ({}));
              throw new Error(err.detail || "创建角色失败");
            }
            const newRole = await res.json();
            roles.value.push(newRole);
            // 绑定新角色到分段
            await onSegmentRoleChange(seg, newRole.id);
            showToast(`已创建角色「${trimmed}」并绑定，请在角色管理中为其绑定音色`);
          } catch (e) {
            console.error(e);
            showToast("创建角色失败: " + e.message);
          }
        }
        return;
      }
      
      // 普通角色选择
      await onSegmentRoleChange(seg, value || null);
    }

    function startEditSegment(seg, field) {
      if (!seg || !["text", "emotion_text"].includes(field)) return;
      // 不控制情感模式下，禁止行内编辑情感文本，避免与实际生效模式产生误解
      // B 模式始终使用 text 模式，不受 A 的 emotion_mode 限制
      if (field === "emotion_text" && seg.emotion_mode === "none" && !isAbMode(seg)) return;
      const value = (field === "text") ? (seg.text || "") : (seg.emotion_text || "");
      segmentInlineEdit.value = {
        id: seg.id,
        field,
        value,
      };
    }

    function cancelSegmentEdit() {
      segmentInlineEdit.value = { id: null, field: null, value: "" };
    }

    async function commitSegmentEdit(seg) {
      const editing = segmentInlineEdit.value;
      if (!seg || !editing.id || editing.id !== seg.id || !editing.field) return;
      const payload = {};
      if (editing.field === "text") {
        payload.text = editing.value;
      } else if (editing.field === "emotion_text") {
        payload.emotion_text = editing.value;
      }
      try {
        const updated = await updateSegment(seg.id, payload);
        if (updated) {
          showToast(updated.audio_file ? "分段已保存，重新合成后生效" : "分段已保存");
        }
      } catch (e) {
        console.error(e);
        showToast("保存分段失败: " + e.message);
      } finally {
        cancelSegmentEdit();
      }
    }

    async function openSegmentEmotionModal(seg) {
      if (!seg) return;

      // 每次打开情感设置前，清理上一次情感录制状态，避免串音
      if (micRecording.value && micRecording.value.context === "emotion") {
        cancelMicRecording();
      }

      // 打开弹窗时重置“从音色库选择”的筛选状态，避免残留
      clearSegEmotionVoiceFilter();
      segEmotionVoiceTypeFilter.value = voiceTypeLabels.emotion;

      const resolvedTtsBackend = getSegmentBackend(seg);
      const role = seg.role_id
        ? (roles.value || []).find((item) => item && item.id === seg.role_id)
        : null;

      // B 版本及 Qwen 后端只编辑文本指令；emotion_mode 不参与 A/B swap，
      // Qwen 的合成也直接读取 emotion_text。
      const inBMode = seg.active_version === "b";
      const allowedModes = ["text", "vector", "audio", "none"];
      const storedMode = allowedModes.includes(seg.emotion_mode) ? seg.emotion_mode : "text";
      const isQwenBackend = resolvedTtsBackend === "qwen3tts" || resolvedTtsBackend === "qwen3tts_cv";
      const mode = (inBMode || isQwenBackend) ? "text" : storedMode;
      const normalizedMode = allowedModes.includes(mode) ? mode : "text";
      let vector = [0, 0, 0, 0, 0, 0, 0, 0];
      if (Array.isArray(seg.emotion_vector) && seg.emotion_vector.length === 8) {
        vector = seg.emotion_vector.map((x) =>
          typeof x === "number" && Number.isFinite(x) ? x : 0,
        );
      }

      // 加载音色列表
      if (!voices.value || !voices.value.length) {
        try {
          await loadVoices();
        } catch (e) {
          console.error(e);
        }
      }

      // 尝试根据 emotion_audio 路径找到对应的音色
      let matchedVoiceId = null;
      if (seg.emotion_audio && voices.value && voices.value.length) {
        const matched = findVoiceByAudioPath(seg.emotion_audio);
        if (matched) {
          matchedVoiceId = matched.id;
        }
      }

      // 确定音频来源
      let audioSource = null;
      let savedSource = null;
      if (matchedVoiceId) {
        audioSource = 'voice';
      } else if (seg.emotion_audio) {
        audioSource = 'saved';
        if (typeof seg.emotion_audio === "string" && seg.emotion_audio.includes('.webm')) {
          savedSource = 'recording';
        } else {
          savedSource = 'upload';
        }
      }

      segEmotionModal.value = {
        id: seg.id,
        mode: normalizedMode,
        text: seg.emotion_text || "",
        vector,
        audioSource: inBMode ? null : audioSource,
        audioVoiceId: inBMode ? null : matchedVoiceId,
        voiceSelectId: inBMode ? null : matchedVoiceId,
        savedAudioPath: inBMode ? null : (seg.emotion_audio || null),
        savedSource: inBMode ? null : savedSource,
        selectedFile: null,
        uploadPreviewUrl: null,
        recordedFile: null,
        recordedUrl: null,
        audioVersion: segEmotionModal.value.audioVersion || 0,
        ttsBackend: resolvedTtsBackend,
        emotionAlpha: seg.emotion_alpha ?? 0.65,
        qwen3Speaker: seg.qwen3_speaker || (role && role.qwen3_speaker) || 'vivian',
        qwen3Language: seg.qwen3_language || (role && role.qwen3_language) || getQwen3DefaultLanguage(),
        original: {
          text: seg.emotion_text || "",
          qwen3Speaker: seg.qwen3_speaker || (role && role.qwen3_speaker) || 'vivian',
          qwen3Language: seg.qwen3_language || (role && role.qwen3_language) || getQwen3DefaultLanguage(),
        },
      };

      // 尝试恢复历史 webm 录音：只要磁盘上存在 segment_id.webm，就认为“之前录制过”
      // 与当前 emotion_audio 指向无关（即使当前使用的是音色库）。
      // 加时间戳避免浏览器缓存旧的录音文件。
      if (currentProject.value && seg.id) {
        try {
          const ts = Date.now();
          const resp = await fetch(
            `/api/v1/projects/${currentProject.value.id}/segments/${seg.id}/emotion-recording?t=${ts}`,
          );
          if (resp.ok) {
            const blob = await resp.blob();
            if (micRecording.value.recordedUrl) {
              URL.revokeObjectURL(micRecording.value.recordedUrl);
            }
            micRecording.value.context = "emotion";
            micRecording.value.recordedBlob = blob;
            micRecording.value.recordedUrl = URL.createObjectURL(blob);
            micRecording.value.duration = 0;
          }
        } catch (e) {
          // 没有录音文件或请求失败时忽略，保持默认“开始录制”状态
          console.error("恢复历史情感录音失败:", e);
        }
      }

      // 重置 AI 生成状态
      resetSegEmotionAiGenerate();

      modals.value.editSegEmotion = true;
    }

    // 重置分段情感 AI 生成状态
    function resetSegEmotionAiGenerate() {
      segEmotionAiGenerate.value = {
        instruct: "",
        previewText: "",
        language: getQwen3DefaultLanguage(),
        generating: false,
        previewUrl: null,
        previewFile: null,
        error: null,
      };
    }

    // 切换分段情感 AI 生成快捷标签
    function toggleSegEmotionAiTag(tag) {
      const current = segEmotionAiGenerate.value.instruct || "";
      if (current.includes(tag)) {
        segEmotionAiGenerate.value.instruct = current.replace(tag, "").replace(/，+/g, "，").replace(/^，|，$/g, "").trim();
      } else {
        segEmotionAiGenerate.value.instruct = current ? `${current}，${tag}` : tag;
      }
    }

    // 使用 AI 生成情感音频
    async function generateSegEmotionWithAi() {
      const state = segEmotionAiGenerate.value;
      if (!state.instruct.trim()) {
        state.error = "请输入音色/情感描述";
        return;
      }

      // 使用分段文本作为默认试听文本
      let previewText = state.previewText;
      if (!previewText && segEmotionModal.value.id) {
        const seg = segments.value.find(s => s.id === segEmotionModal.value.id);
        if (seg && seg.text) {
          previewText = seg.text.slice(0, 100);
        }
      }
      if (!previewText) {
        previewText = "大家好，这是一段测试音频。";
      }

      state.generating = true;
      state.error = null;
      state.previewUrl = null;
      state.previewFile = null;

      try {
        // 自动启动 Qwen3-TTS 服务（如果未运行或运行的是其他后端）
        if (!ttsService.value.running || ttsService.value.backend !== 'qwen3tts') {
          showToast("正在切换到 Qwen3-TTS...");
          // 后端会自动停止旧服务，直接启动新服务即可
          await startTtsService('qwen3tts', { persistDefault: false });
          // 等待服务就绪（包括模型加载）
          const ready = await waitForTtsServiceReady('qwen3tts', 120000);
          if (!ready) {
            state.error = "Qwen3-TTS 服务或模型加载超时，请稍后重试";
            state.generating = false;
            return;
          }
        }

        const resp = await fetchJson("/api/v1/qwen3tts/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: previewText,
            instruct: state.instruct,
            language: state.language,
          }),
        });

        if (resp.success && resp.audio_file) {
          state.previewFile = resp.audio_file;
          state.previewUrl = `/api/v1/qwen3tts/audio?path=${encodeURIComponent(resp.audio_file)}`;
          showToast("AI 情感音频生成成功");
        } else {
          state.error = resp.message || resp.error || "生成失败";
        }
      } catch (err) {
        state.error = err.message || "生成请求失败";
      } finally {
        state.generating = false;
      }
    }

    // 使用 AI 生成的音频作为情感参考
    async function useAiGeneratedEmotionAudio() {
      const state = segEmotionAiGenerate.value;
      if (!state.previewFile || !state.previewUrl) {
        showToast("请先生成音频", "error");
        return;
      }

      const segmentId = segEmotionModal.value.id;
      if (!segmentId || !currentProject.value) {
        showToast("分段信息无效", "error");
        return;
      }

      try {
        showToast("正在上传情感音频...");
        
        // 下载AI生成的音频文件
        const audioResp = await fetch(state.previewUrl);
        if (!audioResp.ok) {
          throw new Error("获取音频文件失败");
        }
        const audioBlob = await audioResp.blob();
        
        // 上传到项目的emotion目录
        const formData = new FormData();
        formData.append("file", audioBlob, `${segmentId}_ai.wav`);
        
        const uploadResp = await fetch(
          `/api/v1/projects/${currentProject.value.id}/segments/${segmentId}/emotion-audio`,
          { method: "POST", body: formData }
        );
        
        if (!uploadResp.ok) {
          const errText = await uploadResp.text();
          throw new Error(errText || "上传失败");
        }
        
        const updated = await uploadResp.json();
        const idx = segments.value.findIndex((s) => s.id === updated.id);
        if (idx !== -1) {
          Object.assign(segments.value[idx], updated);
        }
        
        // 更新弹窗状态
        segEmotionModal.value.mode = 'audio';
        segEmotionModal.value.audioSource = 'saved';
        segEmotionModal.value.savedAudioPath = updated.emotion_audio;
        segEmotionModal.value.audioVersion = (segEmotionModal.value.audioVersion || 0) + 1;
        
        showToast("AI 情感音频已上传并应用");
      } catch (err) {
        console.error("上传AI情感音频失败:", err);
        showToast("上传失败: " + err.message, "error");
      }
    }

    // 将 AI 生成的情感音频添加到音色库
    async function addAiEmotionToVoiceLibrary() {
      const state = segEmotionAiGenerate.value;
      if (!state.previewFile || !state.previewUrl) {
        showToast("请先生成音频", "error");
        return;
      }

      try {
        showToast("正在添加到音色库...");
        
        // 下载AI生成的音频文件
        const audioResp = await fetch(state.previewUrl);
        if (!audioResp.ok) {
          throw new Error("获取音频文件失败");
        }
        const audioBlob = await audioResp.blob();
        
        // 生成音色名称
        const instruct = state.instruct || "";
        const voiceName = instruct.slice(0, 20).replace(/[，,。.]/g, " ").trim() || "AI情感音色";
        
        // 收集标签：情感 + 描述中的标签
        const tags = ["情感"];
        const tagKeywords = ["男性", "女性", "中性", "儿童", "青年", "中年", "老年", "温柔", "沉稳", "活泼", "严肃", "甜美", "磁性", "清脆", "低沉"];
        tagKeywords.forEach(tag => {
          if (instruct.includes(tag)) {
            tags.push(tag);
          }
        });
        
        // 1. 先创建音色元数据
        const createResp = await fetch("/api/v1/voices", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: voiceName,
            tags: tags,
            description: instruct,
          }),
        });
        
        if (!createResp.ok) {
          const errText = await createResp.text();
          throw new Error(errText || "创建音色失败");
        }
        
        const newVoice = await createResp.json();
        
        // 2. 上传音频文件
        const formData = new FormData();
        formData.append("file", audioBlob, `ai_emotion_${Date.now()}.wav`);
        
        const uploadResp = await fetch(`/api/v1/voices/${newVoice.id}/audio`, {
          method: "POST",
          body: formData,
        });
        
        if (!uploadResp.ok) {
          const errText = await uploadResp.text();
          throw new Error(errText || "上传音频失败");
        }
        
        await loadVoices();
        showToast("已添加到音色库");
      } catch (err) {
        console.error("添加到音色库失败:", err);
        showToast("添加失败: " + err.message, "error");
      }
    }

    function closeSegmentEmotionModal() {
      modals.value.editSegEmotion = false;
      segEmotionModal.value = {
        id: null,
        mode: "text",
        text: "",
        vector: [0, 0, 0, 0, 0, 0, 0, 0],
        audioSource: null,
        audioVoiceId: null,
        voiceSelectId: null,
        savedAudioPath: null,
        savedSource: null,
        selectedFile: null,
        uploadPreviewUrl: null,
        recordedFile: null,
        recordedUrl: null,
        audioVersion: 0,
      };
    }

    async function saveSegmentEmotion() {
      const state = segEmotionModal.value;
      if (!state || !state.id) {
        closeSegmentEmotionModal();
        return;
      }

      const payload = {};
      const mode = state.mode || "text";
      // B 模式下 emotion_mode 不参与 swap，不应修改（否则会覆盖 A 的 emotion_mode）
      const seg = (segments.value || []).find((s) => s.id === state.id);
      const inBMode = seg && isAbMode(seg);
      const isQwenBackend = state.ttsBackend === 'qwen3tts' || state.ttsBackend === 'qwen3tts_cv';

      if (isQwenBackend) {
        // Qwen 的音色、语言属于 A/B 共享字段；仅在用户实际修改时提交，
        // 否则后端会将“字段出现”视为需要同时废弃 A/B 两侧音频。
        const original = state.original || {};
        const nextText = state.text || "";
        if (nextText !== (original.text || "")) {
          payload.emotion_text = nextText || null;
        }
        if (state.ttsBackend === 'qwen3tts_cv') {
          const nextSpeaker = state.qwen3Speaker || 'vivian';
          if (nextSpeaker !== (original.qwen3Speaker || 'vivian')) {
            payload.qwen3_speaker = nextSpeaker;
          }
        }
        const nextLanguage = state.qwen3Language || 'Auto';
        if (nextLanguage !== (original.qwen3Language || 'Auto')) {
          payload.qwen3_language = nextLanguage;
        }
      } else if (mode === "text") {
        if (!inBMode) payload.emotion_mode = "text";
        payload.emotion_text = state.text || null;
        payload.emotion_vector = null;
        payload.emotion_audio = null;
      } else if (mode === "audio") {
        if (!inBMode) payload.emotion_mode = "audio";
        payload.emotion_text = state.text || null;
        let audioPath = null;

        // 根据 audioSource 确定音频路径
        if (state.audioSource === 'recording' && state.recordedFile) {
          // 录制的音频需要先上传
          try {
            const formData = new FormData();
            formData.append("file", state.recordedFile);
            const uploadResp = await fetch(
              `/api/v1/projects/${currentProject.value.id}/segments/${state.id}/emotion-audio`,
              { method: "POST", body: formData }
            );
            if (!uploadResp.ok) {
              const errText = await uploadResp.text();
              throw new Error(errText || "上传失败");
            }
            const updated = await uploadResp.json();
            const idx = segments.value.findIndex((s) => s.id === updated.id);
            if (idx !== -1) {
              Object.assign(segments.value[idx], updated);
            }
            showToast("情感参考音频已保存");
            closeSegmentEmotionModal();
            return;
          } catch (e) {
            console.error("上传情感参考音频失败:", e);
            showToast("上传情感参考音频失败: " + e.message);
            return;
          }
        } else if (state.audioSource === 'voice' && state.audioVoiceId) {
          // 从音色库选择
          try {
            if (!voices.value || !voices.value.length) {
              await loadVoices();
            }
          } catch (e) {
            console.error("[Emotion] 加载音色库失败: ", e);
          }
          const vs = voices.value || [];
          const v = vs.find((x) => x && x.id === state.audioVoiceId);
          if (!v || !v.audio_path) {
            showToast("所选音色没有有效的参考音频");
            return;
          }
          audioPath = v.audio_path;
          if (audioPath) {
            const normalized = String(audioPath).replace(/\\/g, "/");
            if (!normalized.startsWith("audio/") && !normalized.startsWith("voices/")) {
              audioPath = `voices/${normalized}`;
            }
          }
          console.debug("[Emotion] saveSegmentEmotion (audio): using voice source", {
            segmentId: state.id,
            audioVoiceId: state.audioVoiceId,
            audioPath,
          });
        } else if (state.audioSource === 'saved' && state.savedAudioPath) {
          // 已保存的音频
          audioPath = state.savedAudioPath;
          console.debug("[Emotion] saveSegmentEmotion (audio): using saved source", {
            segmentId: state.id,
            savedAudioPath: state.savedAudioPath,
          });
        } else if (state.audioSource === 'ai' && state.savedAudioPath) {
          // AI 生成的音频
          audioPath = state.savedAudioPath;
          console.debug("[Emotion] saveSegmentEmotion (audio): using AI generated source", {
            segmentId: state.id,
            savedAudioPath: state.savedAudioPath,
          });
        }

        if (!audioPath) {
          showToast("请选择情感参考音频");
          return;
        }
        payload.emotion_audio = audioPath;
        payload.emotion_vector = null;
        console.debug("[Emotion] saveSegmentEmotion (audio): final payload", payload);
      } else if (mode === "vector") {
        if (!inBMode) payload.emotion_mode = "vector";
        payload.emotion_text = state.text || null;
        const vec = Array.isArray(state.vector) ? state.vector : [];
        const floats = [];
        for (let i = 0; i < 8; i += 1) {
          let v = parseFloat(vec[i]);
          if (!Number.isFinite(v)) v = 0;
          if (v < 0) v = 0;
          if (v > 1) v = 1;
          floats.push(Number(v.toFixed(3)));
        }
        payload.emotion_vector = floats;
        payload.emotion_audio = null;
      } else if (mode === "none") {
        if (!inBMode) payload.emotion_mode = "none";
        // 不再清空情感文本，只关闭情感控制，方便日后参考
        payload.emotion_text = state.text || null;
        payload.emotion_vector = null;
        payload.emotion_audio = null;
      }

      // 保存情感权重（仅 IndexTTS2）
      if (state.ttsBackend !== 'cosyvoice3' && state.ttsBackend !== 'qwen3tts' && state.ttsBackend !== 'qwen3tts_cv') {
        payload.emotion_alpha = state.emotionAlpha ?? 0.65;
      }

      if (!Object.keys(payload).length) {
        showToast("未检测到需要保存的修改");
        closeSegmentEmotionModal();
        return;
      }

      try {
        const updated = await updateSegment(state.id, payload);
        if (updated) {
          showToast("分段情感已更新");
        }
        closeSegmentEmotionModal();
      } catch (e) {
        console.error(e);
        showToast("更新分段情感失败: " + e.message);
      }
    }

    function canMergeDown(seg) {
      if (!seg) return false;
      const list = segments.value;
      if (!Array.isArray(list) || !list.length) return false;
      const idx = list.findIndex((s) => s.id === seg.id);
      if (idx === -1 || idx >= list.length - 1) return false;
      const next = list[idx + 1];
      const a = seg.role_id || null;
      const b = next.role_id || null;
      return a === b;
    }

    async function mergeSegmentDown(seg) {
      if (!currentProject.value || !seg) return;
      if (!ensureProjectMutationAllowed("合并分段")) return;
      if (!canMergeDown(seg)) {
        showToast("当前分段无法向下合并");
        return;
      }
      pushSegmentSnapshot();
      try {
        const data = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/segments/${seg.id}/merge-down`,
          {
            method: "POST",
            body: JSON.stringify({}),
          },
        );
        segments.value = Array.isArray(data) ? data : [];
        reinitWaveformsForSegments();
        showToast("已向下合并分段");
      } catch (e) {
        console.error(e);
        showToast("向下合并失败: " + e.message);
      }
    }

    async function insertSegmentBelow(seg) {
      if (!currentProject.value || !seg) return;
      if (!ensureProjectMutationAllowed("插入分段")) return;
      pushSegmentSnapshot();
      try {
        const data = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/segments/${seg.id}/insert-below`,
          {
            method: "POST",
            body: JSON.stringify({}),
          },
        );
        segments.value = Array.isArray(data) ? data : [];
        reinitWaveformsForSegments();
        showToast("已在当前分段下方插入新分段");
      } catch (e) {
        console.error(e);
        showToast("插入新分段失败: " + e.message);
      }
    }

    async function deleteSegment(seg) {
      if (!currentProject.value || !seg) return;
      if (!ensureProjectMutationAllowed("删除分段")) return;
      const ok = window.confirm("确认要删除该分段吗？（可通过 Ctrl+Z 撤销）");
      if (!ok) return;
      pushSegmentSnapshot();
      try {
        const data = await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/segments/${seg.id}`,
          {
            method: "DELETE",
            body: JSON.stringify({}),
          },
        );
        segments.value = Array.isArray(data) ? data : [];
        reinitWaveformsForSegments();
        showToast("分段已删除");
      } catch (e) {
        console.error(e);
        showToast("删除分段失败: " + e.message);
      }
    }

    async function clearSegmentEdits(seg) {
      if (!currentProject.value || !seg) return;
      const ok = window.confirm("确认清空该分段的所有编辑草稿（速度、音量、区间、静音、淡化和人声预设）吗？此操作不可恢复，但不会删除音频本身。");
      if (!ok) return;
      const st = getWaveformState(seg.id);
      st.speed = getSegmentBaseSpeed(seg);
      st.volume = getSegmentBaseVolume(seg);
      st.region = null;
      st.regions = [];
      st.selectedRegionId = null;
      st.silences = [];
      st.selectedSilenceId = null;
      st.pendingSilenceAt = null;
      st.pendingSilenceDuration = null;
      st.silenceSec = 0;
      st.selecting = false;
      st.fadeIn = 0;
      st.fadeOut = 0;
      st.leadingSilence = 0;
      st.trailingSilence = 0;
      st.processingPreset = "none";
      const ws = wavesurferMap[seg.id];
      if (ws) {
        ws.setPlaybackRate(getSegmentPreviewSpeed(seg));
        ws.setVolume(getSegmentPreviewVolume(seg));
      }
      const pid = selectedProjectId.value;
      if (pid) {
        scheduleWaveformStatePersist(pid);
      }
      showToast("已清空本段编辑草稿");
    }

    async function exportFullWithEdits() {
      if (!currentProject.value) return;
      if (!ensureProjectMutationAllowed("导出音频")) return;
      loading.value.exportFullWithEdits = true;
      try {
        const projectId = currentProject.value.id;
        await flushWaveformStatePersist(projectId);
        if (!currentProject.value || currentProject.value.id !== projectId) return;
        const list = Array.isArray(segments.value) ? segments.value : [];
        const hasEditDrafts = list.some((seg) => hasEffectiveWaveformEditDraft(seg));

        const endpoint = hasEditDrafts ? "export/full-with-edits" : "export/full";
        const data = await fetchJson(
          `/api/v1/projects/${projectId}/${endpoint}`,
          {
            method: "POST",
            body: JSON.stringify({}),
          },
        );
        lastExport.value = data.output_file;
        showToast(hasEditDrafts ? "按当前编辑导出完成" : "导出整书音频完成");
        await loadFinalAudios();
      } catch (e) {
        console.error(e);
        showToast("按当前编辑导出失败: " + e.message);
      } finally {
        loading.value.exportFullWithEdits = false;
      }
    }

    async function openFinalAudioLocation(filename) {
      if (!currentProject.value || !filename) return;
      try {
        await fetchJson(
          `/api/v1/projects/${currentProject.value.id}/exports/final/${encodeURIComponent(filename)}/open-location`,
          {
            method: "POST",
            body: JSON.stringify({}),
          },
        );
        showToast("已在服务器上打开文件所在位置");
      } catch (e) {
        console.error(e);
        showToast("打开所在位置失败: " + e.message);
      }
    }

    async function openProjectLocation() {
      if (!currentProject.value) return;
      try {
        await fetchJson(`/api/v1/projects/${currentProject.value.id}/open-location`, {
          method: "POST",
          body: JSON.stringify({}),
        });
        showToast("已在服务器上打开项目目录");
      } catch (e) {
        console.error(e);
        showToast("打开项目目录失败: " + e.message);
      }
    }

    function onSegmentRowClick(seg) {
      if (!seg || !seg.id) return;
      activePlaySegId.value = seg.id;
    }

    function findRoleName(roleId) {
      if (!roleId) return "";
      const r = roles.value.find((x) => x.id === roleId);
      return r ? r.name : "";
    }

    function isBackendRequiresSpeakerAudio(backend) {
      return backend !== "qwen3tts" && backend !== "qwen3tts_cv";
    }

    // 检查角色是否缺少音色绑定
    function isRoleMissingSpeakerAudio(roleId, backendOverride) {
      if (!roleId) return false;
      const resolvedBackend =
        backendOverride || (ttsBackend.value && ttsBackend.value.current) || "indextts";
      if (!isBackendRequiresSpeakerAudio(resolvedBackend)) return false;
      const r = roles.value.find((x) => x.id === roleId);
      return r && !r.speaker_audio && !r.speaker_voice_id;
    }

    function canSegmentBeSynthesized(seg) {
      if (!seg || !seg.role_id) return false;
      return !isRoleMissingSpeakerAudio(seg.role_id, getSegmentBackend(seg));
    }

    function doesSegmentNeedSynthesis(seg) {
      if (!seg) return false;
      if (seg.status === "pending" || seg.status === "failed") {
        return true;
      }
      if (seg.status === "done" && !seg.audio_file) {
        return true;
      }
      return false;
    }

    // 检查是否有可合成的分段（绑定了角色且角色有音色）
    const hasSynthesizableSegments = computed(() => {
      if (!segments.value.length) return false;
      return segments.value.some((seg) => canSegmentBeSynthesized(seg));
    });

    const hasBatchSynthesisTargets = computed(() => {
      if (!segments.value.length) return false;
      return segments.value.some((seg) => canSegmentBeSynthesized(seg) && doesSegmentNeedSynthesis(seg));
    });

    const batchSynthesizeDisabledReason = computed(() => {
      if (loading.value.autoSegment) {
        return "LLM 分段进行中，请稍候再批量合成";
      }
      if (!segments.value.length) {
        return "当前项目还没有分段，请先进行 LLM 分段或导入 JSON 分段";
      }
      if (hasBatchSynthesisTargets.value) {
        return "";
      }
      if (!hasSynthesizableSegments.value) {
        return "没有可合成的分段（请先绑定角色和音色）";
      }
      return "没有待合成、待重试或音频缺失的分段";
    });

    function getSegmentSynthesizeTitle(seg) {
      if (!seg) return "";
      if (isSegmentSynthRunning(seg)) {
        return "当前分段正在合成";
      }
      if (isSegmentInQueueWaiting(seg)) {
        return "当前分段已在队列中，点击可取消";
      }
      if (!seg.role_id) {
        return "请先为该分段绑定角色";
      }
      if (isRoleMissingSpeakerAudio(seg.role_id, getSegmentBackend(seg))) {
        return "请先在角色管理中为该角色绑定音色";
      }
      return seg.audio_file ? "重新生成当前分段" : "合成当前分段";
    }

    function speakerName(seg) {
      if (!seg || !seg.llm_raw_output) return "";
      const raw = seg.llm_raw_output;
      const name = (raw.speaker_name ?? raw.speaker ?? "").trim();
      return name;
    }

    // 根据 provider_id 获取提供商名称
    function getProviderName(providerId) {
      if (!providerId) {
        // 找默认提供商
        const defaultProvider = llmProviders.value.find((p) => p.is_default);
        return defaultProvider ? defaultProvider.name : "默认";
      }
      const provider = llmProviders.value.find((p) => p.id === providerId);
      return provider ? provider.name : "未知";
    }

    // 格式化 LLM 配置显示名称：提供商-模型（max_tokens=xxxK）
    function formatLlmConfigName(cfg) {
      if (!cfg) return "";
      const providerName = getProviderName(cfg.provider_id);
      const modelName = cfg.model || "未知模型";
      const maxTokens = cfg.max_tokens ? `max_tokens=${Math.round(cfg.max_tokens / 1000)}K` : "max_tokens=默认";
      return `${providerName} - ${modelName}（${maxTokens}）`;
    }

    // TTS 服务状态轮询定时器
    let ttsServicePollTimer = null;
    let ttsServicePollBoostUntil = 0;

    function clearTtsServicePollTimer() {
      if (ttsServicePollTimer) {
        clearTimeout(ttsServicePollTimer);
        ttsServicePollTimer = null;
      }
    }

    function boostTtsServicePolling(durationMs = 30000) {
      ttsServicePollBoostUntil = Date.now() + durationMs;
      scheduleTtsServicePolling(0);
    }

    function getNextTtsServicePollDelayMs() {
      if (typeof document !== "undefined" && document.hidden) {
        return null;
      }

      const now = Date.now();
      if (now < ttsServicePollBoostUntil) {
        return 2000;
      }
      if (ttsService.value && ttsService.value.loading) {
        return 2000;
      }
      if (ttsService.value && !ttsService.value.running) {
        return 15000;
      }
      return 10000;
    }

    function scheduleTtsServicePolling(delayOverrideMs) {
      clearTtsServicePollTimer();

      const nextDelay =
        typeof delayOverrideMs === "number" ? delayOverrideMs : getNextTtsServicePollDelayMs();
      if (nextDelay === null) {
        return;
      }

      ttsServicePollTimer = setTimeout(async () => {
        try {
          await loadTtsServiceStatus();
          await refreshHealth({ force: false, minIntervalMs: 30000 });
        } finally {
          scheduleTtsServicePolling();
        }
      }, nextDelay);
    }

    function onVisibilityChangeForTtsPolling() {
      if (typeof document !== "undefined" && document.hidden) {
        clearTtsServicePollTimer();
        return;
      }
      scheduleTtsServicePolling(0);
    }

    onMounted(async () => {
      loadColumnWidthsFromStorage();
      window.addEventListener("keydown", onGlobalKeydown);
      window.addEventListener("beforeunload", onBeforeUnloadWithPendingWaveformEdits);
      document.addEventListener("visibilitychange", onVisibilityChangeForTtsPolling);
      await refreshHealth();
      await loadPlatformSettings();
      await loadTtsParams();
      await loadTtsBackend();
      await loadTtsServiceStatus();

      await loadGlobalSegPromptTemplate();
      await loadLlmProviders();
      await loadLlmConfigs();
      await loadVoices();

      // 在加载项目列表前尝试恢复上一次打开的项目 ID
      const lastId = restoreLastProjectId();
      if (lastId) {
        selectedProjectId.value = lastId;
      }

      await loadProjects();

      // 启动 TTS 服务状态轮询
      scheduleTtsServicePolling(0);
    });

    onBeforeUnmount(() => {
      window.removeEventListener("keydown", onGlobalKeydown);
      window.removeEventListener("beforeunload", onBeforeUnloadWithPendingWaveformEdits);
      document.removeEventListener("visibilitychange", onVisibilityChangeForTtsPolling);
      clearTtsServicePollTimer();
      clearWaveformPersistTimer();
      clearSoundJobPolling();
      cleanupSynthesisSockets();
    });

    Vue.watch(
      () => ({ left: leftTab.value, tab: activeTab.value }),
      (next) => {
        // 仅当处于「项目工作区」且 Tab 为「分段 & 合成」时，重新初始化波形
        if (next.left === "projects" && next.tab === "text") {
          reinitWaveformsForSegments();
        }
      },
    );

    return {
      apiBase,
      health,
      projects,
      llmConfigs,
      filteredLlmConfigs,
      llmProviders,
      selectedLlmProviderId,
      voices,
      selectedProjectId,
      showProjectList,
      roles,
      segments,
      tabs,
      activeTab,
      leftTab,
      modals,
      forms,
      roleSaving,
      platformSettings,
      ttsParams,
      ttsBackend,
      ttsService,
      displayedTtsBackend,
      ttsServiceNeedsRestart,
      getTtsBackendName,
      isCosyVoice3,
      getSegmentBackend,
      isSegmentCosyVoice3,
      isSegmentQwen3TTS,
      isSegmentQwen3TTSCustomVoice,
      isSegmentQwen3TTSAny,
      qwen3CustomVoiceSpeakers,
      qwen3Languages,
      getLanguageForSpeaker,
      globalSegPromptTemplate,
      platformLoading,
      llmTestResult,
      llmConfigTestStatus,
      llm,
      loading,
      showStopBatchButton,
      stoppingBatchSynthesize,
      lastExport,
      finalAudios,
      masters,
      creatingMaster,
      productionSettings,
      audioAssets,
      qualityReport,
      releasePackages,
      productionBusy,
      pronunciationDraft,
      assetUploadForm,
      assetFileInput,
      coverFileInput,
      timelineDraft,
      soundModelStatus,
      soundPlans,
      activeSoundPlanId,
      activeSoundPlan,
      soundJobs,
      soundGenerationBusy,
      soundPlanDensity,
      soundLicenseForm,
      soundPreview,
      soundPreviewAudioUrl,
      subtitleTask,
      subtitleLanguage,
      subtitleModel,
      subtitles,
      subtitleEditing,
      subtitleEditingLoading,
      subtitleEditingSaving,
      subtitleEditMode,
      subtitleJsonText,
      switchToJsonEditMode,
      toast,
      columnWidths,
      currentProject,
      hasPendingSynthesis,
      hasBatchSynthesisTargets,
      batchSynthesizeDisabledReason,
      projectStats,
      synthTasks,
      llmTasks,
      hasRunningSynthTasks,
      hasRunningLlmTasks,
      hasRunningAnyTasks,
      synthesisResumable,
      resumeSynthesis,
      dismissResumableSynthesis,
      canUndo,
      canRedo,
      undoSegments,
      redoSegments,
      isAbMode,
      toggleAbView,
      getCurrentTakeId,
      isCurrentTakeSelected,
      archivedTakeCount,
      selectCurrentTake,
      takeHistoryState,
      takeLocationLabel,
      takeAudioUrl,
      openTakeHistory,
      closeTakeHistory,
      restoreHistoryTake,
      toggleTakePin,
      clearAlt,
      taskListTab,
      formatTaskStatus,
      removeSynthTask,
      stopSynthTask,
      resumePersistentSynthTask,
      taskSegmentIndex,
      currentRoleForUpload,
      currentRoleForVoice,
      audioInput,
      jsonImportText,
      jsonFileInput,
      llmShowingPrompt,
      llmTextboxCollapsed,
      segPromptTemplate,
      segPromptTemplateSaving,
      editingProjectId,
      editingRoleId,
      segmentInlineEdit,
      segEmotionModal,
      segEmotionUploadInput,
      segEmotionAiGenerate,
      toggleSegEmotionAiTag,
      generateSegEmotionWithAi,
      useAiGeneratedEmotionAudio,
      addAiEmotionToVoiceLibrary,
      segEmotionVoiceSearch,
      segEmotionVoiceTypeFilter,
      filteredEmotionVoices,
      clearSegEmotionVoiceFilter,
      selectProject,
      openCreateProjectModal,
      closeProjectModal,
      submitProject,
      closeRoleModal,
      submitRole,
      openUploadAudio,
      closeUploadAudioModal,
      submitUploadAudio,
      importText,
      autoSegment,
      savePrompt,
      importSegmentsFromJson,
      openImportJsonModal,
      closeImportJsonModal,
      handleJsonFileChange,
      togglePromptView,
      openEditSegPromptTemplateModal,
      closeEditSegPromptTemplateModal,
      saveSegPromptTemplate,
      batchSynthesize,
      stopBatchSynthesize,
      refreshSegmentsStatus,
      batchUpdateEmotionAlpha,
      exportFullWithEdits,
      createReviewedMaster,
      masterAudioUrl,
      loadProductionWorkspace,
      saveContentMode,
      createChapter,
      deleteChapter,
      onSegmentChapterChange,
      savePronunciations,
      addPronunciation,
      removePronunciation,
      uploadSoundAsset,
      assetAudioUrl,
      soundTrackLabel,
      deleteSoundAsset,
      saveSoundDesign,
      addTimelineClip,
      removeTimelineClip,
      isAdoptedSoundClip,
      loadSoundGenerationWorkspace,
      confirmStableAudioLicense,
      draftSoundPlan,
      saveSoundCue,
      normalizeSoundCueTrack,
      generateCueCandidates,
      soundCandidateAudioUrl,
      drawCandidateWaveform,
      adoptSoundCandidate,
      unadoptSoundCandidate,
      deleteSoundCandidate,
      createSoundPreview,
      cueAnchorLabel,
      runQualityCheck,
      saveReleaseMetadata,
      uploadReleaseCover,
      createReleasePackage,
      generateSubtitleForLatestFull,
      generateSubtitleForAudio,
      deleteFinalAudio,
      openSubtitleEditor,
      closeSubtitleEditor,
      saveSubtitleEdits,
      deleteSubtitle,
      synthesizeSegment,
      loadFinalAudios,
      loadMasters,
      loadSubtitles,
      openFinalAudioLocation,
      openProjectLocation,
      findRoleName,
      isRoleMissingSpeakerAudio,
      hasSynthesizableSegments,
      getSegmentSynthesizeTitle,
      speakerName,
      getProviderName,
      formatLlmConfigName,
      onSegmentRowClick,
      currentLlmConfig,
      onLlmConfigChange,
      loadPlatformSettings,
      savePlatformSettings,
      loadTtsParams,
      saveTtsParams,
      resetTtsParams,
      switchTtsBackend,
      loadTtsServiceStatus,
      startTtsService,
      stopTtsService,
      loadGlobalSegPromptTemplate,
      saveGlobalSegPromptTemplate,
      resetGlobalSegPromptTemplate,
      unloadTtsModel,
      unloadWhisperModel,
      testLlm,
      testLlmConfig,
      startCreateLlmConfig,
      startEditLlmConfig,
      closeLlmConfigModal,
      submitLlmConfig,
      deleteLlmConfig,
      setDefaultLlmConfig,
      llmConfigForm,
      llmConfigSaving,
      llmProviderForm,
      llmProviderSaving,
      loadLlmProviders,
      startCreateLlmProvider,
      startEditLlmProvider,
      closeLlmProviderModal,
      submitLlmProvider,
      deleteLlmProvider,
      setDefaultLlmProvider,
      findProviderName,
      selectLlmProvider,
      voiceForm,
      voiceSaving,
      voiceFileInput,
      voiceLocalPreviewUrl,
      voiceFormCard,
      voiceTagPresets,
      voiceTypeLabels,
      voiceGenderTags,
      voiceAgeTags,
      voiceAiGenerate,
      voiceAiTags,
      toggleVoiceAiTag,
      generateVoiceWithAi,
      useAiGeneratedVoice,
      voiceFilterTags,
      voiceOtherSearch,
      filteredVoices,
      speakerVoices,
      isLegacyUntypedVoice,
      toggleVoiceFilterTag,
      clearVoiceFilter,
      isVoiceTypeTag,
      isGenderTag,
      isAgeTag,
      getVoiceTypeTagClass,
      getVoiceTypeDisplayText,
      getVoiceGenderTagClass,
      getVoiceAgeTagClass,
      getVoiceFilterTagClass,
      sortVoiceTags,
      selectedVoiceId,
      showVoiceForm,
      collapseVoiceForm,
      loadVoices,
      startCreateVoice,
      startEditVoice,
      submitVoice,
      deleteVoice,
      openSelectVoiceModal,
      closeSelectVoiceModal,
      bindVoiceToRole,
      getRoleBoundVoice,
      addRoleSpeakerToVoiceLibrary,
      formatDateTime,
      formatDurationSeconds,
      toggleVoiceTag,
      toggleVoiceTypeTag,
      onVoiceFileChange,
      openCreateRoleModal,
      startEditProject,
      deleteProject,
      startEditRole,
      deleteRole,
      cleanupRolePrompts,
      cleanupSegDebugFiles,
      cleanupAudioTemp,
      cleanupProjectTmpFiles,
      roleBadgeClass,
      segmentStatusBadgeClass,
      segmentReviewLabel,
      onSegmentReviewStatusChange,
      flagSegmentIssue,
      resolveAllReviewIssues,
      batchApproveReviewable,
      jumpToNextReview,
      segmentRowClass,
      segmentWaveformRowClass,
      segmentTextClass,
      segmentEmotionTextClass,
      micRecording,
      startMicRecording,
      stopMicRecording,
      restartMicRecording,
      formatRecordingDuration,
      useMicRecordingForVoice,
      useMicRecordingForEmotion,
      saveRecordingToVoiceLibrary,
      uploadRecordingToSegment,
      applySelectedEmotionVoice,
      onEmotionFileSelected,
      getEmotionAudioUrl,
      getEmotionAudioLabel,
      getVoiceNameById,
      uploadSegmentEmotionAudio,
      uploadSegmentEmotionAudioToSegment,
      startColumnResize,
      startEditSegment,
      cancelSegmentEdit,
      commitSegmentEdit,
      openSegmentEmotionModal,
      closeSegmentEmotionModal,
      saveSegmentEmotion,
      onSegmentEmotionAlphaChange,
      onSegmentTtsBackendChange,
      onSegmentRoleChange,
      onSegmentRoleSelectChange,
      isRoleUnused,
      checkUnusedRoles,
      canMergeDown,
      mergeSegmentDown,
      insertSegmentBelow,
      deleteSegment,
      isSegmentSynthRunning,
      isSegmentPlaying,
      isSegmentInQueueWaiting,
      cancelSegmentSynthesis,
      onClickSegmentSynthesize,
      waveformStates,
      getWaveformState,
      toggleSegmentPlay,
      onWaveSpeedChange,
      onWaveVolumeChange,
      startDeleteRegionSelection,
      clearWaveformRegion,
      waveformRegionStyle,
      waveformSilenceRegionStyle,
      getWaveformTimeInfo,
      getSelectedSilence,
      onWaveformMouseDown,
      isSegmentAudioEditPending,
      applySegmentAudioProcessing,
      renderSegmentAsNewTake,
      deleteWaveformRegion,
      addSilenceAtCursor,
      clearSilenceAt,
      onSilenceMarkerClick,
      previewSegment,
      previewSegmentEmotionAudio,
      formatEmotionVector,
      singlePreviewSegmentId,
      singlePreviewPlaying,
      stopSinglePreview,
      sequentialPreviewRunning,
      sequentialPreviewStartSegmentId,
      sequentialPreviewCurrentSegmentId,
      sequentialPreviewAllSegments,
      sequentialPreviewFromSegment,
      clearSegmentEdits,
    };
  },
}).mount("#app");

if (typeof window !== "undefined") {
  window.__WEBUI_MOUNTED__ = true;
  window.__WEBUI_INIT_ERROR__ = null;
}
