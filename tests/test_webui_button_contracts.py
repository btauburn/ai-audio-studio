import re
import unittest
from pathlib import Path

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.main import create_app


class TestWebUIButtonContracts(unittest.TestCase):
    def test_click_handlers_have_function_and_return_binding(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        index_html = (webui_root / "index.html").read_text(encoding="utf-8")
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")

        click_exprs = re.findall(r'@click="([^"]+)"', index_html)
        called_handlers = {
            m.group(1)
            for expr in click_exprs
            if (m := re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", expr.strip()))
        }

        function_defs = set(
            re.findall(
                r"\b(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                app_js,
            )
        )

        missing_defs = sorted(name for name in called_handlers if name not in function_defs)
        self.assertEqual(
            missing_defs,
            [],
            f"index.html 中存在未定义的点击处理函数: {missing_defs}",
        )

        ret_match = re.search(
            r"return\s*\{(?P<body>.*?)\}\s*;\s*\n\s*\},\s*\n\}\)\.mount\(\"#app\"\);",
            app_js,
            re.S,
        )
        self.assertIsNotNone(ret_match, "无法定位 app.js 中 setup() 的 return 对象")
        return_body = ret_match.group("body")
        returned_names = set(
            m.group(1)
            for m in re.finditer(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*$", return_body, re.M)
        )

        missing_returns = sorted(name for name in called_handlers if name not in returned_names)
        self.assertEqual(
            missing_returns,
            [],
            f"点击处理函数未暴露到模板 return 对象: {missing_returns}",
        )

    def test_button_related_api_routes_registered(self) -> None:
        clear_config_caches()
        app = create_app()

        route_methods: dict[str, set[str]] = {}
        for route in app.routes:
            path = getattr(route, "path", None)
            if not path:
                continue
            methods = set(getattr(route, "methods", set()) or set())
            if path not in route_methods:
                route_methods[path] = set()
            route_methods[path].update(methods)

        expected: dict[str, set[str]] = {
            "/api/v1/ready": {"GET"},
            "/api/v1/health": {"GET"},
            "/api/v1/projects": {"GET", "POST"},
            "/api/v1/projects/{project_id}": {"PUT", "DELETE"},
            "/api/v1/projects/{project_id}/open-location": {"POST"},
            "/api/v1/projects/{project_id}/cleanup-tmp-files": {"POST"},
            "/api/v1/projects/{project_id}/audio/cleanup-temp": {"POST"},
            "/api/v1/projects/{project_id}/segments/cleanup-debug-files": {"POST"},
            "/api/v1/projects/{project_id}/roles/cleanup-prompts": {"POST"},
            "/api/v1/projects/{project_id}/roles": {"GET", "POST"},
            "/api/v1/projects/{project_id}/roles/{role_id}": {"PUT", "DELETE"},
            "/api/v1/projects/{project_id}/roles/{role_id}/voice-profile": {"PATCH"},
            "/api/v1/projects/{project_id}/roles/{role_id}/speaker-audio": {"GET", "POST"},
            "/api/v1/projects/{project_id}/roles/{role_id}/bind-voice": {"POST"},
            "/api/v1/projects/{project_id}/text": {"GET"},
            "/api/v1/projects/{project_id}/text/import": {"POST"},
            "/api/v1/projects/{project_id}/segments": {"GET"},
            "/api/v1/projects/{project_id}/segments/import-json": {"POST"},
            "/api/v1/projects/{project_id}/segments/auto-generate": {"POST"},
            "/api/v1/projects/{project_id}/segments/auto-generate/status": {"GET"},
            "/api/v1/projects/{project_id}/segments/refresh-status": {"POST"},
            "/api/v1/projects/{project_id}/segments/edit-state": {"POST"},
            "/api/v1/projects/{project_id}/segments/restore": {"PUT"},
            "/api/v1/projects/{project_id}/segments/reset-stale": {"POST"},
            "/api/v1/projects/{project_id}/segments/batch-update-emotion-alpha": {"POST"},
            "/api/v1/projects/{project_id}/segments/synthesize/plan": {"POST"},
            "/api/v1/projects/{project_id}/segments/synthesize": {"POST"},
            "/api/v1/projects/{project_id}/segments/{segment_id}/synthesize-alt": {"POST"},
            "/api/v1/projects/{project_id}/segments/{segment_id}/swap-alt": {"POST"},
            "/api/v1/projects/{project_id}/segments/{segment_id}/alt": {"DELETE"},
            "/api/v1/projects/{project_id}/segments/{segment_id}/audio": {"GET"},
            "/api/v1/projects/{project_id}/segments/{segment_id}/alt-audio": {"GET"},
            "/api/v1/projects/{project_id}/segments/{segment_id}/edit-audio": {"POST"},
            "/api/v1/projects/{project_id}/segments/{segment_id}/preview-audio": {"POST"},
            "/api/v1/projects/{project_id}/segments/{segment_id}/emotion-audio": {"GET", "POST"},
            "/api/v1/projects/{project_id}/segments/{segment_id}/emotion-recording": {"GET", "POST"},
            "/api/v1/projects/{project_id}/export/full-with-edits": {"POST"},
            "/api/v1/projects/{project_id}/exports/final": {"GET"},
            "/api/v1/projects/{project_id}/exports/final/{filename}": {"GET", "DELETE"},
            "/api/v1/projects/{project_id}/exports/final/{filename}/open-location": {"POST"},
            "/api/v1/projects/{project_id}/subtitles": {"GET"},
            "/api/v1/projects/{project_id}/subtitles/generate-from-final": {"POST"},
            "/api/v1/projects/{project_id}/subtitles/{subtitle_id}": {"GET", "PUT", "DELETE"},
            "/api/v1/voices": {"GET", "POST"},
            "/api/v1/voices/{voice_id}": {"PUT", "DELETE"},
            "/api/v1/voices/{voice_id}/audio": {"GET", "POST"},
            "/api/v1/settings/platform": {"GET", "PUT"},
            "/api/v1/settings/tts-params": {"GET", "PUT"},
            "/api/v1/settings/tts-params/reset": {"POST"},
            "/api/v1/settings/tts-backend": {"GET", "PUT"},
            "/api/v1/settings/tts-service/status": {"GET"},
            "/api/v1/settings/tts-service/start": {"POST"},
            "/api/v1/settings/tts-service/load-model": {"POST"},
            "/api/v1/settings/tts-service/stop": {"POST"},
            "/api/v1/settings/seg-prompt-template": {"GET", "PUT"},
            "/api/v1/settings/seg-prompt-template/reset": {"POST"},
            "/api/v1/settings/gpu/unload-tts": {"POST"},
            "/api/v1/settings/gpu/unload-whisper": {"POST"},
            "/api/v1/settings/test-llm": {"POST"},
            "/api/v1/llm-providers": {"GET", "POST"},
            "/api/v1/llm-providers/{provider_id}": {"PUT", "DELETE"},
            "/api/v1/llm-providers/{provider_id}/set-default": {"POST"},
            "/api/v1/llm-configs": {"GET", "POST"},
            "/api/v1/llm-configs/{config_id}": {"PUT", "DELETE"},
            "/api/v1/llm-configs/{config_id}/set-default": {"POST"},
            "/api/v1/llm-configs/{config_id}/test": {"POST"},
            "/api/v1/qwen3tts/generate": {"POST"},
            "/api/v1/qwen3tts/audio": {"GET"},
        }

        missing_paths = sorted(path for path in expected if path not in route_methods)
        self.assertEqual(missing_paths, [], f"缺少按钮依赖的 API 路由: {missing_paths}")

        missing_methods: list[str] = []
        for path, required_methods in expected.items():
            methods = route_methods.get(path, set())
            if not required_methods.issubset(methods):
                missing = sorted(required_methods - methods)
                missing_methods.append(f"{path}: 缺少 {missing}, 实际 {sorted(methods)}")

        self.assertEqual(
            missing_methods,
            [],
            "按钮依赖的 API 路由方法不完整:\n" + "\n".join(missing_methods),
        )

    def test_frontend_api_paths_have_backend_routes(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")

        clear_config_caches()
        app = create_app()
        route_paths = {
            path
            for route in app.routes
            if (path := getattr(route, "path", None))
        }

        literal_paths = {
            m.group(1)
            for m in re.finditer(r"[\"'`](/api/v1/[^\"'`]+)[\"'`]", app_js)
        }

        # 这些路径是直连外部 TTS 服务（由 base_url + path 组成），不属于本后端路由。
        ignored_prefixes = (
            "/api/v1/tts/",
        )

        unmatched: list[str] = []
        for raw in sorted(literal_paths):
            path = raw.split("?", 1)[0]
            if path.startswith(ignored_prefixes):
                continue

            # 将模板字符串中的 ${...} 视作任意单段路径。
            escaped = re.escape(path)
            escaped = re.sub(r"\\\$\\\{[^}]+\\\}", r"[^/]+", escaped)
            pattern = re.compile(rf"^{escaped}$")

            if not any(pattern.match(route_path) for route_path in route_paths):
                unmatched.append(raw)

        self.assertEqual(
            unmatched,
            [],
            "前端引用了未注册的 /api/v1 路径: " + ", ".join(unmatched),
        )

    def test_segment_backend_inherits_segment_role_then_global(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")
        index_html = (webui_root / "index.html").read_text(encoding="utf-8")

        self.assertIn(
            "function getSegmentBackend(seg)",
            app_js,
            "应存在分段后端解析函数，统一处理显式后端与全局默认后端。",
        )

        self.assertIn(
            "const validBackends = new Set([\"indextts\", \"cosyvoice3\", \"qwen3tts\", \"qwen3tts_cv\"]);",
            app_js,
            "getSegmentBackend 应显式限制可用后端集合，避免非法值污染调用链路。",
        )
        segment_pos = app_js.index("validBackends.has(segmentBackend)")
        role_pos = app_js.index("validBackends.has(roleBackend)")
        global_pos = app_js.index("validBackends.has(globalBackend)")
        self.assertLess(segment_pos, role_pos)
        self.assertLess(role_pos, global_pos)
        self.assertIn(
            "(roles.value || []).find((item) => item && item.id === seg.role_id)",
            app_js,
            "未显式配置后端的分段应继承所属角色的后端。",
        )

        bad_patterns = [
            "seg.tts_backend || 'indextts'",
            "seg.tts_backend || \"indextts\"",
        ]
        for pattern in bad_patterns:
            self.assertNotIn(
                pattern,
                app_js,
                f"检测到硬编码后端回退写法，会导致分段未配置后端时错误强制使用 indextts: {pattern}",
            )

        self.assertIn(
            ":value=\"getSegmentBackend(seg)\"",
            index_html,
            "分段后端下拉应展示解析后的实际后端（分段、角色或全局默认）。",
        )
        self.assertIn(
            "isRoleMissingSpeakerAudio(seg.role_id, getSegmentBackend(seg))",
            app_js,
            "缺少参考音频判定必须使用与 WebSocket payload 相同的后端继承结果。",
        )

    def test_automatic_worker_switch_does_not_change_platform_default(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")
        index_html = (webui_root / "index.html").read_text(encoding="utf-8")

        self.assertIn("async function startTtsService(backend, options = {})", app_js)
        self.assertIn("persist_default: persistDefault", app_js)
        self.assertGreaterEqual(
            app_js.count("{ persistDefault: false }"),
            4,
            "AI 预览、批量及单段合成的自动 worker 切换均不应持久化平台默认后端。",
        )
        self.assertIn(
            '@click="startTtsService(backend.id)"',
            index_html,
            "用户在平台设置页主动启动后端时仍应使用默认的持久化行为。",
        )

    def test_destructive_resegmentation_requires_confirmation_and_keeps_failed_input(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")

        self.assertIn("function confirmReplacingExistingSegments(actionLabel)", app_js)
        self.assertIn('confirmReplacingExistingSegments("JSON 导入")', app_js)
        self.assertIn('confirmReplacingExistingSegments("LLM 重新分段")', app_js)

        import_match = re.search(
            r"async function importSegmentsFromJson\(\) \{(?P<body>.*?)\n\s*async function autoSegment",
            app_js,
            re.S,
        )
        self.assertIsNotNone(import_match)
        import_body = import_match.group("body")
        catch_pos = import_body.index("} catch (e) {")
        self.assertLess(import_body.index("closeImportJsonModal();"), catch_pos)
        self.assertNotIn("modals.value.importJson = false", import_body[catch_pos:])
        self.assertNotIn("jsonImportText.value =", import_body[catch_pos:])

    def test_all_engine_specific_tts_params_have_save_actions(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        index_html = (webui_root / "index.html").read_text(encoding="utf-8")

        self.assertIn('v-model.number="ttsParams.cosyvoice_speed"', index_html)
        self.assertIn('v-model="ttsParams.qwen3_default_language"', index_html)
        self.assertIn(
            '<div v-if="ttsParams && ttsParams.defaults" class="flex gap-2 mt-2 text-xs justify-end">',
            index_html,
            "保存/重载/重置按钮应位于通用参数区，而不是仅在 IndexTTS2 条件块内显示。",
        )

    def test_qwen_emotion_save_only_submits_changed_shared_fields(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")

        self.assertIn("const original = state.original || {};", app_js)
        self.assertIn("if (nextSpeaker !== (original.qwen3Speaker || 'vivian'))", app_js)
        self.assertIn("if (nextLanguage !== (original.qwen3Language || 'Auto'))", app_js)
        self.assertIn("if (!Object.keys(payload).length)", app_js)

    def test_role_voice_picker_excludes_explicit_emotion_voices(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")
        index_html = (webui_root / "index.html").read_text(encoding="utf-8")

        self.assertIn("const speakerVoices = computed(() => {", app_js)
        self.assertIn("tags.includes(voiceTypeLabels.speaker) || isLegacyUntypedVoice(voice)", app_js)
        self.assertIn("v-for=\"(v, idx) in speakerVoices\"", index_html)
        self.assertIn("isLegacyUntypedVoice(v)", index_html)
        self.assertIn("情感音色不能绑定为角色主音色", app_js)

    def test_refresh_segments_status_syncs_canonical_take_state(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")

        self.assertIn(
            "...updated,",
            app_js,
            "refreshSegmentsStatus 应同步后端完整 Take/slot 状态，避免前端残留过期的 A/B 视图。",
        )
        self.assertIn("/active-version/${nextSlot}", app_js)

    def test_tts_backend_switch_ui_warns_restart_when_runtime_differs(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")
        index_html = (webui_root / "index.html").read_text(encoding="utf-8")

        self.assertIn(
            "data.runtime_backend || data.backend || null",
            app_js,
            "前端应优先使用 runtime_backend 表示当前运行中的 TTS 后端。",
        )
        self.assertIn(
            "重启 TTS 服务后生效",
            app_js,
            "切换默认后端时，前端提示应明确说明需要重启 TTS 服务后才生效。",
        )
        self.assertIn(
            "ttsServiceNeedsRestart",
            index_html,
            "平台页应展示运行后端与默认配置不一致的提醒。",
        )

    def test_wait_for_tts_service_ready_uses_platform_routes_for_lan_clients(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")

        self.assertIn(
            'fetchJson("/api/v1/health"',
            app_js,
            "waitForTtsServiceReady 应通过主平台 health 接口轮询，避免局域网客户端直连 localhost TTS 子服务。",
        )
        self.assertIn(
            'fetchJson("/api/v1/settings/tts-service/load-model"',
            app_js,
            "waitForTtsServiceReady 应通过主平台接口代触发 TTS 模型加载，避免浏览器跨机器请求 127.0.0.1。",
        )
        self.assertNotIn(
            'const healthUrl = `${baseUrl}/health`',
            app_js,
            "前端不应再在浏览器中拼接 TTS 子服务 health URL。",
        )
        self.assertNotIn(
            'const loadModelUrl = `${baseUrl}/api/v1/tts/load-model`',
            app_js,
            "前端不应再在浏览器中拼接 TTS 子服务 load-model URL。",
        )

    def test_background_tts_polling_throttles_heavy_health_refresh(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")

        self.assertIn(
            "let healthRefreshPromise = null;",
            app_js,
            "前端应合并并发 health 请求，避免后台轮询与用户操作重复触发重型探测。",
        )
        self.assertIn(
            'await refreshHealth({ force: false, minIntervalMs: 30000 });',
            app_js,
            "TTS 后台轮询应对重型 health 做节流，避免高频重复探测。",
        )
        self.assertIn(
            'const data = await fetchJson("/api/v1/health", { headers: { "Content-Type": undefined } });',
            app_js,
            "详细 health 能力仍应保留，供平台配置与就绪检测使用。",
        )

    def test_project_activation_flow_is_unified_for_switch_create_delete(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")

        self.assertIn(
            "async function activateProject(id, options = {})",
            app_js,
            "项目切换/新建/删除后的加载流程应统一收敛到 activateProject，避免工作区状态清理逻辑分叉。",
        )
        self.assertIn(
            "await flushWaveformStatePersist(previousProjectId);",
            app_js,
            "切换到其他项目之前应等待当前项目的波形编辑状态持久化完成，避免快速切换时丢草稿。",
        )
        self.assertIn(
            'await activateProject(projects.value[0].id, { persistCurrent: false });',
            app_js,
            "删除当前项目后切换到下一个项目时，也应复用统一的项目激活流程。",
        )

    def test_synthesis_buttons_use_actual_target_and_reason_helpers(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")
        index_html = (webui_root / "index.html").read_text(encoding="utf-8")

        self.assertIn(
            "const hasBatchSynthesisTargets = computed(() => {",
            app_js,
            "前端应区分“存在可合成分段”和“当前确实有待合成目标”，避免按钮可点后才提示无任务。",
        )
        self.assertIn(
            "const batchSynthesizeDisabledReason = computed(() => {",
            app_js,
            "批量合成入口应给出明确的禁用原因，减少用户猜测为什么当前不能执行。",
        )
        self.assertIn(
            'function getSegmentSynthesizeTitle(seg) {',
            app_js,
            "单段合成按钮应存在统一提示函数，明确当前是可合成、需先绑定角色，还是已经在队列中。",
        )
        self.assertIn(
            ':disabled="!hasBatchSynthesisTargets || loading.autoSegment"',
            index_html,
            "批量合成按钮应按实际待处理目标决定可用性，而不是只看是否存在已绑角色的分段。",
        )
        self.assertIn(
            ':title="batchSynthesizeDisabledReason"',
            index_html,
            "批量合成按钮禁用时应显示具体原因。",
        )
        self.assertIn(
            ':title="getSegmentSynthesizeTitle(seg)"',
            index_html,
            "单段合成按钮应在悬停时说明当前为何能或不能执行。",
        )

    def test_segment_edit_feedback_clarifies_resynthesis_requirement(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")

        self.assertIn(
            'showToast(updated.audio_file ? "分段已保存，重新合成后生效" : "分段已保存");',
            app_js,
            "编辑分段文本后，如果当前已有音频，前端应明确提示需要重新合成后才会反映到声音上。",
        )
        self.assertIn(
            'showToast(updated.audio_file ? "分段角色已更新，重新合成后生效" : "分段角色已更新");',
            app_js,
            "修改分段角色后，如果当前已有音频，前端应提示需要重新合成，避免用户误以为声音会立刻变化。",
        )
        self.assertIn(
            'showToast(updated.audio_file ? "情感权重已更新，重新合成后生效" : "情感权重已更新");',
            app_js,
            "修改情感权重后，如果当前已有音频，前端应提示需要重新合成后生效。",
        )
        self.assertIn(
            "const suffix = updated.audio_file ? '，重新合成后生效' : '';",
            app_js,
            "切换分段 TTS 后端后，如果当前已有音频，前端应提示需要重新合成后生效。",
        )

    def test_platform_service_defaults_to_loopback_with_explicit_lan_mode(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        launcher_py = (repo_root / "tools" / "platform_launcher.py").read_text(encoding="utf-8")
        main_py = (repo_root / "dubbing_service" / "app" / "main.py").read_text(encoding="utf-8")
        batch = (repo_root / "start_platform.bat").read_text(encoding="utf-8")

        self.assertIn(
            'host: str = "127.0.0.1"',
            launcher_py,
            "启动器应默认仅监听本机，避免无鉴权服务意外暴露到局域网。",
        )
        self.assertIn(
            'choices=("127.0.0.1", "0.0.0.0")',
            launcher_py,
            "启动器应保留显式、受限的局域网绑定选项。",
        )
        self.assertIn(
            'host="127.0.0.1"',
            main_py,
            "直接运行 main.py 时也应默认仅监听本机。",
        )
        self.assertIn(
            'if /I "%~1"=="--lan" set "PLATFORM_HOST=0.0.0.0"',
            batch,
            "Mac/局域网访问必须由用户显式启用。",
        )

    def test_export_full_with_edits_syncs_waveform_state_before_request(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")

        self.assertIn(
            "await flushWaveformStatePersist(projectId);",
            app_js,
            "导出前应先等待当前 waveformStates 持久化，避免 full-with-edits 读到旧 edit_state。",
        )
        self.assertIn(
            "const hasEditDrafts = list.some((seg) => hasEffectiveWaveformEditDraft(seg));",
            app_js,
            "导出路径应基于当前波形编辑态判断，而不是依赖可能滞后的 seg.edit_state。",
        )
        self.assertIn(
            "function hasEffectiveWaveformEditDraft(seg)",
            app_js,
            "前端应存在统一的当前波形草稿判定函数，避免导出和 UI 各自用不同口径。",
        )

    def test_waveform_playback_uses_relative_preview_targets(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")

        self.assertIn(
            "function getSegmentPreviewSpeed(seg)",
            app_js,
            "前端应统一计算相对当前已生效音频的预览速度，避免正式应用后再被本地试听二次叠加。",
        )
        self.assertIn(
            "function getSegmentPreviewVolume(seg)",
            app_js,
            "前端应统一计算相对当前已生效音频的预览音量，避免正式应用后本地试听继续叠加。",
        )
        self.assertIn(
            "ws.setPlaybackRate(getSegmentPreviewSpeed(seg));",
            app_js,
            "WaveSurfer 初始化和实时调节都应使用相对预览速度，而不是直接拿绝对目标值。",
        )
        self.assertIn(
            "ws.setVolume(getSegmentPreviewVolume(seg));",
            app_js,
            "WaveSurfer 本地播放音量应基于相对预览值计算，避免误导用户对最终导出结果的判断。",
        )

    def test_delete_final_audio_refreshes_subtitle_state(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")

        self.assertIn(
            "if (subtitleEditing.value && subtitleEditing.value.audio_filename === fa.filename) {",
            app_js,
            "删除整书音频后，应关闭仍绑定该源音频的字幕编辑态，避免页面保留已失效字幕。",
        )
        self.assertIn(
            "await loadSubtitles();",
            app_js,
            "删除整书音频后应刷新字幕列表，确保被级联删除的字幕不会继续显示在前端。",
        )

    def test_clear_segment_edits_resets_speed_and_volume_drafts(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")

        self.assertIn(
            "st.speed = getSegmentBaseSpeed(seg);",
            app_js,
            "清空本段编辑时应将速度草稿恢复到当前已生效音频的基线值，而不是残留旧草稿。",
        )
        self.assertIn(
            "st.volume = getSegmentBaseVolume(seg);",
            app_js,
            "清空本段编辑时应将音量草稿恢复到当前已生效音频的基线值，而不是残留旧草稿。",
        )
        self.assertIn(
            "showToast(\"已清空本段编辑草稿\")",
            app_js,
            "清空按钮的提示文案应明确它清除的是编辑草稿，避免用户误解为删除音频本身。",
        )
        self.assertIn(
            "ws.setPlaybackRate(getSegmentPreviewSpeed(seg));",
            app_js,
            "清空本段编辑后应立即刷新波形本地预览倍率，避免页面继续以旧草稿试听。",
        )

    def test_initial_project_load_runs_resumable_detection(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")

        self.assertIn(
            "async function activateProject(id, options = {})",
            app_js,
            "项目激活流程应统一收敛到 activateProject，避免首次载入、手动切换和删除后切换各自维护一套状态逻辑。",
        )
        self.assertIn(
            "await detectResumableSynthesis(id);",
            app_js,
            "浏览器刷新后首次载入项目时也应执行断点恢复检测，避免只在手动切项目时才检查。",
        )
        self.assertIn(
            "const activeGenerating = Number(data.active_generating_count || 0);",
            app_js,
            "前端应识别 reset-stale 返回的 active_generating_count，区分真正卡死与仍在后台执行的 generating。",
        )
        self.assertIn(
            "if (Number.isFinite(activeGenerating) && activeGenerating > 0) {",
            app_js,
            "当后端仍报告活跃 generating 段落时，前端不应弹出可续传提示误导用户立即重跑。",
        )

    def test_tts_param_round_trip_keeps_engine_specific_values(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")

        self.assertIn(
            "function normalizeTtsParamsResponse(data, fallback, resetToDefaults = false)",
            app_js,
            "TTS 参数响应应经过统一归一化，避免各加载/保存/重置分支使用不同回退条件。",
        )
        self.assertIn(
            "normalizeTtsParamsResponse(data, ttsParams.value, false)",
            app_js,
            "保存响应缺少新字段时应保留用户刚提交的 CosyVoice 语速和 Qwen 默认语言。",
        )
        self.assertIn(
            "normalizeTtsParamsResponse(data, ttsParams.value, true)",
            app_js,
            "重置响应缺少新字段时应回退后端 defaults，而不是继续保留旧值。",
        )

    def test_mount_only_observes_tts_status_without_starting_service(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")

        mounted_match = re.search(
            r"onMounted\(async \(\) => \{(?P<body>.*?)\n\s*\}\);",
            app_js,
            re.S,
        )
        self.assertIsNotNone(mounted_match, "无法定位 Vue onMounted 生命周期")
        mounted_body = mounted_match.group("body")
        self.assertIn("await loadTtsServiceStatus();", mounted_body)
        self.assertNotIn(
            "startTtsService(",
            mounted_body,
            "打开页面只能读取 TTS 状态，不能产生启动模型服务的副作用。",
        )

    def test_websocket_timeout_and_socket_are_cleaned_up(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")

        self.assertIn("const activeSynthesisSockets = new Set();", app_js)
        self.assertIn("const synthesisSocketTimeouts = new Set();", app_js)
        self.assertIn("clearTimeout(timeoutId);", app_js)
        self.assertIn("synthesisSocketTimeouts.delete(timeoutId);", app_js)
        self.assertIn("cleanupSynthesisSockets();", app_js)

    def test_waveform_persistence_is_debounced_and_protected_on_unload(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")

        self.assertIn("function scheduleWaveformStatePersist(projectId, delayMs = 400)", app_js)
        self.assertIn("async function flushWaveformStatePersist(projectId)", app_js)
        self.assertIn("scheduleWaveformStatePersist(pid);", app_js)
        self.assertIn(
            'window.addEventListener("beforeunload", onBeforeUnloadWithPendingWaveformEdits);',
            app_js,
        )
        self.assertIn(
            'window.removeEventListener("beforeunload", onBeforeUnloadWithPendingWaveformEdits);',
            app_js,
        )

    def test_waveform_audio_edit_submission_has_per_segment_guard(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")
        index_html = (webui_root / "index.html").read_text(encoding="utf-8")

        self.assertIn("function isSegmentAudioEditPending(seg)", app_js)
        self.assertIn("if (isSegmentAudioEditPending(seg)) return;", app_js)
        self.assertIn(':disabled="isSegmentAudioEditPending(seg)"', index_html)

    def test_browser_dependencies_are_local_and_version_pinned(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        index_html = (webui_root / "index.html").read_text(encoding="utf-8")
        expected_assets = [
            "vendor/vue-3.5.39/vue.global.prod.js",
            "vendor/tailwindcss-3.4.17/tailwindcss.min.css",
            "vendor/wavesurfer-7.12.10/wavesurfer.min.js",
            "vendor/wavesurfer-7.12.10/regions.min.js",
        ]

        external_script_sources = re.findall(
            r'<script[^>]+src=["\']https?://[^"\']+["\']',
            index_html,
            re.I,
        )
        self.assertEqual(external_script_sources, [], "生产 WebUI 不应依赖公网脚本 CDN")
        self.assertNotIn("cdn.tailwindcss.com", index_html)
        self.assertNotIn("tailwindcss.min.js", index_html)
        for relative_path in expected_assets:
            self.assertIn(f'/web/{relative_path}', index_html)
            asset_path = webui_root / relative_path
            self.assertTrue(asset_path.is_file(), f"缺少本地浏览器依赖: {relative_path}")
            self.assertGreater(asset_path.stat().st_size, 1000, f"本地依赖内容异常: {relative_path}")

        self.assertTrue((webui_root / "vendor" / "VERSIONS.md").is_file())

    def test_platform_settings_support_dedicated_cosyvoice_url(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")
        index_html = (webui_root / "index.html").read_text(encoding="utf-8")

        self.assertIn('cosyvoice_base_url: ""', app_js)
        self.assertIn('platformSettings.value?.cosyvoice_base_url', app_js)
        self.assertIn('replacePort(base, ["8001", "8003", "8004"], "8002")', app_js)
        self.assertIn('v-model="platformSettings.cosyvoice_base_url"', index_html)
        self.assertIn('placeholder="http://127.0.0.1:8002"', index_html)

    def test_role_editor_saves_backend_voice_profile(self) -> None:
        webui_root = Path(__file__).resolve().parents[1] / "dubbing_service" / "webui"
        app_js = (webui_root / "app.js").read_text(encoding="utf-8")
        index_html = (webui_root / "index.html").read_text(encoding="utf-8")

        for field_name in (
            "tts_backend",
            "qwen3_speaker",
            "qwen3_language",
            "voice_description",
            "style_instruction",
        ):
            self.assertIn(field_name, app_js)
            self.assertIn(f"forms.role.{field_name}", index_html)

        self.assertIn(
            "`/api/v1/projects/${currentProject.value.id}/roles/${savedRoleId}/voice-profile`",
            app_js,
        )
        self.assertIn('method: "PATCH"', app_js)
        self.assertIn('value="">继承平台默认</option>', index_html)


if __name__ == "__main__":
    unittest.main()
