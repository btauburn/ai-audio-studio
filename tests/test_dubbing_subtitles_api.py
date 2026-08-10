import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.main import create_app
from dubbing_service.app.models.subtitles import SubtitleFile, SubtitleSegment, SubtitleWord
from dubbing_service.app.storage.subtitles import save_subtitle


class TestSubtitlesApiGenerateFromFinal(unittest.TestCase):
    def _make_subtitle(self, *, project_id: str, subtitle_id: str, audio_rel_path: str, audio_filename: str) -> SubtitleFile:
        return SubtitleFile(
            id=subtitle_id,
            project_id=project_id,
            audio_rel_path=audio_rel_path,
            audio_filename=audio_filename,
            language="zh",
            segments=[
                SubtitleSegment(
                    index=0,
                    start=0.0,
                    end=1.0,
                    text="你好",
                    asr_text="你好",
                    words=[SubtitleWord(index=0, text="你", start=0.0, end=0.5)],
                )
            ],
        )

    def _setup_project(self, tmp_root: str, project_id: str) -> Path:
        project_root = Path(tmp_root) / project_id
        (project_root / "audio" / "final").mkdir(parents=True, exist_ok=True)
        return project_root

    def test_generate_from_final_prefers_timeline_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()

            project_id = "p1"
            project_root = self._setup_project(tmp_root, project_id)

            audio_filename = "full_edit.wav"
            (project_root / "audio" / "final" / audio_filename).write_bytes(b"")

            (project_root / "dialogue_timeline").mkdir(parents=True, exist_ok=True)
            timeline_path = project_root / "dialogue_timeline" / f"dialogue_timeline_{Path(audio_filename).stem}.json"
            timeline_path.write_text(
                json.dumps({"items": [{"start": 0.0, "end": 1.0, "text": "你好"}]}),
                encoding="utf-8",
            )

            app = create_app()

            def _fake_generate(*, subtitle_id: str, project_id: str, audio_path: Path, audio_rel_path: str, audio_filename: str, language=None, task=None):
                return self._make_subtitle(
                    project_id=project_id,
                    subtitle_id=subtitle_id,
                    audio_rel_path=audio_rel_path,
                    audio_filename=audio_filename,
                )

            with patch("dubbing_service.app.routers.subtitles.generate_subtitle_id", return_value="s1"), patch(
                "dubbing_service.app.routers.subtitles.FastWhisperSubtitleProvider"
            ) as ProviderCls, patch(
                "dubbing_service.app.routers.subtitles.apply_timeline_items_to_subtitle"
            ) as apply_timeline, patch(
                "dubbing_service.app.routers.subtitles.save_subtitle"
            ) as save_subtitle, patch(
                "dubbing_service.app.routers.subtitles.write_text_atomic"
            ) as write_text:
                ProviderCls.return_value.generate_subtitles.side_effect = _fake_generate
                apply_timeline.side_effect = lambda sub, items: sub
                save_subtitle.return_value = None
                write_text.return_value = None

                from fastapi.testclient import TestClient

                client = TestClient(app)
                resp = client.post(
                    f"/api/v1/projects/{project_id}/subtitles/generate-from-final",
                    json={
                        "audio_filename": audio_filename,
                        "language": "zh",
                        "task": "transcribe",
                        "model_size": "large-v3",
                        "text_source": "segments",
                    },
                )

                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertTrue(data.get("success"))
                self.assertEqual(data.get("subtitle_id"), "s1")
                self.assertEqual(data.get("json_rel_path"), "subtitles/s1.json")
                self.assertEqual(data.get("srt_rel_path"), "subtitles/s1.srt")

                apply_timeline.assert_called_once()

    def test_generate_from_final_fallback_to_segments_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()

            project_id = "p1"
            project_root = self._setup_project(tmp_root, project_id)

            audio_filename = "full_edit.wav"
            (project_root / "audio" / "final" / audio_filename).write_bytes(b"")

            seg_audio_rel = "audio/segments/seg_0001.wav"
            seg_audio_path = project_root / seg_audio_rel
            seg_audio_path.parent.mkdir(parents=True, exist_ok=True)
            seg_audio_path.write_bytes(b"")

            (project_root / "segments.json").write_text(
                json.dumps(
                    [
                        {
                            "id": "seg1",
                            "index": 0,
                            "text": "你好",
                            "status": "done",
                            "audio_file": seg_audio_rel,
                            "emotion_mode": "text",
                            "emotion_text": None,
                            "emotion_vector": None,
                            "emotion_audio": None,
                            "emotion_alpha": 0.65,
                            "playback_speed": 1.0,
                            "volume_gain": 1.0,
                            "chapter_id": None,
                            "role_id": None,
                            "edit_state": None,
                            "llm_raw_output": None,
                            "created_at": "2020-01-01T00:00:00Z",
                            "updated_at": "2020-01-01T00:00:00Z"
                        }
                    ]
                ),
                encoding="utf-8",
            )

            app = create_app()

            def _fake_generate(*, subtitle_id: str, project_id: str, audio_path: Path, audio_rel_path: str, audio_filename: str, language=None, task=None):
                return self._make_subtitle(
                    project_id=project_id,
                    subtitle_id=subtitle_id,
                    audio_rel_path=audio_rel_path,
                    audio_filename=audio_filename,
                )

            with patch("dubbing_service.app.routers.subtitles.generate_subtitle_id", return_value="s1"), patch(
                "dubbing_service.app.routers.subtitles.FastWhisperSubtitleProvider"
            ) as ProviderCls, patch(
                "dubbing_service.app.routers.subtitles.apply_source_text_to_subtitle_by_alignment"
            ) as apply_align, patch(
                "dubbing_service.app.routers.subtitles.save_subtitle"
            ) as save_subtitle, patch(
                "dubbing_service.app.routers.subtitles.write_text_atomic"
            ) as write_text:
                ProviderCls.return_value.generate_subtitles.side_effect = _fake_generate
                apply_align.side_effect = lambda sub, src: (sub, 1.0)
                save_subtitle.return_value = None
                write_text.return_value = None

                from fastapi.testclient import TestClient

                client = TestClient(app)
                resp = client.post(
                    f"/api/v1/projects/{project_id}/subtitles/generate-from-final",
                    json={
                        "audio_filename": audio_filename,
                        "language": "zh",
                        "task": "transcribe",
                        "model_size": "large-v3",
                        "text_source": "segments",
                    },
                )

                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertTrue(data.get("success"))
                apply_align.assert_called_once()

    def test_generate_from_final_text_source_asr_skips_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()

            project_id = "p1"
            project_root = self._setup_project(tmp_root, project_id)

            audio_filename = "full_edit.wav"
            (project_root / "audio" / "final" / audio_filename).write_bytes(b"")

            app = create_app()

            def _fake_generate(*, subtitle_id: str, project_id: str, audio_path: Path, audio_rel_path: str, audio_filename: str, language=None, task=None):
                return self._make_subtitle(
                    project_id=project_id,
                    subtitle_id=subtitle_id,
                    audio_rel_path=audio_rel_path,
                    audio_filename=audio_filename,
                )

            with patch("dubbing_service.app.routers.subtitles.generate_subtitle_id", return_value="s1"), patch(
                "dubbing_service.app.routers.subtitles.FastWhisperSubtitleProvider"
            ) as ProviderCls, patch(
                "dubbing_service.app.routers.subtitles.apply_timeline_items_to_subtitle"
            ) as apply_timeline, patch(
                "dubbing_service.app.routers.subtitles.apply_source_text_to_subtitle_by_alignment"
            ) as apply_align, patch(
                "dubbing_service.app.routers.subtitles.save_subtitle"
            ) as save_subtitle, patch(
                "dubbing_service.app.routers.subtitles.write_text_atomic"
            ) as write_text:
                ProviderCls.return_value.generate_subtitles.side_effect = _fake_generate
                save_subtitle.return_value = None
                write_text.return_value = None

                from fastapi.testclient import TestClient

                client = TestClient(app)
                resp = client.post(
                    f"/api/v1/projects/{project_id}/subtitles/generate-from-final",
                    json={
                        "audio_filename": audio_filename,
                        "language": "zh",
                        "task": "transcribe",
                        "model_size": "large-v3",
                        "text_source": "asr",
                    },
                )

                self.assertEqual(resp.status_code, 200)
                self.assertTrue(resp.json().get("success"))
                apply_timeline.assert_not_called()
                apply_align.assert_not_called()

    def test_delete_final_audio_cascades_related_subtitles_and_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()

            project_id = "p1"
            project_root = self._setup_project(tmp_root, project_id)
            (project_root / "project.json").write_text("{}", encoding="utf-8")

            audio_filename = "full_edit.wav"
            audio_path = project_root / "audio" / "final" / audio_filename
            audio_path.write_bytes(b"fake-audio")

            timeline_dir = project_root / "dialogue_timeline"
            timeline_dir.mkdir(parents=True, exist_ok=True)
            named_timeline = timeline_dir / f"dialogue_timeline_{Path(audio_filename).stem}.json"
            named_timeline.write_text(json.dumps({"items": []}), encoding="utf-8")
            latest_timeline = project_root / "dialogue_timeline.json"
            latest_timeline.write_text(
                json.dumps({"audio_filename": audio_filename, "items": []}),
                encoding="utf-8",
            )

            subtitle = self._make_subtitle(
                project_id=project_id,
                subtitle_id="s1",
                audio_rel_path=f"audio/final/{audio_filename}",
                audio_filename=audio_filename,
            )
            subtitle.json_rel_path = "subtitles/s1.json"
            subtitle.srt_rel_path = "subtitles/s1.srt"
            save_subtitle(project_id, subtitle)
            srt_path = project_root / "subtitles" / "s1.srt"
            srt_path.parent.mkdir(parents=True, exist_ok=True)
            srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")

            app = create_app()

            from fastapi.testclient import TestClient

            client = TestClient(app)
            resp = client.delete(
                f"/api/v1/projects/{project_id}/exports/final/{audio_filename}",
            )

            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json(), {"success": True})
            self.assertFalse(audio_path.exists())
            self.assertFalse(named_timeline.exists())
            self.assertFalse(latest_timeline.exists())
            self.assertFalse((project_root / "subtitles" / "s1.json").exists())
            self.assertFalse(srt_path.exists())

            list_resp = client.get(f"/api/v1/projects/{project_id}/subtitles")
            self.assertEqual(list_resp.status_code, 200)
            self.assertEqual(list_resp.json(), [])

    def test_generate_from_final_rejects_illegal_audio_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            os.environ["DUBBING_PROJECTS_ROOT"] = tmp_root
            clear_config_caches()

            project_id = "p1"
            project_root = self._setup_project(tmp_root, project_id)
            (project_root / "audio" / "final").mkdir(parents=True, exist_ok=True)

            app = create_app()

            from fastapi.testclient import TestClient

            client = TestClient(app)
            resp = client.post(
                f"/api/v1/projects/{project_id}/subtitles/generate-from-final",
                json={
                    "audio_filename": "../evil.wav",
                    "language": "zh",
                    "task": "transcribe",
                    "model_size": "large-v3",
                    "text_source": "segments",
                },
            )
            self.assertEqual(resp.status_code, 400)
            self.assertIn("非法文件名", resp.text)


if __name__ == "__main__":
    unittest.main()
