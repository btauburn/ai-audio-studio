import sys
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import service.server as server


class _NoInferModel:
    def normalize_emo_vec(self, vector, apply_bias=True):
        return vector

    def infer(self, *args, **kwargs):
        raise AssertionError("infer should not be called for invalid input")


class TestServiceServerRobustness(unittest.TestCase):
    def test_parse_server_args_ignores_unknown_flags(self) -> None:
        with patch.object(sys, "argv", ["server.py", "--port", "9001", "--unknown-flag", "x"]):
            args = server._parse_server_args()
        self.assertEqual(args.port, 9001)

    def test_synthesize_file_invalid_emotion_vector_length_returns_400(self) -> None:
        with patch.object(server, "tts_model", _NoInferModel()), patch.object(
            server, "_load_tts_model_if_needed", return_value=None
        ):
            with TestClient(server.app) as client:
                resp = client.post(
                    "/api/v1/tts/synthesize-file",
                    data={
                        "text": "test",
                        "emotion_mode": "2",
                        "emotion_vector": "[1, 0]",
                    },
                    files={
                        "speaker_audio": ("speaker.wav", b"RIFF\x00\x00\x00\x00WAVE", "audio/wav"),
                    },
                )

        self.assertEqual(resp.status_code, 400)
        self.assertIn("情感向量必须包含8个浮点数", resp.text)

    def test_synthesize_file_invalid_emotion_vector_json_returns_400(self) -> None:
        with patch.object(server, "tts_model", _NoInferModel()), patch.object(
            server, "_load_tts_model_if_needed", return_value=None
        ):
            with TestClient(server.app) as client:
                resp = client.post(
                    "/api/v1/tts/synthesize-file",
                    data={
                        "text": "test",
                        "emotion_mode": "2",
                        "emotion_vector": "not-a-json-array",
                    },
                    files={
                        "speaker_audio": ("speaker.wav", b"RIFF\x00\x00\x00\x00WAVE", "audio/wav"),
                    },
                )

        self.assertEqual(resp.status_code, 400)
        self.assertIn("情感向量格式错误", resp.text)


if __name__ == "__main__":
    unittest.main()
