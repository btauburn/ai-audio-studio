#!/bin/bash
# IndexTTS2 API cURL调用示例

API_BASE="http://localhost:8001"

echo "======================================"
echo "IndexTTS2 API cURL示例"
echo "======================================"
echo ""

# 1. 健康检查
echo "1. 健康检查"
curl -X GET "${API_BASE}/health"
echo -e "\n\n"

# 2. 基础语音合成（文件上传）
echo "2. 基础语音合成（文件上传）"
curl -X POST "${API_BASE}/api/v1/tts/synthesize-file" \
  -F "text=你好，这是cURL调用测试" \
  -F "speaker_audio=@../../examples/sample_prompt.wav" \
  -F "emotion_mode=0" \
  --output output_curl.wav
echo "已保存到 output_curl.wav"
echo -e "\n"

# 3. 使用情感参考音频
echo "3. 使用情感参考音频"
curl -X POST "${API_BASE}/api/v1/tts/synthesize-file" \
  -F "text=这是带情感的语音" \
  -F "speaker_audio=@../../examples/sample_prompt.wav" \
  -F "emotion_audio=@../../examples/sample_prompt.wav" \
  -F "emotion_mode=1" \
  -F "emotion_alpha=0.7" \
  --output output_emotion.wav
echo "已保存到 output_emotion.wav"
echo -e "\n"

# 4. JSON方式调用
echo "4. JSON方式调用"
curl -X POST "${API_BASE}/api/v1/tts/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "JSON方式调用测试",
    "speaker_audio": "../../examples/sample_prompt.wav",
    "emotion_mode": 0,
    "temperature": 0.8,
    "save_file": true,
    "return_base64": false
  }'
echo -e "\n\n"

# 5. 高级参数调用
echo "5. 高级参数调用"
curl -X POST "${API_BASE}/api/v1/tts/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "使用高级参数进行精细控制",
    "speaker_audio": "../../examples/sample_prompt.wav",
    "emotion_mode": 0,
    "temperature": 0.9,
    "top_p": 0.85,
    "top_k": 40,
    "repetition_penalty": 12.0,
    "max_mel_tokens": 1200
  }'
echo -e "\n\n"

# 6. 列出参考音频
echo "6. 列出参考音频"
curl -X GET "${API_BASE}/api/v1/prompts/list"
echo -e "\n\n"

echo "======================================"
echo "测试完成！"
echo "======================================"
