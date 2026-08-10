#!/usr/bin/env python3
"""
测试 /api/v1/tts/synthesize-file 端点的完整参数支持
验证文件上传方式是否支持所有高级参数
"""

import os
import sys
import json
import requests

API_BASE = "http://localhost:8001"


def test_file_upload_with_emotion_vector():
    """测试文件上传 + 情感向量"""
    print("=" * 60)
    print("测试1: 文件上传 + 情感向量（emotion_vector）")
    print("=" * 60)
    
    speaker_audio_path = os.path.abspath("../../examples/sample_prompt.wav")
    
    if not os.path.exists(speaker_audio_path):
        print(f"❌ 参考音频不存在: {speaker_audio_path}")
        return
    
    with open(speaker_audio_path, "rb") as f:
        files = {"speaker_audio": ("voice.wav", f, "audio/wav")}
        data = {
            "text": "这是测试情感向量的语音合成",
            "emotion_mode": 2,  # 使用情感向量
            "emotion_vector": json.dumps([0.8, 0, 0, 0, 0, 0, 0.2, 0]),  # 80%快乐 + 20%平静
            "emotion_alpha": 0.75,
        }
        
        response = requests.post(
            f"{API_BASE}/api/v1/tts/synthesize-file",
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        output_path = "../outputs/test_file_emotion_vector.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"✅ 成功！文件已保存到: {output_path}")
        print(f"   大小: {len(response.content)} bytes")
    else:
        print(f"❌ 失败: {response.status_code}")
        print(f"   响应: {response.text}")
    print()


def test_file_upload_with_advanced_params():
    """测试文件上传 + 所有高级生成参数"""
    print("=" * 60)
    print("测试2: 文件上传 + 完整高级参数")
    print("=" * 60)
    
    speaker_audio_path = os.path.abspath("../../examples/sample_prompt.wav")
    
    if not os.path.exists(speaker_audio_path):
        print(f"❌ 参考音频不存在: {speaker_audio_path}")
        return
    
    with open(speaker_audio_path, "rb") as f:
        files = {"speaker_audio": ("voice.wav", f, "audio/wav")}
        data = {
            "text": "测试完整高级参数：温度、采样、惩罚系数等。",
            "emotion_mode": 2,
            "emotion_vector": json.dumps([0.6, 0, 0, 0, 0, 0, 0.4, 0]),
            "emotion_alpha": 0.7,
            "use_random": False,
            # 高级生成参数
            "max_text_tokens_per_segment": 100,
            "max_mel_tokens": 1200,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.85,
            "top_k": 15,
            "num_beams": 3,
            "repetition_penalty": 1.8,
            "length_penalty": 0.0,
        }
        
        print("📊 使用参数:")
        print(f"   temperature: {data['temperature']}")
        print(f"   top_k: {data['top_k']}")
        print(f"   top_p: {data['top_p']}")
        print(f"   repetition_penalty: {data['repetition_penalty']}")
        print(f"   max_mel_tokens: {data['max_mel_tokens']}")
        
        response = requests.post(
            f"{API_BASE}/api/v1/tts/synthesize-file",
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        output_path = "../outputs/test_file_advanced_params.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"✅ 成功！文件已保存到: {output_path}")
        print(f"   大小: {len(response.content)} bytes")
    else:
        print(f"❌ 失败: {response.status_code}")
        print(f"   响应: {response.text}")
    print()


def test_file_upload_with_emotion_audio():
    """测试文件上传 + 分离音色和情感"""
    print("=" * 60)
    print("测试3: 文件上传 + 分离音色和情感（两个文件）")
    print("=" * 60)
    
    speaker_audio_path = os.path.abspath("../../examples/sample_prompt.wav")
    emotion_audio_path = speaker_audio_path  # 演示用，实际应该是不同的文件
    
    if not os.path.exists(speaker_audio_path):
        print(f"❌ 参考音频不存在: {speaker_audio_path}")
        return
    
    with open(speaker_audio_path, "rb") as sf, open(emotion_audio_path, "rb") as ef:
        files = {
            "speaker_audio": ("voice.wav", sf, "audio/wav"),
            "emotion_audio": ("emotion.wav", ef, "audio/wav")
        }
        data = {
            "text": "测试分离音色和情感的文件上传",
            "emotion_mode": 1,  # 使用情感参考音频
            "emotion_alpha": 0.8,
            "temperature": 0.8,
            "top_k": 20,
        }
        
        response = requests.post(
            f"{API_BASE}/api/v1/tts/synthesize-file",
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        output_path = "../outputs/test_file_separate_emotion.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"✅ 成功！文件已保存到: {output_path}")
        print(f"   大小: {len(response.content)} bytes")
    else:
        print(f"❌ 失败: {response.status_code}")
        print(f"   响应: {response.text}")
    print()


def main():
    print("\n" + "=" * 60)
    print("🧪 测试 synthesize-file 端点的完整参数支持")
    print("=" * 60)
    print()
    
    # 检查服务器是否运行
    try:
        response = requests.get(f"{API_BASE}/health", timeout=2)
        if response.status_code != 200:
            print("❌ 服务器未响应，请先启动服务器")
            print("   uv run server.py")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print("❌ 无法连接到服务器，请先启动服务器")
        print("   uv run server.py")
        sys.exit(1)
    
    print("✅ 服务器运行正常\n")
    
    # 运行测试
    test_file_upload_with_emotion_vector()
    test_file_upload_with_advanced_params()
    test_file_upload_with_emotion_audio()
    
    print("=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
