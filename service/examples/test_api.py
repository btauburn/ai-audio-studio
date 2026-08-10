#!/usr/bin/env python3
"""
IndexTTS2 API测试脚本
演示如何调用API进行语音合成
"""

import requests
import base64
import os
import sys

API_BASE = "http://localhost:8001"

def test_health():
    """测试健康检查"""
    print("=" * 50)
    print("测试1: 健康检查")
    print("=" * 50)
    
    response = requests.get(f"{API_BASE}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()


def test_synthesize_with_file_path():
    """测试语音合成 - 使用文件路径"""
    print("=" * 50)
    print("测试2: 语音合成（文件路径）")
    print("=" * 50)
    
    # 使用项目中的示例音频（转换为绝对路径）
    speaker_audio_path = os.path.abspath("../../examples/sample_prompt.wav")
    
    if not os.path.exists(speaker_audio_path):
        print(f"❌ 参考音频不存在: {speaker_audio_path}")
        return
    
    request_data = {
        "text": "你好，这是IndexTTS2的API测试。今天天气真不错！",
        "speaker_audio": speaker_audio_path,
        "emotion_mode": 0,
        "temperature": 0.8,
        "save_file": True,
        "return_base64": False
    }
    
    response = requests.post(f"{API_BASE}/api/v1/tts/synthesize", json=request_data)
    
    print(f"状态码: {response.status_code}")
    result = response.json()
    
    if result.get("success"):
        print("✅ 合成成功！")
        print(f"   文件: {result['audio_file']}")
        print(f"   耗时: {result['duration_seconds']}秒")
    else:
        print(f"❌ 合成失败: {result.get('message')}")
    print()


def test_synthesize_with_base64():
    """测试语音合成 - 使用base64编码"""
    print("=" * 50)
    print("测试3: 语音合成（base64编码）")
    print("=" * 50)
    
    # 使用绝对路径
    speaker_audio_path = os.path.abspath("../../examples/sample_prompt.wav")
    
    if not os.path.exists(speaker_audio_path):
        print(f"❌ 参考音频不存在: {speaker_audio_path}")
        return
    
    # 读取音频并转为base64
    with open(speaker_audio_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode()
    
    request_data = {
        "text": "使用base64编码传输音频数据",
        "speaker_audio": audio_base64,
        "emotion_mode": 0,
        "return_base64": True,
        "save_file": False
    }
    
    response = requests.post(f"{API_BASE}/api/v1/tts/synthesize", json=request_data)
    result = response.json()
    
    if result.get("success"):
        print("✅ 合成成功！")
        print("   返回了base64编码的音频")
        print(f"   耗时: {result['duration_seconds']}秒")
        
        # 保存base64音频
        if result.get("audio_base64"):
            output_path = "../outputs/test_output_base64.wav"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(result["audio_base64"]))
            print(f"   已保存到: {output_path}")
    else:
        print("❌ 合成失败")
    print()


def test_synthesize_with_emotion_vector():
    """测试情感向量控制"""
    print("=" * 50)
    print("测试4: 情感向量控制")
    print("=" * 50)
    
    # 使用绝对路径
    speaker_audio_path = os.path.abspath("../../examples/sample_prompt.wav")
    
    if not os.path.exists(speaker_audio_path):
        print("❌ 参考音频不存在")
        return
    
    # 测试不同情感
    emotions = {
        "高兴": [1.0, 0, 0, 0, 0, 0, 0, 0],
        "愤怒": [0, 1.0, 0, 0, 0, 0, 0, 0],
        "悲伤": [0, 0, 1.0, 0, 0, 0, 0, 0],
        "平静": [0, 0, 0, 0, 0, 0, 0, 1.0]
    }
    
    for emotion_name, emotion_vec in emotions.items():
        request_data = {
            "text": f"这是{emotion_name}的情感测试",
            "speaker_audio": speaker_audio_path,
            "emotion_mode": 2,  # 使用情感向量
            "emotion_vector": emotion_vec,
            "emotion_alpha": 0.8,
            "save_file": True
        }
        
        response = requests.post(f"{API_BASE}/api/v1/tts/synthesize", json=request_data)
        result = response.json()
        
        if result.get("success"):
            print(f"✅ {emotion_name}: {result['audio_file']}")
        else:
            print(f"❌ {emotion_name}: 失败")
    print()


def test_file_upload():
    """测试文件上传方式"""
    print("=" * 50)
    print("测试5: 文件上传方式")
    print("=" * 50)
    
    # 使用绝对路径
    speaker_audio_path = os.path.abspath("../../examples/sample_prompt.wav")
    
    if not os.path.exists(speaker_audio_path):
        print("❌ 参考音频不存在")
        return
    
    with open(speaker_audio_path, "rb") as f:
        files = {"speaker_audio": ("voice.wav", f, "audio/wav")}
        data = {
            "text": "这是文件上传方式的测试",
            "emotion_mode": 0,
            "temperature": 0.8
        }
        
        response = requests.post(
            f"{API_BASE}/api/v1/tts/synthesize-file",
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        output_path = "../outputs/test_output_upload.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        print("✅ 文件上传方式成功！")
        print(f"   已保存到: {output_path}")
    else:
        print(f"❌ 失败: {response.text}")
    print()


def test_separate_voice_emotion():
    """测试分离音色和情感参考"""
    print("=" * 50)
    print("测试6: 分离音色和情感参考")
    print("=" * 50)
    
    # 使用绝对路径
    speaker_audio_path = os.path.abspath("../../examples/sample_prompt.wav")
    
    if not os.path.exists(speaker_audio_path):
        print("❌ 参考音频不存在")
        return
    
    # 测试：使用同一个音频作为音色和情感参考（作为对比）
    print("\n方式1: 使用情感参考音频模式")
    print("说明: 可以用A音频的音色 + B音频的情感")
    print("这里演示用相同音频，实际使用时可以是不同音频\n")
    
    request_data = {
        "text": "这是分离音色和情感的测试",
        "speaker_audio": speaker_audio_path,      # 音色来源
        "emotion_mode": 1,                          # 使用情感参考音频
        "emotion_audio": speaker_audio_path,       # 情感来源（可以是不同的音频）
        "emotion_alpha": 0.65,                     # 情感强度
        "save_file": True
    }
    
    response = requests.post(f"{API_BASE}/api/v1/tts/synthesize", json=request_data)
    result = response.json()
    
    if result.get("success"):
        print(f"✅ 情感音频模式成功: {result['audio_file']}")
        print("   音色来源: speaker_audio")
        print("   情感来源: emotion_audio")
        print(f"   情感强度: {request_data['emotion_alpha']}")
    else:
        print("❌ 失败")
    print()


def test_list_prompts():
    """测试列出参考音频"""
    print("=" * 50)
    print("测试7: 列出参考音频")
    print("=" * 50)
    
    response = requests.get(f"{API_BASE}/api/v1/prompts/list")
    result = response.json()
    
    print(f"找到 {result['count']} 个参考音频:")
    for prompt in result['prompts']:
        print(f"  - {prompt['filename']} ({prompt['size_mb']} MB)")
    print()


if __name__ == "__main__":
    print("\n🧪 IndexTTS2 API 测试套件\n")
    
    try:
        test_health()
        test_synthesize_with_file_path()
        test_synthesize_with_base64()
        test_synthesize_with_emotion_vector()
        test_file_upload()
        test_separate_voice_emotion()
        test_list_prompts()
        
        print("=" * 50)
        print("✅ 所有测试完成！")
        print("=" * 50)
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到API服务器")
        print("   请确保服务器正在运行: uv run server.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        sys.exit(1)
