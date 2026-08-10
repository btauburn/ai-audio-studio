#!/usr/bin/env python3
"""
IndexTTS2 API Python客户端封装
提供简洁的Python接口调用API
"""

import requests
import base64
from typing import Optional, List
from pathlib import Path


class IndexTTS2Client:
    """IndexTTS2 API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip('/')
        
    def health_check(self) -> dict:
        """健康检查"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def synthesize(
        self,
        text: str,
        speaker_audio: str,
        emotion_mode: int = 0,
        emotion_audio: Optional[str] = None,
        emotion_vector: Optional[List[float]] = None,
        emotion_text: Optional[str] = None,
        emotion_alpha: float = 0.65,
        temperature: float = 0.8,
        save_file: bool = True,
        return_base64: bool = False,
        **kwargs
    ) -> dict:
        """
        语音合成
        
        Args:
            text: 要合成的文本
            speaker_audio: 说话人参考音频路径或base64
            emotion_mode: 情感控制模式 (0-3)
            emotion_audio: 情感参考音频
            emotion_vector: 情感向量 [喜,怒,哀,惧,厌恶,低落,惊喜,平静]
            emotion_text: 情感描述文本
            emotion_alpha: 情感权重
            temperature: 温度参数
            save_file: 是否保存文件
            return_base64: 是否返回base64
            **kwargs: 其他高级参数
        """
        # 处理文件路径
        if Path(speaker_audio).exists():
            with open(speaker_audio, 'rb') as f:
                speaker_audio = base64.b64encode(f.read()).decode()
        
        if emotion_audio and Path(emotion_audio).exists():
            with open(emotion_audio, 'rb') as f:
                emotion_audio = base64.b64encode(f.read()).decode()
        
        payload = {
            "text": text,
            "speaker_audio": speaker_audio,
            "emotion_mode": emotion_mode,
            "emotion_audio": emotion_audio,
            "emotion_vector": emotion_vector,
            "emotion_text": emotion_text,
            "emotion_alpha": emotion_alpha,
            "temperature": temperature,
            "save_file": save_file,
            "return_base64": return_base64,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/tts/synthesize",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def synthesize_file(
        self,
        text: str,
        speaker_audio_path: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        语音合成并下载文件
        
        Args:
            text: 要合成的文本
            speaker_audio_path: 说话人参考音频文件路径
            output_path: 输出文件路径
            **kwargs: 其他参数
        
        Returns:
            输出文件路径
        """
        with open(speaker_audio_path, 'rb') as f:
            files = {'speaker_audio': f}
            data = {'text': text, **kwargs}
            
            response = requests.post(
                f"{self.base_url}/api/v1/tts/synthesize-file",
                files=files,
                data=data
            )
        
        response.raise_for_status()
        
        if output_path is None:
            output_path = f"tts_output_{int(time.time())}.wav"
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return output_path
    
    def clone_voice(
        self,
        text: str,
        voice_sample: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        快速音色克隆（简化接口）
        
        Args:
            text: 要说的话
            voice_sample: 参考音频路径
            output_path: 输出路径
        """
        return self.synthesize_file(text, voice_sample, output_path)
    
    def emotional_speech(
        self,
        text: str,
        speaker_audio: str,
        emotion: str,
        intensity: float = 0.8,
        output_path: Optional[str] = None
    ) -> dict:
        """
        生成情感语音
        
        Args:
            text: 文本
            speaker_audio: 说话人音频
            emotion: 情感类型 (happy/angry/sad/afraid/disgusted/melancholic/surprised/calm)
            intensity: 情感强度 (0.0-1.0)
            output_path: 输出路径
        """
        emotion_map = {
            "happy": [1, 0, 0, 0, 0, 0, 0, 0],
            "angry": [0, 1, 0, 0, 0, 0, 0, 0],
            "sad": [0, 0, 1, 0, 0, 0, 0, 0],
            "afraid": [0, 0, 0, 1, 0, 0, 0, 0],
            "disgusted": [0, 0, 0, 0, 1, 0, 0, 0],
            "melancholic": [0, 0, 0, 0, 0, 1, 0, 0],
            "surprised": [0, 0, 0, 0, 0, 0, 1, 0],
            "calm": [0, 0, 0, 0, 0, 0, 0, 1]
        }
        
        if emotion not in emotion_map:
            raise ValueError(f"未知情感: {emotion}")
        
        emotion_vector = [v * intensity for v in emotion_map[emotion]]
        
        return self.synthesize(
            text=text,
            speaker_audio=speaker_audio,
            emotion_mode=2,
            emotion_vector=emotion_vector,
            emotion_alpha=0.8
        )
    
    def list_prompts(self) -> List[dict]:
        """列出所有参考音频"""
        response = requests.get(f"{self.base_url}/api/v1/prompts/list")
        response.raise_for_status()
        return response.json()['prompts']
    
    def upload_prompt(self, name: str, audio_path: str) -> dict:
        """上传参考音频"""
        with open(audio_path, 'rb') as f:
            files = {'audio': f}
            data = {'name': name}
            
            response = requests.post(
                f"{self.base_url}/api/v1/prompts/upload",
                files=files,
                data=data
            )
        
        response.raise_for_status()
        return response.json()


# ==================== 使用示例 ====================
if __name__ == "__main__":
    import time
    
    # 创建客户端
    client = IndexTTS2Client()
    
    # 检查服务状态
    print("检查服务状态...")
    health = client.health_check()
    print(f"✅ 服务正常: {health['model_version']}\n")
    
    # 示例1: 基础音色克隆
    print("示例1: 基础音色克隆")
    result = client.clone_voice(
        text="你好，这是Python客户端的测试",
        voice_sample="../../examples/sample_prompt.wav",
        output_path="client_output_1.wav"
    )
    print(f"✅ 生成成功: {result}\n")
    
    # 示例2: 情感语音生成
    print("示例2: 情感语音生成")
    result = client.emotional_speech(
        text="我今天非常高兴！",
        speaker_audio="../../examples/sample_prompt.wav",
        emotion="happy",
        intensity=0.9
    )
    if result['success']:
        print(f"✅ 高兴情感: {result['audio_file']}\n")
    
    # 示例3: 自定义情感向量
    print("示例3: 自定义情感向量")
    result = client.synthesize(
        text="混合情感测试：既高兴又有点惊讶",
        speaker_audio="../../examples/sample_prompt.wav",
        emotion_mode=2,
        emotion_vector=[0.6, 0, 0, 0, 0, 0, 0.4, 0],  # 60%高兴 + 40%惊讶
        temperature=0.85
    )
    if result['success']:
        print(f"✅ 混合情感: {result['audio_file']}\n")
    
    print("所有示例完成！")
