# Dify集成指南

## 在Dify工作流中使用IndexTTS2 API

### 方法1: HTTP请求节点

#### 配置步骤

1. **添加HTTP请求节点**

2. **基础配置**:
   ```
   URL: http://localhost:8001/api/v1/tts/synthesize
   Method: POST
   Headers: 
     Content-Type: application/json
   ```

3. **请求Body**:
   ```json
   {
     "text": "{{text}}",
     "speaker_audio": "{{speaker_audio_path}}",
     "emotion_mode": 0,
     "temperature": 0.8,
     "save_file": true
   }
   ```

4. **变量映射**:
   - `{{text}}`: 从上游节点获取要合成的文本
   - `{{speaker_audio_path}}`: 参考音频文件的服务器路径

### 方法2: 代码节点

使用Python代码节点直接调用：

```python
import requests

def main(params):
    text = params.get('text', '')
    speaker_audio = params.get('speaker_audio', '')
    
    response = requests.post(
        'http://localhost:8001/api/v1/tts/synthesize',
        json={
            'text': text,
            'speaker_audio': speaker_audio,
            'emotion_mode': 0,
            'save_file': True
        }
    )
    
    result = response.json()
    return {
        'audio_file': result.get('audio_file'),
        'success': result.get('success'),
        'duration': result.get('duration_seconds')
    }
```

## 工作流场景

### 场景1: AI对话转语音

```
[用户输入] → [LLM生成回复] → [IndexTTS2 API] → [返回语音]
```

配置：
1. **LLM节点** 生成对话文本
2. **HTTP请求节点** 调用TTS API
3. **输出节点** 返回音频URL

### 场景2: 情感化内容创作

```
[输入文本] → [情感分析] → [选择情感向量] → [TTS合成] → [输出]
```

示例配置：
```json
{
  "text": "{{content}}",
  "speaker_audio": "narrator_voice.wav",
  "emotion_mode": 2,
  "emotion_vector": "{{emotion_vector}}",
  "emotion_alpha": 0.7
}
```

情感向量映射：
- 积极文本: `[0.8, 0, 0, 0, 0, 0, 0.2, 0]`
- 消极文本: `[0, 0, 0.7, 0, 0, 0.3, 0, 0]`
- 中性文本: `[0, 0, 0, 0, 0, 0, 0, 1]`

### 场景3: 批量有声内容生成

```
[读取文章列表] → [循环处理] → [TTS API] → [保存音频] → [生成播放列表]
```

## 环境变量配置

在Dify中设置环境变量：

```
TTS_API_URL=http://localhost:8001
TTS_SPEAKER_AUDIO=/path/to/default_speaker.wav
TTS_TEMPERATURE=0.8
```

## 错误处理

在HTTP请求节点后添加条件节点：

```
IF response.success == true:
    继续后续流程
ELSE:
    发送错误通知
    返回错误信息
```

## 性能优化建议

1. **缓存参考音频**: 将常用的参考音频上传到服务器，使用路径而非base64
2. **批量处理**: 使用循环节点批量生成，避免频繁调用
3. **异步处理**: 对于长文本，考虑异步生成

## 完整示例工作流

```json
{
  "nodes": [
    {
      "id": "input",
      "type": "input",
      "data": {
        "title": "输入文本",
        "variables": ["text"]
      }
    },
    {
      "id": "llm",
      "type": "llm",
      "data": {
        "model": "gpt-4",
        "prompt": "优化以下文本使其更适合语音播报：{{text}}"
      }
    },
    {
      "id": "tts",
      "type": "http-request",
      "data": {
        "url": "http://localhost:8001/api/v1/tts/synthesize",
        "method": "POST",
        "body": {
          "text": "{{llm.output}}",
          "speaker_audio": "/prompts/narrator.wav",
          "emotion_mode": 0,
          "temperature": 0.8
        }
      }
    },
    {
      "id": "output",
      "type": "output",
      "data": {
        "audio_url": "{{tts.audio_file}}",
        "text": "{{llm.output}}"
      }
    }
  ]
}
```

## 常见问题

**Q: 如何处理长文本？**
A: IndexTTS2会自动分段处理，但建议控制在500字以内以获得最佳效果。

**Q: 可以并发调用吗？**
A: 可以，但需要注意GPU显存限制。

**Q: 如何选择合适的情感向量？**
A: 可以先在API文档中测试不同组合，找到最适合的配置。
