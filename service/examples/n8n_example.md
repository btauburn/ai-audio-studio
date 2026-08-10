# n8n调用示例

## 方法1: HTTP Request节点（推荐）

### 配置步骤

1. **添加HTTP Request节点**

2. **配置节点**:
   - Method: `POST`
   - URL: `http://localhost:8001/api/v1/tts/synthesize-file`
   - Authentication: None

3. **配置Body**:
   - Body Content Type: `Form-Data`
   - 添加参数:
     - `text`: `{{ $json.text }}` （从上一节点获取）
     - `speaker_audio`: 选择文件（Binary Data）
     - `emotion_mode`: `0`
     - `temperature`: `0.8`

4. **设置响应**:
   - Response Format: `File`
   - Put Output in Field: `audio`

### 完整工作流示例

```json
{
  "nodes": [
    {
      "parameters": {
        "values": {
          "string": [
            {
              "name": "text",
              "value": "你好，这是n8n工作流生成的语音"
            }
          ]
        }
      },
      "name": "设置文本",
      "type": "n8n-nodes-base.set",
      "position": [250, 300]
    },
    {
      "parameters": {
        "url": "http://localhost:8001/api/v1/tts/synthesize-file",
        "method": "POST",
        "sendBody": true,
        "contentType": "multipart-form-data",
        "bodyParameters": {
          "parameters": [
            {
              "name": "text",
              "value": "={{ $json.text }}"
            },
            {
              "name": "speaker_audio",
              "value": "={{ $binary.audio }}"
            },
            {
              "name": "emotion_mode",
              "value": "0"
            }
          ]
        }
      },
      "name": "调用TTS API",
      "type": "n8n-nodes-base.httpRequest",
      "position": [450, 300]
    }
  ]
}
```

## 方法2: JSON Body

### 配置

```json
{
  "parameters": {
    "url": "http://localhost:8001/api/v1/tts/synthesize",
    "method": "POST",
    "sendBody": true,
    "contentType": "application/json",
    "bodyParameters": {
      "parameters": [
        {
          "name": "text",
          "value": "={{ $json.text }}"
        },
        {
          "name": "speaker_audio",
          "value": "={{ $json.speaker_audio_path }}"
        },
        {
          "name": "emotion_mode",
          "value": 0
        },
        {
          "name": "save_file",
          "value": true
        }
      ]
    }
  }
}
```

## 常见使用场景

### 场景1: 批量语音生成

1. **Spreadsheet节点** - 读取文本列表
2. **Split In Batches** - 分批处理
3. **HTTP Request** - 调用TTS API
4. **Write Binary File** - 保存音频文件

### 场景2: 定时语音播报

1. **Cron节点** - 定时触发
2. **Code节点** - 生成播报文本
3. **HTTP Request** - 调用TTS API
4. **发送到播放系统**

### 场景3: 情感对话生成

1. **Webhook** - 接收对话文本
2. **IF节点** - 判断情感
3. **HTTP Request** - 使用对应情感向量生成语音
4. **返回音频文件**

## 提示

- 使用文件路径方式更快，适合本地部署
- 使用base64方式适合跨网络调用
- 建议在n8n中配置错误处理节点
