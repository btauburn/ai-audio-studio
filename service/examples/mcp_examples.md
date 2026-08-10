# IndexTTS2 MCP 使用示例

## 基础用法

### 1. 快速克隆音色
```
用户：生成语音"你好世界"，参考音频examples/sample_prompt.wav
Claude：[调用clone_voice] 已生成
```

### 2. 情感语音
```
用户：生成高兴的语音"今天天气真好"
Claude：[调用emotional_speech, emotion=happy]
```

### 3. 批量生成
```
用户：批量生成三段文本的语音
Claude：[调用batch_synthesize]
```

### 4. 混合情感
```
用户：60%高兴+40%惊喜
Claude：[调用mixed_emotion_speech]
```

## 完整示例请查看MCP_README.md
