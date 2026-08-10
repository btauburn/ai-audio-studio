#!/usr/bin/env python3
"""
IndexTTS2 MCP服务测试脚本
模拟MCP客户端测试服务器功能
"""

import json
import os
import subprocess
import sys
import time
from typing import Dict, Any


class MCPTestClient:
    """MCP测试客户端"""
    
    def __init__(self, server_script: str = "mcp_server.py"):
        self.server_script = server_script
        self.process = None
        self.request_id = 0
        # 获取示例音频的绝对路径
        self.sample_audio = os.path.abspath("../examples/sample_prompt.wav")
    
    def start_server(self):
        """启动MCP服务器"""
        print("🚀 启动MCP服务器...")
        self.process = subprocess.Popen(
            [sys.executable, self.server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        time.sleep(3)  # 等待服务器初始化
        print("✅ 服务器已启动\n")
    
    def stop_server(self):
        """停止MCP服务器"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("\n🛑 服务器已停止")
    
    def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """发送JSON-RPC请求"""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }
        
        # 发送请求
        request_json = json.dumps(request)
        print(f"📤 发送请求: {method}")
        self.process.stdin.write(request_json + "\n")
        self.process.stdin.flush()
        
        # 接收响应
        response_line = self.process.stdout.readline()
        response = json.loads(response_line)
        
        return response
    
    def test_initialize(self):
        """测试初始化"""
        print("=" * 60)
        print("测试1: 初始化")
        print("=" * 60)
        
        response = self.send_request("initialize", {
            "protocolVersion": "1.0.0",
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        })
        
        if "result" in response:
            print("✅ 初始化成功")
            print(f"   服务器: {response['result']['serverInfo']['name']}")
            print(f"   版本: {response['result']['serverInfo']['version']}")
        else:
            print(f"❌ 初始化失败: {response.get('error')}")
        print()
    
    def test_list_tools(self):
        """测试列出工具"""
        print("=" * 60)
        print("测试2: 列出可用工具")
        print("=" * 60)
        
        response = self.send_request("tools/list")
        
        if "result" in response:
            tools = response["result"]["tools"]
            print(f"✅ 找到 {len(tools)} 个工具:")
            for tool in tools:
                print(f"   - {tool['name']}: {tool['description'][:50]}...")
        else:
            print(f"❌ 失败: {response.get('error')}")
        print()
    
    def test_clone_voice(self):
        """测试音色克隆"""
        print("=" * 60)
        print("测试3: 音色克隆")
        print("=" * 60)
        
        response = self.send_request("tools/call", {
            "name": "clone_voice",
            "arguments": {
                "text": "这是MCP测试生成的语音",
                "speaker_audio": self.sample_audio
            }
        })
        
        if "result" in response:
            content = response["result"]["content"][0]["text"]
            result = json.loads(content)
            if result.get("success"):
                print("✅ 音色克隆成功")
                print(f"   文件: {result['output_file']}")
            else:
                print(f"❌ 失败: {result}")
        else:
            print(f"❌ 失败: {response.get('error')}")
        print()
    
    def test_emotional_speech(self):
        """测试情感语音"""
        print("=" * 60)
        print("测试4: 情感语音生成")
        print("=" * 60)
        
        emotions = ["happy", "sad", "calm"]
        
        for emotion in emotions:
            response = self.send_request("tools/call", {
                "name": "emotional_speech",
                "arguments": {
                    "text": f"这是{emotion}情感的测试",
                    "speaker_audio": self.sample_audio,
                    "emotion": emotion,
                    "emotion_intensity": 0.8
                }
            })
            
            if "result" in response:
                content = response["result"]["content"][0]["text"]
                result = json.loads(content)
                if result.get("success"):
                    print(f"✅ {emotion}: {result['output_file']}")
                else:
                    print(f"❌ {emotion}: 失败")
            else:
                print(f"❌ {emotion}: {response.get('error')}")
        print()
    
    def test_mixed_emotion(self):
        """测试混合情感"""
        print("=" * 60)
        print("测试5: 混合情感语音")
        print("=" * 60)
        
        response = self.send_request("tools/call", {
            "name": "mixed_emotion_speech",
            "arguments": {
                "text": "这真是让人又惊又喜",
                "speaker_audio": self.sample_audio,
                "emotion_mix": {
                    "happy": 0.6,
                    "surprised": 0.4
                }
            }
        })
        
        if "result" in response:
            content = response["result"]["content"][0]["text"]
            result = json.loads(content)
            if result.get("success"):
                print("✅ 混合情感成功")
                print(f"   情感配比: {result['emotion_mix']}")
                print(f"   文件: {result['output_file']}")
            else:
                print("❌ 失败")
        else:
            print(f"❌ 失败: {response.get('error')}")
        print()
    
    def test_batch_synthesize(self):
        """测试批量合成"""
        print("=" * 60)
        print("测试6: 批量语音合成")
        print("=" * 60)
        
        response = self.send_request("tools/call", {
            "name": "batch_synthesize",
            "arguments": {
                "text_list": [
                    "第一段测试文本",
                    "第二段测试文本",
                    "第三段测试文本"
                ],
                "speaker_audio": self.sample_audio
            }
        })
        
        if "result" in response:
            content = response["result"]["content"][0]["text"]
            result = json.loads(content)
            print("✅ 批量合成完成")
            print(f"   总数: {result['total']}")
            print(f"   成功: {result['success_count']}")
            print(f"   失败: {result['failed_count']}")
        else:
            print(f"❌ 失败: {response.get('error')}")
        print()
    
    def test_list_generated(self):
        """测试列出生成的音频"""
        print("=" * 60)
        print("测试7: 列出生成的音频文件")
        print("=" * 60)
        
        response = self.send_request("tools/call", {
            "name": "list_generated_audio",
            "arguments": {
                "limit": 5
            }
        })
        
        if "result" in response:
            content = response["result"]["content"][0]["text"]
            result = json.loads(content)
            print(f"✅ 找到 {result['count']} 个文件:")
            for audio in result['audio_files']:
                print(f"   - {audio['filename']} ({audio['size_mb']} MB)")
        else:
            print(f"❌ 失败: {response.get('error')}")
        print()
    
    def test_synthesize_speech(self):
        """测试完整语音合成"""
        print("=" * 60)
        print("测试8: 完整语音合成（带参数）")
        print("=" * 60)
        
        response = self.send_request("tools/call", {
            "name": "synthesize_speech",
            "arguments": {
                "text": "这是带有自定义参数的语音合成测试",
                "speaker_audio_path": self.sample_audio,
                "emotion_mode": "speaker",
                "temperature": 0.85,
                "output_filename": "test_synthesize.wav"
            }
        })
        
        if "result" in response:
            content = response["result"]["content"][0]["text"]
            result = json.loads(content)
            if result.get("success"):
                print("✅ 合成成功")
                print(f"   文件: {result['output_file']}")
                print(f"   情感模式: {result['details']['emotion_mode']}")
            else:
                print("❌ 失败")
        else:
            print(f"❌ 失败: {response.get('error')}")
        print()


def main():
    """主函数"""
    print("""
╔════════════════════════════════════════════════╗
║     IndexTTS2 MCP 服务测试套件                 ║
╚════════════════════════════════════════════════╝
    """)
    
    client = MCPTestClient()
    
    try:
        # 启动服务器
        client.start_server()
        
        # 运行测试
        client.test_initialize()
        client.test_list_tools()
        client.test_clone_voice()
        client.test_emotional_speech()
        client.test_mixed_emotion()
        client.test_batch_synthesize()
        client.test_list_generated()
        client.test_synthesize_speech()
        
        print("=" * 60)
        print("✅ 所有测试完成！")
        print("=" * 60)
        print("\n生成的文件位置: mcp_outputs/")
        print("服务器日志: mcp_server.log")
        
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止服务器
        client.stop_server()


if __name__ == "__main__":
    main()
