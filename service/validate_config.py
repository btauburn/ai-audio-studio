#!/usr/bin/env python3
"""
Claude Desktop MCP配置验证工具
"""

import os
import sys
import json
from pathlib import Path


def find_claude_config():
    """查找Claude Desktop配置文件"""
    if sys.platform == 'win32':
        config_path = Path(os.environ.get('APPDATA', '')) / 'Claude' / 'claude_desktop_config.json'
    elif sys.platform == 'darwin':
        config_path = Path.home() / 'Library' / 'Application Support' / 'Claude' / 'claude_desktop_config.json'
    else:
        config_path = Path.home() / '.config' / 'Claude' / 'claude_desktop_config.json'
    
    return config_path


def validate_config(config_path):
    """验证配置文件"""
    print(f"检查配置文件: {config_path}")
    print()
    
    if not config_path.exists():
        print("❌ 配置文件不存在")
        print()
        print("需要创建配置文件，内容示例：")
        print_example_config()
        return False
    
    # 读取配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ 配置文件JSON格式错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return False
    
    # 检查结构
    if 'mcpServers' not in config:
        print("❌ 配置缺少 'mcpServers' 字段")
        return False
    
    if 'indextts2' not in config['mcpServers']:
        print("⚠️  配置中未找到 'indextts2' 服务")
        print()
        print("需要添加IndexTTS2配置，示例：")
        print_example_config()
        return False
    
    # 检查IndexTTS2配置
    indextts2_config = config['mcpServers']['indextts2']
    
    # 检查command
    if 'command' not in indextts2_config:
        print("❌ 缺少 'command' 字段")
        return False
    
    command = str(indextts2_config['command'])
    command_name = Path(command).name.lower()
    if command_name in ['python', 'python.exe', 'python3', 'python3.exe', 'uv', 'uv.exe']:
        print(f"✅ command: {command}")
    else:
        print(f"⚠️  command值为 '{command}'，确保这是有效的命令（推荐使用 uv run）")
    
    # 检查args
    if 'args' not in indextts2_config:
        print("❌ 缺少 'args' 字段")
        return False
    
    args = indextts2_config['args']
    if not args or len(args) == 0:
        print("❌ 'args' 不能为空")
        return False

    script_path: Path
    if command_name in ['uv', 'uv.exe']:
        # uv run 配置：args 中会包含 run/--directory/.../mcp_server.py
        detected = None
        for arg in args:
            if isinstance(arg, str) and arg.lower().endswith('.py'):
                detected = Path(arg)
                break
        if detected is None:
            print("❌ 未在 args 中找到 .py 脚本路径（uv run 配置示例见下方）")
            print()
            print_example_config()
            return False
        script_path = detected
    else:
        # python / python3 配置：args[0] 通常为脚本路径
        script_path = Path(str(args[0]))

    if not script_path.exists():
        print(f"❌ MCP服务器脚本不存在: {script_path}")
        print()
        print("请确保路径正确，当前配置的脚本路径是：")
        print(f"  {script_path}")
        return False
    print(f"✅ MCP服务器脚本存在: {script_path}")
    
    # 检查env
    if 'env' in indextts2_config:
        env = indextts2_config['env']
        if 'PYTHONPATH' in env:
            pythonpath = Path(env['PYTHONPATH'])
            if pythonpath.exists():
                print(f"✅ PYTHONPATH存在: {pythonpath}")
            else:
                print(f"⚠️  PYTHONPATH不存在: {pythonpath}")
        else:
            print("⚠️  建议添加 PYTHONPATH 环境变量")
    else:
        print("⚠️  建议添加 'env' 字段设置PYTHONPATH")
    
    print()
    print("✅ 配置文件验证通过！")
    print()
    print("下一步:")
    print("1. 完全退出Claude Desktop")
    print("2. 重新启动Claude Desktop")
    print("3. 在对话中询问：你有哪些工具可以使用？")
    print("4. 查看是否有IndexTTS2相关工具")
    
    return True


def print_example_config():
    """打印示例配置"""
    project_root = Path(__file__).parent.parent.absolute()
    mcp_script = project_root / 'service' / 'mcp_server.py'
    python_exe = project_root / '.venv' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')

    example = {
        "mcpServers": {
            "indextts2": {
                "command": str(python_exe),
                "args": [
                    str(mcp_script),
                    "--model_dir",
                    str(project_root / 'models' / 'index_tts'),
                ],
                "env": {
                    "PYTHONPATH": str(project_root),
                    "PYTHONUTF8": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "HF_HOME": str(project_root / 'models' / '.runtime_cache'),
                    "HF_HUB_CACHE": str(project_root / 'models' / '.runtime_cache' / 'hub'),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                }
            }
        }
    }
    
    print(json.dumps(example, indent=2, ensure_ascii=False))


def main():
    """主函数"""
    print("=" * 60)
    print("Claude Desktop MCP配置验证工具")
    print("=" * 60)
    print()
    
    config_path = find_claude_config()
    success = validate_config(config_path)
    
    if not success:
        print()
        print("配置文件路径:")
        print(f"  {config_path}")
        print()
        print("可以手动创建或修改此文件")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
