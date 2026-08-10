#!/usr/bin/env python3
"""
MCP服务自动配置工具
帮助用户快速配置Claude Desktop
"""

import os
import sys
import json
from pathlib import Path
import shutil


def get_claude_config_path():
    """获取Claude配置文件路径"""
    if sys.platform == 'win32':
        return Path(os.environ.get('APPDATA', '')) / 'Claude' / 'claude_desktop_config.json'
    elif sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support' / 'Claude' / 'claude_desktop_config.json'
    else:
        return Path.home() / '.config' / 'Claude' / 'claude_desktop_config.json'


def create_mcp_config():
    """创建MCP配置"""
    project_root = Path(__file__).parent.parent.absolute()
    
    # 使用service目录
    mcp_script = project_root / 'service' / 'mcp_server.py'

    python_exe = project_root / '.venv' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    model_dir = project_root / 'models' / 'index_tts'

    config = {
        "indextts2": {
            "command": str(python_exe),
            "args": [
                str(mcp_script),
                "--model_dir",
                str(model_dir),
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
    
    return config


def setup_mcp():
    """设置MCP配置"""
    print("=" * 60)
    print("IndexTTS2 MCP服务配置工具")
    print("=" * 60)
    print()
    
    config_path = get_claude_config_path()
    print(f"配置文件位置: {config_path}")
    print()
    
    # 检查配置文件目录
    config_dir = config_path.parent
    if not config_dir.exists():
        print("⚠️  配置目录不存在，创建中...")
        config_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 已创建目录: {config_dir}")
    
    # 读取或创建配置
    existing_config = {}
    if config_path.exists():
        print("✅ 配置文件已存在")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                existing_config = json.load(f)
            print("✅ 成功读取现有配置")
            
            # 备份现有配置
            backup_path = config_path.with_suffix('.json.backup')
            shutil.copy2(config_path, backup_path)
            print(f"✅ 已备份到: {backup_path}")
        except Exception as e:
            print(f"⚠️  读取配置文件失败: {e}")
            print("   将创建新配置")
    else:
        print("⚠️  配置文件不存在，将创建新文件")
    
    # 创建IndexTTS2配置
    indextts2_config = create_mcp_config()
    
    # 合并配置
    if 'mcpServers' not in existing_config:
        existing_config['mcpServers'] = {}
    
    # 检查是否已有配置
    if 'indextts2' in existing_config['mcpServers']:
        print()
        print("⚠️  检测到已有IndexTTS2配置:")
        print(json.dumps(existing_config['mcpServers']['indextts2'], indent=2, ensure_ascii=False))
        print()
        response = input("是否覆盖现有配置？(y/n): ").strip().lower()
        if response != 'y':
            print("❌ 已取消配置")
            return False
    
    # 更新配置
    existing_config['mcpServers']['indextts2'] = indextts2_config['indextts2']
    
    # 写入配置文件
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(existing_config, f, indent=2, ensure_ascii=False)
        print()
        print("✅ 配置已成功写入！")
        print()
        print("新的IndexTTS2配置:")
        print(json.dumps(indextts2_config, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"❌ 写入配置失败: {e}")
        return False
    
    # 显示后续步骤
    print()
    print("=" * 60)
    print("配置完成！接下来:")
    print("=" * 60)
    print("1. 完全退出Claude Desktop（确保进程完全关闭）")
    print("2. 重新启动Claude Desktop")
    print("3. 在对话中询问：你有哪些工具可以使用？")
    print("4. 查看是否显示IndexTTS2相关工具")
    print()
    print("如果没有显示工具:")
    print("- 运行验证工具: .venv\\Scripts\\python.exe service\\validate_config.py")
    print("- 查看日志: service/mcp_server.log")
    print()
    
    return True


def main():
    """主函数"""
    try:
        success = setup_mcp()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n已取消配置")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 配置过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
