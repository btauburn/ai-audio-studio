@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%~dp0"
set "PROJECT_ROOT=%~dp0.."
set "HF_HOME=%PROJECT_ROOT%\models\.runtime_cache"
set "HF_HUB_CACHE=%PROJECT_ROOT%\models\.runtime_cache\hub"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "NO_PROXY=127.0.0.1,localhost"
set "no_proxy=127.0.0.1,localhost"

if not exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    echo [错误] 平台虚拟环境不存在，请先运行 scripts\bootstrap.ps1。
    pause
    exit /b 1
)

if /I "%~1"=="--check" (
    "%PROJECT_ROOT%\.venv\Scripts\python.exe" "%PROJECT_ROOT%\service\mcp_server.py" --help >nul 2>nul
    if errorlevel 1 exit /b 1
    echo [OK] MCP entrypoint preflight passed.
    exit /b 0
)

echo ====================================
echo IndexTTS2 MCP 服务器
echo ====================================
echo.
echo 注意：MCP服务器通常不需要手动启动
echo 它会被Claude Desktop自动调用
echo.
echo 此脚本仅用于测试和调试
echo.
pause

echo 启动MCP服务器...
echo 日志将保存到 mcp_server.log
echo.

"%PROJECT_ROOT%\.venv\Scripts\python.exe" "%PROJECT_ROOT%\service\mcp_server.py"

pause
