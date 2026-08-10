@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%~dp0"
echo ====================================
echo AI 有声工作室 - 一键启动器
echo ====================================
echo.
echo 正在执行启动预检...
echo.

set "HF_HOME=%CD%\models\.runtime_cache"
set "HF_HUB_CACHE=%CD%\models\.runtime_cache\hub"
set "MODELSCOPE_CACHE=%CD%\models\.modelscope_cache"
set "COSYVOICE_WETEXT_DIR=%CD%\models\cosyvoice3\wetext"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "NO_PROXY=127.0.0.1,localhost"
set "no_proxy=127.0.0.1,localhost"

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 平台虚拟环境不存在。
    echo 请先在 PowerShell 中运行: .\scripts\bootstrap.ps1
    echo.
    pause
    exit /b 1
)

"%CD%\.venv\Scripts\python.exe" "%CD%\scripts\platform_doctor.py" --startup
if errorlevel 1 (
    echo.
    echo [错误] 启动预检失败。请根据上方提示修复后重试。
    echo.
    pause
    exit /b 1
)

if /I "%~1"=="--check" (
    echo [OK] Startup preflight passed.
    exit /b 0
)

set "PLATFORM_HOST=127.0.0.1"
if /I "%~1"=="--lan" set "PLATFORM_HOST=0.0.0.0"

echo.
echo 正在启动平台界面...
echo.

"%CD%\.venv\Scripts\python.exe" "%CD%\tools\platform_launcher.py" --host "%PLATFORM_HOST%"
