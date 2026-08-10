@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
echo ====================================
echo IndexTTS2 API 服务启动脚本
echo ====================================
echo.

cd /d "%~dp0"

REM 固定使用项目内模型和缓存；运行时保持离线，不隐式下载模型。
set "PROJECT_ROOT=%~dp0.."
set "HF_HOME=%PROJECT_ROOT%\models\.runtime_cache"
set "HF_HUB_CACHE=%PROJECT_ROOT%\models\.runtime_cache\hub"
set "MODELSCOPE_CACHE=%PROJECT_ROOT%\models\.modelscope_cache"
set "COSYVOICE_WETEXT_DIR=%PROJECT_ROOT%\models\cosyvoice3\wetext"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"

set "PRECISION=%~1"
if "%PRECISION%"=="" set "PRECISION=fp16"

if /I "%PRECISION%"=="fp32" (
    set "FP16_ARG=--no-fp16"
    set "PRECISION_LABEL=FP32(全精度)"
) else (
    set "FP16_ARG=--fp16"
    set "PRECISION_LABEL=FP16(半精度)"
)

echo.
echo 启动API服务器...
echo 精度模式: %PRECISION_LABEL%
echo 访问 http://localhost:8001/docs 查看API文档
echo 按 Ctrl+C 停止服务
echo.

if not exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    echo [错误] 平台虚拟环境不存在，请先运行 scripts\bootstrap.ps1。
    pause
    exit /b 1
)

"%PROJECT_ROOT%\.venv\Scripts\python.exe" "%PROJECT_ROOT%\service\server.py" --host 127.0.0.1 --port 8001 --model_dir "%PROJECT_ROOT%\models\index_tts" %FP16_ARG% --no-cuda-kernel

pause
