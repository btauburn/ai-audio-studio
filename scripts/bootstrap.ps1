[CmdletBinding()]
param(
    [ValidateSet("All", "Platform", "Qwen", "CosyVoice")]
    [string]$Runtime = "All"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $ProjectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not available on PATH. Install uv before running bootstrap."
}

$env:UV_LINK_MODE = "copy"
uv python install 3.12.13 3.10.19

if ($Runtime -in @("All", "Platform")) {
    uv sync --locked --all-extras
    uv pip check --python (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
}

if ($Runtime -in @("All", "Qwen")) {
    uv sync --project (Join-Path $ProjectRoot "environments\qwen3tts") --locked --no-dev
    uv pip check --python (Join-Path $ProjectRoot "environments\qwen3tts\.venv\Scripts\python.exe")
}

if ($Runtime -in @("All", "CosyVoice")) {
    uv sync --project (Join-Path $ProjectRoot "environments\cosyvoice") --locked --no-dev
    uv pip check --python (Join-Path $ProjectRoot "environments\cosyvoice\.venv\Scripts\python.exe")
}

Write-Host "Environment bootstrap completed. Run scripts\platform_doctor.py before startup."
