$ErrorActionPreference = 'Stop'
$studioRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $studioRoot
& (Join-Path $studioRoot '.venv\Scripts\python.exe') -m dubbing_service.app.audiobook.delivery start
exit $LASTEXITCODE
