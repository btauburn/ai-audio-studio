[CmdletBinding()]
param(
    [ValidateSet("Full", "Tts", "Core", "Subtitles", "Sfx")]
    [string]$Profile = "Full",

    [switch]$AcceptThirdPartyLicenses,

    [ValidateSet("community", "enterprise")]
    [string]$StableAudioLicenseTier = "community",

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Provisioner = Join-Path $ProjectRoot "scripts\model_provision.py"
$ManifestRoot = Join-Path $ProjectRoot "models\manifests"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Platform environment is missing. Run .\scripts\bootstrap.ps1 -Runtime All first."
}

$profiles = @{
    Full = @(
        "index_tts2.json",
        "qwen3_tts_customvoice.json",
        "qwen3_tts_voicedesign.json",
        "cosyvoice3.json",
        "whisper_distil_large_v3.json",
        "whisper_large_v3.json",
        "whisper_large_v3_turbo.json",
        "whisperx_zh_alignment.json",
        "stable_audio_3_small_sfx.json"
    )
    Tts = @(
        "index_tts2.json",
        "qwen3_tts_customvoice.json",
        "qwen3_tts_voicedesign.json",
        "cosyvoice3.json"
    )
    Core = @(
        "qwen3_tts_customvoice.json",
        "qwen3_tts_voicedesign.json",
        "cosyvoice3.json"
    )
    Subtitles = @(
        "whisper_distil_large_v3.json",
        "whisper_large_v3.json",
        "whisper_large_v3_turbo.json",
        "whisperx_zh_alignment.json"
    )
    Sfx = @(
        "stable_audio_3_small_sfx.json"
    )
}

$manifestNames = $profiles[$Profile]
$manifestPaths = foreach ($name in $manifestNames) {
    $path = Join-Path $ManifestRoot $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Model manifest is missing: $path"
    }
    $path
}

Write-Host "Model profile: $Profile ($($manifestPaths.Count) manifests)"
Write-Host "Model files are downloaded only from manifest-pinned sources and verified by size and SHA256."
Write-Host "Model licenses are independent from the platform source license."
Write-Host "IndexTTS2 currently includes MaskGCT material marked CC-BY-NC-4.0."

if (-not $DryRun -and -not $AcceptThirdPartyLicenses) {
    throw (
        "Review LICENSES, THIRD_PARTY_NOTICES.md, and every manifest's source/license metadata, " +
        "then rerun with -AcceptThirdPartyLicenses. This switch records acknowledgement only; " +
        "it does not grant commercial or redistribution rights."
    )
}

Write-Host ""
Write-Host "Preflight: validating all approved download sources..."
foreach ($manifest in $manifestPaths) {
    $manifestData = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
    if ($manifestData.sources.model.provider -eq "huggingface") {
        & $Python $Provisioner `
            --manifest $manifest `
            --allow-huggingface-gated-download `
            --dry-run
    }
    else {
        & $Python $Provisioner `
            --manifest $manifest `
            --allow-modelscope-download `
            --dry-run
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Model preflight failed: $(Split-Path -Leaf $manifest)"
    }
}

if ($DryRun) {
    Write-Host ""
    Write-Host "Dry run completed. No model files were changed."
    exit 0
}

Write-Host ""
Write-Host "Provisioning models. Interrupted downloads can be retried safely."
foreach ($manifest in $manifestPaths) {
    Write-Host ""
    Write-Host "==> $(Split-Path -Leaf $manifest)"
    $manifestData = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
    if ($manifestData.sources.model.provider -eq "huggingface") {
        $confirmationHash = $manifestData.license_policy.confirmation_sha256
        & $Python $Provisioner `
            --manifest $manifest `
            --allow-huggingface-gated-download `
            --accept-license-hash $confirmationHash `
            --license-tier $StableAudioLicenseTier
    }
    else {
        & $Python $Provisioner `
            --manifest $manifest `
            --allow-modelscope-download
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Model provisioning failed: $(Split-Path -Leaf $manifest)"
    }
}

Write-Host ""
Write-Host "Model provisioning completed for profile: $Profile"
if ($Profile -ne "Full") {
    Write-Warning "start_platform.bat performs the Full-profile startup check. Partial profiles are intended for backend development and manual startup."
}
