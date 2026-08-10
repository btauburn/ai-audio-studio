# Copyright 2026 TadashiJLU.

[CmdletBinding()]
param(
    [ValidateRange(1, 99)]
    [int]$MaximumNonLfsSizeMiB = 50
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $ProjectRoot
try {
    $trackedOrPending = @(
        git -c core.quotepath=false ls-files --cached --others --exclude-standard
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to enumerate repository files with git ls-files."
    }
    $files = @($trackedOrPending | Where-Object { $_ } | Sort-Object -Unique)
    if ($files.Count -eq 0) {
        throw "Repository file guard found no tracked or pending files."
    }

    $forbiddenExactPaths = [Collections.Generic.HashSet[string]]::new(
        [string[]]@(
            ".env",
            "dubbing_service/config.json",
            "dubbing_service/llm_providers.json",
            "dubbing_service/voices.json"
        ),
        [StringComparer]::OrdinalIgnoreCase
    )
    $forbiddenPrefixes = @(
        ".venv",
        "checkpoints/",
        "dubbing_projects/",
        "dubbing_service/logs/",
        "dubbing_service/voices/",
        "models/.download_cache/",
        "models/.modelscope_cache/",
        "models/.quarantine/",
        "models/.runtime_cache/",
        "models/cosyvoice3/",
        "models/index_tts/",
        "models/qwen3_tts/",
        "outputs/",
        "service/mcp_outputs/",
        "service/mcp_prompts/",
        "service/outputs/",
        "service/prompts/",
        "service/temp/",
        "temp/"
    )
    $secretPatterns = @(
        '(?i)-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        '(?i)\bAKIA[0-9A-Z]{16}\b',
        '(?i)\b(?:sk|rk)-[A-Za-z0-9_-]{20,}\b',
        '(?i)\bgh[pousr]_[A-Za-z0-9]{20,}\b',
        '(?i)\bxox[baprs]-[A-Za-z0-9-]{20,}\b'
    )
    $textExtensions = [Collections.Generic.HashSet[string]]::new(
        [string[]]@(
            ".bat", ".cfg", ".cjs", ".cmd", ".conf", ".css", ".html",
            ".ini", ".ipynb", ".js", ".json", ".jsonl", ".lock", ".md",
            ".mjs", ".ps1", ".py", ".pyi", ".rst", ".sh", ".toml",
            ".txt", ".xml", ".yaml", ".yml"
        ),
        [StringComparer]::OrdinalIgnoreCase
    )
    $textNames = [Collections.Generic.HashSet[string]]::new(
        [string[]]@("COPYING", "DISCLAIMER", "Dockerfile", "LICENSE", "NOTICE"),
        [StringComparer]::OrdinalIgnoreCase
    )

    $violations = [Collections.Generic.List[string]]::new()
    $maximumBytes = [int64]$MaximumNonLfsSizeMiB * 1MB
    foreach ($relativePath in $files) {
        $normalized = $relativePath.Replace('\', '/').TrimStart('/')
        $forbidden = $forbiddenExactPaths.Contains($normalized)
        foreach ($prefix in $forbiddenPrefixes) {
            if ($normalized.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
                $forbidden = $true
                break
            }
        }
        if (
            $normalized.StartsWith("models/", [StringComparison]::OrdinalIgnoreCase) -and
            -not $normalized.Equals("models/README.md", [StringComparison]::OrdinalIgnoreCase) -and
            -not $normalized.StartsWith("models/manifests/", [StringComparison]::OrdinalIgnoreCase)
        ) {
            $forbidden = $true
        }
        if ($forbidden) {
            $violations.Add("forbidden project/model data path: $normalized")
            continue
        }

        $fullPath = Join-Path $ProjectRoot $relativePath
        if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
            continue
        }
        $file = Get-Item -LiteralPath $fullPath -Force
        if ($file.Length -gt $maximumBytes) {
            $filter = git check-attr filter -- $relativePath
            if ($LASTEXITCODE -ne 0 -or $filter -notmatch ': filter: lfs$') {
                $violations.Add(
                    "non-LFS file exceeds ${MaximumNonLfsSizeMiB} MiB: $normalized ($([Math]::Round($file.Length / 1MB, 2)) MiB)"
                )
            }
        }

        if (-not ($textExtensions.Contains($file.Extension) -or $textNames.Contains($file.Name))) {
            continue
        }
        $content = [IO.File]::ReadAllText($file.FullName)
        foreach ($pattern in $secretPatterns) {
            if ([Text.RegularExpressions.Regex]::IsMatch($content, $pattern)) {
                $violations.Add("potential credential in repository text: $normalized")
                break
            }
        }
    }

    if ($violations.Count -gt 0) {
        $details = $violations | Sort-Object -Unique | ForEach-Object { " - $_" }
        throw "Repository source guard failed:`n$($details -join "`n")"
    }

    Write-Host "Repository source guard passed: $($files.Count) tracked or pending files inspected."
}
finally {
    Pop-Location
}
