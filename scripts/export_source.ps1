[CmdletBinding()]
param(
    [string]$DestinationRoot = "",
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ProjectRootFull = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\', '/')

# This is an allow-list, not a mirror with exclusions.  New top-level folders
# are deliberately absent until they are reviewed and added here.
$AllowedRootFiles = @(
    ".gitattributes",
    ".gitignore",
    ".python-version",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "DISCLAIMER",
    "LICENSE",
    "MANIFEST.in",
    "NOTICE.md",
    "README.md",
    "README_EN.md",
    "SECURITY.md",
    "SUPPORT.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "start_platform.bat",
    "uv.lock",
    "webui.py",
    "dubbing_service/__init__.py",
    "dubbing_service/config.example.json",
    "dubbing_service/llm_models.json",
    "models/README.md"
)

$AllowedTrees = @(
    ".github",
    "archive",
    "cosyvoice",
    "docs",
    "dubbing_service/app",
    "dubbing_service/docs",
    "dubbing_service/webui",
    "environments",
    "examples",
    "indextts",
    "LICENSES",
    "models/manifests",
    "scripts",
    "service",
    "tests",
    "tools"
)

$AllowedExtensions = [Collections.Generic.HashSet[string]]::new(
    [string[]]@(
        ".bat", ".c", ".cc", ".cfg", ".cmake", ".cmd", ".conf", ".cpp",
        ".css", ".cu", ".cuh", ".h", ".hpp", ".html", ".ini", ".j2",
        ".jinja", ".js", ".json", ".jsonl", ".lock", ".md", ".mjs",
        ".proto", ".ps1", ".py", ".pyi", ".rst", ".sh", ".svg", ".toml", ".txt",
        ".xml", ".yaml", ".yml"
    ),
    [StringComparer]::OrdinalIgnoreCase
)

$AllowedExtensionlessNames = [Collections.Generic.HashSet[string]]::new(
    [string[]]@("COPYING", "Dockerfile", "LICENSE", "NOTICE"),
    [StringComparer]::OrdinalIgnoreCase
)

$ExcludedDirectoryNames = [Collections.Generic.HashSet[string]]::new(
    [string[]]@(
        ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv",
        ".venv_qwen3tts", "__pycache__", "build", "checkpoints", "conda_env",
        "dist", "dubbing_projects", "logs", "mcp_outputs", "mcp_prompts",
        "node_modules", "outputs", "prompts", "temp", "venv", "voices"
    ),
    [StringComparer]::OrdinalIgnoreCase
)

# These mutable files may contain API credentials or private voice/model choices.
$ExcludedFileNames = [Collections.Generic.HashSet[string]]::new(
    [string[]]@(".env"),
    [StringComparer]::OrdinalIgnoreCase
)

$ExcludedRelativeFiles = [Collections.Generic.HashSet[string]]::new(
    [string[]]@(
        "dubbing_service/config.json",
        "dubbing_service/llm_providers.json",
        "dubbing_service/voices.json"
    ),
    [StringComparer]::OrdinalIgnoreCase
)

$ForbiddenRelativePrefixes = @(
    ".git/",
    ".venv",
    "checkpoints/",
    "dubbing_projects/",
    "dubbing_service/logs/",
    "dubbing_service/voices/",
    "dubbing_service/webui/node_modules/",
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

$ForbiddenBinaryExtensions = [Collections.Generic.HashSet[string]]::new(
    [string[]]@(
        ".7z", ".bin", ".ckpt", ".flac", ".fst", ".gguf", ".mp3", ".mp4",
        ".npy", ".npz", ".onnx", ".pth", ".pt", ".safetensors", ".tar",
        ".wav", ".webm", ".weights", ".zip"
    ),
    [StringComparer]::OrdinalIgnoreCase
)

$AllowedPublicBinaryFiles = [Collections.Generic.HashSet[string]]::new(
    [string[]]@(
        "docs/assets/screenshots/publish-workflow.png",
        "docs/assets/screenshots/sound-design.png",
        "docs/assets/screenshots/studio-overview.png",
        "dubbing_service/webui/brand/logo-mark-256.png",
        "dubbing_service/webui/brand/social-preview.png"
    ),
    [StringComparer]::OrdinalIgnoreCase
)

$SecretPatterns = @(
    '(?i)-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    '(?i)\bAKIA[0-9A-Z]{16}\b',
    '(?i)\b(?:sk|rk)-[A-Za-z0-9_-]{20,}\b',
    '(?i)\bgh[pousr]_[A-Za-z0-9]{20,}\b',
    '(?i)\bxox[baprs]-[A-Za-z0-9-]{20,}\b'
)

function Convert-ToRelativePath {
    param([Parameter(Mandatory = $true)][string]$FullName)
    $candidate = [IO.Path]::GetFullPath($FullName)
    $rootPrefix = $ProjectRootFull + [IO.Path]::DirectorySeparatorChar
    if (-not $candidate.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Source path escapes the project root: $candidate"
    }
    return $candidate.Substring($rootPrefix.Length).Replace('\', '/')
}

function Test-ForbiddenRelativePath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $normalized = $RelativePath.Replace('\', '/').TrimStart('/')
    foreach ($prefix in $ForbiddenRelativePrefixes) {
        if ($normalized.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }

    $parts = $normalized.Split('/', [StringSplitOptions]::RemoveEmptyEntries)
    foreach ($part in $parts) {
        if ($ExcludedDirectoryNames.Contains($part)) {
            return $true
        }
    }
    return $false
}

function Test-AllowedSourceFile {
    param([Parameter(Mandatory = $true)][IO.FileInfo]$File)

    if (($File.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        return $false
    }
    $relative = Convert-ToRelativePath -FullName $File.FullName
    if (Test-ForbiddenRelativePath -RelativePath $relative) {
        return $false
    }
    if ($ExcludedRelativeFiles.Contains($relative)) {
        return $false
    }
    if ($ExcludedFileNames.Contains($File.Name)) {
        return $false
    }
    if ($AllowedPublicBinaryFiles.Contains($relative)) {
        return $true
    }
    if ($ForbiddenBinaryExtensions.Contains($File.Extension)) {
        return $false
    }
    if ($AllowedExtensions.Contains($File.Extension)) {
        return $true
    }
    return $AllowedExtensionlessNames.Contains($File.Name)
}

function Assert-NoEmbeddedSecret {
    param(
        [Parameter(Mandatory = $true)][IO.FileInfo]$File,
        [Parameter(Mandatory = $true)][string]$RelativePath
    )

    $content = [IO.File]::ReadAllText($File.FullName)
    foreach ($pattern in $SecretPatterns) {
        if ([Text.RegularExpressions.Regex]::IsMatch($content, $pattern)) {
            throw "Potential credential found in allow-listed source file: $RelativePath"
        }
    }
}

$selectedByPath = [Collections.Generic.SortedDictionary[string, IO.FileInfo]]::new(
    [StringComparer]::OrdinalIgnoreCase
)

foreach ($relative in $AllowedRootFiles) {
    $candidate = Join-Path $ProjectRoot $relative
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        continue
    }
    $file = Get-Item -LiteralPath $candidate -Force
    if (Test-AllowedSourceFile -File $file) {
        $normalized = Convert-ToRelativePath -FullName $file.FullName
        $selectedByPath[$normalized] = $file
    }
}

foreach ($tree in $AllowedTrees) {
    $sourceRoot = Join-Path $ProjectRoot $tree
    if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
        continue
    }
    foreach ($file in Get-ChildItem -LiteralPath $sourceRoot -File -Recurse -Force) {
        if (-not (Test-AllowedSourceFile -File $file)) {
            continue
        }
        $normalized = Convert-ToRelativePath -FullName $file.FullName
        $selectedByPath[$normalized] = $file
    }
}

if ($selectedByPath.Count -eq 0) {
    throw "Source allow-list selected no files."
}

foreach ($entry in $selectedByPath.GetEnumerator()) {
    $relative = $entry.Key
    if (Test-ForbiddenRelativePath -RelativePath $relative) {
        throw "Internal error: forbidden path passed the allow-list: $relative"
    }
    if ($ForbiddenBinaryExtensions.Contains($entry.Value.Extension)) {
        throw "Internal error: model/media binary passed the allow-list: $relative"
    }
    if (-not $AllowedPublicBinaryFiles.Contains($relative)) {
        Assert-NoEmbeddedSecret -File $entry.Value -RelativePath $relative
    }
}

if ($ValidateOnly) {
    Write-Host "Source export validation passed: $($selectedByPath.Count) allow-listed source and public brand files."
    return
}

$ProjectParent = Split-Path -Parent $ProjectRootFull
if (-not $DestinationRoot) {
    $DestinationRoot = Join-Path $ProjectParent "index-tts-source-exports"
}

$destinationRootFull = [IO.Path]::GetFullPath($DestinationRoot).TrimEnd('\', '/')
if (
    $destinationRootFull -eq $ProjectRootFull -or
    $destinationRootFull.StartsWith($ProjectRootFull + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)
) {
    throw "Source export destination must be outside the project directory."
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$destination = Join-Path $destinationRootFull $stamp
if (Test-Path -LiteralPath $destination) {
    throw "Source export destination already exists: $destination"
}
New-Item -ItemType Directory -Path $destination -Force | Out-Null

$manifestFiles = [Collections.Generic.List[object]]::new()
foreach ($entry in $selectedByPath.GetEnumerator()) {
    $relative = $entry.Key
    $source = $entry.Value
    $target = Join-Path $destination ($relative.Replace('/', [IO.Path]::DirectorySeparatorChar))
    $targetParent = Split-Path -Parent $target
    New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
    Copy-Item -LiteralPath $source.FullName -Destination $target
    $manifestFiles.Add([ordered]@{
        path = $relative
        size = $source.Length
        sha256 = (Get-FileHash -LiteralPath $source.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    })
}

$manifest = [ordered]@{
    schema_version = 1
    created_at = (Get-Date).ToUniversalTime().ToString("o")
    source_root_name = Split-Path -Leaf $ProjectRootFull
    file_count = $manifestFiles.Count
    files = $manifestFiles
    excluded_by_design = @(
        "model weights and model caches",
        "virtual environments and dependency caches",
        "API keys, .env files, config.json and provider configuration",
        "dubbing projects, voices, prompts, generated audio and logs",
        "all non-allow-listed binary/media files"
    )
    notice = "This source-only export does not grant additional license rights. Review repository and component licenses before redistribution."
}
$manifestPath = Join-Path $destination "source-export-manifest.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8

Write-Host "Source-only export completed: $destination"
Write-Host "Files copied: $($manifestFiles.Count)"
