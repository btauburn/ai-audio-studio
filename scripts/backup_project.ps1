[CmdletBinding()]
param(
    [string]$DestinationRoot = "",
    [switch]$IncludeModels,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ProjectRootFull = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\', '/')
$ProjectRootPrefix = $ProjectRootFull + [IO.Path]::DirectorySeparatorChar

$ExcludedDirectoryNames = [Collections.Generic.HashSet[string]]::new(
    [string[]]@(
        ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv",
        ".venv_qwen3tts", "__pycache__", "build", "dist", "logs",
        "node_modules", "outputs", "temp", "venv",
        ".download_cache", ".modelscope_cache", ".quarantine", ".runtime_cache"
    ),
    [StringComparer]::OrdinalIgnoreCase
)

# A cold backup contains private projects and voices, but credentials must be
# restored from the operator's secret store instead of being copied into a
# broadly readable backup directory.
$SensitiveRelativeFiles = [Collections.Generic.HashSet[string]]::new(
    [string[]]@(
        ".env",
        "dubbing_service/config.json",
        "dubbing_service/llm_providers.json",
        "service/claude_desktop_config.json"
    ),
    [StringComparer]::OrdinalIgnoreCase
)

$SensitiveFileNames = [Collections.Generic.HashSet[string]]::new(
    [string[]]@(
        "credentials.json", "credential.json", "secrets.json", "secret.json",
        "tokens.json", "token.json", "id_rsa", "id_ed25519"
    ),
    [StringComparer]::OrdinalIgnoreCase
)

$SensitiveExtensions = [Collections.Generic.HashSet[string]]::new(
    [string[]]@(".key", ".pem", ".p12", ".pfx"),
    [StringComparer]::OrdinalIgnoreCase
)

$ExcludedFileExtensions = [Collections.Generic.HashSet[string]]::new(
    [string[]]@(".pyc", ".pyo", ".log"),
    [StringComparer]::OrdinalIgnoreCase
)

$CredentialScanExtensions = [Collections.Generic.HashSet[string]]::new(
    [string[]]@(
        "", ".bat", ".cfg", ".cmd", ".conf", ".css", ".html", ".ini",
        ".js", ".json", ".jsonl", ".md", ".mjs", ".ps1", ".py",
        ".rst", ".sh", ".toml", ".txt", ".xml", ".yaml", ".yml"
    ),
    [StringComparer]::OrdinalIgnoreCase
)
$CredentialScanExcludedPrefixes = @(
    "dubbing_projects/",
    "dubbing_service/voices/",
    "prompts/",
    "service/prompts/",
    "VioceTimbre/"
)
$CredentialPatterns = @(
    '(?i)-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    '(?i)\bAKIA[0-9A-Z]{16}\b',
    '(?i)\b(?:sk|rk)-[A-Za-z0-9_-]{20,}\b',
    '(?i)\bgh[pousr]_[A-Za-z0-9]{20,}\b',
    '(?i)\bxox[baprs]-[A-Za-z0-9-]{20,}\b'
)

$DefaultModelFilePrefixes = @(
    "models/manifests/",
    "models/recovery_blobs/"
)
$DefaultModelRootFiles = [Collections.Generic.HashSet[string]]::new(
    [string[]]@("models/README.md"),
    [StringComparer]::OrdinalIgnoreCase
)

$RequiredRecoveryInstructions = @(
    "models/README.md",
    "docs/operations.md"
)

function Convert-ToProjectRelativePath {
    param([Parameter(Mandatory = $true)][string]$FullName)

    $candidate = [IO.Path]::GetFullPath($FullName)
    if (-not $candidate.StartsWith($ProjectRootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Source path escapes the project root: $candidate"
    }
    return $candidate.Substring($ProjectRootPrefix.Length).Replace('\', '/')
}

function Normalize-ManifestRelativePath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    if ([IO.Path]::IsPathRooted($RelativePath)) {
        throw "Manifest recovery path must be relative: $RelativePath"
    }
    $normalized = $RelativePath.Replace('\', '/').TrimStart('/')
    $parts = $normalized.Split('/', [StringSplitOptions]::RemoveEmptyEntries)
    if ($parts.Count -eq 0) {
        throw "Manifest recovery path is empty."
    }
    foreach ($part in $parts) {
        if ($part -eq "." -or $part -eq "..") {
            throw "Manifest recovery path contains an unsafe component: $RelativePath"
        }
    }
    return [string]::Join('/', $parts)
}

function Test-IsDefaultAllowedModelPath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $normalized = $RelativePath.Replace('\', '/').TrimStart('/')
    if ($DefaultModelRootFiles.Contains($normalized)) {
        return $true
    }
    foreach ($prefix in $DefaultModelFilePrefixes) {
        if ($normalized.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

function Test-IsExcludedDirectory {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $normalized = $RelativePath.Replace('\', '/').Trim('/')
    $parts = $normalized.Split('/', [StringSplitOptions]::RemoveEmptyEntries)
    foreach ($part in $parts) {
        if ($ExcludedDirectoryNames.Contains($part)) {
            return $true
        }
    }

    if (-not $IncludeModels -and $normalized.StartsWith("models/", [StringComparison]::OrdinalIgnoreCase)) {
        if (
            $normalized.Equals("models/manifests", [StringComparison]::OrdinalIgnoreCase) -or
            $normalized.StartsWith("models/manifests/", [StringComparison]::OrdinalIgnoreCase) -or
            $normalized.Equals("models/recovery_blobs", [StringComparison]::OrdinalIgnoreCase) -or
            $normalized.StartsWith("models/recovery_blobs/", [StringComparison]::OrdinalIgnoreCase)
        ) {
            return $false
        }
        return $true
    }
    return $false
}

function Test-IsSensitiveFile {
    param(
        [Parameter(Mandatory = $true)][IO.FileInfo]$File,
        [Parameter(Mandatory = $true)][string]$RelativePath
    )

    if ($SensitiveRelativeFiles.Contains($RelativePath)) {
        return $true
    }
    if ($SensitiveFileNames.Contains($File.Name)) {
        return $true
    }
    if ($SensitiveExtensions.Contains($File.Extension)) {
        return $true
    }
    if ($File.Name.Equals(".env", [StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    if (
        $File.Name.StartsWith(".env.", [StringComparison]::OrdinalIgnoreCase) -and
        -not $File.Name.EndsWith(".example", [StringComparison]::OrdinalIgnoreCase)
    ) {
        return $true
    }
    return $false
}

function Test-IsExcludedFile {
    param(
        [Parameter(Mandatory = $true)][IO.FileInfo]$File,
        [Parameter(Mandatory = $true)][string]$RelativePath
    )

    if ($ExcludedFileExtensions.Contains($File.Extension)) {
        return $true
    }
    if (Test-IsSensitiveFile -File $File -RelativePath $RelativePath) {
        return $true
    }
    if (-not $IncludeModels -and $RelativePath.StartsWith("models/", [StringComparison]::OrdinalIgnoreCase)) {
        return -not (Test-IsDefaultAllowedModelPath -RelativePath $RelativePath)
    }
    return $false
}

function Test-ShouldScanForCredentials {
    param(
        [Parameter(Mandatory = $true)][IO.FileInfo]$File,
        [Parameter(Mandatory = $true)][string]$RelativePath
    )

    if (-not $CredentialScanExtensions.Contains($File.Extension)) {
        return $false
    }
    foreach ($prefix in $CredentialScanExcludedPrefixes) {
        if ($RelativePath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            return $false
        }
    }
    return $true
}

function Assert-NoEmbeddedCredential {
    param(
        [Parameter(Mandatory = $true)][IO.FileInfo]$File,
        [Parameter(Mandatory = $true)][string]$RelativePath
    )

    $content = [IO.File]::ReadAllText($File.FullName)
    foreach ($pattern in $CredentialPatterns) {
        if ([Text.RegularExpressions.Regex]::IsMatch($content, $pattern)) {
            throw "Potential credential found in selected backup source file: $RelativePath"
        }
    }
}

function Assert-NoReparsePoint {
    param(
        [Parameter(Mandatory = $true)][IO.FileSystemInfo]$Item,
        [Parameter(Mandatory = $true)][string]$Context
    )

    if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "$Context is a reparse point and will not be backed up: $($Item.FullName)"
    }
}

function Assert-NoReparsePathComponents {
    param([Parameter(Mandatory = $true)][string]$Path)

    $current = [IO.Path]::GetFullPath($Path)
    while ($current) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            Assert-NoReparsePoint -Item $item -Context "Destination path component"
        }
        $parent = Split-Path -Parent $current
        if (-not $parent -or $parent -eq $current) {
            break
        }
        $current = $parent
    }
}

function Get-Sha256Hex {
    param([Parameter(Mandatory = $true)][string]$LiteralPath)

    $stream = $null
    $sha256 = $null
    try {
        $stream = [IO.File]::Open(
            $LiteralPath,
            [IO.FileMode]::Open,
            [IO.FileAccess]::Read,
            [IO.FileShare]::Read
        )
        $sha256 = [Security.Cryptography.SHA256]::Create()
        $hashBytes = $sha256.ComputeHash($stream)
        return ([BitConverter]::ToString($hashBytes)).Replace('-', '').ToLowerInvariant()
    }
    finally {
        if ($null -ne $sha256) {
            $sha256.Dispose()
        }
        if ($null -ne $stream) {
            $stream.Dispose()
        }
    }
}

$projectRootItem = Get-Item -LiteralPath $ProjectRootFull -Force
Assert-NoReparsePoint -Item $projectRootItem -Context "Project root"

$SelectedByPath = [Collections.Generic.SortedDictionary[string, IO.FileInfo]]::new(
    [StringComparer]::OrdinalIgnoreCase
)
$ExcludedSensitivePaths = [Collections.Generic.List[string]]::new()

function Add-SafeDirectoryFiles {
    param([Parameter(Mandatory = $true)][string]$DirectoryPath)

    foreach ($item in Get-ChildItem -LiteralPath $DirectoryPath -Force) {
        $relative = Convert-ToProjectRelativePath -FullName $item.FullName
        if ($item.PSIsContainer) {
            if (Test-IsExcludedDirectory -RelativePath $relative) {
                continue
            }
            Assert-NoReparsePoint -Item $item -Context "Source directory"
            Add-SafeDirectoryFiles -DirectoryPath $item.FullName
            continue
        }

        if (Test-IsSensitiveFile -File $item -RelativePath $relative) {
            $ExcludedSensitivePaths.Add($relative)
            continue
        }
        if (Test-IsExcludedFile -File $item -RelativePath $relative) {
            continue
        }
        Assert-NoReparsePoint -Item $item -Context "Source file"
        $SelectedByPath[$relative] = $item
    }
}

Add-SafeDirectoryFiles -DirectoryPath $ProjectRootFull

if ($SelectedByPath.Count -eq 0) {
    throw "Backup selection is empty."
}

$secretPathsSelected = [Collections.Generic.List[string]]::new()
$ordinaryModelPathsSelected = [Collections.Generic.List[string]]::new()
foreach ($entry in $SelectedByPath.GetEnumerator()) {
    $relative = $entry.Key
    if (Test-IsSensitiveFile -File $entry.Value -RelativePath $relative) {
        $secretPathsSelected.Add($relative)
    }
    if (
        -not $IncludeModels -and
        $relative.StartsWith("models/", [StringComparison]::OrdinalIgnoreCase) -and
        -not (Test-IsDefaultAllowedModelPath -RelativePath $relative)
    ) {
        $ordinaryModelPathsSelected.Add($relative)
    }
}
if ($secretPathsSelected.Count -ne 0) {
    throw "Internal error: credential-bearing paths passed the backup selection."
}
if ($ordinaryModelPathsSelected.Count -ne 0) {
    throw "Internal error: ordinary model files passed the default backup selection."
}

$credentialScanFileCount = 0
foreach ($entry in $SelectedByPath.GetEnumerator()) {
    if (Test-ShouldScanForCredentials -File $entry.Value -RelativePath $entry.Key) {
        Assert-NoEmbeddedCredential -File $entry.Value -RelativePath $entry.Key
        $credentialScanFileCount += 1
    }
}

foreach ($instructionPath in $RequiredRecoveryInstructions) {
    if (-not $SelectedByPath.ContainsKey($instructionPath)) {
        throw "Required recovery instruction is missing from the backup selection: $instructionPath"
    }
}

$RecoveryReferences = [Collections.Generic.SortedDictionary[string, object]]::new(
    [StringComparer]::OrdinalIgnoreCase
)
$manifestPaths = [Collections.Generic.List[string]]::new()
$ordinaryManifestModelPathCount = 0
$ordinaryManifestModelPathsSelected = [Collections.Generic.List[string]]::new()
$manifestDirectory = Join-Path $ProjectRootFull "models\manifests"
foreach ($manifestFile in Get-ChildItem -LiteralPath $manifestDirectory -File -Filter "*.json" -Force) {
    Assert-NoReparsePoint -Item $manifestFile -Context "Model manifest"
    $manifestRelative = Convert-ToProjectRelativePath -FullName $manifestFile.FullName
    if (-not $SelectedByPath.ContainsKey($manifestRelative)) {
        throw "Model manifest is missing from the backup selection: $manifestRelative"
    }
    $manifestPaths.Add($manifestRelative)

    try {
        $manifest = Get-Content -LiteralPath $manifestFile.FullName -Raw | ConvertFrom-Json
    }
    catch {
        throw "Invalid model manifest JSON: $manifestRelative ($($_.Exception.Message))"
    }

    foreach ($modelFile in @($manifest.files)) {
        $runtimePath = Normalize-ManifestRelativePath -RelativePath ([string]$modelFile.path)
        $sourceProperty = $null
        if ($null -ne $manifest.sources -and $null -ne $modelFile.source) {
            $sourceProperty = $manifest.sources.PSObject.Properties[[string]$modelFile.source]
        }
        $hasModelScopeRecovery = (
            $null -ne $sourceProperty -and
            ([string]$sourceProperty.Value.provider).Equals("modelscope", [StringComparison]::OrdinalIgnoreCase)
        )
        if (
            $runtimePath.StartsWith("models/", [StringComparison]::OrdinalIgnoreCase) -and
            -not (Test-IsDefaultAllowedModelPath -RelativePath $runtimePath)
        ) {
            $ordinaryManifestModelPathCount += 1
            if (-not $IncludeModels -and $SelectedByPath.ContainsKey($runtimePath)) {
                $ordinaryManifestModelPathsSelected.Add($runtimePath)
            }
        }
        foreach ($recovery in @($modelFile.recovery)) {
            if ($null -eq $recovery -or -not ([string]$recovery.provider).Equals("project_cache", [StringComparison]::OrdinalIgnoreCase)) {
                continue
            }
            $recoveryPath = Normalize-ManifestRelativePath -RelativePath ([string]$recovery.path)
            if (-not $recoveryPath.StartsWith("models/recovery_blobs/", [StringComparison]::OrdinalIgnoreCase)) {
                throw "Project-cache recovery path is outside models/recovery_blobs: $recoveryPath"
            }
            if (-not $SelectedByPath.ContainsKey($recoveryPath)) {
                if ($hasModelScopeRecovery) {
                    continue
                }
                throw "Required non-downloadable recovery blob is missing from the backup selection: $recoveryPath"
            }

            $reference = [ordered]@{
                manifest = $manifestRelative
                size = [long]$modelFile.size
                sha256 = ([string]$modelFile.sha256).ToLowerInvariant()
            }
            if ($reference.size -lt 0 -or $reference.sha256 -notmatch '^[0-9a-f]{64}$') {
                throw "Recovery metadata is invalid for $recoveryPath in $manifestRelative"
            }
            if ($RecoveryReferences.ContainsKey($recoveryPath)) {
                $previous = $RecoveryReferences[$recoveryPath]
                if ($previous.size -ne $reference.size -or $previous.sha256 -ne $reference.sha256) {
                    throw "Conflicting recovery metadata for $recoveryPath"
                }
            }
            else {
                $RecoveryReferences[$recoveryPath] = $reference
            }
        }
    }
}

if (-not $IncludeModels -and $ordinaryManifestModelPathsSelected.Count -ne 0) {
    throw "Internal error: a manifest-declared ordinary model path passed the default backup selection."
}

$validatedRecoveryBytes = [long]0
foreach ($entry in $RecoveryReferences.GetEnumerator()) {
    $relative = $entry.Key
    $source = $SelectedByPath[$relative]
    $expected = $entry.Value
    if ($source.Length -ne $expected.size) {
        throw "Recovery blob size mismatch: $relative (expected $($expected.size), got $($source.Length))"
    }
    $actualHash = Get-Sha256Hex -LiteralPath $source.FullName
    if ($actualHash -ne $expected.sha256) {
        throw "Recovery blob SHA256 mismatch: $relative"
    }
    $validatedRecoveryBytes += $source.Length
}

foreach ($entry in $SelectedByPath.GetEnumerator()) {
    if (
        $entry.Key.StartsWith("models/recovery_blobs/", [StringComparison]::OrdinalIgnoreCase) -and
        -not $RecoveryReferences.ContainsKey($entry.Key)
    ) {
        throw "Unreferenced file exists in models/recovery_blobs: $($entry.Key)"
    }
}

$selectedBytes = [long]0
foreach ($entry in $SelectedByPath.GetEnumerator()) {
    $selectedBytes += $entry.Value.Length
}

$validationReport = [ordered]@{
    schema_version = 1
    mode = $(if ($IncludeModels) { "full-model-backup" } else { "recoverable-default" })
    includes_models = [bool]$IncludeModels
    selected_file_count = $SelectedByPath.Count
    selected_bytes = $selectedBytes
    manifest_paths = @($manifestPaths)
    recovery_blob_paths = @($RecoveryReferences.Keys)
    recovery_blob_count = $RecoveryReferences.Count
    validated_recovery_bytes = $validatedRecoveryBytes
    excluded_sensitive_paths = @($ExcludedSensitivePaths)
    secret_paths_selected = @($secretPathsSelected)
    ordinary_model_paths_selected = @($ordinaryModelPathsSelected)
    ordinary_manifest_model_path_count = $ordinaryManifestModelPathCount
    ordinary_manifest_model_paths_selected = @($ordinaryManifestModelPathsSelected)
    credential_scan_file_count = $credentialScanFileCount
    credentials_excluded = $true
}

if ($ValidateOnly) {
    Write-Host "Backup validation passed. No files were copied."
    Write-Host "Selected files: $($SelectedByPath.Count); selected bytes: $selectedBytes"
    Write-Host "Verified project-local recovery blobs: $($RecoveryReferences.Count); bytes: $validatedRecoveryBytes"
    Write-Output ("BACKUP_VALIDATION_JSON=" + ($validationReport | ConvertTo-Json -Depth 6 -Compress))
    return
}

$ProjectParent = Split-Path -Parent $ProjectRootFull
if (-not $DestinationRoot) {
    $DestinationRoot = Join-Path $ProjectParent "index-tts-backups"
}
$destinationRootFull = [IO.Path]::GetFullPath($DestinationRoot).TrimEnd('\', '/')
if (
    $destinationRootFull.Equals($ProjectRootFull, [StringComparison]::OrdinalIgnoreCase) -or
    $destinationRootFull.StartsWith($ProjectRootPrefix, [StringComparison]::OrdinalIgnoreCase)
) {
    throw "Backup destination must be outside the project directory."
}
Assert-NoReparsePathComponents -Path $destinationRootFull
if (Test-Path -LiteralPath $destinationRootFull -PathType Leaf) {
    throw "Backup destination root is a file: $destinationRootFull"
}
New-Item -ItemType Directory -Path $destinationRootFull -Force | Out-Null
Assert-NoReparsePathComponents -Path $destinationRootFull

$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$destination = Join-Path $destinationRootFull $stamp
$partialDestination = "$destination.partial"
if ((Test-Path -LiteralPath $destination) -or (Test-Path -LiteralPath $partialDestination)) {
    throw "Backup destination already exists: $destination"
}
New-Item -ItemType Directory -Path $partialDestination | Out-Null
Assert-NoReparsePathComponents -Path $partialDestination
$partialPrefix = [IO.Path]::GetFullPath($partialDestination).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar

$manifestFiles = [Collections.Generic.List[object]]::new()
try {
    foreach ($entry in $SelectedByPath.GetEnumerator()) {
        $relative = $entry.Key
        $source = $entry.Value
        Assert-NoReparsePoint -Item (Get-Item -LiteralPath $source.FullName -Force) -Context "Source file"

        $target = Join-Path $partialDestination ($relative.Replace('/', [IO.Path]::DirectorySeparatorChar))
        $targetFull = [IO.Path]::GetFullPath($target)
        if (-not $targetFull.StartsWith($partialPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Backup target escapes the destination: $relative"
        }
        $targetParent = Split-Path -Parent $targetFull
        New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
        Assert-NoReparsePathComponents -Path $targetParent

        $sourceHash = Get-Sha256Hex -LiteralPath $source.FullName
        Copy-Item -LiteralPath $source.FullName -Destination $targetFull
        $targetItem = Get-Item -LiteralPath $targetFull -Force
        Assert-NoReparsePoint -Item $targetItem -Context "Backup file"
        $targetHash = Get-Sha256Hex -LiteralPath $targetFull
        if ($targetItem.Length -ne $source.Length -or $targetHash -ne $sourceHash) {
            throw "Backup verification failed after copying: $relative"
        }

        $manifestFiles.Add([ordered]@{
            path = $relative
            size = $targetItem.Length
            sha256 = $targetHash
            kind = $(
                if ($relative.StartsWith("models/recovery_blobs/", [StringComparison]::OrdinalIgnoreCase)) { "recovery_blob" }
                elseif ($relative.StartsWith("models/manifests/", [StringComparison]::OrdinalIgnoreCase)) { "model_manifest" }
                else { "project_file" }
            )
        })
    }

    $environmentDirectory = Join-Path $partialDestination "environment-snapshots"
    New-Item -ItemType Directory -Path $environmentDirectory -Force | Out-Null
    $runtimePythons = [ordered]@{
        platform = Join-Path $ProjectRootFull ".venv\Scripts\python.exe"
        qwen3tts = Join-Path $ProjectRootFull "environments\qwen3tts\.venv\Scripts\python.exe"
        cosyvoice = Join-Path $ProjectRootFull "environments\cosyvoice\.venv\Scripts\python.exe"
    }
    $uvCommand = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -ne $uvCommand) {
        foreach ($entry in $runtimePythons.GetEnumerator()) {
            if (-not (Test-Path -LiteralPath $entry.Value -PathType Leaf)) {
                continue
            }
            $snapshotLines = @(& $uvCommand.Source pip list --python $entry.Value --format freeze 2>&1)
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to create environment snapshot for $($entry.Key)."
            }
            $snapshotPath = Join-Path $environmentDirectory "$($entry.Key).txt"
            ($snapshotLines -join [Environment]::NewLine) | Set-Content -LiteralPath $snapshotPath -Encoding UTF8
            $snapshotItem = Get-Item -LiteralPath $snapshotPath -Force
            $snapshotHash = Get-Sha256Hex -LiteralPath $snapshotPath
            $manifestFiles.Add([ordered]@{
                path = "environment-snapshots/$($entry.Key).txt"
                size = $snapshotItem.Length
                sha256 = $snapshotHash
                kind = "environment_snapshot"
            })
        }
    }

    $restoreNotesPath = Join-Path $partialDestination "BACKUP_RESTORE.md"
    $restoreNotes = @'
# Backup restore notes

- Credentials are intentionally excluded. Recreate local configuration from `dubbing_service/config.example.json` and inject API keys from the operator's secret store.
- Default backups omit ordinary model weights. Restore those files with the pinned manifests and ModelScope-only provisioning commands in `docs/operations.md`.
- Any present `models/recovery_blobs/` entries are included and hash-verified as optional local recovery cache. Downloadable model files can also be reconstructed from the pinned ModelScope sources in `models/manifests/*.json`.
- User projects, voices, and prompts may contain private data. Keep this backup encrypted and access-controlled.
- Verify `backup-manifest.json`, then run the project data audit and full platform doctor before using a restored copy.
'@
    $restoreNotes | Set-Content -LiteralPath $restoreNotesPath -Encoding UTF8
    $restoreNotesItem = Get-Item -LiteralPath $restoreNotesPath -Force
    $restoreNotesHash = Get-Sha256Hex -LiteralPath $restoreNotesPath
    $manifestFiles.Add([ordered]@{
        path = "BACKUP_RESTORE.md"
        size = $restoreNotesItem.Length
        sha256 = $restoreNotesHash
        kind = "restore_instructions"
    })

    $backupManifest = [ordered]@{
        schema_version = 2
        created_at = (Get-Date).ToUniversalTime().ToString("o")
        source_root_name = Split-Path -Leaf $ProjectRootFull
        hash_algorithm = "SHA256"
        includes_models = [bool]$IncludeModels
        contains_credentials = $false
        file_count = $manifestFiles.Count
        files = $manifestFiles
        recovery_blobs = @($RecoveryReferences.Keys)
        excluded_by_design = @(
            "API keys, .env files, private keys, local provider/config files",
            "virtual environments, dependency caches, logs, outputs and temporary files",
            $(if ($IncludeModels) { "model download/runtime caches and quarantine" } else { "ordinary model weights; pinned manifests and non-downloadable recovery blobs are retained" })
        )
    }
    $manifestPath = Join-Path $partialDestination "backup-manifest.json"
    $backupManifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    $manifestHash = Get-Sha256Hex -LiteralPath $manifestPath

    $summary = [ordered]@{
        schema_version = 2
        created_at = (Get-Date).ToUniversalTime().ToString("o")
        destination_name = Split-Path -Leaf $destination
        includes_models = [bool]$IncludeModels
        contains_credentials = $false
        recovery_blob_count = $RecoveryReferences.Count
        backup_manifest_sha256 = $manifestHash
        note = "This backup excludes API credentials but may contain private voices, prompts, and project data. Keep it encrypted and access-controlled."
    }
    $summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $partialDestination "backup.json") -Encoding UTF8

    Move-Item -LiteralPath $partialDestination -Destination $destination
}
catch {
    Write-Warning "Backup did not complete. An incomplete directory may remain at: $partialDestination"
    throw
}

Write-Host "Backup completed and SHA256-verified: $destination"
Write-Host "Files recorded in backup-manifest.json: $($manifestFiles.Count)"
