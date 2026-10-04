$ErrorActionPreference = 'Stop'
$studioRoot = Split-Path -Parent $PSScriptRoot
$pythonWindowless = Join-Path $studioRoot '.venv\Scripts\pythonw.exe'
$launcher = Join-Path $PSScriptRoot 'voice_library_launcher.py'
if (!(Test-Path -LiteralPath $pythonWindowless) -or !(Test-Path -LiteralPath $launcher)) {
    throw 'Voice Library runtime or launcher is missing.'
}
$shortcutShell = New-Object -ComObject WScript.Shell
$targets = @(
    @{Path=(Join-Path ([Environment]::GetFolderPath('Desktop')) 'Voice Library.lnk'); Arguments='"' + $launcher + '"'},
    @{Path=(Join-Path ([Environment]::GetFolderPath('Startup')) 'Voice Library Services.lnk'); Arguments='"' + $launcher + '" --no-browser'}
)
foreach ($target in $targets) {
    if (Test-Path -LiteralPath $target.Path) {
        $existing = $shortcutShell.CreateShortcut($target.Path)
        if ($existing.TargetPath -ne $pythonWindowless -or $existing.Arguments -ne $target.Arguments) {
            throw "A different shortcut already exists at $($target.Path); it was preserved."
        }
    }
    $shortcut = $shortcutShell.CreateShortcut($target.Path)
    $shortcut.TargetPath = $pythonWindowless
    $shortcut.Arguments = $target.Arguments
    $shortcut.WorkingDirectory = $studioRoot
    $shortcut.Description = 'Open Voice Library, preview character voices, and create book MP3s.'
    $shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,116"
    $shortcut.WindowStyle = 7
    $shortcut.Save()
    Write-Output $target.Path
}
