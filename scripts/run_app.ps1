param(
    [string]$Configuration = "Debug"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$appProject = Join-Path $repoRoot "src\DreamAirTracking.App\DreamAirTracking.App.csproj"
$targetFramework = "net9.0-windows10.0.26100.0"
$runtimeIdentifier = "win-x64"

dotnet build $appProject --configuration $Configuration --runtime $runtimeIdentifier

$appExe = Join-Path $repoRoot "src\DreamAirTracking.App\bin\$Configuration\$targetFramework\$runtimeIdentifier\DreamAirTracking.App.exe"
if (-not (Test-Path $appExe)) {
    throw "DreamAirTracking.App.exe not found: $appExe"
}

Start-Process -FilePath $appExe -WorkingDirectory (Split-Path $appExe)
