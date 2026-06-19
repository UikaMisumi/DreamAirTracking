param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [string]$VrcftCustomLibs = "$env:APPDATA\VRCFaceTracking\CustomLibs",
    [string]$ModuleDirName = "b24a50f2-36bd-4a56-88f0-daa2a3727b5d"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$projectPath = Join-Path $repoRoot "src\DreamAirTracking.VrcftModule\DreamAirTracking.VrcftModule.csproj"
$outputDir = Join-Path $repoRoot "src\DreamAirTracking.VrcftModule\bin\$Configuration\net7.0"
$moduleDir = Join-Path $VrcftCustomLibs $ModuleDirName
$moduleDll = Join-Path $outputDir "DreamAirTracking.VrcftModule.dll"
$modulePdb = Join-Path $outputDir "DreamAirTracking.VrcftModule.pdb"

dotnet build $projectPath -c $Configuration

if (!(Test-Path $moduleDll)) {
    throw "Module DLL was not produced: $moduleDll"
}

New-Item -ItemType Directory -Force -Path $moduleDir | Out-Null
Copy-Item -Force -Path $moduleDll -Destination (Join-Path $moduleDir "DreamAirTracking.VrcftModule.dll")
if (Test-Path $modulePdb) {
    Copy-Item -Force -Path $modulePdb -Destination (Join-Path $moduleDir "DreamAirTracking.VrcftModule.pdb")
}

$moduleJson = [ordered]@{
    InstallationState = 0
    ModuleId = "b24a50f2-36bd-4a56-88f0-daa2a3727b5d"
    LastUpdated = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Version = "0.1.0-local"
    Downloads = 0
    Ratings = 0
    Rating = 0
    AuthorName = "DreamAirTracking"
    ModuleName = "DreamAirTracking"
    ModuleDescription = "Local BrokenEye raw-image eye tracking bridge for VRCFaceTracking."
    UsageInstructions = "Start BrokenEye, then start DreamAirTracking Eye Tracking from the DreamAirTracking app. This module only provides eye gaze, openness, and pupil diameter over UDP 127.0.0.1:9400; use SRanipalTrackingModule for face/expression capture. If App calibration is centered but VRChat eyes are biased, adjust DreamAirTracking VRCFT output trim."
    DownloadUrl = "local"
    ModulePageUrl = "file:///$($moduleDir -replace '\\','/')"
    DllFileName = "DreamAirTracking.VrcftModule.dll"
    FileHash = $null
} | ConvertTo-Json -Depth 4

Set-Content -Encoding UTF8 -Path (Join-Path $moduleDir "module.json") -Value $moduleJson

Write-Host "Installed DreamAirTracking VRCFT module to:"
Write-Host "  $moduleDir"
Write-Host "Restart VRCFaceTracking after installing."
