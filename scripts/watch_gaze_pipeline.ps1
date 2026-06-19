param(
    [string]$OutputRoot = "runs\gaze_watch",
    [string]$AuditRoot = "$HOME\Documents\DreamAirTracking\calibration_data",
    [int]$Epochs = 60,
    [int]$IntervalSeconds = 20,
    [int]$MaxWaitSeconds = 0,
    [double]$SettleSeconds = 5.0,
    [ValidateSet("none", "fair", "good")]
    [string]$ImageSessionQuality = "none",
    [ValidateSet("fair", "good", "none")]
    [string]$RequireMetricsQuality = "fair",
    [switch]$RunImmediately,
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ArgsList = @(
    (Join-Path $Root "scripts\ml\watch_gaze_pipeline.py"),
    "--output-root", (Join-Path $Root $OutputRoot),
    "--audit-root", $AuditRoot,
    "--epochs", "$Epochs",
    "--interval-seconds", "$IntervalSeconds",
    "--max-wait-seconds", "$MaxWaitSeconds",
    "--settle-seconds", "$SettleSeconds",
    "--image-session-quality", $ImageSessionQuality,
    "--require-metrics-quality", $RequireMetricsQuality
)

if ($RunImmediately) {
    $ArgsList += "--run-immediately"
}

if ($Cpu) {
    $ArgsList += "--cpu"
}

python @ArgsList
exit $LASTEXITCODE
