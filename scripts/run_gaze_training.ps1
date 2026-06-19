param(
    [string[]]$Pairs = @(
        "calibration_session_01\calibration_pairs.csv",
        "calibration_session_fix01\calibration_pairs.csv",
        "calibration_session_final\calibration_pairs.csv"
    ),
    [string[]]$ValSession = @("calibration_session_final"),
    [ValidateSet("session", "interleaved")]
    [string]$SplitMode = "session",
    [string]$OutDir = "runs\gaze_baseline",
    [int]$Epochs = 40,
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Manifest = Join-Path $OutDir "manifest.csv"
$Summary = Join-Path $OutDir "manifest.summary.json"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$PrepareArgs = @(
    (Join-Path $Root "scripts\ml\prepare_gaze_dataset.py"),
    "--output", $Manifest,
    "--summary", $Summary,
    "--split-mode", $SplitMode
)

foreach ($Pair in $Pairs) {
    $PrepareArgs += @("--pairs", (Join-Path $Root $Pair))
}

foreach ($Session in $ValSession) {
    if (-not [string]::IsNullOrWhiteSpace($Session)) {
        $PrepareArgs += @("--val-session", $Session)
    }
}

python @PrepareArgs

$TrainArgs = @(
    (Join-Path $Root "scripts\ml\train_gaze_baseline.py"),
    "--manifest", $Manifest,
    "--output-dir", $OutDir,
    "--epochs", "$Epochs"
)

if ($Cpu) {
    $TrainArgs += "--cpu"
}

python @TrainArgs
