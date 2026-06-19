param(
    [switch]$Live,
    [int]$DurationSeconds = 30,
    [string]$NormalizationMode = "pupil_center",
    [string]$OutputDir = "runs\live_acceptance_check",
    [string]$Onnx = "runs\gaze_mobilenetv3_small_live_round1_round2\gaze_baseline.onnx",
    [int]$MonitorUdpPort = 9401
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Body
    )

    Write-Host ""
    Write-Host "== $Name ==" -ForegroundColor Cyan
    & $Body
}

Invoke-Step "Python syntax" {
    python -m py_compile `
        scripts\ml\eye_normalization.py `
        scripts\ml\wear_templates.py `
        scripts\ml\analyze_live_validation.py `
        scripts\ml\debug_eye_normalization.py `
        scripts\ml\predict_live.py
}

Invoke-Step "Runtime help" {
    python scripts\ml\predict_live.py --help | Out-Null
    python scripts\ml\wear_templates.py --help | Out-Null
    python scripts\ml\analyze_live_validation.py --help | Out-Null
    python scripts\ml\debug_eye_normalization.py --help | Out-Null
}

Invoke-Step "Core tests" {
    dotnet test .\tests\DreamAirTracking.Tests\DreamAirTracking.Tests.csproj
}

Invoke-Step "App release build" {
    dotnet build .\src\DreamAirTracking.App\DreamAirTracking.App.csproj -c Release
}

if ($Live) {
    Invoke-Step "Live dry-run" {
        New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
        python scripts\ml\predict_live.py `
            --onnx $Onnx `
            --duration-seconds $DurationSeconds `
            --monitor-udp-port $MonitorUdpPort `
            --normalization-mode $NormalizationMode `
            --output-dir $OutputDir `
            --print-every 30 `
            --snapshot-every 0
    }

    Invoke-Step "Live CSV analysis" {
        $LatestCsv = Get-ChildItem -Path $OutputDir -Filter "predict_live_*.csv" |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1
        if ($null -eq $LatestCsv) {
            throw "No predict_live CSV found in $OutputDir"
        }

        $ReportPath = Join-Path $OutputDir "validation_report.json"
        python scripts\ml\analyze_live_validation.py `
            --csv $LatestCsv.FullName `
            --templates runs\wear_templates\wear_templates.json `
            --output $ReportPath
        Write-Host "Validation report: $((Resolve-Path $ReportPath).Path)"
    }
}
else {
    Write-Host ""
    Write-Host "Static acceptance checks passed. Run with -Live after BrokenEye is streaming to collect live metrics." -ForegroundColor Green
}
