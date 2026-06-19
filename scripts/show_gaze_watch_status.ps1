param(
    [string]$OutputRoot = "runs\gaze_watch"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$StatusPath = Join-Path (Join-Path $Root $OutputRoot) "watch_status.json"

if (-not (Test-Path $StatusPath)) {
    Write-Host "No watch status found: $StatusPath"
    Write-Host "Start watcher with: .\scripts\watch_gaze_pipeline.ps1 -RunImmediately"
    exit 2
}

$Status = Get-Content $StatusPath -Raw | ConvertFrom-Json
Write-Host "State: $($Status.state)"
Write-Host "Return code: $($Status.return_code)"
Write-Host "Attempts: $($Status.attempt_count)"
Write-Host "Updated: $($Status.updated_at)"
if ($Status.reason) {
    Write-Host "Reason: $($Status.reason)"
}
if ($Status.attempt_dir) {
    Write-Host "Attempt: $($Status.attempt_dir)"
}
if ($Status.recording_request) {
    Write-Host "Recording request: $($Status.recording_request)"
}
if ($Status.latest_audit_report) {
    Write-Host "Latest audit: $($Status.latest_audit_report)"
}

function Show-MatchingLines {
    param(
        [string]$Path,
        [string[]]$Patterns
    )

    if (-not $Path -or -not (Test-Path $Path)) {
        return
    }

    $Lines = Get-Content $Path
    foreach ($Pattern in $Patterns) {
        $Match = $Lines | Where-Object { $_ -like $Pattern } | Select-Object -First 1
        if ($Match) {
            Write-Host $Match
        }
    }
}

function Show-SectionTableRows {
    param(
        [string]$Path,
        [string]$Section,
        [int]$MaxRows = 4
    )

    if (-not $Path -or -not (Test-Path $Path)) {
        return
    }

    $Lines = Get-Content $Path
    $InSection = $false
    $Rows = 0
    foreach ($Line in $Lines) {
        if ($Line -eq $Section) {
            $InSection = $true
            continue
        }
        if (-not $InSection) {
            continue
        }
        if ($Line -like "## *") {
            break
        }
        if ($Line -like "| * |" -and $Line -notlike "|---*") {
            Write-Host $Line
            $Rows++
            if ($Rows -ge ($MaxRows + 1)) {
                break
            }
        }
    }
}

if ($Status.recording_request) {
    Write-Host ""
    Write-Host "Recording request summary:"
    Show-MatchingLines $Status.recording_request @(
        "- Complete 9-point sessions with fair/good metrics:*",
        "- Required for train/validation:*",
        "- Weak stages to watch:*"
    )
    Show-SectionTableRows $Status.recording_request "### Weak Stage Fix Plan"
}

if ($Status.latest_audit_report) {
    Write-Host ""
    Write-Host "Audit summary:"
    Show-MatchingLines $Status.latest_audit_report @(
        "- Strict usable sessions:*",
        "- Weak stages:*",
        "Next recording plan:*"
    )
}

exit ([int]$Status.return_code)
