param(
    [string]$Profile = "tracking_profile_final.json",
    [string]$HostName = "127.0.0.1",
    [int]$BrokenEyePort = 5555,
    [string]$UdpHost = "127.0.0.1",
    [int]$UdpPort = 9400,
    [string]$MonitorUdpHost = "127.0.0.1",
    [int]$MonitorUdpPort = 9401,
    [double]$MaxDeltaMs = 20,
    [int]$PrintEvery = 30
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$bridgeProject = Join-Path $repoRoot "src\DreamAirTracking.Bridge\DreamAirTracking.Bridge.csproj"
$profilePath = Join-Path $repoRoot $Profile

dotnet run --project $bridgeProject -- run `
    --profile $profilePath `
    --host $HostName `
    --port $BrokenEyePort `
    --udp-host $UdpHost `
    --udp-port $UdpPort `
    --monitor-udp-host $MonitorUdpHost `
    --monitor-udp-port $MonitorUdpPort `
    --max-delta-ms $MaxDeltaMs `
    --print-every $PrintEvery
