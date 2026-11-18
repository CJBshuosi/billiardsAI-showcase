[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$EvidenceRoot,
    [Parameter(Mandatory = $true)] [string]$OutputRoot,
    [ValidateRange(1, 720)] [int]$MaximumCalibrationAgeHours = 24
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Read-EvidenceJson {
    param([string]$Root, [string]$Name)
    $Path = Join-Path $Root $Name
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required acceptance evidence is missing: $Name"
    }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Add-Failure {
    param([System.Collections.Generic.List[string]]$Failures, [string]$Code)
    if (-not $Failures.Contains($Code)) { $Failures.Add($Code) }
}

$ResolvedEvidence = [IO.Path]::GetFullPath($EvidenceRoot)
$ResolvedOutput = [IO.Path]::GetFullPath($OutputRoot)
if (-not (Test-Path -LiteralPath $ResolvedEvidence -PathType Container)) {
    throw "EvidenceRoot is missing: $ResolvedEvidence"
}
if (Test-Path -LiteralPath $ResolvedOutput) {
    throw "OutputRoot already exists; acceptance bundles are immutable: $ResolvedOutput"
}

$Probe = Read-EvidenceJson $ResolvedEvidence "probe.json"
$Cadence = Read-EvidenceJson $ResolvedEvidence "cadence.json"
$Calibration = Read-EvidenceJson $ResolvedEvidence "calibration.json"
$Shots = Read-EvidenceJson $ResolvedEvidence "shots.json"
$Disconnect = Read-EvidenceJson $ResolvedEvidence "disconnect.json"
$Sessions = Read-EvidenceJson $ResolvedEvidence "sessions.json"
$Failures = [System.Collections.Generic.List[string]]::new()

$ProbeReady = ([string]$Probe.status).ToLowerInvariant() -eq "ready"
$ProbeExact = $Probe.requested_width -eq 1920 -and $Probe.requested_height -eq 1080 -and [Math]::Abs([double]$Probe.requested_fps - 60.0) -le 0.5 -and $Probe.observed_width -eq 1920 -and $Probe.observed_height -eq 1080 -and [Math]::Abs([double]$Probe.observed_fps - 60.0) -le 0.5
$CadenceReady = ([string]$Cadence.status).ToLowerInvariant() -eq "ready"
$CadenceExact = $Cadence.observed_width -eq 1920 -and $Cadence.observed_height -eq 1080 -and [Math]::Abs([double]$Cadence.observed_fps - 60.0) -le 0.5
if (-not ($ProbeReady -and $ProbeExact -and $CadenceReady -and $CadenceExact)) { Add-Failure $Failures "CAPTURE_MODE_NOT_1080P60" }
$ProbeMeasured = $null -ne $Probe.PSObject.Properties["measured_fps"] -and [Math]::Abs([double]$Probe.measured_fps - 60.0) -le 3.0
$CadenceMeasured = $null -ne $Cadence.PSObject.Properties["measured_fps"] -and [Math]::Abs([double]$Cadence.measured_fps - 60.0) -le 3.0
if (-not ($ProbeMeasured -and $CadenceMeasured)) { Add-Failure $Failures "CAPTURE_CADENCE_NOT_60FPS" }
if ([int]$Probe.timestamp_reversals -ne 0 -or [int]$Cadence.timestamp_reversals -ne 0) { Add-Failure $Failures "TIMESTAMP_REVERSAL" }
if ([double]$Cadence.duration_seconds -lt 1800.0) { Add-Failure $Failures "CADENCE_DURATION_UNDER_30_MINUTES" }

$DistortionReady = ([string]$Calibration.distortion.model).ToUpperInvariant() -eq "OPENCV" -and [bool]$Calibration.distortion.verified
if (-not $DistortionReady) { Add-Failure $Failures "DISTORTION_NOT_VERIFIED" }
if ([double]$Calibration.maximum_reference_error_mm -gt 20.0) { Add-Failure $Failures "REFERENCE_ERROR_EXCEEDS_20MM" }
if ([string]$Calibration.gate_status -ne "READY") { Add-Failure $Failures "CALIBRATION_GATE_NOT_READY" }
$CalibrationAge = ([DateTimeOffset]::UtcNow - [DateTimeOffset]::Parse([string]$Calibration.captured_at_utc)).TotalHours
if ($CalibrationAge -lt -1.0 -or $CalibrationAge -gt $MaximumCalibrationAgeHours) { Add-Failure $Failures "CALIBRATION_STALE" }

$ShotTotal = [int]$Shots.published + [int]$Shots.abstained
if ([int]$Shots.attempted -ne 10 -or $ShotTotal -ne 10) { Add-Failure $Failures "TEN_SHOT_SAMPLE_INCOMPLETE" }
if ([int]$Shots.incomplete_authoritative_count -ne 0 -or [int]$Shots.cross_shot_contamination_count -ne 0) { Add-Failure $Failures "INCOMPLETE_SHOT_PUBLISHED" }

$DisconnectReady = [bool]$Disconnect.idle_disconnect_abstained -and [bool]$Disconnect.in_shot_disconnect_abstained -and [bool]$Disconnect.reconnect_ready -and -not [bool]$Disconnect.synthesized_shot_end
if (-not $DisconnectReady) { Add-Failure $Failures "DISCONNECT_NOT_FAIL_CLOSED" }

$SessionIds = @($Sessions.session_ids)
$DistinctSessionIds = @($SessionIds | Sort-Object -Unique)
$SessionsReady = $SessionIds.Count -eq 2 -and $DistinctSessionIds.Count -eq 2 -and [bool]$Sessions.second_session_joinable -and [int]$Sessions.cross_session_leak_count -eq 0
if (-not $SessionsReady) { Add-Failure $Failures "TWO_SESSION_ROLLOVER_FAILED" }

$Status = if ($Failures.Count -eq 0) { "PASS" } else { "FAIL" }
$Summary = [ordered]@{
    schema_version = "1.0"
    kind = "mainline1_usb_live_hardware_acceptance"
    status = $Status
    failures = @($Failures)
    evaluated_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
    source = "raytheon_usb_live"
}

$Parent = Split-Path -Parent $ResolvedOutput
if (-not $Parent) { $Parent = (Get-Location).Path }
New-Item -ItemType Directory -Force -Path $Parent | Out-Null
$Pending = Join-Path $Parent (".mainline1-usb-acceptance-" + [Guid]::NewGuid().ToString("N"))
try {
    New-Item -ItemType Directory -Path $Pending | Out-Null
    foreach ($Name in @("probe.json", "cadence.json", "calibration.json", "shots.json", "disconnect.json", "sessions.json")) {
        Copy-Item -LiteralPath (Join-Path $ResolvedEvidence $Name) -Destination (Join-Path $Pending $Name)
    }
    $Summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Pending "acceptance.json") -Encoding UTF8
    Move-Item -LiteralPath $Pending -Destination $ResolvedOutput
} finally {
    if (Test-Path -LiteralPath $Pending) { Remove-Item -LiteralPath $Pending -Recurse -Force }
}

$Summary | ConvertTo-Json -Depth 6 -Compress | Write-Output
if ($Status -ne "PASS") { exit 2 }
