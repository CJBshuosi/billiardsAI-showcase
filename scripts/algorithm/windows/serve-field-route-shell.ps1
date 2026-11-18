param(
  [Parameter(Mandatory = $true)][string]$ProjectRoot,
  [string]$Artifact = "",
  [string]$RuntimeRoot = "",
  [string]$HostAddress = "0.0.0.0",
  [ValidateRange(1, 65535)][int]$Port = 8787,
  [string]$DisplayHost = "",
  [string]$ControlToken = "",
  [string]$StateDb = "",
  [string]$JoinQrAsset = "",
  [string]$SyncBaseUrl = "",
  [string]$DeviceId = "",
  [string]$CredentialTarget = "",
  [string]$PythonPath = "",
  [string]$CaptureProfilePath = "",
  [string]$Video = "",
  [string]$Mainline1Artifact = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ResolvedProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$RuntimeConfiguration = $null
$CaptureProfile = $null
if (-not $Artifact) {
  if (-not $RuntimeRoot) { throw "Artifact or RuntimeRoot is required" }
  $ResolvedRuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
  $CurrentPointer = Join-Path $ResolvedRuntimeRoot "current.json"
  if (-not (Test-Path -LiteralPath $CurrentPointer -PathType Leaf)) { throw "Field runtime current.json is missing" }
  $RuntimeConfiguration = Get-Content -LiteralPath $CurrentPointer -Raw | ConvertFrom-Json
  $Artifact = [string]$RuntimeConfiguration.artifact
  if (-not $JoinQrAsset) { $JoinQrAsset = [string]$RuntimeConfiguration.join_qr_asset }
  if (-not $SyncBaseUrl) { $SyncBaseUrl = [string]$RuntimeConfiguration.sync_base_url }
  if (-not $DeviceId) { $DeviceId = [string]$RuntimeConfiguration.device_id }
  if (-not $CredentialTarget) { $CredentialTarget = [string]$RuntimeConfiguration.credential_target }
  if (-not $PythonPath) { $PythonPath = [string]$RuntimeConfiguration.python_path }
  if ($null -ne $RuntimeConfiguration.PSObject.Properties["capture_profile"]) {
    $CaptureProfile = $RuntimeConfiguration.capture_profile
  }
}
$ResolvedCaptureProfilePath = ""
if ($CaptureProfilePath) {
  $ResolvedCaptureProfilePath = (Resolve-Path -LiteralPath $CaptureProfilePath).Path
  $CaptureProfile = Get-Content -LiteralPath $ResolvedCaptureProfilePath -Raw | ConvertFrom-Json
}
$ResolvedArtifact = (Resolve-Path -LiteralPath $Artifact).Path
$SourceRoot = Join-Path $ResolvedProjectRoot "src"
$Python = if ($PythonPath) { (Resolve-Path -LiteralPath $PythonPath).Path } else { Join-Path $ResolvedProjectRoot ".venv-rfdetr\Scripts\python.exe" }
$Runner = Join-Path $ResolvedProjectRoot "scripts\algorithm\serve_mainline1_usb_field.py"

if ($StateDb) {
  $ResolvedStateDb = [System.IO.Path]::GetFullPath($StateDb)
  $StateDbParent = Split-Path -Parent $ResolvedStateDb
} else {
  $StateDbParent = Join-Path $env:LOCALAPPDATA "billiardsAI\field-route"
  $ResolvedStateDb = Join-Path $StateDbParent "field-state.sqlite3"
}
New-Item -ItemType Directory -Path $StateDbParent -Force | Out-Null
$ErrorLog = Join-Path $StateDbParent "field-service-error.log"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  throw "Python does not exist: $Python"
}
if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
  throw "Field route shell runner does not exist: $Runner"
}
if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) {
  throw "Field route shell source does not exist: $SourceRoot"
}

$SyncValues = @(@($SyncBaseUrl, $DeviceId, $CredentialTarget) | Where-Object { $_ })
if (($SyncValues.Count -ne 0) -and ($SyncValues.Count -ne 3)) {
  throw "SyncBaseUrl, DeviceId, and CredentialTarget must be configured together"
}
$ResolvedJoinQrAsset = if ($JoinQrAsset) { (Resolve-Path -LiteralPath $JoinQrAsset).Path } else { "" }
$PythonPrefixArguments = @()

if ($null -eq $CaptureProfile) {
  throw "The commercial-pilot field service requires the installed RER U3 capture profile"
}
if ($null -ne $CaptureProfile) {
  if ([string]$CaptureProfile.schema_version -ne "1.0") {
    throw "Unsupported installed capture profile schema."
  }
  if ([string]$CaptureProfile.mode -ne "mainline1_usb_native") {
    throw "Unsupported installed capture profile mode: $($CaptureProfile.mode)"
  }
  $Runner = Join-Path $ResolvedProjectRoot "scripts\algorithm\serve_mainline1_usb_field.py"
  if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
    throw "Installed USB field runner does not exist: $Runner"
  }
  $ResolvedNativeHelper = (Resolve-Path -LiteralPath ([string]$CaptureProfile.native_helper)).Path
  $ResolvedCaptureSetup = (Resolve-Path -LiteralPath ([string]$CaptureProfile.capture_setup)).Path
  $ResolvedCalibrationRoot = (Resolve-Path -LiteralPath ([string]$CaptureProfile.calibration_root)).Path
  $ResolvedDetectorTrainingRun = (Resolve-Path -LiteralPath ([string]$CaptureProfile.detector_training_run)).Path
  $ResolvedIdentityTrainingRun = (Resolve-Path -LiteralPath ([string]$CaptureProfile.identity_training_run)).Path
  $ResolvedDinov2Assets = (Resolve-Path -LiteralPath ([string]$CaptureProfile.dinov2_assets)).Path
  $ResolvedShotConfig = (Resolve-Path -LiteralPath ([string]$CaptureProfile.shot_config)).Path
  $ResolvedRingRoot = [System.IO.Path]::GetFullPath([string]$CaptureProfile.ring_root)
  $ResolvedJobsRoot = [System.IO.Path]::GetFullPath([string]$CaptureProfile.jobs_root)
  $ResolvedLiveArtifactRoot = [System.IO.Path]::GetFullPath([string]$CaptureProfile.live_artifact_root)
  foreach ($Directory in @($ResolvedRingRoot, $ResolvedJobsRoot, $ResolvedLiveArtifactRoot)) {
    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
  }
  $PythonPrefixArguments = @("-u")
  $RunnerArguments = @(
    "--native-helper", $ResolvedNativeHelper,
    "--device-locator", [string]$CaptureProfile.device_locator,
    "--device-fingerprint", [string]$CaptureProfile.device_fingerprint,
    "--capture-setup", $ResolvedCaptureSetup,
    "--calibration-root", $ResolvedCalibrationRoot,
    "--detector-training-run", $ResolvedDetectorTrainingRun,
    "--identity-training-run", $ResolvedIdentityTrainingRun,
    "--dinov2-assets", $ResolvedDinov2Assets,
    "--config", $ResolvedShotConfig,
    "--state-db", $ResolvedStateDb,
    "--ring-root", $ResolvedRingRoot,
    "--jobs-root", $ResolvedJobsRoot,
    "--artifact-root", $ResolvedLiveArtifactRoot,
    "--host", $HostAddress,
    "--port", [string]$Port,
    "--device", [string]$CaptureProfile.device,
    "--batch-size", [string]$CaptureProfile.batch_size,
    "--identity-gate-status", [string]$CaptureProfile.identity_gate_status,
    "--rule-profile", "cbsa-chinese-pool-2017-observable-v1"
  )
  if ([string]$CaptureProfile.identity_gate_status -ne "PASS") {
    # Keep capture and evidence available without falsely granting automatic
    # rule/ranking authority to an identity model that failed its gate.
    $RunnerArguments += "--observation-only"
  }
  if ($ControlToken) { $RunnerArguments += @("--control-token", $ControlToken) }
  if ($ResolvedJoinQrAsset) { $RunnerArguments += @("--join-qr-asset", $ResolvedJoinQrAsset) }
  if ($SyncValues.Count -eq 3) {
    $RunnerArguments += @(
      "--sync-base-url", $SyncBaseUrl,
      "--device-id", $DeviceId,
      "--credential-target", $CredentialTarget,
      "--runtime-environment", [string]$CaptureProfile.runtime_environment,
      "--cloudbase-environment-id", [string]$CaptureProfile.cloudbase_environment_id,
      "--expected-development-environment-id", [string]$CaptureProfile.expected_development_environment_id
    )
  }
}

$env:PYTHONPATH = $SourceRoot
Set-Content -LiteralPath $ErrorLog -Value "" -Encoding UTF8
$PreviousErrorActionPreference = $ErrorActionPreference
$RunnerExitCode = 125
try {
  $ErrorActionPreference = "Continue"
  & $Python @PythonPrefixArguments $Runner @RunnerArguments *>&1 | Tee-Object -FilePath $ErrorLog
  $RunnerExitCode = $LASTEXITCODE
}
finally {
  $ErrorActionPreference = $PreviousErrorActionPreference
}
exit $RunnerExitCode
