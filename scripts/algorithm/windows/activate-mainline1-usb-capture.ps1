#requires -Version 5.1
#requires -RunAsAdministrator
[CmdletBinding()]
param(
  [string]$BundleRoot = "",
  [string]$RuntimeRoot = (Join-Path $env:LOCALAPPDATA "billiardsAI\field-runtime")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if ([string]::IsNullOrWhiteSpace($BundleRoot)) { $BundleRoot = Split-Path -Parent $PSCommandPath }
$Bundle = (Resolve-Path -LiteralPath $BundleRoot).Path
$ResolvedRuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$CurrentPointer = Join-Path $ResolvedRuntimeRoot "current.json"
$ServiceErrorLog = Join-Path $ResolvedRuntimeRoot "field-service-error.log"
$ServiceTaskName = "billiardsAI-field-service"
$KioskTaskName = "billiardsAI-field-kiosk"
$Port = 8787
$InstallClock = [System.Diagnostics.Stopwatch]::StartNew()

function Write-InstallProgress {
  param(
    [Parameter(Mandatory = $true)][string]$Stage,
    [Parameter(Mandatory = $true)][string]$Message,
    [System.Collections.IDictionary]$Details = $null
  )
  $Progress = [ordered]@{
    status = "RUNNING"
    stage = $Stage
    message = $Message
    elapsed_seconds = [Math]::Round($InstallClock.Elapsed.TotalSeconds, 1)
  }
  if ($null -ne $Details) {
    foreach ($Key in $Details.Keys) { $Progress[[string]$Key] = $Details[$Key] }
  }
  Write-Host ($Progress | ConvertTo-Json -Compress)
}

function Get-OptionalPropertyValue {
  param(
    $Object,
    [Parameter(Mandatory = $true)][string]$Name
  )
  if ($null -eq $Object) { return $null }
  $Property = $Object.PSObject.Properties[$Name]
  if ($null -eq $Property) { return $null }
  return $Property.Value
}

function Get-ServiceStartupStage {
  if (-not (Test-Path -LiteralPath $ServiceErrorLog -PathType Leaf)) { return $null }
  $Lines = @(Get-Content -LiteralPath $ServiceErrorLog -Tail 80 -ErrorAction SilentlyContinue)
  [array]::Reverse($Lines)
  foreach ($Line in $Lines) {
    if (-not ([string]$Line).TrimStart().StartsWith([string][char]123)) { continue }
    try {
      $Payload = ([string]$Line) | ConvertFrom-Json
      if ([string]$Payload.status -eq "USB_FIELD_STARTING") {
        return [string]$Payload.stage
      }
    } catch {
      # Third-party model logs are not required to be JSON.
    }
  }
  return $null
}

function Write-ServiceHealthProgress {
  $Details = [ordered]@{
    current_version = $null
    task_state = $null
    source_kind = $null
    media_frame_id = $null
    frame_phase = $null
    display_phase = $null
    recording_status = $null
    video_status = $null
    analysis_status = $null
    route_status = $null
    degraded_reason = $null
    startup_stage = Get-ServiceStartupStage
  }
  try {
    if (Test-Path -LiteralPath $CurrentPointer -PathType Leaf) {
      $Configuration = Get-Content -LiteralPath $CurrentPointer -Raw -Encoding UTF8 | ConvertFrom-Json
      $Details.current_version = Get-OptionalPropertyValue -Object $Configuration -Name "version"
    }
  } catch {
    $Details.current_version = "unreadable"
  }
  try {
    $Task = Get-ScheduledTask -TaskName $ServiceTaskName -ErrorAction Stop
    $Details.task_state = [string]$Task.State
  } catch {
    $Details.task_state = "unavailable"
  }
  try {
    $Health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/screen" -TimeoutSec 2
    $Frame = Get-OptionalPropertyValue -Object $Health -Name "frame"
    $Display = Get-OptionalPropertyValue -Object $Health -Name "display"
    $Recording = Get-OptionalPropertyValue -Object $Health -Name "recording_health"
    $Video = Get-OptionalPropertyValue -Object $Health -Name "video_health"
    $Analysis = Get-OptionalPropertyValue -Object $Health -Name "analysis_health"
    $Route = Get-OptionalPropertyValue -Object $Health -Name "route_health"
    $Details.source_kind = Get-OptionalPropertyValue -Object $Frame -Name "source_kind"
    $Details.media_frame_id = Get-OptionalPropertyValue -Object $Frame -Name "media_frame_id"
    $Details.frame_phase = Get-OptionalPropertyValue -Object $Frame -Name "phase"
    $Details.display_phase = Get-OptionalPropertyValue -Object $Display -Name "phase"
    $Details.recording_status = Get-OptionalPropertyValue -Object $Recording -Name "status"
    $Details.video_status = Get-OptionalPropertyValue -Object $Video -Name "status"
    $Details.analysis_status = Get-OptionalPropertyValue -Object $Analysis -Name "status"
    $Details.route_status = Get-OptionalPropertyValue -Object $Route -Name "status"
    $Details.degraded_reason = Get-OptionalPropertyValue -Object $Health -Name "degraded_reason"
  } catch {
    # The service is expected to be briefly unavailable while its task restarts.
  }
  Write-InstallProgress -Stage "service_health" -Message "Waiting for the requested release to become healthy" -Details $Details
}

function ConvertTo-ProcessArgument {
  param([Parameter(Mandatory = $true)][string]$Value)
  if ($Value -match '[\s"]') { return '"' + $Value.Replace('"', '\"') + '"' }
  return $Value
}

function Invoke-InstallerProcess {
  param(
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [Parameter(Mandatory = $true)][string]$Description,
    [switch]$ReportServiceHealth
  )
  $Stdout = Join-Path $env:TEMP ("billiards-service-installer-" + [Guid]::NewGuid().ToString("N") + ".out")
  $Stderr = Join-Path $env:TEMP ("billiards-service-installer-" + [Guid]::NewGuid().ToString("N") + ".err")
  $Process = $null
  try {
    $CommandLine = @($Arguments | ForEach-Object { ConvertTo-ProcessArgument -Value ([string]$_) }) -join " "
    $Process = Start-Process -FilePath "powershell.exe" -ArgumentList $CommandLine -NoNewWindow -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    $LastProgress = [DateTime]::UtcNow
    while (-not $Process.WaitForExit(1000)) {
      if (([DateTime]::UtcNow - $LastProgress).TotalSeconds -ge 10) {
        if ($ReportServiceHealth) {
          Write-ServiceHealthProgress
        } else {
          Write-InstallProgress -Stage "service_process_running" -Message "Still running: $Description"
        }
        $LastProgress = [DateTime]::UtcNow
      }
    }
    $Process.WaitForExit()
    foreach ($Path in @($Stdout, $Stderr)) {
      if (Test-Path -LiteralPath $Path -PathType Leaf) {
        Get-Content -LiteralPath $Path | ForEach-Object { Write-Host ([string]$_) }
      }
    }
    return [int]$Process.ExitCode
  } finally {
    if ($null -ne $Process -and -not $Process.HasExited) {
      Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
      $Process.WaitForExit()
    }
    if ($null -ne $Process) { $Process.Dispose() }
    foreach ($Path in @($Stdout, $Stderr)) {
      if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Force }
    }
  }
}

function Resolve-BundlePath {
  param([Parameter(Mandatory = $true)][string]$RelativePath)
  if ([System.IO.Path]::IsPathRooted($RelativePath) -or $RelativePath -match '(^|[\\/])\.\.([\\/]|$)') {
    throw "Unsafe bundle path: $RelativePath"
  }
  $Candidate = [System.IO.Path]::GetFullPath((Join-Path $Bundle $RelativePath))
  $Boundary = $Bundle.TrimEnd('\') + '\'
  if (-not $Candidate.StartsWith($Boundary, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Bundle path escaped its root: $RelativePath"
  }
  return $Candidate
}

function Assert-FileHash {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Expected
  )
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Required file is missing: $Path" }
  $Actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($Actual -ne $Expected.ToLowerInvariant()) { throw "File SHA256 mismatch: $Path" }
}

function Resolve-ApprovedNativeCaptureHelper {
  param(
    [Parameter(Mandatory = $true)]$CurrentConfiguration,
    [Parameter(Mandatory = $true)][string]$ExpectedSha256
  )
  if (
    $null -eq $CurrentConfiguration.PSObject.Properties["capture_profile"] -or
    $null -eq $CurrentConfiguration.capture_profile.PSObject.Properties["native_helper"]
  ) {
    throw "The original installed field service has no approved native camera helper"
  }
  $ConfiguredHelper = [string]$CurrentConfiguration.capture_profile.native_helper
  if ([string]::IsNullOrWhiteSpace($ConfiguredHelper)) {
    throw "The original installed field service has an empty native camera helper path"
  }
  $Candidate = (Resolve-Path -LiteralPath $ConfiguredHelper).Path
  $CaptureBoundary = [System.IO.Path]::GetFullPath((Join-Path $ResolvedRuntimeRoot "capture")).TrimEnd('\') + '\'
  if (-not $Candidate.StartsWith($CaptureBoundary, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "The approved native camera helper escaped the installed capture root"
  }
  Assert-FileHash -Path $Candidate -Expected $ExpectedSha256
  return $Candidate
}

function Test-RecorderPreflight {
  param([Parameter(Mandatory = $true)][string]$Root)
  New-Item -ItemType Directory -Path $Root -Force | Out-Null
  $Drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($Root)))
  if ([long]$Drive.AvailableFreeSpace -lt 2GB) { throw "Session recorder does not have its 2 GiB reserve" }
  $Probe = Join-Path $Root (".recording-preflight-" + [Guid]::NewGuid().ToString("N"))
  $Buffer = New-Object byte[] (1MB)
  $Watch = [System.Diagnostics.Stopwatch]::StartNew()
  try {
    $Stream = New-Object System.IO.FileStream($Probe, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None, 1MB, [System.IO.FileOptions]::SequentialScan)
    try {
      for ($Index = 0; $Index -lt 16; $Index++) { $Stream.Write($Buffer, 0, $Buffer.Length) }
      $Stream.Flush($true)
    } finally { $Stream.Dispose() }
  } finally {
    $Watch.Stop()
    if (Test-Path -LiteralPath $Probe) { Remove-Item -LiteralPath $Probe -Force }
  }
  $WriteMbps = 16.0 / [Math]::Max($Watch.Elapsed.TotalSeconds, 0.001)
  if ($WriteMbps -lt 64.0) { throw "Session recorder write throughput is below 64 MiB/s: $([Math]::Round($WriteMbps, 2))" }
  return [pscustomobject]@{ status = "READY"; available_bytes = [long]$Drive.AvailableFreeSpace; measured_write_mibps = [Math]::Round($WriteMbps, 2) }
}

function Wait-AdvancingScreenFrames {
  param([Parameter(Mandatory = $true)][int]$Port)
  $FirstMediaFrameId = $null
  $Deadline = [DateTime]::UtcNow.AddSeconds(15)
  do {
    $Screen = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/screen" -TimeoutSec 3
    $FrameId = $Screen.frame.media_frame_id
    if ($null -ne $FrameId) {
      if ($null -eq $FirstMediaFrameId) { $FirstMediaFrameId = [long]$FrameId }
      elseif ([long]$FrameId -gt $FirstMediaFrameId) { return [pscustomobject]@{ first = $FirstMediaFrameId; last = [long]$FrameId } }
    }
    Start-Sleep -Milliseconds 250
  } while ([DateTime]::UtcNow -lt $Deadline)
  throw "Original field service did not expose advancing camera frames"
}

function Stop-PreviousSidecarProcesses {
  $Deadline = [DateTime]::UtcNow.AddSeconds(15)
  do {
    $Processes = @(
      Get-CimInstance Win32_Process | Where-Object {
        $CommandLine = [string]$_.CommandLine
        $CommandLine -and
          $CommandLine.IndexOf("serve_mainline1_usb_field.py", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
          $CommandLine.IndexOf("billiardsAI-raytheon-eight-ball-field-", [System.StringComparison]::OrdinalIgnoreCase) -ge 0
      }
    )
    foreach ($Process in $Processes) {
      Invoke-CimMethod -InputObject $Process -MethodName Terminate -ErrorAction SilentlyContinue | Out-Null
    }
    if ($Processes.Count -eq 0) { return }
    Start-Sleep -Milliseconds 250
  } while ([DateTime]::UtcNow -lt $Deadline)
  throw "The obsolete USB sidecar did not stop"
}

function Remove-InactiveCaptureRelease {
  param(
    [Parameter(Mandatory = $true)][string]$CaptureRoot,
    [Parameter(Mandatory = $true)]$CurrentConfiguration
  )
  if (-not (Test-Path -LiteralPath $CaptureRoot)) { return }
  $CaptureBoundary = [System.IO.Path]::GetFullPath((Join-Path $ResolvedRuntimeRoot "capture")).TrimEnd('\') + '\'
  $ResolvedCaptureRoot = [System.IO.Path]::GetFullPath($CaptureRoot)
  if (-not $ResolvedCaptureRoot.StartsWith($CaptureBoundary, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Capture release escaped the mutable capture root"
  }
  if ($null -ne $CurrentConfiguration.PSObject.Properties["capture_profile"]) {
    $ActiveCaptureSetup = ""
    if ($null -ne $CurrentConfiguration.capture_profile.PSObject.Properties["capture_setup"]) {
      $ActiveCaptureSetup = [string]$CurrentConfiguration.capture_profile.capture_setup
    }
    if (-not [string]::IsNullOrWhiteSpace($ActiveCaptureSetup)) {
      $ActiveCaptureRoot = Split-Path -Parent $ActiveCaptureSetup
      if ([System.IO.Path]::GetFullPath($ActiveCaptureRoot) -eq $ResolvedCaptureRoot) {
        throw "Capture release is already active: $ResolvedCaptureRoot"
      }
    }
  }
  Remove-Item -LiteralPath $CaptureRoot -Recurse -Force
}

if (-not (Test-Path -LiteralPath $CurrentPointer -PathType Leaf)) {
  throw "The original installed field service is missing current.json: $CurrentPointer"
}
$CurrentConfiguration = Get-Content -LiteralPath $CurrentPointer -Raw -Encoding UTF8 | ConvertFrom-Json

$ManifestPath = Join-Path $Bundle "installed-usb-upgrade-manifest.json"
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) { throw "Installed USB upgrade manifest is missing" }
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ([string]$Manifest.schema_version -ne "1.0" -or [string]$Manifest.kind -ne "commercial_pilot_field_service") {
  throw "Installed USB upgrade manifest is invalid"
}
if ([int]$Manifest.service.port -ne $Port -or [string]$Manifest.service.task_name -ne $ServiceTaskName) {
  throw "Installed USB upgrade does not target the original service"
}
Stop-PreviousSidecarProcesses
foreach ($Entry in $Manifest.files.PSObject.Properties) {
  $Path = Resolve-BundlePath -RelativePath ([string]$Entry.Name)
  Assert-FileHash -Path $Path -Expected ([string]$Entry.Value.sha256)
  if ([long](Get-Item -LiteralPath $Path).Length -ne [long]$Entry.Value.bytes) {
    throw "File byte count mismatch: $Path"
  }
}
Write-InstallProgress -Stage "bundle_verified" -Message "Bundle integrity verified"

$JoinQrCodeKind = [string]$Manifest.join.code_kind
$JoinQrFieldTestOnly = [bool]$Manifest.join.field_test_only
$JoinQrLaunchGateStatus = [string]$Manifest.join.launch_gate_status
if (
  $JoinQrCodeKind -notin @("permanent", "development_preview") -or
  [string]$Manifest.join.environment_id -ne [string]$Manifest.cloud.development_environment_id -or
  [string]$Manifest.join.device_id -ne "raytheon-01" -or
  ($JoinQrCodeKind -eq "permanent" -and ($JoinQrFieldTestOnly -or $JoinQrLaunchGateStatus -ne "PASS")) -or
  ($JoinQrCodeKind -eq "development_preview" -and (-not $JoinQrFieldTestOnly -or $JoinQrLaunchGateStatus -ne "FAIL"))
) {
  throw "Commercial field service table code metadata is invalid"
}
$JoinQrSource = Resolve-BundlePath -RelativePath ([string]$Manifest.join.image_path)
$ExpectedJoinQrSha256 = [string]$Manifest.join.image_sha256
Assert-FileHash -Path $JoinQrSource -Expected $ExpectedJoinQrSha256
$JoinQrRoot = Join-Path $ResolvedRuntimeRoot ("join\" + [string]$Manifest.version)
New-Item -ItemType Directory -Path $JoinQrRoot -Force | Out-Null
$InstalledJoinQr = Join-Path $JoinQrRoot "table-code.png"
Copy-Item -LiteralPath $JoinQrSource -Destination $InstalledJoinQr -Force
Assert-FileHash -Path $InstalledJoinQr -Expected $ExpectedJoinQrSha256

$ReleaseArchive = Resolve-BundlePath -RelativePath ([string]$Manifest.release.path)
Assert-FileHash -Path $ReleaseArchive -Expected ([string]$Manifest.release.sha256)
$Installer = Resolve-BundlePath -RelativePath "install-field-route-service.ps1"
$ExpectedNativeHelperSha256 = [string]$Manifest.capture.approved_native_helper_sha256
if ([string]::IsNullOrWhiteSpace($ExpectedNativeHelperSha256)) {
  throw "Installed USB upgrade did not pin an approved native camera helper"
}
Write-InstallProgress -Stage "approved_native_capture_started" -Message "Checking the previously approved camera helper"
$NativeHelper = Resolve-ApprovedNativeCaptureHelper -CurrentConfiguration $CurrentConfiguration -ExpectedSha256 $ExpectedNativeHelperSha256
Write-InstallProgress -Stage "approved_native_capture_ready" -Message "Previously approved camera helper hash verified"
$CaptureRoot = Join-Path $ResolvedRuntimeRoot ("capture\" + [string]$Manifest.version)
Remove-InactiveCaptureRelease -CaptureRoot $CaptureRoot -CurrentConfiguration $CurrentConfiguration
New-Item -ItemType Directory -Path $CaptureRoot -Force | Out-Null

$CaptureSetupSource = Resolve-BundlePath -RelativePath ([string]$Manifest.capture.capture_setup)
$CalibrationSource = Resolve-BundlePath -RelativePath ([string]$Manifest.capture.calibration_root)
$ShotConfigSource = Resolve-BundlePath -RelativePath ([string]$Manifest.capture.shot_config)
$CaptureSetup = Join-Path $CaptureRoot "capture-setup.json"
$CalibrationRoot = Join-Path $CaptureRoot "calibrations"
$ShotConfig = Join-Path $CaptureRoot "shot-local.yaml"
Copy-Item -LiteralPath $CaptureSetupSource -Destination $CaptureSetup
Copy-Item -LiteralPath $CalibrationSource -Destination $CalibrationRoot -Recurse
Copy-Item -LiteralPath $ShotConfigSource -Destination $ShotConfig
Write-InstallProgress -Stage "camera_contract_started" -Message "Checking the pinned camera mode and identity contract"
$CaptureSetupContract = Get-Content -LiteralPath $CaptureSetup -Raw -Encoding UTF8 | ConvertFrom-Json
if (
  [string]$CaptureSetupContract.schema_version -ne "1.0" -or
  [string]$CaptureSetupContract.source_kind -ne "usb" -or
  [string]$CaptureSetupContract.device_locator_fingerprint -ne [string]$Manifest.capture.device_fingerprint -or
  [int]$CaptureSetupContract.geometry.width -ne 1920 -or
  [int]$CaptureSetupContract.geometry.height -ne 1080 -or
  [Math]::Abs([double]$CaptureSetupContract.requested_fps - 60.0) -gt 0.01
) {
  throw "Pinned camera setup contract is invalid"
}
$CameraContract = [ordered]@{
  status = "READY"
  source_kind = [string]$CaptureSetupContract.source_kind
  device_fingerprint = [string]$CaptureSetupContract.device_locator_fingerprint
  width = [int]$CaptureSetupContract.geometry.width
  height = [int]$CaptureSetupContract.geometry.height
  fps = [double]$CaptureSetupContract.requested_fps
}
Write-InstallProgress -Stage "camera_contract_ready" -Message "Camera contract verified; the service will open and validate one continuous capture"

$AlgorithmRoot = Join-Path $env:LOCALAPPDATA "billiardsAI\algorithm-artifacts"
$ApprovedRoot = Join-Path $env:LOCALAPPDATA "billiardsAI\approved-assets"
$DetectorRoot = Join-Path $AlgorithmRoot ([string]$Manifest.assets.detector.relative_root)
$IdentityRoot = Join-Path $AlgorithmRoot ([string]$Manifest.assets.identity.relative_root)
$Dinov2Root = Join-Path $ApprovedRoot ([string]$Manifest.assets.dinov2.relative_root)
Assert-FileHash -Path (Join-Path $DetectorRoot "training-manifest.json") -Expected ([string]$Manifest.assets.detector.manifest_sha256)
Assert-FileHash -Path (Join-Path $DetectorRoot "checkpoint_best.pth") -Expected ([string]$Manifest.assets.detector.checkpoint_sha256)
Assert-FileHash -Path (Join-Path $IdentityRoot "manifest.json") -Expected ([string]$Manifest.assets.identity.manifest_sha256)
Assert-FileHash -Path (Join-Path $IdentityRoot "group-head.onnx") -Expected ([string]$Manifest.assets.identity.head_sha256)
Assert-FileHash -Path (Join-Path $IdentityRoot "metrics.json") -Expected ([string]$Manifest.assets.identity.metrics_sha256)
Assert-FileHash -Path (Join-Path $Dinov2Root "manifest.json") -Expected ([string]$Manifest.assets.dinov2.manifest_sha256)
Assert-FileHash -Path (Join-Path $Dinov2Root "model.safetensors") -Expected ([string]$Manifest.assets.dinov2.model_sha256)
Assert-FileHash -Path (Join-Path $Dinov2Root "config.json") -Expected ([string]$Manifest.assets.dinov2.config_sha256)
Assert-FileHash -Path (Join-Path $Dinov2Root "preprocessor_config.json") -Expected ([string]$Manifest.assets.dinov2.preprocessor_sha256)

$CaptureProfilePath = Join-Path $CaptureRoot "capture-profile.json"
$CaptureProfile = [ordered]@{
  schema_version = "1.0"
  mode = "mainline1_usb_native"
  native_helper = $NativeHelper
  device_locator = [string]$Manifest.capture.device_locator
  device_fingerprint = [string]$Manifest.capture.device_fingerprint
  capture_setup = $CaptureSetup
  calibration_root = $CalibrationRoot
  detector_training_run = $DetectorRoot
  identity_training_run = $IdentityRoot
  dinov2_assets = $Dinov2Root
  shot_config = $ShotConfig
  ring_root = (Join-Path $ResolvedRuntimeRoot "capture-ring")
  jobs_root = (Join-Path $ResolvedRuntimeRoot "shot-jobs")
  live_artifact_root = (Join-Path $ResolvedRuntimeRoot "live-artifacts")
  device = [string]$Manifest.capture.inference_device
  batch_size = [int]$Manifest.capture.batch_size
  identity_gate_status = [string]$Manifest.assets.identity.gate_status
  runtime_environment = "development"
  cloudbase_environment_id = [string]$Manifest.cloud.development_environment_id
  expected_development_environment_id = [string]$Manifest.cloud.development_environment_id
}
$CaptureProfile | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $CaptureProfilePath -Encoding UTF8
Write-InstallProgress -Stage "recorder_preflight_started" -Message "Checking recording disk capacity and write speed"
$RecorderPreflight = Test-RecorderPreflight -Root ([string]$CaptureProfile.ring_root)
Write-InstallProgress -Stage "recorder_preflight_ready" -Message "Recording disk check passed"

$InstallArguments = @(
  "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Installer,
  "-ReleaseArchive", $ReleaseArchive,
  "-ExpectedSha256", [string]$Manifest.release.sha256,
  "-CaptureProfile", $CaptureProfilePath,
  "-JoinQrAsset", $InstalledJoinQr,
  "-RuntimeRoot", $ResolvedRuntimeRoot,
  "-Port", [string]$Port
)
Write-InstallProgress -Stage "service_preflight_started" -Message "Checking the original service, models, and install prerequisites"
$PreflightExitCode = Invoke-InstallerProcess -Arguments @($InstallArguments + "-CheckOnly") -Description "service preflight"
if ($PreflightExitCode -ne 0) { throw "Installed USB upgrade preflight failed" }
Write-InstallProgress -Stage "service_preflight_ready" -Message "Original service install prerequisites passed"
Write-InstallProgress -Stage "service_install_started" -Message "Updating the original service; port 8787 will restart"
$InstallExitCode = Invoke-InstallerProcess -Arguments $InstallArguments -Description "service installation and health verification" -ReportServiceHealth
if ($InstallExitCode -ne 0) { throw "Installed USB upgrade failed; see the installer error above for rollback status" }
$InstalledConfiguration = Get-Content -LiteralPath $CurrentPointer -Raw -Encoding UTF8 | ConvertFrom-Json
$ExpectedInstalledRelease = [System.IO.Path]::GetFullPath((Join-Path (Join-Path (Split-Path -Parent $ResolvedRuntimeRoot) "field-releases") ([string]$Manifest.version)))
$ObservedInstalledRelease = [string]$InstalledConfiguration.current_release
if (
  [string]$InstalledConfiguration.version -ne [string]$Manifest.version -or
  [string]::IsNullOrWhiteSpace($ObservedInstalledRelease) -or
  -not [string]::Equals(
    [System.IO.Path]::GetFullPath($ObservedInstalledRelease),
    $ExpectedInstalledRelease,
    [System.StringComparison]::OrdinalIgnoreCase
  )
) {
  throw "Installed service release mismatch: expected version=$([string]$Manifest.version), release=$ExpectedInstalledRelease; observed version=$([string]$InstalledConfiguration.version), release=$ObservedInstalledRelease"
}
Write-InstallProgress -Stage "service_install_ready" -Message "Original service update and internal health checks passed"

Write-InstallProgress -Stage "advancing_frames_started" -Message "Confirming that live camera frame IDs keep advancing"
$Frames = Wait-AdvancingScreenFrames -Port $Port
Write-InstallProgress -Stage "advancing_frames_ready" -Message "Live camera frame IDs are advancing"
$Screen = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/screen" -TimeoutSec 10
$Qr = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/api/table-code.png" -TimeoutSec 10
if ($Qr.StatusCode -ne 200) { throw "Original field service HTTP contract is not healthy" }
$InstalledJoinQrAfter = (Resolve-Path -LiteralPath ([string]$InstalledConfiguration.join_qr_asset)).Path
$JoinQrSha256After = (Get-FileHash -LiteralPath $InstalledJoinQrAfter -Algorithm SHA256).Hash.ToLowerInvariant()
if ($InstalledJoinQrAfter -ne $InstalledJoinQr -or $JoinQrSha256After -ne $ExpectedJoinQrSha256) {
  throw "Manifest-pinned table QR was not activated"
}
Write-InstallProgress -Stage "permanent_qr_verified" -Message "Manifest-pinned table QR is active ($JoinQrCodeKind)"
$Receipt = [ordered]@{
  status = "PASS"
  release = [string]$Manifest.version
  task_name = $ServiceTaskName
  kiosk_task_name = $KioskTaskName
  port = $Port
  capture_mode = "mainline1_usb_native"
  identity_gate_status = [string]$Manifest.assets.identity.gate_status
  first_media_frame_id = [long]$Frames.first
  last_media_frame_id = [long]$Frames.last
  recording_preflight = $RecorderPreflight
  camera_contract = $CameraContract
  join_qr_sha256_after = $JoinQrSha256After
  join_qr_code_kind = [string]$Manifest.join.code_kind
  join_qr_launch_gate_status = $JoinQrLaunchGateStatus
}
$Receipt | ConvertTo-Json -Depth 8 -Compress
Write-Host "PASS: RER U3 is now the capture source of the original installed field service on port $Port"
