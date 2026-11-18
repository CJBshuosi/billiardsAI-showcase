[CmdletBinding()]
param(
  [string]$ReleaseArchive = "",
  [string]$ExpectedSha256 = "",
  [string]$Artifact = "",
  [string]$JoinQrAsset = "",
  [string]$SyncBaseUrl = "",
  [string]$DeviceId = "",
  [string]$CredentialTarget = "",
  [string]$BrowserPath = "",
  [string]$PythonPath = "",
  [string]$CaptureProfile = "",
  [string]$RuntimeRoot = (Join-Path $env:LOCALAPPDATA "billiardsAI\field-runtime"),
  [string]$ReleasesRoot = (Join-Path $env:LOCALAPPDATA "billiardsAI\field-releases"),
  [ValidateRange(1, 65535)][int]$Port = 8787,
  [ValidateRange(-1, 2147483647)][int]$ExpectedShotCount = -1,
  [switch]$CheckOnly,
  [switch]$SwitchArtifact,
  [switch]$Rollback
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ServiceTaskName = "billiardsAI-field-service"
$KioskTaskName = "billiardsAI-field-kiosk"
$ScheduledTaskRunningResult = 267009
$ResolvedRuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$ResolvedReleasesRoot = [System.IO.Path]::GetFullPath($ReleasesRoot)
$CurrentPointer = Join-Path $ResolvedRuntimeRoot "current.json"
$StateDbPath = Join-Path $ResolvedRuntimeRoot "field-state.sqlite3"
$StateDbCompanionPaths = @(
  $StateDbPath,
  (Join-Path $ResolvedRuntimeRoot "field-state.sqlite3-wal"),
  (Join-Path $ResolvedRuntimeRoot "field-state.sqlite3-shm")
)
$ServiceErrorLog = Join-Path $ResolvedRuntimeRoot "field-service-error.log"
$JoinQrAssetExplicit = $PSBoundParameters.ContainsKey("JoinQrAsset") -and -not [string]::IsNullOrWhiteSpace($JoinQrAsset)

function ConvertTo-TaskArgument {
  param([Parameter(Mandatory = $true)][string]$Value)
  return '"' + $Value.Replace('"', '\"') + '"'
}

function Assert-UnderRoot {
  param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Root)
  $FullPath = [System.IO.Path]::GetFullPath($Path)
  $Separator = [System.IO.Path]::DirectorySeparatorChar
  $FullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd($Separator) + $Separator
  $Comparison = if ($Separator -eq '\') { [System.StringComparison]::OrdinalIgnoreCase } else { [System.StringComparison]::Ordinal }
  if (-not $FullPath.StartsWith($FullRoot, $Comparison)) {
    throw "Path escapes the approved root: $FullPath"
  }
  return $FullPath
}

function Read-ReleaseManifest {
  param([Parameter(Mandatory = $true)][string]$ReleaseRoot)
  $ManifestPath = Join-Path $ReleaseRoot "release-manifest.json"
  if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) { throw "release-manifest.json is missing" }
  $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
  if ($Manifest.schema_version -ne "1.0") { throw "Unsupported release manifest schema" }
  $Version = [string]$Manifest.version
  if ($Version -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$') { throw "Release version is invalid" }
  return $Manifest
}

function Read-CaptureProfile {
  param([Parameter(Mandatory = $true)][string]$Path)
  $ResolvedProfilePath = (Resolve-Path -LiteralPath $Path).Path
  $Profile = Get-Content -LiteralPath $ResolvedProfilePath -Raw | ConvertFrom-Json
  if ([string]$Profile.schema_version -ne "1.0") { throw "Unsupported capture profile schema" }
  if ([string]$Profile.mode -ne "mainline1_usb_native") { throw "Unsupported capture profile mode" }
  if ([string]$Profile.device_fingerprint -notmatch '^[0-9a-fA-F]{64}$') { throw "Capture device fingerprint is invalid" }
  if ([string]::IsNullOrWhiteSpace([string]$Profile.device_locator)) { throw "Capture device locator is missing" }
  if ([int]$Profile.batch_size -lt 1) { throw "Capture batch size is invalid" }
  if ([string]$Profile.device -notin @("cuda", "cpu")) { throw "Capture inference device is invalid" }
  foreach ($PropertyName in @("native_helper", "capture_setup", "shot_config")) {
    $Candidate = [string]$Profile.$PropertyName
    if (-not $Candidate -or -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
      throw "Capture profile file is missing: $PropertyName"
    }
    $Profile.$PropertyName = (Resolve-Path -LiteralPath $Candidate).Path
  }
  foreach ($PropertyName in @("calibration_root", "detector_training_run", "identity_training_run", "dinov2_assets")) {
    $Candidate = [string]$Profile.$PropertyName
    if (-not $Candidate -or -not (Test-Path -LiteralPath $Candidate -PathType Container)) {
      throw "Capture profile directory is missing: $PropertyName"
    }
    $Profile.$PropertyName = (Resolve-Path -LiteralPath $Candidate).Path
  }
  foreach ($PropertyName in @("ring_root", "jobs_root", "live_artifact_root")) {
    $Candidate = Assert-UnderRoot -Path ([string]$Profile.$PropertyName) -Root $ResolvedRuntimeRoot
    New-Item -ItemType Directory -Path $Candidate -Force | Out-Null
    $Profile.$PropertyName = $Candidate
  }
  return $Profile
}

function Register-FieldTasks {
  param([Parameter(Mandatory = $true)]$Configuration)
  $ReleaseRoot = Assert-UnderRoot -Path ([string]$Configuration.current_release) -Root $ResolvedReleasesRoot
  $Launcher = Join-Path $ReleaseRoot "scripts\algorithm\windows\serve-field-route-shell.ps1"
  $KioskLauncher = Join-Path $ReleaseRoot "scripts\algorithm\windows\start-field-route-kiosk.ps1"
  if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) { throw "Service launcher missing: $Launcher" }
  if (-not (Test-Path -LiteralPath $KioskLauncher -PathType Leaf)) { throw "Kiosk launcher missing: $KioskLauncher" }
  $ServiceParts = @("-NoProfile", "-WindowStyle", "Hidden", "-ExecutionPolicy", "Bypass", "-File", (ConvertTo-TaskArgument $Launcher), "-ProjectRoot", (ConvertTo-TaskArgument $ReleaseRoot), "-RuntimeRoot", (ConvertTo-TaskArgument $ResolvedRuntimeRoot), "-StateDb", (ConvertTo-TaskArgument (Join-Path $ResolvedRuntimeRoot "field-state.sqlite3")), "-JoinQrAsset", (ConvertTo-TaskArgument ([string]$Configuration.join_qr_asset)), "-HostAddress", "127.0.0.1", "-Port", [string]$Port)
  $CaptureProfileProperty = $Configuration.PSObject.Properties["capture_profile_path"]
  if ($null -ne $CaptureProfileProperty -and -not [string]::IsNullOrWhiteSpace([string]$Configuration.capture_profile_path)) {
    $ServiceParts += @("-CaptureProfilePath", (ConvertTo-TaskArgument ([string]$Configuration.capture_profile_path)))
  }
  if ([string]$Configuration.python_path) { $ServiceParts += @("-PythonPath", (ConvertTo-TaskArgument ([string]$Configuration.python_path))) }
  if ([string]$Configuration.sync_base_url) { $ServiceParts += @("-SyncBaseUrl", (ConvertTo-TaskArgument ([string]$Configuration.sync_base_url)), "-DeviceId", (ConvertTo-TaskArgument ([string]$Configuration.device_id)), "-CredentialTarget", (ConvertTo-TaskArgument ([string]$Configuration.credential_target))) }
  $KioskParts = @("-NoProfile", "-WindowStyle", "Hidden", "-ExecutionPolicy", "Bypass", "-File", (ConvertTo-TaskArgument $KioskLauncher), "-BrowserPath", (ConvertTo-TaskArgument ([string]$Configuration.browser_path)), "-RuntimeRoot", (ConvertTo-TaskArgument $ResolvedRuntimeRoot))
  $ServiceAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument ($ServiceParts -join " ")
  $KioskAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument ($KioskParts -join " ")
  $Trigger = New-ScheduledTaskTrigger -AtLogOn
  $Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
  $Settings = New-ScheduledTaskSettingsSet -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 3650) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
  foreach ($TaskName in @($ServiceTaskName, $KioskTaskName)) {
    if ($null -ne (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue)) {
      Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
  }
  Register-ScheduledTask -TaskName $ServiceTaskName -Action $ServiceAction -Trigger $Trigger -Principal $Principal -Settings $Settings -Force | Out-Null
  Register-ScheduledTask -TaskName $KioskTaskName -Action $KioskAction -Trigger $Trigger -Principal $Principal -Settings $Settings -Force | Out-Null
}

function Export-FieldTaskDefinitions {
  $Snapshot = @{}
  foreach ($TaskName in @($ServiceTaskName, $KioskTaskName)) {
    $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    $Snapshot[$TaskName] = if ($null -eq $Task) { $null } else { Export-ScheduledTask -TaskName $TaskName }
  }
  return $Snapshot
}

function Restore-FieldTaskDefinitions {
  param([Parameter(Mandatory = $true)][hashtable]$Snapshot)
  foreach ($TaskName in @($ServiceTaskName, $KioskTaskName)) {
    if ($null -ne (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue)) {
      Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
    if ($Snapshot.ContainsKey($TaskName) -and $null -ne $Snapshot[$TaskName]) {
      Register-ScheduledTask -TaskName $TaskName -Xml ([string]$Snapshot[$TaskName]) -Force | Out-Null
    }
  }
}

function Retire-ExistingFieldSession {
  param(
    [Parameter(Mandatory = $true)][string]$ReleaseRoot,
    [Parameter(Mandatory = $true)][string]$StateDb,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$SyncBaseUrl,
    [Parameter(Mandatory = $true)][string]$DeviceId,
    [Parameter(Mandatory = $true)][string]$CredentialTarget
  )
  if (-not (Test-Path -LiteralPath $StateDb -PathType Leaf)) { return $null }
  foreach ($Value in @($Python, $SyncBaseUrl, $DeviceId, $CredentialTarget)) {
    if ([string]::IsNullOrWhiteSpace($Value)) {
      throw "Existing field session cannot be safely retired without cloud sync configuration"
    }
  }
  $ResolvedPython = (Resolve-Path -LiteralPath $Python).Path
  $RetirementScript = Join-Path $ReleaseRoot "scripts\algorithm\retire_existing_field_session.py"
  if (-not (Test-Path -LiteralPath $RetirementScript -PathType Leaf)) {
    throw "Installed field session retirement helper is missing"
  }
  $ReceiptRoot = Join-Path $ResolvedRuntimeRoot "session-retirement-receipts"
  New-Item -ItemType Directory -Path $ReceiptRoot -Force | Out-Null
  $ReceiptPath = Join-Path $ReceiptRoot ("retirement-" + [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmssfff") + ".json")
  $OldPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
  try {
    [Environment]::SetEnvironmentVariable("PYTHONPATH", (Join-Path $ReleaseRoot "src"), "Process")
    $RetirementArguments = @(
      $RetirementScript,
      "--state-db", $StateDb,
      "--sync-base-url", $SyncBaseUrl,
      "--device-id", $DeviceId,
      "--credential-target", $CredentialTarget,
      "--receipt", $ReceiptPath
    )
    $RetirementOutput = @(& $ResolvedPython @RetirementArguments 2>&1)
    $RetirementExitCode = $LASTEXITCODE
    if ($RetirementExitCode -ne 0) {
      throw "Existing field session retirement failed: $($RetirementOutput -join ' ')"
    }
    if (-not (Test-Path -LiteralPath $ReceiptPath -PathType Leaf)) {
      throw "Existing field session retirement did not produce a receipt"
    }
    $Receipt = Get-Content -LiteralPath $ReceiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (
      [string]$Receipt.schema_version -ne "1.0" -or
      [string]$Receipt.status -notin @("RETIRED", "NOT_NEEDED") -or
      [string]$Receipt.table_id -ne $DeviceId
    ) {
      throw "Existing field session retirement receipt is invalid"
    }
    Write-Host ($Receipt | ConvertTo-Json -Compress)
    return $Receipt
  } finally {
    [Environment]::SetEnvironmentVariable("PYTHONPATH", $OldPythonPath, "Process")
  }
}

function Wait-FieldTaskStopped {
  param([Parameter(Mandatory = $true)][string]$TaskName)
  $Deadline = [DateTime]::UtcNow.AddSeconds(30)
  do {
    $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -eq $Task) { return }
    $TaskInfo = Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction SilentlyContinue
    $SchedulerInstanceStopped = $null -eq $TaskInfo -or [long]$TaskInfo.LastTaskResult -ne $ScheduledTaskRunningResult
    if ($Task.State -ne "Running" -and $SchedulerInstanceStopped) { return }
    Start-Sleep -Milliseconds 250
  } while ([DateTime]::UtcNow -lt $Deadline)
  $LastResult = if ($null -eq $TaskInfo) { "unknown" } else { [string]$TaskInfo.LastTaskResult }
  throw "Scheduled task instance did not finish stopping: task=$TaskName state=$($Task.State) last_task_result=$LastResult"
}

function Stop-FieldServiceProcesses {
  $ExpectedStateDb = [System.IO.Path]::GetFullPath((Join-Path $ResolvedRuntimeRoot "field-state.sqlite3"))
  $Deadline = [DateTime]::UtcNow.AddSeconds(15)
  do {
    $ResidualProcesses = @(
      Get-CimInstance Win32_Process | Where-Object {
        $CommandLine = [string]$_.CommandLine
        $CommandLine -and
          ($CommandLine.IndexOf("serve_field_route_shell.py", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
            $CommandLine.IndexOf("serve_mainline1_usb_field.py", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) -and
          $CommandLine.IndexOf($ExpectedStateDb, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
      }
    )
    foreach ($Process in $ResidualProcesses) {
      Invoke-CimMethod -InputObject $Process -MethodName Terminate -ErrorAction SilentlyContinue | Out-Null
    }
    if ($ResidualProcesses.Count -eq 0) { return }
    Start-Sleep -Milliseconds 250
  } while ([DateTime]::UtcNow -lt $Deadline)
  throw "Residual field service process did not terminate: $ExpectedStateDb"
}

function Stop-FieldTasks {
  foreach ($TaskName in @($ServiceTaskName, $KioskTaskName)) {
    $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -eq $Task) { continue }
    if ($Task.State -eq "Running") { Stop-ScheduledTask -TaskName $TaskName }
    Wait-FieldTaskStopped -TaskName $TaskName
  }
  Stop-FieldServiceProcesses
  $Deadline = [DateTime]::UtcNow.AddSeconds(15)
  do {
    $Listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    if ($Listeners.Count -eq 0) { return }
    Start-Sleep -Milliseconds 250
  } while ([DateTime]::UtcNow -lt $Deadline)
  $ListenerPids = @($Listeners | ForEach-Object { [string]$_.OwningProcess }) -join ","
  throw "Field service port did not become available: port=$Port listener_pids=$ListenerPids"
}

function Write-CurrentConfigurationAtomically {
  param([Parameter(Mandatory = $true)]$Configuration)
  New-Item -ItemType Directory -Path $ResolvedRuntimeRoot -Force | Out-Null
  $TemporaryPointer = Join-Path $ResolvedRuntimeRoot (".current-" + [Guid]::NewGuid().ToString("N") + ".json")
  $BackupPointer = Join-Path $ResolvedRuntimeRoot (".current-backup-" + [Guid]::NewGuid().ToString("N") + ".json")
  try {
    $Configuration | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $TemporaryPointer -Encoding UTF8
    if (Test-Path -LiteralPath $CurrentPointer -PathType Leaf) {
      [System.IO.File]::Replace($TemporaryPointer, $CurrentPointer, $BackupPointer, $true)
    } else {
      [System.IO.File]::Move($TemporaryPointer, $CurrentPointer)
    }
  } finally {
    if (Test-Path -LiteralPath $TemporaryPointer -PathType Leaf) {
      Remove-Item -LiteralPath $TemporaryPointer -Force
    }
    if (Test-Path -LiteralPath $BackupPointer -PathType Leaf) {
      Remove-Item -LiteralPath $BackupPointer -Force
    }
  }
}

function Wait-FieldServiceHealthy {
  param(
    [switch]$RequireUsbCapture,
    [ValidateRange(60, 900)][int]$StartupTimeoutSeconds = 300
  )
  # CUDA model loading and the first detector/identity inference are cold-start
  # work, not evidence that the scheduled task is dead.  Give that bounded work
  # enough time while still failing promptly if the task process actually exits.
  $StartedAt = [DateTime]::UtcNow
  $Deadline = $StartedAt.AddSeconds($StartupTimeoutSeconds)
  $Health = $null
  $FirstMediaFrameId = $null
  $LastStartError = ""
  $LastObservedSourceKind = ""
  $StartAttempted = $false
  $ScheduledTaskExited = $false
  Set-Content -LiteralPath $ServiceErrorLog -Value "" -Encoding UTF8
  do {
    $Task = Get-ScheduledTask -TaskName $ServiceTaskName -ErrorAction SilentlyContinue
    if ($null -eq $Task) { throw "Field service scheduled task is missing: $ServiceTaskName" }
    if (-not $StartAttempted) {
      try {
        if ($Task.State -ne "Running") {
          Start-ScheduledTask -TaskName $ServiceTaskName
        }
        $LastStartError = ""
      } catch {
        $LastStartError = $_.Exception.Message
      }
      $StartAttempted = $true
    }
    try { $Health = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/api/screen" -TimeoutSec 2 } catch { $Health = $null }
    if ($null -ne $Health -and $Health.StatusCode -eq 200) {
      $TableCode = $null
      try { $TableCode = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/api/table-code.png" -TimeoutSec 2 } catch { $TableCode = $null }
      if ($null -eq $TableCode -or $TableCode.StatusCode -ne 200) {
        Start-Sleep -Milliseconds 250
        continue
      }
      if (-not $RequireUsbCapture) { return }
      $HealthPayload = $null
      try {
        $HealthPayload = $Health.Content | ConvertFrom-Json
        $LastObservedSourceKind = [string]$HealthPayload.frame.source_kind
      } catch {
        $LastObservedSourceKind = "invalid"
      }
      $CurrentMediaFrameId = if ($null -eq $HealthPayload) { $null } else { $HealthPayload.frame.media_frame_id }
      if ($null -ne $CurrentMediaFrameId -and $null -eq $FirstMediaFrameId) {
        $FirstMediaFrameId = [long]$CurrentMediaFrameId
      }
      $RecordingStatus = if ($null -eq $HealthPayload) { "unavailable" } else { [string]$HealthPayload.recording_health.status }
      $RecordingReady = $RecordingStatus -eq "ready" -or (
        [string]$HealthPayload.display.phase -eq "WAITING_FOR_PLAYERS" -and $RecordingStatus -eq "unavailable"
      )
      if (
        $null -ne $HealthPayload -and
        [string]$HealthPayload.frame.source_kind -eq "usb_live" -and
        $null -ne $CurrentMediaFrameId -and
        $null -ne $FirstMediaFrameId -and
        [long]$CurrentMediaFrameId -gt [long]$FirstMediaFrameId -and
        $RecordingReady -and
        $HealthPayload.frame.phase -notin @("degraded", "recovering") -and
        $HealthPayload.frame.rule_status -ne "DEGRADED" -and
        [string]::IsNullOrWhiteSpace($HealthPayload.frame.degraded_reason)
      ) { return }
    }
    $Task = Get-ScheduledTask -TaskName $ServiceTaskName -ErrorAction SilentlyContinue
    $TaskInfo = Get-ScheduledTaskInfo -TaskName $ServiceTaskName -ErrorAction SilentlyContinue
    $ScheduledTaskExited = (
      $StartAttempted -and
      ([DateTime]::UtcNow - $StartedAt).TotalSeconds -ge 3 -and
      ($null -eq $Task -or $Task.State -ne "Running") -and
      ($null -eq $TaskInfo -or [long]$TaskInfo.LastTaskResult -ne $ScheduledTaskRunningResult)
    )
    if ($ScheduledTaskExited) { break }
    Start-Sleep -Seconds 1
  } while ([DateTime]::UtcNow -lt $Deadline)

  $Task = Get-ScheduledTask -TaskName $ServiceTaskName -ErrorAction SilentlyContinue
  $TaskInfo = Get-ScheduledTaskInfo -TaskName $ServiceTaskName -ErrorAction SilentlyContinue
  $TaskState = if ($null -eq $Task) { "missing" } else { [string]$Task.State }
  $LastTaskResult = if ($null -eq $TaskInfo) { "unknown" } else { [string]$TaskInfo.LastTaskResult }
  $Listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
  $ListenerPids = @($Listeners | ForEach-Object { [string]$_.OwningProcess }) -join ","
  if (-not $ListenerPids) { $ListenerPids = "none" }
  $ExpectedStateDb = [System.IO.Path]::GetFullPath((Join-Path $ResolvedRuntimeRoot "field-state.sqlite3"))
  $ServiceProcesses = @(
    Get-CimInstance Win32_Process | Where-Object {
      $CommandLine = [string]$_.CommandLine
      $CommandLine -and
        ($CommandLine.IndexOf("serve_field_route_shell.py", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
          $CommandLine.IndexOf("serve_mainline1_usb_field.py", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) -and
        $CommandLine.IndexOf($ExpectedStateDb, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    }
  )
  $ServicePids = @($ServiceProcesses | ForEach-Object { [string]$_.ProcessId }) -join ","
  if (-not $ServicePids) { $ServicePids = "none" }
  $Details = @("task_state=$TaskState", "last_task_result=$LastTaskResult", "listener_pids=$ListenerPids", "service_pids=$ServicePids")
  if ($ScheduledTaskExited) { $Details += "failure_kind=scheduled_task_exited" }
  if ($RequireUsbCapture) { $Details += "observed_source_kind=$LastObservedSourceKind" }
  if ($LastStartError) { $Details += "start_error=$LastStartError" }
  if (Test-Path -LiteralPath $ServiceErrorLog -PathType Leaf) {
    $LogDetail = ((Get-Content -LiteralPath $ServiceErrorLog -Tail 20) -join " | ").Trim()
    if ($LogDetail) { $Details += "stderr=$LogDetail" }
  }
  throw "Installed service did not become healthy: $($Details -join '; ')"
}

function Assert-FieldReplayArtifact {
  param(
    [Parameter(Mandatory = $true)][string]$ArtifactRoot,
    [int]$RequiredShotCount = -1
  )
  $ResolvedArtifact = (Resolve-Path -LiteralPath $ArtifactRoot).Path
  $ArtifactItem = Get-Item -LiteralPath $ResolvedArtifact
  if (-not $ArtifactItem.PSIsContainer -or ($ArtifactItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
    throw "Field replay artifact is missing or unsafe: $ResolvedArtifact"
  }
  $ManifestPath = Join-Path $ResolvedArtifact "manifest.json"
  if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) { throw "Field replay manifest is missing" }
  $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
  if ($Manifest.schema_version -ne "1.0") { throw "Unsupported field replay manifest schema" }
  if ($Manifest.kind -ne "field_route_file_replay") { throw "Field replay manifest kind is invalid" }
  if ($Manifest.pipeline_version -ne "field-route-file-replay/2") { throw "Field replay pipeline is not the authoritative v2 replay" }
  if ($null -eq $Manifest.inventory) { throw "Field replay inventory is invalid" }
  $Inventory = @($Manifest.inventory.PSObject.Properties)
  if ($Inventory.Count -eq 0) { throw "Field replay inventory is empty" }
  foreach ($Entry in $Inventory) {
    $Name = [string]$Entry.Name
    if ([System.IO.Path]::GetFileName($Name) -ne $Name) { throw "Field replay inventory path is unsafe: $Name" }
    $Candidate = Join-Path $ResolvedArtifact $Name
    if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { throw "Field replay inventory file is missing: $Name" }
    $CandidateItem = Get-Item -LiteralPath $Candidate
    if ($CandidateItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw "Field replay inventory file is unsafe: $Name" }
    if ([long]$Entry.Value.bytes -ne [long]$CandidateItem.Length) { throw "Field replay inventory byte count mismatch: $Name" }
    $ActualHash = (Get-FileHash -LiteralPath $Candidate -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualHash -ne ([string]$Entry.Value.sha256).ToLowerInvariant()) { throw "Field replay inventory hash mismatch: $Name" }
  }
  if ($null -eq $Manifest.inventory.PSObject.Properties["summary.json"]) { throw "Field replay summary is not inventoried" }
  $Summary = Get-Content -LiteralPath (Join-Path $ResolvedArtifact "summary.json") -Raw | ConvertFrom-Json
  if ($Summary.schema_version -ne "1.0" -or $Summary.status -ne "PASS") { throw "Field replay summary is not a passing v1 summary" }
  if ($RequiredShotCount -ge 0 -and [int]$Summary.shot_count -ne $RequiredShotCount) {
    throw "Field replay shot_count $($Summary.shot_count) does not equal expected $RequiredShotCount"
  }
  return [pscustomobject]@{ artifact_root = $ResolvedArtifact; manifest = $Manifest; summary = $Summary }
}

function Restore-FieldConfiguration {
  param([Parameter(Mandatory = $true)][string]$OriginalJson)
  $OriginalConfiguration = $OriginalJson | ConvertFrom-Json
  Stop-FieldTasks
  Write-CurrentConfigurationAtomically -Configuration $OriginalConfiguration
  Wait-FieldServiceHealthy
  Start-ScheduledTask -TaskName $KioskTaskName
}

if ($Rollback) {
  if (-not (Test-Path -LiteralPath $CurrentPointer -PathType Leaf)) { throw "No current.json pointer exists" }
  $Configuration = Get-Content -LiteralPath $CurrentPointer -Raw | ConvertFrom-Json
  if (-not [string]$Configuration.previous_release) { throw "No previous release is available" }
  $Previous = Assert-UnderRoot -Path ([string]$Configuration.previous_release) -Root $ResolvedReleasesRoot
  Read-ReleaseManifest -ReleaseRoot $Previous | Out-Null
  if ($CheckOnly) { Write-Host "READY: rollback target $Previous"; exit 0 }
  $OldCurrent = [string]$Configuration.current_release
  $Configuration.current_release = $Previous
  $Configuration.previous_release = $OldCurrent
  Stop-FieldTasks
  $Configuration | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $CurrentPointer -Encoding UTF8
  Register-FieldTasks -Configuration $Configuration
  Wait-FieldServiceHealthy
  Write-Host "PASS: rolled back to $Previous; $ResolvedRuntimeRoot was preserved"
  exit 0
}

function Restore-FieldReleaseConfiguration {
  param(
    [Parameter(Mandatory = $true)][string]$OriginalJson,
    [byte[]]$OriginalPointerBytes = @(),
    [string]$StateDbBackup = "",
    [hashtable]$TaskDefinitionSnapshot = $null
  )
  $OriginalConfiguration = $OriginalJson | ConvertFrom-Json
  Stop-FieldTasks
  if ($OriginalPointerBytes.Count -gt 0) {
    [System.IO.File]::WriteAllBytes($CurrentPointer, $OriginalPointerBytes)
  } else {
    Write-CurrentConfigurationAtomically -Configuration $OriginalConfiguration
  }
  if ($StateDbBackup -and (Test-Path -LiteralPath $StateDbBackup -PathType Container)) {
    foreach ($StatePath in $StateDbCompanionPaths) {
      if (Test-Path -LiteralPath $StatePath -PathType Leaf) {
        Remove-Item -LiteralPath $StatePath -Force
      }
      $BackupPath = Join-Path $StateDbBackup ([System.IO.Path]::GetFileName($StatePath))
      if (Test-Path -LiteralPath $BackupPath -PathType Leaf) {
        Copy-Item -LiteralPath $BackupPath -Destination $StatePath -Force
      }
    }
  }
  if ($null -ne $TaskDefinitionSnapshot) {
    Restore-FieldTaskDefinitions -Snapshot $TaskDefinitionSnapshot
  } else {
    Register-FieldTasks -Configuration $OriginalConfiguration
  }
  Wait-FieldServiceHealthy
  Start-ScheduledTask -TaskName $KioskTaskName
}

if ($SwitchArtifact) {
  if (-not (Test-Path -LiteralPath $CurrentPointer -PathType Leaf)) { throw "No current.json pointer exists" }
  if (-not $Artifact) { throw "Artifact is required for an artifact switch" }
  $VerifiedArtifact = Assert-FieldReplayArtifact -ArtifactRoot $Artifact -RequiredShotCount $ExpectedShotCount
  $OriginalJson = Get-Content -LiteralPath $CurrentPointer -Raw
  $Configuration = $OriginalJson | ConvertFrom-Json
  Read-ReleaseManifest -ReleaseRoot (Assert-UnderRoot -Path ([string]$Configuration.current_release) -Root $ResolvedReleasesRoot) | Out-Null
  if ($CheckOnly) {
    [pscustomobject]@{ status = "READY"; artifact = $VerifiedArtifact.artifact_root; shot_count = [int]$VerifiedArtifact.summary.shot_count; current_release = [string]$Configuration.current_release } | ConvertTo-Json -Compress
    exit 0
  }
  $Configuration.artifact = $VerifiedArtifact.artifact_root
  try {
    Stop-FieldTasks
    Write-CurrentConfigurationAtomically -Configuration $Configuration
    Wait-FieldServiceHealthy
    Start-ScheduledTask -TaskName $KioskTaskName
  } catch {
    $SwitchFailure = $_
    try {
      Restore-FieldConfiguration -OriginalJson $OriginalJson
    } catch {
      throw "Field replay switch failed ($($SwitchFailure.Exception.Message)) and rollback failed ($($_.Exception.Message))"
    }
    throw "Field replay switch failed and the previous artifact was restored: $($SwitchFailure.Exception.Message)"
  }
  [pscustomobject]@{ status = "PASS"; artifact = $VerifiedArtifact.artifact_root; shot_count = [int]$VerifiedArtifact.summary.shot_count; current_release = [string]$Configuration.current_release } | ConvertTo-Json -Compress
  exit 0
}

if (-not $ReleaseArchive -or -not $ExpectedSha256) { throw "ReleaseArchive and ExpectedSha256 are required" }
if ($ExpectedSha256 -notmatch '^[0-9a-fA-F]{64}$') { throw "ExpectedSha256 must be 64 hexadecimal characters" }
$ResolvedArchive = (Resolve-Path -LiteralPath $ReleaseArchive).Path
$ActualSha256 = (Get-FileHash -LiteralPath $ResolvedArchive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualSha256 -ne $ExpectedSha256.ToLowerInvariant()) { throw "Release SHA256 mismatch: $ActualSha256" }
# Code upgrades must not replace the input selected by SwitchArtifact/refresh.
# An invalid current selection needs an explicit repair, never a silent fallback.
$ExistingConfiguration = $null
$OriginalJoinQrPath = ""
$OriginalJoinQrSha256 = ""
if (Test-Path -LiteralPath $CurrentPointer -PathType Leaf) {
  $ExistingConfiguration = Get-Content -LiteralPath $CurrentPointer -Raw | ConvertFrom-Json
  if (($DeviceId -and [string]$ExistingConfiguration.device_id -ne $DeviceId) -or
      ($SyncBaseUrl -and [string]$ExistingConfiguration.sync_base_url -ne $SyncBaseUrl) -or
      ($CredentialTarget -and [string]$ExistingConfiguration.credential_target -ne $CredentialTarget)) {
    throw "Deployment identity differs from current configuration; refusing to reuse its replay"
  }
  $Artifact = [string]$ExistingConfiguration.artifact
  if (-not $JoinQrAssetExplicit) {
    $JoinQrAsset = [string]$ExistingConfiguration.join_qr_asset
  }
  $SyncBaseUrl = [string]$ExistingConfiguration.sync_base_url
  $DeviceId = [string]$ExistingConfiguration.device_id
  $CredentialTarget = [string]$ExistingConfiguration.credential_target
  $BrowserPath = [string]$ExistingConfiguration.browser_path
  $PythonPath = [string]$ExistingConfiguration.python_path
  $OriginalJoinQrPath = (Resolve-Path -LiteralPath $JoinQrAsset).Path
  $OriginalJoinQrSha256 = (Get-FileHash -LiteralPath $OriginalJoinQrPath -Algorithm SHA256).Hash.ToLowerInvariant()
}
if (-not $Artifact -or -not (Test-Path -LiteralPath $Artifact -PathType Container)) { throw "Required replay artifact is missing: $Artifact" }
$SelectedCaptureProfile = $null
$SelectedCaptureProfilePath = ""
if ($CaptureProfile) {
  $SelectedCaptureProfilePath = (Resolve-Path -LiteralPath $CaptureProfile).Path
  $SelectedCaptureProfile = Read-CaptureProfile -Path $CaptureProfile
} elseif ($null -ne $ExistingConfiguration -and $null -ne $ExistingConfiguration.PSObject.Properties["capture_profile"]) {
  $SelectedCaptureProfile = $ExistingConfiguration.capture_profile
  if ($null -ne $ExistingConfiguration.PSObject.Properties["capture_profile_path"]) {
    $SelectedCaptureProfilePath = [string]$ExistingConfiguration.capture_profile_path
  }
}
$SelectedReplay = Assert-FieldReplayArtifact -ArtifactRoot $Artifact -RequiredShotCount $ExpectedShotCount
$Artifact = $SelectedReplay.artifact_root
foreach ($RequiredPath in @($JoinQrAsset, $BrowserPath)) {
  if (-not $RequiredPath -or -not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) { throw "Required deployment file is missing: $RequiredPath" }
}
$SyncValues = @($SyncBaseUrl, $DeviceId, $CredentialTarget) | Where-Object { $_ }
if (($SyncValues.Count -ne 0) -and ($SyncValues.Count -ne 3)) { throw "SyncBaseUrl, DeviceId, and CredentialTarget must be configured together" }

Add-Type -AssemblyName System.IO.Compression.FileSystem
$ArchiveHandle = [System.IO.Compression.ZipFile]::OpenRead($ResolvedArchive)
try {
  $ManifestEntry = $ArchiveHandle.Entries | Where-Object { $_.FullName.Replace('\', '/') -eq "release-manifest.json" } | Select-Object -First 1
  if ($null -eq $ManifestEntry) { throw "release-manifest.json is missing from release archive root" }
} finally {
  $ArchiveHandle.Dispose()
}
if ($CheckOnly) {
  [pscustomobject]@{ status = "READY"; sha256 = $ActualSha256; runtime_root = $ResolvedRuntimeRoot; releases_root = $ResolvedReleasesRoot; credential_target = $CredentialTarget; artifact = $Artifact; shot_count = [int]$SelectedReplay.summary.shot_count; capture_mode = if ($null -eq $SelectedCaptureProfile) { "replay" } else { [string]$SelectedCaptureProfile.mode } } | ConvertTo-Json -Compress
  exit 0
}

New-Item -ItemType Directory -Path $ResolvedRuntimeRoot -Force | Out-Null
New-Item -ItemType Directory -Path $ResolvedReleasesRoot -Force | Out-Null
$StagingRoot = Join-Path $ResolvedReleasesRoot (".staging-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $StagingRoot -Force | Out-Null
try {
  Expand-Archive -LiteralPath $ResolvedArchive -DestinationPath $StagingRoot
  $Manifest = Read-ReleaseManifest -ReleaseRoot $StagingRoot
  $ReleaseRoot = Assert-UnderRoot -Path (Join-Path $ResolvedReleasesRoot ([string]$Manifest.version)) -Root $ResolvedReleasesRoot
  if (Test-Path -LiteralPath $ReleaseRoot) { throw "Immutable release already exists: $ReleaseRoot" }
  Move-Item -LiteralPath $StagingRoot -Destination $ReleaseRoot
  $StagingRoot = ""
  $PreviousRelease = ""
  $OriginalJson = ""
  [byte[]]$OriginalPointerBytes = @()
  $StateDbBackup = ""
  if (Test-Path -LiteralPath $CurrentPointer -PathType Leaf) {
    $OriginalJson = Get-Content -LiteralPath $CurrentPointer -Raw
    $OriginalPointerBytes = [System.IO.File]::ReadAllBytes($CurrentPointer)
    $PreviousRelease = [string]($OriginalJson | ConvertFrom-Json).current_release
  }
  $TaskDefinitionSnapshot = Export-FieldTaskDefinitions
  $Configuration = [ordered]@{
    schema_version = "1.0"
    version = [string]$Manifest.version
    current_release = $ReleaseRoot
    previous_release = $PreviousRelease
    artifact = (Resolve-Path -LiteralPath $Artifact).Path
    join_qr_asset = (Resolve-Path -LiteralPath $JoinQrAsset).Path
    sync_base_url = $SyncBaseUrl
    device_id = $DeviceId
    credential_target = $CredentialTarget
    browser_path = (Resolve-Path -LiteralPath $BrowserPath).Path
    python_path = if ($PythonPath) { (Resolve-Path -LiteralPath $PythonPath).Path } else { "" }
    capture_profile = $null
    capture_profile_path = ""
  }
  if ($null -ne $SelectedCaptureProfile) {
    $Configuration.capture_profile = $SelectedCaptureProfile
    $Configuration.capture_profile_path = $SelectedCaptureProfilePath
  }
  try {
    Stop-FieldTasks
    if (Test-Path -LiteralPath $StateDbPath -PathType Leaf) {
      $SessionRetirement = Retire-ExistingFieldSession -ReleaseRoot $ReleaseRoot -StateDb $StateDbPath -Python $PythonPath -SyncBaseUrl $SyncBaseUrl -DeviceId $DeviceId -CredentialTarget $CredentialTarget
      $BackupRoot = Join-Path $ResolvedRuntimeRoot "backups"
      New-Item -ItemType Directory -Path $BackupRoot -Force | Out-Null
      $PendingStateDbBackup = Join-Path $BackupRoot ("field-state-before-" + [string]$Manifest.version + "-" + [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmssfff"))
      New-Item -ItemType Directory -Path $PendingStateDbBackup | Out-Null
      foreach ($StatePath in $StateDbCompanionPaths) {
        if (-not (Test-Path -LiteralPath $StatePath -PathType Leaf)) { continue }
        Copy-Item -LiteralPath $StatePath -Destination $PendingStateDbBackup
        $BackupPath = Join-Path $PendingStateDbBackup ([System.IO.Path]::GetFileName($StatePath))
        if ((Get-FileHash -LiteralPath $StatePath -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $BackupPath -Algorithm SHA256).Hash) {
          throw "Field state archive verification failed: $StatePath"
        }
      }
      $StateDbBackup = $PendingStateDbBackup
      foreach ($StatePath in $StateDbCompanionPaths) {
        if (Test-Path -LiteralPath $StatePath -PathType Leaf) {
          Remove-Item -LiteralPath $StatePath -Force
        }
      }
    }
    Write-CurrentConfigurationAtomically -Configuration ([pscustomobject]$Configuration)
    Register-FieldTasks -Configuration ([pscustomobject]$Configuration)
    Wait-FieldServiceHealthy -RequireUsbCapture:($null -ne $SelectedCaptureProfile)
    if ($OriginalJoinQrPath -and -not $JoinQrAssetExplicit) {
      $ActiveConfiguration = Get-Content -LiteralPath $CurrentPointer -Raw | ConvertFrom-Json
      $ActiveJoinQrPath = (Resolve-Path -LiteralPath ([string]$ActiveConfiguration.join_qr_asset)).Path
      $ActiveJoinQrSha256 = (Get-FileHash -LiteralPath $ActiveJoinQrPath -Algorithm SHA256).Hash.ToLowerInvariant()
      if ($ActiveJoinQrPath -ne $OriginalJoinQrPath -or $ActiveJoinQrSha256 -ne $OriginalJoinQrSha256) {
        throw "Permanent table QR changed during release activation"
      }
    }
    Start-ScheduledTask -TaskName $KioskTaskName
  } catch {
    $InstallFailure = $_
    if ($OriginalJson) {
      try {
        Restore-FieldReleaseConfiguration -OriginalJson $OriginalJson -OriginalPointerBytes $OriginalPointerBytes -StateDbBackup $StateDbBackup -TaskDefinitionSnapshot $TaskDefinitionSnapshot
      } catch {
        throw "Release install failed ($($InstallFailure.Exception.Message)) and rollback failed ($($_.Exception.Message))"
      }
      throw "Release install failed and the previous release was restored: $($InstallFailure.Exception.Message)"
    }
    throw
  }
  Write-Host "PASS: installed release $($Manifest.version); mutable data remains at $ResolvedRuntimeRoot; state backup=$StateDbBackup"
} finally {
  if ($StagingRoot -and (Test-Path -LiteralPath $StagingRoot)) { Remove-Item -LiteralPath $StagingRoot -Recurse -Force }
}
