[CmdletBinding()]
param(
  [string]$BundleRoot = "",
  [string]$RuntimeRoot = (Join-Path $env:LOCALAPPDATA "billiardsAI\field-runtime")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if ([string]::IsNullOrWhiteSpace($BundleRoot)) {
  $BundleRoot = Split-Path -Parent $PSCommandPath
}
$Bundle = (Resolve-Path -LiteralPath $BundleRoot).Path
$Receipt = Get-Content -LiteralPath (Join-Path $Bundle "upgrade-receipt.json") -Raw | ConvertFrom-Json
if ([System.IO.Path]::GetFileName([string]$Receipt.release) -ne [string]$Receipt.release) { throw "Invalid upgrade archive path" }
$Archive = Join-Path $Bundle ([string]$Receipt.release)
if ((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne [string]$Receipt.sha256) { throw "Upgrade archive SHA256 mismatch" }
$Pointer = Join-Path $RuntimeRoot "current.json"
$Configuration = Get-Content -LiteralPath $Pointer -Raw | ConvertFrom-Json
$Manifest = Get-Content -LiteralPath (Join-Path $Configuration.artifact "manifest.json") -Raw | ConvertFrom-Json
$Summary = Get-Content -LiteralPath (Join-Path $Configuration.artifact "summary.json") -Raw | ConvertFrom-Json
if ($Manifest.pipeline_version -ne "field-route-file-replay/2" -or [int]$Summary.shot_count -ne 1) {
  # Reuse the installed, verified Mainline 1 adapter and its atomic artifact switch.
  $Refresh = Join-Path $Configuration.current_release "scripts\algorithm\windows\refresh-video43-field-replay.ps1"
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Refresh -RuntimeRoot $RuntimeRoot -ExpectedShotCount 1
  if ($LASTEXITCODE -ne 0) { throw "video43 refresh failed; code upgrade was not started" }
  $Configuration = Get-Content -LiteralPath $Pointer -Raw | ConvertFrom-Json
}
$Installer = Join-Path $Bundle "install-field-route-service.ps1"
$InstallArguments = @(
  "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Installer,
  "-ReleaseArchive", $Archive, "-ExpectedSha256", [string]$Receipt.sha256,
  "-Artifact", [string]$Configuration.artifact, "-ExpectedShotCount", 1,
  "-JoinQrAsset", [string]$Configuration.join_qr_asset,
  "-SyncBaseUrl", [string]$Configuration.sync_base_url,
  "-DeviceId", [string]$Configuration.device_id,
  "-CredentialTarget", [string]$Configuration.credential_target,
  "-BrowserPath", [string]$Configuration.browser_path,
  "-PythonPath", [string]$Configuration.python_path,
  "-RuntimeRoot", $RuntimeRoot
)
& powershell.exe @InstallArguments -CheckOnly
if ($LASTEXITCODE -ne 0) { throw "Upgrade preflight failed; current release was not replaced" }
& powershell.exe @InstallArguments
if ($LASTEXITCODE -ne 0) { throw "Code upgrade failed; keep the output for diagnosis" }
