[CmdletBinding()]
param(
  [string]$BundleRoot = $PSScriptRoot,
  [string]$PythonPath = "",
  [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ResolvedBundleRoot = (Resolve-Path -LiteralPath $BundleRoot).Path

function Assert-UnderBundleRoot {
  param([Parameter(Mandatory = $true)][string]$RelativePath, [switch]$Directory)
  if ([System.IO.Path]::IsPathRooted($RelativePath)) { throw "Receipt path must be relative: $RelativePath" }
  $FullPath = [System.IO.Path]::GetFullPath((Join-Path $ResolvedBundleRoot $RelativePath))
  $ApprovedPrefix = $ResolvedBundleRoot.TrimEnd('\') + '\'
  if (-not $FullPath.StartsWith($ApprovedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Receipt path escapes BundleRoot: $RelativePath"
  }
  $PathType = if ($Directory) { "Container" } else { "Leaf" }
  if (-not (Test-Path -LiteralPath $FullPath -PathType $PathType)) {
    throw "Required bundle asset is missing: $RelativePath"
  }
  return $FullPath
}

function Assert-Sha256 {
  param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Expected)
  if ($Expected -notmatch '^[0-9a-fA-F]{64}$') { throw "Receipt SHA256 is invalid" }
  $Actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($Actual -ne $Expected.ToLowerInvariant()) { throw "Bundle SHA256 mismatch: $Path" }
}

function Resolve-EdgePath {
  $Candidates = @(
    (Join-Path ${env:ProgramFiles(x86)} "Microsoft\Edge\Application\msedge.exe"),
    (Join-Path $env:ProgramFiles "Microsoft\Edge\Application\msedge.exe")
  )
  foreach ($Candidate in $Candidates) {
    if ($Candidate -and (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
      return (Resolve-Path -LiteralPath $Candidate).Path
    }
  }
  throw "Microsoft Edge msedge.exe was not found"
}

function Resolve-PythonPath {
  param([string]$Override)
  $Candidate = if ($Override) {
    $Override
  } else {
    Join-Path $env:USERPROFILE "Documents\billiardsAI-raytheon3\.venv-rfdetr\Scripts\python.exe"
  }
  if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
    throw "Raytheon .venv-rfdetr Python was not found: $Candidate"
  }
  return (Resolve-Path -LiteralPath $Candidate).Path
}

$ReceiptPath = Join-Path $ResolvedBundleRoot "deployment-receipt.json"
if (-not (Test-Path -LiteralPath $ReceiptPath -PathType Leaf)) {
  throw "deployment-receipt.json is missing"
}
$Receipt = Get-Content -LiteralPath $ReceiptPath -Raw | ConvertFrom-Json
if ($Receipt.schema_version -ne "1.0") { throw "Unsupported deployment receipt schema" }

$Gateway = $null
if (-not [Uri]::TryCreate([string]$Receipt.sync_base_url, [UriKind]::Absolute, [ref]$Gateway) -or $Gateway.Scheme -ne "https") {
  throw "Sync gateway must use HTTPS"
}
$ReleaseArchive = Assert-UnderBundleRoot -RelativePath ([string]$Receipt.release.path)
Assert-Sha256 -Path $ReleaseArchive -Expected ([string]$Receipt.release.sha256)
$ReplayRoot = Assert-UnderBundleRoot -RelativePath ([string]$Receipt.replay.path) -Directory
$ReplayManifestPath = Join-Path $ReplayRoot "manifest.json"
Assert-Sha256 -Path $ReplayManifestPath -Expected ([string]$Receipt.replay.manifest_sha256)
$ReplayManifest = Get-Content -LiteralPath $ReplayManifestPath -Raw | ConvertFrom-Json
if ($ReplayManifest.kind -ne "field_route_file_replay") { throw "Replay kind is invalid" }
if ($Receipt.replay.mode -ne "development_replay") { throw "Replay mode is invalid" }
foreach ($Property in $ReplayManifest.inventory.psobject.Properties) {
  $Name = [string]$Property.Name
  if ([System.IO.Path]::GetFileName($Name) -ne $Name) { throw "Replay inventory path is invalid: $Name" }
  $Asset = Join-Path $ReplayRoot $Name
  if (-not (Test-Path -LiteralPath $Asset -PathType Leaf)) { throw "Replay asset is missing: $Name" }
  if ((Get-Item -LiteralPath $Asset).Length -ne [long]$Property.Value.bytes) { throw "Replay size mismatch: $Name" }
  Assert-Sha256 -Path $Asset -Expected ([string]$Property.Value.sha256)
}
$JoinQrAsset = Assert-UnderBundleRoot -RelativePath ([string]$Receipt.table_join_qr.path)
Assert-Sha256 -Path $JoinQrAsset -Expected ([string]$Receipt.table_join_qr.sha256)
$QrBytes = [System.IO.File]::ReadAllBytes($JoinQrAsset)
$IsPng = $QrBytes.Length -ge 8 -and $QrBytes[0] -eq 0x89 -and $QrBytes[1] -eq 0x50 -and $QrBytes[2] -eq 0x4E -and $QrBytes[3] -eq 0x47
$IsJpeg = $QrBytes.Length -ge 2 -and $QrBytes[0] -eq 0xFF -and $QrBytes[1] -eq 0xD8
if (-not $IsPng -and -not $IsJpeg) { throw "Table join QR image is invalid" }

$CredentialTarget = [string]$Receipt.credential_target
if (-not $CredentialTarget) { throw "Credential target is missing" }
$CredentialListing = (& cmdkey.exe "/list:$CredentialTarget" | Out-String)
if ($LASTEXITCODE -ne 0 -or $CredentialListing -notmatch [regex]::Escape($CredentialTarget)) {
  throw "Windows Credential Manager target is missing: $CredentialTarget"
}
$BrowserPath = Resolve-EdgePath
$ResolvedPythonPath = Resolve-PythonPath -Override $PythonPath

Add-Type -AssemblyName System.IO.Compression.FileSystem
$ArchiveHandle = [System.IO.Compression.ZipFile]::OpenRead($ReleaseArchive)
$TemporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("billiardsAI-install-" + [Guid]::NewGuid().ToString("N"))
try {
  $InstallerEntry = $ArchiveHandle.Entries | Where-Object { $_.FullName.Replace('\', '/') -eq "scripts/algorithm/windows/install-field-route-service.ps1" } | Select-Object -First 1
  if ($null -eq $InstallerEntry) { throw "install-field-route-service.ps1 is missing from release archive" }
  New-Item -ItemType Directory -Path $TemporaryRoot | Out-Null
  $InstallerPath = Join-Path $TemporaryRoot "install-field-route-service.ps1"
  [System.IO.Compression.ZipFileExtensions]::ExtractToFile($InstallerEntry, $InstallerPath)
} finally {
  $ArchiveHandle.Dispose()
}

$InstallerArguments = @(
  "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $InstallerPath,
  "-ReleaseArchive", $ReleaseArchive,
  "-ExpectedSha256", ([string]$Receipt.release.sha256),
  "-Artifact", $ReplayRoot,
  "-JoinQrAsset", $JoinQrAsset,
  "-SyncBaseUrl", ([string]$Receipt.sync_base_url),
  "-DeviceId", ([string]$Receipt.device_id),
  "-CredentialTarget", $CredentialTarget,
  "-BrowserPath", $BrowserPath,
  "-PythonPath", $ResolvedPythonPath
)
if ($null -ne $Receipt.replay.PSObject.Properties["expected_shot_count"] -and $null -ne $Receipt.replay.expected_shot_count) {
  $InstallerArguments += @("-ExpectedShotCount", [string]$Receipt.replay.expected_shot_count)
}
try {
  & powershell.exe @InstallerArguments -CheckOnly
  if ($LASTEXITCODE -ne 0) { throw "Field installer preflight failed with exit $LASTEXITCODE" }
  if ($CheckOnly) {
    [pscustomobject]@{ status = "READY"; bundle_root = $ResolvedBundleRoot; device_id = [string]$Receipt.device_id } | ConvertTo-Json -Compress
    exit 0
  }
  & powershell.exe @InstallerArguments
  if ($LASTEXITCODE -ne 0) { throw "Field installer failed with exit $LASTEXITCODE" }
} finally {
  if (Test-Path -LiteralPath $TemporaryRoot) { Remove-Item -LiteralPath $TemporaryRoot -Recurse -Force }
}
