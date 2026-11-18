[CmdletBinding()]
param(
  [string]$ProjectRoot = "C:\Users\developer\Documents\billiardsAI-raytheon3",
  [string]$Video = "",
  [string]$Mainline1Artifact = "",
  [string]$Mainline1Root = (Join-Path $env:LOCALAPPDATA "billiardsAI\algorithm-artifacts\mainline1-shot-local-video-runs"),
  [string]$OutputRoot = (Join-Path $env:LOCALAPPDATA "billiardsAI\algorithm-artifacts\field-route-replay-runs"),
  [string]$RuntimeRoot = (Join-Path $env:LOCALAPPDATA "billiardsAI\field-runtime"),
  [ValidateSet("open", "solid", "stripe")][string]$ContextGroup = "open",
  [ValidateRange(0, 2147483647)][int]$ExpectedShotCount = 1,
  [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-UnderRoot {
  param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Root)
  $FullPath = [System.IO.Path]::GetFullPath($Path)
  $FullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
  if (-not $FullPath.StartsWith($FullRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Path escapes the approved root: $FullPath"
  }
  return $FullPath
}

$CurrentPointer = Join-Path ([System.IO.Path]::GetFullPath($RuntimeRoot)) "current.json"
if (-not (Test-Path -LiteralPath $CurrentPointer -PathType Leaf)) { throw "Field runtime current.json is missing" }
$Configuration = Get-Content -LiteralPath $CurrentPointer -Raw | ConvertFrom-Json
$ReleaseRoot = [System.IO.Path]::GetFullPath([string]$Configuration.current_release)
$PythonPath = [System.IO.Path]::GetFullPath([string]$Configuration.python_path)
$ReplayScript = Join-Path $ReleaseRoot "scripts\algorithm\run_field_route_replay.py"
$Installer = Join-Path $ReleaseRoot "scripts\algorithm\windows\install-field-route-service.ps1"
foreach ($RequiredFile in @($PythonPath, $ReplayScript, $Installer)) {
  if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) { throw "Required replay runtime file is missing: $RequiredFile" }
}
if (-not $Video) { $Video = Join-Path $ProjectRoot "private-data\external-dev\det-sam2-evaluation\videos\video43.mp4" }
if (-not (Test-Path -LiteralPath $Video -PathType Leaf)) { throw "video43 is missing: $Video" }
$VideoSha256 = (Get-FileHash -LiteralPath $Video -Algorithm SHA256).Hash.ToLowerInvariant()
if (-not $Mainline1Artifact) {
  if (-not (Test-Path -LiteralPath $Mainline1Root -PathType Container)) { throw "Mainline 1 artifact root is missing: $Mainline1Root" }
  $Latest = Get-ChildItem -LiteralPath $Mainline1Root -Directory | Where-Object { if (-not ((Test-Path -LiteralPath (Join-Path $_.FullName "manifest.json") -PathType Leaf) -and (Test-Path -LiteralPath (Join-Path $_.FullName "shot-windows.json") -PathType Leaf) -and (Test-Path -LiteralPath (Join-Path $_.FullName "world-events.json") -PathType Leaf))) { return $false }; try { $CandidateManifest = Get-Content -LiteralPath (Join-Path $_.FullName "manifest.json") -Raw | ConvertFrom-Json; return ([string]$CandidateManifest.video_sha256).ToLowerInvariant() -eq $VideoSha256 } catch { return $false } } | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
  if ($null -eq $Latest) { throw "No complete Mainline 1 artifact was found" }
  $Mainline1Artifact = $Latest.FullName
}
if (-not (Test-Path -LiteralPath $Mainline1Artifact -PathType Container)) { throw "Mainline 1 artifact is missing: $Mainline1Artifact" }
$SourceManifest = Get-Content -LiteralPath (Join-Path $Mainline1Artifact "manifest.json") -Raw | ConvertFrom-Json
if (([string]$SourceManifest.video_sha256).ToLowerInvariant() -ne $VideoSha256) { throw "Mainline 1 artifact does not belong to video43" }
$ResolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

if ($CheckOnly) {
  [pscustomobject]@{ status = "READY"; video = (Resolve-Path -LiteralPath $Video).Path; mainline1_artifact = (Resolve-Path -LiteralPath $Mainline1Artifact).Path; current_release = $ReleaseRoot } | ConvertTo-Json -Compress
  exit 0
}

New-Item -ItemType Directory -Path $ResolvedOutputRoot -Force | Out-Null
$OldPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
try {
  $ReleaseSource = Join-Path $ReleaseRoot "src"
  $ProcessPythonPath = $ReleaseSource
  if ($OldPythonPath) { $ProcessPythonPath = $ReleaseSource + [System.IO.Path]::PathSeparator + $OldPythonPath }
  [Environment]::SetEnvironmentVariable("PYTHONPATH", $ProcessPythonPath, "Process")
  $ReplayOutput = @(& $PythonPath $ReplayScript "--video" $Video "--mainline1-artifact" $Mainline1Artifact "--context-group" $ContextGroup "--output-root" $ResolvedOutputRoot)
  $ReplayExitCode = $LASTEXITCODE
} finally {
  [Environment]::SetEnvironmentVariable("PYTHONPATH", $OldPythonPath, "Process")
}
if ($ReplayExitCode -ne 0) { throw "video43 replay generation failed: $($ReplayOutput -join [Environment]::NewLine)" }
$ResultLine = @($ReplayOutput | ForEach-Object { [string]$_ } | Where-Object { $_.TrimStart().StartsWith([char]123) }) | Select-Object -Last 1
if (-not $ResultLine) { throw "video43 replay did not return a result document" }
$Result = $ResultLine | ConvertFrom-Json
if ($Result.status -ne "PASS") { throw "video43 replay did not pass" }
$ArtifactRoot = Assert-UnderRoot -Path ([string]$Result.output) -Root $ResolvedOutputRoot
$Manifest = Get-Content -LiteralPath (Join-Path $ArtifactRoot "manifest.json") -Raw | ConvertFrom-Json
$Summary = Get-Content -LiteralPath (Join-Path $ArtifactRoot "summary.json") -Raw | ConvertFrom-Json
if ($Manifest.kind -ne "field_route_file_replay" -or $Manifest.pipeline_version -ne "field-route-file-replay/2") { throw "video43 replay is not an authoritative v2 artifact" }
if ([int]$Summary.shot_count -ne $ExpectedShotCount) { throw "video43 replay shot_count $($Summary.shot_count) does not equal expected $ExpectedShotCount" }

$TaskIdentity = "$env:USERDOMAIN\$env:USERNAME"
$OldPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
try {
  $ReleaseSource = Join-Path $ReleaseRoot "src"
  $ProcessPythonPath = $ReleaseSource
  if ($OldPythonPath) { $ProcessPythonPath = $ReleaseSource + [System.IO.Path]::PathSeparator + $OldPythonPath }
  [Environment]::SetEnvironmentVariable("PYTHONPATH", $ProcessPythonPath, "Process")
  $GrantOutput = @(& $PythonPath "-m" "billiards_ai.deployment.windows_acl" "--artifact" $ArtifactRoot "--identity" $TaskIdentity)
  $GrantExitCode = $LASTEXITCODE
} finally {
  [Environment]::SetEnvironmentVariable("PYTHONPATH", $OldPythonPath, "Process")
}
if ($GrantExitCode -ne 0) { throw "artifact permission grant failed: $($GrantOutput -join [Environment]::NewLine)" }

$InstallerArguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Installer, "-SwitchArtifact", "-Artifact", $ArtifactRoot, "-ExpectedShotCount", [string]$ExpectedShotCount, "-RuntimeRoot", $RuntimeRoot)
$SwitchOutput = @(& powershell.exe @InstallerArguments)
if ($LASTEXITCODE -ne 0) { throw "video43 replay was generated but not activated: $($SwitchOutput -join [Environment]::NewLine)" }
[pscustomobject]@{ status = "PASS"; artifact = $ArtifactRoot; shot_count = [int]$Summary.shot_count; guidance_generation_count = [int]$Summary.guidance_generation_count; source_mainline1_artifact = (Resolve-Path -LiteralPath $Mainline1Artifact).Path } | ConvertTo-Json -Compress
