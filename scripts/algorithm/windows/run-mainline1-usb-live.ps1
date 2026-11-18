[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$ProjectRoot,
    [Parameter(Mandatory = $true)] [string]$RfdetrPython,
    [Parameter(Mandatory = $true)] [string]$DeviceLocator,
    [Parameter(Mandatory = $true)] [int]$RuntimeIndex,
    [Parameter(Mandatory = $true)] [string]$CaptureSetup,
    [Parameter(Mandatory = $true)] [string]$DetectorTrainingRun,
    [Parameter(Mandatory = $true)] [string]$IdentityTrainingRun,
    [Parameter(Mandatory = $true)] [string]$Dinov2Assets,
    [Parameter(Mandatory = $true)] [string]$ShotLocalConfig,
    [string]$CalibrationRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\algorithm-artifacts\mainline1-calibrations"),
    [string]$ArtifactRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\algorithm-artifacts\mainline1-usb-live-runs"),
    [string]$SpoolRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\capture-spool"),
    [ValidateRange(1, 64)] [int]$BatchSize = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) { $PSNativeCommandUseErrorActionPreference = $false }

if ($RuntimeIndex -ne 0) {
    throw "Trusted USB binding requires RuntimeIndex 0 and exactly one present camera. Disconnect every other camera before starting Mainline 1."
}

$resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
$runner = Join-Path $resolvedProject "scripts\algorithm\run_mainline1_usb_live.py"
$sourceRoot = Join-Path $resolvedProject "src"
$resolvedSetup = [IO.Path]::GetFullPath($CaptureSetup)
$resolvedCalibrationRoot = [IO.Path]::GetFullPath($CalibrationRoot)
$resolvedArtifactRoot = [IO.Path]::GetFullPath($ArtifactRoot)
$resolvedSpoolRoot = [IO.Path]::GetFullPath($SpoolRoot)
foreach ($required in @($RfdetrPython, $runner, $sourceRoot, $resolvedSetup, $resolvedCalibrationRoot, $DetectorTrainingRun, $IdentityTrainingRun, $Dinov2Assets, $ShotLocalConfig)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required USB Mainline 1 input is missing: $required" }
}

New-Item -ItemType Directory -Force -Path $resolvedArtifactRoot, $resolvedSpoolRoot | Out-Null
$env:PYTHONPATH = $sourceRoot
$arguments = @($runner, "--device-locator", $DeviceLocator, "--runtime-index", $RuntimeIndex, "--capture-setup", $resolvedSetup, "--calibration-root", $resolvedCalibrationRoot, "--detector-training-run", $DetectorTrainingRun, "--identity-training-run", $IdentityTrainingRun, "--dinov2-assets", $Dinov2Assets, "--config", $ShotLocalConfig, "--output-root", $resolvedArtifactRoot, "--spool-root", $resolvedSpoolRoot, "--device", "cuda", "--batch-size", $BatchSize)
& $RfdetrPython @arguments
if ($LASTEXITCODE -ne 0) { throw "USB Mainline 1 runtime stopped with exit $LASTEXITCODE." }
