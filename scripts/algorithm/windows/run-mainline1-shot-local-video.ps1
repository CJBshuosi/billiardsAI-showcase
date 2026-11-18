[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$ProjectRoot,
    [Parameter(Mandatory = $true)] [string]$RfdetrPython,
    [Parameter(Mandatory = $true)] [string]$Video,
    [Parameter(Mandatory = $true)] [string]$CaptureSetup,
    [Parameter(Mandatory = $true)] [string]$DetectorTrainingRun,
    [Parameter(Mandatory = $true)] [string]$IdentityTrainingRun,
    [Parameter(Mandatory = $true)] [string]$Dinov2Assets,
    [Parameter(Mandatory = $true)] [string]$ShotLocalConfig,
    [string]$Calibration,
    [string]$CalibrationRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\algorithm-artifacts\mainline1-calibrations"),
    [string]$ArtifactRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\algorithm-artifacts"),
    [ValidateRange(1, 64)] [int]$BatchSize = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
$runner = Join-Path $resolvedProject "scripts\algorithm\run_mainline1_shot_local_video.py"
$sourceRoot = Join-Path $resolvedProject "src"
$outputRoot = Join-Path ([IO.Path]::GetFullPath($ArtifactRoot)) "mainline1-shot-local-video-runs"
$resolvedVideo = [IO.Path]::GetFullPath($Video)
$resolvedSetup = [IO.Path]::GetFullPath($CaptureSetup)
$resolvedCalibrationRoot = [IO.Path]::GetFullPath($CalibrationRoot)
foreach ($required in @($RfdetrPython, $runner, $sourceRoot, $resolvedVideo, $resolvedSetup, $resolvedCalibrationRoot, $DetectorTrainingRun, $IdentityTrainingRun, $Dinov2Assets, $ShotLocalConfig)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required unified Mainline 1 input is missing: $required"
    }
}

$env:PYTHONPATH = $sourceRoot
$arguments = @($runner, "--video", $resolvedVideo, "--capture-setup", $resolvedSetup, "--calibration-root", $resolvedCalibrationRoot, "--detector-training-run", $DetectorTrainingRun, "--identity-training-run", $IdentityTrainingRun, "--dinov2-assets", $Dinov2Assets, "--config", $ShotLocalConfig, "--output-root", $outputRoot, "--device", "cuda", "--batch-size", $BatchSize)
if ($Calibration) {
    $resolvedCalibration = [IO.Path]::GetFullPath($Calibration)
    if (-not (Test-Path -LiteralPath $resolvedCalibration -PathType Leaf)) { throw "Explicit calibration is missing: $resolvedCalibration" }
    $arguments += @("--calibration", $resolvedCalibration)
}
& $RfdetrPython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Unified Mainline 1 video analysis stopped with exit $LASTEXITCODE."
}
