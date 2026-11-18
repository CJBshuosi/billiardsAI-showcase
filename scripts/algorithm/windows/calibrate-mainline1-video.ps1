[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$ProjectRoot,
    [Parameter(Mandatory = $true)] [string]$RfdetrPython,
    [Parameter(Mandatory = $true)] [string]$Video,
    [Parameter(Mandatory = $true)] [string]$CaptureSetup,
    [string]$DeviceId = "",
    [ValidateRange(1, 65535)] [int]$Port = 8767,
    [string]$ArtifactRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\algorithm-artifacts")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
$runner = Join-Path $resolvedProject "scripts\algorithm\calibrate_mainline1_video.py"
$sourceRoot = Join-Path $resolvedProject "src"
$outputRoot = Join-Path ([IO.Path]::GetFullPath($ArtifactRoot)) "mainline1-calibrations"
$resolvedVideo = [IO.Path]::GetFullPath($Video)
$resolvedSetup = [IO.Path]::GetFullPath($CaptureSetup)
foreach ($required in @($RfdetrPython, $runner, $resolvedVideo, $resolvedSetup)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required Mainline 1 calibration input is missing: $required"
    }
}
if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
    throw "Required Mainline 1 calibration source root is missing: $sourceRoot"
}

$env:PYTHONPATH = $sourceRoot
$RunnerArguments = @($runner, "--video", $resolvedVideo, "--capture-setup", $resolvedSetup, "--port", $Port, "--output-root", $outputRoot)
if ($DeviceId -ne "") { $RunnerArguments += @("--device-id", $DeviceId) }
& $RfdetrPython @RunnerArguments
if ($LASTEXITCODE -ne 0) {
    throw "Mainline 1 calibration stopped with exit $LASTEXITCODE."
}
