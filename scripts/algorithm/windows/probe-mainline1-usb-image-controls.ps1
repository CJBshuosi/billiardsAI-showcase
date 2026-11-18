[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$ProjectRoot,
    [Parameter(Mandatory = $true)] [string]$PythonPath,
    [Parameter(Mandatory = $true)] [string]$DeviceLocator,
    [Parameter(Mandatory = $true)] [int]$RuntimeIndex,
    [Parameter(Mandatory = $true)] [string]$OutputDir,
    [ValidateRange(1, 120000)] [int]$WarmupFrames = 60,
    [string[]]$Control = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
$runner = Join-Path $resolvedProject "scripts\algorithm\probe_mainline1_usb_image_controls.py"
$sourceRoot = Join-Path $resolvedProject "src"
foreach ($required in @($PythonPath, $runner, $sourceRoot)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required USB image-control probe input is missing: $required"
    }
}

$env:PYTHONPATH = $sourceRoot
$arguments = @($runner, "--device-locator", $DeviceLocator, "--runtime-index", $RuntimeIndex, "--output-dir", $OutputDir, "--width", "1920", "--height", "1080", "--fps", "60", "--warmup-frames", $WarmupFrames)
foreach ($setting in $Control) {
    $arguments += @("--control", $setting)
}
& $PythonPath @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Mainline 1 USB image-control probe stopped with exit $LASTEXITCODE."
}
