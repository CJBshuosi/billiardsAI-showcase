[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$ProjectRoot,
    [Parameter(Mandatory = $true)] [string]$PythonPath,
    [Parameter(Mandatory = $true)] [string]$DeviceLocator,
    [Parameter(Mandatory = $true)] [int]$RuntimeIndex,
    [Parameter(Mandatory = $true)] [string]$Output,
    [ValidateRange(8, 100)] [int]$TargetSamples = 12,
    [ValidateRange(2, 30)] [int]$CheckerboardColumns = 9,
    [ValidateRange(2, 30)] [int]$CheckerboardRows = 6,
    [switch]$RerProfile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
$runner = Join-Path $resolvedProject "scripts\algorithm\capture_mainline1_usb_lens_samples.py"
$sourceRoot = Join-Path $resolvedProject "src"
foreach ($required in @($PythonPath, $runner, $sourceRoot)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required USB lens sample capture input is missing: $required"
    }
}

$env:PYTHONPATH = $sourceRoot
$arguments = @(
    $runner,
    "--device-locator", $DeviceLocator,
    "--runtime-index", $RuntimeIndex,
    "--output", $Output,
    "--target-samples", $TargetSamples,
    "--checkerboard-columns", $CheckerboardColumns,
    "--checkerboard-rows", $CheckerboardRows
)
if ($RerProfile) { $arguments += "--rer-profile" }
& $PythonPath @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Mainline 1 USB lens sample capture stopped with exit $LASTEXITCODE."
}
