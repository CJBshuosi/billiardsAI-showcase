[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$ProjectRoot,
    [Parameter(Mandatory = $true)] [string]$PythonPath,
    [Parameter(Mandatory = $true)] [string]$DeviceLocator,
    [Parameter(Mandatory = $true)] [int]$RuntimeIndex,
    [ValidateRange(1, 120000)] [int]$SampleFrames = 120
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
$runner = Join-Path $resolvedProject "scripts\algorithm\probe_mainline1_usb.py"
$sourceRoot = Join-Path $resolvedProject "src"
foreach ($required in @($PythonPath, $runner, $sourceRoot)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required USB probe input is missing: $required"
    }
}

$env:PYTHONPATH = $sourceRoot
$arguments = @($runner, "--device-locator", $DeviceLocator, "--runtime-index", $RuntimeIndex, "--width", "1920", "--height", "1080", "--fps", "60", "--sample-frames", $SampleFrames)
& $PythonPath @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Mainline 1 USB quality probe stopped with exit $LASTEXITCODE."
}
