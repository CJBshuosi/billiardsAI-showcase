[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$ProjectRoot,
    [Parameter(Mandatory = $true)] [string]$PythonPath,
    [Parameter(Mandatory = $true)] [string]$Samples,
    [Parameter(Mandatory = $true)] [string]$Output
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
$runner = Join-Path $resolvedProject "scripts\algorithm\calibrate_mainline1_usb_lens.py"
$sourceRoot = Join-Path $resolvedProject "src"
foreach ($required in @($PythonPath, $runner, $sourceRoot, $Samples)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required USB lens calibration input is missing: $required"
    }
}

$env:PYTHONPATH = $sourceRoot
$arguments = @($runner, "--samples", $Samples, "--output", $Output)
& $PythonPath @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Mainline 1 USB lens calibration stopped with exit $LASTEXITCODE."
}
