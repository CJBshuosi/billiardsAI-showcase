[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [Parameter(Mandatory = $true)]
    [string]$RfdetrPython,

    [string]$ArtifactRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\algorithm-artifacts")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
$resolvedArtifacts = [IO.Path]::GetFullPath($ArtifactRoot)
$runner = Join-Path $resolvedProject "scripts\algorithm\export_reviewed_expansion_bootstrap.py"
$sourceRoot = Join-Path $resolvedProject "src"
foreach ($required in @($RfdetrPython, $runner, $sourceRoot)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required reviewed expansion export input is missing: $required"
    }
}

$env:PYTHONPATH = $sourceRoot
& $RfdetrPython $runner --project-root $resolvedProject --artifact-root $resolvedArtifacts --frame-stride 15 --split-seed 20260823
if ($LASTEXITCODE -ne 0) {
    throw "Reviewed expansion bootstrap export stopped with exit $LASTEXITCODE."
}
