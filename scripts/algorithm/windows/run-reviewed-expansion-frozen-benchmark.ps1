[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [Parameter(Mandatory = $true)]
    [string]$RfdetrPython,

    [Parameter(Mandatory = $true)]
    [string]$DatasetId,

    [string]$ArtifactRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\algorithm-artifacts"),

    [string]$DocumentsRoot = [Environment]::GetFolderPath("MyDocuments")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
$resolvedArtifacts = [IO.Path]::GetFullPath($ArtifactRoot)
$resolvedDocuments = [IO.Path]::GetFullPath($DocumentsRoot)
$sourceRoot = Join-Path $resolvedProject "src"
$runner = Join-Path $resolvedProject "scripts\algorithm\run_reviewed_expansion_frozen_benchmark.py"
foreach ($required in @($RfdetrPython, $sourceRoot, $runner, $resolvedArtifacts, $resolvedDocuments)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required reviewed expansion frozen benchmark input is missing: $required"
    }
}
if ($DatasetId -cnotmatch "^[0-9a-f]{64}$") {
    throw "DatasetId must be 64 lowercase hexadecimal characters."
}

$env:PYTHONPATH = $sourceRoot
$runnerArguments = @(
    "--project-root", $resolvedProject,
    "--artifact-root", $resolvedArtifacts,
    "--documents-root", $resolvedDocuments,
    "--dataset-id", $DatasetId
)
& $RfdetrPython $runner @runnerArguments
if ($LASTEXITCODE -ne 0) {
    throw "Reviewed expansion frozen benchmark stopped with exit $LASTEXITCODE."
}
