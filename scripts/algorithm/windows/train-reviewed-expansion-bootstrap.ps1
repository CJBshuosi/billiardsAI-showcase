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
$sourceRoot = Join-Path $resolvedProject "src"
$runner = Join-Path $resolvedProject "scripts\algorithm\train_rfdetr_bootstrap_short.py"
$weight = Join-Path $resolvedProject "rf-detr-seg-s-ft.pth"
$datasetRoot = Join-Path $resolvedArtifacts "bootstrap-datasets"
$baseDatasetId = "2759358fe07f948a3c056a18f2b592e0b0ebad76364159f0a774c4eca70ca186"
foreach ($required in @($RfdetrPython, $sourceRoot, $runner, $weight, $datasetRoot)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required reviewed expansion training input is missing: $required"
    }
}

$dataset = Get-ChildItem -LiteralPath $datasetRoot -Directory | Where-Object { $_.Name -ne $baseDatasetId -and (Test-Path -LiteralPath (Join-Path $_.FullName "manifest.json") -PathType Leaf) } | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
if ($null -eq $dataset) {
    throw "No merged reviewed expansion bootstrap dataset was found. Run export first."
}

$env:PYTHONPATH = $sourceRoot
$runsRoot = Join-Path $resolvedArtifacts "bootstrap-training-runs"
& $RfdetrPython $runner --dataset $dataset.FullName --runs-root $runsRoot --weight $weight
if ($LASTEXITCODE -ne 0) {
    throw "Reviewed expansion RF-DETR short training stopped with exit $LASTEXITCODE."
}
