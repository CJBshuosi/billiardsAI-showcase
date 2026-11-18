[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [Parameter(Mandatory = $true)][string]$RfdetrPython,
    [Parameter(Mandatory = $true)][string]$CandidateArtifact,
    [Parameter(Mandatory = $true)][string]$VerdictArtifact,
    [string]$CorrectionVerdictArtifact,
    [Parameter(Mandatory = $true)][string]$TrainingRun,
    [string]$AssetRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\approved-assets\dinov2-small-ed25f3a"),
    [string]$ArtifactRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\algorithm-artifacts"),
    [ValidateRange(1024, 65535)][int]$Port = 8767,
    [ValidateRange(1, 64)][int]$BatchSize = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
$runner = Join-Path $resolvedProject "scripts\algorithm\review_identity_group_errors.py"
$sourceRoot = Join-Path $resolvedProject "src"
$outputRoot = Join-Path ([IO.Path]::GetFullPath($ArtifactRoot)) "identity-group-error-review"
$requiredInputs = @($RfdetrPython, $runner, $sourceRoot, $CandidateArtifact, $VerdictArtifact, $TrainingRun, $AssetRoot)
if ($CorrectionVerdictArtifact) {
    $requiredInputs += $CorrectionVerdictArtifact
}
foreach ($required in $requiredInputs) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required group identity error-review input is missing: $required"
    }
}

$env:PYTHONPATH = $sourceRoot
$reviewArguments = @(
    $runner,
    "--candidate-artifact", $CandidateArtifact,
    "--verdict-artifact", $VerdictArtifact,
    "--training-run", $TrainingRun,
    "--asset-root", $AssetRoot,
    "--output-root", $outputRoot,
    "--device", "cuda",
    "--batch-size", $BatchSize,
    "--port", $Port,
    "--open-browser"
)
if ($CorrectionVerdictArtifact) {
    $reviewArguments += @("--correction-verdict-artifact", $CorrectionVerdictArtifact)
}
& $RfdetrPython @reviewArguments
if ($LASTEXITCODE -ne 0) {
    throw "Group identity error review stopped with exit $LASTEXITCODE."
}
