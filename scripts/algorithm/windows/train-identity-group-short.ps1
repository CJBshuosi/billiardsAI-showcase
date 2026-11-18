[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [Parameter(Mandatory = $true)]
    [string]$RfdetrPython,

    [Parameter(Mandatory = $true)]
    [string]$CandidateArtifact,

    [Parameter(Mandatory = $true)]
    [string]$VerdictArtifact,

    [string]$CorrectionVerdictArtifact,

    [string]$AssetRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\approved-assets\dinov2-small-ed25f3a"),

    [string]$ArtifactRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\algorithm-artifacts"),

    [ValidateRange(1, 64)]
    [int]$BatchSize = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
$runner = Join-Path $resolvedProject "scripts\algorithm\train_identity_group_short.py"
$sourceRoot = Join-Path $resolvedProject "src"
$outputRoot = Join-Path ([IO.Path]::GetFullPath($ArtifactRoot)) "identity-group-training-runs"
foreach ($required in @($RfdetrPython, $runner, $sourceRoot, $CandidateArtifact, $VerdictArtifact, $AssetRoot)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required group identity training input is missing: $required"
    }
}
if ($CorrectionVerdictArtifact -and -not (Test-Path -LiteralPath $CorrectionVerdictArtifact)) {
    throw "Required group identity correction verdict is missing: $CorrectionVerdictArtifact"
}

$env:PYTHONPATH = $sourceRoot
$arguments = @($runner, "--candidate-artifact", $CandidateArtifact, "--verdict-artifact", $VerdictArtifact, "--asset-root", $AssetRoot, "--output-root", $outputRoot, "--device", "cuda", "--batch-size", $BatchSize)
if ($CorrectionVerdictArtifact) {
    $arguments += @("--correction-verdict-artifact", $CorrectionVerdictArtifact)
}
& $RfdetrPython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Group identity short training stopped with exit $LASTEXITCODE."
}
