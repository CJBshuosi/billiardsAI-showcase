[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [Parameter(Mandatory = $true)][string]$RfdetrPython,
    [Parameter(Mandatory = $true)][string]$CandidateArtifact,
    [Parameter(Mandatory = $true)][string]$BaseCorrectionVerdictArtifact,
    [Parameter(Mandatory = $true)][string]$OverlayCorrectionVerdictArtifact,
    [string]$ArtifactRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\algorithm-artifacts")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
$runner = Join-Path $resolvedProject "scripts\algorithm\merge_identity_correction_verdicts.py"
$sourceRoot = Join-Path $resolvedProject "src"
$outputRoot = Join-Path ([IO.Path]::GetFullPath($ArtifactRoot)) "identity-group-cumulative-corrections"
foreach ($required in @($RfdetrPython, $runner, $sourceRoot, $CandidateArtifact, $BaseCorrectionVerdictArtifact, $OverlayCorrectionVerdictArtifact)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required identity correction merge input is missing: $required"
    }
}

$env:PYTHONPATH = $sourceRoot
& $RfdetrPython $runner --candidate-artifact $CandidateArtifact --base-correction-verdict-artifact $BaseCorrectionVerdictArtifact --overlay-correction-verdict-artifact $OverlayCorrectionVerdictArtifact --output-root $outputRoot
if ($LASTEXITCODE -ne 0) {
    throw "Identity correction merge stopped with exit $LASTEXITCODE."
}
