[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$ProjectRoot,
    [Parameter(Mandatory = $true)] [string]$Python,
    [Parameter(Mandatory = $true)] [string]$Video,
    [Parameter(Mandatory = $true)] [string]$SourceArtifact,
    [Parameter(Mandatory = $true)] [string]$ShotLocalConfig,
    [string]$ArtifactRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\algorithm-artifacts")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
$env:PYTHONPATH = Join-Path $resolvedProject "src"
$runner = Join-Path $resolvedProject "scripts\algorithm\reobserve_mainline1_shot_local_artifact.py"
$outputRoot = Join-Path ([IO.Path]::GetFullPath($ArtifactRoot)) "mainline1-shot-local-video-runs"
foreach ($required in @($Python, $runner, $Video, $SourceArtifact, $ShotLocalConfig)) { if (-not (Test-Path -LiteralPath $required)) { throw "Required cached re-observation input is missing: $required" } }
& $Python $runner --video $Video --source-artifact $SourceArtifact --config $ShotLocalConfig --output-root $outputRoot
if ($LASTEXITCODE -ne 0) { throw "Shot-local cached re-observation stopped with exit $LASTEXITCODE." }
