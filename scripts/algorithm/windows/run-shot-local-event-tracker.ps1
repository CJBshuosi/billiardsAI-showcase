[CmdletBinding()]
# Compatibility wrapper only. New product runs must use run-mainline1-shot-local-video.ps1.
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
$runner = Join-Path $resolvedProject "scripts\algorithm\run_shot_local_event_tracker.py"
$outputRoot = Join-Path ([IO.Path]::GetFullPath($ArtifactRoot)) "mainline1-shot-local-video-runs"
foreach ($required in @($Python, $runner, $Video, $SourceArtifact, $ShotLocalConfig)) { if (-not (Test-Path -LiteralPath $required)) { throw "Required unified Mainline 1 cache input is missing: $required" } }
& $Python $runner --video $Video --source-artifact $SourceArtifact --shot-local-config $ShotLocalConfig --output-root $outputRoot
if ($LASTEXITCODE -ne 0) { throw "Unified Mainline 1 cache replay stopped with exit $LASTEXITCODE." }
