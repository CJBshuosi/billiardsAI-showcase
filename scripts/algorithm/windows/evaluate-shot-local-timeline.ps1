[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$ProjectRoot,
    [Parameter(Mandatory = $true)] [string]$Python,
    [Parameter(Mandatory = $true)] [string]$Timeline,
    [Parameter(Mandatory = $true)] [string]$Truth,
    [Parameter(Mandatory = $true)] [string]$Output,
    [ValidateRange(0, 5000)] [int]$ToleranceMs = 120
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
$env:PYTHONPATH = Join-Path $resolvedProject "src"
$runner = Join-Path $resolvedProject "scripts\algorithm\evaluate_shot_local_timeline.py"
& $Python $runner --timeline $Timeline --truth $Truth --output $Output --tolerance-ms $ToleranceMs
if ($LASTEXITCODE -ne 0) { throw "Shot-local timeline evaluation stopped with exit $LASTEXITCODE." }
