[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$ProjectRoot,
    [Parameter(Mandatory = $true)] [string]$SourceArtifact,
    [Parameter(Mandatory = $true)] [string]$UserId,
    [Parameter(Mandatory = $true)] [string]$SessionId,
    [string]$PythonPath = "",
    [string]$ArtifactRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\algorithm-artifacts")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$resolvedProject = (Resolve-Path -LiteralPath $ProjectRoot).Path
$resolvedSource = (Resolve-Path -LiteralPath $SourceArtifact).Path
$python = if ($PythonPath) { (Resolve-Path -LiteralPath $PythonPath).Path } else { Join-Path $resolvedProject ".venv-rfdetr\Scripts\python.exe" }
$runner = Join-Path $resolvedProject "scripts\algorithm\replay_lightweight_realtime_chain.py"
$sourceRoot = Join-Path $resolvedProject "src"
$outputRoot = Join-Path ([IO.Path]::GetFullPath($ArtifactRoot)) "lightweight-realtime-chain-runs"
foreach ($requiredFile in @($python, $runner)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required lightweight realtime-chain file is missing: $requiredFile"
    }
}
foreach ($requiredDirectory in @($resolvedSource, $sourceRoot)) {
    if (-not (Test-Path -LiteralPath $requiredDirectory -PathType Container)) {
        throw "Required lightweight realtime-chain directory is missing: $requiredDirectory"
    }
}
if (-not $UserId.Trim() -or -not $SessionId.Trim()) {
    throw "UserId and SessionId must not be empty."
}

$env:PYTHONPATH = $sourceRoot
& $python $runner --source-artifact $resolvedSource --output-root $outputRoot --user-id $UserId --session-id $SessionId
if ($LASTEXITCODE -ne 0) {
    throw "Lightweight realtime-chain replay stopped with exit $LASTEXITCODE."
}

