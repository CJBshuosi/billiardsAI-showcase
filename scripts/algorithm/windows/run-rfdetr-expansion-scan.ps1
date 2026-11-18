param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,
    [Parameter(Mandatory = $true)]
    [string]$RfdetrPython,
    [string]$ArtifactRoot = "$env:LOCALAPPDATA\billiardsAI\algorithm-artifacts"
)

$ErrorActionPreference = "Stop"

$splitManifest = Join-Path $ProjectRoot "configs\algorithm\datasets\detsam2-training-split-v1.json"
$reviewRoot = Join-Path $ProjectRoot "private-data\algorithm-artifacts\detsam2-expansion-review-pack\run-3d46ea28f62173cc4726653281a35a941eb22926fe4649e8562d1354ebb409c1"
$runner = Join-Path $ProjectRoot "scripts\algorithm\run_rfdetr_expansion_scan.py"
$sourceRoot = Join-Path $ProjectRoot "src"
$outputRoot = Join-Path $ArtifactRoot "detsam2-expansion-rfdetr-scans"

foreach ($required in @($RfdetrPython, $splitManifest, $reviewRoot, $runner, $sourceRoot)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required expansion-scan input is missing: $required"
    }
}

$trainingManifests = Get-ChildItem -LiteralPath (Join-Path $ArtifactRoot "bootstrap-training-runs") -Recurse -File -Filter "training-manifest.json" | ForEach-Object {
    try {
        $payload = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
        if ($payload.candidate_id -eq "rfdetr-seg-small-bootstrap-short-r1") {
            [pscustomobject]@{ Path = $_.FullName; Directory = $_.DirectoryName; Modified = $_.LastWriteTimeUtc }
        }
    }
    catch {
    }
} | Sort-Object Modified -Descending

$trainingManifest = $trainingManifests | Select-Object -First 1
if ($null -eq $trainingManifest) {
    throw "Approved RF-DETR bootstrap training manifest was not found."
}

$checkpoint = Join-Path $trainingManifest.Directory "checkpoint_best.pth"
if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
    throw "checkpoint_best.pth is missing beside the approved training manifest."
}
$checkpointSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $checkpoint).Hash.ToLowerInvariant()

$env:PYTHONPATH = $sourceRoot
& $RfdetrPython $runner --split-manifest $splitManifest --review-root $reviewRoot --checkpoint $checkpoint --checkpoint-sha256 $checkpointSha256 --output-root $outputRoot --threshold 0.1 --low-confidence-below 0.6 --review-limit 200
if ($LASTEXITCODE -ne 0) {
    throw "RF-DETR expansion scan stopped with exit $LASTEXITCODE."
}
