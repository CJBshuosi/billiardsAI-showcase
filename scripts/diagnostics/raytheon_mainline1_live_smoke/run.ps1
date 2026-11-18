[CmdletBinding()]
param(
    [string]$PythonPath = "C:\Users\developer\Documents\billiardsAI-raytheon3\.venv-rfdetr\Scripts\python.exe",
    [string]$OutputParent = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) { $PSNativeCommandUseErrorActionPreference = $false }

function Assert-FileHash {
    param([string]$Path, [string]$Expected)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Required file is missing: $Path" }
    $Actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Expected.ToLowerInvariant()) { throw "File hash mismatch: $Path" }
}

$PacketRoot = [IO.Path]::GetFullPath($PSScriptRoot)
if ([string]::IsNullOrWhiteSpace($OutputParent)) { $OutputParent = $PacketRoot }
$ResolvedOutputParent = [IO.Path]::GetFullPath($OutputParent)
$ManifestPath = Join-Path $PacketRoot "packet-manifest.json"
$ResultName = "mainline1-live-smoke-result-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + [Guid]::NewGuid().ToString("N").Substring(0,6)
$Result = Join-Path $ResolvedOutputParent $ResultName
New-Item -ItemType Directory -Path $Result | Out-Null
$ExitCode = 2

try {
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw "Python is missing: $PythonPath" }
    Get-Process | Where-Object { $_.ProcessName -match "^(WindowsCamera|Camera|AMCap|obs64)$" } | ForEach-Object { throw "Close camera app before running: $($_.ProcessName)" }
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    foreach ($Entry in $Manifest.files.PSObject.Properties) {
        $Name = [string]$Entry.Name
        if ([IO.Path]::IsPathRooted($Name) -or $Name -match '(^|[\/])\.\.([\/]|$)') { throw "Unsafe packet inventory path: $Name" }
        $Path = [IO.Path]::GetFullPath((Join-Path $PacketRoot $Name))
        if (-not $Path.StartsWith($PacketRoot, [StringComparison]::OrdinalIgnoreCase)) { throw "Packet path escaped root: $Name" }
        Assert-FileHash -Path $Path -Expected ([string]$Entry.Value.sha256)
    }

    $AlgorithmRoot = Join-Path $env:LOCALAPPDATA "billiardsAI\algorithm-artifacts"
    $ApprovedRoot = Join-Path $env:LOCALAPPDATA "billiardsAI\approved-assets"
    $Detector = Join-Path $AlgorithmRoot ([string]$Manifest.assets.detector.relative_root)
    $Identity = Join-Path $AlgorithmRoot ([string]$Manifest.assets.identity.relative_root)
    $Dinov2 = Join-Path $ApprovedRoot ([string]$Manifest.assets.dinov2.relative_root)
    Assert-FileHash -Path (Join-Path $Detector "training-manifest.json") -Expected ([string]$Manifest.assets.detector.manifest_sha256)
    Assert-FileHash -Path (Join-Path $Detector "checkpoint_best.pth") -Expected ([string]$Manifest.assets.detector.checkpoint_sha256)
    Assert-FileHash -Path (Join-Path $Identity "manifest.json") -Expected ([string]$Manifest.assets.identity.manifest_sha256)
    Assert-FileHash -Path (Join-Path $Identity "group-head.onnx") -Expected ([string]$Manifest.assets.identity.head_sha256)
    Assert-FileHash -Path (Join-Path $Identity "metrics.json") -Expected ([string]$Manifest.assets.identity.metrics_sha256)
    Assert-FileHash -Path (Join-Path $Dinov2 "manifest.json") -Expected ([string]$Manifest.assets.dinov2.manifest_sha256)
    Assert-FileHash -Path (Join-Path $Dinov2 "model.safetensors") -Expected ([string]$Manifest.assets.dinov2.model_sha256)
    Assert-FileHash -Path (Join-Path $Dinov2 "config.json") -Expected ([string]$Manifest.assets.dinov2.config_sha256)
    Assert-FileHash -Path (Join-Path $Dinov2 "preprocessor_config.json") -Expected ([string]$Manifest.assets.dinov2.preprocessor_sha256)

    $Runner = Join-Path $PacketRoot "scripts\algorithm\run_mainline1_usb_live.py"
    $Setup = Join-Path $PacketRoot "candidate\capture-setup.json"
    $CalibrationRoot = Join-Path $PacketRoot "candidate\calibrations"
    $Config = Join-Path $PacketRoot "configs\algorithm\shot-local.yaml"
    $ArtifactRoot = Join-Path $Result "artifacts"
    $SpoolRoot = Join-Path $Result "spool"
    $RuntimeLog = Join-Path $Result "runtime.log"
    New-Item -ItemType Directory -Force -Path $ArtifactRoot, $SpoolRoot | Out-Null
    Copy-Item -LiteralPath $ManifestPath -Destination (Join-Path $Result "packet-manifest.json")
    $env:PYTHONPATH = Join-Path $PacketRoot "src"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $Arguments = @(
        $Runner,
        "--device-locator", ([string]$Manifest.run.device_locator),
        "--runtime-index", ([string]$Manifest.run.runtime_index),
        "--capture-setup", $Setup,
        "--calibration-root", $CalibrationRoot,
        "--detector-training-run", $Detector,
        "--identity-training-run", $Identity,
        "--dinov2-assets", $Dinov2,
        "--config", $Config,
        "--output-root", $ArtifactRoot,
        "--spool-root", $SpoolRoot,
        "--device", "cuda",
        "--batch-size", "8",
        "--max-frames", ([string]$Manifest.run.max_frames)
    )
    Write-Host "This is a side-by-side live Mainline 1 diagnostic; installed service and state DB are untouched."
    Write-Host "Wait for LIVE_RUNTIME_READY, then make ONE clear shot. The run stops automatically."
    Write-Host "Result folder: $Result"
    $NativeErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $PythonPath @Arguments 2>&1 | Tee-Object -LiteralPath $RuntimeLog
        $ExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $NativeErrorPreference
    }

    $Parsed = @()
    foreach ($Line in (Get-Content -LiteralPath $RuntimeLog)) {
        try { $Parsed += ($Line | ConvertFrom-Json -ErrorAction Stop) } catch { }
    }
    $ReadyCount = @($Parsed | Where-Object { $_.PSObject.Properties["status"] -and $_.status -eq "LIVE_RUNTIME_READY" }).Count
    $StartedCount = @($Parsed | Where-Object { $_.PSObject.Properties["kind"] -and $_.kind -eq "SHOT_STARTED" }).Count
    $PublishedCount = @($Parsed | Where-Object { $_.PSObject.Properties["kind"] -and $_.kind -eq "SHOT_PUBLISHED" }).Count
    $AbstainedCount = @($Parsed | Where-Object { $_.PSObject.Properties["kind"] -and $_.kind -eq "SHOT_ABSTAINED" }).Count
    $DegradedCount = @($Parsed | Where-Object { $_.PSObject.Properties["kind"] -and $_.kind -eq "CAPTURE_DEGRADED" }).Count
    if ($ExitCode -ne 0) { $Outcome = "RUNTIME_FAILED" }
    elseif ($ReadyCount -ne 1) { $Outcome = "READY_SIGNAL_MISSING" }
    elseif (($PublishedCount + $AbstainedCount) -lt 1) { $Outcome = "NO_SHOT_OBSERVED" }
    else { $Outcome = "LIVE_CHAIN_OBSERVED" }
    $Summary = [ordered]@{
        schema_version = "1.0"
        status = "COLLECTED_REVIEW_REQUIRED"
        live_chain_outcome = $Outcome
        exit_code = $ExitCode
        live_runtime_ready_count = $ReadyCount
        shot_started_count = $StartedCount
        shot_published_count = $PublishedCount
        shot_abstained_count = $AbstainedCount
        capture_degraded_count = $DegradedCount
        development_identity_gate_status = "FAIL"
        official_match_or_score_written = $false
        installed_service_or_state_db_written = $false
        production_ready = $false
    }
    $Summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Result "summary.json") -Encoding UTF8
} catch {
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $Result "powershell-error.txt") -Encoding UTF8
}

$Zip = Join-Path $ResolvedOutputParent ("SEND_BACK-" + $ResultName + ".zip")
Compress-Archive -LiteralPath $Result -DestinationPath $Zip -CompressionLevel Optimal
Write-Host "SEND THIS ZIP BACK: $Zip" -ForegroundColor Green
Write-Host "Result SHA256: $((Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToLowerInvariant())"
exit $ExitCode
