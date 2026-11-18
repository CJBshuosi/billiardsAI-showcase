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
if ([string]::IsNullOrWhiteSpace($OutputParent)) { $OutputParent = Join-Path ([Environment]::GetFolderPath("UserProfile")) "Downloads" }
$ResolvedOutputParent = [IO.Path]::GetFullPath($OutputParent)
$ManifestPath = Join-Path $PacketRoot "packet-manifest.json"
$ResultName = "mainline1-product-realtime-result-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + [Guid]::NewGuid().ToString("N").Substring(0,6)
$Result = Join-Path $ResolvedOutputParent $ResultName
$Evidence = Join-Path $Result "evidence"
$Work = Join-Path $Result "work"
New-Item -ItemType Directory -Force -Path $Evidence, $Work | Out-Null
$ExitCode = 2

try {
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw "Python is missing: $PythonPath" }
    Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -match "^(WindowsCamera|Camera|AMCap|obs32|obs64)$" } | ForEach-Object { throw "Close camera app before running: $($_.ProcessName)" }
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    if ($Manifest.production_ready -ne $false -or $Manifest.development_evidence_only -ne $true) { throw "This runner accepts only the bounded development packet." }
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

    $Compiler = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
    if (-not (Test-Path -LiteralPath $Compiler -PathType Leaf)) { throw "Verified built-in C# compiler is missing." }
    $NativeSource = Join-Path $PacketRoot "native\windows\mainline1_uvc_capture_csharp\main.cs"
    $NativeHelper = Join-Path $Work "mainline1_uvc_capture.exe"
    $OutArgument = "/out:" + $NativeHelper
    $CompilerOutput = (& $Compiler /nologo /platform:x64 /optimize+ /target:exe $OutArgument $NativeSource 2>&1 | Out-String).Trim()
    $CompilerOutput | Set-Content -LiteralPath (Join-Path $Evidence "compiler.log") -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw "Native capture compilation failed with code $LASTEXITCODE." }
    [ordered]@{
        helper_sha256 = (Get-FileHash -LiteralPath $NativeHelper -Algorithm SHA256).Hash.ToLowerInvariant()
        source_sha256 = (Get-FileHash -LiteralPath $NativeSource -Algorithm SHA256).Hash.ToLowerInvariant()
        authenticode_status = [string](Get-AuthenticodeSignature -LiteralPath $NativeHelper).Status
        production_signature_required = $true
        development_evidence_only = $true
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Evidence "native-helper.json") -Encoding UTF8

    $Runner = Join-Path $PacketRoot "scripts\algorithm\run_mainline1_product_realtime.py"
    $Setup = Join-Path $PacketRoot "candidate\capture-setup.json"
    $CalibrationRoot = Join-Path $PacketRoot "candidate\calibrations"
    $Config = Join-Path $PacketRoot "configs\algorithm\shot-local.yaml"
    $ArtifactRoot = Join-Path $Evidence "artifacts"
    $RingRoot = Join-Path $Work "ring"
    $RuntimeLog = Join-Path $Evidence "runtime.log"
    New-Item -ItemType Directory -Force -Path $ArtifactRoot, $RingRoot | Out-Null
    Copy-Item -LiteralPath $ManifestPath -Destination (Join-Path $Evidence "packet-manifest.json")
    $env:PYTHONPATH = Join-Path $PacketRoot "src"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONWARNINGS = "ignore::FutureWarning"
    $Arguments = @(
        $Runner,
        "--native-helper", $NativeHelper,
        "--device-locator", ([string]$Manifest.run.device_locator),
        "--device-fingerprint", ([string]$Manifest.run.device_fingerprint),
        "--capture-setup", $Setup,
        "--calibration-root", $CalibrationRoot,
        "--detector-training-run", $Detector,
        "--identity-training-run", $Identity,
        "--dinov2-assets", $Dinov2,
        "--config", $Config,
        "--output-root", $ArtifactRoot,
        "--ring-root", $RingRoot,
        "--device", "cuda",
        "--batch-size", "8",
        "--stop-after-terminal-shots", ([string]$Manifest.run.stop_after_terminal_shots),
        "--diagnostic-duration-seconds", ([string]$Manifest.run.diagnostic_duration_seconds)
    )
    Write-Host "Side-by-side product realtime Mainline 1; installed service and state DB are untouched."
    Write-Host "Wait for LIVE_RUNTIME_READY, make one to three clear shots, then wait for automatic stop."
    Write-Host "The run stops automatically after one terminal shot or 60 seconds. Do not press Ctrl+C."
    Write-Host "Result folder: $Result"
    $PriorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $PythonPath @Arguments 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -LiteralPath $RuntimeLog
        $ExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PriorPreference
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
    $ProductSummary = @($Parsed | Where-Object { $_.PSObject.Properties["status"] -and $_.status -eq "PRODUCT_REALTIME_SUMMARY" } | Select-Object -Last 1)
    if ($ExitCode -ne 0) { $Outcome = "RUNTIME_FAILED" }
    elseif ($ReadyCount -ne 1) { $Outcome = "READY_SIGNAL_MISSING" }
    elseif (($PublishedCount + $AbstainedCount) -lt 1) { $Outcome = "NO_SHOT_OBSERVED" }
    else { $Outcome = "PRODUCT_REALTIME_CHAIN_OBSERVED" }
    [ordered]@{
        schema_version = "1.0"
        status = "COLLECTED_REVIEW_REQUIRED"
        product_realtime_outcome = $Outcome
        exit_code = $ExitCode
        live_runtime_ready_count = $ReadyCount
        shot_started_count = $StartedCount
        shot_published_count = $PublishedCount
        shot_abstained_count = $AbstainedCount
        capture_degraded_count = $DegradedCount
        realtime_diagnostics = $(if ($ProductSummary.Count -eq 1) { $ProductSummary.diagnostics } else { $null })
        development_identity_gate_status = "FAIL"
        official_match_or_score_written = $false
        installed_service_or_state_db_written = $false
        production_ready = $false
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Evidence "summary.json") -Encoding UTF8
} catch {
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $Evidence "powershell-error.txt") -Encoding UTF8
    $ExitCode = 2
}

$Zip = Join-Path $ResolvedOutputParent ("SEND_BACK-" + $ResultName + ".zip")
Compress-Archive -LiteralPath $Evidence -DestinationPath $Zip -CompressionLevel Optimal -Force
Write-Host "SEND THIS ZIP BACK: $Zip" -ForegroundColor Green
Write-Host "Result SHA256: $((Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToLowerInvariant())"
exit $ExitCode
