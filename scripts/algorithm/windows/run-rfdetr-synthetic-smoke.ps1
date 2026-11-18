[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [Parameter(Mandatory = $true)]
    [string]$WeightPath,

    [string]$ArtifactsRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\algorithm-artifacts")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$artifactDirectory = $null
$transcriptStarted = $false
$frozenInputs = $null

function Invoke-Captured {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$StdoutPath,
        [string]$StderrPath,
        [string]$ManifestPath,
        [string]$ArtifactDirectory
    )

    $captureArguments = @(
        "-3.12",
        $helper,
        "run-captured",
        "--artifact-directory", $ArtifactDirectory,
        "--stdout", $StdoutPath,
        "--stderr", $StderrPath,
        "--manifest", $ManifestPath,
        "--",
        $FilePath
    )
    $captureArguments += $Arguments
    $previousErrorActionPreference = $ErrorActionPreference
    $captureHelperExit = 125
    try {
        $ErrorActionPreference = "Continue"
        & py @captureArguments
        $captureHelperExit = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($captureHelperExit -ne 0) {
        throw "Raw capture helper failed (exit $captureHelperExit): $FilePath"
    }

    $exitArguments = @(
        "-3.12",
        $helper,
        "read-capture-exit",
        "--artifact-directory", $ArtifactDirectory,
        "--manifest", $ManifestPath
    )
    $exitHelperExit = 125
    try {
        $ErrorActionPreference = "Continue"
        $targetExitRaw = & py @exitArguments
        $exitHelperExit = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitHelperExit -ne 0) {
        throw "Capture manifest validation failed (exit $exitHelperExit): $ManifestPath"
    }
    $targetExit = 0
    $targetExitText = [string]($targetExitRaw | Select-Object -Last 1)
    if (-not [int]::TryParse($targetExitText, [ref]$targetExit)) {
        throw "Capture manifest returned an invalid target exit: $targetExitText"
    }
    return $targetExit
}

function Invoke-Classification {
    param(
        [string]$CaptureManifestPath
    )

    $classificationArguments = @(
        "classify-smoke",
        "--project-root", $resolvedProject,
        "--capture-manifest", $CaptureManifestPath,
        "--artifact-directory", $artifactDirectory,
        "--output", (Join-Path $artifactDirectory "result.json")
    )
    & py -3.12 $helper @classificationArguments
    return $LASTEXITCODE
}

try {
    $resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
    $resolvedWeight = [IO.Path]::GetFullPath($WeightPath)
    $resolvedArtifacts = [IO.Path]::GetFullPath($ArtifactsRoot)
    $helper = Join-Path $resolvedProject "src\billiards_ai\lab\raytheon_handoff.py"
    $python = Join-Path $resolvedProject ".venv-rfdetr\Scripts\python.exe"
    $venvRoot = Join-Path $resolvedProject ".venv-rfdetr"
    $smokeScript = Join-Path $resolvedProject "scripts\algorithm\rfdetr_smoke.py"
    foreach ($requiredPath in @($helper, $python, $smokeScript, $resolvedWeight)) {
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Required file is missing: $requiredPath"
        }
    }
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "py -3.12 was not found; frozen inputs and evidence cannot be verified."
    }

    $artifactDirectoryRaw = & py -3.12 $helper create-artifact --project-root $resolvedProject --root $resolvedArtifacts --kind smoke
    if ($LASTEXITCODE -ne 0) { throw "Could not create a unique smoke evidence directory." }
    $artifactDirectory = [IO.Path]::GetFullPath(($artifactDirectoryRaw | Select-Object -Last 1))
    Start-Transcript -Path (Join-Path $artifactDirectory "smoke-transcript.txt") | Out-Null
    $transcriptStarted = $true

    $frozenInputsRaw = & py -3.12 $helper verify-inputs --project-root $resolvedProject --weight $resolvedWeight
    if ($LASTEXITCODE -ne 0) { throw "Frozen config, weight, hash, or seed validation failed before model loading." }
    $frozenInputsRaw | Set-Content -LiteralPath (Join-Path $artifactDirectory "frozen-inputs.json") -Encoding UTF8
    $frozenInputs = $frozenInputsRaw | ConvertFrom-Json

    $venvItem = Get-Item -LiteralPath $venvRoot
    if (($venvItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw ".venv-rfdetr is a reparse point; refusing to run."
    }
    $venvProbePath = Join-Path $artifactDirectory "venv-identity.json"
    $venvProbeRaw = & $python $helper interpreter-probe
    if ($LASTEXITCODE -ne 0) { throw "Could not read .venv-rfdetr interpreter identity." }
    $venvProbeRaw | Set-Content -LiteralPath $venvProbePath -Encoding UTF8
    & py -3.12 $helper validate-venv --venv $venvRoot --probe $venvProbePath
    if ($LASTEXITCODE -ne 0) { throw ".venv-rfdetr identity validation failed." }

    $workspace = Join-Path $artifactDirectory "workspace"
    $runsRoot = Join-Path $artifactDirectory "runs"
    New-Item -ItemType Directory -Path $workspace | Out-Null
    New-Item -ItemType Directory -Path $runsRoot | Out-Null
    $env:PYTHONPATH = Join-Path $resolvedProject "src"

    $deviceSummaryPath = Join-Path $artifactDirectory "cuda-before.json"
    $deviceStderrPath = Join-Path $artifactDirectory "cuda-before.stderr.txt"
    $deviceManifestPath = Join-Path $artifactDirectory "cuda-before.capture.json"
    $deviceArguments = @($helper, "cuda-preflight", "--seed", [string]$frozenInputs.seed)
    $deviceExit = Invoke-Captured $python $deviceArguments $deviceSummaryPath `
        $deviceStderrPath $deviceManifestPath $artifactDirectory
    if ($deviceExit -ne 0) {
        $classifiedExit = Invoke-Classification $deviceManifestPath
        Write-Host "CUDA preflight stopped. Return only this evidence directory: $artifactDirectory"
        exit $classifiedExit
    }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $vramBeforeExit = Invoke-Captured "nvidia-smi" `
            @("--query-gpu=name,driver_version,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits") `
            (Join-Path $artifactDirectory "vram-before.csv") `
            (Join-Path $artifactDirectory "vram-before.stderr.txt") `
            (Join-Path $artifactDirectory "vram-before.capture.json") `
            $artifactDirectory
        if ($vramBeforeExit -ne 0) { throw "Pre-run nvidia-smi diagnostics failed." }
    }

    $stdoutPath = Join-Path $artifactDirectory "smoke-summary.json"
    $stderrPath = Join-Path $artifactDirectory "smoke.stderr.txt"
    $smokeManifestPath = Join-Path $artifactDirectory "smoke.capture.json"
    $smokeArguments = @(
        $smokeScript,
        "--workspace", $workspace,
        "--runs-root", $runsRoot,
        "--weight", $resolvedWeight,
        "--accelerator", "cuda",
        "--amp"
    )
    $smokeExit = Invoke-Captured $python $smokeArguments $stdoutPath $stderrPath `
        $smokeManifestPath $artifactDirectory

    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $vramAfterExit = Invoke-Captured "nvidia-smi" `
            @("--query-gpu=name,driver_version,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits") `
            (Join-Path $artifactDirectory "vram-after.csv") `
            (Join-Path $artifactDirectory "vram-after.stderr.txt") `
            (Join-Path $artifactDirectory "vram-after.capture.json") `
            $artifactDirectory
        if ($vramAfterExit -ne 0) { throw "Post-run nvidia-smi diagnostics failed." }
    }
    $classifiedExit = Invoke-Classification $smokeManifestPath
    if ($classifiedExit -eq 0) {
        Write-Host "PASS: synthetic single-batch lifecycle completed. Unique evidence directory: $artifactDirectory"
        Write-Host "This is not model accuracy, real-video quality, or production-release evidence."
    }
    else {
        Write-Host "Smoke was classified from explicit JSON. Return only this evidence directory: $artifactDirectory"
    }
    exit $classifiedExit
}
catch {
    [Console]::Error.WriteLine("Smoke stopped: $($_.Exception.Message)")
    if ($null -ne $artifactDirectory) {
        $failureSummaryPath = Join-Path $artifactDirectory "failure-summary.json"
        [ordered]@{
            schema_version = "1.0"
            status = "OTHER_FAILURE"
            exit_code = 22
            detail = $_.Exception.Message
            synthetic_only = $true
        } | ConvertTo-Json | Set-Content -LiteralPath $failureSummaryPath -Encoding UTF8
        $failureArguments = @(
            "classify-smoke",
            "--project-root", $resolvedProject,
            "--native-exit-code", "22",
            "--summary", $failureSummaryPath,
            "--artifact-directory", $artifactDirectory,
            "--output", (Join-Path $artifactDirectory "result.json")
        )
        & py -3.12 $helper @failureArguments
        Write-Host "Return only this evidence directory: $artifactDirectory"
    }
    exit 22
}
finally {
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
}
