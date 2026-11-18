[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [string]$ArtifactsRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\algorithm-artifacts")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Invoke-Native {
    param([string]$FilePath, [string[]]$Arguments)

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $FilePath $($Arguments -join ' ')"
    }
}

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

$artifactDirectory = $null
$transcriptStarted = $false

try {
    $resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
    $resolvedArtifacts = [IO.Path]::GetFullPath($ArtifactsRoot)
    $helper = Join-Path $resolvedProject "src\billiards_ai\lab\raytheon_handoff.py"
    if (-not (Test-Path -LiteralPath (Join-Path $resolvedProject "pyproject.toml") -PathType Leaf) -or
        -not (Test-Path -LiteralPath $helper -PathType Leaf)) {
        throw "ProjectRoot is not a complete billiardsAI repository: $resolvedProject"
    }
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Windows Python Launcher (py) not found. Install 64-bit CPython 3.12; do not install a standalone CUDA Toolkit."
    }

    $baseProbeRaw = & py -3.12 $helper interpreter-probe
    if ($LASTEXITCODE -ne 0) {
        throw "py -3.12 is unavailable. Install 64-bit CPython 3.12 and retry."
    }
    $baseProbe = $baseProbeRaw | ConvertFrom-Json
    if ($baseProbe.implementation -ne "cpython" -or $baseProbe.major -ne 3 -or
        $baseProbe.minor -ne 12 -or $baseProbe.pointer_bits -ne 64) {
        throw "The frozen environment requires 64-bit CPython 3.12."
    }

    $artifactDirectoryRaw = & py -3.12 $helper create-artifact --project-root $resolvedProject --root $resolvedArtifacts --kind install
    if ($LASTEXITCODE -ne 0) { throw "Could not create a unique install evidence directory." }
    $artifactDirectory = [IO.Path]::GetFullPath(($artifactDirectoryRaw | Select-Object -Last 1))
    Start-Transcript -Path (Join-Path $artifactDirectory "install-transcript.txt") | Out-Null
    $transcriptStarted = $true

    $packagePlanRaw = & py -3.12 $helper package-plan --project-root $resolvedProject
    if ($LASTEXITCODE -ne 0) { throw "Frozen package plan validation failed." }
    $packagePlan = $packagePlanRaw | ConvertFrom-Json
    $packagePlanRaw | Set-Content -LiteralPath (Join-Path $artifactDirectory "package-plan.json") -Encoding UTF8

    $venvRoot = Join-Path $resolvedProject ".venv-rfdetr"
    $venvPython = Join-Path $venvRoot "Scripts\python.exe"
    if (Test-Path -LiteralPath $venvRoot) {
        $venvItem = Get-Item -LiteralPath $venvRoot
        if (($venvItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
            -not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
            throw "Existing .venv-rfdetr is not a verifiable normal venv; refusing reuse or overwrite."
        }
    }
    else {
        Invoke-Native "py" @("-3.12", "-m", "venv", $venvRoot)
    }

    $venvProbePath = Join-Path $artifactDirectory "venv-identity.json"
    $venvProbeRaw = & $venvPython $helper interpreter-probe
    if ($LASTEXITCODE -ne 0) { throw "Could not read .venv-rfdetr interpreter identity." }
    $venvProbeRaw | Set-Content -LiteralPath $venvProbePath -Encoding UTF8
    Invoke-Native "py" @("-3.12", $helper, "validate-venv", "--venv", $venvRoot, "--probe", $venvProbePath)

    $constraintsPath = Join-Path $artifactDirectory "selected-torch-constraints.txt"
    @($packagePlan.torch, $packagePlan.torchvision) | Set-Content -LiteralPath $constraintsPath -Encoding ASCII

    Invoke-Native $venvPython @("-m", "pip", "install", "--upgrade", "--index-url", $packagePlan.pypi_index, "pip", "setuptools", "wheel")
    Invoke-Native $venvPython @("-m", "pip", "install", "--index-url", $packagePlan.pytorch_index, $packagePlan.torch, $packagePlan.torchvision)
    Invoke-Native $venvPython @("-m", "pip", "install", "--index-url", $packagePlan.pypi_index, "--extra-index-url", $packagePlan.pytorch_index, "--constraint", $constraintsPath, $packagePlan.rfdetr)
    Invoke-Native $venvPython @("-m", "pip", "install", "--no-deps", "--editable", $resolvedProject, "--index-url", $packagePlan.pypi_index)

    $pipFreezeExit = Invoke-Captured $venvPython @("-m", "pip", "freeze") `
        (Join-Path $artifactDirectory "pip-freeze.txt") `
        (Join-Path $artifactDirectory "pip-freeze.stderr.txt") `
        (Join-Path $artifactDirectory "pip-freeze.capture.json") `
        $artifactDirectory
    if ($pipFreezeExit -ne 0) { throw "pip freeze failed." }
    $pipCheckExit = Invoke-Captured $venvPython @("-m", "pip", "check") `
        (Join-Path $artifactDirectory "pip-check.txt") `
        (Join-Path $artifactDirectory "pip-check.stderr.txt") `
        (Join-Path $artifactDirectory "pip-check.capture.json") `
        $artifactDirectory
    if ($pipCheckExit -ne 0) { throw "pip check failed. Return this evidence directory and do not continue to smoke." }

    $diagnosticsRaw = & $venvPython $helper runtime-diagnostics
    if ($LASTEXITCODE -ne 0) { throw "Torch/CUDA diagnostics failed. Return this evidence directory and do not continue to smoke." }
    $diagnosticsRaw | Set-Content -LiteralPath (Join-Path $artifactDirectory "runtime-diagnostics.json") -Encoding UTF8
    $diagnostics = $diagnosticsRaw | ConvertFrom-Json
    if ($diagnostics.torch -ne ($packagePlan.torch -replace '^torch==', '') -or
        $diagnostics.torchvision -ne ($packagePlan.torchvision -replace '^torchvision==', '') -or
        $diagnostics.rfdetr -ne ($packagePlan.rfdetr -replace '^rfdetr\[train,onnx\]==', '') -or
        $diagnostics.cuda_runtime -ne "12.6") {
        throw "Installed versions do not match the frozen package plan."
    }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $nvidiaSmiExit = Invoke-Captured "nvidia-smi" @("-q") `
            (Join-Path $artifactDirectory "nvidia-smi.txt") `
            (Join-Path $artifactDirectory "nvidia-smi.stderr.txt") `
            (Join-Path $artifactDirectory "nvidia-smi.capture.json") `
            $artifactDirectory
        if ($nvidiaSmiExit -ne 0) { throw "nvidia-smi diagnostics failed. Return this evidence directory and do not continue to smoke." }
    }

    Write-Host "Installation complete. Unique evidence directory: $artifactDirectory"
    Write-Host "Next, follow the runbook for synthetic smoke. Do not use real video or claim model accuracy."
}
catch {
    [Console]::Error.WriteLine("Installation stopped: $($_.Exception.Message)")
    if ($null -ne $artifactDirectory) {
        Write-Host "Return only logs from this evidence directory: $artifactDirectory"
    }
    Write-Host "Do not continue to smoke and do not install a standalone CUDA Toolkit."
    exit 1
}
finally {
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
}
