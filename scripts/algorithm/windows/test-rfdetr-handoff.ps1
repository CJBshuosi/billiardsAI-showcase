[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [string]$ArtifactsRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\algorithm-artifacts")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$transcriptStarted = $false
$cleanupSentinel = $null
$selfTestDirectory = $null
$transcriptPath = $null
$failureMessage = $null

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

try {
    $resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
    $helper = Join-Path $resolvedProject "src\billiards_ai\lab\raytheon_handoff.py"
    if (-not (Get-Command py -ErrorAction SilentlyContinue) -or
        -not (Test-Path -LiteralPath $helper -PathType Leaf)) {
        throw "py -3.12 and the Raytheon handoff helper are required."
    }
    $selfTestRaw = & py -3.12 $helper create-artifact --project-root $resolvedProject --root ([IO.Path]::GetFullPath($ArtifactsRoot)) --kind selftest
    if ($LASTEXITCODE -ne 0) { throw "Could not create a self-test evidence directory." }
    $selfTestDirectory = [IO.Path]::GetFullPath(($selfTestRaw | Select-Object -Last 1))
    $transcriptPath = Join-Path $selfTestDirectory "self-test-transcript.txt"
    Start-Transcript -Path $transcriptPath | Out-Null
    $transcriptStarted = $true

    $warningStdoutPath = Join-Path $selfTestDirectory "native-warning.stdout.json"
    $warningStderrPath = Join-Path $selfTestDirectory "native-warning.stderr.txt"
    $warningManifestPath = Join-Path $selfTestDirectory "native-warning.capture.json"
    $warningArguments = @("-3.12", $helper, "native-warning-probe")
    $warningExit = Invoke-Captured "py" $warningArguments $warningStdoutPath `
        $warningStderrPath $warningManifestPath $selfTestDirectory
    if ($warningExit -ne 0) {
        throw "Native warning capture did not preserve exit code 0."
    }
    $warningText = Get-Content -LiteralPath $warningStderrPath -Raw -Encoding UTF8
    if ($warningText -notmatch "nonfatal native stderr warning") {
        throw "Native warning stderr evidence was not retained."
    }
    $warningResultPath = Join-Path $selfTestDirectory "native-warning-result.json"
    & py -3.12 $helper classify-smoke `
        --project-root $resolvedProject `
        --capture-manifest $warningManifestPath `
        --artifact-directory $selfTestDirectory `
        --output $warningResultPath
    if ($LASTEXITCODE -ne 0) {
        throw "Native warning exit 0 did not continue to typed PASS classification."
    }
    $warningResult = Get-Content -LiteralPath $warningResultPath -Raw | ConvertFrom-Json
    if ($warningResult.status -ne "PASS" -or $warningResult.exit_code -ne 0) {
        throw "Native warning classification did not produce typed PASS."
    }
    if ($ErrorActionPreference -ne "Stop") {
        throw "Native warning capture did not restore ErrorActionPreference."
    }

    $quoted = "argument with spaces"
    $quotedStdoutPath = Join-Path $selfTestDirectory "spaced-argument.stdout.txt"
    $quotedStderrPath = Join-Path $selfTestDirectory "spaced-argument.stderr.txt"
    $quotedManifestPath = Join-Path $selfTestDirectory "spaced-argument.capture.json"
    $quotedArguments = @("-3.12", $helper, "echo-argument", "--value", $quoted)
    $quotedExit = Invoke-Captured "py" $quotedArguments $quotedStdoutPath `
        $quotedStderrPath $quotedManifestPath $selfTestDirectory
    $roundTrip = Get-Content -LiteralPath $quotedStdoutPath -Encoding UTF8 | `
        Select-Object -Last 1
    if ($quotedExit -ne 0 -or $roundTrip -ne $quoted) {
        throw "PowerShell native argument quoting self-test failed."
    }

    $exitStdoutPath = Join-Path $selfTestDirectory "native-exit.stdout.txt"
    $exitStderrPath = Join-Path $selfTestDirectory "native-exit.stderr.txt"
    $exitManifestPath = Join-Path $selfTestDirectory "native-exit.capture.json"
    $exitArguments = @("-3.12", $helper, "exit-with-code", "--code", "7")
    $exitProbeExit = Invoke-Captured "py" $exitArguments $exitStdoutPath `
        $exitStderrPath $exitManifestPath $selfTestDirectory
    if ($exitProbeExit -ne 7) {
        throw "PowerShell native exit-code self-test failed."
    }
    if ($ErrorActionPreference -ne "Stop") {
        throw "Native nonzero capture did not restore ErrorActionPreference."
    }

    $cleanupSentinel = Join-Path $selfTestDirectory "cleanup-sentinel.tmp"
    New-Item -ItemType File -Path $cleanupSentinel | Out-Null
}
catch {
    $failureMessage = $_.Exception.Message
}
finally {
    if ($null -ne $cleanupSentinel -and (Test-Path -LiteralPath $cleanupSentinel)) {
        Remove-Item -LiteralPath $cleanupSentinel -Force
    }
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
}

$cleanupStillExists = $null -ne $cleanupSentinel -and (Test-Path -LiteralPath $cleanupSentinel)
if ($cleanupStillExists) {
    $failureMessage = "PowerShell finally cleanup self-test failed."
}
if ($null -eq $transcriptPath -or -not (Test-Path -LiteralPath $transcriptPath -PathType Leaf)) {
    $failureMessage = "PowerShell transcript file does not exist."
}
else {
    try {
        Add-Content -LiteralPath $transcriptPath -Value "transcript-closed=true" -Encoding UTF8
    }
    catch {
        $failureMessage = "Stop-Transcript did not close the transcript file: $($_.Exception.Message)"
    }
}
if ($null -ne $failureMessage) {
    [Console]::Error.WriteLine("PowerShell handoff self-test failed: $failureMessage")
    exit 1
}
Write-Host "PASS: Python raw capture, quoting, transcript, exit, classification, and cleanup self-test."
Write-Host "Evidence directory: $selfTestDirectory"
exit 0
