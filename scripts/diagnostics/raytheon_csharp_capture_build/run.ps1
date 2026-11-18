[CmdletBinding()]
param(
    [string]$OutputParent = (Join-Path ([Environment]::GetFolderPath("UserProfile")) "Downloads")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) { $PSNativeCommandUseErrorActionPreference = $false }

$packetRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$resultRoot = Join-Path ([IO.Path]::GetFullPath($OutputParent)) "raytheon-csharp-capture-build-result-$stamp"
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null
$status = "FAIL"
$reason = $null
$exitCode = 0
$compilerOutput = $null
$selfTestOutput = $null
$compiler = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
$source = Join-Path $packetRoot "native\windows\mainline1_uvc_capture_csharp\main.cs"
$binary = Join-Path $resultRoot "mainline1_uvc_capture.exe"

try {
    if (-not (Test-Path -LiteralPath $compiler)) { throw "The verified built-in 64-bit C# compiler is missing." }
    $outArgument = "/out:" + $binary
    $compilerOutput = (& $compiler /nologo /platform:x64 /optimize+ /target:exe $outArgument $source 2>&1 | Out-String).Trim()
    $compilerOutput | Set-Content -LiteralPath (Join-Path $resultRoot "compiler.log") -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw "Native capture compilation failed with code $LASTEXITCODE." }
    if (-not (Test-Path -LiteralPath $binary)) { throw "Native capture compiler did not create the helper." }

    $selfTestError = Join-Path $resultRoot "self-test.stderr.log"
    $selfTestOutput = (& $binary 2> $selfTestError | Out-String).Trim()
    $selfTestExit = $LASTEXITCODE
    $selfTestErrorText = (Get-Content -LiteralPath $selfTestError -Raw).Trim()
    if ($selfTestExit -ne 2 -or $selfTestErrorText -notlike '*"kind":"native_capture_error"*') {
        throw "Native capture argument self-test did not fail closed as expected."
    }

    $signature = Get-AuthenticodeSignature -LiteralPath $binary
    [ordered]@{
        bytes = (Get-Item -LiteralPath $binary).Length
        sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
        authenticode_status = [string]$signature.Status
        signer_subject = if ($null -eq $signature.SignerCertificate) { $null } else { $signature.SignerCertificate.Subject }
        production_ready = $false
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $resultRoot "binary.json") -Encoding UTF8
    $status = "PASS"
} catch {
    $reason = $_.Exception.Message
    $exitCode = 2
    $reason | Set-Content -LiteralPath (Join-Path $resultRoot "error.txt") -Encoding UTF8
}

[ordered]@{
    schema_version = "1.0"
    kind = "raytheon_csharp_capture_build"
    status = $status
    reason = $reason
    captured_at_utc = [DateTime]::UtcNow.ToString("o")
    computer_name = $env:COMPUTERNAME
    os = [Environment]::OSVersion.VersionString
    powershell = $PSVersionTable.PSVersion.ToString()
    clr = [Environment]::Version.ToString()
    compiler = $compiler
    compiler_output = $compilerOutput
    source_sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    production_ready = $false
    camera_opened = $false
    service_or_production_data_mutated = $false
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $resultRoot "environment.json") -Encoding UTF8

$zipPath = Join-Path ([IO.Path]::GetFullPath($OutputParent)) ("SEND_BACK-" + (Split-Path -Leaf $resultRoot) + ".zip")
Compress-Archive -LiteralPath $resultRoot -DestinationPath $zipPath -Force
Write-Host "SEND BACK: $zipPath"
exit $exitCode
