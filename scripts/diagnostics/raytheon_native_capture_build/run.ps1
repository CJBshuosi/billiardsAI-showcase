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
$resultRoot = Join-Path ([IO.Path]::GetFullPath($OutputParent)) "raytheon-native-capture-build-result-$stamp"
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null
$status = "FAIL"
$reason = $null
$exitCode = 0
$cmake = $null
$generator = $null

try {
    $cmakeCommand = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmakeCommand) { $cmake = $cmakeCommand.Source }
    if ($null -eq $cmake) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path -LiteralPath $vswhere) {
            $installation = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
            if ($installation) {
                $candidate = Join-Path $installation "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
                if (Test-Path -LiteralPath $candidate) { $cmake = $candidate }
            }
        }
    }
    if ($null -eq $cmake) { throw "CMake with Visual C++ x64 build tools was not found." }

    $source = Join-Path $packetRoot "native\windows\mainline1_uvc_capture"
    $build = Join-Path $resultRoot "build"
    $configureLog = Join-Path $resultRoot "configure.log"
    $buildLog = Join-Path $resultRoot "build.log"
    & $cmake -S $source -B $build -A x64 2>&1 | Tee-Object -LiteralPath $configureLog
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed with exit $LASTEXITCODE." }
    & $cmake --build $build --config Release 2>&1 | Tee-Object -LiteralPath $buildLog
    if ($LASTEXITCODE -ne 0) { throw "Visual C++ build failed with exit $LASTEXITCODE." }

    $binary = Join-Path $build "Release\mainline1_uvc_capture.exe"
    if (-not (Test-Path -LiteralPath $binary)) { throw "Compiled native helper is missing." }
    $destination = Join-Path $resultRoot "mainline1_uvc_capture.exe"
    Copy-Item -LiteralPath $binary -Destination $destination
    $hash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    $signature = Get-AuthenticodeSignature -LiteralPath $destination
    $binaryRecord = [ordered]@{
        bytes = (Get-Item -LiteralPath $destination).Length
        sha256 = $hash
        authenticode_status = [string]$signature.Status
        signer_subject = if ($null -eq $signature.SignerCertificate) { $null } else { $signature.SignerCertificate.Subject }
    }
    $binaryRecord | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $resultRoot "binary.json") -Encoding UTF8
    $status = "PASS"
} catch {
    $reason = $_.Exception.Message
    $exitCode = 2
    $reason | Set-Content -LiteralPath (Join-Path $resultRoot "error.txt") -Encoding UTF8
}

$environment = [ordered]@{
    schema_version = "1.0"
    kind = "raytheon_native_capture_build"
    status = $status
    reason = $reason
    captured_at_utc = [DateTime]::UtcNow.ToString("o")
    computer_name = $env:COMPUTERNAME
    os = [Environment]::OSVersion.VersionString
    powershell = $PSVersionTable.PSVersion.ToString()
    cmake = $cmake
    source_sha256 = (Get-FileHash -LiteralPath (Join-Path $packetRoot "native\windows\mainline1_uvc_capture\main.cpp") -Algorithm SHA256).Hash.ToLowerInvariant()
    cmake_sha256 = (Get-FileHash -LiteralPath (Join-Path $packetRoot "native\windows\mainline1_uvc_capture\CMakeLists.txt") -Algorithm SHA256).Hash.ToLowerInvariant()
}
$environment | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $resultRoot "environment.json") -Encoding UTF8
$zipPath = Join-Path ([IO.Path]::GetFullPath($OutputParent)) ("SEND_BACK-" + (Split-Path -Leaf $resultRoot) + ".zip")
Compress-Archive -LiteralPath $resultRoot -DestinationPath $zipPath -Force
Write-Host "SEND BACK: $zipPath"
exit $exitCode
