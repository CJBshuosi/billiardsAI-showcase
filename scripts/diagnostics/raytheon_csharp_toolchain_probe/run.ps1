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
$resultRoot = Join-Path ([IO.Path]::GetFullPath($OutputParent)) "raytheon-csharp-toolchain-probe-result-$stamp"
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null
$status = "FAIL"
$reason = $null
$exitCode = 0
$compilerRoute = $null
$compilerOutput = $null
$probeOutput = $null
$frameworkCsc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
$source = Join-Path $packetRoot "probe.cs"
$binary = Join-Path $resultRoot "probe.exe"

try {
    if (-not (Test-Path -LiteralPath $frameworkCsc)) { throw "The built-in 64-bit .NET Framework C# compiler was not found." }
    $outArgument = "/out:" + $binary
    $compilerOutput = (& $frameworkCsc /nologo /target:exe $outArgument $source 2>&1 | Out-String).Trim()
    $compilerOutput | Set-Content -LiteralPath (Join-Path $resultRoot "compiler.log") -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw "The built-in C# compiler exited with code $LASTEXITCODE." }
    if (-not (Test-Path -LiteralPath $binary)) { throw "The built-in C# compiler did not create probe.exe." }
    $compilerRoute = "Framework64.csc.exe"
    $probeOutput = (& $binary 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Compiled probe exited with code $LASTEXITCODE." }
    if ($probeOutput -ne "BILLIARDSAI_CSHARP_PROBE_OK") { throw "Compiled probe returned unexpected output." }
    $status = "PASS"
} catch {
    $reason = $_.Exception.Message
    $exitCode = 2
    $reason | Set-Content -LiteralPath (Join-Path $resultRoot "error.txt") -Encoding UTF8
}

$environment = [ordered]@{
    schema_version = "1.0"
    kind = "raytheon_csharp_toolchain_probe"
    status = $status
    reason = $reason
    captured_at_utc = [DateTime]::UtcNow.ToString("o")
    computer_name = $env:COMPUTERNAME
    os = [Environment]::OSVersion.VersionString
    powershell = $PSVersionTable.PSVersion.ToString()
    clr = [Environment]::Version.ToString()
    powershell_edition = if ($PSVersionTable.PSObject.Properties.Name -contains "PSEdition") { $PSVersionTable.PSEdition } else { "Desktop" }
    language_mode = [string]$ExecutionContext.SessionState.LanguageMode
    framework64_csc_exists = Test-Path -LiteralPath $frameworkCsc
    framework64_csc = if (Test-Path -LiteralPath $frameworkCsc) { $frameworkCsc } else { $null }
    compiler_route = $compilerRoute
    compiler_output = $compilerOutput
    probe_output = $probeOutput
    source_sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    binary_sha256 = if (Test-Path -LiteralPath $binary) { (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
}
$environment | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $resultRoot "environment.json") -Encoding UTF8
$zipPath = Join-Path ([IO.Path]::GetFullPath($OutputParent)) ("SEND_BACK-" + (Split-Path -Leaf $resultRoot) + ".zip")
Compress-Archive -LiteralPath $resultRoot -DestinationPath $zipPath -Force
Write-Host "SEND BACK: $zipPath"
exit $exitCode
