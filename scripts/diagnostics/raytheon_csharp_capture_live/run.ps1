[CmdletBinding()]
param([string]$OutputParent = (Join-Path ([Environment]::GetFolderPath("UserProfile")) "Downloads"))

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) { $PSNativeCommandUseErrorActionPreference = $false }

$packetRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$resultRoot = Join-Path ([IO.Path]::GetFullPath($OutputParent)) "raytheon-csharp-capture-live-result-$stamp"
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null
$status = "FAIL"
$reason = $null
$exitCode = 0
$compiler = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
$source = Join-Path $packetRoot "native\windows\mainline1_uvc_capture_csharp\main.cs"
$binary = Join-Path $resultRoot "mainline1_uvc_capture.exe"

try {
    $cameraApps = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -match '^(AMCap|WindowsCamera|obs32|obs64)$' })
    if ($cameraApps.Count -gt 0) { throw "Close AMCap, Windows Camera and OBS before the live test." }
    if (-not (Test-Path -LiteralPath $compiler)) { throw "The verified built-in C# compiler is missing." }
    $outArgument = "/out:" + $binary
    $compilerOutput = (& $compiler /nologo /platform:x64 /optimize+ /target:exe $outArgument $source 2>&1 | Out-String).Trim()
    $compilerOutput | Set-Content -LiteralPath (Join-Path $resultRoot "compiler.log") -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw "Native capture compilation failed with code $LASTEXITCODE." }

    $python = "C:\Users\developer\Documents\billiardsAI-raytheon3\.venv-rfdetr\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python)) {
        $python = (Get-ChildItem -LiteralPath (Join-Path $env:USERPROFILE "Documents") -Recurse -File -Filter "python.exe" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -like '*billiardsAI*\.venv-rfdetr\Scripts\python.exe' } | Select-Object -First 1).FullName
    }
    if (-not $python -or -not (Test-Path -LiteralPath $python)) { throw "The Raytheon RF-DETR Python environment was not found." }
    $liveOutput = Join-Path $resultRoot "live"
    $priorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $python (Join-Path $packetRoot "smoke.py") --helper $binary --output $liveOutput --frames 120 2>&1 | Tee-Object -LiteralPath (Join-Path $resultRoot "smoke.log")
        $smokeExit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $priorPreference
    }
    if ($smokeExit -ne 0) { throw "Native capture live smoke failed with code $smokeExit." }
    $status = "PASS"
} catch {
    $reason = $_.Exception.Message
    $exitCode = 2
    $reason | Set-Content -LiteralPath (Join-Path $resultRoot "error.txt") -Encoding UTF8
}

[ordered]@{
    schema_version = "1.0"
    kind = "raytheon_csharp_capture_live_wrapper"
    status = $status
    reason = $reason
    captured_at_utc = [DateTime]::UtcNow.ToString("o")
    computer_name = $env:COMPUTERNAME
    helper_sha256 = if (Test-Path -LiteralPath $binary) { (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
    source_sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    production_ready = $false
    service_or_production_data_mutated = $false
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $resultRoot "wrapper.json") -Encoding UTF8

$zipPath = Join-Path ([IO.Path]::GetFullPath($OutputParent)) ("SEND_BACK-" + (Split-Path -Leaf $resultRoot) + ".zip")
Compress-Archive -LiteralPath $resultRoot -DestinationPath $zipPath -Force
Write-Host "SEND BACK: $zipPath"
exit $exitCode
