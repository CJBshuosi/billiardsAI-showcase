[CmdletBinding()]
param(
    [string]$PythonPath = "C:\Users\developer\Documents\billiardsAI-raytheon3\.venv-rfdetr\Scripts\python.exe",
    [string]$OutputParent = $PSScriptRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) { $PSNativeCommandUseErrorActionPreference = $false }

$PacketRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$Result = Join-Path ([IO.Path]::GetFullPath($OutputParent)) ("mainline1-usb-gate-result-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + [Guid]::NewGuid().ToString("N").Substring(0,6))
New-Item -ItemType Directory -Path $Result | Out-Null
$env:PYTHONPATH = Join-Path $PacketRoot "src"
$GateScript = Join-Path $PacketRoot "gate_check.py"
$Setup = Join-Path $PacketRoot "candidate\capture-setup.json"
$CalibrationRoot = Join-Path $PacketRoot "candidate\calibrations"
$GateOutput = Join-Path $Result "gate.json"

$ExitCode = 2
try {
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw "Python is missing: $PythonPath" }
    Get-Process | Where-Object { $_.ProcessName -match "^(WindowsCamera|Camera|AMCap|obs64)$" } | ForEach-Object { throw "Close camera app before running: $($_.ProcessName)" }
    & $PythonPath $GateScript --capture-setup $Setup --calibration-root $CalibrationRoot --output $GateOutput 2>&1 | Tee-Object -LiteralPath (Join-Path $Result "gate.log")
    $ExitCode = $LASTEXITCODE
} catch {
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $Result "powershell-error.txt") -Encoding UTF8
}

Copy-Item -LiteralPath (Join-Path $PacketRoot "packet-manifest.json") -Destination (Join-Path $Result "packet-manifest.json")
$Zip = Join-Path ([IO.Path]::GetFullPath($OutputParent)) ("SEND_BACK-mainline1-usb-gate-" + (Split-Path -Leaf $Result) + ".zip")
Compress-Archive -LiteralPath $Result -DestinationPath $Zip -CompressionLevel Optimal
Write-Host "SEND BACK: $Zip" -ForegroundColor Green
exit $ExitCode
