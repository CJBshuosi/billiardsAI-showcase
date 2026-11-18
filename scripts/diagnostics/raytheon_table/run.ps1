param(
    [Parameter(Mandatory=$true)][string]$PythonPath,
    [string]$OutputParent = "",
    [switch]$BoardCheck,
    [switch]$FrozenCheck,
    [switch]$MotionCheck
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw "Existing Python not found: $PythonPath. Nothing was installed." }
if (-not $OutputParent) { $OutputParent = $PSScriptRoot }
$TableOutputRoot = [System.IO.Path]::GetFullPath($OutputParent)
if (-not (Test-Path -LiteralPath $TableOutputRoot -PathType Container)) { throw "Output parent not found: $TableOutputRoot" }
$TableResult = Join-Path $TableOutputRoot ("rer-table-result-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + [guid]::NewGuid().ToString("N").Substring(0,8))
$TableOldPythonPath = $env:PYTHONPATH
$TableOldNoBytecode = $env:PYTHONDONTWRITEBYTECODE
$TableOldUtf8 = $env:PYTHONUTF8
$TableExit = 2
try {
    $env:PYTHONPATH = Join-Path $PSScriptRoot "src"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    $env:PYTHONUTF8 = "1"
    Write-Host "Table diagnostic only. No service, registry, or business data changes."
    $TableEntry = "scripts/diagnostics/raytheon_table/table_check.py"
    if ($BoardCheck) { $TableEntry = "scripts/diagnostics/raytheon_table/board_check.py" }
    if ($FrozenCheck) { $TableEntry = "scripts/diagnostics/raytheon_table/frozen_check.py" }
    if ($MotionCheck) { $TableEntry = "scripts/diagnostics/raytheon_table/motion_check.py" }
    & $PythonPath (Join-Path $PSScriptRoot $TableEntry) --output $TableResult
    $TableExit = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $TableOldPythonPath
    $env:PYTHONDONTWRITEBYTECODE = $TableOldNoBytecode
    $env:PYTHONUTF8 = $TableOldUtf8
}
if ($TableExit -ne 0) { Write-Host "Diagnostic did not finish. Keep the console error and any SEND_BACK ZIP. Do not activate calibration." }
exit $TableExit
