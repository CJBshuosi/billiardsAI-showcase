[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$Manifest,
  [Parameter(Mandatory = $true)][string]$OutputRoot,
  [string]$CalibrationRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\algorithm-artifacts\mainline1-calibrations"),
  [string]$ProjectRoot = (Get-Location).Path,
  [string]$Python = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($Python)) {
  $Python = Join-Path $ProjectRoot ".venv-rfdetr\Scripts\python.exe"
}
$Runner = Join-Path $ProjectRoot "scripts\algorithm\run_shot_local_regression_suite.py"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) { throw "Regression runner not found: $Runner" }
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) { throw "Suite manifest not found: $Manifest" }
if (-not (Test-Path -LiteralPath $CalibrationRoot -PathType Container)) { throw "Calibration registry not found: $CalibrationRoot" }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
& $Python $Runner --manifest $Manifest --output-root $OutputRoot "--calibration-root" $CalibrationRoot
if ($LASTEXITCODE -ne 0) { throw "Shot-local regression suite exited with code $LASTEXITCODE" }
