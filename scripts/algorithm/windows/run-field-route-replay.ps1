param(
  [Parameter(Mandatory = $true)][string]$ProjectRoot,
  [Parameter(Mandatory = $true)][string]$Video,
  [string]$Mainline1Artifact = "",
  [string]$Observations = "",
  [string]$ShotWindows = "",
  [string]$Calibration = "",
  [Parameter(Mandatory = $true)][ValidateSet("open", "solid", "stripe")][string]$ContextGroup,
  [Parameter(Mandatory = $true)][string]$OutputRoot,
  [double]$MotionThresholdMm = 8,
  [int]$MotionConfirmFrames = 2,
  [int]$StableDurationMs = 400,
  [int]$VisualPixelDeltaThreshold = 18,
  [double]$VisualChangedPixelRatio = 0.0005
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ResolvedProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$ResolvedVideo = (Resolve-Path -LiteralPath $Video).Path
$Python = Join-Path $ResolvedProjectRoot ".venv-rfdetr\Scripts\python.exe"
$Runner = Join-Path $ResolvedProjectRoot "scripts\algorithm\run_field_route_replay.py"

$HasMainline1Artifact = -not [string]::IsNullOrWhiteSpace($Mainline1Artifact)
$HasObservations = -not [string]::IsNullOrWhiteSpace($Observations)
if ($HasMainline1Artifact -eq $HasObservations) {
  throw "Specify exactly one of -Mainline1Artifact or -Observations."
}
if ($HasMainline1Artifact -and ((-not [string]::IsNullOrWhiteSpace($Calibration)) -or (-not [string]::IsNullOrWhiteSpace($ShotWindows)))) {
  throw "-Mainline1Artifact cannot be combined with -Calibration or -ShotWindows."
}
if ($HasObservations -and [string]::IsNullOrWhiteSpace($Calibration)) {
  throw "Legacy -Observations mode requires -Calibration."
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  throw "Python does not exist: $Python"
}
if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
  throw "Field route replay runner does not exist: $Runner"
}

$RunnerArguments = @("--video", $ResolvedVideo, "--context-group", $ContextGroup, "--output-root", $OutputRoot, "--motion-threshold-mm", $MotionThresholdMm, "--motion-confirm-frames", $MotionConfirmFrames, "--stable-duration-ms", $StableDurationMs, "--visual-pixel-delta-threshold", $VisualPixelDeltaThreshold, "--visual-changed-pixel-ratio", $VisualChangedPixelRatio)
if ($HasMainline1Artifact) {
  $ResolvedMainline1Artifact = (Resolve-Path -LiteralPath $Mainline1Artifact).Path
  $RunnerArguments += @("--mainline1-artifact", $ResolvedMainline1Artifact)
} else {
  $ResolvedObservations = (Resolve-Path -LiteralPath $Observations).Path
  $ResolvedCalibration = (Resolve-Path -LiteralPath $Calibration).Path
  $RunnerArguments += @("--observations", $ResolvedObservations, "--calibration", $ResolvedCalibration)
  if ($ShotWindows) { $RunnerArguments += @("--shot-windows", (Resolve-Path -LiteralPath $ShotWindows).Path) }
}
& $Python $Runner @RunnerArguments
exit $LASTEXITCODE
