#requires -Version 7.2
param(
  [Parameter(Mandatory)][string]$ProjectRoot,
  [Parameter(Mandatory)][string]$PythonPath,
  [Parameter(Mandatory)][string]$NativeHelper,
  [Parameter(Mandatory)][ValidatePattern('^[a-fA-F0-9]{64}$')][string]$NativeHelperSha256,
  [Parameter(Mandatory)][string]$NativeHelperSigner,
  [Parameter(Mandatory)][string]$DeviceLocator,
  [Parameter(Mandatory)][string]$DeviceFingerprint,
  [Parameter(Mandatory)][string]$CaptureSetup,
  [Parameter(Mandatory)][string]$CalibrationRoot,
  [Parameter(Mandatory)][string]$DetectorTrainingRun,
  [Parameter(Mandatory)][string]$IdentityTrainingRun,
  [Parameter(Mandatory)][string]$Dinov2Assets,
  [Parameter(Mandatory)][string]$Config,
  [Parameter(Mandatory)][string]$StateDb,
  [Parameter(Mandatory)][string]$RingRoot,
  [Parameter(Mandatory)][string]$JobsRoot,
  [Parameter(Mandatory)][string]$ArtifactRoot,
  [Parameter(Mandatory)][string]$JoinQrAsset,
  [Parameter(Mandatory)][string]$HostAddress,
  [Parameter(Mandatory)][ValidateRange(1, 65535)][int]$Port,
  [Parameter(Mandatory)][string]$ControlToken,
  [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
  [string]$SyncBaseUrl = '',
  [string]$DeviceId = '',
  [string]$CredentialTarget = ''
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
function Require-Path([string]$Value, [string]$Kind) {
  if (-not (Test-Path -LiteralPath $Value -PathType $Kind)) { throw "Required path is missing: $Value" }
  return (Resolve-Path -LiteralPath $Value).Path
}
$ProjectRoot = Require-Path $ProjectRoot 'Container'
$PythonPath = Require-Path $PythonPath 'Leaf'
$NativeHelper = Require-Path $NativeHelper 'Leaf'
$CaptureSetup = Require-Path $CaptureSetup 'Leaf'
$Config = Require-Path $Config 'Leaf'
$JoinQrAsset = Require-Path $JoinQrAsset 'Leaf'
$CalibrationRoot = Require-Path $CalibrationRoot 'Container'
$DetectorTrainingRun = Require-Path $DetectorTrainingRun 'Container'
$IdentityTrainingRun = Require-Path $IdentityTrainingRun 'Container'
$Dinov2Assets = Require-Path $Dinov2Assets 'Container'
$RingRoot = Require-Path $RingRoot 'Container'
$JobsRoot = Require-Path $JobsRoot 'Container'
$ArtifactRoot = Require-Path $ArtifactRoot 'Container'
$StateDb = [IO.Path]::GetFullPath($StateDb)
$null = Require-Path ([IO.Path]::GetDirectoryName($StateDb)) 'Container'
$Runner = Require-Path (Join-Path $ProjectRoot 'scripts/algorithm/serve_mainline1_usb_field.py') 'Leaf'
$ActualHash = (Get-FileHash -LiteralPath $NativeHelper -Algorithm SHA256).Hash
if ($ActualHash -ne $NativeHelperSha256) { throw 'Native helper SHA-256 mismatch' }
$Signature = Get-AuthenticodeSignature -LiteralPath $NativeHelper
if ($Signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid -or
    $Signature.SignerCertificate.Subject -ne $NativeHelperSigner) { throw 'Native helper Authenticode signer mismatch' }
$RunnerArguments = @(
  '-u', $Runner, '--native-helper', $NativeHelper, '--device-locator', $DeviceLocator,
  '--device-fingerprint', $DeviceFingerprint, '--capture-setup', $CaptureSetup,
  '--calibration-root', $CalibrationRoot, '--detector-training-run', $DetectorTrainingRun,
  '--identity-training-run', $IdentityTrainingRun, '--dinov2-assets', $Dinov2Assets,
  '--config', $Config, '--state-db', $StateDb, '--ring-root', $RingRoot,
  '--jobs-root', $JobsRoot, '--artifact-root', $ArtifactRoot, '--join-qr-asset', $JoinQrAsset,
  '--host', $HostAddress, '--port', [string]$Port, '--control-token', $ControlToken,
  '--device', $Device, '--shutdown-stdin'
)
$SyncValues = @(@($SyncBaseUrl, $DeviceId, $CredentialTarget) | Where-Object { $_ })
if ($SyncValues.Count -ne 0 -and $SyncValues.Count -ne 3) { throw 'All cloud identity inputs are required together' }
if ($SyncValues.Count -eq 3) {
  $RunnerArguments += @('--sync-base-url', $SyncBaseUrl, '--device-id', $DeviceId, '--credential-target', $CredentialTarget)
}
$StartInfo = [Diagnostics.ProcessStartInfo]::new()
$StartInfo.FileName = $PythonPath
$StartInfo.UseShellExecute = $false
$StartInfo.RedirectStandardInput = $true
$StartInfo.RedirectStandardOutput = $true
$StartInfo.RedirectStandardError = $true
foreach ($Argument in $RunnerArguments) { $StartInfo.ArgumentList.Add($Argument) }
$Process = [Diagnostics.Process]::new()
$Process.StartInfo = $StartInfo
$Started = $false
$ExitCode = 2
try {
  $Started = $Process.Start()
  $OutputLine = $Process.StandardOutput.ReadLineAsync()
  $ErrorLine = $Process.StandardError.ReadLineAsync()
  while (-not $Process.HasExited -or $null -ne $OutputLine -or $null -ne $ErrorLine) {
    if ($null -ne $OutputLine -and $OutputLine.IsCompleted) {
      $Line = $OutputLine.GetAwaiter().GetResult()
      if ($null -eq $Line) { $OutputLine = $null } else {
        [Console]::Out.WriteLine($Line)
        $OutputLine = $Process.StandardOutput.ReadLineAsync()
      }
    }
    if ($null -ne $ErrorLine -and $ErrorLine.IsCompleted) {
      $Line = $ErrorLine.GetAwaiter().GetResult()
      if ($null -eq $Line) { $ErrorLine = $null } else {
        [Console]::Error.WriteLine($Line)
        $ErrorLine = $Process.StandardError.ReadLineAsync()
      }
    }
    Start-Sleep -Milliseconds 20
  }
  $ExitCode = $Process.ExitCode
} finally {
  if ($Started -and -not $Process.HasExited) {
    try { $Process.StandardInput.WriteLine('STOP'); $Process.StandardInput.Flush() } catch { }
    if (-not $Process.WaitForExit(8000)) { $Process.Kill($true); $Process.WaitForExit() }
  }
  $Process.Dispose()
}
exit $ExitCode
