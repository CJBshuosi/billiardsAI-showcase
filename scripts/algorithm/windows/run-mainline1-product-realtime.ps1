[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$ProjectRoot,
    [Parameter(Mandatory = $true)] [string]$RfdetrPython,
    [Parameter(Mandatory = $true)] [string]$NativeCaptureHelper,
    [Parameter(Mandatory = $true)] [string]$NativeCaptureHelperSha256,
    [Parameter(Mandatory = $true)] [string]$NativeCaptureHelperSigner,
    [Parameter(Mandatory = $true)] [string]$DeviceLocator,
    [Parameter(Mandatory = $true)] [string]$DeviceFingerprint,
    [Parameter(Mandatory = $true)] [string]$CaptureSetup,
    [Parameter(Mandatory = $true)] [string]$CalibrationRoot,
    [Parameter(Mandatory = $true)] [string]$DetectorTrainingRun,
    [Parameter(Mandatory = $true)] [string]$IdentityTrainingRun,
    [Parameter(Mandatory = $true)] [string]$Dinov2Assets,
    [Parameter(Mandatory = $true)] [string]$ShotLocalConfig,
    [Parameter(Mandatory = $true)] [string]$ArtifactRoot,
    [Parameter(Mandatory = $true)] [string]$RingRoot,
    [ValidateRange(1, 64)] [int]$BatchSize = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) { $PSNativeCommandUseErrorActionPreference = $false }

$resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
$resolvedHelper = [IO.Path]::GetFullPath($NativeCaptureHelper)
$runner = Join-Path $resolvedProject "scripts\algorithm\run_mainline1_product_realtime.py"
$sourceRoot = Join-Path $resolvedProject "src"
foreach ($required in @($RfdetrPython, $resolvedHelper, $runner, $sourceRoot, $CaptureSetup, $CalibrationRoot, $DetectorTrainingRun, $IdentityTrainingRun, $Dinov2Assets, $ShotLocalConfig)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required product realtime input is missing: $required" }
}
$actualHelperSha256 = (Get-FileHash -LiteralPath $resolvedHelper -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHelperSha256 -ne $NativeCaptureHelperSha256.ToLowerInvariant()) { throw "Native capture helper SHA-256 mismatch." }
$signature = Get-AuthenticodeSignature -LiteralPath $resolvedHelper
if ($signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) { throw "Native capture helper Authenticode signature is not valid." }
if ($null -eq $signature.SignerCertificate -or $signature.SignerCertificate.Subject -ne $NativeCaptureHelperSigner) { throw "Native capture helper signer is not authorized." }

New-Item -ItemType Directory -Force -Path $ArtifactRoot, $RingRoot | Out-Null
$env:PYTHONPATH = $sourceRoot
$arguments = @($runner, "--native-helper", $resolvedHelper, "--device-locator", $DeviceLocator, "--device-fingerprint", $DeviceFingerprint, "--capture-setup", $CaptureSetup, "--calibration-root", $CalibrationRoot, "--detector-training-run", $DetectorTrainingRun, "--identity-training-run", $IdentityTrainingRun, "--dinov2-assets", $Dinov2Assets, "--config", $ShotLocalConfig, "--output-root", $ArtifactRoot, "--ring-root", $RingRoot, "--device", "cuda", "--batch-size", $BatchSize)
& $RfdetrPython @arguments
if ($LASTEXITCODE -ne 0) { throw "Product realtime Mainline 1 stopped with exit $LASTEXITCODE." }
