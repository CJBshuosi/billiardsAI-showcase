[CmdletBinding()]
param(
    [string]$ProjectRoot = "C:\Users\developer\Documents\billiardsAI-raytheon3",
    [string]$PythonPath = "",
    [string]$RuntimeRoot = (Join-Path $env:LOCALAPPDATA "billiardsAI\field-runtime"),
    [string]$OutputParent = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS = "0"
$ProgressPreference = "SilentlyContinue"
if ([string]::IsNullOrWhiteSpace($OutputParent)) { $OutputParent = $PSScriptRoot }
$Runner = Join-Path $PSScriptRoot "camera_check.py"
$RunName = "camera-result-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + [Guid]::NewGuid().ToString("N").Substring(0, 6)
$Output = Join-Path ([IO.Path]::GetFullPath($OutputParent)) $RunName
New-Item -ItemType Directory -Path $Output | Out-Null

try {
    $ManifestPath = Join-Path $PSScriptRoot "packet-manifest.json"
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    foreach ($Entry in $Manifest.files.PSObject.Properties) {
        $Name = [string]$Entry.Name
        if ([IO.Path]::GetFileName($Name) -ne $Name) { throw "Unsafe packet inventory path" }
        $Path = Join-Path $PSScriptRoot $Name
        $Actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Actual -ne [string]$Entry.Value.sha256) { throw "Packet file hash mismatch: $Name" }
    }
    Copy-Item -LiteralPath $ManifestPath -Destination (Join-Path $Output "packet-manifest.json")
    $Candidates = @()
    if ($PythonPath) {
        $Candidates = @($PythonPath)
    } else {
        $Pointer = Join-Path $RuntimeRoot "current.json"
        if (Test-Path -LiteralPath $Pointer -PathType Leaf) {
            $Current = Get-Content -LiteralPath $Pointer -Raw | ConvertFrom-Json
            if ($null -ne $Current.PSObject.Properties["python_path"] -and $Current.python_path) {
                $Candidates += [string]$Current.python_path
            }
        }
        $Candidates += (Join-Path $ProjectRoot ".venv-rfdetr\Scripts\python.exe")
        $Candidates += (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
    }
    $Selected = $Candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $Selected) {
        throw "Python not found. Rerun with -PythonPath pointing to the existing .venv-rfdetr\Scripts\python.exe. Nothing was installed."
    }
    Write-Host "Camera diagnostic only; no service upgrade or business-data writes."
    Write-Host "V4: DirectShow 1080p60 MJPG: compare manual exposure -7 and -8. NO 30-minute test."
    Write-Host "Each exposure: 10 seconds preparation after camera setup, then 30 seconds recording."
    Write-Host "Change exposure ONLY when prompted; autofocus and auto white balance remain OFF."
    Write-Host "Close AMCap, Windows Camera and OBS. Keep the new camera connected."
    Write-Host "Result folder: $Output"
    Read-Host "Press ENTER when ready" | Out-Null
    & $Selected -u $Runner --output $Output --project-root $ProjectRoot --runtime-root $RuntimeRoot
    $Code = $LASTEXITCODE
    $Zip = Join-Path (Split-Path -Parent $Output) ("SEND_BACK-" + $RunName + ".zip")
    if (-not (Test-Path -LiteralPath $Zip -PathType Leaf)) {
        throw "Python exited before producing a result ZIP (exit code $Code)"
    }
    Write-Host ""
    Write-Host "SEND THIS ZIP BACK: $Zip" -ForegroundColor Green
    Write-Host "Result SHA256: $((Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToLowerInvariant())"
    if ($Code -ne 0) { Write-Host "Some diagnostics failed. Keep the failure ZIP and send it back." -ForegroundColor Yellow }
} catch {
    $Failure = [ordered]@{ status="BOOTSTRAP_FAILED"; acceptance="NOT_EVALUATED"; error=$_.Exception.Message; project_root=$ProjectRoot; runtime_root=$RuntimeRoot }
    $Failure | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Output "bootstrap-error.json") -Encoding UTF8
    $Zip = Join-Path (Split-Path -Parent $Output) ("SEND_BACK-" + $RunName + "-bootstrap.zip")
    Compress-Archive -Path $Output -DestinationPath $Zip
    Write-Host "Diagnostic could not start: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "SEND THIS ZIP BACK: $Zip" -ForegroundColor Green
}
