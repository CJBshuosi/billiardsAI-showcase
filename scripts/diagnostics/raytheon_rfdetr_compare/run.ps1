[CmdletBinding()]
param(
    [string]$PythonPath = "C:\Users\developer\Documents\billiardsAI-raytheon3\.venv-rfdetr\Scripts\python.exe",
    [string]$DownloadsRoot = (Join-Path $env:USERPROFILE "Downloads"),
    [string]$AlgorithmRoot = (Join-Path $env:LOCALAPPDATA "billiardsAI\algorithm-artifacts"),
    [string]$OutputParent = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"
$ProgressPreference = "SilentlyContinue"
if ([string]::IsNullOrWhiteSpace($OutputParent)) { $OutputParent = $PSScriptRoot }

function Assert-LocalPathWithoutReparse {
    param([Parameter(Mandatory=$true)][string]$Candidate, [switch]$RequireFile, [switch]$RequireDirectory)
    if ($Candidate -match '(^|[\\/])\.\.([\\/]|$)') { throw "Parent traversal is not allowed in diagnostic paths." }
    $Full = [IO.Path]::GetFullPath($Candidate)
    if ($env:OS -eq "Windows_NT") {
        if ($Full -notmatch '^[A-Za-z]:\\' -or $Full.Substring(2).Contains(":")) {
            throw "Only local drive paths are allowed; UNC and device paths are rejected."
        }
    }
    $Root = [IO.Path]::GetPathRoot($Full)
    $Current = $Root
    foreach ($Segment in ($Full.Substring($Root.Length) -split '[\\/]' | Where-Object { $_ })) {
        $Current = Join-Path $Current $Segment
        if (-not (Test-Path -LiteralPath $Current)) { break }
        $Item = Get-Item -LiteralPath $Current -Force
        if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Reparse point is not allowed in diagnostic input path: $Current"
        }
    }
    if ($RequireFile -and -not (Test-Path -LiteralPath $Full -PathType Leaf)) { throw "Required file not found: $Full" }
    if ($RequireDirectory -and -not (Test-Path -LiteralPath $Full -PathType Container)) { throw "Required directory not found: $Full" }
}

Assert-LocalPathWithoutReparse -Candidate $PSScriptRoot -RequireDirectory
Assert-LocalPathWithoutReparse -Candidate $OutputParent -RequireDirectory
$Runner = Join-Path $PSScriptRoot "rfdetr_compare.py"
$RunName = "raytheon-rfdetr-compare-result-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + [Guid]::NewGuid().ToString("N").Substring(0, 6)
$Output = Join-Path ([IO.Path]::GetFullPath($OutputParent)) $RunName
New-Item -ItemType Directory -Path $Output | Out-Null

try {
    Assert-LocalPathWithoutReparse -Candidate $PythonPath -RequireFile
    Assert-LocalPathWithoutReparse -Candidate $DownloadsRoot -RequireDirectory
    Assert-LocalPathWithoutReparse -Candidate $AlgorithmRoot -RequireDirectory
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
    Write-Host "Offline RF-DETR comparison only; identity heads and official field stores are not used."
    Write-Host "The two verified local checkpoints will process the same frozen frames."
    Write-Host "This can take several minutes. Progress is written after every 10 selected frames."
    Write-Host "Result folder: $Output"
    & $PythonPath $Runner --output $Output --downloads-root $DownloadsRoot --algorithm-root $AlgorithmRoot
    $Code = $LASTEXITCODE
    $Zip = Join-Path (Split-Path -Parent $Output) ("SEND_BACK-" + $RunName + ".zip")
    if (-not (Test-Path -LiteralPath $Zip -PathType Leaf)) { throw "Comparison exited before producing a result ZIP (exit code $Code)" }
    Write-Host ""
    Write-Host "SEND THIS ZIP BACK: $Zip" -ForegroundColor Green
    Write-Host "Result SHA256: $((Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToLowerInvariant())"
    if ($Code -ne 0) { Write-Host "One or more diagnostics failed. Send the ZIP back anyway." -ForegroundColor Yellow }
} catch {
    $Failure = [ordered]@{ status="BOOTSTRAP_FAILED"; acceptance="NOT_EVALUATED"; identity_evaluated=$false; official_results_written=$false; production_ready=$false; error=$_.Exception.Message }
    $Failure | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Output "bootstrap-error.json") -Encoding UTF8
    $Zip = Join-Path (Split-Path -Parent $Output) ("SEND_BACK-" + $RunName + "-bootstrap.zip")
    Compress-Archive -Path $Output -DestinationPath $Zip
    Write-Host "Comparison could not finish: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "SEND THIS ZIP BACK: $Zip" -ForegroundColor Green
}
