[CmdletBinding()]
param(
    [string]$ProjectRoot = "C:\Users\developer\Documents\billiardsAI-raytheon3",
    [string]$PythonPath = "",
    [string]$RuntimeRoot = (Join-Path $env:LOCALAPPDATA "billiardsAI\field-runtime"),
    [string]$AlgorithmRoot = (Join-Path $env:LOCALAPPDATA "billiardsAI\algorithm-artifacts"),
    [string]$ApprovedAssetsRoot = (Join-Path $env:LOCALAPPDATA "billiardsAI\approved-assets"),
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
    param([Parameter(Mandatory=$true)][string]$Candidate, [switch]$RequireFile)
    if ($Candidate -match '(^|[\\/])\.\.([\\/]|$)') { throw "Parent traversal is not allowed in diagnostic paths." }
    $Full = [IO.Path]::GetFullPath($Candidate)
    if ($env:OS -eq "Windows_NT") {
        if ($Full -notmatch '^[A-Za-z]:\\' -or $Full.Substring(2).Contains(":")) {
            throw "Only local drive paths are allowed; UNC and device paths are rejected."
        }
    }
    $Root = [IO.Path]::GetPathRoot($Full)
    $Current = $Root
    $Remainder = $Full.Substring($Root.Length)
    foreach ($Segment in ($Remainder -split '[\\/]' | Where-Object { $_ })) {
        $Current = Join-Path $Current $Segment
        if (-not (Test-Path -LiteralPath $Current)) { break }
        $Item = Get-Item -LiteralPath $Current -Force
        if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Reparse point is not allowed in diagnostic input path: $Current"
        }
    }
    if ($RequireFile -and -not (Test-Path -LiteralPath $Full -PathType Leaf)) {
        throw "Required local file not found: $Full"
    }
}

Assert-LocalPathWithoutReparse -Candidate $PSScriptRoot
Assert-LocalPathWithoutReparse -Candidate $OutputParent
$Runner = Join-Path $PSScriptRoot "inventory_check.py"
$RunName = "raytheon-asset-inventory-result-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + [Guid]::NewGuid().ToString("N").Substring(0, 6)
$Output = Join-Path ([IO.Path]::GetFullPath($OutputParent)) $RunName
New-Item -ItemType Directory -Path $Output | Out-Null

try {
    Assert-LocalPathWithoutReparse -Candidate $ProjectRoot
    Assert-LocalPathWithoutReparse -Candidate $RuntimeRoot
    Assert-LocalPathWithoutReparse -Candidate $AlgorithmRoot
    Assert-LocalPathWithoutReparse -Candidate $ApprovedAssetsRoot
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
    if ([string]::IsNullOrWhiteSpace($PythonPath)) {
        $Candidates = @()
        $Pointer = Join-Path $RuntimeRoot "current.json"
        if (Test-Path -LiteralPath $Pointer -PathType Leaf) {
            Assert-LocalPathWithoutReparse -Candidate $Pointer -RequireFile
            $Current = Get-Content -LiteralPath $Pointer -Raw | ConvertFrom-Json
            if ($null -ne $Current.PSObject.Properties["python_path"] -and $Current.python_path) { $Candidates += [string]$Current.python_path }
        }
        $Candidates += (Join-Path $ProjectRoot ".venv-rfdetr\Scripts\python.exe")
        $Candidates += (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
        foreach ($Candidate in $Candidates) {
            try {
                Assert-LocalPathWithoutReparse -Candidate $Candidate
                if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
                    $PythonPath = $Candidate
                    break
                }
            } catch {
                # Never probe a rejected UNC, device or reparse-point candidate.
            }
        }
    }
    Assert-LocalPathWithoutReparse -Candidate $PythonPath -RequireFile
    Write-Host "Read-only asset inventory; services, calibration, models and state data will not be changed."
    Write-Host "Result folder: $Output"
    & $PythonPath $Runner --output $Output --project-root $ProjectRoot --runtime-root $RuntimeRoot --algorithm-root $AlgorithmRoot --approved-assets-root $ApprovedAssetsRoot
    $Code = $LASTEXITCODE
    $Zip = Join-Path (Split-Path -Parent $Output) ("SEND_BACK-" + $RunName + ".zip")
    if (-not (Test-Path -LiteralPath $Zip -PathType Leaf)) { throw "Inventory exited before producing a result ZIP (exit code $Code)" }
    Write-Host ""
    Write-Host "SEND THIS ZIP BACK: $Zip" -ForegroundColor Green
    Write-Host "Result SHA256: $((Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToLowerInvariant())"
    if ($Code -ne 0) { Write-Host "Some read-only checks were partial. Send the ZIP back anyway." -ForegroundColor Yellow }
} catch {
    $Failure = [ordered]@{ status="BOOTSTRAP_FAILED"; acceptance="NOT_EVALUATED"; production_ready=$false; error=$_.Exception.Message }
    $Failure | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Output "bootstrap-error.json") -Encoding UTF8
    $Zip = Join-Path (Split-Path -Parent $Output) ("SEND_BACK-" + $RunName + "-bootstrap.zip")
    Compress-Archive -Path $Output -DestinationPath $Zip
    Write-Host "Inventory could not finish: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "SEND THIS ZIP BACK: $Zip" -ForegroundColor Green
}
