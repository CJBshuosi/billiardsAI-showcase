[CmdletBinding()]
param(
  [string]$BrowserPath = "",
  [string]$RuntimeRoot = (Join-Path $env:LOCALAPPDATA "billiardsAI\field-runtime"),
  [ValidateRange(1, 600)][int]$HealthTimeoutSeconds = 60,
  [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScreenUrl = "http://127.0.0.1:8787/screen"
$HealthUrl = "http://127.0.0.1:8787/api/screen"
$ProfileRoot = Join-Path ([System.IO.Path]::GetFullPath($RuntimeRoot)) "kiosk-profile"

if (-not $BrowserPath) {
  $Candidates = @(
    (Join-Path ${env:ProgramFiles(x86)} "Microsoft\Edge\Application\msedge.exe"),
    (Join-Path $env:ProgramFiles "Microsoft\Edge\Application\msedge.exe"),
    (Join-Path $env:ProgramFiles "Google\Chrome\Application\chrome.exe")
  )
  $BrowserPath = $Candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } | Select-Object -First 1
}
if (-not $BrowserPath -or -not (Test-Path -LiteralPath $BrowserPath -PathType Leaf)) {
  throw "Edge or Chrome executable was not found"
}
$ResolvedBrowser = (Resolve-Path -LiteralPath $BrowserPath).Path

if ($CheckOnly) {
  [pscustomobject]@{ status = "READY"; browser = $ResolvedBrowser; screen_url = $ScreenUrl; profile = $ProfileRoot } | ConvertTo-Json -Compress
  exit 0
}

New-Item -ItemType Directory -Path $ProfileRoot -Force | Out-Null
$ResidualDeadline = [DateTime]::UtcNow.AddSeconds(5)
do {
  $ResidualBrowsers = @(
    Get-CimInstance Win32_Process | Where-Object {
      $_.CommandLine -like "*--user-data-dir=$ProfileRoot*"
    }
  )
  foreach ($Process in $ResidualBrowsers) {
    Invoke-CimMethod -InputObject $Process -MethodName Terminate -ErrorAction SilentlyContinue | Out-Null
  }
  if ($ResidualBrowsers.Count -eq 0) { break }
  Start-Sleep -Milliseconds 250
} while ([DateTime]::UtcNow -lt $ResidualDeadline)
$ResidualBrowsers = @(
  Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*--user-data-dir=$ProfileRoot*"
  }
)
if ($ResidualBrowsers.Count -ne 0) {
  throw "Residual kiosk browser did not terminate: $ProfileRoot"
}
$Deadline = [DateTime]::UtcNow.AddSeconds($HealthTimeoutSeconds)
while ([DateTime]::UtcNow -lt $Deadline) {
  try {
    $Response = Invoke-WebRequest -UseBasicParsing -Uri $HealthUrl -TimeoutSec 2
    if ($Response.StatusCode -eq 200) { break }
  } catch {
    Start-Sleep -Seconds 1
  }
}
if ([DateTime]::UtcNow -ge $Deadline) {
  throw "Field service health check did not become ready: $HealthUrl"
}

$BrowserArguments = @("--kiosk", $ScreenUrl, "--user-data-dir=$ProfileRoot", "--no-first-run", "--disable-session-crashed-bubble")
while ($true) {
  $Process = Start-Process -FilePath $ResolvedBrowser -ArgumentList $BrowserArguments -PassThru
  $Process.WaitForExit()
  Start-Sleep -Seconds 2
}
