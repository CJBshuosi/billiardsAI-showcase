[CmdletBinding()]
param(
  [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$TaskNames = @("billiardsAI-field-service", "billiardsAI-field-kiosk")
$RuntimeRoot = Join-Path $env:LOCALAPPDATA "billiardsAI\field-runtime"

if ($CheckOnly) {
  foreach ($TaskName in $TaskNames) {
    $Exists = $null -ne (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue)
    [pscustomobject]@{ task = $TaskName; exists = $Exists; action = "would_unregister" }
  }
  Write-Host "Runtime data will be preserved at $RuntimeRoot"
  exit 0
}

foreach ($TaskName in $TaskNames) {
  $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if ($null -ne $Task) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
  }
}
Write-Host "PASS: scheduled tasks removed; runtime data preserved at $RuntimeRoot"
