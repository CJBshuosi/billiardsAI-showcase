[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$probePath = $null

try {
    $resolvedProject = [IO.Path]::GetFullPath($ProjectRoot)
    $helper = Join-Path $resolvedProject "src\billiards_ai\lab\raytheon_handoff.py"
    $venvRoot = Join-Path $resolvedProject ".venv-rfdetr"
    $venvPython = Join-Path $venvRoot "Scripts\python.exe"
    if (-not (Get-Command py -ErrorAction SilentlyContinue) -or
        -not (Test-Path -LiteralPath $helper -PathType Leaf) -or
        -not (Test-Path -LiteralPath (Join-Path $resolvedProject "pyproject.toml") -PathType Leaf)) {
        throw "ProjectRoot or py -3.12 cannot be verified; refusing uninstall."
    }
    if ((Split-Path -Leaf $venvRoot) -ne ".venv-rfdetr" -or
        -not (Test-Path -LiteralPath (Join-Path $venvRoot "pyvenv.cfg") -PathType Leaf) -or
        -not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        throw "Target is not a complete project-owned .venv-rfdetr; refusing uninstall."
    }
    $venvItem = Get-Item -LiteralPath $venvRoot
    if (($venvItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw ".venv-rfdetr is a reparse point; refusing uninstall."
    }

    $probePath = Join-Path ([IO.Path]::GetTempPath()) ("rfdetr-uninstall-{0}.json" -f [Guid]::NewGuid().ToString("N"))
    $probeRaw = & $venvPython $helper interpreter-probe
    if ($LASTEXITCODE -ne 0) { throw "Could not read target venv identity; refusing uninstall." }
    $probeRaw | Set-Content -LiteralPath $probePath -Encoding UTF8
    & py -3.12 $helper validate-uninstall --project-root $resolvedProject --venv $venvRoot --probe $probePath
    if ($LASTEXITCODE -ne 0) { throw "Cross-platform uninstall contract validation failed; refusing uninstall." }

    Remove-Item -LiteralPath $venvRoot -Recurse -Force
    Write-Host "Removed verified project-owned venv: $venvRoot"
}
catch {
    [Console]::Error.WriteLine("Uninstall stopped: $($_.Exception.Message)")
    exit 1
}
finally {
    if ($null -ne $probePath -and (Test-Path -LiteralPath $probePath -PathType Leaf)) {
        Remove-Item -LiteralPath $probePath -Force
    }
}
