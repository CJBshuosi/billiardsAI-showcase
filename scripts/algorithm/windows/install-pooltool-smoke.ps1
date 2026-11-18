param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [Parameter(Mandatory = $true)][string]$Packet,
    [Parameter(Mandatory = $true)][string]$PacketSha256,
    [Parameter(Mandatory = $true)][string]$RfdetrPython
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$Packet = (Resolve-Path -LiteralPath $Packet).Path
$RfdetrPython = (Resolve-Path -LiteralPath $RfdetrPython).Path
& py -3.12 --version
if ($LASTEXITCODE -ne 0) {
    throw "64-bit CPython 3.12 is required."
}

$ArtifactRoot = Join-Path $env:LOCALAPPDATA "billiardsAI\algorithm-artifacts\pooltool-smoke"
$BundledWheelhouse = Join-Path $ProjectRoot "_handoff\wheelhouse"
$Wheelhouse = if (Test-Path -LiteralPath $BundledWheelhouse -PathType Container) { $BundledWheelhouse } else { Join-Path $ArtifactRoot ("wheelhouse-" + $PacketSha256) }
$Venv = Join-Path $ProjectRoot ".venv-pooltool"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$PooltoolSitePackages = Join-Path $Venv "Lib\site-packages"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

& py -3.12 (Join-Path $ProjectRoot "scripts\algorithm\pooltool_packet_handoff.py") --packet $Packet --packet-sha256 $PacketSha256 --wheelhouse $Wheelhouse
if ($LASTEXITCODE -ne 0) { throw "Approved pooltool packet download failed." }

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    & py -3.12 -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "Pooltool virtual environment creation failed." }
}

& $VenvPython -m pip install --no-index --find-links $Wheelhouse "pooltool-billiards==0.6.0"
if ($LASTEXITCODE -ne 0) { throw "Offline pooltool installation failed." }

$NativeSmokeRoot = Join-Path $ArtifactRoot "native-runs"
& $VenvPython (Join-Path $ProjectRoot "scripts\algorithm\pooltool_native_smoke.py") --output-root $NativeSmokeRoot
if ($LASTEXITCODE -ne 0) { throw "Native pooltool synthetic smoke failed." }

if (-not (Test-Path -LiteralPath $RfdetrPython -PathType Leaf)) {
    throw "Existing .venv-rfdetr Python is required for the project adapter smoke."
}
$env:PYTHONPATH = $PooltoolSitePackages + ";" + (Join-Path $ProjectRoot "src")
$AdapterSmokeRoot = Join-Path $ArtifactRoot "adapter-runs"
& $RfdetrPython (Join-Path $ProjectRoot "scripts\algorithm\pooltool_real_smoke.py") --output-root $AdapterSmokeRoot
if ($LASTEXITCODE -ne 0) { throw "Project pooltool adapter synthetic smoke failed." }

$GuidanceSmokeRoot = Join-Path $ArtifactRoot "guidance-runs"
$GuidanceOutput = & $RfdetrPython (Join-Path $ProjectRoot "scripts\algorithm\run_mainline1_guidance.py") "--table-state" (Join-Path $ProjectRoot "tests\fixtures\guidance\table-state.json") "--context-group" "solid" "--physics" "pooltool" "--output-root" $GuidanceSmokeRoot
$GuidanceExit = $LASTEXITCODE
$GuidanceOutput | ForEach-Object { Write-Output $_ }
if ($GuidanceExit -ne 0) { throw "Mainline 1 real pooltool guidance smoke failed." }
$GuidanceSummary = $GuidanceOutput | Select-Object -Last 1 | ConvertFrom-Json
if ($GuidanceSummary.recommendation_status -ne "simulated" -or $GuidanceSummary.candidate_count -ne 3 -or $GuidanceSummary.physics_scored_count -ne 3 -or $GuidanceSummary.backend_versions -notcontains "pooltool/0.6.0") {
    throw "Mainline 1 real pooltool guidance smoke did not prove three physics-ranked candidates."
}
