#requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet("Start", "EndSession", "Worker")][string]$Mode = "Start",
    [string]$ResultRoot = "",
    [string]$PythonPath = "C:\Users\developer\Documents\billiardsAI-raytheon3\.venv-rfdetr\Scripts\python.exe",
    [string]$OutputParent = "",
    [string]$HostAddress = "0.0.0.0",
    [string]$PublicHost = "",
    [ValidateRange(0, 65535)][int]$Port = 8765,
    [ValidateSet("cuda", "cpu")][string]$Device = "cuda",
    [ValidateRange(1, 64)][int]$BatchSize = 8,
    [double]$Fps = 60.0,
    [string]$SyncBaseUrl = "",
    [string]$DeviceId = "",
    [string]$CredentialTarget = "",
    [ValidateSet("development")][string]$RuntimeEnvironment = "development",
    [string]$CloudbaseEnvironmentId = "",
    [string]$JoinQrAsset = "",
    [switch]$ObservationOnly,
    [switch]$LocalOnly,
    [switch]$ExistingDevelopmentControlOnly,
    [switch]$KeepRunning,
    [string]$WorkerSessionId = "",
    [string]$WorkerIdentity = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$RuleProfile = "cbsa-chinese-pool-2017-observable-v1"
$PacketRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$ManifestPath = Join-Path $PacketRoot "packet-manifest.json"

function Assert-FileHash {
    param([string]$Path, [string]$Expected)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file is missing: $Path"
    }
    $Actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Expected.ToLowerInvariant()) { throw "File hash mismatch: $Path" }
}

function Read-JsonFile {
    param([string]$Path)
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Write-JsonFile {
    param([object]$Value, [string]$Path)
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Write-JsonFileAtomic {
    param([object]$Value, [string]$Path)
    $Temporary = $Path + ".tmp-" + [Guid]::NewGuid().ToString("N")
    try {
        $Value | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $Temporary -Encoding UTF8
        Move-Item -LiteralPath $Temporary -Destination $Path -Force
    } finally {
        if (Test-Path -LiteralPath $Temporary) { Remove-Item -LiteralPath $Temporary -Force }
    }
}

function Update-RunStateLocked {
    param([string]$StatePath, [scriptblock]$Update)
    $LockPath = $StatePath + ".lock"
    $Lock = $null
    $LockDeadline = [DateTimeOffset]::UtcNow.AddSeconds(10)
    while ($null -eq $Lock -and [DateTimeOffset]::UtcNow -lt $LockDeadline) {
        try {
            $Lock = [IO.File]::Open($LockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
        } catch [IO.IOException] {
            Start-Sleep -Milliseconds 50
        }
    }
    if ($null -eq $Lock) { throw "Timed out acquiring the run-state lock." }
    try {
        $Current = Read-JsonFile $StatePath
        & $Update $Current
        Write-JsonFileAtomic $Current $StatePath
        return $Current
    } finally {
        $Lock.Dispose()
    }
}

function Get-StringSha256 {
    param([string]$Value)
    $Sha = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($Sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($Value)))).Replace("-", "").ToLowerInvariant()
    } finally {
        $Sha.Dispose()
    }
}

function Protect-PrivateFile {
    param([string]$Path)
    $Identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    if ($null -eq $Identity.User) { throw "Current Windows user SID is unavailable." }
    $Security = New-Object Security.AccessControl.FileSecurity
    $Security.SetOwner($Identity.User)
    $Security.SetAccessRuleProtection($true, $false)
    $Rights = [Security.AccessControl.FileSystemRights]::Read -bor [Security.AccessControl.FileSystemRights]::Write
    $Rule = [Security.AccessControl.FileSystemAccessRule]::new(
        $Identity.User, $Rights, [Security.AccessControl.AccessControlType]::Allow
    )
    [void]$Security.AddAccessRule($Rule)
    [IO.File]::SetAccessControl($Path, $Security)
}

function ConvertTo-WindowsProcessArgument {
    param([AllowNull()][AllowEmptyString()][string]$Value)
    if ($null -eq $Value) { $Value = "" }
    $Builder = New-Object System.Text.StringBuilder
    [void]$Builder.Append([char]34)
    $Backslashes = 0
    for ($Index = 0; $Index -lt $Value.Length; $Index++) {
        $Character = $Value[$Index]
        if ([int][char]$Character -eq 92) {
            $Backslashes += 1
            continue
        }
        if ([int][char]$Character -eq 34) {
            for ($Count = 0; $Count -lt (2 * $Backslashes + 1); $Count++) {
                [void]$Builder.Append([char]92)
            }
        } else {
            for ($Count = 0; $Count -lt $Backslashes; $Count++) {
                [void]$Builder.Append([char]92)
            }
        }
        $Backslashes = 0
        [void]$Builder.Append($Character)
    }
    for ($Count = 0; $Count -lt (2 * $Backslashes); $Count++) {
        [void]$Builder.Append([char]92)
    }
    [void]$Builder.Append([char]34)
    return $Builder.ToString()
}

function Join-WindowsProcessArguments {
    param([object[]]$Values)
    return (@($Values | ForEach-Object {
        ConvertTo-WindowsProcessArgument ([string]$_)
    }) -join " ")
}

function Resolve-PublicHost {
    param([string]$Requested)
    if (-not [string]::IsNullOrWhiteSpace($Requested)) { return $Requested }
    $Address = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object {
            $_.IPAddress -ne "127.0.0.1" -and
            $_.AddressState -eq "Preferred" -and
            $_.PrefixOrigin -ne "WellKnown"
        } |
        Select-Object -First 1 -ExpandProperty IPAddress
    if ([string]::IsNullOrWhiteSpace($Address)) { return "127.0.0.1" }
    return $Address
}

function Resolve-VerifiedQrFile {
    param([string]$Path, [string]$Source)
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Source join QR asset is missing."
    }
    $Item = Get-Item -LiteralPath $Path
    if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "$Source join QR asset must not be a link."
    }
    $Resolved = (Resolve-Path -LiteralPath $Path).Path
    if ([IO.Path]::GetExtension($Resolved).ToLowerInvariant() -notin @(".png", ".jpg", ".jpeg")) {
        throw "$Source join QR asset must be PNG or JPEG."
    }
    $Bytes = [IO.File]::ReadAllBytes($Resolved)
    $Png = $Bytes.Length -ge 8 -and $Bytes[0] -eq 0x89 -and $Bytes[1] -eq 0x50 -and $Bytes[2] -eq 0x4e -and $Bytes[3] -eq 0x47 -and $Bytes[4] -eq 0x0d -and $Bytes[5] -eq 0x0a -and $Bytes[6] -eq 0x1a -and $Bytes[7] -eq 0x0a
    $Jpeg = $Bytes.Length -ge 3 -and $Bytes[0] -eq 0xff -and $Bytes[1] -eq 0xd8 -and $Bytes[2] -eq 0xff
    if (-not $Png -and -not $Jpeg) { throw "$Source join QR asset content is invalid." }
    return $Resolved
}

function Read-ExistingFieldRuntime {
    $CurrentPointer = Join-Path $env:LOCALAPPDATA "billiardsAI\field-runtime\current.json"
    if (-not (Test-Path -LiteralPath $CurrentPointer -PathType Leaf)) { return $null }
    $PointerItem = Get-Item -LiteralPath $CurrentPointer
    if (($PointerItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Existing field-runtime current.json must not be a link."
    }
    $Current = Read-JsonFile $CurrentPointer
    $Current | Add-Member -NotePropertyName _current_pointer_path -NotePropertyValue $CurrentPointer -Force
    $Current | Add-Member -NotePropertyName _current_pointer_sha256 -NotePropertyValue ((Get-FileHash -LiteralPath $CurrentPointer -Algorithm SHA256).Hash.ToLowerInvariant()) -Force
    return $Current
}

function Assert-ExistingFieldServiceStopped {
    $Task = Get-ScheduledTask -TaskName "billiardsAI-field-service" -ErrorAction SilentlyContinue
    if ($null -ne $Task -and [string]$Task.State -eq "Running") {
        throw "Installed field service is still running; stop it before control-only testing."
    }
    $LegacyProcesses = @(
        Get-CimInstance Win32_Process -ErrorAction Stop |
            Where-Object { [string]$_.CommandLine -like "*serve_field_route_shell.py*" }
    )
    if ($LegacyProcesses.Count -ne 0) {
        throw "A legacy field service process is still running."
    }
    $Listeners = @(
        Get-NetTCPConnection -State Listen -ErrorAction Stop |
            Where-Object { $_.LocalPort -eq 8787 }
    )
    if ($Listeners.Count -ne 0) {
        throw "Installed field service port 8787 is still listening."
    }
}

function Stop-PreviousUsbFieldPackets {
    $Deadline = [DateTimeOffset]::UtcNow.AddSeconds(15)
    do {
        $Processes = @(
            Get-CimInstance Win32_Process -ErrorAction Stop |
                Where-Object {
                    $CommandLine = [string]$_.CommandLine
                    $CommandLine -and
                    $CommandLine.IndexOf("serve_mainline1_usb_field.py", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                    $CommandLine.IndexOf("billiardsAI-raytheon-eight-ball-field-", [StringComparison]::OrdinalIgnoreCase) -ge 0
                }
        )
        foreach ($Process in $Processes) {
            Invoke-CimMethod -InputObject $Process -MethodName Terminate -ErrorAction Stop | Out-Null
        }
        if ($Processes.Count -eq 0) { return }
        Start-Sleep -Milliseconds 250
    } while ([DateTimeOffset]::UtcNow -lt $Deadline)
    throw "A previous USB field packet process did not stop within timeout."
}

function Invoke-ExistingSessionRetirement {
    param([object]$ResolvedSync, [string]$EvidenceRoot, [string]$ResultParent)
    $InstalledStateDb = Join-Path $env:LOCALAPPDATA "billiardsAI\field-runtime\field-state.sqlite3"
    $Retirer = Join-Path $PacketRoot "retire_existing_session.py"
    if (-not (Test-Path -LiteralPath $Retirer -PathType Leaf)) {
        throw "Installed-session retirement helper is missing."
    }
    $PriorPacketStateDbs = @(
        Get-ChildItem -LiteralPath $ResultParent -Directory -Filter "eight-ball-field-result-*" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTimeUtc -Descending |
            ForEach-Object { Join-Path $_.FullName "work\field-state.sqlite3" } |
            Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
    )
    $Candidates = @($PriorPacketStateDbs)
    if (Test-Path -LiteralPath $InstalledStateDb -PathType Leaf) {
        $Candidates += $InstalledStateDb
    }
    if ($Candidates.Count -eq 0) {
        throw "No prior development field state database is available for safe retirement."
    }
    $Failures = @()
    $LastNotNeeded = $null
    for ($Index = 0; $Index -lt $Candidates.Count; $Index += 1) {
        $Receipt = Join-Path $EvidenceRoot ("existing-session-retirement-" + $Index + ".json")
        $RetireArguments = @(
            "-u", $Retirer,
            "--state-db", [string]$Candidates[$Index],
            "--sync-base-url", [string]$ResolvedSync.sync_base_url,
            "--device-id", [string]$ResolvedSync.device_id,
            "--credential-target", [string]$ResolvedSync.credential_target,
            "--receipt", $Receipt
        )
        $Output = (& $PythonPath @RetireArguments 2>&1 | Out-String).Trim()
        Add-Content -LiteralPath (Join-Path $EvidenceRoot "existing-session-retirement.log") -Value $Output -Encoding UTF8
        if ($LASTEXITCODE -ne 0) {
            $Failures += $Output
            continue
        }
        $Result = Read-JsonFile $Receipt
        if ([string]$Result.status -notin @("RETIRED", "NOT_NEEDED")) {
            $Failures += "Existing development session retirement receipt is invalid."
            continue
        }
        if ([string]$Result.status -eq "RETIRED") {
            Copy-Item -LiteralPath $Receipt -Destination (Join-Path $EvidenceRoot "existing-session-retirement.json") -Force
            Write-Host "EXISTING SESSION: RETIRED" -ForegroundColor Green
            return $Result
        }
        $LastNotNeeded = $Result
    }
    if ($Failures.Count -ne 0) {
        throw ("Existing development session could not be retired: " + ($Failures -join " | "))
    }
    if ($null -eq $LastNotNeeded) {
        throw "Existing development session retirement produced no receipt."
    }
    Write-JsonFile $LastNotNeeded (Join-Path $EvidenceRoot "existing-session-retirement.json")
    Write-Host "EXISTING SESSION: NOT_NEEDED" -ForegroundColor Green
    return $LastNotNeeded
}

function Resolve-JoinQrAsset {
    param([object]$Manifest, [object]$ExistingRuntime)
    if ($ExistingDevelopmentControlOnly) {
        if (-not [string]::IsNullOrWhiteSpace($JoinQrAsset)) {
            throw "ExistingDevelopmentControlOnly rejects an explicit join QR override."
        }
        $PacketQr = [string]$Manifest.run.join_qr_asset
        if (-not [string]::IsNullOrWhiteSpace($PacketQr)) {
            $Candidate = [IO.Path]::GetFullPath((Join-Path $PacketRoot $PacketQr))
            if (-not $Candidate.StartsWith($PacketRoot, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Packet join QR asset escaped the packet root."
            }
            return Resolve-VerifiedQrFile $Candidate "packet"
        }
        if ($null -eq $ExistingRuntime) {
            throw "ExistingDevelopmentControlOnly requires existing field-runtime current.json."
        }
        $Existing = [string]$ExistingRuntime.join_qr_asset
        if (-not [IO.Path]::IsPathRooted($Existing)) {
            throw "Existing field-runtime join_qr_asset must be an absolute path."
        }
        return Resolve-VerifiedQrFile $Existing "existing field-runtime"
    }
    if (-not [string]::IsNullOrWhiteSpace($JoinQrAsset)) {
        return Resolve-VerifiedQrFile $JoinQrAsset "explicit"
    }
    $PacketQr = [string]$Manifest.run.join_qr_asset
    if (-not [string]::IsNullOrWhiteSpace($PacketQr)) {
        $Candidate = [IO.Path]::GetFullPath((Join-Path $PacketRoot $PacketQr))
        if (-not $Candidate.StartsWith($PacketRoot, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Packet join QR asset escaped the packet root."
        }
        return Resolve-VerifiedQrFile $Candidate "packet"
    }
    if ($LocalOnly) { return $null }
    if ($null -eq $ExistingRuntime) {
        throw "No explicit, packet, or existing field-runtime join QR asset is available."
    }
    $Existing = [string]$ExistingRuntime.join_qr_asset
    if (-not [IO.Path]::IsPathRooted($Existing)) {
        throw "Existing field-runtime join_qr_asset must be an absolute path."
    }
    return Resolve-VerifiedQrFile $Existing "existing field-runtime"
}

function Assert-DevelopmentCloudScope {
    param([object]$Resolved, [object]$Manifest)
    $ExpectedEnvironmentId = [string]$Manifest.run.development_environment_id
    if ([string]::IsNullOrWhiteSpace($ExpectedEnvironmentId)) {
        throw "Packet development environment ID is missing."
    }
    if ([string]$Resolved.environment -ne "development") {
        throw "Field packet may write only to the development environment."
    }
    if ([string]$Resolved.cloudbase_environment_id -ne $ExpectedEnvironmentId) {
        throw "CloudBase environment is not the packet-approved development environment."
    }
    if ([string]$Resolved.credential_target -notmatch '^billiardsAI/development/') {
        throw "Credential target is not development scoped."
    }
    $ExpectedCredentialTarget = "billiardsAI/development/" + [string]$Resolved.device_id
    if ([string]$Resolved.credential_target -ne $ExpectedCredentialTarget) {
        throw "Credential target is not scoped to the exact development device."
    }
    if ($ExistingDevelopmentControlOnly) {
        if (
            [string]$Resolved.sync_base_url -ne [string]$Manifest.run.existing_development_control_base_url -or
            [string]$Resolved.device_id -ne [string]$Manifest.run.existing_development_device_id -or
            [string]$Resolved.credential_target -ne [string]$Manifest.run.existing_development_credential_target
        ) {
            throw "Existing development control configuration is not exactly packet pinned."
        }
    }
    $Gateway = $null
    if (-not [Uri]::TryCreate([string]$Resolved.sync_base_url, [UriKind]::Absolute, [ref]$Gateway) -or $Gateway.Scheme -ne "https") {
        throw "Field sync base URL must be absolute HTTPS."
    }
    if (-not [string]::IsNullOrWhiteSpace($Gateway.UserInfo) -or -not [string]::IsNullOrWhiteSpace($Gateway.Query) -or -not [string]::IsNullOrWhiteSpace($Gateway.Fragment)) {
        throw "Field sync base URL must not contain credentials, query, or fragment."
    }
    $EndpointBinding = $Gateway.DnsSafeHost + $Gateway.AbsolutePath
    if ($EndpointBinding.IndexOf($ExpectedEnvironmentId, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
        throw "Field sync endpoint is not bound to the approved development environment."
    }
}

function Resolve-SyncConfiguration {
    param([object]$ExistingRuntime, [object]$Manifest)
    $Explicit = @(@($SyncBaseUrl, $DeviceId, $CredentialTarget) | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
    if ($Explicit.Count -ne 0 -and $Explicit.Count -ne 3) {
        throw "SyncBaseUrl, DeviceId, and CredentialTarget must all be explicit or all be omitted."
    }
    if ($ExistingDevelopmentControlOnly) {
        if ($Explicit.Count -ne 0 -or -not [string]::IsNullOrWhiteSpace($CloudbaseEnvironmentId)) {
            throw "ExistingDevelopmentControlOnly rejects explicit sync and cloud arguments."
        }
        if ($null -eq $ExistingRuntime) {
            throw "ExistingDevelopmentControlOnly requires existing field-runtime current.json."
        }
        $Resolved = [ordered]@{
            source = "existing_field_runtime_control_only"
            sync_base_url = [string]$ExistingRuntime.sync_base_url
            device_id = [string]$ExistingRuntime.device_id
            credential_target = [string]$ExistingRuntime.credential_target
            environment = "development"
            cloudbase_environment_id = [string]$Manifest.run.development_environment_id
        }
    } elseif ($Explicit.Count -eq 3) {
        $Resolved = [ordered]@{
            source = "explicit"
            sync_base_url = $SyncBaseUrl
            device_id = $DeviceId
            credential_target = $CredentialTarget
        }
    } else {
        if ($null -eq $ExistingRuntime) {
            throw "Existing field-runtime current.json is required when sync inputs are omitted."
        }
        $Resolved = [ordered]@{
            source = "existing_field_runtime"
            sync_base_url = [string]$ExistingRuntime.sync_base_url
            device_id = [string]$ExistingRuntime.device_id
            credential_target = [string]$ExistingRuntime.credential_target
        }
        if (@($Resolved.Values | Where-Object { [string]::IsNullOrWhiteSpace([string]$_) }).Count -ne 0) {
            throw "Existing field-runtime cloud configuration is incomplete."
        }
    }
    if (-not $ExistingDevelopmentControlOnly -and [string]::IsNullOrWhiteSpace($CloudbaseEnvironmentId)) {
        if ($null -eq $ExistingRuntime) {
            throw "CloudBase development environment ID is required."
        }
        $Resolved["environment"] = [string]$ExistingRuntime.environment
        $Resolved["cloudbase_environment_id"] = [string]$ExistingRuntime.cloudbase_environment_id
    } elseif (-not $ExistingDevelopmentControlOnly) {
        $Resolved["environment"] = $RuntimeEnvironment
        $Resolved["cloudbase_environment_id"] = $CloudbaseEnvironmentId
    }
    Assert-DevelopmentCloudScope ([pscustomobject]$Resolved) $Manifest
    return [pscustomobject]$Resolved
}

function Assert-FrozenRunState {
    param([object]$State, [object]$Manifest)
    if ($State.local_only -eq $true) {
        if ($State.observation_only -ne $true -or [string]$State.sync_source -ne "local_only") {
            throw "Frozen LocalOnly run must remain ObservationOnly and cloud-free."
        }
        foreach ($Value in @(
            $State.cloudbase_environment_id, $State.sync_base_url,
            $State.device_id, $State.credential_target
        )) {
            if (-not [string]::IsNullOrWhiteSpace([string]$Value)) {
                throw "Frozen LocalOnly run contains forbidden cloud configuration."
            }
        }
    } elseif ($State.existing_development_control_only -eq $true) {
        if (
            $State.observation_only -ne $true -or
            $State.analysis_sync_enabled -ne $false -or
            [string]$State.sync_source -ne "existing_field_runtime_control_only" -or
            [string]$State.join_qr_source -ne "existing_field_runtime" -or
            [string]$State.runtime_environment -ne "development" -or
            [string]$State.cloudbase_environment_id -ne [string]$Manifest.run.development_environment_id -or
            [string]$State.sync_base_url -ne [string]$Manifest.run.existing_development_control_base_url -or
            [string]$State.device_id -ne [string]$Manifest.run.existing_development_device_id -or
            [string]$State.credential_target -ne [string]$Manifest.run.existing_development_credential_target
        ) {
            throw "Frozen existing development control-only run is not exactly packet pinned."
        }
    } else {
        $FrozenCloud = [pscustomobject][ordered]@{
            environment = [string]$State.runtime_environment
            cloudbase_environment_id = [string]$State.cloudbase_environment_id
            sync_base_url = [string]$State.sync_base_url
            device_id = [string]$State.device_id
            credential_target = [string]$State.credential_target
        }
        Assert-DevelopmentCloudScope $FrozenCloud $Manifest
    }
    $HasQrAsset = -not [string]::IsNullOrWhiteSpace([string]$State.join_qr_asset)
    $HasQrHash = -not [string]::IsNullOrWhiteSpace([string]$State.join_qr_sha256)
    if ($HasQrAsset -ne $HasQrHash) {
        throw "Frozen join QR asset and hash must either both be present or both be absent."
    }
    if (-not $State.local_only -and -not $HasQrAsset) {
        throw "Frozen non-LocalOnly run is missing its required join QR."
    }
    if ([string]$State.identity_gate_status -ne [string]$Manifest.assets.identity.gate_status) {
        throw "Frozen run identity gate conflicts with the packet manifest."
    }
    if ($State.observation_only -isnot [bool]) {
        throw "Frozen run ObservationOnly value is invalid."
    }
    if (-not $State.observation_only -and [string]$State.identity_gate_status -ne "PASS") {
        throw "Frozen run cannot claim rule authority without identity PASS."
    }
}

function Verify-Packet {
    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        throw "Packet manifest is missing: $ManifestPath"
    }
    $Manifest = Read-JsonFile $ManifestPath
    if ($Manifest.production_ready -ne $false -or $Manifest.development_evidence_only -ne $true) {
        throw "This runner accepts only the development evidence packet."
    }
    if ($Manifest.native_helper.trust -ne "unsigned_source_compiled_at_runtime") {
        throw "Unsigned development helper trust declaration is missing."
    }
    if ($Manifest.run.rule_profile -ne $RuleProfile) {
        throw "Packet rule profile is not the approved fixed profile."
    }
    $DeclaredFiles = @{}
    foreach ($Entry in $Manifest.files.PSObject.Properties) {
        $Name = [string]$Entry.Name
        if ([IO.Path]::IsPathRooted($Name) -or $Name -match '(^|[\/])\.\.([\/]|$)') {
            throw "Unsafe packet inventory path: $Name"
        }
        $Path = [IO.Path]::GetFullPath((Join-Path $PacketRoot $Name))
        if (-not $Path.StartsWith($PacketRoot, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Packet path escaped root: $Name"
        }
        $DeclaredFiles[$Name.Replace('\', '/')] = $true
        Assert-FileHash $Path ([string]$Entry.Value.sha256)
    }
    foreach ($Item in Get-ChildItem -LiteralPath $PacketRoot -Recurse -File) {
        $Relative = $Item.FullName.Substring($PacketRoot.Length).TrimStart([char[]]@('\', '/')).Replace('\', '/')
        if ($Relative -ne "packet-manifest.json" -and -not $DeclaredFiles.ContainsKey($Relative)) {
            throw "Packet extraction contains an undeclared file: $Relative"
        }
    }
    return $Manifest
}

function Resolve-ExternalAssets {
    param([object]$Manifest)
    $AlgorithmRoot = Join-Path $env:LOCALAPPDATA "billiardsAI\algorithm-artifacts"
    $ApprovedRoot = Join-Path $env:LOCALAPPDATA "billiardsAI\approved-assets"
    $Detector = Join-Path $AlgorithmRoot ([string]$Manifest.assets.detector.relative_root)
    $Identity = Join-Path $AlgorithmRoot ([string]$Manifest.assets.identity.relative_root)
    $Dinov2 = Join-Path $ApprovedRoot ([string]$Manifest.assets.dinov2.relative_root)
    Assert-FileHash (Join-Path $Detector "training-manifest.json") ([string]$Manifest.assets.detector.manifest_sha256)
    Assert-FileHash (Join-Path $Detector "checkpoint_best.pth") ([string]$Manifest.assets.detector.checkpoint_sha256)
    Assert-FileHash (Join-Path $Identity "manifest.json") ([string]$Manifest.assets.identity.manifest_sha256)
    Assert-FileHash (Join-Path $Identity "group-head.onnx") ([string]$Manifest.assets.identity.head_sha256)
    Assert-FileHash (Join-Path $Identity "metrics.json") ([string]$Manifest.assets.identity.metrics_sha256)
    $IdentityManifest = Read-JsonFile (Join-Path $Identity "manifest.json")
    $IdentityGateStatus = [string]$IdentityManifest.gate_status
    if ($IdentityGateStatus -ne [string]$Manifest.assets.identity.gate_status) {
        throw "Identity gate status does not match the frozen packet manifest."
    }
    if ($IdentityGateStatus -ne "PASS" -and -not $ObservationOnly) {
        throw "The frozen identity gate is not PASS; automatic rule authority is blocked. Use -ObservationOnly only for non-acceptance diagnostics."
    }
    Assert-FileHash (Join-Path $Dinov2 "manifest.json") ([string]$Manifest.assets.dinov2.manifest_sha256)
    Assert-FileHash (Join-Path $Dinov2 "model.safetensors") ([string]$Manifest.assets.dinov2.model_sha256)
    Assert-FileHash (Join-Path $Dinov2 "config.json") ([string]$Manifest.assets.dinov2.config_sha256)
    Assert-FileHash (Join-Path $Dinov2 "preprocessor_config.json") ([string]$Manifest.assets.dinov2.preprocessor_sha256)
    return [ordered]@{
        detector = $Detector
        identity = $Identity
        identity_gate_status = $IdentityGateStatus
        dinov2 = $Dinov2
    }
}

function Invoke-EvidenceExport {
    param([object]$State, [string]$SessionId, [string]$Output)
    $Exporter = Join-Path $PacketRoot "export_evidence.py"
    $OutputText = & $PythonPath $Exporter --state-db $State.state_db --jobs-db $State.jobs_db --session-id $SessionId --output $Output 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) { throw "Evidence export failed: $OutputText" }
    return Read-JsonFile $Output
}

function Initialize-OperatorGateEvidence {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) { return }
    Write-JsonFile ([ordered]@{
        schema_version = "1.0"
        capture_30m = [ordered]@{
            status = "NOT_RUN"
            observed_at_utc = $null
            continuous_healthy_capture = $false
            evidence = @()
            reason = "Operator has not attested 30 minutes of continuous healthy capture."
        }
        disconnect_recovery = [ordered]@{
            status = "NOT_RUN"
            observed_at_utc = $null
            idle_disconnect_abstained = $false
            in_shot_disconnect_abstained = $false
            reconnect_ready = $false
            synthesized_shot_end = $null
            evidence = @()
            reason = "Operator has not completed both disconnect/recovery checks."
        }
    }) $Path
}

function New-FailedFieldGateReceipt {
    param([string]$Phase, [string]$Reason)
    $GateIds = @(
        "healthy_capture_30m", "rule_decision_latency_2s", "missing_fact_abstain",
        "disconnect_recovery", "settled_shots_10", "same_process_two_session_rollover",
        "permanent_qr_hash", "session_artifact_isolation", "postgame_120s",
        "development_cloud_publication"
    )
    return [ordered]@{
        schema_version = "1.0"
        generated_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
        phase = $Phase
        development_evidence_only = $true
        production_ready = $false
        gates = @($GateIds | ForEach-Object {
            [ordered]@{
                gate_id = $_
                required = $true
                status = "NOT_RUN"
                determination = "automatic"
                evidence = [ordered]@{}
                reason = $Reason
            }
        })
        all_required_pass = $false
        rule_authority_accepted = $false
        accepted = $false
    }
}

function Invoke-FieldGateEvaluation {
    param(
        [object]$State,
        [object[]]$SessionResults,
        [string]$Evidence,
        [ValidateSet("INTERIM", "FINAL")][string]$Phase
    )
    $Work = Join-Path ([string]$State.result_root) "work"
    $LaunchAuditPath = Join-Path $Evidence "launch-audit.json"
    $ServerExitPath = Join-Path $Work "server-exit.json"
    $OperatorPath = Join-Path $Work "operator-gate-evidence.json"
    $ContextPath = Join-Path $Work "field-gate-context.json"
    $OutputPath = Join-Path $Evidence "field-gates.json"
    try {
        $LaunchAudit = Read-JsonFile $LaunchAuditPath
        $ServerExit = if (Test-Path -LiteralPath $ServerExitPath -PathType Leaf) {
            Read-JsonFile $ServerExitPath
        } else { [ordered]@{} }
        $CurrentQrHash = $null
        if (-not [string]::IsNullOrWhiteSpace([string]$State.join_qr_asset) -and
            (Test-Path -LiteralPath ([string]$State.join_qr_asset) -PathType Leaf)) {
            $CurrentQrHash = (Get-FileHash -LiteralPath ([string]$State.join_qr_asset) -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        Write-JsonFile ([ordered]@{
            phase = $Phase
            state = $State
            launch_audit = $LaunchAudit
            server_exit = $ServerExit
            session_results = @($SessionResults)
            current_join_qr_sha256 = $CurrentQrHash
        }) $ContextPath
        $Evaluator = Join-Path $PacketRoot "field_gate_evaluator.py"
        $Arguments = @(
            $Evaluator, "--evaluate-field-gates", "--state-db", [string]$State.state_db,
            "--context", $ContextPath, "--operator-evidence", $OperatorPath,
            "--output", $OutputPath
        )
        $OutputText = & $PythonPath @Arguments 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $OutputPath -PathType Leaf)) {
            throw "Field gate evaluator failed: $OutputText"
        }
        return Read-JsonFile $OutputPath
    } catch {
        $Failure = New-FailedFieldGateReceipt $Phase ("field gate evaluation failed: " + $_.Exception.GetType().Name)
        Write-JsonFile $Failure $OutputPath
        return $Failure
    }
}

function Invoke-WorkerPublicationInspection {
    param([object]$State, [string]$SessionId, [string]$Evidence)
    $OutputPath = Join-Path $Evidence ("worker-publication-" + $SessionId + ".json")
    if ($State.analysis_sync_enabled -ne $true) {
        $NotRun = [ordered]@{
            schema_version = "1.0"
            session_id = $SessionId
            status = "NOT_RUN"
            field_sync_status = $null
            session_analysis_pending_count = $null
            cloud_analysis_status = $null
            verified = $false
            reason = "This run mode explicitly disables development cloud publication."
        }
        Write-JsonFile $NotRun $OutputPath
        return $NotRun
    }
    $Evaluator = Join-Path $PacketRoot "field_gate_evaluator.py"
    $Arguments = @(
        $Evaluator, "--inspect-worker-publication",
        "--state-db", [string]$State.state_db,
        "--jobs-db", [string]$State.jobs_db,
        "--session-id", $SessionId,
        "--output", $OutputPath
    )
    $OutputText = & $PythonPath @Arguments 2>&1 | Out-String
    $InspectionExitCode = $LASTEXITCODE
    if (Test-Path -LiteralPath $OutputPath -PathType Leaf) {
        $Receipt = Read-JsonFile $OutputPath
        if ($InspectionExitCode -eq 0 -and $Receipt.verified -ne $true) {
            throw "Worker publication inspector returned an inconsistent success."
        }
        return $Receipt
    }
    $Failure = [ordered]@{
        schema_version = "1.0"
        session_id = $SessionId
        status = "FAIL"
        field_sync_status = $null
        session_analysis_pending_count = $null
        cloud_analysis_status = $null
        verified = $false
        reason = "worker publication inspector failed: $OutputText"
    }
    Write-JsonFile $Failure $OutputPath
    return $Failure
}

function Start-FieldSession {
    if ($LocalOnly -and $ExistingDevelopmentControlOnly) {
        throw "LocalOnly and ExistingDevelopmentControlOnly are mutually exclusive."
    }
    if ($LocalOnly -and -not $ObservationOnly) {
        throw "LocalOnly is permitted only with ObservationOnly."
    }
    if ($ExistingDevelopmentControlOnly -and -not $ObservationOnly) {
        throw "ExistingDevelopmentControlOnly is permitted only with ObservationOnly."
    }
    $ForbiddenCloudArguments = @(
        @($SyncBaseUrl, $DeviceId, $CredentialTarget, $CloudbaseEnvironmentId) |
            Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }
    )
    if ($LocalOnly -and $ForbiddenCloudArguments.Count -ne 0) {
        throw "LocalOnly rejects all sync and cloud arguments."
    }
    if ($ExistingDevelopmentControlOnly -and $ForbiddenCloudArguments.Count -ne 0) {
        throw "ExistingDevelopmentControlOnly rejects explicit sync and cloud arguments."
    }
    $Manifest = Verify-Packet
    if ($ExistingDevelopmentControlOnly) {
        Stop-PreviousUsbFieldPackets
        Assert-ExistingFieldServiceStopped
    }
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw "Python is missing: $PythonPath"
    }
    $Assets = Resolve-ExternalAssets $Manifest
    $ExistingRuntime = if ($LocalOnly) { $null } else { Read-ExistingFieldRuntime }
    $ResolvedJoinQr = Resolve-JoinQrAsset $Manifest $ExistingRuntime
    $ResolvedSync = if ($LocalOnly) {
        [pscustomobject][ordered]@{
            source = "local_only"
            environment = "local_only"
            cloudbase_environment_id = $null
            sync_base_url = $null
            device_id = $null
            credential_target = $null
        }
    } else {
        Resolve-SyncConfiguration $ExistingRuntime $Manifest
    }
    if ([string]::IsNullOrWhiteSpace($OutputParent)) {
        $OutputParent = Join-Path ([Environment]::GetFolderPath("UserProfile")) "Downloads"
    }
    $ResolvedOutput = [IO.Path]::GetFullPath($OutputParent)
    New-Item -ItemType Directory -Force -Path $ResolvedOutput | Out-Null
    $Name = "eight-ball-field-result-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + [Guid]::NewGuid().ToString("N").Substring(0, 6)
    $Result = Join-Path $ResolvedOutput $Name
    $Work = Join-Path $Result "work"
    $Evidence = Join-Path $Result "evidence"
    $Logs = Join-Path $Evidence "logs"
    $LiveArtifacts = Join-Path $Evidence "live-field"
    $PostgameArtifacts = Join-Path $Evidence "postgame-artifacts"
    $Ring = Join-Path $Work "ring"
    $Jobs = Join-Path $Work "jobs"
    $StateDb = Join-Path $Work "field-state.sqlite3"
    $OperatorGateEvidence = Join-Path $Work "operator-gate-evidence.json"
    foreach ($Directory in @($Work, $Evidence, $Logs, $LiveArtifacts, $PostgameArtifacts, $Ring, $Jobs)) {
        New-Item -ItemType Directory -Force -Path $Directory | Out-Null
    }
    Initialize-OperatorGateEvidence $OperatorGateEvidence
    Copy-Item -LiteralPath $ManifestPath -Destination (Join-Path $Evidence "packet-manifest.json")

    $ExistingSessionRetirement = $null
    if ($ExistingDevelopmentControlOnly) {
        $env:PYTHONPATH = Join-Path $PacketRoot "src"
        $env:PYTHONUTF8 = "1"
        $env:PYTHONIOENCODING = "utf-8"
        $ExistingSessionRetirement = Invoke-ExistingSessionRetirement $ResolvedSync $Evidence $ResolvedOutput
    }

    $Compiler = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
    if (-not (Test-Path -LiteralPath $Compiler -PathType Leaf)) {
        throw "Verified built-in C# compiler is missing."
    }
    $NativeSource = Join-Path $PacketRoot "native\windows\mainline1_uvc_capture_csharp\main.cs"
    $NativeHelper = Join-Path $Work "mainline1_uvc_capture.exe"
    $CompilerOutput = (& $Compiler /nologo /platform:x64 /optimize+ /target:exe ("/out:" + $NativeHelper) $NativeSource 2>&1 | Out-String).Trim()
    $CompilerOutput | Set-Content -LiteralPath (Join-Path $Logs "compiler.log") -Encoding UTF8
    if ($LASTEXITCODE -ne 0) {
        throw "Native capture compilation failed with code $LASTEXITCODE."
    }
    $Signature = Get-AuthenticodeSignature -LiteralPath $NativeHelper
    if ($Signature.Status -ne [System.Management.Automation.SignatureStatus]::NotSigned) {
        throw "Development packet requires the locally compiled helper to remain unsigned."
    }
    Write-JsonFile ([ordered]@{
        development_only = $true
        production_ready = $false
        trust = "unsigned_source_compiled_at_runtime"
        authenticode_status = [string]$Signature.Status
        helper_sha256 = (Get-FileHash -LiteralPath $NativeHelper -Algorithm SHA256).Hash.ToLowerInvariant()
        source_sha256 = (Get-FileHash -LiteralPath $NativeSource -Algorithm SHA256).Hash.ToLowerInvariant()
    }) (Join-Path $Evidence "native-helper.json")

    $TokenBytes = New-Object byte[] 32
    $Generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $Generator.GetBytes($TokenBytes) }
    finally { $Generator.Dispose() }
    $ControlToken = [Convert]::ToBase64String($TokenBytes)
    $SecretPath = Join-Path $Work "control-token.txt"
    Set-Content -LiteralPath $SecretPath -Value $ControlToken -Encoding ASCII -NoNewline
    Protect-PrivateFile $SecretPath
    $Runner = Join-Path $PacketRoot "scripts\algorithm\serve_mainline1_usb_field.py"
    $Setup = Join-Path $PacketRoot "candidate\capture-setup.json"
    $Calibration = Join-Path $PacketRoot "candidate\calibrations"
    $Config = Join-Path $PacketRoot "configs\algorithm\shot-local.yaml"
    $Arguments = @(
        "-u", $Runner,
        "--native-helper", $NativeHelper,
        "--device-locator", ([string]$Manifest.run.device_locator),
        "--device-fingerprint", ([string]$Manifest.run.device_fingerprint),
        "--capture-setup", $Setup,
        "--calibration-root", $Calibration,
        "--detector-training-run", $Assets.detector,
        "--identity-training-run", $Assets.identity,
        "--dinov2-assets", $Assets.dinov2,
        "--config", $Config,
        "--state-db", $StateDb,
        "--ring-root", $Ring,
        "--jobs-root", $Jobs,
        "--artifact-root", $LiveArtifacts,
        "--host", $HostAddress,
        "--port", [string]$Port,
        "--control-token-file", $SecretPath,
        "--device", $Device,
        "--batch-size", [string]$BatchSize,
        "--shutdown-stdin",
        "--identity-gate-status", $Assets.identity_gate_status
    )
    if (-not [string]::IsNullOrWhiteSpace([string]$ResolvedJoinQr)) {
        $Arguments += @("--join-qr-asset", $ResolvedJoinQr)
    }
    if ($ObservationOnly) { $Arguments += "--observation-only" }
    else { $Arguments += @("--rule-profile", $RuleProfile) }
    if (-not $LocalOnly) {
        $Arguments += @(
            "--runtime-environment", $ResolvedSync.environment,
            "--cloudbase-environment-id", $ResolvedSync.cloudbase_environment_id,
            "--expected-development-environment-id", ([string]$Manifest.run.development_environment_id),
            "--sync-base-url", $ResolvedSync.sync_base_url,
            "--device-id", $ResolvedSync.device_id,
            "--credential-target", $ResolvedSync.credential_target
        )
        if ($ExistingDevelopmentControlOnly) {
            $Arguments += "--existing-development-control-only"
        }
    }
    $RedactedArguments = @()
    $RedactNext = $false
    foreach ($Argument in $Arguments) {
        if ($RedactNext) {
            $RedactedArguments += "<redacted>"
            $RedactNext = $false
            continue
        }
        $RedactedArguments += [string]$Argument
        if ([string]$Argument -eq "--control-token") { $RedactNext = $true }
    }
    if ($RedactNext) { throw "Launch argument redaction inventory is incomplete." }

    $env:PYTHONPATH = Join-Path $PacketRoot "src"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $StartInfo = [Diagnostics.ProcessStartInfo]::new()
    $StartInfo.FileName = $PythonPath
    $StartInfo.WorkingDirectory = $PacketRoot
    $StartInfo.UseShellExecute = $false
    $StartInfo.RedirectStandardInput = $true
    $StartInfo.RedirectStandardOutput = $true
    $StartInfo.RedirectStandardError = $true
    $StartInfo.Arguments = Join-WindowsProcessArguments $Arguments
    $Process = [Diagnostics.Process]::new()
    $Process.StartInfo = $StartInfo
    $Ready = $false
    $StopSent = $false
    $ReadyDeadline = [DateTimeOffset]::UtcNow.AddMinutes(3)
    $RuntimeLog = Join-Path $Logs "field-service.log"
    $ErrorLog = Join-Path $Logs "field-service.stderr.log"
    $StopRequest = Join-Path $Work "stop.request"
    Write-Host "Development-only field result: $Result"
    try {
        if (-not $Process.Start()) { throw "Field service process did not start." }
        Write-JsonFile ([ordered]@{
            schema_version = "1.0"
            development_evidence_only = $true
            production_ready = $false
            packet_version = [string]$Manifest.packet_version
            packet_manifest_sha256 = (Get-FileHash -LiteralPath $ManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
            process_executable = $PythonPath
            process_executable_sha256 = (Get-FileHash -LiteralPath $PythonPath -Algorithm SHA256).Hash.ToLowerInvariant()
            process_working_directory = $PacketRoot
            process_arguments = $RedactedArguments
            control_token_redacted = $true
            server_pid = $Process.Id
            started_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
        }) (Join-Path $Evidence "launch-audit.json")
        $OutputLine = $Process.StandardOutput.ReadLineAsync()
        $ErrorLine = $Process.StandardError.ReadLineAsync()
        while (-not $Process.HasExited -or $null -ne $OutputLine -or $null -ne $ErrorLine) {
            if ($null -ne $OutputLine -and $OutputLine.IsCompleted) {
                $Line = $OutputLine.GetAwaiter().GetResult()
                if ($null -eq $Line) { $OutputLine = $null } else {
                    Add-Content -LiteralPath $RuntimeLog -Value $Line -Encoding UTF8
                    [Console]::Out.WriteLine($Line)
                    try { $Parsed = $Line | ConvertFrom-Json -ErrorAction Stop } catch { $Parsed = $null }
                    if (-not $Ready -and $null -ne $Parsed -and $Parsed.status -eq "USB_FIELD_READY") {
                        $Ready = $true
                        $ResolvedPublicHost = Resolve-PublicHost $PublicHost
                        $PublicBase = "http://${ResolvedPublicHost}:$($Parsed.port)"
                        $ControlBase = "http://127.0.0.1:$($Parsed.port)"
                        $InitialScreen = Invoke-RestMethod -Method Get -Uri ($ControlBase + "/api/screen")
                        $InitialSessionId = [string]$InitialScreen.session.session_id
                        if ([string]::IsNullOrWhiteSpace($InitialSessionId)) {
                            throw "Initial field screen did not identify its session."
                        }
                        if ($ObservationOnly) {
                            if ($null -ne $InitialScreen.match.rule_profile_id) {
                                throw "Observation-only run unexpectedly received rule authority."
                            }
                        } elseif ([string]$InitialScreen.match.rule_profile_id -ne $RuleProfile) {
                            throw "Initial field screen did not persist the fixed rule profile."
                        }
                        $JoinQrSha256 = $null
                        if (-not [string]::IsNullOrWhiteSpace([string]$ResolvedJoinQr)) {
                            $JoinQrSha256 = (Get-FileHash -LiteralPath $ResolvedJoinQr -Algorithm SHA256).Hash.ToLowerInvariant()
                        }
                        $State = [ordered]@{
                            schema_version = "1.0"
                            result_root = $Result
                            python_path = $PythonPath
                            server_pid = $Process.Id
                            public_base_url = $PublicBase
                            control_base_url = $ControlBase
                            state_db = $StateDb
                            jobs_db = (Join-Path $Jobs "analysis-jobs.sqlite3")
                            live_artifact_root = $LiveArtifacts
                            postgame_artifact_root = $PostgameArtifacts
                            capture_setup = $Setup
                            calibration_root = $Calibration
                            config = $Config
                            detector_training_run = $Assets.detector
                            identity_training_run = $Assets.identity
                            dinov2_assets = $Assets.dinov2
                            fps = $Fps
                            device = $Device
                            batch_size = $BatchSize
                            rule_profile = $RuleProfile
                            local_only = [bool]$LocalOnly
                            existing_development_control_only = [bool]$ExistingDevelopmentControlOnly
                            analysis_sync_enabled = [bool](-not $LocalOnly -and -not $ExistingDevelopmentControlOnly)
                            observation_only = [bool]$ObservationOnly
                            identity_gate_status = $Assets.identity_gate_status
                            runtime_environment = $ResolvedSync.environment
                            cloudbase_environment_id = $ResolvedSync.cloudbase_environment_id
                            sync_source = $ResolvedSync.source
                            sync_base_url = $ResolvedSync.sync_base_url
                            device_id = $ResolvedSync.device_id
                            credential_target = $ResolvedSync.credential_target
                            join_qr_asset = $ResolvedJoinQr
                            join_qr_sha256 = $JoinQrSha256
                            join_qr_source = if ($ExistingDevelopmentControlOnly -and -not [string]::IsNullOrWhiteSpace([string]$Manifest.run.join_qr_asset)) { "packet_private_asset" } elseif ($ExistingDevelopmentControlOnly) { "existing_field_runtime" } elseif ($null -eq $ResolvedJoinQr) { "none" } else { "packet_or_explicit" }
                            current_pointer_path = if ($null -ne $ExistingRuntime) { [string]$ExistingRuntime._current_pointer_path } else { $null }
                            current_pointer_sha256 = if ($null -ne $ExistingRuntime) { [string]$ExistingRuntime._current_pointer_sha256 } else { $null }
                            existing_session_retirement = $ExistingSessionRetirement
                            initial_session_id = $InitialSessionId
                            current_session_id = $InitialSessionId
                            session_receipts = @(
                                [ordered]@{
                                    session_id = $InitialSessionId
                                    join_qr_sha256 = $JoinQrSha256
                                    worker_launch = $null
                                }
                            )
                            started_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
                        }
                        Write-JsonFileAtomic $State (Join-Path $Work "run-state.json")
                        Write-Host ("BIG SCREEN: " + $PublicBase + "/screen") -ForegroundColor Green
                        if (-not [string]::IsNullOrWhiteSpace([string]$ResolvedJoinQr)) {
                            Write-Host ("JOIN QR IMAGE: " + $PublicBase + "/api/table-code.png") -ForegroundColor Green
                        }
                        Write-Host "No participant or breaker is synthesized by this packet." -ForegroundColor Yellow
                        Write-Host "OPERATOR GATES: keep capture healthy for 30 minutes; exercise missing facts; perform idle and in-shot disconnect/recovery; complete at least 10 settled shots across exactly two sessions." -ForegroundColor Yellow
                        Write-Host ("OPERATOR EVIDENCE TEMPLATE: " + $OperatorGateEvidence) -ForegroundColor Yellow
                        $EndCommand = '& "{0}" -Mode EndSession -ResultRoot "{1}"' -f (Join-Path $PacketRoot "run.ps1"), $Result
                        Write-Host ("END COMMAND (run in a second PowerShell): " + $EndCommand)
                    }
                    $OutputLine = $Process.StandardOutput.ReadLineAsync()
                }
            }
            if ($null -ne $ErrorLine -and $ErrorLine.IsCompleted) {
                $Line = $ErrorLine.GetAwaiter().GetResult()
                if ($null -eq $Line) { $ErrorLine = $null } else {
                    Add-Content -LiteralPath $ErrorLog -Value $Line -Encoding UTF8
                    [Console]::Error.WriteLine($Line)
                    $ErrorLine = $Process.StandardError.ReadLineAsync()
                }
            }
            if (-not $Ready -and [DateTimeOffset]::UtcNow -ge $ReadyDeadline) {
                throw "USB_FIELD_READY was not observed within three minutes."
            }
            if (-not $StopSent -and (Test-Path -LiteralPath $StopRequest -PathType Leaf)) {
                $Process.StandardInput.WriteLine("STOP")
                $Process.StandardInput.Flush()
                $StopSent = $true
            }
            Start-Sleep -Milliseconds 20
        }
        $Process.WaitForExit()
        Write-JsonFile ([ordered]@{
            result_root = $Result
            state_db = $StateDb
            server_pid = $Process.Id
            exit_code = $Process.ExitCode
            ready_observed = $Ready
            stopped_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
        }) (Join-Path $Work "server-exit.json")
        if (-not $Ready -or -not $StopSent -or $Process.ExitCode -ne 0) { exit 2 }
        exit 0
    } finally {
        if (-not $Process.HasExited) {
            try { $Process.StandardInput.WriteLine("STOP"); $Process.StandardInput.Flush() } catch { }
            if (-not $Process.WaitForExit(8000)) { $Process.Kill(); $Process.WaitForExit() }
        }
        $Process.Dispose()
    }
}

function Complete-FrozenSession {
    param([object]$State, [string]$SessionId, [string]$Token, [string]$Evidence)
    $ReportPath = Join-Path $Evidence ("session-evidence-" + $SessionId + ".json")
    $Report = Invoke-EvidenceExport $State $SessionId $ReportPath
    if ($Report.session.status -eq "ended") { return $Report }
    if ($Report.session.status -ne "open") { throw "Frozen session has an invalid status." }
    $Live = Invoke-RestMethod -Method Get -Uri ($State.control_base_url + "/api/screen")
    if ([string]$Live.session.session_id -ne $SessionId) {
        throw "Open frozen session no longer matches the live service; refusing to end another session."
    }
    $Headers = @{ "X-Control-Token" = $Token }
    $Response = Invoke-RestMethod -Method Post -Uri ($State.control_base_url + "/api/control") -Headers $Headers -ContentType "application/json" -Body '{"command":"end_session"}'
    if ([string]$Response.session_id -ne $SessionId) {
        throw "End-session response does not match the frozen session."
    }
    Write-JsonFile $Response (Join-Path $Evidence ("end-session-" + $SessionId + ".json"))
    $Report = Invoke-EvidenceExport $State $SessionId $ReportPath
    if ($Report.session.status -ne "ended") { throw "Frozen session did not end durably." }
    return $Report
}

function Wait-ForSessionRollover {
    param([object]$State, [string]$EndedSessionId, [string]$Token)
    $Deadline = [DateTimeOffset]::UtcNow.AddSeconds(15)
    while ([DateTimeOffset]::UtcNow -lt $Deadline) {
        $Live = Invoke-RestMethod -Method Get -Uri ($State.control_base_url + "/api/screen")
        $NewSessionId = [string]$Live.session.session_id
        if (-not [string]::IsNullOrWhiteSpace($NewSessionId) -and $NewSessionId -ne $EndedSessionId) {
            if ($State.observation_only) {
                if ($null -ne $Live.match.rule_profile_id) {
                    throw "Observation-only rollover unexpectedly received rule authority."
                }
            } elseif ([string]$Live.match.rule_profile_id -ne $RuleProfile) {
                throw "Rollover session lost the fixed rule profile."
            }
            $Headers = @{ "X-Control-Token" = $Token }
            $Shots = Invoke-RestMethod -Method Get -Uri ($State.control_base_url + "/api/local/shots") -Headers $Headers
            if ([string]$Shots.session_id -ne $NewSessionId -or @($Shots.shots).Count -ne 0) {
                throw "Rollover session is not the expected empty next session."
            }
            if ($null -ne $State.join_qr_sha256) {
                $QrHash = (Get-FileHash -LiteralPath $State.join_qr_asset -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($QrHash -ne $State.join_qr_sha256) { throw "Join QR changed across session rollover." }
            }
            if (@($State.session_receipts | Where-Object { $_.session_id -eq $NewSessionId }).Count -ne 0) {
                throw "Rollover reused a prior session ID."
            }
            return $NewSessionId
        }
        Start-Sleep -Milliseconds 200
    }
    throw "A distinct rollover session was not observed within fifteen seconds."
}

function Invoke-SessionWorker {
    param(
        [object]$State,
        [string]$SessionId,
        [string]$ServerExitPath,
        [string]$Evidence,
        [string]$Logs,
        [string]$WorkerIdentity,
        [bool]$LiveMode
    )
    $ArtifactRoot = Join-Path $State.postgame_artifact_root $SessionId
    New-Item -ItemType Directory -Force -Path $ArtifactRoot | Out-Null
    $ReportPath = Join-Path $Evidence ("session-evidence-" + $SessionId + ".json")
    $InitialReport = Invoke-EvidenceExport $State $SessionId $ReportPath
    if ($InitialReport.session.status -ne "ended" -or $null -eq $InitialReport.postgame_manifest) {
        throw "Worker target is not the exact ended frozen session."
    }
    $Deadline = [DateTimeOffset]::Parse([string]$InitialReport.postgame_manifest.deadline_at_utc)
    $ExpiredBeforeWorker = [DateTimeOffset]::UtcNow -ge $Deadline
    if ($ExpiredBeforeWorker) { throw "Worker manifest deadline already expired." }
    $EmptyPolls = if ($LiveMode) { "480" } else { "1" }
    $Worker = Join-Path $PacketRoot "scripts\algorithm\run_mainline1_shot_worker.py"
    $Arguments = @(
        "-u", $Worker, "--state-db", $State.state_db, "--jobs-db", $State.jobs_db,
        "--artifact-root", $ArtifactRoot, "--session-id", $SessionId,
        "--worker-id", ("raytheon-dev-" + $WorkerIdentity),
        "--capture-setup", $State.capture_setup, "--calibration-root", $State.calibration_root,
        "--config", $State.config, "--detector-training-run", $State.detector_training_run,
        "--identity-training-run", $State.identity_training_run, "--dinov2-assets", $State.dinov2_assets,
        "--fps", [string]$State.fps, "--device", $State.device,
        "--batch-size", [string]$State.batch_size, "--max-empty-polls", $EmptyPolls,
        "--poll-seconds", "0.25"
    )
    $LiveResourceAuditPath = Join-Path $Evidence ("live-resource-" + $SessionId + ".jsonl")
    if ($LiveMode) {
        $Arguments += @(
            "--live-resource-url", ($State.control_base_url + "/api/screen"),
            "--live-resource-audit", $LiveResourceAuditPath,
            "--live-server-pid", [string]$State.server_pid
        )
    } else {
        $Arguments += @("--server-exit-receipt", $ServerExitPath)
    }
    if ($State.analysis_sync_enabled -eq $true) {
        $Arguments += @(
            "--development-analysis-sync", "--sync-base-url", $State.sync_base_url,
            "--device-id", $State.device_id, "--credential-target", $State.credential_target,
            "--development-cloudbase-environment-id", $State.cloudbase_environment_id
        )
    }
    $StartInfo = [Diagnostics.ProcessStartInfo]::new()
    $StartInfo.FileName = $PythonPath
    $StartInfo.WorkingDirectory = $PacketRoot
    $StartInfo.UseShellExecute = $false
    $StartInfo.RedirectStandardOutput = $true
    $StartInfo.RedirectStandardError = $true
    $StartInfo.Arguments = Join-WindowsProcessArguments $Arguments
    $Process = [Diagnostics.Process]::new()
    $Process.StartInfo = $StartInfo
    if (-not $Process.Start()) { throw "Postgame worker did not start for $SessionId." }
    $Process.PriorityClass = [Diagnostics.ProcessPriorityClass]::BelowNormal
    $WorkerPid = $Process.Id
    $Stdout = $Process.StandardOutput.ReadToEndAsync()
    $Stderr = $Process.StandardError.ReadToEndAsync()
    $WorkerTimedOut = $false
    $HardStopAt = $Deadline.AddSeconds(5)
    try {
        while ($true) {
            if ($Process.HasExited) { break }
            if ([DateTimeOffset]::UtcNow -ge $HardStopAt) {
                $WorkerTimedOut = $true
                break
            }
            Start-Sleep -Seconds 1
        }
    } finally {
        if (-not $Process.HasExited) { $Process.Kill(); $Process.WaitForExit() }
        $Stdout.GetAwaiter().GetResult() | Set-Content -LiteralPath (Join-Path $Logs ("shot-worker-" + $SessionId + ".log")) -Encoding UTF8
        $Stderr.GetAwaiter().GetResult() | Set-Content -LiteralPath (Join-Path $Logs ("shot-worker-" + $SessionId + ".stderr.log")) -Encoding UTF8
        $ExitCode = $Process.ExitCode
        $Process.Dispose()
    }
    $Final = Invoke-EvidenceExport $State $SessionId $ReportPath
    $Publication = Invoke-WorkerPublicationInspection $State $SessionId $Evidence
    $LocalAccepted = (
        -not $WorkerTimedOut -and
        $ExitCode -eq 0 -and
        $Final.analysis.status -eq "AVAILABLE" -and
        $Final.analysis.finalized -eq $true -and
        $Final.analysis.available_count -eq $Final.analysis.expected_count
    )
    $Accepted = (
        $LocalAccepted -and
        $State.analysis_sync_enabled -eq $true -and
        $Publication.verified -eq $true
    )
    return [ordered]@{
        session_id = $SessionId
        worker_identity = $WorkerIdentity
        worker_execution_mode = if ($LiveMode) { "live_priority" } else { "offline" }
        worker_pid = $WorkerPid
        report_session_id = [string]$Final.session.session_id
        artifact_root = $ArtifactRoot
        worker_exit_code = $ExitCode
        worker_timed_out = $WorkerTimedOut
        analysis_status = $Final.analysis.status
        expected_count = $Final.analysis.expected_count
        available_count = $Final.analysis.available_count
        finalized = $Final.analysis.finalized
        job_ids = @($Final.jobs | ForEach-Object { [string]$_.job_id })
        artifact_ids = @($Final.jobs | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_.artifact_id) } | ForEach-Object { [string]$_.artifact_id })
        shot_count = @($Final.shots).Count
        ended_at_utc = [string]$Final.postgame_manifest.ended_at_utc
        deadline_at_utc = [string]$Final.postgame_manifest.deadline_at_utc
        completed_at_utc = [string]$Final.exported_at_utc
        field_sync_status = $Publication.field_sync_status
        session_analysis_pending_count = $Publication.session_analysis_pending_count
        cloud_analysis_status = $Publication.cloud_analysis_status
        publication_verified = $Publication.verified
        publication_reason = $Publication.reason
        local_accepted = $LocalAccepted
        accepted = $Accepted
    }
}

function Get-FrozenSessionReceipt {
    param([object]$State, [string]$SessionId)
    $Matches = @($State.session_receipts | Where-Object { [string]$_.session_id -eq $SessionId })
    if ($Matches.Count -ne 1) { throw "Frozen session receipt is missing or duplicated." }
    return $Matches[0]
}

function Get-WorkerLauncherArguments {
    param([object]$State, [string]$SessionId, [string]$Identity)
    return @(
        "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $PacketRoot "run.ps1"), "-Mode", "Worker",
        "-ResultRoot", [string]$State.result_root,
        "-PythonPath", [string]$State.python_path,
        "-WorkerSessionId", $SessionId, "-WorkerIdentity", $Identity
    )
}

function Start-SessionWorkerBackground {
    param(
        [string]$StatePath,
        [object]$State,
        [string]$SessionId,
        [ValidateSet("live_priority", "offline")][string]$ExecutionMode,
        [string]$Evidence,
        [string]$Logs
    )
    $ReportPath = Join-Path $Evidence ("session-evidence-" + $SessionId + ".json")
    $InitialReport = Invoke-EvidenceExport $State $SessionId $ReportPath
    if ($InitialReport.session.status -ne "ended" -or $null -eq $InitialReport.postgame_manifest) {
        throw "Background worker target is not the exact ended frozen session."
    }
    $Deadline = [string]$InitialReport.postgame_manifest.deadline_at_utc
    $WorkerIdentityValue = [Guid]::NewGuid().ToString("N")
    $ResultPath = Join-Path $Evidence ("session-worker-result-" + $SessionId + ".json")
    $StdoutPath = Join-Path $Logs ("shot-worker-" + $SessionId + ".log")
    $StderrPath = Join-Path $Logs ("shot-worker-" + $SessionId + ".stderr.log")
    $LiveResourceAuditPath = Join-Path $Evidence ("live-resource-" + $SessionId + ".jsonl")
    $PowerShellPath = Join-Path $PSHOME "powershell.exe"
    $LauncherArguments = Get-WorkerLauncherArguments $State $SessionId $WorkerIdentityValue
    $CommandLine = Join-WindowsProcessArguments $LauncherArguments
    $CommandIdentitySha256 = Get-StringSha256 ($PowerShellPath + "`n" + $CommandLine)
    $Launch = [ordered]@{
        schema_version = "1.0"
        session_id = $SessionId
        worker_identity = $WorkerIdentityValue
        execution_mode = $ExecutionMode
        status = "STARTING"
        launcher_pid = $null
        launcher_executable = $PowerShellPath
        command_identity_sha256 = $CommandIdentitySha256
        artifact_root = (Join-Path $State.postgame_artifact_root $SessionId)
        stdout_path = $StdoutPath
        stderr_path = $StderrPath
        live_resource_audit_path = if ($ExecutionMode -eq "live_priority") { $LiveResourceAuditPath } else { $null }
        resource_isolation = "development_best_effort_not_proven"
        result_path = $ResultPath
        deadline_at_utc = $Deadline
        started_at_utc = $null
        completed_at_utc = $null
        exit_code = $null
        deadline_missed = $null
        diagnostic = $null
    }
    $Registered = Update-RunStateLocked $StatePath {
        param($Current)
        $Receipt = Get-FrozenSessionReceipt $Current $SessionId
        if ($Receipt.PSObject.Properties.Name -contains "worker_launch" -and $null -ne $Receipt.worker_launch) {
            throw "Session worker was already launched; refusing a duplicate."
        }
        $Receipt | Add-Member -NotePropertyName worker_launch -NotePropertyValue ([pscustomobject]$Launch) -Force
    }
    $Process = $null
    try {
        $StartInfo = [Diagnostics.ProcessStartInfo]::new()
        $StartInfo.FileName = $PowerShellPath
        $StartInfo.WorkingDirectory = $PacketRoot
        $StartInfo.UseShellExecute = $false
        $StartInfo.CreateNoWindow = $true
        $StartInfo.Arguments = $CommandLine
        $Process = [Diagnostics.Process]::new()
        $Process.StartInfo = $StartInfo
        if (-not $Process.Start()) { throw "Worker supervisor process did not start." }
        $Process.PriorityClass = [Diagnostics.ProcessPriorityClass]::BelowNormal
        $LauncherPid = $Process.Id
        $StartedAt = [DateTimeOffset]::UtcNow.ToString("o")
        $Process.Dispose()
        return Update-RunStateLocked $StatePath {
            param($Current)
            $Receipt = Get-FrozenSessionReceipt $Current $SessionId
            if ([string]$Receipt.worker_launch.worker_identity -ne $WorkerIdentityValue -or [string]$Receipt.worker_launch.status -ne "STARTING") {
                throw "Worker launch registration changed before PID attachment."
            }
            $Receipt.worker_launch.launcher_pid = $LauncherPid
            $Receipt.worker_launch.started_at_utc = $StartedAt
            $Receipt.worker_launch.status = "RUNNING"
        }
    } catch {
        if ($null -ne $Process) { $Process.Dispose() }
        $Diagnostic = $_.Exception.GetType().Name + ": " + $_.Exception.Message
        return Update-RunStateLocked $StatePath {
            param($Current)
            $Receipt = Get-FrozenSessionReceipt $Current $SessionId
            if ([string]$Receipt.worker_launch.worker_identity -eq $WorkerIdentityValue) {
                $Receipt.worker_launch.status = "START_FAILED"
                $Receipt.worker_launch.diagnostic = $Diagnostic
            }
        }
    }
}

function Assert-WorkerProcessIdentity {
    param([object]$State, [object]$Launch)
    if ($null -eq $Launch.launcher_pid -or [int]$Launch.launcher_pid -le 0) {
        throw "Worker launch PID is missing."
    }
    $ExpectedArguments = Get-WorkerLauncherArguments $State ([string]$Launch.session_id) ([string]$Launch.worker_identity)
    $ExpectedCommandLine = Join-WindowsProcessArguments $ExpectedArguments
    $ExpectedHash = Get-StringSha256 ([string]$Launch.launcher_executable + "`n" + $ExpectedCommandLine)
    if ($ExpectedHash -ne [string]$Launch.command_identity_sha256) {
        throw "Frozen worker command identity hash is invalid."
    }
    $Candidate = Get-CimInstance Win32_Process -Filter ("ProcessId = " + [int]$Launch.launcher_pid) -ErrorAction SilentlyContinue
    if ($null -eq $Candidate) { return $false }
    $CommandLine = [string]$Candidate.CommandLine
    if (
        -not $CommandLine.Contains([string]$Launch.worker_identity) -or
        -not $CommandLine.Contains([string]$Launch.session_id) -or
        -not $CommandLine.Contains((Join-Path $PacketRoot "run.ps1"))
    ) {
        throw "Live worker PID command identity does not match the frozen session."
    }
    return $true
}

function New-WorkerFailureResult {
    param([object]$State, [string]$SessionId, [object]$Launch, [string]$Reason)
    $ReportPath = Join-Path ([string]$State.result_root) ("evidence\session-evidence-" + $SessionId + ".json")
    $Report = if (Test-Path -LiteralPath $ReportPath -PathType Leaf) { Read-JsonFile $ReportPath } else { $null }
    $Deadline = if ($null -ne $Launch) { [string]$Launch.deadline_at_utc } elseif ($null -ne $Report) { [string]$Report.postgame_manifest.deadline_at_utc } else { $null }
    return [ordered]@{
        session_id = $SessionId
        report_session_id = if ($null -ne $Report) { [string]$Report.session.session_id } else { $null }
        artifact_root = if ($null -ne $Launch) { [string]$Launch.artifact_root } else { Join-Path $State.postgame_artifact_root $SessionId }
        worker_identity = if ($null -ne $Launch) { [string]$Launch.worker_identity } else { $null }
        worker_execution_mode = if ($null -ne $Launch) { [string]$Launch.execution_mode } else { $null }
        command_identity_sha256 = if ($null -ne $Launch) { [string]$Launch.command_identity_sha256 } else { $null }
        launcher_pid = if ($null -ne $Launch) { $Launch.launcher_pid } else { $null }
        worker_exit_code = 2
        worker_timed_out = $false
        analysis_status = if ($null -ne $Report) { [string]$Report.analysis.status } else { "PROCESSING" }
        expected_count = if ($null -ne $Report) { $Report.analysis.expected_count } else { $null }
        available_count = if ($null -ne $Report) { $Report.analysis.available_count } else { $null }
        finalized = if ($null -ne $Report) { $Report.analysis.finalized } else { $false }
        job_ids = @()
        artifact_ids = @()
        shot_count = if ($null -ne $Report) { @($Report.shots).Count } else { 0 }
        ended_at_utc = if ($null -ne $Report) { [string]$Report.postgame_manifest.ended_at_utc } else { $null }
        deadline_at_utc = $Deadline
        completed_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
        field_sync_status = $null
        session_analysis_pending_count = $null
        cloud_analysis_status = $null
        publication_verified = $false
        publication_reason = if ($State.analysis_sync_enabled -ne $true) { "This run mode explicitly disables development cloud publication." } else { $Reason }
        local_accepted = $false
        accepted = $false
        worker_launch = $Launch
        worker_diagnostics = [ordered]@{
            status = "FAIL"
            reason = $Reason
            deadline_missed = if ([string]::IsNullOrWhiteSpace($Deadline)) { $true } else { [DateTimeOffset]::UtcNow -gt [DateTimeOffset]::Parse($Deadline) }
        }
    }
}

function Wait-SessionWorkerResult {
    param([object]$State, [object]$SessionReceipt)
    $Launch = $SessionReceipt.worker_launch
    if ($null -eq $Launch) { return New-WorkerFailureResult $State ([string]$SessionReceipt.session_id) $null "worker was never launched" }
    if ([string]$Launch.status -eq "STARTING" -and $null -eq $Launch.launcher_pid) {
        return New-WorkerFailureResult $State ([string]$SessionReceipt.session_id) $Launch "worker registration was interrupted before PID attachment"
    }
    $ResultPath = [string]$Launch.result_path
    $WaitUntil = [DateTimeOffset]::Parse([string]$Launch.deadline_at_utc).AddSeconds(15)
    while ([DateTimeOffset]::UtcNow -lt $WaitUntil) {
        if (Test-Path -LiteralPath $ResultPath -PathType Leaf) { break }
        if ([string]$Launch.status -eq "START_FAILED") {
            return New-WorkerFailureResult $State ([string]$SessionReceipt.session_id) $Launch ([string]$Launch.diagnostic)
        }
        $Alive = Assert-WorkerProcessIdentity $State $Launch
        if (-not $Alive) { break }
        Start-Sleep -Milliseconds 200
    }
    if (-not (Test-Path -LiteralPath $ResultPath -PathType Leaf)) {
        return New-WorkerFailureResult $State ([string]$SessionReceipt.session_id) $Launch "worker exited or exceeded its manifest deadline without a durable result"
    }
    $Result = Read-JsonFile $ResultPath
    if (
        [string]$Result.session_id -ne [string]$SessionReceipt.session_id -or
        [string]$Result.worker_identity -ne [string]$Launch.worker_identity -or
        [string]$Result.worker_execution_mode -ne [string]$Launch.execution_mode -or
        [string]$Result.command_identity_sha256 -ne [string]$Launch.command_identity_sha256 -or
        [int]$Result.launcher_pid -ne [int]$Launch.launcher_pid
    ) {
        throw "worker result identity does not match the frozen launch"
    }
    $Completed = [DateTimeOffset]::Parse([string]$Result.completed_at_utc)
    $Deadline = [DateTimeOffset]::Parse([string]$Result.deadline_at_utc)
    $Result | Add-Member -NotePropertyName worker_diagnostics -NotePropertyValue ([pscustomobject][ordered]@{
        status = if ($Result.worker_exit_code -eq 0 -and $Completed -le $Deadline) { "PASS" } else { "FAIL" }
        reason = if ($Completed -le $Deadline) { $null } else { "worker completed after the frozen manifest deadline" }
        deadline_missed = $Completed -gt $Deadline
    }) -Force
    return $Result
}

function Record-WorkerHarvest {
    param([string]$StatePath, [string]$SessionId, [object]$Result)
    return Update-RunStateLocked $StatePath {
        param($Current)
        $Receipt = Get-FrozenSessionReceipt $Current $SessionId
        if (
            $null -eq $Receipt.worker_launch -or
            [string]$Receipt.worker_launch.worker_identity -ne [string]$Result.worker_identity -or
            [int]$Receipt.worker_launch.launcher_pid -ne [int]$Result.launcher_pid
        ) {
            throw "Worker harvest identity does not match the frozen launch."
        }
        $Receipt.worker_launch.status = if ($Result.worker_exit_code -eq 0 -and -not $Result.worker_diagnostics.deadline_missed) { "COMPLETED" } else { "FAILED" }
        $Receipt.worker_launch.completed_at_utc = [string]$Result.completed_at_utc
        $Receipt.worker_launch.exit_code = $Result.worker_exit_code
        $Receipt.worker_launch.deadline_missed = [bool]$Result.worker_diagnostics.deadline_missed
    }
}

function Invoke-WorkerMode {
    $Manifest = Verify-Packet
    if ([string]::IsNullOrWhiteSpace($ResultRoot) -or [string]::IsNullOrWhiteSpace($WorkerSessionId) -or [string]::IsNullOrWhiteSpace($WorkerIdentity)) {
        throw "Worker mode requires exact result, session, and worker identities."
    }
    $ResolvedResult = [IO.Path]::GetFullPath($ResultRoot)
    $StatePath = Join-Path $ResolvedResult "work\run-state.json"
    $State = $null
    $Launch = $null
    $RegistrationDeadline = [DateTimeOffset]::UtcNow.AddSeconds(10)
    while ([DateTimeOffset]::UtcNow -lt $RegistrationDeadline) {
        $State = Read-JsonFile $StatePath
        $Receipt = Get-FrozenSessionReceipt $State $WorkerSessionId
        if ($Receipt.PSObject.Properties.Name -contains "worker_launch") {
            $Launch = $Receipt.worker_launch
            if ([string]$Launch.worker_identity -eq $WorkerIdentity -and [int]$Launch.launcher_pid -eq $PID) { break }
        }
        Start-Sleep -Milliseconds 50
    }
    if ($null -eq $Launch -or [string]$Launch.worker_identity -ne $WorkerIdentity -or [int]$Launch.launcher_pid -ne $PID) {
        throw "Worker process identity was not atomically registered."
    }
    $null = Assert-WorkerProcessIdentity $State $Launch
    Assert-FrozenRunState $State $Manifest
    $Evidence = Join-Path $ResolvedResult "evidence"
    $Logs = Join-Path $Evidence "logs"
    $ServerExitPath = Join-Path $ResolvedResult "work\server-exit.json"
    try {
        $Result = Invoke-SessionWorker $State $WorkerSessionId $ServerExitPath $Evidence $Logs $WorkerIdentity ([string]$Launch.execution_mode -eq "live_priority")
        $Result["command_identity_sha256"] = [string]$Launch.command_identity_sha256
        $Result["launcher_pid"] = $PID
        $Result["worker_launch"] = $Launch
        Write-JsonFileAtomic $Result ([string]$Launch.result_path)
        exit 0
    } catch {
        $Failure = New-WorkerFailureResult $State $WorkerSessionId $Launch ($_.Exception.GetType().Name + ": " + $_.Exception.Message)
        $Failure["command_identity_sha256"] = [string]$Launch.command_identity_sha256
        $Failure["launcher_pid"] = $PID
        $Failure["worker_launch"] = $Launch
        Write-JsonFileAtomic $Failure ([string]$Launch.result_path)
        exit 2
    }
}

function End-FieldSession {
    $Manifest = Verify-Packet
    if ([string]::IsNullOrWhiteSpace($ResultRoot)) {
        throw "EndSession requires -ResultRoot from the Start command."
    }
    $ResolvedResult = [IO.Path]::GetFullPath($ResultRoot)
    $Work = Join-Path $ResolvedResult "work"
    $Evidence = Join-Path $ResolvedResult "evidence"
    $Logs = Join-Path $Evidence "logs"
    $StatePath = Join-Path $Work "run-state.json"
    $SecretPath = Join-Path $Work "control-token.txt"
    if (-not (Test-Path -LiteralPath $StatePath -PathType Leaf) -or -not (Test-Path -LiteralPath $SecretPath -PathType Leaf)) {
        throw "Result root is not an active packet result."
    }
    $State = Read-JsonFile $StatePath
    if ($State.result_root -ne $ResolvedResult -or $State.rule_profile -ne $RuleProfile) {
        throw "Result identity or rule profile mismatch."
    }
    Assert-FrozenRunState $State $Manifest
    $FrozenSessionId = [string]$State.current_session_id
    if ([string]::IsNullOrWhiteSpace($FrozenSessionId)) { throw "Frozen current session is missing." }
    $Token = Get-Content -LiteralPath $SecretPath -Raw -Encoding ASCII
    $FrozenReport = Complete-FrozenSession $State $FrozenSessionId $Token $Evidence
    if ($KeepRunning) {
        $NewSessionId = Wait-ForSessionRollover $State $FrozenSessionId $Token
        $State = Update-RunStateLocked $StatePath {
            param($Current)
            if ([string]$Current.current_session_id -ne $FrozenSessionId) {
                throw "Frozen current session changed during rollover state update."
            }
            if (@($Current.session_receipts | Where-Object { [string]$_.session_id -eq $NewSessionId }).Count -ne 0) {
                throw "Rollover session was already registered."
            }
            $Current.current_session_id = $NewSessionId
            $Current.session_receipts = @($Current.session_receipts) + @(
                [pscustomobject][ordered]@{
                    session_id = $NewSessionId
                    join_qr_sha256 = $Current.join_qr_sha256
                    worker_launch = $null
                }
            )
        }
        try {
            $State = Start-SessionWorkerBackground $StatePath $State $FrozenSessionId "live_priority" $Evidence $Logs
        } catch {
            Write-Warning ("Rollover remains available, but the ended-session worker could not be registered: " + $_.Exception.Message)
            $State = Read-JsonFile $StatePath
        }
        $null = Invoke-FieldGateEvaluation -State $State -SessionResults @() -Evidence $Evidence -Phase "INTERIM"
        Write-Host "ROLLOVER READY: $FrozenSessionId -> $NewSessionId" -ForegroundColor Green
        if ($null -ne $State.join_qr_sha256) {
            Write-Host "QR SHA256 UNCHANGED: $($State.join_qr_sha256)" -ForegroundColor Green
        } else {
            Write-Host "QR REUSE: NOT RUN (LocalOnly without a join QR)" -ForegroundColor Yellow
        }
        Write-Host "INTERIM FIELD GATES: $Evidence\field-gates.json (uncompleted gates remain NOT_RUN)" -ForegroundColor Yellow
        return
    }

    if ($null -ne $State.join_qr_sha256) {
        $QrHash = (Get-FileHash -LiteralPath $State.join_qr_asset -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($QrHash -ne $State.join_qr_sha256) { throw "Join QR changed before finalization." }
    }
    $SessionIds = @($State.session_receipts | ForEach-Object { [string]$_.session_id })
    if ($SessionIds.Count -eq 0 -or @($SessionIds | Select-Object -Unique).Count -ne $SessionIds.Count) {
        throw "Frozen session receipt inventory is empty or duplicated."
    }
    if (@($State.session_receipts | Where-Object { $_.join_qr_sha256 -ne $State.join_qr_sha256 }).Count -ne 0) {
        throw "Frozen sessions do not share the same join QR hash."
    }
    $SessionResultsById = @{}
    foreach ($SessionReceipt in @($State.session_receipts)) {
        if ($null -ne $SessionReceipt.worker_launch) {
            $ExistingResult = Wait-SessionWorkerResult $State $SessionReceipt
            $SessionResultsById[[string]$SessionReceipt.session_id] = $ExistingResult
            $State = Record-WorkerHarvest $StatePath ([string]$SessionReceipt.session_id) $ExistingResult
            $ExistingResult.worker_launch = (Get-FrozenSessionReceipt $State ([string]$SessionReceipt.session_id)).worker_launch
        }
    }
    Set-Content -LiteralPath (Join-Path $Work "stop.request") -Value ([DateTimeOffset]::UtcNow.ToString("o")) -Encoding ASCII
    $ServerExitPath = Join-Path $Work "server-exit.json"
    $ServerStopDeadline = [DateTimeOffset]::UtcNow.AddSeconds(20)
    while (-not (Test-Path -LiteralPath $ServerExitPath -PathType Leaf) -and [DateTimeOffset]::UtcNow -lt $ServerStopDeadline) {
        Start-Sleep -Milliseconds 100
    }
    if (-not (Test-Path -LiteralPath $ServerExitPath -PathType Leaf)) {
        throw "Field service did not stop after durable session end."
    }
    Copy-Item -LiteralPath $ServerExitPath -Destination (Join-Path $Evidence "server-exit.json") -Force
    $State = Read-JsonFile $StatePath
    foreach ($SessionReceipt in @($State.session_receipts)) {
        $SessionId = [string]$SessionReceipt.session_id
        if ($null -eq $SessionReceipt.worker_launch) {
            $State = Start-SessionWorkerBackground $StatePath $State $SessionId "offline" $Evidence $Logs
            $SessionReceipt = Get-FrozenSessionReceipt $State $SessionId
            $SessionResultsById[$SessionId] = Wait-SessionWorkerResult $State $SessionReceipt
            $State = Record-WorkerHarvest $StatePath $SessionId $SessionResultsById[$SessionId]
            $SessionResultsById[$SessionId].worker_launch = (Get-FrozenSessionReceipt $State $SessionId).worker_launch
        }
    }
    $SessionResults = @($SessionIds | ForEach-Object { $SessionResultsById[[string]$_] })
    $FieldGates = Invoke-FieldGateEvaluation -State $State -SessionResults $SessionResults -Evidence $Evidence -Phase "FINAL"
    $RuleAuthorityAccepted = $FieldGates.rule_authority_accepted -eq $true
    $Accepted = $FieldGates.accepted -eq $true
    $Summary = [ordered]@{
        schema_version = "1.0"
        development_evidence_only = $true
        production_ready = $false
        initial_session_id = $State.initial_session_id
        final_session_id = $FrozenSessionId
        distinct_session_count = @($SessionIds | Select-Object -Unique).Count
        join_qr_sha256 = $State.join_qr_sha256
        rule_profile = $RuleProfile
        rule_authority_accepted = $RuleAuthorityAccepted
        local_only = [bool]$State.local_only
        observation_only = [bool]$State.observation_only
        identity_gate_status = [string]$State.identity_gate_status
        runtime_environment = [string]$State.runtime_environment
        cloudbase_environment_id = [string]$State.cloudbase_environment_id
        server_exit_code = (Read-JsonFile $ServerExitPath).exit_code
        sessions = $SessionResults
        field_gates_path = "field-gates.json"
        all_required_field_gates_pass = $FieldGates.all_required_pass -eq $true
        accepted = $Accepted
    }
    Write-JsonFile $Summary (Join-Path $Evidence "summary.json")
    $DatabaseEvidence = Join-Path $Evidence "databases"
    New-Item -ItemType Directory -Force -Path $DatabaseEvidence | Out-Null
    Copy-Item -LiteralPath $State.state_db -Destination (Join-Path $DatabaseEvidence "field-state.sqlite3")
    Copy-Item -LiteralPath $State.jobs_db -Destination (Join-Path $DatabaseEvidence "analysis-jobs.sqlite3")
    Write-JsonFile ([ordered]@{
        schema_version = "1.0"
        development_evidence_only = $true
        production_ready = $false
        packet_manifest_sha256 = (Get-FileHash -LiteralPath (Join-Path $Evidence "packet-manifest.json") -Algorithm SHA256).Hash.ToLowerInvariant()
        server_exit_sha256 = (Get-FileHash -LiteralPath (Join-Path $Evidence "server-exit.json") -Algorithm SHA256).Hash.ToLowerInvariant()
        field_gates_sha256 = (Get-FileHash -LiteralPath (Join-Path $Evidence "field-gates.json") -Algorithm SHA256).Hash.ToLowerInvariant()
        state_db_sha256 = (Get-FileHash -LiteralPath (Join-Path $DatabaseEvidence "field-state.sqlite3") -Algorithm SHA256).Hash.ToLowerInvariant()
        jobs_db_sha256 = (Get-FileHash -LiteralPath (Join-Path $DatabaseEvidence "analysis-jobs.sqlite3") -Algorithm SHA256).Hash.ToLowerInvariant()
        session_ids = $SessionIds
        audited_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
    }) (Join-Path $Evidence "state-audit.json")
    $SendBack = Join-Path (Split-Path -Parent $ResolvedResult) ("SEND_BACK-" + (Split-Path -Leaf $ResolvedResult) + ".zip")
    Compress-Archive -LiteralPath $Evidence -DestinationPath $SendBack -CompressionLevel Optimal -Force
    Write-Host "SEND BACK: $SendBack" -ForegroundColor Green
    Write-Host "SEND BACK SHA256: $((Get-FileHash -LiteralPath $SendBack -Algorithm SHA256).Hash.ToLowerInvariant())"
    if (-not $Accepted) {
        Write-Host "ONE OR MORE REQUIRED FIELD GATES WERE NOT ACCEPTED" -ForegroundColor Red
        exit 2
    }
    Write-Host "ALL REQUIRED DEVELOPMENT FIELD GATES ACCEPTED" -ForegroundColor Green
    exit 0
}

if ($MyInvocation.InvocationName -eq ".") { return }
if ($Mode -eq "Start") { Start-FieldSession }
elseif ($Mode -eq "EndSession") { End-FieldSession }
else { Invoke-WorkerMode }
