# Read-only, finite diagnostic for one already-started actual screen case.
# Does not launch models, change priority/settings, or alter the owned deadlines.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateRange(1,2147483647)][int]$ControllerId,
    [Parameter(Mandatory=$true)][long]$ControllerCreatedUtcTicks,
    [Parameter(Mandatory=$true)][string]$TraceFile,
    [Parameter(Mandatory=$true)][string]$OutputFile
)
$ErrorActionPreference = 'Stop'
$Clock = [Diagnostics.Stopwatch]::StartNew()
$Utf8 = New-Object Text.UTF8Encoding($false, $true)
$ModelId = $null
$ModelCreated = $null
$EpochTicks = [DateTime]::SpecifyKind([DateTime]'1970-01-01', [DateTimeKind]::Utc).Ticks
foreach ($Path in @($TraceFile, $OutputFile)) {
    if ($Path -notmatch '^[A-Za-z]:[\\/]' -or $Path -match '[\r\n]' -or
        (($Path -split '[\\/]') -contains '..')) { throw 'Local absolute path required' }
    $Drive = New-Object IO.DriveInfo ($Path.Substring(0, 3))
    if ($Drive.DriveType -eq [IO.DriveType]::Network) { throw 'Network drive refused' }
    $Current = [IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($Path))
    while ($Current) {
        if ([IO.Directory]::Exists($Current)) {
            $Item = Get-Item -LiteralPath $Current -Force
            if ($Item.Name -ieq '.project-archive' -or
                ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                throw 'Unsafe diagnostic path'
            }
        }
        $Current = [IO.Path]::GetDirectoryName($Current)
    }
}
$Stream = [IO.File]::Open($OutputFile, [IO.FileMode]::CreateNew,
    [IO.FileAccess]::Write, [IO.FileShare]::Read)
function Write-Sample($Value) {
    $Bytes = $Utf8.GetBytes(($Value | ConvertTo-Json -Compress -Depth 6) + "`n")
    if ($Stream.Length + $Bytes.Length -gt 131072) { throw 'Diagnostic output budget exceeded' }
    $Stream.Write($Bytes, 0, $Bytes.Length)
    $Stream.Flush()
}
try {
    Write-Sample ([ordered]@{event='START'; utc=[DateTime]::UtcNow.ToString('o');
        controller_id=$ControllerId; controller_created_utc_ticks=$ControllerCreatedUtcTicks;
        cadence_seconds=5; max_seconds=80; cim_timeout_seconds=2;
        scope='owned-model-and-global-memory-only'; io_counters='NOT_CAPTURED'})
    # Leave two seconds for the final bounded CIM query; never wait for a model.
    while ($Clock.Elapsed.TotalSeconds -lt 78) {
        $Iteration = $Clock.Elapsed.TotalSeconds
        $Controller = $null
        $Model = $null
        $Row = [ordered]@{event='WAITING_FOR_MODEL'; seconds=$Iteration;
            utc=[DateTime]::UtcNow.ToString('o')}
        $Stop = $false
        try {
            $Controller = [Diagnostics.Process]::GetProcessById($ControllerId)
            if ($Controller.StartTime.ToUniversalTime().Ticks -ne $ControllerCreatedUtcTicks) {
                $Row.event='CONTROLLER_IDENTITY_CHANGED'; $Stop=$true
            } elseif ($Controller.HasExited) {
                $Row.event='CONTROLLER_EXITED'; $Stop=$true
            } else {
                if ($null -eq $ModelId -and [IO.File]::Exists($TraceFile)) {
                    $Info = Get-Item -LiteralPath $TraceFile -Force
                    if (($Info.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
                        $Info.Length -gt 65536) { throw 'Unsafe or excessive call trace' }
                    # Incomplete last lines are ignored until the next sample.
                    $Text = [IO.File]::ReadAllText($TraceFile, $Utf8)
                    foreach ($Line in ($Text -split "`n" | Select-Object -SkipLast 1)) {
                        $Entry = $Line | ConvertFrom-Json
                        if ($Entry.event -ceq 'factory_enter') {
                            if ($Entry.pid -isnot [int] -or $Entry.pid -le 0 -or
                                $Entry.wall_time_ns -isnot [long]) { throw 'Invalid factory identity' }
                            $Model = [Diagnostics.Process]::GetProcessById($Entry.pid)
                            $Created = $Model.StartTime.ToUniversalTime().Ticks
                            $EventTicks = $EpochTicks + [long][Math]::Floor($Entry.wall_time_ns / 100)
                            if ($Created -lt $ControllerCreatedUtcTicks -or
                                $Created -gt $EventTicks) { throw 'Factory PID creation mismatch' }
                            $ModelId = $Entry.pid; $ModelCreated = $Created
                            break
                        }
                    }
                }
                if ($null -ne $ModelId) {
                    if ($null -eq $Model) { $Model = [Diagnostics.Process]::GetProcessById($ModelId) }
                    if ($Model.StartTime.ToUniversalTime().Ticks -ne $ModelCreated) {
                        $Row.event='MODEL_IDENTITY_CHANGED'; $Stop=$true
                    } elseif ($Model.HasExited) {
                        $Row.event='MODEL_EXITED'; $Stop=$true
                    } else {
                        $Model.Refresh()
                        $Row.event='RUNNING'; $Row.model_id=$ModelId
                        $Row.model_created_utc_ticks=$ModelCreated
                        $Row.process_read_utc=[DateTime]::UtcNow.ToString('o')
                        $Row.cpu_ticks=$Model.TotalProcessorTime.Ticks
                        $Row.working_set_bytes=$Model.WorkingSet64
                        $Row.private_memory_bytes=$Model.PrivateMemorySize64
                        $Row.thread_count=$Model.Threads.Count
                        try {
                            $Memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -OperationTimeoutSec 2
                            $Row.memory_read_utc=[DateTime]::UtcNow.ToString('o')
                            $Row.available_mib=$Memory.AvailableMBytes
                            $Row.committed_bytes=$Memory.CommittedBytes
                            $Row.commit_limit=$Memory.CommitLimit
                            $Row.page_reads_per_second=$Memory.PageReadsPersec
                            $Row.memory_read='AVAILABLE'
                        } catch { $Row.memory_read='READ_ERROR' }
                        $Row.model_alive_after_memory_read= -not $Model.HasExited
                    }
                }
            }
        } catch {
            # An exited process can disappear before GetProcessById/StartTime.
            $Row.event='READ_ERROR'; $Row.error_type=$_.Exception.GetType().FullName
            if ($null -ne $ModelId) { $Stop=$true }
        } finally {
            if ($null -ne $Model) { $Model.Dispose() }
            if ($null -ne $Controller) { $Controller.Dispose() }
        }
        $Row.read_seconds=$Clock.Elapsed.TotalSeconds - $Iteration
        Write-Sample $Row
        if ($Stop) { break }
        $Remaining = [Math]::Min(5 - $Row.read_seconds, 78 - $Clock.Elapsed.TotalSeconds)
        if ($Remaining -gt 0) { Start-Sleep -Milliseconds ([int]($Remaining * 1000)) }
    }
    Write-Sample ([ordered]@{event='STOP'; seconds=$Clock.Elapsed.TotalSeconds;
        utc=[DateTime]::UtcNow.ToString('o'); model_bound=($null -ne $ModelId)})
} finally { $Stream.Dispose(); $Clock.Stop() }
