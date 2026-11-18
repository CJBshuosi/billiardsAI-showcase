# Developer entry for an audited code-only synchronization clone, not an installed product.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$Commit,
    [Parameter(Mandatory=$true)][string]$Config,
    [ValidateSet('Preflight','Fullgame','FullgameComplete','CaptureEnd')][string]$Profile = 'Preflight'
)
$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
function Assert-LocalAbsolute([string]$Path) {
    # Bootstrap reads happen before the owned setup deadline. Require local drive paths;
    # do not normalize a relative, drive-relative, UNC or device path into acceptance.
    if ([string]::IsNullOrWhiteSpace($Path) -or $Path -notmatch '^[A-Za-z]:[\\/]' `
        -or $Path.Contains("`n") -or $Path.Contains("`r") `
        -or (($Path -split '[\\/]') -contains '..')) {
        throw 'Local absolute path required'
    }
    foreach ($Part in ($Path -split '[\\/]')) {
        if ($Part.TrimEnd([char[]]' .') -ieq '.project-archive') {
            throw 'Archive path refused'
        }
    }
    $Drive = New-Object IO.DriveInfo ($Path.Substring(0, 3))
    if ($Drive.DriveType -eq [IO.DriveType]::Network) { throw 'Network drive refused' }
    $Current = [IO.Path]::GetFullPath($Path)
    $Absolute = $Current
    $Parents = @()
    while ($Current) {
        $Parents = @($Current) + $Parents
        $Current = [IO.Path]::GetDirectoryName($Current)
    }
    # Refuse a junction before accessing any descendant through it.
    foreach ($Current in $Parents) {
        $Item = Get-Item -LiteralPath $Current -Force
        if ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw 'Reparse path refused'
        }
    }
    return $Absolute
}
function Read-BoundedLocalFile([string]$Path, [int]$Limit) {
    $Absolute = Assert-LocalAbsolute $Path
    $Item = Get-Item -LiteralPath $Absolute -Force
    if ($Item -isnot [IO.FileInfo] -or $Item.Length -gt $Limit) {
        throw 'Bootstrap file size budget exceeded'
    }
    $Buffer = New-Object byte[] ($Limit + 1)
    $Stream = [IO.File]::Open($Absolute, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    try {
        $Count = 0
        while ($Count -lt $Buffer.Length) {
            $Read = $Stream.Read($Buffer, $Count, $Buffer.Length - $Count)
            if ($Read -eq 0) { break }
            $Count += $Read
        }
        if ($Count -gt $Limit) { throw 'Bootstrap file size budget exceeded' }
        $Result = New-Object byte[] $Count
        [Array]::Copy($Buffer, $Result, $Count)
        return ,$Result
    }
    finally { $Stream.Dispose() }
}
$Config = Assert-LocalAbsolute $Config
$ConfigBytes = Read-BoundedLocalFile $Config (2 * 1024 * 1024)
$Machine = [Text.Encoding]::UTF8.GetString($ConfigBytes) | ConvertFrom-Json
if ($Machine.contract -cne 'blackbox-dev-machine/v1') { throw 'Invalid machine configuration' }
# The parent creates the report/lock here before either Job starts.
$null = Assert-LocalAbsolute $Machine.runs
$null = Assert-LocalAbsolute $Machine.ownership
$Bootstrap = Assert-LocalAbsolute $Machine.bootstrap_python
if ($Machine.bootstrap_python_sha256 -cnotmatch '^[0-9a-f]{64}$') {
    throw 'Bootstrap SHA256 required'
}
$BootstrapBytes = Read-BoundedLocalFile $Bootstrap (8 * 1024 * 1024)
$Hasher = [Security.Cryptography.SHA256]::Create()
try {
    $Digest = [BitConverter]::ToString($Hasher.ComputeHash($BootstrapBytes)).Replace('-', '').ToLowerInvariant()
}
finally { $Hasher.Dispose() }
if ($Digest -cne $Machine.bootstrap_python_sha256) { throw 'Bootstrap runtime identity changed' }
# Git identity, clean HEAD, allowed-branch ancestry, hashes and copying are checked in
# Python's owned setup Job. Never invoke Git here or switch this running script's files.
$Entry = Join-Path $Repo 'scripts/development/blackbox_validation.py'
$Entry = Assert-LocalAbsolute $Entry
& $Bootstrap -I -B $Entry --repo $Repo --commit $Commit --config $Config --profile $Profile
exit $LASTEXITCODE
