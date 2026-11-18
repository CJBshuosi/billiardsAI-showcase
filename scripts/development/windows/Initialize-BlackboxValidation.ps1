# Initialize local assets/configuration from an already verified Git synchronization clone.
# Run once; subsequent development uses Git fetch/select plus Invoke-BlackboxValidation.ps1.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Root,
    [Parameter(Mandatory=$true)][string]$AssetsRoot,
    [Parameter(Mandatory=$true)][string]$Dependencies,
    [Parameter(Mandatory=$true)][string]$Git,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$GitSha256,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$Transport,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$Source,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$LockSha,
    [Parameter(Mandatory=$true)][ValidateRange(1,2097152)][long]$LockBytes,
    [Parameter(Mandatory=$true)][ValidateRange(1,5000)][int]$ExpectedDependencyFiles,
    [Parameter(Mandatory=$true)][ValidateRange(1,536870912)][long]$ExpectedDependencyBytes
)
$ErrorActionPreference='Stop'
$Old=$AssetsRoot
$Repository=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
function Assert-PlainLocal([string]$Path) {
    if($Path -notmatch '^[A-Za-z]:[\\/]' -or ($Path -split '[\\/]') -contains '..') {throw 'Local absolute path required'}
    $Drive=New-Object IO.DriveInfo ($Path.Substring(0,3))
    if($Drive.DriveType -eq [IO.DriveType]::Network) {throw 'Network drive refused'}
    $Current=[IO.Path]::GetFullPath($Path);$Parents=@()
    while($Current){$Parents=@($Current)+$Parents;$Current=[IO.Path]::GetDirectoryName($Current)}
    foreach($Part in $Parents){
        if([IO.Path]::GetFileName($Part).TrimEnd([char[]]' .') -ieq '.project-archive'){throw 'Archive refused'}
        $Item=Get-Item -LiteralPath $Part -Force
        if($Item.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Reparse path refused'}
    }
}
function Pin-File([string]$Path,[long]$ExpectedBytes=-1,[string]$ExpectedSha='') {
    Assert-PlainLocal $Path
    $Before=Get-Item -LiteralPath $Path -Force
    if($Before -isnot [IO.FileInfo] -or $Before.Length -gt 536870912){throw 'Regular bounded file required'}
    if($ExpectedBytes -ge 0 -and $Before.Length -ne $ExpectedBytes){throw 'Asset size changed'}
    $Size=$Before.Length;$Stamp=$Before.LastWriteTimeUtc.Ticks
    $Sha=(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    $After=Get-Item -LiteralPath $Path -Force
    if($After.Length -ne $Size -or $After.LastWriteTimeUtc.Ticks -ne $Stamp){throw 'File changed during hash'}
    if($ExpectedSha -and $Sha -cne $ExpectedSha){throw 'Asset SHA changed'}
    return @{bytes=$Size;sha256=$Sha}
}
function Get-PlainFiles([string]$Path) {
    Assert-PlainLocal $Path
    $Item=Get-Item -LiteralPath $Path -Force
    if(-not $Item.PSIsContainer){
        if($Item.Extension -notin @('.pyc','.pth','._pth')){$Item}
        return
    }
    foreach($Child in Get-ChildItem -LiteralPath $Path -Force){
        # Check each directory before entering it; never recurse through a junction.
        Assert-PlainLocal $Child.FullName
        if($Child.PSIsContainer){
            if($Child.Name -cne '__pycache__'){Get-PlainFiles $Child.FullName}
        }elseif($Child.Extension -notin @('.pyc','.pth','._pth')){$Child}
    }
}
Assert-PlainLocal $Root
if([IO.Path]::GetFullPath((Join-Path $Root 'repository')) -ine $Repository){throw 'Unexpected initialization clone path'}
Assert-PlainLocal $Repository
$RootItems=@(Get-ChildItem -LiteralPath $Root -Force)
if($RootItems.Count -ne 1 -or $RootItems[0].Name -cne 'repository'){throw 'Root contains existing preparation or unexpected files; preserve it'}
if((New-Object IO.DriveInfo ([IO.Path]::GetPathRoot($Root))).AvailableFreeSpace -lt 10737418240){throw '10 GiB free space required'}
$GitPin=Pin-File $Git -ExpectedSha $GitSha256
$Head=& $Git -C $Repository rev-parse HEAD
if($LASTEXITCODE -ne 0 -or $Head.Trim() -cne $Transport){throw 'Transport identity changed'}
$Origin=& $Git -C $Repository remote get-url origin
if($LASTEXITCODE -ne 0 -or $Origin.Trim() -cne 'https://github.com/CJBshuosi/billiardsAI.git'){throw 'Unexpected origin'}
& $Git -C $Repository merge-base --is-ancestor $Transport refs/remotes/origin/codex/windows-blackbox-sync
if($LASTEXITCODE -ne 0){throw 'Transport outside synchronization branch'}
$Dirty=& $Git -C $Repository status --porcelain --untracked-files=all
if($LASTEXITCODE -ne 0 -or $Dirty){throw 'Checkout not clean'}
$null=Pin-File (Join-Path $Repository 'SOURCE-LOCK.json') $LockBytes $LockSha
$Lock=Get-Content -LiteralPath (Join-Path $Repository 'SOURCE-LOCK.json') -Raw|ConvertFrom-Json
if($Lock.commit -cne $Source -or $Lock.contract -cne 'recording-first-source-lock/v1'){throw 'Source lock identity changed'}
$Members=@($Lock.files.PSObject.Properties)
if($Members.Count -lt 1 -or $Members.Count -gt 1000){throw 'Source lock member budget'}
foreach($Member in $Members){$null=Pin-File (Join-Path $Repository $Member.Name) $Member.Value.bytes $Member.Value.sha256}
$Runtime=Join-Path $Old 'packet/runtime'
$RuntimePins=@{
    'LICENSE.txt'=@{bytes=36874;sha256='e502c6b880ff58d614901495a9009c136539cd0b1e2a2abb8fc00b934c203419'}
    '_asyncio.pyd'=@{bytes=74096;sha256='c77de90cad749087a0c5dc5e19d78d24f6ef44e7e5ac19b8420cfc0cea62395d'}
    '_bz2.pyd'=@{bytes=86896;sha256='65f610646d10d5babea080e060d244b546ec52c54012e5e937dc72dcc2e2e7bc'}
    '_ctypes.pyd'=@{bytes=127480;sha256='0e56107a891100535f2da79d85a5ae65ff1227b4765bd108f8593ac1a1110d81'}
    '_decimal.pyd'=@{bytes=262000;sha256='aeacfead2befcc0768063a037bb64bd32109ae09cb144e5f571b04dd7965db9d'}
    '_elementtree.pyd'=@{bytes=135160;sha256='5009a9a53c03ef8387443b5cf6ce08e5c3a79727466a0b931c5bbca915bf89ca'}
    '_hashlib.pyd'=@{bytes=68976;sha256='03d233dbace599168eaffec823de6ed7beec8c2a4ebfe9b8a3d8e042a59af3ba'}
    '_lzma.pyd'=@{bytes=161136;sha256='e3deecc1f59a3b811c608861decd84de1e8df8bafd5e2b39188861318c5a38cf'}
    '_msi.pyd'=@{bytes=46584;sha256='b49332bae154ebdd8c3d5e7ddfdde336d9c47f91ba028dc67bafb2420b3dbb5c'}
    '_multiprocessing.pyd'=@{bytes=37368;sha256='2c636df7059cb0dd7e978dd735c7c5c8ddb09cf94754cf9e232296b96c5c545a'}
    '_overlapped.pyd'=@{bytes=57336;sha256='662126ff80c3a1b4cedf74074029c12ae4743e6d3b37376e39ac70aacb7d4d3c'}
    '_queue.pyd'=@{bytes=33784;sha256='48e76e0ad63f6932b4a77b5819f41c1e7595636ca8d9e824f0943a14b8bb5035'}
    '_socket.pyd'=@{bytes=85496;sha256='3efa4636cf9e350856f6927d67a4c2564bf9bd2758a00bbf17d3aece5b60016d'}
    '_sqlite3.pyd'=@{bytes=126184;sha256='64b51df2805c58b023a875e8ae3be7ac7f61ef664f32ca6c90f6315da47ef0bc'}
    '_ssl.pyd'=@{bytes=179432;sha256='0bc527d2dbc50009f43b2ed5d5982c4f63f4b647647ffa9eda612dcda80f6040'}
    '_uuid.pyd'=@{bytes=27128;sha256='acce4f281d7ffb2dec0997527faf25b1df62ad7bd3032c568ddcf65c90edc867'}
    '_wmi.pyd'=@{bytes=40304;sha256='a980f1e7a29842bd946863fc2693fbeb555a9dc0f9e1630dfed9a2fcdef3ece2'}
    '_zoneinfo.pyd'=@{bytes=49656;sha256='76f1439cda82ab6976df6ba505c4079444703f6d44d8b0259ce59c0ff4279fb4'}
    'libcrypto-3.dll'=@{bytes=5231472;sha256='ccfffddcd3defb8d899026298af9af43bc186130f8483d77e97c93233d5f27d7'}
    'libffi-8.dll'=@{bytes=39696;sha256='eff52743773eb550fcc6ce3efc37c85724502233b6b002a35496d828bd7b280a'}
    'libssl-3.dll'=@{bytes=794992;sha256='007142039f04d04e0ed607bda53de095e5bc6a8a10d26ecedde94ea7d2d7eefe'}
    'pyexpat.pyd'=@{bytes=205304;sha256='d615e193d1affcdbfe504e90d3621cbdcff94bf72b79d899126534507fd6d27a'}
    'python.cat'=@{bytes=574231;sha256='5d85439736ae088d9680b391e2bca55ca7e9cc5b41a12ebe8e2f554506c7e72e'}
    'python.exe'=@{bytes=104952;sha256='4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a'}
    'python3.dll'=@{bytes=70376;sha256='fb975a606e7fbf74f64260e3f60c3490b4f74a183c0926fd6ed1ac4c52ac7b1c'}
    'python312.dll'=@{bytes=6945272;sha256='9a0e3435aaa680d868150f87ab3e388ad2eebc22f87e036155c7b4eda8cd2120'}
    'python312.zip'=@{bytes=3835482;sha256='fb131c0ef7e35cc5250a74c8cd18744bf4115fb8163710711f3758d7df3d1f88'}
    'pythonw.exe'=@{bytes=104304;sha256='d72294fb338bc2fc8896d25a7395a4db466425427e1559e77185d5135a830681'}
    'select.pyd'=@{bytes=33136;sha256='d4d7181b11db0bd22a6cd5cda6f8583d1f5c77615fd0a6ca33350619d6e0438a'}
    'sqlite3.dll'=@{bytes=1583608;sha256='8daa45684bf6688fca28d956b2e8942f2fa98f656675ce68e6d61dd869848a7a'}
    'unicodedata.pyd'=@{bytes=1140592;sha256='208ee1cd3cbad0828433ca7246b05be2b1ff5569983d27d296c0b38888585488'}
    'vcruntime140.dll'=@{bytes=120400;sha256='052ad6a20d375957e82aa6a3c441ea548d89be0981516ca7eb306e063d5027f4'}
    'vcruntime140_1.dll'=@{bytes=49776;sha256='6a99bc0128e0c7d6cbbf615fcc26909565e17d4ca3451b97f8987f9c6acbc6c8'}
    'winsound.pyd'=@{bytes=31736;sha256='100e8d676777603c7258daf9b2f23ef001f09f1596d8708de359326fff72bddd'}
}
foreach($Name in $RuntimePins.Keys){$null=Pin-File (Join-Path $Runtime $Name) $RuntimePins[$Name].bytes $RuntimePins[$Name].sha256}
$Ownership=Join-Path $Old 'packet/ownership.py'
$null=Pin-File $Ownership 8087 '76e90eb5490e7c899788a87040bf827bac7b5ae057efbcba59475efe9b011a8f'
$Codec=Join-Path $Old 'codec'
$null=Pin-File (Join-Path $Codec 'openh264-2.5.0-win64.dll') 986712 'bf91c488638b32448f7d89611f6bb9c232dce82975d7d2827c78f78fc2a7fb2b'
$null=Pin-File (Join-Path $Codec 'BINARY_LICENSE.txt') 4612 'bd9f363c5ea11ef723d0304cddacb5273c43c0e1194097c7a045d05273635418'
$Fixtures=Join-Path $Old 'packet/input'
$null=Pin-File (Join-Path $Fixtures 'input-preflight.json') 2705 '8f583bdad79865b7f670d69af70e5cbddb2e2329b77a5027927da8e179e3470a'
# Dependency pins freeze this observed installation; they do not certify its vendor provenance.
$PackageRoots=@('cv2','numpy','numpy.libs','pydantic','pydantic_core','annotated_types','typing_extensions.py','typing_inspection','yaml','_yaml')
Assert-PlainLocal $Dependencies
$Dependencies=[IO.Path]::GetFullPath($Dependencies)
$DependencyPrefix=$Dependencies.TrimEnd([char[]]@([char]92,[char]47))+[IO.Path]::DirectorySeparatorChar
$DependencyPins=[ordered]@{};$Total=0L
foreach($PackageRoot in $PackageRoots){
    $Path=Join-Path $Dependencies $PackageRoot;Assert-PlainLocal $Path
    $Files=@(Get-PlainFiles $Path)
    if($Files.Count -gt $ExpectedDependencyFiles){throw 'Dependency file count budget'}
    foreach($File in $Files){
        $FullName=[IO.Path]::GetFullPath($File.FullName)
        if(-not $FullName.StartsWith($DependencyPrefix,[StringComparison]::OrdinalIgnoreCase)){throw 'Dependency path escaped'}
        $Name=$FullName.Substring($DependencyPrefix.Length).Replace([char]92,[char]47)
        if($Name -cmatch '[\x00-\x1f\x7f-\uffff\\:<>"|?*]' -or ($Name -split '/') -contains '..' -or $DependencyPins.Contains($Name)){throw 'Dependency member refused'}
        if($File.Name -in @('sitecustomize.py','usercustomize.py')){throw 'Startup injection refused'}
        $Total+=$File.Length;if($Total -gt 536870912){throw 'Dependency size budget'}
        $DependencyPins[$Name]=Pin-File $File.FullName $File.Length
    }
}
if($DependencyPins.Count -ne $ExpectedDependencyFiles -or $Total -ne $ExpectedDependencyBytes){throw 'Observed dependency inventory changed'}
$Runs=Join-Path $Root 'runs';$null=New-Item -ItemType Directory -Path $Runs
$Machine=[ordered]@{
    contract='blackbox-dev-machine/v1';bootstrap_python=(Join-Path $Runtime 'python.exe')
    bootstrap_python_sha256=$RuntimePins['python.exe'].sha256;git_executable=$Git;git_sha256=$GitPin.sha256
    runtime=$Runtime;runtime_pins=$RuntimePins;dependencies=$Dependencies;dependency_pins=$DependencyPins
    fixtures=$Fixtures;codec=$Codec;ownership=$Ownership;runs=$Runs
}
$Config=Join-Path $Root 'machine.private.json'
$Json=$Machine|ConvertTo-Json -Depth 8
if([Text.Encoding]::UTF8.GetByteCount($Json) -gt 2097152){throw 'Config size budget'}
$ConfigBytes=(New-Object Text.UTF8Encoding $false).GetBytes($Json)
$ConfigStream=[IO.File]::Open($Config,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try{$ConfigStream.Write($ConfigBytes,0,$ConfigBytes.Length);$ConfigStream.Flush()}finally{$ConfigStream.Dispose()}
$ConfigPin=Pin-File $Config
[ordered]@{
    contract='blackbox-dev-preparation/v1';status='CONFIGURATION_READY';source_commit=$Source
    transport_commit=$Transport;source_lock_sha256=$LockSha;config=$Config;config_identity=$ConfigPin
    runtime_files=$RuntimePins.Count;dependency_files=$DependencyPins.Count;dependency_bytes=$Total
    product_executed=$false;old_once_reused=$false
}|ConvertTo-Json -Depth 5 -Compress
