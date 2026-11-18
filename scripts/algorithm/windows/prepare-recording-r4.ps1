function Invoke-RecordingR4 {
param([ValidateSet('Prepare','Run')][string]$Mode,
      [string]$Zip,[string]$ZipSha,[Parameter(Mandatory=$true)][string]$ManifestSha,[string]$Root)
$ErrorActionPreference='Stop'
function Assert-Plain([string]$P) {
 $Current=[IO.Path]::GetFullPath($P)
 while($Current){
  $Item=Get-Item -LiteralPath $Current -Force
  if($Item.Attributes -band [IO.FileAttributes]::ReparsePoint){throw "Reparse point: $P"}
  $Current=[IO.Path]::GetDirectoryName($Current)
 }
}
function Hash([string]$P){(Get-FileHash -LiteralPath $P -Algorithm SHA256).Hash.ToLowerInvariant()}
function Safe-Name([string]$N){
 if(-not $N -or $N.Contains('\') -or $N.Contains(':') -or $N.StartsWith('/') -or $N.EndsWith('/') -or $N -match '[^\x20-\x7e]' -or $N -match '[<>"|?*]'){throw 'Unsafe ZIP name'}
 foreach($Part in $N.Split('/')){
  if(-not $Part -or $Part -in @('.','..','.git','.env','.project-archive') -or $Part -match '[. ]$' -or $Part -match '^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)'){throw 'Unsafe Windows path component'}
 }
}
function Check-Packet([string]$Packet){
 Assert-Plain $Packet
 $Manifest=Join-Path $Packet 'packet-manifest.json'
 Assert-Plain $Manifest
 if((Get-Item -LiteralPath $Manifest).Length -gt 2097152 -or (Hash $Manifest) -ne $ManifestSha){throw 'Manifest identity mismatch'}
 $M=Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
 if($M.contract -ne 'recording-first-windows-packet/v1'){throw 'Manifest contract mismatch'}
 $Names=@($M.members.PSObject.Properties.Name)+@('packet-manifest.json')
 $Actual=New-Object 'System.Collections.Generic.List[string]'
 $Directories=New-Object 'System.Collections.Generic.Stack[string]';$Directories.Push($Packet)
 while($Directories.Count -gt 0){
  $Directory=$Directories.Pop();Assert-Plain $Directory
  foreach($Item in Get-ChildItem -LiteralPath $Directory -Force){
   Assert-Plain $Item.FullName
   if($Item.PSIsContainer){$Directories.Push($Item.FullName)}else{$Actual.Add($Item.FullName.Substring($Packet.Length+1).Replace('\','/'))}
  }
 }
 if(@(Compare-Object ($Names | Sort-Object) ($Actual | Sort-Object)).Count -ne 0){throw 'Packet inventory mismatch'}
 foreach($Property in $M.members.PSObject.Properties){
  Safe-Name $Property.Name
  $P=Join-Path $Packet $Property.Name;Assert-Plain $P
  if((Get-Item -LiteralPath $P).Length -ne $Property.Value.bytes -or (Hash $P) -ne $Property.Value.sha256){throw "Member mismatch $($Property.Name)"}
 }
 foreach($P in @("$Packet\pyvenv.cfg","$Packet\runtime\pyvenv.cfg","$Packet\runtime\Lib")){
  if(Test-Path -LiteralPath $P){throw 'Unexpected startup prefix'}
 }
 if(@(Get-ChildItem -LiteralPath "$Packet\runtime" -File -Filter '*.pth' -Force).Count -ne 0){throw 'Extra startup .pth refused'}
}
$Documents='C:\Users\developer\Documents'
Assert-Plain $Documents
if($Mode -eq 'Prepare'){
 if(-not $Zip -or -not $ZipSha -or $Root){throw 'Prepare needs exact ZIP/SHA and no root'}
 Assert-Plain $Zip
 if((Hash $Zip) -ne $ZipSha){throw 'ZIP SHA mismatch'}
 Add-Type -AssemblyName System.IO.Compression.FileSystem
 $Archive=[IO.Compression.ZipFile]::OpenRead($Zip)
 try{
  $Names=New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
  if($Archive.Entries.Count -gt 2000){throw 'ZIP entry budget exceeded'}
  [long]$Total=0
  foreach($Entry in $Archive.Entries){
   Safe-Name $Entry.FullName
   if(-not $Names.Add($Entry.FullName) -or (($Entry.ExternalAttributes -shr 16) -band 61440) -eq 40960){throw 'Duplicate or link ZIP entry'}
   $Total+=$Entry.Length
   if($Entry.Length -gt 536870912 -or $Total -gt 734003200){throw 'ZIP expansion budget exceeded'}
  }
  $Root=Join-Path $Documents ('billiardsAI-recording-r4-'+[guid]::NewGuid().ToString())
  if(Test-Path -LiteralPath $Root){throw 'Root reuse refused'}
  [IO.Directory]::CreateDirectory($Root)|Out-Null
  $Packet=Join-Path $Root 'packet';[IO.Directory]::CreateDirectory($Packet)|Out-Null
  foreach($Entry in $Archive.Entries){
   $Target=[IO.Path]::GetFullPath((Join-Path $Packet $Entry.FullName))
   if(-not $Target.StartsWith($Packet+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Extraction escape'}
   [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Target))|Out-Null
   [IO.Compression.ZipFileExtensions]::ExtractToFile($Entry,$Target,$false)
  }
 }finally{$Archive.Dispose()}
 Check-Packet $Packet
 @{manifest_sha256=$ManifestSha;zip_sha256=$ZipSha;native_authorization_inferred=$false}|ConvertTo-Json|Set-Content -LiteralPath "$Root\PREPARED.json" -Encoding UTF8
 Write-Output $Root
}elseif($Mode -eq 'Run'){
 if([IO.Path]::GetDirectoryName($Root) -ne $Documents -or [IO.Path]::GetFileName($Root) -notmatch '^billiardsAI-recording-r4-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$'){throw 'Fresh expected Documents root required'}
 Assert-Plain $Root
 foreach($Name in @('once.json','output','recording','profile','temp','worker-stdout.txt','worker-stderr.txt','parent-stdout.txt','parent-stderr.txt')){
  if(Test-Path -LiteralPath (Join-Path $Root $Name)){throw 'Single run consumed or root not fresh'}
 }
 $Prepared=Get-Content -LiteralPath "$Root\PREPARED.json" -Raw | ConvertFrom-Json
 if($Prepared.manifest_sha256 -ne $ManifestSha){throw 'Prepared manifest mismatch'}
 $Packet=Join-Path $Root 'packet';Check-Packet $Packet
 & "$Packet\runtime\python.exe" -I -B "$Packet\entry.py" launch $Root $ManifestSha 1> "$Root\parent-stdout.txt" 2> "$Root\parent-stderr.txt"
 if($LASTEXITCODE -ne 0){throw "Owned R4 run failed ($LASTEXITCODE); preserve root; do not retry"}
}else{throw 'Explicit mode required'}
}
