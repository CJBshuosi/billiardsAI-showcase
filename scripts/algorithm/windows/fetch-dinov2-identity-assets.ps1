[CmdletBinding()]
param(
    [string]$OutputRoot = (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "billiardsAI\approved-assets\dinov2-small-ed25f3a")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Revision = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
$BaseUrl = "https://huggingface.co/facebook/dinov2-small/resolve/$Revision"
$Assets = @(
    [pscustomobject]@{ Name = "model.safetensors"; Bytes = 88249960; Sha256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1" },
    [pscustomobject]@{ Name = "config.json"; Bytes = 547; Sha256 = "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d" },
    [pscustomobject]@{ Name = "preprocessor_config.json"; Bytes = 436; Sha256 = "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828" }
)

[IO.Directory]::CreateDirectory($OutputRoot) | Out-Null
foreach ($Asset in $Assets) {
    $Target = Join-Path $OutputRoot $Asset.Name
    $Partial = "$Target.partial"
    if (Test-Path -LiteralPath $Target -PathType Leaf) {
        $Existing = Get-Item -LiteralPath $Target
        $ExistingHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Target).Hash.ToLowerInvariant()
        if ($Existing.Length -eq $Asset.Bytes -and $ExistingHash -eq $Asset.Sha256) {
            continue
        }
        throw "Existing DINOv2 payload failed identity verification: $Target"
    }
    if (Test-Path -LiteralPath $Partial) {
        Remove-Item -LiteralPath $Partial -Force
    }
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/$($Asset.Name)" -OutFile $Partial
        $Downloaded = Get-Item -LiteralPath $Partial
        $DownloadedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Partial).Hash.ToLowerInvariant()
        if ($Downloaded.Length -ne $Asset.Bytes -or $DownloadedHash -ne $Asset.Sha256) {
            throw "Downloaded DINOv2 payload failed identity verification: $($Asset.Name)"
        }
        Move-Item -LiteralPath $Partial -Destination $Target
    }
    finally {
        if (Test-Path -LiteralPath $Partial) {
            Remove-Item -LiteralPath $Partial -Force
        }
    }
}

$Inventory = @($Assets | ForEach-Object {
    $Path = Join-Path $OutputRoot $_.Name
    [ordered]@{
        name = $_.Name
        bytes = (Get-Item -LiteralPath $Path).Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    }
})
$Manifest = [ordered]@{
    schema_version = "1.0"
    source = "facebook/dinov2-small"
    revision = $Revision
    license = "Apache-2.0"
    total_bytes = 88250943
    payloads = $Inventory
    use_scope = "local_identity_research_only"
}
$ManifestJson = $Manifest | ConvertTo-Json -Depth 5
[IO.File]::WriteAllText((Join-Path $OutputRoot "manifest.json"), $ManifestJson, (New-Object Text.UTF8Encoding($false)))
[pscustomobject]@{ Status = "PASS"; Output = $OutputRoot; PayloadCount = 3; TotalBytes = 88250943 } | ConvertTo-Json -Compress
