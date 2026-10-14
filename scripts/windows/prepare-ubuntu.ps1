param([string]$Distribution = 'TeachingUbuntu')
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$bootstrap = Join-Path $projectRoot '.runtime\bootstrap'
$installDirectory = Join-Path $projectRoot ('.runtime\wsl\' + $Distribution)
New-Item -ItemType Directory -Path $bootstrap -Force | Out-Null
$manifestPath = Join-Path $bootstrap 'distributions.json'
if (-not (Test-Path -LiteralPath $manifestPath)) {
    Invoke-WebRequest 'https://raw.githubusercontent.com/microsoft/WSL/master/distributions/DistributionInfo.json' -OutFile $manifestPath
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$ubuntu = $manifest.ModernDistributions.Ubuntu | Where-Object Name -eq 'Ubuntu-24.04'
if (-not $ubuntu) { throw 'Ubuntu-24.04 is missing from the official WSL distribution manifest.' }
$archivePath = Join-Path $bootstrap 'ubuntu-24.04-amd64.wsl'
if (-not (Test-Path -LiteralPath $archivePath) -or
    (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ubuntu.Amd64Url.Sha256) {
    & curl.exe --fail --location --retry 2 --connect-timeout 20 --max-time 1200 --output $archivePath $ubuntu.Amd64Url.Url
    if ($LASTEXITCODE -ne 0) { throw "Ubuntu download failed: $LASTEXITCODE" }
}
if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ubuntu.Amd64Url.Sha256) {
    throw 'Ubuntu image checksum mismatch.'
}
$registry = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss'
$existing = @()
if (Test-Path $registry) {
    $existing = @(Get-ChildItem $registry | Get-ItemProperty | Where-Object DistributionName -eq $Distribution)
}
if (-not $existing.Count) {
    New-Item -ItemType Directory -Path $installDirectory -Force | Out-Null
    & wsl.exe --import $Distribution $installDirectory $archivePath --version 2
    if ($LASTEXITCODE -ne 0) { throw "WSL import failed: $LASTEXITCODE. A Windows restart may be required." }
}
& wsl.exe --distribution $Distribution --user root --exec uname -a
if ($LASTEXITCODE -ne 0) { throw 'The distribution cannot boot yet. Save your work and restart Windows before continuing.' }
