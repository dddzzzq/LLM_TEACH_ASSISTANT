param([string]$Distribution = 'TeachingUbuntu')
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'configure-wsl.ps1')
& (Join-Path $PSScriptRoot 'prepare-ubuntu.ps1') -Distribution $Distribution
if ($LASTEXITCODE -ne 0) { throw 'Ubuntu preparation failed.' }
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$drive = $projectRoot.Substring(0, 1).ToLowerInvariant()
$linuxRoot = '/mnt/' + $drive + $projectRoot.Substring(2).Replace('\', '/')
& wsl.exe --distribution $Distribution --user root --exec bash ($linuxRoot + '/scripts/bootstrap-wsl.sh')
if ($LASTEXITCODE -ne 0) { throw "Deployment failed with exit code $LASTEXITCODE. Check the command output." }
& wsl.exe --distribution $Distribution --user teacher --exec bash ($linuxRoot + '/scripts/verify-wsl.sh')
if ($LASTEXITCODE -ne 0) { throw "WSL verification failed with exit code $LASTEXITCODE." }
& (Join-Path $PSScriptRoot 'start.ps1') -Distribution $Distribution
