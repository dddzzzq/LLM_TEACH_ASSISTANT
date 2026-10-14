param([string]$Distribution = 'TeachingUbuntu')
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$drive = $projectRoot.Substring(0, 1).ToLowerInvariant()
$linuxRoot = '/mnt/' + $drive + $projectRoot.Substring(2).Replace('\', '/')
& wsl.exe --distribution $Distribution --user teacher --exec bash ($linuxRoot + '/scripts/start-wsl.sh')
if ($LASTEXITCODE -ne 0) { throw "Startup failed with exit code $LASTEXITCODE." }
