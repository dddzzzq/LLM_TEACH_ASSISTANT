param()
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$swapDirectory = Join-Path $projectRoot '.runtime\wsl'
New-Item -ItemType Directory -Path $swapDirectory -Force | Out-Null
$configPath = Join-Path ([Environment]::GetFolderPath('UserProfile')) '.wslconfig'
$content = ''
if (Test-Path -LiteralPath $configPath) {
    $content = [IO.File]::ReadAllText($configPath)
}
# Preserve other sections while configuring D-drive swap and Windows networking.
$swapPath = (Join-Path $swapDirectory 'swap.vhdx').Replace('\', '\\')
$desired = [ordered]@{
    swapFile = $swapPath
    networkingMode = 'mirrored'
    autoProxy = 'true'
    dnsTunneling = 'true'
}
$lines = [Collections.Generic.List[string]]::new()
$insideWsl = $false
$hasWsl = $false
$seen = @{}
foreach ($line in ($content -split '\r?\n')) {
    if ($line -match '^\s*\[(.+)\]\s*$') {
        if ($insideWsl) {
            foreach ($key in $desired.Keys) {
                if (-not $seen.ContainsKey($key)) { $lines.Add($key + '=' + $desired[$key]); $seen[$key] = $true }
            }
        }
        $insideWsl = $Matches[1] -eq 'wsl2'
        $hasWsl = $hasWsl -or $insideWsl
    }
    if ($insideWsl -and $line -match '^\s*([^=\s]+)\s*=' -and $desired.Contains($Matches[1])) {
        $key = $Matches[1]
        $lines.Add($key + '=' + $desired[$key])
        $seen[$key] = $true
    } else {
        $lines.Add($line)
    }
}
if (-not $hasWsl) {
    $lines.Add('[wsl2]')
    $lines.Add('memory=20GB')
    $lines.Add('processors=12')
    $lines.Add('swap=8GB')
}
foreach ($key in $desired.Keys) {
    if (-not $seen.ContainsKey($key)) { $lines.Add($key + '=' + $desired[$key]) }
}
[IO.File]::WriteAllText($configPath, ($lines -join "`r`n"), [Text.UTF8Encoding]::new($false))
Write-Output "WSL swap configured on the project drive; mirrored networking and automatic proxy enabled: $swapDirectory"
