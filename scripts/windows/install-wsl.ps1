param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\..\.runtime\bootstrap'),
    [string]$MsiPath = ''
)

$ErrorActionPreference = 'Stop'
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$statusPath = Join-Path $OutputDirectory 'wsl-install-status.json'
$result = @{ completed = $false; restartRequired = $false; error = $null }
Start-Transcript -Path (Join-Path $OutputDirectory 'wsl-install.log') -Append | Out-Null
try {
    $principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'This script requires administrator privileges to enable Windows WSL features.'
    }
    foreach ($feature in @('VirtualMachinePlatform', 'Microsoft-Windows-Subsystem-Linux')) {
        $state = Get-WindowsOptionalFeature -Online -FeatureName $feature
        if ($state.State -eq 'EnablePending') {
            $result.restartRequired = $true
        } elseif ($state.State -ne 'Enabled') {
            $enabled = Enable-WindowsOptionalFeature -Online -FeatureName $feature -All -NoRestart
            $result.restartRequired = $result.restartRequired -or $enabled.RestartNeeded
        }
    }
    # Install only the Windows WSL component. The large Linux distribution is
    # installed separately with --location on the project drive.
    if ($MsiPath) {
        $MsiPath = (Resolve-Path -LiteralPath $MsiPath).Path
        $signature = Get-AuthenticodeSignature -LiteralPath $MsiPath
        if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Microsoft Corporation') {
            throw 'The WSL MSI does not have a valid Microsoft signature.'
        }
        $msiLog = Join-Path $OutputDirectory 'wsl-msi.log'
        $msi = Start-Process msiexec.exe -ArgumentList @('/i', ('"' + $MsiPath + '"'), '/qn', '/norestart', '/l*v', ('"' + $msiLog + '"')) -WindowStyle Hidden -Wait -PassThru
        $installExitCode = $msi.ExitCode
    } else {
        & wsl.exe --install --no-distribution --web-download
        $installExitCode = $LASTEXITCODE
    }
    if ($installExitCode -notin @(0, 3010)) {
        throw "WSL installation failed with exit code $installExitCode."
    }
    $result.restartRequired = $result.restartRequired -or ($installExitCode -eq 3010)
    $result.completed = $true
} catch {
    $result.error = $_.Exception.Message
    Write-Error $_ -ErrorAction Continue
} finally {
    $result | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
    Stop-Transcript | Out-Null
}
if (-not $result.completed) { exit 1 }
if ($result.restartRequired) { exit 3010 }
