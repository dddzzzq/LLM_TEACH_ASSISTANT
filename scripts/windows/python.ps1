param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PythonArguments)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$env:UV_CACHE_DIR = Join-Path $projectRoot '.runtime\cache\uv-windows'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.runtime\python-windows'
$env:UV_PYTHON_BIN_DIR = Join-Path $projectRoot '.runtime\tools\python'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $projectRoot '.runtime\python-env-windows'
$env:TEMP = Join-Path $projectRoot '.runtime\tmp'
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$uv = Join-Path $projectRoot '.runtime\tools\uv\uv.exe'
if (-not (Test-Path -LiteralPath $uv)) { throw 'The project-local uv binary is not installed.' }
& $uv run --project (Join-Path $projectRoot 'ai_engine_python') --locked --no-dev python @PythonArguments
exit $LASTEXITCODE
