@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title LLM Teaching Assistant - Stop

pushd "%~dp0"
if errorlevel 1 (
    echo 无法进入项目目录：%~dp0
    set "TEACH_EXIT_CODE=1"
    goto finish
)

echo 正在停止本机教学助理服务...
echo 请先结束正在运行的批改或浏览器任务。
echo.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\stop.ps1"
set "TEACH_EXIT_CODE=%ERRORLEVEL%"
popd

echo.
if "%TEACH_EXIT_CODE%"=="0" (
    echo 服务已停止。
) else (
    echo 停止失败，请查看上方错误信息和 WINDOWS_DEPLOYMENT.md 中的排障说明。
)

:finish
if /I not "%~1"=="--no-pause" pause
exit /b %TEACH_EXIT_CODE%
