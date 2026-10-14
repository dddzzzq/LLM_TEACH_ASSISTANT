@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title LLM Teaching Assistant - Start

pushd "%~dp0"
if errorlevel 1 (
    echo 无法进入项目目录：%~dp0
    set "TEACH_EXIT_CODE=1"
    goto finish
)

echo 正在启动本机教学助理服务，请等待模型加载完成...
echo.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\start.ps1"
set "TEACH_EXIT_CODE=%ERRORLEVEL%"
popd

echo.
if "%TEACH_EXIT_CODE%"=="0" (
    echo 启动成功，访问：http://127.0.0.1:6006
) else (
    echo 启动失败，请查看上方错误信息和 WINDOWS_DEPLOYMENT.md 中的排障说明。
)

:finish
if /I not "%~1"=="--no-pause" pause
exit /b %TEACH_EXIT_CODE%
