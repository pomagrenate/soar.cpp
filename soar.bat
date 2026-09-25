@echo off
if exist "%~dp0build\soar_engine.exe" (
    "%~dp0build\soar_engine.exe" %*
) else (
    echo [ERROR] soar_engine.exe not found in build directory. Please build first using cmake.
    exit /b 1
)
