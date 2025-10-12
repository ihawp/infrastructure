@echo off
title Security Monitor - Simple Startup
echo Starting Security Monitor...
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Wait a bit for system to fully load
timeout /t 5 /nobreak >nul

:restart_loop
echo Starting security monitor...
set PYTHONIOENCODING=utf-8
"C:\Python312\python.exe" -u "%~dp0security_monitor.py" 1>> "%~dp0startup_log.txt" 2>>&1

REM If the script exits, restart it
echo Security monitor stopped. Restarting in 3 seconds...
timeout /t 3 /nobreak >nul
goto restart_loop
