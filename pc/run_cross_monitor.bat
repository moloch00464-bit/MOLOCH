@echo off
setlocal
set VENV=%USERPROFILE%\moloch_pc_env
set HERE=%~dp0
title MOLOCH Cross-Session Monitor

if not exist "%VENV%\Scripts\python.exe" (
  echo [run_cross_monitor] venv not found at %VENV% - run pc\setup.bat first
  exit /b 1
)

"%VENV%\Scripts\python.exe" "%HERE%cross_session_monitor.py"
endlocal
