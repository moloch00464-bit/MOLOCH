@echo off
setlocal
set VENV=%USERPROFILE%\moloch_pc_env
set HERE=%~dp0

if not exist "%VENV%\Scripts\python.exe" (
  echo [run_auto_researcher_server] venv not found at %VENV% - run pc\setup.bat first
  exit /b 1
)

cd /d "%HERE%\.."
"%VENV%\Scripts\python.exe" -m pc.auto_researcher_server
endlocal
