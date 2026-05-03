@echo off
setlocal
set VENV=%USERPROFILE%\moloch_pc_env
set HERE=%~dp0

if not exist "%VENV%\Scripts\python.exe" (
  echo [run_avatar_v2] venv not found at %VENV% - run pc\setup.bat first
  exit /b 1
)

REM avatar_v2 als Modul (braucht moloch_repo-Root im sys.path)
cd /d "%HERE%\.."
"%VENV%\Scripts\python.exe" -m pc.avatar_v2.server
endlocal
