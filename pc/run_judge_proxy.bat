@echo off
setlocal
set VENV=%USERPROFILE%\moloch_pc_env
set HERE=%~dp0

if not exist "%VENV%\Scripts\python.exe" (
  echo [run_judge_proxy] venv not found at %VENV% - run pc\setup.bat first
  exit /b 1
)

REM judge_proxy als Modul (braucht moloch_repo-Root im sys.path)
cd /d "%HERE%\.."
"%VENV%\Scripts\python.exe" -m pc.judge_proxy
endlocal
