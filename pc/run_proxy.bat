@echo off
setlocal
set VENV=%USERPROFILE%\moloch_pc_env
set HERE=%~dp0

if not exist "%VENV%\Scripts\python.exe" (
  echo [run_proxy] venv not found at %VENV% - run pc\setup.bat first
  exit /b 1
)

"%VENV%\Scripts\python.exe" "%HERE%adapter_inference_proxy.py"
endlocal
