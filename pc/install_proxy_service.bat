@echo off
setlocal
set NAME=MolochAdapterProxy
set VENV=%USERPROFILE%\moloch_pc_env
set HERE=%~dp0
set LOGDIR=%USERPROFILE%\moloch_adapters

where nssm >nul 2>&1
if errorlevel 1 (
  echo [install] nssm not found in PATH.
  echo           Download: https://nssm.cc/download
  echo           Then either add nssm.exe to PATH or rerun from its directory.
  exit /b 2
)

if not exist "%VENV%\Scripts\python.exe" (
  echo [install] venv not found at %VENV% - run pc\setup.bat first.
  exit /b 3
)

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

nssm install %NAME% "%VENV%\Scripts\python.exe" "%HERE%adapter_inference_proxy.py"
nssm set %NAME% AppDirectory "%HERE%"
nssm set %NAME% AppStdout "%LOGDIR%\proxy.log"
nssm set %NAME% AppStderr "%LOGDIR%\proxy.err.log"
nssm set %NAME% Start SERVICE_AUTO_START
nssm start %NAME%

echo [install] %NAME% installed and started on :11600
echo           logs: %LOGDIR%\proxy.log / proxy.err.log
endlocal
