@echo off
setlocal
set HERE=%~dp0
set NAME=MolochAdapterProxy
set RUNNER=%HERE%run_proxy.bat

if not exist "%RUNNER%" (
  echo [install] %RUNNER% not found
  exit /b 1
)

echo [install] registering Scheduled Task "%NAME%" (trigger: at logon)
schtasks /create /tn "%NAME%" /tr "\"%RUNNER%\"" /sc onlogon /f
if errorlevel 1 (
  echo [install] schtasks /create failed
  exit /b 1
)

echo [install] starting task once now...
schtasks /run /tn "%NAME%" >nul 2>&1

echo.
echo [install] done.
echo   - autostart bei jedem Windows-Login
echo   - status check / restart: pc\moloch_status.bat (oder Desktop-Button)
echo   - to remove: schtasks /delete /tn "%NAME%" /f
endlocal
