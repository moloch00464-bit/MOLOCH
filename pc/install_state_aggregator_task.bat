@echo off
setlocal
set HERE=%~dp0
set NAME=MolochStateAggregator
set RUNNER=%HERE%run_state_aggregator_hidden.vbs

if not exist "%RUNNER%" (
  echo [install] %RUNNER% not found
  exit /b 1
)

echo [install] registering Scheduled Task "%NAME%" (trigger: at logon, silent via VBS)
schtasks /create /tn "%NAME%" /tr "wscript.exe \"%RUNNER%\"" /sc onlogon /f
if errorlevel 1 (
  echo [install] schtasks /create failed
  exit /b 1
)

echo [install] starting task once now...
schtasks /run /tn "%NAME%" >nul 2>&1

echo.
echo [install] done.
echo   - autostart bei jedem Windows-Login
echo   - State-Aggregator laeuft auf http://localhost:11652
echo   - to remove: schtasks /delete /tn "%NAME%" /f
echo   - health-check: curl http://localhost:11652/health
echo   - state-file: %%LOCALAPPDATA%%\moloch_pc_state\state_vector.json
endlocal
