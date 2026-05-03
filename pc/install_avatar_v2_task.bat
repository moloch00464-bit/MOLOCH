@echo off
setlocal
set HERE=%~dp0
set NAME=MolochAvatarV2
set RUNNER=%HERE%run_avatar_v2_hidden.vbs

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
echo   - Avatar 2.0 laeuft auf http://localhost:11801
echo   - parallel zum Legacy-Avatar auf :11800 (kein Konflikt)
echo   - to remove: schtasks /delete /tn "%NAME%" /f
echo   - health-check: curl http://localhost:11801/health
endlocal
