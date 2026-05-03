@echo off
setlocal
set HERE=%~dp0
set NAME=MolochAutoResearcher
set RUNNER=%HERE%run_auto_researcher_hidden.vbs

if not exist "%RUNNER%" (
  echo [install] %RUNNER% not found
  exit /b 1
)

REM Daily 09:00 (Markus' typischer Morgenwakeup) - Markus kann uhrzeit anpassen via XML-Edit
echo [install] registering Scheduled Task "%NAME%" (DAILY 09:00, silent via VBS)
schtasks /create /tn "%NAME%" /tr "wscript.exe \"%RUNNER%\"" /sc DAILY /st 09:00 /f
if errorlevel 1 (
  echo [install] schtasks /create failed
  exit /b 1
)

echo.
echo [install] done.
echo   - Auto-Researcher laeuft taeglich 09:00 (Stufe 1 passiv)
echo   - Output: %USERPROFILE%\moloch_repo\staging\research_proposals\YYYY-MM-DD.md
echo   - Manueller Trigger: pc\run_auto_researcher.bat
echo   - to remove: schtasks /delete /tn "%NAME%" /f
echo   - Stage check: python -m pc.auto_researcher --stage
endlocal
