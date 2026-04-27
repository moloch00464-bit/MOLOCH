@echo off
setlocal enabledelayedexpansion
title MOLOCH Cockpit
color 0A

echo ==========================================
echo   MOLOCH Cockpit - One-Click Start
echo ==========================================
echo.

:: --- 1. Services starten (silent, ignore wenn schon up) ---
echo [moloch] Services pruefen...
for %%T in (MolochAdapterProxy MolochDashboard MolochAvatar MolochPiTunnel MolochOllama MolochCrossMonitor MolochSampleSync) do (
    schtasks /run /tn "%%T" >nul 2>&1
)

:: --- 2. Mic-Permission check ---
set "PYEXE=%USERPROFILE%\moloch_pc_env\Scripts\python.exe"
set "MIC_OK=0"
"%PYEXE%" -c "import json,os,pathlib; p=pathlib.Path(os.environ['LOCALAPPDATA'])/'Google'/'Chrome'/'User Data'/'Default'/'Preferences'; d=json.loads(p.read_text(encoding='utf-8')); m=d.get('profile',{}).get('content_settings',{}).get('exceptions',{}).get('media_stream_mic',{}); print(1) if m.get('http://localhost:9000,*',{}).get('setting')==1 else print(0)" > "%TEMP%\moloch_mic_check.txt" 2>nul
set /p MIC_OK=<"%TEMP%\moloch_mic_check.txt"
del "%TEMP%\moloch_mic_check.txt" >nul 2>&1

if "%MIC_OK%"=="1" (
    echo [moloch] Mic-Permission OK
    goto :open_browser
)

:: --- 3. Mic-Permission fehlt: prefs editieren wenn Chrome zu ---
tasklist /fi "IMAGENAME eq chrome.exe" 2>nul | find /i "chrome.exe" >nul
if not errorlevel 1 (
    echo.
    echo [moloch] Mic-Permission fehlt UND Chrome ist offen.
    echo [moloch] Schliesse ALLE Chrome-Fenster und klicke MOLOCH erneut.
    echo.
    echo Druecke Taste zum Beenden...
    pause >nul
    exit /b 1
)

echo [moloch] Mic-Permission setzen ^(Chrome ist zu, Pruefung OK^)...
"%PYEXE%" "%~dp0fix_chrome_mic_prefs.py"
if errorlevel 1 (
    echo [moloch] Mic-Fix fehlgeschlagen - siehe Output oben
    pause
    exit /b 1
)
echo [moloch] Mic-Permission gesetzt.

:open_browser
echo.
echo [moloch] Oeffne http://localhost:9000/ ...
start "" "http://localhost:9000/"

:: --- 4. self-close ---
timeout /t 2 /nobreak >nul
endlocal
exit /b 0
