@echo off
REM MOLOCH Search-Proxy — Scheduled Task installer (AtLogOn, silent via VBS).
REM Idempotent: deletes existing task before creating new one.

set TASKNAME=MolochSearchProxy
set SCRIPT=%USERPROFILE%\moloch_repo\pc\run_search_proxy_hidden.vbs

if not exist "%SCRIPT%" (
    echo [ERROR] %SCRIPT% nicht gefunden.
    exit /b 1
)

REM Loesche existing task wenn da
schtasks /query /tn "%TASKNAME%" >nul 2>&1
if %errorlevel%==0 (
    echo [info] Loesche existing Task %TASKNAME% ...
    schtasks /delete /tn "%TASKNAME%" /f >nul
)

REM Anlegen: AtLogOn, kein Admin noetig, wscript fuer silent
schtasks /create ^
    /tn "%TASKNAME%" ^
    /tr "wscript.exe \"%SCRIPT%\"" ^
    /sc onlogon ^
    /rl limited ^
    /f >nul

if %errorlevel% neq 0 (
    echo [ERROR] schtasks /create fehlgeschlagen.
    exit /b 1
)

echo [ok] Scheduled Task %TASKNAME% angelegt (AtLogOn, silent VBS).
echo [hint] Sofort starten: schtasks /run /tn %TASKNAME%
echo [hint] Status:         schtasks /query /tn %TASKNAME% /v ^| findstr Status
echo [hint] Health-Check:   curl http://localhost:11650/health
