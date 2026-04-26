@echo off
setlocal enabledelayedexpansion
title MOLOCH Adapter
color 0A

echo ==========================================
echo   MOLOCH Adapter Proxy Status
echo ==========================================
echo.

curl -s --max-time 3 http://localhost:11600/health
if errorlevel 1 (
  echo.
  echo [moloch] proxy is DOWN - starting via Scheduled Task...
  schtasks /run /tn "MolochAdapterProxy" >nul 2>&1
  if errorlevel 1 (
    echo [moloch] could not trigger Scheduled Task ^(install missing?^).
    echo          Try: pc\install_scheduled_task.bat
    goto :end
  )
  echo [moloch] waiting for /health...
  set TRY=0
  :wait
  set /a TRY+=1
  curl -s --max-time 2 http://localhost:11600/health >nul 2>&1
  if not errorlevel 1 goto :up
  if !TRY! geq 60 goto :fail
  timeout /t 1 /nobreak >nul
  goto :wait
  :up
  echo.
  echo [moloch] OK - proxy is UP
  echo.
  curl -s http://localhost:11600/health
  goto :end
  :fail
  echo.
  echo [moloch] timeout 60s - check %USERPROFILE%\moloch_adapters\proxy.err.log
  goto :end
)

echo.
echo [moloch] proxy is UP and healthy.
echo.
echo Adapters:
curl -s http://localhost:11600/list

:end
echo.
echo.
echo ------------------------------------------
echo  R = Restart proxy   ^|   any other key = close
set /p CHOICE=Wahl:
if /i "%CHOICE%"=="R" (
  schtasks /end /tn "MolochAdapterProxy" >nul 2>&1
  timeout /t 2 /nobreak >nul
  schtasks /run /tn "MolochAdapterProxy" >nul 2>&1
  echo [moloch] restart triggered.
  timeout /t 3 /nobreak >nul
)
endlocal
