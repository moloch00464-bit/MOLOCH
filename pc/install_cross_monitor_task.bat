@echo off
setlocal
set HERE=%~dp0
set NAME=MolochCrossMonitor
set RUNNER=%HERE%run_cross_monitor.bat

if not exist "%RUNNER%" (
  echo [install] %RUNNER% not found
  exit /b 1
)

echo [install] registering Scheduled Task "%NAME%" via PowerShell ...

powershell.exe -NoProfile -Command ^
  "$a = New-ScheduledTaskAction -Execute '%RUNNER%';" ^
  "$t = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME;" ^
  "$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartInterval (New-TimeSpan -Minutes 1) -RestartCount 9999;" ^
  "Register-ScheduledTask -TaskName '%NAME%' -Action $a -Trigger $t -Settings $s -Force | Out-Null;" ^
  "Write-Host '[install] OK - %NAME% registered (logon, RestartCount=9999)'"

if errorlevel 1 (
  echo [install] FAILED
  exit /b 1
)
endlocal
