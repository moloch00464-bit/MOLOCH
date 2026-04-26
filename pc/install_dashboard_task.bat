@echo off
setlocal
set HERE=%~dp0
set NAME=MolochDashboard
set RUNNER=%HERE%run_dashboard.bat

if not exist "%RUNNER%" (
  echo [install] %RUNNER% not found
  exit /b 1
)

echo [install] registering Scheduled Task "%NAME%" via PowerShell ...

powershell.exe -NoProfile -Command ^
  "$a = New-ScheduledTaskAction -Execute '%RUNNER%';" ^
  "$t = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME;" ^
  "$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero);" ^
  "Register-ScheduledTask -TaskName '%NAME%' -Action $a -Trigger $t -Settings $s -Force | Out-Null;" ^
  "Write-Host '[install] OK - %NAME% registered'"

if errorlevel 1 (
  echo [install] FAILED
  exit /b 1
)
endlocal
