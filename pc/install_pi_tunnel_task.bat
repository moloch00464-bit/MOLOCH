@echo off
setlocal
set HERE=%~dp0
set NAME=MolochPiTunnel
set RUNNER=%HERE%start_pi_tunnel.bat

if not exist "%RUNNER%" (
  echo [install] %RUNNER% not found
  exit /b 1
)

echo [install] registering Scheduled Task "%NAME%" via PowerShell ...
echo            (logon trigger, ExecutionTimeLimit unbegrenzt)

powershell.exe -NoProfile -Command ^
  "$a = New-ScheduledTaskAction -Execute '%RUNNER%';" ^
  "$t = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME;" ^
  "$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartInterval (New-TimeSpan -Minutes 1) -RestartCount 3;" ^
  "Register-ScheduledTask -TaskName '%NAME%' -Action $a -Trigger $t -Settings $s -Force | Out-Null;" ^
  "Write-Host '[install] OK - %NAME% registered'"

if errorlevel 1 (
  echo [install] FAILED
  exit /b 1
)
endlocal
