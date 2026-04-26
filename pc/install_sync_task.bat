@echo off
setlocal
set HERE=%~dp0
set NAME=MolochSampleSync
set RUNNER=%HERE%sync_samples.bat

if not exist "%RUNNER%" (
  echo [install] %RUNNER% not found
  exit /b 1
)

echo [install] registering Scheduled Task "%NAME%" via PowerShell ...
echo            (logon trigger + every 6h repetition)

powershell.exe -NoProfile -Command ^
  "$a = New-ScheduledTaskAction -Execute '%RUNNER%';" ^
  "$tLogon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME;" ^
  "$tPeriod = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Hours 6) -RepetitionDuration (New-TimeSpan -Days 9999);" ^
  "$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 5);" ^
  "Register-ScheduledTask -TaskName '%NAME%' -Action $a -Trigger @($tLogon,$tPeriod) -Settings $s -Force | Out-Null;" ^
  "Write-Host '[install] OK - %NAME% registered'"

if errorlevel 1 (
  echo [install] FAILED
  exit /b 1
)
endlocal
