@echo off
setlocal
set NAME=MolochOllama
set OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe

if not exist "%OLLAMA%" (
  echo [install] Ollama nicht gefunden unter %OLLAMA%
  exit /b 1
)

echo [install] registering Scheduled Task "%NAME%" via PowerShell ...
echo            Ollama-Server :11434 startet bei jedem Login automatisch

powershell.exe -NoProfile -Command ^
  "$a = New-ScheduledTaskAction -Execute '%OLLAMA%' -Argument 'serve';" ^
  "$t = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME;" ^
  "$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartInterval (New-TimeSpan -Minutes 2) -RestartCount 9999;" ^
  "Register-ScheduledTask -TaskName '%NAME%' -Action $a -Trigger $t -Settings $s -Force | Out-Null;" ^
  "Write-Host '[install] OK - %NAME% registered'"

if errorlevel 1 (
  echo [install] FAILED
  exit /b 1
)
endlocal
