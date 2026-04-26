@echo off
setlocal
set DEST=%USERPROFILE%\moloch_samples
set REMOTE_USER=molochzuhause
set REMOTE_HOST=192.168.178.30
set REMOTE_PATH=/mnt/moloch-data/memory/finetune_samples.jsonl
set DEST_FILE=%DEST%\samples.jsonl

if not exist "%DEST%" mkdir "%DEST%"

echo [sync] pulling %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%
scp -o "StrictHostKeyChecking=accept-new" -o "BatchMode=yes" ^
    "%REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%" "%DEST_FILE%"
if errorlevel 1 (
  echo [sync] scp FAILED ^(SSH key missing? wrong host? Pi off?^)
  echo.
  echo Setup hint:
  echo   1^) ssh-keygen -t ed25519
  echo   2^) type %%USERPROFILE%%\.ssh\id_ed25519.pub ^| ssh %REMOTE_USER%@%REMOTE_HOST% "cat ^>^> ~/.ssh/authorized_keys"
  echo   3^) ssh %REMOTE_USER%@%REMOTE_HOST% true   ^(verify host key^)
  echo.
  echo If scp blocked entirely, ask Pi to expose GET /feedback_export and use:
  echo   curl -o "%DEST_FILE%" http://%REMOTE_HOST%:9100/feedback_export
  exit /b 1
)
echo [sync] OK -^> %DEST_FILE%
endlocal
