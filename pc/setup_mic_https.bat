@echo off
setlocal
set BIN=%USERPROFILE%\bin
set CERTS=%USERPROFILE%\moloch_certs
set MKCERT=%BIN%\mkcert.exe
set MKCERT_URL=https://github.com/FiloSottile/mkcert/releases/download/v1.4.4/mkcert-v1.4.4-windows-amd64.exe
set REMOTE=molochzuhause@192.168.178.30
set REMOTE_CERTS=/home/molochzuhause/moloch/config/certs

if not exist "%BIN%" mkdir "%BIN%"
if not exist "%CERTS%" mkdir "%CERTS%"

if not exist "%MKCERT%" (
  echo [mic] downloading mkcert v1.4.4 ^(~5 MB^) ...
  powershell.exe -NoProfile -Command "Invoke-WebRequest -Uri '%MKCERT_URL%' -OutFile '%MKCERT%' -UseBasicParsing"
  if errorlevel 1 (
    echo [mic] download FAILED
    exit /b 1
  )
) else (
  echo [mic] mkcert already at %MKCERT%
)

echo.
echo [mic] installing local Root CA into Windows Cert Store
echo        ^(UAC prompt erwartet - kurz "Ja" klicken^)
"%MKCERT%" -install
if errorlevel 1 (
  echo [mic] mkcert -install FAILED ^(UAC abgebrochen?^)
  exit /b 1
)

echo.
echo [mic] generating cert for 192.168.178.30 + moloch.local + localhost
pushd "%CERTS%"
"%MKCERT%" -key-file moloch_chat.key -cert-file moloch_chat.crt 192.168.178.30 moloch.local localhost
set GENERR=%errorlevel%
popd
if not "%GENERR%"=="0" (
  echo [mic] mkcert generate FAILED
  exit /b 1
)

echo.
echo [mic] scp cert + key to Pi
scp -o "StrictHostKeyChecking=accept-new" -o "BatchMode=yes" "%CERTS%\moloch_chat.crt" "%CERTS%\moloch_chat.key" "%REMOTE%:%REMOTE_CERTS%/"
if errorlevel 1 (
  echo [mic] scp FAILED ^(SSH-Key-Problem?^)
  exit /b 1
)

echo.
echo [mic] restarting Pi HTTPS service via SSH ^(mit Pi-Lock-Convention^)
ssh -o "StrictHostKeyChecking=accept-new" -o "BatchMode=yes" "%REMOTE%" "touch /tmp/moloch_agent_bridge && sudo systemctl restart moloch-chat-https && rm -f /tmp/moloch_agent_bridge"
if errorlevel 1 (
  echo.
  echo [mic] ssh restart FAILED - sudo braucht Passwort?
  echo        Manuell auf Pi: sudo systemctl restart moloch-chat-https
  exit /b 2
)

echo.
echo [mic] verify https://192.168.178.30:9443/
curl -sS --max-time 5 https://192.168.178.30:9443/ -o NUL -w "        HTTP %%{http_code}\n"

echo.
echo [mic] DONE.
echo        Browser oeffnen: https://192.168.178.30:9443/
echo        -^> kein Sicherheits-Warning, Mic-Permissions klickbar.
endlocal
