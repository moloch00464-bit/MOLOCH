@echo off
setlocal
title MOLOCH Pi-Tunnel (localhost:9000 -> 192.168.178.30:9100)

:loop
echo [tunnel] connecting molochzuhause@192.168.178.30 ^(L 9000 -^> 9100^) ...
ssh -L 9000:localhost:9100 -N ^
    -o BatchMode=yes ^
    -o ExitOnForwardFailure=yes ^
    -o ServerAliveInterval=30 ^
    -o ServerAliveCountMax=3 ^
    -o StrictHostKeyChecking=accept-new ^
    molochzuhause@192.168.178.30
echo [tunnel] disconnected ^(exit %errorlevel%^), reconnect in 5s
timeout /t 5 /nobreak >nul
goto loop
