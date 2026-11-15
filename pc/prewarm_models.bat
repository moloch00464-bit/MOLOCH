@echo off
REM MOLOCH - Pre-warm Ollama-Modelle beim Boot (v2, 2026-06-11).
REM v1-Bug: setzte laufenden Ollama-Server voraus - beim Boot-Race war er oft
REM noch nicht da, Prewarm lief still ins Leere (>nul verschluckte den Fehler).
REM v2: wartet bis zu 120s auf den Server, waermt alle 3 Tentakel-Modelle
REM (Welle-5-Routing: complex/code/web).

setlocal enabledelayedexpansion
set OLLAMA=ollama
set DUMMY="x"

echo [prewarm] Warte auf Ollama-Server (max 120s)...
set /a tries=0
:wait_server
curl.exe -s --max-time 3 http://127.0.0.1:11434/api/tags >nul 2>&1
if not errorlevel 1 goto server_up
set /a tries+=1
if !tries! geq 24 (
    echo [prewarm] FAIL: Ollama-Server nach 120s nicht erreichbar - Abbruch.
    exit /b 1
)
timeout /t 5 /nobreak >nul
goto wait_server

:server_up
echo [prewarm] Server da. Waerme Modelle...

REM v3 (2026-11-15, KI-Zentrale): nur das Haupthirn dauerhaft laden (keep_alive -1).
REM Die uebrigen Modelle laden bei Bedarf (Pi-Coder, moondream) oder per Knopf in
REM der KI-Zentrale (http://localhost:11700/zentrale). "ollama run" haette nur 5 min gehalten.
set BRAIN=qwen2.5:7b-instruct
echo [prewarm] %BRAIN% (dauerhaft) ...
curl.exe -s --max-time 300 -o nul -w "%%{http_code}" http://127.0.0.1:11434/api/generate -d "{\"model\":\"%BRAIN%\",\"prompt\":\"\",\"keep_alive\":-1}" > "%TEMP%\moloch_prewarm_code.txt"
set /p CODE=<"%TEMP%\moloch_prewarm_code.txt"
if "!CODE!"=="200" (
    echo [prewarm] %BRAIN% OK
) else (
    echo [prewarm] %BRAIN% FAIL HTTP !CODE! ^(installiert? "ollama pull %BRAIN%"^)
)

echo [prewarm] done.
endlocal
