@echo off
REM MOLOCH — Pre-warm Ollama-Modelle beim Boot.
REM Verhindert Cold-Load-Latenz beim ersten code_query / complex_smalltalk.
REM Lauft ~30-60s einmalig nach Login. Danach sind Modelle in Ollama-RAM-Cache.

setlocal
set OLLAMA=ollama
set DUMMY="x"

echo [prewarm] Warming up Ollama models...

REM Code-Modell (deepseek-coder:6.7b) — wird bei prompt_type=code_query gerufen.
echo [prewarm] deepseek-coder:6.7b ...
%OLLAMA% run deepseek-coder:6.7b %DUMMY% >nul 2>&1
if errorlevel 1 (
    echo [prewarm] deepseek-coder:6.7b FAIL ^(Modell nicht installiert? "ollama pull deepseek-coder:6.7b"^)
) else (
    echo [prewarm] deepseek-coder:6.7b OK
)

REM Konversations-Modell (dolphin-llama3:8b) — Default fuer complex_smalltalk + web_research.
echo [prewarm] dolphin-llama3:8b ...
%OLLAMA% run dolphin-llama3:8b %DUMMY% >nul 2>&1
if errorlevel 1 (
    echo [prewarm] dolphin-llama3:8b FAIL
) else (
    echo [prewarm] dolphin-llama3:8b OK
)

echo [prewarm] done.
endlocal
