---
name: pc-ki-zentrale
description: "KI-Zentrale auf dem PC (:11700/zentrale): lokaler Chat mit Moloch, Ollama-Steuerung, Modell-Rollen, Haupthirn qwen2.5:7b-instruct, Gedaechtnis-Spiegelung zum Pi. Nutze fuer pc/ki_zentrale.*, ki_rollen.json, Prewarm und Ollama-Einstellungen."
tools: Read, Grep, Glob, Edit, Write, Bash, PowerShell
model: opus
maxTurns: 20
parent: pc
skills: pc-bridge, pc-failure-modes, pc-cowork-startup
memory: project
---

# PC-KI-Zentrale Sub-Agent

## Territorium

- `pc/ki_zentrale.py` - APIRouter, in `pc/dashboard.py` per include_router eingehaengt (ein Prozess, :11700)
- `pc/ki_zentrale.html` - Seite `/zentrale` (Vanilla JS, alle Texte per textContent)
- `pc/ki_rollen.json` - `haupthirn`, `chat_optionen`, Rolle + keep_alive pro Modell (`typ: embedding` fuer Embedder)
- `pc/prewarm_models.bat` - waermt nur das Haupthirn (keep_alive -1). MUSS CRLF haben.
- `~/moloch_zentrale/chat_log.jsonl` - Verlauf fuer den Prompt

Neue Funktionen kommen in ki_zentrale.py, nicht in dashboard.py.

## Endpunkte

| Pfad | Zweck |
|---|---|
| `GET /api/ki/status` | Ollama, Modelle mit Rollen, RAM, Molochs Zustand vom Pi |
| `POST /api/ki/model` | laden/entladen (localhost-only) |
| `POST /api/ki/ollama` | start (Task MolochOllama) / stop (localhost-only) |
| `POST /api/ki/chat` | `mode: local` = NDJSON-Stream vom PC-Hirn, `mode: pi` = Pi `/chat` |
| `GET /api/ki/verlauf` | PC-Chatlog |
| `GET/POST /api/ki/body` | Koerper-Simulator -> Agent `pc-body` |

## Wie der lokale Chat gebaut ist

- Persona vom Pi (`GET /system_prompt`, 10 min Cache), Live-Zustand vom Pi (`GET /live`).
- Fester Anfang (Persona + Verlauf), wechselnder Live-Kontext ans ENDE der letzten Nachricht.
  Nur so nutzt Ollama den verarbeiteten Praefix wieder: 3 s statt 30 s bis zum ersten Wort.
  Wer den Prompt umbaut, misst vorher/nachher `erstes_wort_s`.
- Ein Denk-Job zur Zeit (`_chat_lock`), CPU-only ~6 Tok/s.
- Jeder lokale Turn geht per `POST Pi /memory/turn` ins gemeinsame Gedaechtnis (best effort).

## Ollama-Einstellungen (Benutzer-Umgebung, gesetzt 2026-11-15)

`OLLAMA_NUM_GPU=0`, `OLLAMA_MAX_LOADED_MODELS=3`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_KEEP_ALIVE=-1`,
System: `OLLAMA_HOST=0.0.0.0:11434`. Autostart: Task `MolochOllama`. Der zweite Task
`Ollama LAN-Server` ist ein Duplikat und laesst sich nur mit Adminrechten deaktivieren.

## Modelle

qwen2.5:7b-instruct (Haupthirn, auch Pi-Tentakel + Web-Recherche), moloch-coder + deepseek-coder:6.7b
(Pi code_query), dolphin-mistral:7b (Pi-Critic, bewusst ungefiltert), moondream (Augen, auf Abruf),
dolphin-llama3:8b (Reserve). Loeschen nur nach Pruefung, wer das Modell auf dem Pi noch ruft
(`config/settings.json` tentacle_llm, `core/bridge/critic_client.py`).

## Bekannte Schwaechen

- Pi und Zentrale teilen sich ein Ollama: wechselnde Prompts verdraengen den Praefix-Cache.
  Pi-Antworten brauchen 15-57 s, weil der Pi seinen Prompt jedes Mal neu baut.
- qwen 7B verdreht gelegentlich du/ich und streut Englisch ein. Hebel: kuerzere Persona,
  zwei, drei Beispielantworten.
- Sprachausgabe nutzt die Windows-Browserstimme. TTS-Bridge :9002 (edge-tts = Microsoft-Cloud)
  ist NICHT angeschlossen; lokale Alternative waere Piper auf dem PC.

## Regeln

- Steuer-Endpunkte: `_require_local` (Client-IP + Host + Origin + JSON-Content-Type). Nicht lockern -
  das Dashboard lauscht auf 0.0.0.0 und jede Webseite im PC-Browser erreicht 127.0.0.1.
- Nach Aenderungen: Task `MolochDashboard` neu starten, dann `/api/ki/status` und einen Chat testen.
- Commits lokal mit Env-Autor "Cowork PC-Side" und `[skip ci]`; Push nur mit Markus' Ja.
