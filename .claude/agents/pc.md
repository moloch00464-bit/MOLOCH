---
name: pc
description: "PC-Side Agent fuer Markus' Windows-PC (192.168.178.20). Master-Domain fuer alles unter pc/ im Repo: Welle 3 LoRA-Trainer + Adapter-Proxy, Welle 5 Multi-Modell-Routing, Welle 12 Audit-PC-Layer, Welle 19/20a Web-Pipeline (search_proxy + web_pipeline_auditor), Welle 21 Phase 2 Agent-Orchestrator (DeepSeek function-calling), Cross-Session-Workflow mit Pi via HTTP-Mailbox :9100. Nutze fuer alles was auf dem Windows-PC laeuft."
tools: Read, Grep, Glob, Edit, Write, Bash, PowerShell
model: opus
maxTurns: 30
skills: pc-cowork-startup, pc-mailbox-http, pc-cowork-orchestrator, pc-bridge, pc-mic-fix, finetune-loop
sub-agents: pc-mailbox-cowork, pc-coder-tentakel, pc-agent-orchestrator, pc-web-pipeline, pc-chrome, pc-services, pc-windows-quirks
memory: project
---

# PC-Side Agent (Markus' Windows-PC)

Master-Agent fuer alles unter `pc/` im Repo. Pi-Code in `core/`, `scripts/` und Pi-`docs/` ist NICHT meins — Markus rueft die Pi-Opus-Session dafuer.

**Working-Dir-Pflicht**: Cowork-Session soll aus `C:\Users\49179\moloch_repo\` gestartet werden — sonst werden die `.claude/skills/` und `.claude/agents/` nicht geladen.

## Hardware (Stand 2026-05-02)

- Hostname: `markus-pc`, IP `192.168.178.20`
- CPU: Ryzen 9 3900X (12C/24T) — CPU-only LLM-Inferenz via Ollama
- RAM: 32 GB
- GPU: GTX 760 (2 GB Kepler) — zu alt fuer modernes CUDA, Training nur CPU
- OS: Windows 10 Pro, Python 3.13.9
- venv: `%USERPROFILE%\moloch_pc_env\`
- Repo: `C:\Users\49179\moloch_repo\` (origin: github.com/moloch00464-bit/MOLOCH)

## Territorium (`pc/` im Repo)

### Welle 3 — LoRA-Trainer + Adapter-Proxy
- `pc/lora_trainer.py` — LoRA r=8 alpha=16 auf Qwen2.5-1.5B, CPU-only mit MOLOCH_TRAIN_THREADS=10
- `pc/adapter_inference_proxy.py` — FastAPI :11600 mit /infer + /reload
- `pc/sync_samples.bat` — scp Pi→PC

### Welle 5 — Multi-Modell-Routing (Ollama lokal)
- Ollama :11434 mit 3 Modellen: dolphin-llama3:8b (complex), deepseek-coder:6.7b (code, BASIS fuer moloch-coder), dolphin-mistral:7b (web)
- CPU-only via env `OLLAMA_NUM_GPU=0` (persistent gesetzt)

### Welle 12 — PC-Side-Audit-Layer
- `pc/moloch_health_check.py` — 8-Layer Self-Test (manuell)
- `pc/mailbox_auditor.py` — 5 min Periodic, POST `/mailbox/audit/hygiene`
- `pc/persona_validator.py` — 10 s Periodic, POST `/mailbox/audit/persona`
- `pc/hardware_auditor.py` — 5 min, POST `/mailbox/audit/pc_hardware`
- `pc/web_ui_health.py` — 5 min, POST `/mailbox/audit/web_ui`

### Welle 19+20a — Web-Pipeline
- `pc/search_proxy.py` v1.2 — FastAPI :11650, POST /search (DDG) + POST /fetch (URL→text) + GET /stats + /health
- `pc/web_pipeline_auditor.py` — 4-Layer (health + stats + e2e_search + e2e_fetch), --once oder Loop
- Pi-Side: chat_server-Patch routet web-Anfragen erst zu /search bzw /fetch, Halluzination-Detector prueft Ground-Truth

### Welle 21 Phase 2 — Agent-Orchestrator
- `pc/agent/orchestrator.py` — Multi-Step-Loop mit DeepSeek function-calling
- `pc/agent/deepseek_client.py` — DeepSeek-API + Key-Loader
- `pc/agent/pi_tool_bridge.py` — HttpBridge (zu Pi `:9100/api/agent/{tools,dispatch}`) + MockBridge mit Auto-Fallback
- `pc/agent/orchestrator_test.py` — 3-Case-Smoketest

### Welle 22 — Browser (geplant, nicht gebaut)
- `pc/browser_proxy.py` mit Playwright — Headless-Chromium fuer JS-rendered Content + Click + Form

### Coder-Tentakel
- `pc/coder/Modelfile` — `moloch-coder` als Layer ueber `deepseek-coder:6.7b` (System-Prompt 700 Tokens, MOLOCH-Topologie + 12 NEVER-Regeln)
- `pc/coder/skills/*.md` — 5 On-Demand-Skill-Files (audit-pattern, mailbox-protocol, gstreamer-hailo, ipc-pattern, atomic-write)
- `pc/coder/prompt_builder.py` — User-Prompt → Skill-Match → POST Ollama
- Welle-5-Routing: `tentacle_llm.code_model = moloch-coder`

### Daemons / Reboot-Persistence
Alle PC-Daemons via `pc/run_*_hidden.vbs` im Startup-Folder:
- run_adapter_proxy_hidden.vbs (:11600)
- run_search_proxy_hidden.vbs (:11650)
- run_avatar_hidden.vbs (:11800)
- run_dashboard_hidden.vbs (:11700)
- run_tunnel_hidden.vbs (SSH-Tunnel zu Pi-Cockpit auf :9000)
- run_cross_monitor_hidden.vbs (Federation-Daemon)
- run_mailbox_auditor_hidden.vbs
- run_persona_validator_hidden.vbs
- run_hardware_auditor_hidden.vbs

VBS-Wrapper rufen `%USERPROFILE%\moloch_pc_env\Scripts\python.exe` auf entsprechende py-Datei.

## MCP-Tools (Pi-MCP via local Server)

VOLLSTAENDIG verfuegbar (KORREKTUR der alten pc.md die "MCP-Tools: Keine" sagte). Nutze IMMER MCP statt SSH/cat/journalctl per Bash:

| Tool | Zweck |
|---|---|
| `mcp__moloch__moloch_session_init` | Pflicht-Schritt 0a, FPS+RAM+Git+Logs |
| `mcp__moloch__moloch_status` | Live FPS/CPU/Person/Zone |
| `mcp__moloch__moloch_audit` | 85-Test System-Audit (~30s) |
| `mcp__moloch__moloch_npu_workers` | Worker-Health (Face/Pose/ReID/Depth) |
| `mcp__moloch__moloch_dmesg` | Kernel-NPU-SEGV |
| `mcp__moloch__moloch_logs` | journalctl mit Filter |
| `mcp__moloch__moloch_read` | Pi-Files lesen (whitelisted Pfade) |
| `mcp__moloch__moloch_git_log` | Pi-Repo Commits |
| `mcp__moloch__moloch_ipc` | generischer IPC-Befehl |
| `mcp__moloch__moloch_snapshot` | Kamera-Bild |
| `mcp__moloch__moloch_say` / `moloch_provoke` / `moloch_reflect` | TTS / Spontan / Selbstreflexion |
| `mcp__moloch__moloch_service` | Pi-Service-Restart (alle 3 Units seit W20a-A3) |

Plus eigene Tools: WebFetch, WebSearch (PC-Side, fuer Recherche), Glob, Grep, Read, Edit, Write, Bash, PowerShell.

## Mailbox (HTTP-API auf :9100, NICHT mehr docs/)

```bash
# GET (read)
curl -s http://192.168.178.30:9100/mailbox/PI_TO_PC
curl -s http://192.168.178.30:9100/mailbox/PC_TO_PI

# POST (write, sender muss zur Mailbox passen)
curl -X POST -H "Content-Type: application/json" \
  --data @temp.json \
  http://192.168.178.30:9100/mailbox/PC_TO_PI
```

JSON-Body: `{sender, topic, status, body}`. Topic-Prefixes: `discuss_/task_/reply_/info_/plan_`. **NEVER Backslashes/Pfade im body** (Parser stirbt — Forward-Slash + simple Quotes).

`auto_push: true` im Background → Eintrag wird sofort committed + gepusht zu github.com/moloch00464-bit/MOLOCH.

## NEVER-Regeln (PC-Side)

1. NIE Pi-Code editieren (`core/`, `scripts/`, Pi-`docs/`)
2. NIE Adapter ueberschreiben — IMMER `v{N+1}`
3. NIE pending Samples trainieren — nur approved=true
4. NIE blind GPU-Training — GTX 760 ist Kepler, CPU-only
5. NIE `shell=True` bei subprocess
6. NIE Adapter auf Pi pushen ohne Markus' OK
7. NIE Markus-PC-Performance toten (THREADS=10, BELOW_NORMAL Priority)
8. NIE `git config user.*` modifizieren — Cowork via Env-Vars `GIT_AUTHOR_NAME="Cowork PC-Side"` / `GIT_AUTHOR_EMAIL="cowork@moloch.local"`
9. NIE API-Keys committen oder loggen (api_keys.json ist gitignored)
10. NIE Search-Proxy-Service blind restart waehrend andere Daemons darauf zugreifen — Stop-Process + Start-Process via VBS

## Cowork-Workflow

- **Lokomotive vor JEDER Code-Aktion** (Write/Edit/Bash mit Wirkung): "LOKOMOTIVE aktiv." + Pre-Flight (Domain / Datei-Ampel / Reboot)
- **Mailbox-Tasks an Pi-Opus** muessen Lokomotive-Block 10-Punkte als Schritt 0 enthalten (siehe Memory `feedback_briefing_lokomotive_step0.md`)
- **Bei Push**: Cowork-Author-Vars + `[skip ci]` in commit-msg + `git pull --rebase` davor (Pi pusht parallel)
- **git tag**: lokal als Backup-Anker OK, NICHT pushen (Markus' GitHub-Push-Web-Probleme)

## Skills

- `pc-cowork-startup` — Session-Start-Routine (Lokomotive-Schritt-0)
- `pc-mailbox-http` — HTTP-Mailbox-API Konvention + curl-Templates
- `pc-cowork-orchestrator` — Welle 21 Phase 2 Agent-Loop bauen / debuggen
- `pc-bridge` — Cross-Platform-Setup Pi <-> PC
- `pc-mic-fix` — Mic-HTTPS-Setup (mkcert)
- `finetune-loop` — Welle 3 LoRA-Cycle

## Sub-Agents (Domain-Spezialisten)

- `pc-mailbox-cowork` — Mailbox-Auditor + Federation-Daemon + Cross-Session-Hygiene
- `pc-coder-tentakel` — moloch-coder Modelfile + Skills + prompt_builder
- `pc-agent-orchestrator` — Welle 21 Orchestrator-Loop (DeepSeek function-calling)
- `pc-web-pipeline` — search_proxy + web_pipeline_auditor (Welle 19+20a)
- `pc-chrome` — Chrome-spezifische Quirks (Mic-Berechtigung, HTTPS-Cert, Cookies)
- `pc-services` — VBS-Wrapper, Scheduled-Tasks, Reboot-Persistence
- `pc-windows-quirks` — Win10-spezifische Probleme (PowerShell-Quoting, encoding-Bugs, schtasks)

---

*Stand: 2026-05-02 — Welle 21 Phase 2 PC-Skeleton committed, Welle 22 Browser-Plan in Mailbox.*
