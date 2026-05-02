# Pi-Opus Handoff — 2026-05-02 Nachmittag

**Copy-Paste in neue Session damit Pi-Opus nahtlos weitermacht.**

---

## SOFORT-PRE-FLIGHT (PFLICHT)

```
1. moloch_session_init() via MCP
2. moloch_status() + moloch_npu_workers()
3. /moloch-dev Skill laden
4. /moloch-agent Skill laden
5. /moloch-mcp Skill laden
6. git fetch -q origin && git log --oneline HEAD..origin/deepseek_architecture_overhaul
7. head -40 docs/PC_TO_PI.md (neue PC-Topics)
8. cat /dev/shm/audit_state.json | python3 -m json.tool | head -80
9. git tag als Backup-Anker setzen vor Code-Edit
10. Domain-Lock: touch /tmp/moloch_agent_<name> VOR jedem Edit
```

---

## STAND HEUTE (Push-HEAD: aa9a621)

### Audit-Verteilung
- **27 Layer total**
- **21 PASS** / 5 WARN / 0 FAIL / 1 PENDING
- overall: warn (alarm: silent)

### 5 verbleibende WARN-Layer (alle akzeptabel)
| Layer | Reason | Owner |
|-------|--------|-------|
| mailbox WARN 4/4 | 73 PC-Topics + 138 Pi-Topics + 24 stale | PC-Hoheit |
| personality WARN 3/4 | tension=-1.0 (Sentinel: kein Person im Frame) | idle |
| reflection WARN 15/20 | aktive Dev-Day, beruhigt sich | self-resolve |
| self_diagnosis WARN 3/4 | Pytest-Suite fehlt | externes Setup |
| voice WARN 3/4 | tts_calls_1h=0 (idle) | idle |

### 1 PENDING
- web_search PENDING — wartet auf PC web_pipeline_auditor-Daemon-POST (5min-Intervall, run_web_pipeline_auditor_hidden.vbs)

---

## 27-PUNKTE-PLAN (PC-Topic 14:04 plan_27_punkte_alles_fertig_aufteilung_pi_pc)

### Pi-Side erledigt heute (~14 Punkte)
- #2 Bug B Spotify Device-404-Recovery (`d2c4dcf`)
- #3 9 Spotify-Tools (`64d2c74`)
- #4 7 Hardware-Tools + 4 System-Tools (`d0d16bb`+`40f186a`+`49eab67`)
- #5 agent_loop_verify (`871ae4b`) PASS 4/4
- #6 settings.json agent_loop config-flag (`aaff2a7`)
- #9 3 Browser-Tools (`aa9a621`) — PC-Proxy :11680 Live-Roundtrip
- #11 awareness Schwellen-Tuning (`740fc89`)
- #13 capability dynamic max + 65% (`94ef146`)
- #14 cross thread-threshold 100→150 (`b3e087c`)
- #16 memory idle-PASS (`9e2caa0`)
- #18 reflection score>max + Schwelle (`975bf15`+`8838980`)
- #21 voice wifi_mic_singleton-Probe (`f550dcb`+`18ab757`)
- #22 persona Initial-Event Hook (`ef2ea26`)
- #23 federation Heartbeat (`417beaf`) schon gestern Mittag

### Pi-Side offen
- **#24 Hand-Erkennung** — Pi 12/13 aktiv, vision-Domain, **GROßER BROCKEN**, eigene Welle
- 5 WARN-Layer sind strukturell-akzeptabel (siehe oben)

### PC-Side erledigt heute (7/7)
- #1, #3-PC-Audit, #7 Token-Budget, #8 browser_proxy, #10 Vision-Bridge-Stub, #25 STT, #26 TTS

### Markus-Decision-blocked
- #10 Vision-Backend-Wahl (moondream2 lokal vs Claude vs OpenRouter)
- #27 Claude-API-Fallback (wontfix ohne Anthropic-Key)

---

## TOOL-CATALOG: 28 TOOLS

```
Web (2):       web_search, web_fetch
Spotify (11):  top_artists, play, pause, next, prev, volume, search, 
               now_playing, top_tracks, recommend, play_genre
Hardware (7):  ptz_pan, ptz_tilt, led_set, thermal_set_tension_pwm,
               camera_snapshot, get_face_id, get_npu_status
System (4):    get_audit_state, moloch_status_summary, read_memory, tts_say
Mood (1):      get_mood
Browser (3):   browser_open, browser_click, browser_screenshot
```

Alle dispatchen via `/api/agent/dispatch` HTTP + `core.agent.tool_dispatcher.dispatch()`.

---

## MAILBOX-STAND

### Letzte Pi-Posts (PI_TO_PC.md)
- 14:43 `info_pi_sprint_update_layer_5_fixed_tools_25` (Auto-Push via :9100/mailbox/PI_TO_PC)
- 14:29 `info_pi_sprint1_2_3_progress`
- 13:18 `info_pi_session_drift7_qdrant_heartbeat_done`

### Letzte PC-Topics (PC_TO_PI.md offen/info)
- 14:38 `info_pc_sprint_status_12_von_27_done` — PC-Anteil komplett
- 14:17 `info_pc_phase3_tool_catalog_audit_live`
- 14:04 `plan_27_punkte_alles_fertig_aufteilung_pi_pc` (Master-Plan)

---

## ARCHITEKTUR-STAND

### W21 Agent-Loop (Cloud-Orchestrator)
- DeepSeek function-calling-Loop läuft auf PC: `pc/agent/orchestrator.py` (`f872e77`)
- Pi-Tool-API: GET `/api/agent/tools`, POST `/api/agent/dispatch` (`a5327cc`)
- Tool-Dispatcher: `core/agent/tool_dispatcher.py` (`2e2f482`)
- closed_loop: `agent_loop_verify.py` PASS 4/4

### W22 Browser
- PC: browser_proxy :11680 (Playwright) — `bc40c45`
- PC: vision-Bridge :9003 Stub — `7007f6f`
- Pi: 3 Browser-Tools im Catalog — `aa9a621`

### Cross-Audit (gestern Pi-PC-Drift)
- 7 Drifts identifiziert + alle resolved (4 PC + 3 Pi)
- Pi `/audit/transition` Endpoint
- transition-Layer: PASS 7/7 alive (federation_heartbeat via audit_orchestrator-Tick)

### MCP-Tool (Service-Restart)
- `moloch_service(action="restart")` restartet 3 Units (moloch + moloch-chat + moloch-chat-https) mit per-unit-Timeout (https=60s SSL-Init)

---

## NEXT ACTIONS (Reihenfolge)

### A) #24 Hand-Erkennung (wenn Markus will)
- Domain: vision (`core/perception/*.py`, `core/inference_engine.py`)
- Pi 12/13 hat Hand-HEF-Modell aktiv
- Pre-Flight: NPU-Worker pruefen
- Aufwand: groß — eigene Welle, ggf. mehrere Sub-Agents

### B) Verbleibende WARN-Layer (akzeptabel — Markus-Direktive nötig)
- Wenn Markus 100% PASS will: voice-WARN-Schwelle für `tts_calls_1h=0` als idle akzeptieren (1 Edit, 1 Commit)
- mailbox-WARN: PC's Hoheit, gemeinsame Direktive nötig

### C) Folge-Issues (warten auf PC)
- web_search PENDING bleibt bis PC-Daemon postet
- Drift 7 federation strukturell ist durch heartbeat-append OK, aber PC's cross_session_monitor schreibt eigene Datei (Doppel-Stream)

---

## CLAUDE.md-PFLICHTEN
- Pi 4GB RAM — sparsam
- NEVER-Regeln 1-12 (siehe moloch-dev Skill)
- subprocess timeout=30
- atomic JSON-write (NEVER 6)
- 1 ROT-Datei = 1 Commit (NEVER 4)
- KEIN shell=True
- __pycache__ nach Edit löschen

---

## SUB-AGENTS BENUTZT HEUTE

audit (Layer-Auditor-Tunings + Closed-Loop), bridge (chat_server-Endpoints), music (Bug B + Spotify-Tools), tentacle (wifi_mic-Probe-Fix), service (mcp-Tool 3-Units), autonomy (Tool-Catalog 17→28), hardware (thermal_manager device-API)

---

## SESSION 31 ABSCHLUSS-COMMITS (Push-Reihenfolge)

```
aa9a621 #9 3 Browser-Tools — Catalog 25→28 (W22 PC :11680)
49eab67 4 System-Tools (audit_state, status_summary, read_memory, tts_say) — 21→25
40f186a 4 Hardware-Tools (ptz_tilt, thermal, face_id, npu_status) — 17→21
f550dcb #21 voice_auditor wifi_mic-Probe-Fix
18ab757 #21 voice Cross-Process-Schema (vorher)
8838980 #18 reflection FAIL-Schwelle 10→20
9e2caa0 #16 memory idle-PASS
b3e087c #14 cross thread-threshold 100→150
94ef146 #13 capability dynamic max
740fc89 #11 awareness Idle-Toleranz
ef2ea26 #22 persona Initial-Event Hook
975bf15 #18 reflection score>max-Bug
d0d16bb #4 3 Hardware-Tools — 14→17
64d2c74 #3 9 Spotify-Tools — 5→14
aaff2a7 #6 settings.json agent_loop config-flag
871ae4b #5 agent_loop_verify
d2c4dcf #2 Bug B Device-404-Recovery
```

---

**Pi-Side ist substanziell durch.** Markus-Direktive für #24 abwarten.
Bei "weitermachen" → vision-Sub-Agent für Hand-Erkennung anfeuern (eigene Welle, groß).
