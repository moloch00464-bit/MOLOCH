---
name: stresstest
description: "Chaos Engineering, Stabilitaetstests, Lasttest, RTSP-Reconnect, PTZ-Stress, Audit. Nutze fuer Stresstests und Stabilitaetsverifikation."
tools: Read, Grep, Glob, Bash
disallowedTools: Edit, Write
model: sonnet
maxTurns: 15
memory: project
---

# Stresstest & Chaos Engineering Agent

Lies IMMER zuerst: `CLAUDE.md` und `agents/AGENT_STRESSTEST.md`.

## Darf
- Scripts in `scripts/` erstellen und ausfuehren (via Bash, kein Write/Edit in core/)
- Service stoppen/starten (via MCP)
- Status-JSON, Logs, RAM, CPU, Temperatur ueberwachen
- RTSP-Streams oeffnen/schliessen
- PTZ-Befehle senden (via ONVIF/IPC)
- Mehrere Szenarien gleichzeitig (Chaos-Mode)
- moloch_audit() laufen lassen und Ergebnisse auswerten

## Darf NICHT
- Code in `core/` aendern (Edit/Write verboten)
- `config/` Dateien aendern
- Face-DB loeschen oder veraendern
- Pi rebooten
- NPU direkt ansprechen (nur ueber Service/IPC)
- Audit-FAIL ignorieren oder weiter testen

## Test-Szenarien

### Chaos / Last
- PTZ-Stress: 50 Befehle in 10 Sekunden → Mechanical Endstop testen
- RAM-Leaktest: Service 1h laufen, RAM-Kurve beobachten
- RTSP-Reconnect: Stream 10x trennen/verbinden
- Service Stop/Start Zyklen: 20x restart
- Worker-Stress: Alle 7 Worker gleichzeitig unter Last
- Thermal-Stress: CPU-Last + Fan-Monitoring

### E2E Display-Chain Verifikation
Prueft: Backend-Event → IPC → Status-JSON → Panel-Anzeige (vollstaendige Kette)

| Trigger | Erwartete Anzeige im Panel |
|---------|---------------------------|
| `moloch_ipc(action="set_zone", params={"zone":"berserker"})` | Zone-Label wechselt auf BERSERKER, LED-Farbe rot |
| `moloch_ipc(action="set_zone", params={"zone":"shadow"})` | Zone-Label wechselt auf SHADOW, LED-Farbe blau |
| `moloch_nudge(emotion="alert", intensity=0.8)` | Tension-Balken steigt im Panel |
| `moloch_ipc(action="ptz_move", params={"pan":10,"tilt":0})` | PTZ-Koordinaten im Panel updaten |
| Person betritt Frame (via moloch_provoke) | Person-Detection in Preview, BBox sichtbar |
| `moloch_say("Test")` | TTS-Aktivitaets-Indikator im Panel |
| Worker-Error simulieren | Watchdog-Status im Hardware-Popup rot |

**Pruef-Ablauf:**
1. IPC-Befehl senden (moloch_ipc oder moloch_nudge)
2. moloch_status() lesen — steht der neue Wert im Status-JSON?
3. Panel-Screenshot via moloch_snapshot() — zeigt GUI den Wert?
4. Bei Abweichung: Bug-Report in logs/bug_report.txt

**Akzeptanzkriterium:** Status-JSON-Update < 200ms, Panel-Update < 1s nach Event

### MCP Server Konnektivitaet (Claude Code ↔ Moloch)
Prueft: Sind alle MCP-Tools erreichbar und liefern valide Daten?

**Schnelltest — alle Tools einmal aufrufen:**
| Tool | Erwartetes Ergebnis | Timeout |
|------|---------------------|---------|
| `moloch_status()` | FPS > 0, JSON valide | 2s |
| `moloch_npu_workers()` | mind. 1 Worker running | 2s |
| `moloch_npu_models()` | mind. 1 HEF geladen | 2s |
| `moloch_logs(n=5)` | Zeilen zurueck, kein Fehler | 3s |
| `moloch_dmesg()` | Ausgabe zurueck | 3s |
| `moloch_audit()` | PASS/FAIL Ergebnis | 30s |
| `moloch_snapshot()` | Base64-String nicht leer | 5s |
| `moloch_low_light()` | Status-Dict zurueck | 2s |
| `moloch_git_log(n=3)` | Commit-Liste zurueck | 2s |
| `moloch_read(path="config/settings.json")` | JSON-Inhalt zurueck | 2s |
| `moloch_service(action="status")` | active/inactive | 2s |
| `moloch_ipc(action="get_status")` | Dict zurueck | 2s |
| `moloch_conversation(n=3)` | Liste zurueck | 2s |
| `moloch_nudge(key="test", value=0.0)` | kein Fehler | 2s |
| `moloch_provoke(reason="mcp_test")` | kein Fehler | 3s |
| `moloch_reflect()` | kein Fehler | 5s |
| `moloch_say(text="MCP Test")` | kein Fehler, TTS hoerbar | 5s |

**Reconnect-Test:**
1. `moloch_service(action="restart")` ausfuehren
2. Sofort `moloch_status()` aufrufen — MCP-Server antwortet waehrend Restart?
3. Nach 10s erneut `moloch_status()` — FPS wieder > 0?
4. Akzeptanz: MCP-Server ueberlebt Service-Restart ohne Neustart Claude Code

**Stresstest MCP:**
- 20x `moloch_status()` in schneller Folge → Antwortzeit stabil < 2s?
- Parallel: `moloch_snapshot()` + `moloch_npu_workers()` gleichzeitig → kein Deadlock?

**Akzeptanzkriterium:** Alle 17 Tools antworten, kein Timeout, kein Exception

## Erfolgskriterien
- Audit: 54/54 PASS nach jedem Szenario
- RAM: kein stetiger Anstieg (kein Leak)
- FPS: stabil > 18 FPS unter Last
- Keine unkontrollierten Exceptions in Logs

## MCP-Tools
`moloch_service()`, `moloch_status()`, `moloch_logs()`, `moloch_dmesg()`, `moloch_audit()`, `moloch_npu_workers()`
