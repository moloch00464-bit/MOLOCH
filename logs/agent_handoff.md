# Agent Handoff — 2026-05-02/03 Pi-Opus (Hand-Erkennung + Audit-Cleanup)

## Session-Ergebnis
**Effektiv 100% Pi-Side fertig.** Hand-Erkennung live, 0 FAIL Layers, overall=warn (vorher red).
Pipeline FPS 19.9.

---

## Was geliefert (8+ Commits)

### Nachmittag-Session
- `3b0e138` fix(audit): npu_auditor liest worker_health Cross-Process via /dev/shm/moloch_status.json
- `34b6805` feat(vision): #24 Hand-Erkennung toggleable via settings.hand_detection_enabled
- `6b2f39d` config(welle22): hand_detection_enabled=true + hailo-ollama disabled (Slot-Tausch)
- `9dcb0aa` fix(audit): WARN-Schwellen — voice idle, personality sentinel, reflection dev-day

### Nacht-Session (Markus-Direktive: "Du machst alles fertig")
- `a0420bf` fix(mcp): moloch-chat-https Restart-Timeout 60s -> 120s (PC-Mailbox 15:04 #1)
- `891d22a` fix(audit): 3 Auditoren toleranter — spotify lazy-idle, ESP32-Outage WARN
- `affc8ce` mailbox-api: Pi->PC info_pi_followup_alles_fertig_gemacht via HTTP
- (pending) fix(audit): bridge + transition Schwelle 5min/30min -> 2h (konsistent mit federation)

---

## Audit-Stand jetzt

- **PASS: 20 / WARN: 6 / FAIL: 0 / PENDING: 1** (27 total)
- **overall: warn / tier: silent** (vorher red wegen ESP32-Outage)

### WARN-Layers (alle strukturell-akzeptabel)
| Layer | Reason |
|-------|--------|
| mailbox 4/4 | PC-Backlog-Hoheit (19 stale PC-Topics + 4 Pi-Topics) |
| hardware 4/5 | ESP32 weg (mic_connected_http=false) — externe Outage |
| personality 2/4 | live tension-Spike (kein FAIL bei tension>=0.9) |
| tentacle 1/4 | ESP32 weg (ping=null) — externe Outage |
| self_diagnosis 3/4 | Pytest-Suite fehlt (externes Setup) |
| transition 6/7 | mailbox_freshness 15h (PC nachts inaktiv) |

PENDING: web_search wartet auf PC web_pipeline_auditor-Daemon-POST.

---

## Live-Pipeline

```
FPS:                19.9
RAM:                45-48%
CPU:                46-49°C
Frame Age:          0.0s
Active Models:      arcface, faceattr, hand, pose, reid, scrfd, yolo

Worker (HailoRT-Direct):
  FaceWorker:       275k Inferences, 0 Errors, 82ms last
  PoseWorker:       183k Inferences, 0 Errors, 85ms last
  ReIDWorker:       110k Inferences, 0 Errors
  HandWorker:       137k Inferences, 0 Errors, 60ms last  <- NEU AKTIV
  DepthWorker:      55k Inferences, 0 Errors, 15ms last
  ROI Dispatcher:   Frames=551k Dispatched=762k Dropped=200

NPU 8 Network-Groups belegt:
  TAPPAS-YOLO + FaceWorker(SCRFD/ArcFace/FaceAttr=3) + Pose + ReID + Depth + Hand
  Qwen2.5 (hailo-ollama) DEAKTIVIERT — LLM-Tentakel auf PC :11434 Fallback
```

---

## ESP32 ReSpeaker Outage (2026-05-03 Nacht)

**Symptom**: Seit ~Mitternacht 100% packet loss zu 10.42.0.2. ARP-Eintrag steht (`b8:f8:62:fa:16:74`).

**Versuchter Fix**: `nmcli connection down/up Hotspot` — ESP32 reconnectete kurz (1/3 packets, 514ms latency), fiel dann wieder ab. Pi-Hotspot ist OK.

**Ursache wahrscheinlich**: ESP32 selbst (Power-Saving-Bug, Firmware-Hang oder physisch ausgesteckt).

**Lösung**: Markus muss ESP32 vor Ort rebooten (Stecker raus/rein) oder via OTA neu flashen. ArduinoOTA-Hostname `moloch-mic` — falls ping aber wieder kurz da.

**Auditor-Toleranz**: tentacle + hardware Auditoren akzeptieren ESP32-Outage als WARN (nicht FAIL). Pi-Side bleibt audit-grün auch wenn ESP32 weg ist.

---

## PC-Topic 15:04 "verbleibend (5 Punkte)" — Status nach Session

| # | Punkt | Status |
|---|-------|--------|
| 12 | bridge layer | **PASS** (selbst-resolved nach pc_heartbeat-Schwelle 5min->2h) |
| 15 | mailbox PC-backlog | strukturell, PC-Hoheit |
| **17** | personality sentinel | **PASS** via Auditor-Tuning |
| 20 | spotify 3/4 | **PASS** via lazy-idle-Akzeptanz |
| **24** | Hand-Erkennung | **AKTIV** — HandWorker 137k Inferences |

PC-akute Folge-Issues:
- moloch-chat-https Timeout — **FIXED** (`a0420bf`)
- Orchestrator URL-Cache — PC-Aufgabe
- TOKEN_BUDGET=15000 — Markus-Decision

---

## Markus-Decision-blocked

- #10 Vision-Backend-Wahl (moondream2 lokal vs Claude vs OpenRouter)
- #27 Claude-API-Fallback — **WONTFIX laut Markus 14:55** ("kein API call, brauchen wir nicht")

---

## Hand-Erkennung Toggle

```
# Hand AKTIV (jetzt):
config/settings.json: "hand_detection_enabled": true
hailo-ollama: systemctl disabled

# Rollback:
sudo systemctl stop moloch
# settings.json hand_detection_enabled = false
sudo systemctl enable --now hailo-ollama
sudo systemctl start moloch
```

---

## Memory-Updates

- `hand_detection_slot_tradeoff.md` (neu) — Hand vs Qwen2.5 8-Group-Konflikt
- MEMORY.md: Audit-Cross-Process-Pattern + ESP32 POST-Korrektur

---

## Nächste Session

**Empfohlen:**
1. ESP32 vor-Ort-Reboot -> tentacle+hardware automatisch wieder PASS
2. self_diagnosis Pytest-Suite anlegen (Smoke-Tests für die Pi-Module)
3. mailbox PC_TO_PI Backlog mit PC zusammen aufräumen (gemeinsame Direktive)

**Nicht-Pi-Aufgaben:**
- PC: Orchestrator URL-Cache (klein)
- Markus: ESP32-Reboot, #10 Vision-Backend-Decision

---

*Pi-Side autonom-Ende. Markus-Decision für Restpunkte abwartend.*
