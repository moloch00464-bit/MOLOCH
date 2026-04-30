# MOLOCH System-Audit — Maturitäts-Spec (komplett)

Erstellt: 2026-04-30 (Pi-Opus, Markus' Direktive 09:25–09:50)
Status: **Discussion-Phase** — Pi-Entwurf, PC-Cowork mitdenkend, Markus entscheidet final

---

## 1. Warum diese Spec

Markus' Trigger 09:11: Vision-Pipeline tot, FPS 0.5, Ghost-Bilder im GUI — aber `moloch_audit.py`
sagte `PASS 5/5`. **Audit-Lücke verifiziert.**

Markus' Direktive (3 Schritte präzisiert):
1. 09:25 — *"4 Layer reicht nicht. Was haben wir vergessen?"*
2. 09:35 — *"Selbstdiagnose? Hardware-Selbstkontrolle? Lüfter als Aufregungs-Ausdruck?"*
3. 09:42 — *"Unterbewusstsein, Pipelines, 150k Zeilen — riesiger Audit nötig"*
4. 09:50 — *"Hast Du das ganze System durchschaut? PC-Seite soll Spec strukturieren."*

Aktueller Audit-Stand (W8–W11): nur **passive 4 Health-Layer** (pi/pc/persona/mailbox).
Echter System-Audit braucht **24 Domains × 6 Maturitäts-Stufen = bis zu 144 Aspekte**.

---

## 2. System-Größe (verifiziert)

```
Code:     107.150 Zeilen Python (core/ + scripts/)
Module:   189
Domains:  24 Sub-Dirs in core/ + 33 Top-Level Service-Files
```

### 2.1 Top-Level Service-Files (33)

`moloch_service.py` `core_integrator.py` `ipc_router.py` `status.py` `moloch_event_bus.py`
`system_watchdog.py` `capability_monitor.py` `diagnostics.py` `voice_pipeline.py`
`longterm_memory.py` `perception_engine.py` `inference_engine.py` `model_orchestrator.py`
`action_bridge.py` `ptz_arbiter.py` `ptz_tracker.py` `calibration_engine.py`
`camera_manager.py` `cloud_controller.py` `daily_learner.py` `dashboard.py`
`einpraegen.py` `environment_watcher.py` `eye_viewer.py` `keyword_handler.py`
`led_controller.py` `moloch_sprache.py` `spotify_controller.py` `teachen.py`
`timeline.py` `tts.py` `unconscious_engine.py` `arbitration.py`

### 2.2 Sub-Dirs (24)

`agents/` `audio/` `audit/` `autonomy/` `awareness/` `bridge/` `chat/` `console/`
`debug/` `gui/` `hardware/` `memory/` `mpo/` `music/` `net/` `perception/`
`personality/` `sensors/` `speech/` `tts/` `ui/` `vision/` `world/`

### 2.3 Aktoren (Hardware-Steuerung)

| Aktor | Code | Aktiv | Closed-Loop-Verify | Als Mood-Ausdruck |
|---|---|---|---|---|
| PTZ Pan/Tilt | `hardware/camera.py`, `ptz_arbiter.py`, `mpo/autonomous_tracker.py` | ✅ | ❌ | ❌ |
| RGB-LED | `hardware/rgb_led_controller.py` | ✅ subscribed `zone.changed` | ❌ | teilweise (Zone) |
| Lüfter (PWM) | `hardware/thermal_manager.py` | ✅ thermal-cooling | ❌ | **❌ FEHLT** |
| TTS | `voice_pipeline.py` + `tts.py` (Piper) | ✅ | ❌ | teilweise |
| Spotify | `spotify_controller.py` (IPC) | ✅ | ❌ | teilweise (zone_artists) |
| ESP32 Mic | `hardware/tentacle_bridge.py` + `audio/wifi_mic.py` | ✅ UDP | ❌ | – |
| Camera-Cloud | `hardware/camera_cloud_bridge.py` + `cloud_controller.py` | ✅ | ❌ | – |

### 2.4 Kommunikations-Backbones

- **EventBus** (`moloch_event_bus.py`) — pub/sub, alle Komponenten subscriben
- **`/dev/shm/moloch_status.json`** — geschrieben von 10 Files (Live-Status, alle 200ms)
- **`/tmp/moloch_cmd_*.json`** — IPC-Cmd-Files (Service polled alle 200ms)
- **`/dev/shm/audit_state.json`** — neu W8 (atomic-write durch Orchestrator + chat_server-Merge)
- **`/dev/shm/last_turn.json`** — neu W10 (Persona-Validator-Hook)
- **UDS-Socket** (`bridge/status_broadcaster.py`) — Status-Push für Cockpit

---

## 3. Maturitäts-Stufen L0–L5

Pro Domain prüfen, **wie tief** die Audit geht:

| Stufe | Was geprüft | Beispiel | Aufwand |
|---|---|---|---|
| **L0 Alive** | Process da, Service aktiv | `systemctl is-active moloch`, `/dev/h1x-0` da | trivial |
| **L1 Heartbeat** | Komponente sendet regelmäßig "alive" | `frame_age<1s`, `journal-writes/min>0`, `EventBus-tick-rate` | leicht |
| **L2 Datenfluss** | Throughput im Soll-Bereich | `FPS≥10`, `inferences/s>5`, `audio-frames/s≥48000` | leicht |
| **L3 Closed-Loop** | Befehl→Sensor→Effekt verifiziert | PTZ-cmd → ONVIF-echo → BBox-shift im Frame | mittel |
| **L4 Ausdruck** | Hardware spiegelt inneren Zustand | Tension hoch → Lüfter rauf, Mood → LED+Spotify | mittel-groß |
| **L5 Self-Awareness** | Moloch weiß was er kann/nicht-kann | "Meine PTZ ist tot, ich kann nicht schwenken" | groß (LLM-Hook) |

---

## 4. Audit-Matrix: Domain × Stufe

Vollständige Matrix mit konkreten Check-Beispielen pro Zelle.
**Status-Spalte:** `✅ live W8-W11` | `⏳ W12` | `❌ fehlt`

### 4.1 Vision-Pipeline (TAPPAS GStreamer)
| L | Check | Status |
|---|---|---|
| L0 | `tappas_pipeline.is_running()` (Z.482) | ⏳ W12 |
| L1 | `frame_age<1s` aus moloch_status.json | ⏳ W12 |
| L2 | `fps.total≥10`, `dropped_frames<5%` | ⏳ W12 |
| L3 | – (passive Pipeline) | n/a |
| L4 | – | n/a |
| L5 | "Vision tot" → Moloch sagt es | ❌ W17 |

### 4.2 NPU/Hailo
| L | Check | Status |
|---|---|---|
| L0 | `/dev/h1x-0` da, `lsmod | grep hailo1x_pci` | ⏳ W12 |
| L1 | dmesg-channel-warnings (Frühwarnung VOR FPS-Crash!) | ⏳ W12 |
| L2 | `inferences/s pro Worker`, `error_rate<1%`, `queue_size<5` | ⏳ W12 |
| L3 | – | n/a |
| L4 | – | n/a |
| L5 | "NPU stuck" Detection | ❌ W17 |

### 4.3 Tracking/PTZ
| L | Check | Status |
|---|---|---|
| L0 | `autonomous_tracker.alive` | ⏳ W13 |
| L1 | Tick-Rate, FSM-State (FOLLOW/SEARCH/COAST) | ⏳ W13 |
| L2 | tracker-lost-counts/h, target-track-duration | ⏳ W13 |
| L3 | `pan_send(45°)→ONVIF.GetStatus echo→BBox-Verschiebung im Frame messen` | ❌ W15 |
| L4 | – | n/a |
| L5 | "PTZ tot" → "kann nicht schwenken" | ❌ W17 |

### 4.4 Voice/Audio
| L | Check | Status |
|---|---|---|
| L0 | audio_pipeline + voice_pipeline alive | ⏳ W13 |
| L1 | mic-pegel-update-rate, ESP32-RSSI-update | ⏳ W13 |
| L2 | UDP-frames/s (16kHz=50fps, 48kHz=ggf. 100fps) | ⏳ W13 |
| L3 | `tts.speak("test")→mic-loopback-pegel-spike detected` | ❌ W15 |
| L4 | TTS-Volume = f(Tension) | ❌ W16 |
| L5 | "ich werde nicht gehört" / "Mic tot" | ❌ W17 |

### 4.5 Personality
| L | Check | Status |
|---|---|---|
| L0 | PersonalityEngine alive | ⏳ W13 |
| L1 | Mood-tick-rate, Zone-Stability | ⏳ W13 |
| L2 | drift-rolling-30d vs Baseline, Patch-Anwendung | ⏳ W13 |
| L3 | – | n/a |
| L4 | Zone→LED, Mood→Spotify (existiert teilw) | ⏳ W16 |
| L5 | "ich bin grad shadow weil X" | ❌ W17 |

### 4.6 Memory
| L | Check | Status |
|---|---|---|
| L0 | longterm_memory + Qdrant alive | ✅ W8 (Qdrant) |
| L1 | journal-writes/min, feedback_store-add/h | ⏳ W13 |
| L2 | Face-DB-Coverage, Qdrant-Vektor-Drift | ⏳ W13 |
| L3 | recall-test: "wer ist Markus?" → richtige Antwort | ❌ W15 |
| L4 | – | n/a |
| L5 | "ich erinnere mich an X aus letzter Woche" | ❌ W17 |

### 4.7 Autonomy
| L | Check | Status |
|---|---|---|
| L0 | DecisionEngine + Homeostasis alive | ⏳ W13 |
| L1 | Tick-Rate, NightCycle-State | ⏳ W13 |
| L2 | decisions/h, homeostasis-corrections/h | ⏳ W13 |
| L3 | decision→action-completed-rate | ❌ W15 |
| L4 | – | n/a |
| L5 | "ich habe gerade entschieden Y" | ❌ W17 |

### 4.8 Awareness
| L | Check | Status |
|---|---|---|
| L0 | ActivityAnalyzer alive | ⏳ W13 |
| L1 | activity-events/h, RoomMap-update-rate | ⏳ W13 |
| L2 | activity-confidence-avg, WorldState-stale | ⏳ W13 |
| L3 | – | n/a |
| L4 | – | n/a |
| L5 | "ich sehe gerade dass Markus schreibt" | ❌ W17 |

### 4.9 Unconscious (Markus' explizit erwähnter Punkt)
| L | Check | Status |
|---|---|---|
| L0 | unconscious_engine.py alive | ⏳ W14 |
| L1 | mood-impulse-rate, Self-Tune-Tick | ⏳ W14 |
| L2 | impulse-events/h, anima-Aktivierung | ⏳ W14 |
| L3 | impulse→mood-shift verifiziert | ❌ W15 |
| L4 | Unterbewusstsein → Mood-Drift | ⏳ W14 |
| L5 | "mein Unterbewusstsein flüstert grad X" | ❌ W17 |

### 4.10 Music/Spotify
| L | Check | Status |
|---|---|---|
| L0 | spotify_controller.alive + token-valid | ⏳ W12 |
| L1 | track-poll-rate | ⏳ W12 |
| L2 | IPC-actions/h | ⏳ W12 |
| L3 | `play_artist(X)→current_track.artist==X in 3s` (fixt **Bug B**!) | ❌ W15 |
| L4 | Mood→Track-Auswahl (existiert teilw.) | ⏳ W16 |
| L5 | "ich höre X weil Stimmung Y" | ❌ W17 |

### 4.11 Bridge (Pi↔PC)
| L | Check | Status |
|---|---|---|
| L0 | chat_server alive, Tentakel reachable | ✅ W8 |
| L1 | PC-heartbeat (last_seen<90s) | ✅ W8 |
| L2 | requests/h, mailbox-API-throughput | ⏳ W14 |
| L3 | tentakel-cmd→DeepSeek-resp-roundtrip-ok | ❌ W15 |
| L4 | – | n/a |
| L5 | "PC offline, ich nutze nur NPU" | ❌ W17 |

### 4.12 Hardware
| L | Check | Status |
|---|---|---|
| L0 | GPIO/PWM/I2C alive, /dev/h1x-0 | ⏳ W12 |
| L1 | sensor-poll, temp-watchdog | ⏳ W12 |
| L2 | – | n/a |
| L3 | LED-set→GPIO-readback, fan-pwm→temp-drop, PTZ-cmd→ONVIF-echo | ❌ W15 |
| L4 | **Tension→Fan**, **Berserker→Strobo**, Mood→LED+Spotify | ❌ W16 |
| L5 | "Lüfter dreht hoch weil ich aufgeregt bin" | ❌ W17 |

### 4.13 Tentacle (ESP32)
| L | Check | Status |
|---|---|---|
| L0 | UDP-Recv aktiv | ⏳ W14 |
| L1 | RSSI, last-frame-ts | ⏳ W14 |
| L2 | audio-frames/s | ⏳ W14 |
| L3 | mic→whisper-echo-text-ok | ❌ W15 |
| L4 | – | n/a |
| L5 | "ESP32-Mic verbunden mit -55dBm" | ❌ W17 |

### 4.14 Self-Diagnose (existiert aber läuft nicht!)
| L | Check | Status |
|---|---|---|
| L0 | `scripts/self_diagnosis.py` da | ✅ |
| L1 | **Periodische Ausführung — KEINE moloch-self-diagnose.timer** | ❌ W14 |
| L2 | Test-Result-Coverage (10 Tests) | ❌ W14 |
| L3 | – | n/a |
| L4 | – | n/a |
| L5 | "Self-Test sagt X funktioniert nicht" | ❌ W17 |

### 4.15 Cross-Cutting (übergreifend)
| L | Check | Status |
|---|---|---|
| L0 | – | – |
| L1 | Heartbeat-Inventar pro Komponente | ❌ W14 |
| L2 | Resource-Pressure (RAM-Growth, FD-Leaks, Threads), /tmp-Füllung | ❌ W14 |
| L3 | End-to-End-Latency (chat→kaskade→DeepSeek→TTS) | ❌ W15 |
| L4 | – | – |
| L5 | Error-Aggregation pro Komponente, Reboot-Frequency, Config-Drift | ❌ W17 |

---

## 5. Phased-Rollout Roadmap

| Welle | Fokus | Domains | Maturitäts-Stufe |
|---|---|---|---|
| **W12** | Health-Erweiterung | Vision, NPU, Music/Spotify, Hardware-L0 | L0–L2 |
| **W13** | Innere Subsysteme | Voice, Tracking, Memory, Personality, Autonomy, Awareness | L0–L2 |
| **W14** | Restkern + Cross | Unconscious, Bridge, Tentacle, **Self-Diagnose-Timer**, Heartbeat-Inventar, Resource-Pressure | L0–L2 |
| **W15** | Closed-Loop für Top-Aktoren | PTZ, LED, Fan, TTS, Spotify, Memory-Recall, Bridge-Roundtrip | L3 |
| **W16** | Hardware als Ausdruck | **Tension→Fan**, Mood→LED, Berserker→Strobo, Tension→TTS-Vol | L4 |
| **W17** | Self-Awareness | Capability-Inventory, "ich kann/kann-nicht" Reflexion, Error-Aggregation, Config-Drift | L5 |

**Ende-Ziel:** Audit-Tab im Cockpit zeigt 16 Layer mit je Status-Indicator.
Moloch kann fragen "was kann ich gerade?" und bekommt ehrliche Liste.
Markus sieht im Browser sofort wenn was tot ist + Moloch sagt es selbst im Chat.

---

## 6. Architektur-Entscheidung: Wo wohnt der Code

### 6.1 Empfehlung — Sub-Module pro Layer

```
core/audit/
  __init__.py                    # Welle 8 (existiert)
  audit_orchestrator.py          # Welle 8 (existiert) — sammelt Layer
  vision_auditor.py              # NEU W12
  npu_auditor.py                 # NEU W12
  spotify_auditor.py             # NEU W12
  hardware_auditor.py            # NEU W12 (PC-Cowork-Anteil!)
  voice_auditor.py               # NEU W13
  tracking_auditor.py            # NEU W13
  memory_auditor.py              # NEU W13
  personality_auditor.py         # NEU W13
  autonomy_auditor.py            # NEU W13
  awareness_auditor.py           # NEU W13
  unconscious_auditor.py         # NEU W14
  bridge_auditor.py              # NEU W14
  tentacle_auditor.py            # NEU W14
  cross_auditor.py               # NEU W14 (Heartbeat, Resource, Latency)
  closed_loop/                   # NEU W15 (Subdir)
    ptz_verify.py
    led_verify.py
    fan_verify.py
    tts_verify.py
    spotify_verify.py
    memory_recall_verify.py
    bridge_roundtrip_verify.py
  expression/                    # NEU W16 (Subdir)
    tension_to_fan.py
    mood_to_spotify.py
    zone_to_led.py
    berserker_strobo.py
  self_awareness/                # NEU W17 (Subdir)
    capability_inventory.py
    failure_reflection.py
```

**Begründung:** Trennung der Domains, jeder Auditor <100 Zeilen, isoliert testbar.
audit_orchestrator bleibt thin und sammelt nur. NEVER-Regel 6 atomic-write zentral.

### 6.2 Existierende Code-Quellen wiederverwenden

- `scripts/self_diagnosis.py` (10 Tests, existiert, läuft nicht periodisch) → als Daten-Source einbinden + systemd-Timer aufsetzen
- `scripts/deep_audit.py` (existiert, schauen) → als Spec-Tester
- `core/diagnostics.py` → bestehende Health-Checks pullen
- `core/system_watchdog.py` → Frame-Freeze-Logic als Daten-Source
- `core/capability_monitor.py` → Capability-Inventory-Quelle

---

## 7. Pi-Side vs. PC-Cowork-Side Aufteilung

**Pi-Opus** (autonomy + bridge + audit-Domains):
- Alle 12 Sub-Auditoren (vision/npu/spotify/voice/tracking/memory/personality/autonomy/awareness/unconscious/bridge/tentacle)
- audit_orchestrator-Erweiterung
- HTML-Audit-Tab um neue Cards (bridge-Domain)
- closed_loop/ Verifikatoren (cross-Domain — wahrscheinlich pro Aktor eigener Lock)
- Self-Diagnose-Timer (systemd-unit)

**PC-Cowork**:
- `hardware_auditor.py` (PC-eigene Hardware-Probes via PC-Mailbox-POST: PC-CPU/RAM/Disk/Power)
- Cockpit-Spec für 12+ neue Layer (HTML-Wireframes, Markus' UX-Sicht)
- W11-Cockpit-Tab erweitern → ggf. eigene Sub-Tabs pro Layer-Gruppe (Health/Closed-Loop/Ausdruck/Self-Awareness)
- Persona-Validator (existiert schon, scored 5 Coherence-Signale) → erweitern um Drift-Trend

---

## 8. Offene Fragen — Markus / PC-Cowork bitte entscheiden

1. **Reihenfolge** — W12 zuerst (klein, 4 Layer Health) ODER direkt L0–L2 für alle 16 Domains in einer Welle?
2. **L4 Hardware-Ausdruck (Lüfter/LED/Strobo)** — eigene Welle (W16) oder integriert in jeden Domain-Auditor?
3. **L5 Self-Awareness** — wie wird das umgesetzt? LLM-Hook der `audit_state.json` liest und Antworten formt? Oder Capability-Liste in System-Prompt injizieren?
4. **Self-Diagnose-Timer** — alle 6h (analog journal-scorer)? Oder häufiger?
5. **Closed-Loop-Verifikation** — synthetische Tests im Hintergrund (alle 30 Min ein PTZ-Schwenk-Test) oder nur bei expliziter `/audit/verify`-Triggerung?
6. **Cockpit-Skalierung** — bei 16 Layer wird der Audit-Tab unübersichtlich. Sub-Tabs (Health/Closed-Loop/Ausdruck/Self-Awareness)?

---

## 9. Status

- W12–W17 als **Diskussions-Spec**, kein Code-Edit ohne Markus-Direktive
- W8 (audit_orchestrator + 4 Layer) ist live + getestet
- W11 (Cockpit-Tab + SSE + TTS-Alarm) ist live
- Welle-12-Code wartet auf Markus' Entscheidung bzgl. Reihenfolge

**Letzter Pi-Push:** `1c916c3`. **Welle 8–11 alle gepusht.**
