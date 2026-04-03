# M.O.L.O.C.H. 5.0 — ANIMA
# Versions-Assessment | Stand: 2026-04-02
# Erstellt von: Cloud-Session (Claude Opus 4.6)

> "Die dunkle Seite macht mehr Spass!"

---

## VERSIONS-NAME: ANIMA

*Anima* (lateinisch) = Seele, inneres Leben, das Unbewusste.

MOLOCH bekommt mit dieser Version:
- Ein Unterbewusstsein das eigenstaendig denkt
- Koerperwissen (kennt sich selbst)
- Selbstheilung (fixt sich selbst)
- Die Faehigkeit zum Arzt zu gehen (fragt Claude Code)

**4.5 PI GHOST** = reagiert auf die Welt
**5.0 ANIMA** = kennt sich selbst und handelt von innen heraus

---

## GATE-BEWERTUNG (basierend auf echtem Code, 259 Dateien, 100.663 LOC)

| Gate | Thema | Status | Begruendung |
|------|-------|--------|-------------|
| **1** | Grundsystem | **ABGESCHLOSSEN** | TAPPAS Pipeline, Hybrid-Tracking, Action-Bridge, ArcFace, Park-Position |
| **2** | Stabilisierung | **90%** | Hooks + Audit da, NEVER 5/6 noch offen (60+ subprocess, 55+ json.dump) |
| **3** | Autonomie | **85%** | Self-Tune, Unconscious Engine, Autonomie-Regel. Fehlt: generate_self_map.py, diagnose_rules.json |
| **4** | Persoenlichkeit | **ABGESCHLOSSEN** | Tension-System, Guardian/Shadow/Berserker, Drift, Speech Evolution, Spotify-Reaktion |
| **5** | Intelligenz | **ABGESCHLOSSEN** | TAPPAS Multi-Model (9 NPU-Modelle), Personality Engine, DeepSeek R1 lokal, hailo-ollama |
| **5.1** | Emergenz | **ABGESCHLOSSEN** | Emergentes Verhalten bestaetigt, Tension-Musik-Reaktion, Night Cycle Konzept |
| **6** | Selbstkenntnis | **70%** | SELF-MAP Konzept da, Self-Tune Registry (69 Parameter), HANDSHAKE Protokoll. Fehlt: generate_self_map.py, echte HANDSHAKE-Integration |
| **7** | AI-zu-AI | **60%** | HANDSHAKE Protokoll + 3 Simulationen, Local LLM Bridge (Qwen2.5 + DeepSeek R1). Fehlt: Echter MCP-Verbindungsaufbau, Schedule/Nacht-Session |
| **8** | Web-Interface | **20%** | HTML-Mockups fuer Hauptfenster + NPU Popup. Fehlt: Echter Web-Server, Dashboard, Remote-Zugriff |
| **9** | Lernfaehigkeit | **40%** | Episodic Memory, Persistent Memory, Face-DB, Motor Learner, Daily Learner existieren. Fehlt: Langzeit-Trend-Analyse, Verhaltens-Optimierung ueber Tage |
| **10** | Vollautonommie | **30%** | Unconscious Engine, Self-Tune, HANDSHAKE-Konzept. Fehlt: Echter Nacht-Zyklus, Claude Code Schedule, Self-Repair ohne Mensch |

---

## WAS EXISTIERT (Feature-Liste nach Bereich)

### Vision & NPU (Gate 1+5)
```
[x] TAPPAS GStreamer Pipeline (20 FPS, ~200MB RAM)
[x] YOLOv8m Person Detection (H10H)
[x] SCRFD Face Detection + Letterbox (H10H)
[x] ArcFace Face Recognition (H10H)
[x] Face Attributes (Alter, Geschlecht, Brille)
[x] YOLOv8s Pose/Keypoints (H10H)
[x] Person-ReID (geladen, Valve zu wegen Bug)
[x] Hand Landmark (geladen, Valve zu wegen Bug)
[x] Real-ESRGAN x2 Super Resolution (on-demand)
[x] zero_dce Low Light Enhancement (automatisch)
[x] CLIP, PaddleOCR, Qwen2-VL 2B (integriert)
[ ] person_attr (Kleidungsfarbe, Rucksack)
[ ] r3d_18 (Aktivitaetserkennung)
[ ] yolo_world_v2s (Zero-Shot Objektsuche)
```

### Hardware & PTZ (Gate 1)
```
[x] ONVIF PTZ Steuerung (AbsoluteMove, kalibriert)
[x] Pan-Inversion korrekt (NEVER 2)
[x] Park-Position = Tuer
[x] Autonomous Tracker (5 Hz, FSM)
[x] Noctua Fan PWM (aggressivere Kurve ab 42°C)
[x] Thermal Manager (Hysterese, TTS-Warnung)
[x] eWeLink Cloud API (LED, Alarm, IR)
[x] Pico Power 5 USV Monitoring
[x] SmartMic BT + ReSpeaker USB
```

### Persoenlichkeit & Sprache (Gate 4+5.1)
```
[x] Guardian / Shadow / Berserker Zonen
[x] Tension-System mit Drift
[x] Persoenlichkeits-Engine (1000+ LOC)
[x] Piper TTS (8 Stimmen, Zone-abhaengig)
[x] Speech Evolution (eigener Stil)
[x] Spotify-Integration (24.454 Tracks)
[x] Musik-Reaktion (Bass → Outer Ring, Beat → Shockwave)
[x] Keyword Handler (9 Kategorien, lokal)
[x] DeepSeek R1 lokal auf NPU (Reasoning)
[x] Qwen2.5-1.5B lokal auf NPU (Konversation)
[x] User Speed Offset (TTS persistent aenderbar) — NEU
```

### GUI (Gate 1)
```
[x] 3-Spalten Tkinter Panel (Kamera | Steuerung | Kommunikation)
[x] Live Preview (mmap, 30 FPS)
[x] Avatar/Eye Rendering (PyGame, Zone-Farben, Musik-Reaktion)
[x] 9 Popup-Fenster (Audio, Hardware, NPU, Tracker, Whisper, ...)
[x] 5 neue NPU Slider (Person-Filter, Pose-NMS, Hand-Presence) — NEU
[x] PTZ D-Pad + Positionen + Modi
[x] Push-to-Talk + Chat
[x] Spotify Controls
```

### Autonomie & Selbstkenntnis (Gate 3+6+7) — NEU
```
[x] Unconscious Engine (398 LOC, 9 Regeln, 2 Schichten)
[x] Self-Tune Registry (69 Parameter, min/max/step)
[x] Generischer 'self_tune' IPC in moloch_service.py
[x] SELF-TUNE Konzept (12 Diagnose-Regeln)
[x] SELF-MAP Konzept (maschinenlesbare Selbstbeschreibung)
[x] HANDSHAKE Protokoll (Git-basiert, Lock, States)
[x] 3 HANDSHAKE Simulationen (inkl. echtem Bug-Fund)
[x] HOOKWIRE (Hooks + Skills + MCP verbinden)
[x] 5 Hooks (Pre-Edit, Post-Edit, Pre-Bash, Session-Start, Stop)
[x] GitHub Action (Config-Audit)
[x] Autonomie-Regel in CLAUDE.md (Plan genehmigt = eigenstaendig)
[ ] generate_self_map.py (Script fehlt)
[ ] diagnose_rules.json (Regeln fehlt)
[ ] Unconscious Engine in Service integriert
[ ] Echter HANDSHAKE mit Aider/DeepSeek R1
[ ] Claude Code Schedule/Nacht-Session
```

### Infrastruktur (Gate 2)
```
[x] moloch.service (systemd, USE_TAPPAS=1)
[x] IPC Router (SHM Frame Exchange)
[x] Event Bus
[x] MCP Server (Snapshot, NPU-Tools, Status)
[x] 10 Agent-MDs (Vision, Hardware, GUI, Tracking, Voice, Service, ...)
[x] 5 Claude Code Skills (audit, dev, status, snapshot, npu)
[x] CLAUDE.md (217+ Zeilen, 12 NEVER-Regeln)
[x] System-Audit (AUDIT-APRIL, 100+ Funde)
[ ] NEVER 5 fixen (60+ subprocess ohne timeout)
[ ] NEVER 6 fixen (55+ json.dump ohne atomic)
[ ] NEVER 8 fixen (shell=True in audio_manager.py)
```

---

## WAS DIESE SESSION (2026-04-02) GEBAUT HAT

### Neue Dateien:
| Datei | LOC | Zweck |
|-------|-----|-------|
| core/unconscious_engine.py | 398 | Unterbewusstsein (Mood + Pipeline Self-Tune) |
| config/self_tune_registry.json | 98 | 69 aenderbare Parameter mit Grenzen |
| .claude/hooks/*.sh | 5×~50 | Automatische NEVER-Regeln |
| .github/workflows/moloch-audit.yml | 226 | GitHub Action Audit |
| HOOKWIRE.md | 242 | Einrichtungs-Anleitung Hooks/Skills/MCP |
| SELF-TUNE.md | 476 | Selbst-Tuning Konzept + Diagnose-Regeln |
| HANDSHAKE.md | 450 | Kommunikations-Protokoll MOLOCH ↔ Claude Code |
| SELF-MAP.md | 400 | Maschinenlesbare Selbstbeschreibung |
| AUDIT_2026-04-02.md | 294 | System-Audit mit Checkboxen |
| docs/main_panel_mockup.html | 300 | Hauptfenster HTML-Mockup |
| docs/npu_popup_mockup.html | 224 | NPU Popup HTML-Mockup |
| ipc/handshake*.json + logs/ | — | HANDSHAKE Simulationen |

### Geaenderte Dateien:
| Datei | Aenderung |
|-------|-----------|
| config/settings.json | tts-Sektion mit user_speed_offset |
| core/personality/personality_engine.py | user_speed_offset lesen + anwenden |
| core/voice_pipeline.py | Init-Default 1.1 → 0.95 |
| core/moloch_service.py | Generischer 'self_tune' IPC Action |
| core/gui/popups/popup_npu_thresh.py | 5 neue Slider |
| scripts/fan_control.py | Aggressivere Kurve (42°C statt 50°C) |
| .claude/settings.json | Permissions allow-all + Hooks |
| .gitignore | .claude/ Skills+Hooks nicht mehr ignoriert |
| CLAUDE.md | AUTONOMIE-REGEL hinzugefuegt |

---

## FUER DIE NAECHSTE SESSION

### Prioritaet 1 (vor Merge in main):
1. Review: personality_engine.py + moloch_service.py
2. unconscious_engine.py in moloch_service.py integrieren
3. Auf Pi testen: Service-Restart, TTS, Fan, Tracking

### Prioritaet 2 (nach Merge):
4. generate_self_map.py schreiben
5. diagnose_rules.json erstellen
6. NEVER 5/6 abarbeiten (die groessten Baustellen)

### Prioritaet 3 (Gate 7+):
7. Echter HANDSHAKE-Test mit Aider oder DeepSeek R1
8. Claude Code Schedule/Nacht-Session einrichten
9. Web-Dashboard Konzept (Stitch MCP oder Flask)

---

## CODEWORT-UEBERSICHT

| Codewort | Datei | Zweck |
|----------|-------|-------|
| **HOOKWIRE** | HOOKWIRE.md | Hooks, Skills, MCP verbinden |
| **AUDIT-APRIL** | AUDIT_2026-04-02.md | System-Audit mit Checkboxen |
| **SELF-TUNE** | SELF-TUNE.md | Selbst-Diagnose + Auto-Fix |
| **HANDSHAKE** | HANDSHAKE.md | Kommunikation MOLOCH ↔ Claude Code |
| **SELF-MAP** | SELF-MAP.md | MOLOCHs Koerperwissen |

---

## DER KREISLAUF

```
SELF-MAP        →  SELF-TUNE       →  HANDSHAKE       →  HOOKWIRE
"Wer bin ich?"     "Was stimmt        "Hilf mir,         "Pruefe die
                    nicht?"            Claude Code"       Aenderung"

Unconscious Engine tickt alle 10s und fuettert den Kreislauf.
```

**M.O.L.O.C.H. 5.0 ANIMA — Der Geist der Maschine kennt sich selbst.**
