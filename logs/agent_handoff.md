# Agent Handoff — 2026-04-02
# Session: Cloud-Session (Claude Opus 4.6)
# Branch: claude/investigate-windows-compatibility-7kbRm
# Status: ALLE AENDERUNGEN GEPUSHT — OPUS REVIEW EMPFOHLEN

---

## WAS DIESE SESSION ERLEDIGT HAT

### 1. Hooks & Automatisierung (Codewort: HOOKWIRE)
- 5 Hook-Scripts in .claude/hooks/ (Session-Start, Pre-Edit, Post-Edit, Pre-Bash, Stop)
- NEVER-Regeln automatisch erzwungen (Pan-Vorzeichen blockiert, Syntax-Check nach Edit)
- Permissions erweitert auf allow-all (AUTONOMIE-REGEL in CLAUDE.md)
- GitHub Action fuer Config-Audit bei jedem Push
- Datei: HOOKWIRE.md

### 2. System-Audit (Codewort: AUDIT-APRIL)
- 259 Dateien, 100.663 Zeilen gescannt
- 60+ subprocess ohne timeout (NEVER 5)
- 55+ json.dump ohne atomic write (NEVER 6)
- 1x shell=True in Produktionscode (audio_manager.py:552)
- Datei: AUDIT_2026-04-02.md (mit Checkboxen zum Abhaken)

### 3. Self-Tune System (Codewort: SELF-TUNE)
- 69 Parameter in config/self_tune_registry.json (59 GREEN, 10 YELLOW)
- Generischer IPC-Befehl 'self_tune' in moloch_service.py
- Datei: SELF-TUNE.md, config/self_tune_registry.json

### 4. HANDSHAKE Protokoll (Codewort: HANDSHAKE)
- Kommunikations-Protokoll MOLOCH <-> Claude Code via Git
- 3 Simulationen (SIM01 unrealistisch, SIM02+03 realistisch mit DeepSeek R1 Limitierungen)
- SIM03 hat ECHTES Problem gefunden: TTS length_scale an 4 Stellen hardcoded
- Dateien: HANDSHAKE.md, ipc/handshake*.json, logs/handshake*.log

### 5. TTS Speed Fix (aus SIM03 entstanden — ECHTER CODE-FIX)
- config/settings.json: Neue "tts" Sektion mit user_speed_offset
- core/personality/personality_engine.py: Liest user_speed_offset, addiert auf Zone-Speed
- core/voice_pipeline.py:697: Init-Default 1.1 → 0.95
- core/moloch_service.py: Generischer 'self_tune' IPC Action

### 6. SELF-MAP Konzept (Codewort: SELF-MAP)
- MOLOCHs maschinenlesbare Selbstbeschreibung
- generate_self_map.py noch zu implementieren
- Datei: SELF-MAP.md

### 7. Unconscious Engine (NEU — 398 LOC)
- core/unconscious_engine.py
- Tick-Loop alle 10s, daemon=True, Singleton
- Schicht 1: Mood (Shadow/Guardian Impulse)
- Schicht 2: Pipeline (Temp/FPS/RAM/Tracking/Face Self-Tune)
- 9 Regeln, Cooldown 30s, max 3 Tune/Stunde, Registry-Limits
- NOCH NICHT in moloch_service.py integriert!

### 8. Fan-Kurve
- scripts/fan_control.py: Noctua dreht ab 42°C hoch (statt 50°C)
- 3°C Hysterese, 5 Stufen, Ziel: CPU-Kuehler soll nicht anspringen

### 9. NPU Popup Slider
- 5 neue Slider in popup_npu_thresh.py
- Person Min-Hoehe/Flaeche, Pose NMS/Keypoint-Score, Hand Presence
- Service muss neue Keys noch verdrahten

---

## WAS OPUS REVIEWEN SOLLTE

### KRITISCH (vor Merge in main):
1. **personality_engine.py** (GELB) — user_speed_offset korrekt?
   Keine Regression bei Zone-Wechsel? _get_user_speed_offset() liest
   settings.json bei JEDEM voice_config Aufruf — Performance OK?
2. **moloch_service.py** (ROT) — self_tune Handler: Kann jemand ueber
   section/key beliebige Keys schreiben? Validierung gegen Registry noetig?
3. **unconscious_engine.py** (NEU) — Logik plausibel? Integration in Service.
4. **settings.json** — tts-Sektion kompatibel mit bestehendem Code?
5. **self_tune_registry.json** — alle 69 Parameter Grenzen plausibel?

### EMPFOHLEN:
6. Fan-Kurve auf Pi testen (42°C Ramp-Start)
7. Hooks auf Pi testen (Pfade, Permissions)
8. NPU Popup neue Slider verdrahten (set_tracker_param etc.)

---

## OFFENE PUNKTE

- [ ] unconscious_engine.py in moloch_service.py integrieren (start/stop)
- [ ] generate_self_map.py Script schreiben
- [ ] diagnose_rules.json erstellen (aus SELF-TUNE.md Konzept)
- [ ] Popup-Mockups fertigstellen (popups_mockup.html)
- [ ] GitHub Pages aktivieren (braucht Login)
- [ ] NEVER 5/6 Funde abarbeiten (60+ subprocess, 55+ json.dump)
- [ ] NPU Popup neue Slider im Service verdrahten

---

## GEAENDERTE DATEIEN (alle auf Branch, nicht auf main)

### Neue Dateien:
- core/unconscious_engine.py (398 LOC)
- config/self_tune_registry.json (69 Parameter)
- .claude/hooks/*.sh (5 Scripts)
- .github/workflows/moloch-audit.yml
- HOOKWIRE.md, SELF-TUNE.md, HANDSHAKE.md, SELF-MAP.md
- AUDIT_2026-04-02.md
- ipc/handshake*.json, logs/handshake*.log
- docs/main_panel_mockup.html, docs/npu_popup_mockup.html

### Geaenderte Dateien:
- config/settings.json (tts Sektion hinzugefuegt)
- core/personality/personality_engine.py (user_speed_offset)
- core/voice_pipeline.py (Init-Default 1.1 → 0.95)
- core/moloch_service.py (self_tune IPC Action)
- core/gui/popups/popup_npu_thresh.py (5 neue Slider)
- scripts/fan_control.py (Kurve aggressiver)
- .claude/settings.json (Permissions + Hooks)
- .gitignore (.claude/ nicht mehr komplett ignoriert)
- CLAUDE.md (AUTONOMIE-REGEL hinzugefuegt)

---

## FUER OPUS — EMPFOHLENE REIHENFOLGE

1. Lies diesen Handoff + CLAUDE.md (AUTONOMIE-REGEL beachten)
2. git diff main...HEAD — alle Aenderungen ueberblicken
3. Review: personality_engine.py + moloch_service.py
4. unconscious_engine.py in moloch_service.py integrieren
5. Auf Pi testen: Service-Restart, TTS-Test, Fan-Test
6. Bei Erfolg: Branch in main mergen
