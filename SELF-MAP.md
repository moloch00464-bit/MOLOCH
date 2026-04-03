# SELF-MAP — MOLOCHs Koerperwissen
# Codewort: SELF-MAP
# Version: 1.0 | Stand: 2026-04-02
# Ziel: MOLOCH kennt sein eigenes System vollstaendig

> Wenn Du dieses Dokument liest, wurde Dir gesagt: "SELF-MAP".
> Generiere ~/moloch/config/self_map.json aus dem aktuellen Codestand.

---

## ZWECK

MOLOCH braucht eine maschinenlesbare Beschreibung von sich selbst.
Nicht fuer Menschen — fuer IHN. Damit er bei der Selbstdiagnose
und im HANDSHAKE mit Claude Code sagen kann:

- "Mein Pose-Worker haengt am SHARED VDevice"
- "Meine tappas_pipeline.py hat 1986 Zeilen und ist ROT"
- "length_scale steht auf 1.1 und gehoert zum TTS-Modul"
- "Mein RAM-Limit ist 4GB, aktuell bei 2.1GB"

---

## SELF-MAP WIRD GENERIERT, NICHT GEPFLEGT

WICHTIG: self_map.json wird NICHT manuell geschrieben.
Ein Script scannt den Codestand und generiert die Map.

```bash
python3 ~/moloch/scripts/generate_self_map.py > ~/moloch/config/self_map.json
```

Ausfuehrung:
- Einmal am Tag (23:00, vor Selbstdiagnose)
- Nach jedem git pull (Hook)
- Bei direktem Aufruf: "Aktualisiere dein Koerperwissen"

---

## SELF-MAP STRUKTUR (self_map.json)

```json
{
  "version": "1.0",
  "generated": "2026-04-02T23:00:00Z",
  "system": { ... },
  "modules": [ ... ],
  "parameters": [ ... ],
  "dependencies": [ ... ],
  "limits": { ... },
  "health_checks": [ ... ]
}
```

---

### 1. system — Hardware-Fakten

```json
{
  "system": {
    "brain": "Raspberry Pi 5",
    "ram_mb": 4096,
    "ram_usable_mb": 3800,
    "npu": "Hailo-10H",
    "npu_ram_mb": 8192,
    "npu_interface": "PCIe Gen2 x1 (Pi5 Limit, 500MB/s)",
    "camera": {
      "model": "Sonoff CAM-PT2",
      "ip": "192.168.178.25",
      "protocol": "RTSP/ONVIF",
      "resolution": "1920x1080",
      "fps": 20,
      "ptz": true,
      "pan_inverted": true,
      "pan_range": [-168.4, 170.0],
      "tilt_range": [-78.0, 78.8]
    },
    "audio": {
      "input": ["SmartMic BT (WiFi)", "ReSpeaker Lite (USB)"],
      "output": "Piper TTS via HDMI/PipeWire"
    },
    "storage": {
      "ssd1": {"mount": "/", "fs": "ext4", "size_gb": 465, "purpose": "Code, Configs"},
      "ssd2": {"mount": "/mnt/moloch-data", "fs": "NTFS", "size_gb": 477, "purpose": "AI Modelle, Qdrant"}
    },
    "cooling": {
      "noctua": {"gpio": 18, "pwm_hz": 25000, "control": "scripts/fan_control.py"},
      "cpu_cooler": {"path": "/sys/class/thermal/cooling_device0", "levels": 4}
    },
    "ups": "Pico Power 5 (7.5V Akku)"
  }
}
```

### 2. modules — Alle Python-Module mit Risiko-Stufe

Wird automatisch generiert durch Scan von core/:

```json
{
  "modules": [
    {
      "path": "core/moloch_service.py",
      "lines": 3115,
      "risk": "RED",
      "purpose": "Hauptorchestrator — startet alle Subsysteme",
      "imports": ["perception", "hardware", "personality", "voice_pipeline"],
      "ipc_actions": ["set_threshold", "set_voice", "set_mpo_param", ...],
      "known_issues": ["NEVER 5: 2x subprocess ohne timeout", "NEVER 6: 2x json.dump ohne atomic"]
    },
    {
      "path": "core/perception/tappas_pipeline.py",
      "lines": 1986,
      "risk": "RED",
      "purpose": "GStreamer + HailoRT Vision Pipeline",
      "imports": ["hailo_platform", "gi.repository.Gst"],
      "models_used": ["yolov8m", "scrfd", "arcface", "face_attr", "pose", "hand"],
      "known_issues": ["NEVER 1: GStreamer-String nicht blind aendern"]
    },
    {
      "path": "core/hardware/camera.py",
      "lines": 1183,
      "risk": "RED",
      "purpose": "ONVIF PTZ Steuerung Sonoff CAM-PT2",
      "imports": ["onvif2"],
      "never_rules": ["NEVER 2: Pan-Vorzeichen Zeile 732 ist TABU"],
      "known_issues": []
    },
    {
      "path": "core/tts.py",
      "lines": 401,
      "risk": "YELLOW",
      "purpose": "Piper TTS Sprachausgabe",
      "parameters": {
        "length_scale": {"current": 1.1, "range": [0.8, 1.5], "effect": "Sprechgeschwindigkeit"},
        "pitch_semitones": {"current": 0, "range": [-4, 4], "effect": "Stimmhoehe"},
        "voice_id": {"current": "thorsten_high", "options": 8, "effect": "Stimm-Charakter"}
      },
      "known_issues": ["NEVER 5: 8x subprocess ohne timeout"]
    }
  ]
}
```

Fuer JEDES Modul wird erfasst:
- Pfad, Zeilenanzahl, Risikostufe (RED/YELLOW/GREEN)
- Zweck (1 Satz)
- Abhaengigkeiten (imports)
- Aenderbare Parameter (mit Bereich und aktuellem Wert)
- Bekannte Probleme (aus AUDIT)
- NEVER-Regeln die gelten

### 3. parameters — Alle aenderbaren Einstellungen

Wird aus config/settings.json + Code extrahiert:

```json
{
  "parameters": [
    {
      "key": "thresholds.yolo_conf",
      "value": 0.5,
      "min": 0.1, "max": 0.9, "step": 0.05,
      "category": "vision",
      "effect": "Person Detection Empfindlichkeit",
      "side_effects": "Niedriger = mehr Fehlerkennungen, hoeher = uebersieht Personen",
      "changed_by": ["popup_npu_thresh.py", "self_tuner.py"],
      "consumed_by": ["tappas_pipeline.py", "inference_engine.py"]
    },
    {
      "key": "tts.length_scale",
      "value": 1.1,
      "min": 0.8, "max": 1.5, "step": 0.05,
      "category": "tts",
      "effect": "Sprechgeschwindigkeit (hoeher = langsamer)",
      "side_effects": "Zu schnell = unverstaendlich, zu langsam = nervend",
      "changed_by": ["voice_pipeline.py", "self_tuner.py"],
      "consumed_by": ["voice_pipeline.py", "console/moloch_console.py"]
    }
  ]
}
```

### 4. dependencies — Was haengt wovon ab

```json
{
  "dependencies": [
    {
      "from": "tappas_pipeline.py",
      "to": "hailo_manager.py",
      "type": "SHARED VDevice",
      "constraint": "NUR EIN VDevice gleichzeitig, group_id=SHARED"
    },
    {
      "from": "voice_pipeline.py",
      "to": "tts.py",
      "type": "function_call",
      "constraint": "TTS blockiert Audio-Output waehrend Sprachausgabe"
    },
    {
      "from": "spotify_controller.py",
      "to": "spotipy (extern)",
      "type": "API",
      "constraint": "Braucht Internet + gueltige Spotify Tokens"
    },
    {
      "from": "camera.py",
      "to": "Sonoff CAM-PT2",
      "type": "ONVIF",
      "constraint": "Nur EIN RTSP-Slot, kein Doppelzugriff"
    }
  ]
}
```

### 5. limits — Harte Grenzen

```json
{
  "limits": {
    "ram_max_mb": 3800,
    "ram_warning_mb": 3200,
    "ram_critical_mb": 3500,
    "cpu_temp_warning_c": 65,
    "cpu_temp_critical_c": 85,
    "npu_models_max_simultaneous": "alle (8GB reicht)",
    "npu_llm_blocks_vision": true,
    "rtsp_slots": 1,
    "vdevice_count": 1,
    "fps_target": 20,
    "fps_minimum": 10,
    "max_subprocess_timeout_s": 60,
    "max_file_changes_per_session": 5,
    "never_rules": [
      "NEVER 1: GStreamer-String nicht blind aendern",
      "NEVER 2: Pan-Vorzeichen in camera.py TABU",
      "NEVER 3: ArcFace-Threshold nicht als Quick-Fix",
      "NEVER 4: Nie mehrere ROT-Dateien in einem Commit",
      "NEVER 5: Immer timeout bei subprocess",
      "NEVER 6: Immer atomic JSON write",
      "NEVER 7: Runtime-State nicht committen",
      "NEVER 8: Kein shell=True",
      "NEVER 9: HailoRT uint8 vs float32 pruefen",
      "NEVER 10: Kein np.ndarray in moloch_service.py Signaturen",
      "NEVER 11: __pycache__ loeschen vor Restart",
      "NEVER 12: Nicht im Worktree testen"
    ]
  }
}
```

### 6. health_checks — Was MOLOCH bei sich pruefen kann

```json
{
  "health_checks": [
    {
      "id": "fps_check",
      "source": "/dev/shm/moloch_status.json",
      "field": "fps",
      "condition": ">= 10",
      "severity": "critical",
      "diagnosis": "FPS zu niedrig — Pipeline-Problem oder CPU-Ueberlast"
    },
    {
      "id": "ram_check",
      "source": "/proc/meminfo",
      "field": "MemAvailable",
      "condition": ">= 500 MB",
      "severity": "critical",
      "diagnosis": "RAM fast voll — Memory Leak oder zu viele Prozesse"
    },
    {
      "id": "cpu_temp",
      "source": "/sys/class/thermal/thermal_zone0/temp",
      "condition": "< 75000",
      "severity": "warning",
      "diagnosis": "CPU zu warm — Kuehlung pruefen"
    },
    {
      "id": "service_running",
      "source": "systemctl is-active moloch",
      "condition": "== active",
      "severity": "critical",
      "diagnosis": "Service gestoppt — Crash oder manueller Stop"
    },
    {
      "id": "camera_reachable",
      "source": "ping -c1 -W2 192.168.178.25",
      "condition": "returncode == 0",
      "severity": "critical",
      "diagnosis": "Kamera nicht erreichbar — Netzwerk oder Kamera-Neustart"
    },
    {
      "id": "npu_accessible",
      "source": "hailortcli fw-control identify",
      "condition": "returncode == 0",
      "severity": "critical",
      "diagnosis": "NPU nicht erreichbar — PCIe Problem oder Treiber"
    },
    {
      "id": "tracking_stable",
      "source": "/dev/shm/moloch_status.json",
      "field": "tracking_moves_per_minute",
      "condition": "< 60",
      "severity": "warning",
      "diagnosis": "Tracking zu hektisch — Dead Zone zu klein"
    },
    {
      "id": "face_recognition",
      "source": "/dev/shm/moloch_status.json",
      "field": "face_similarity",
      "condition": ">= 0.50 when face_detected",
      "severity": "warning",
      "diagnosis": "Gesichtserkennung unzuverlaessig — Threshold oder Enrollment pruefen"
    }
  ]
}
```

---

## GENERATOR-SCRIPT (generate_self_map.py)

Das Script das self_map.json erzeugt. Scannt automatisch:

```python
#!/usr/bin/env python3
"""
Generiert ~/moloch/config/self_map.json aus dem aktuellen Codestand.
Laeuft einmal am Tag (23:00) oder bei Aufruf.
"""

# Was es tut:
# 1. Zaehlt LOC pro .py Datei in core/
# 2. Liest Risiko-Stufe aus moloch-dev.md (ROT/GELB/GRUEN)
# 3. Extrahiert imports pro Modul
# 4. Liest aktuelle Parameter aus config/settings.json
# 5. Liest Hardware-Info aus /proc, /sys, vcgencmd
# 6. Prueft health_checks live
# 7. Schreibt alles als JSON (atomar!)

# Ergebnis: config/self_map.json (~50KB, vollstaendige Selbstbeschreibung)
```

### Aufruf-Moeglichkeiten:

```bash
# Manuell
python3 ~/moloch/scripts/generate_self_map.py

# Taeglich um 23:00 (cron oder systemd timer)
# 0 23 * * * python3 /home/molochzuhause/moloch/scripts/generate_self_map.py

# Via Sprache
# MOLOCH: "Aktualisiere dein Koerperwissen"
# → keyword_handler → generate_self_map.py

# Via HANDSHAKE (Claude Code fragt)
# Claude Code: "Zeig mir deine aktuelle self_map"
# → MOLOCH generiert neu und schickt sie mit
```

---

## WIE MOLOCH DIE MAP NUTZT

### Bei Selbstdiagnose (SELF-TUNE):

```python
# self_tuner.py
map = load_self_map()

# Finde den Parameter der geaendert werden soll
param = map.find_parameter("tts.length_scale")
if param:
    new_val = param["value"] - 0.05
    if new_val >= param["min"]:
        apply_change(param["key"], new_val)
```

### Bei HANDSHAKE mit Claude Code:

```python
# handshake_client.py
map = load_self_map()

request = {
    "symptom": "RAM bei 3.4GB",
    "self_map_excerpt": {
        "ram_max_mb": map["limits"]["ram_max_mb"],
        "top_ram_modules": map.get_top_ram_consumers(5),
        "known_issues": map.get_issues_by_category("memory"),
    }
}
send_handshake(request)
# Claude Code bekommt genug Kontext um gezielt zu diagnostizieren
```

### Bei Fragen vom User:

```
User: "Wie viel RAM hast du noch?"
MOLOCH: [liest self_map.json]
        "Ich habe 4 GB, davon sind 2.1 GB belegt. 
         Mein Service braucht 380 MB, der Rest ist System.
         Mein Limit ist 3.5 GB bevor es kritisch wird."
```

---

## ZUSAMMENFASSUNG: Die vier Dokumente

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  SELF-MAP   │     │  SELF-TUNE  │     │  HANDSHAKE  │     │  HOOKWIRE   │
│             │     │             │     │             │     │             │
│ "Wer bin    │ ──→ │ "Was stimmt │ ──→ │ "Hilf mir,  │ ──→ │ "Pruefe die │
│  ich?"      │     │  nicht?"    │     │  Claude"    │     │  Aenderung" │
│             │     │             │     │             │     │             │
│ Koerper-    │     │ Diagnose +  │     │ Arztbesuch  │     │ Qualitaets- │
│ wissen      │     │ Selbst-Fix  │     │ Protokoll   │     │ sicherung   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

MOLOCH kennt sich → erkennt Probleme → fixt einfaches selbst → fragt Claude Code bei komplexem → Hooks pruefen die Aenderung → Audit bestaetigt alles.
