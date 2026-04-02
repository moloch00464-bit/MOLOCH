# SELF-TUNE — MOLOCH tuned sich selbst via Claude Code
# Codewort: SELF-TUNE
# Version: 1.0 | Stand: 2026-04-02
# Ziel: MOLOCH erkennt Probleme und fixt sie eigenstaendig

> Wenn Du dieses Dokument liest, wurde Dir gesagt: "SELF-TUNE".
> Implementiere die hier beschriebene Architektur.

---

## VISION

MOLOCH erkennt durch Selbstdiagnose ein Problem und beauftragt
Claude Code es zu fixen. Der Mensch muss nur noch bei kritischen
Aenderungen zustimmen.

```
User: "Du redest zu langsam"
  ↓
MOLOCH LLM versteht das Problem
  ↓
MOLOCH ruft MCP-Tool: self_tune("tts_speed", "schneller")
  ↓
Claude Code aendert length_scale 1.1 → 0.95 in settings.json
  ↓
MOLOCH reloaded Settings → redet wirklich schneller
  ↓
MOLOCH: "Besser so?"
```

---

## ARCHITEKTUR

### Zwei MCP-Server die sich gegenseitig kennen:

```
┌─────────────────────┐         ┌─────────────────────┐
│  MOLOCH (Pi)        │  MCP    │  Claude Code         │
│                     │ ←────── │                      │
│  moloch_mcp_server  │         │  (liest/schreibt     │
│  (schon vorhanden)  │         │   Code und Configs)  │
│                     │ ──────→ │                      │
│  moloch_tune_client │  MCP    │  tune_mcp_server     │
│  (NEU)              │         │  (NEU)               │
└─────────────────────┘         └─────────────────────┘

Pfad 1 (schon da): Claude Code → moloch_mcp_server → MOLOCH Status lesen
Pfad 2 (NEU):      MOLOCH → tune_mcp_server → Claude Code aendert Configs
Pfad 3 (NEU):      MOLOCH Selbstdiagnose → automatisch Pfad 2 triggern
```

---

## PFAD 2: tune_mcp_server (Claude Code Seite)

Neuer MCP-Server den Claude Code bereitstellt. MOLOCH kann ihn aufrufen
um Parameter-Aenderungen anzufragen.

### Tools die der Server anbietet:

```python
# tune_mcp_server.py — laeuft auf dem Pi, Claude Code als Backend

@tool("tune_parameter")
def tune_parameter(category: str, param: str, direction: str, reason: str):
    """
    Aendert einen MOLOCH-Parameter.
    
    category: "tts", "tracking", "vision", "personality", "audio", "fan"
    param:    spezifischer Parameter-Name
    direction: "increase", "decrease", oder konkreter Wert
    reason:   Warum die Aenderung noetig ist (fuer Log)
    
    Returns: {"success": bool, "old_value": X, "new_value": Y, "message": str}
    """

@tool("diagnose_and_fix")
def diagnose_and_fix(symptom: str):
    """
    MOLOCH beschreibt ein Symptom, Claude Code diagnostiziert und fixt.
    
    symptom: Freitext-Beschreibung des Problems
    
    Returns: {"diagnosis": str, "actions_taken": [...], "success": bool}
    """

@tool("get_tunable_params")
def get_tunable_params(category: str = None):
    """
    Listet alle aenderbaren Parameter mit aktuellem Wert und Bereich.
    
    Returns: [{"category": str, "param": str, "value": X, 
               "min": X, "max": X, "description": str}, ...]
    """
```

### Aenderbare Parameter (GRUEN — sicher, kein Reboot noetig):

```
KATEGORIE       PARAMETER              BEREICH         DEFAULT   SETTINGS-KEY
─────────────────────────────────────────────────────────────────────────────
tts             length_scale           0.8 – 1.5       1.1       tts.length_scale
tts             pitch_semitones        -4 – +4         0         tts.pitch_semitones
tts             sentence_silence       0.1 – 1.0       0.3       tts.sentence_silence
tts             voice_id               (8 Stimmen)     thorsten  tts.voice_id

audio           mic_gain               0 – 100         65        audio.mic_gain
audio           noise_gate_db          -60 – 0         -35       audio.noise_gate
audio           agc_enabled            true/false       true      audio.agc

tracking        dead_zone              0.05 – 0.40     0.15      tracking.dead_zone
tracking        tracking_speed         0.1 – 2.0       1.0       tracking.speed
tracking        pan_gain               0.1 – 3.0       1.0       tracking.pan_gain
tracking        tilt_gain              0.1 – 3.0       1.0       tracking.tilt_gain
tracking        coast_timeout          0.5 – 10.0      1.5       tracking.coast_timeout

vision          yolo_conf              0.1 – 0.9       0.5       thresholds.yolo_conf
vision          scrfd_conf             0.1 – 0.9       0.5       thresholds.scrfd_conf
vision          arcface_thresh         0.3 – 0.9       0.65      thresholds.arcface_thresh
vision          pose_conf              0.1 – 0.9       0.6       thresholds.pose_conf

personality     tension_tau            100 – 600       300       mpo.tension_tau
personality     dominance_drift        0.001 – 0.05    0.01      mpo.dominance_drift
personality     berserker_threshold    0.85 – 1.0      0.95      mpo.berserker_threshold

fan             noctua_base_pct        0.20 – 0.50     0.25      fan.base_pct
fan             noctua_ramp_start_c    38 – 50         42        fan.ramp_start
fan             noctua_full_pct_c      55 – 75         60        fan.full_temp

spotify         default_volume         10 – 100        45        spotify.volume
```

### Sicherheitsstufen:

```
GRUEN  — Sofort aenderbar, kein Risiko (alle oben genannten)
GELB   — Aenderbar mit Warnung (GStreamer Properties, Subprocess Timeouts)
ROT    — NUR mit User-Bestaetigung (HEF-Pfade, Service-Neustart, Pipeline-Rebuild)
```

---

## PFAD 3: Selbstdiagnose → Auto-Fix

MOLOCH fuehrt regelmaessig Selbstdiagnose durch (diagnostics.py, moloch_audit.py).
Bei Problemen ruft er automatisch tune_parameter() auf.

### Diagnose-Regeln (in diagnose_rules.json):

```json
{
  "rules": [
    {
      "id": "tts_too_slow",
      "trigger": "user_feedback",
      "keywords": ["zu langsam", "red schneller", "schneller reden", "tempo"],
      "action": {"category": "tts", "param": "length_scale", "direction": "decrease", "step": 0.05},
      "min": 0.8,
      "response": "Ich rede jetzt etwas schneller. Besser so?"
    },
    {
      "id": "tts_too_fast", 
      "trigger": "user_feedback",
      "keywords": ["zu schnell", "red langsamer", "langsamer reden", "nicht so schnell"],
      "action": {"category": "tts", "param": "length_scale", "direction": "increase", "step": 0.05},
      "max": 1.5,
      "response": "Ich rede jetzt langsamer. Besser?"
    },
    {
      "id": "tts_too_quiet",
      "trigger": "user_feedback",
      "keywords": ["zu leise", "lauter", "kann dich nicht hoeren", "hoer dich nicht"],
      "action": {"category": "audio", "param": "tts_volume", "direction": "increase", "step": 10},
      "max": 100,
      "response": "Ist das besser?"
    },
    {
      "id": "tts_too_loud",
      "trigger": "user_feedback",
      "keywords": ["zu laut", "leiser", "nicht so laut"],
      "action": {"category": "audio", "param": "tts_volume", "direction": "decrease", "step": 10},
      "min": 10,
      "response": "Besser?"
    },
    {
      "id": "voice_change",
      "trigger": "user_feedback",
      "keywords": ["andere stimme", "klingt komisch", "stimme aendern", "klingt doof"],
      "action": {"category": "tts", "param": "voice_id", "direction": "next"},
      "response": "Wie klingt diese Stimme?"
    },
    {
      "id": "tracking_jitter",
      "trigger": "self_diagnosis",
      "condition": "tracking_moves_per_minute > 60",
      "action": {"category": "tracking", "param": "dead_zone", "direction": "increase", "step": 0.02},
      "max": 0.40,
      "log": "Tracking zu hektisch — Dead Zone erhoeht"
    },
    {
      "id": "tracking_slow",
      "trigger": "user_feedback",
      "keywords": ["zu langsam", "tracking langsam", "kamera zu langsam", "schneller schwenken"],
      "action": {"category": "tracking", "param": "tracking_speed", "direction": "increase", "step": 0.1},
      "max": 2.0,
      "response": "Ich schwenke jetzt schneller."
    },
    {
      "id": "face_not_recognized",
      "trigger": "user_feedback",
      "keywords": ["erkennst mich nicht", "wer bin ich", "kennst mich nicht mehr"],
      "action": {"category": "vision", "param": "arcface_thresh", "direction": "decrease", "step": 0.05},
      "min": 0.30,
      "response": "Ich habe die Erkennungsschwelle gelockert. Schau nochmal in die Kamera."
    },
    {
      "id": "false_detections",
      "trigger": "self_diagnosis",
      "condition": "false_positive_rate > 0.3",
      "action": {"category": "vision", "param": "yolo_conf", "direction": "increase", "step": 0.05},
      "max": 0.9,
      "log": "Zu viele Fehlerkennungen — YOLO Confidence erhoeht"
    },
    {
      "id": "thermal_warning",
      "trigger": "self_diagnosis",
      "condition": "cpu_temp > 65",
      "action": {"category": "fan", "param": "noctua_base_pct", "direction": "increase", "step": 0.05},
      "max": 0.50,
      "log": "CPU zu warm — Noctua Grunddrehzahl erhoeht"
    },
    {
      "id": "music_too_loud",
      "trigger": "user_feedback",
      "keywords": ["musik leiser", "zu laut die musik", "musik runter"],
      "action": {"category": "spotify", "param": "default_volume", "direction": "decrease", "step": 10},
      "min": 10,
      "response": "Musik leiser."
    },
    {
      "id": "music_too_quiet",
      "trigger": "user_feedback",
      "keywords": ["musik lauter", "musik hoeher", "musik auf"],
      "action": {"category": "spotify", "param": "default_volume", "direction": "increase", "step": 10},
      "max": 100,
      "response": "Musik lauter."
    }
  ]
}
```

### Automatische Diagnose-Zyklen:

```
Alle 60 Sekunden:
  1. CPU Temp pruefen → zu warm? → Fan hochdrehen
  2. Tracking Restless pruefen → zu hektisch? → Dead Zone vergroessern
  3. FPS pruefen → zu niedrig? → Confidence erhoehen (weniger Detections)
  4. RAM pruefen → zu voll? → Warnung an User

Bei User-Feedback (Sprache/Text):
  1. Keywords matchen gegen diagnose_rules.json
  2. Parameter anpassen
  3. Bestaetigung sprechen
  4. Neuen Wert in settings.json speichern (atomar!)
```

---

## IMPLEMENTIERUNG — 4 Dateien

### 1. core/self_tuner.py (NEU)

```python
"""
MOLOCH Self-Tuner — Parameter-Aenderungen aus Diagnose-Regeln.
Liest diagnose_rules.json, matcht User-Feedback und Systemzustand,
aendert Parameter via IPC (set_* Actions).
"""

class SelfTuner:
    def __init__(self, service_proxy):
        self._service = service_proxy
        self._rules = self._load_rules()
        self._change_log = []  # Historie aller Aenderungen
    
    def on_user_feedback(self, text: str) -> Optional[str]:
        """Prueft ob User-Text eine Parameter-Aenderung triggert.
        Returns: Antwort-Text oder None."""
        for rule in self._rules:
            if rule["trigger"] != "user_feedback":
                continue
            for kw in rule.get("keywords", []):
                if kw in text.lower():
                    return self._apply_rule(rule)
        return None
    
    def on_diagnosis_cycle(self, status: dict) -> List[str]:
        """Prueft Systemzustand gegen Auto-Fix Regeln.
        Returns: Liste der durchgefuehrten Aenderungen."""
        actions = []
        for rule in self._rules:
            if rule["trigger"] != "self_diagnosis":
                continue
            if self._evaluate_condition(rule["condition"], status):
                result = self._apply_rule(rule)
                if result:
                    actions.append(result)
        return actions
    
    def _apply_rule(self, rule: dict) -> Optional[str]:
        """Wendet eine Regel an: Parameter aendern, speichern, loggen."""
        action = rule["action"]
        current = self._get_current_value(action["category"], action["param"])
        new_val = self._calculate_new_value(current, action, rule)
        
        if new_val == current:
            return None  # Nichts zu aendern (Limit erreicht)
        
        # IPC Command senden
        self._service._write_command("action", {
            "action": f"set_{action['category']}_param",
            "param": action["param"],
            "value": new_val,
        })
        
        # Atomar in settings.json speichern
        self._save_to_settings(action["category"], action["param"], new_val)
        
        # Loggen
        self._change_log.append({
            "ts": time.time(),
            "rule": rule["id"],
            "param": f"{action['category']}.{action['param']}",
            "old": current,
            "new": new_val,
            "reason": rule.get("log", rule.get("response", "")),
        })
        
        return rule.get("response", f"{action['param']} geaendert: {current} → {new_val}")
```

### 2. config/diagnose_rules.json (NEU)
Die Regeln von oben als JSON-Datei.

### 3. Erweiterung: core/keyword_handler.py
SelfTuner wird VOR dem LLM-API-Call aufgerufen:

```python
# In voice_pipeline.py, _process_text_inner():
# 1. Keywords (schon da)
# 2. NEU: Self-Tune Check
tune_response = self._self_tuner.on_user_feedback(text)
if tune_response:
    self._speak(tune_response)
    return
# 3. Spotify (schon da)
# 4. LLM API (schon da)
```

### 4. Erweiterung: core/moloch_service.py
Diagnose-Zyklus alle 60 Sekunden:

```python
# In _monitoring_loop():
if time.time() - self._last_tune_check > 60:
    status = self._build_status()
    actions = self._self_tuner.on_diagnosis_cycle(status)
    for a in actions:
        logger.info(f"[SELF-TUNE] {a}")
    self._last_tune_check = time.time()
```

---

## SICHERHEIT

### Limits (hart kodiert, nicht ueberschreitbar):
- Jeder Parameter hat min/max in der Regel-Definition
- Maximal 3 Aenderungen pro Parameter pro Stunde (Anti-Oszillation)
- Aenderungen werden geloggt in ~/moloch/logs/self_tune.log
- ROT-Parameter (Pipeline, HEF-Pfade) sind NICHT aenderbar

### Rollback:
- Jede Aenderung speichert den alten Wert
- User kann sagen "mach das rueckgaengig" → letzte Aenderung wird reverted
- Bei Service-Neustart werden nur settings.json-Werte geladen (kein Drift)

### Transparenz:
- MOLOCH sagt WAS er geaendert hat: "Ich habe die Sprechgeschwindigkeit von 1.1 auf 1.05 geaendert"
- User kann fragen "was hast du geaendert?" → zeigt Change-Log

---

## BEISPIEL-SZENARIEN

### Szenario 1: TTS zu langsam
```
User: "Du redest zu langsam"
MOLOCH: [matcht "zu langsam" → tts_too_slow Regel]
        [length_scale 1.1 → 1.05, speichert in settings.json]
        [spricht mit neuer Geschwindigkeit:]
        "Ich rede jetzt etwas schneller. Besser so?"
User: "Noch schneller"
MOLOCH: [matcht wieder → length_scale 1.05 → 1.00]
        "Und jetzt?"
User: "Perfekt"
MOLOCH: [kein Match → geht an LLM]
        "Gut, ich merke mir das."
```

### Szenario 2: Tracking ruckelt (automatisch)
```
[Diagnose-Zyklus: tracking_moves_per_minute = 78 > 60]
MOLOCH: [matcht tracking_jitter Regel]
        [dead_zone 0.15 → 0.17]
        [loggt: "Tracking zu hektisch — Dead Zone erhoeht"]
        [kein TTS — stille Korrektur]
[Naechster Zyklus: tracking_moves_per_minute = 42 < 60]
        [keine Aenderung noetig]
```

### Szenario 3: Gesichtserkennung versagt
```
User: "Du erkennst mich nicht mehr"
MOLOCH: [matcht face_not_recognized → arcface_thresh 0.65 → 0.60]
        "Ich habe die Erkennungsschwelle gelockert. Schau nochmal in die Kamera."
[Falls immer noch nicht erkannt:]
User: "Immer noch nicht"
MOLOCH: [matcht wieder → arcface_thresh 0.60 → 0.55]
        "Nochmal gelockert. Wenn das nicht hilft, machen wir ein neues Enrollment."
```

### Szenario 4: CPU zu warm (automatisch)
```
[Diagnose-Zyklus: cpu_temp = 68 > 65]
MOLOCH: [matcht thermal_warning]
        [noctua_base_pct 0.25 → 0.30]
        [loggt: "CPU zu warm — Noctua Grunddrehzahl erhoeht"]
[Falls User fragt "Was hast du geaendert?":]
MOLOCH: "Mir war zu warm. Ich habe den Luefter von 25% auf 30% Grunddrehzahl erhoeht."
```

### Szenario 5: Rueckgaengig machen
```
User: "Mach das rueckgaengig"
MOLOCH: [letzter Change-Log Eintrag: arcface_thresh 0.65 → 0.55]
        [reverted: arcface_thresh 0.55 → 0.65]
        "Rueckgaengig: Erkennungsschwelle wieder auf 0.65."
```

---

## INTEGRATION MIT CLAUDE CODE (Optional, Gate 3+)

Fuer komplexere Fixes die ueber Parameter-Aenderungen hinausgehen:

```
MOLOCH: "Mein Pose-Worker crashed bei bestimmten Frames"
  → Self-Tuner kann das NICHT fixen (Code-Problem, nicht Parameter)
  → MOLOCH schreibt Problem in ~/moloch/logs/self_tune_escalation.log
  → Naechste Claude Code Session liest das und fixt den Code
```

Oder direkt via MCP (wenn Claude Code lokal laeuft):

```
MOLOCH → MCP: diagnose_and_fix("Pose-Worker crashed bei kleinen BBoxen")
Claude Code → liest pose_worker.py → findet fehlenden Size-Check
Claude Code → editiert pose_worker.py → testet Syntax
Claude Code → antwortet: "Habe einen Min-Size Check eingefuegt"
MOLOCH → "Code-Fix angewendet. Teste..."
```

Das ist Gate 3+ Material — erfordert Claude Code lokal auf dem Pi.
