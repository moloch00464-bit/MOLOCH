# TaoEngine Implementation Plan
## M.O.L.O.C.H. 5.0 ANIMA — Unterbewusstsein via Event Bus

### Context

MOLOCH hat ein 2-Achsen-Modell (Tension + Dominance) in `core/core_integrator.py` das auf
externe Reize reagiert (Person erkannt, Alarm, Markus da, etc.). Was fehlt: ein internes
"Unterbewusstsein" das den Systemzustand bewertet und subtile Stimmungs-Offsets erzeugt.

Die existierende `unconscious_engine.py` (398 LOC, Feature-Branch) ist zu gross, liest
direkt aus Dateien, und ist nicht in den Service integriert. Sie wird durch eine schlanke
TaoEngine ersetzt (max 150 LOC), die NUR ueber den Event Bus kommuniziert.

**Kritische Erkenntnis aus RECON:** Kein asyncio in der Codebase. Alles threading-basiert.
Der Spec forderte asyncio.create_task() — das ist FALSCH. TaoEngine wird ein Daemon-Thread.

---

### Dateien (Reihenfolge = Commit-Reihenfolge)

| # | Datei | Aktion | Zeilen-Delta |
|---|-------|--------|-------------|
| 1 | `config/settings.json` | Edit: `tao_engine` Sektion hinzufuegen | +4 |
| 2 | `core/unconscious_engine.py` | REPLACE: 398 LOC → ~140 LOC TaoEngine | -398, +140 |
| 3 | `core/core_integrator.py` | Edit: tao.tension_offset Consumer | +15 |
| 4 | `core/moloch_service.py` | Edit: TaoEngine Lifecycle (4 Zeilen) | +8 |
| 5 | `config/anima_mappings.json` | NEW: Behavior-Mapping Config | +30 |

---

### Schritt 1: Kill Switch in settings.json

**Datei:** `config/settings.json`
**Wo:** Nach `"teach"` Sektion (Zeile 88), vor `"internet"`

```json
"tao_engine": {
  "enabled": true,
  "tick_interval_ms": 500
},
```

**Warum zuerst:** Alles andere prueft dieses Flag. Ohne Flag kein sicheres Testen.

---

### Schritt 2: TaoEngine (core/unconscious_engine.py)

**REPLACE** die existierende 398-Zeilen-Datei komplett. Neue Klasse: `TaoEngine`.

#### Architektur:
```
TaoEngine(event_bus)
  ├── 4 State-Vars: yin=0.6, yang=0.4, wu_wei=0.5, ziran=0.5
  ├── Daemon-Thread, tick alle 500ms
  ├── Subscribes: perception.*, mood.changed, music.*, health_alert
  ├── Publishes: tao.state_update (PRIO_INFO=5), tao.tension_offset (PRIO_BRIDGE=3)
  └── Max 150 LOC
```

#### State-Variablen:
- `yin` (0.0-1.0, init 0.6): Ruhe, Innenschau, Zuhoeren
- `yang` (0.0-1.0, init 0.4): Aktion, Ausdruck, Reaktion
- `wu_wei` (0.0-1.0, init 0.5): Nicht-Handeln, Geduld
- `ziran` (0.0-1.0, init 0.5): Natuerlichkeit, Spontanitaet

#### 6 Evaluation Rules (Klassenkonstanten als Thresholds):
1. **silence_high**: Keine Person >60s → yin+0.01, yang-0.01
2. **music_mismatch**: Mood=dark + Musik=euphoric (oder umgekehrt) → wu_wei-0.01
3. **high_tension**: Tension >0.7 (aus letztem Event) → yang+0.01, wu_wei-0.01
4. **low_tension**: Tension <0.2 → yin+0.005, ziran+0.005
5. **system_stress**: health_alert empfangen → yang-0.01, wu_wei+0.01
6. **high_activity**: >3 person events in 10s → yang+0.01, ziran+0.01

#### Dynamics pro Tick:
1. Rules evaluieren → Deltas sammeln
2. **max_delta_per_tick = 0.02** (NICHT 0.12!)
3. Decay anwenden: `var *= (1.0 - 0.015)` (rate=0.015)
4. Memory: `var = var * 0.8 + prev_var * 0.2`
5. Noise: `var += random.uniform(-0.008, 0.008)`
6. Clamp auf [0.0, 1.0]

#### Derived Metrics:
- `balance = yin - yang` (-1.0 bis +1.0)
- `flow = (wu_wei + ziran) / 2.0`
- `stability = yin * wu_wei`
- `activity = yang * ziran`

#### Events:
- **Publish `tao.state_update`** (PRIO_INFO): alle 4 Vars + 4 Metrics, nur wenn Delta > 0.05
- **Publish `tao.tension_offset`** (PRIO_BRIDGE): `offset = abs(balance) * 0.05`, max ±0.02

#### Event-Daten sammeln (KEIN direkter Import):
- Subscribe `perception.person_detected` → Zaehler hochzaehlen
- Subscribe `perception.target_lost` → Silence-Timer starten
- Subscribe `mood.changed` → Aktuellen Mood merken
- Subscribe `music.playing` / `music.stopped` → Musik-State
- Subscribe `health_alert` → System-Stress Flag
- Tension-Wert: aus letztem `tao.state_update` oder CoreIntegrator-Status in /dev/shm

#### Wichtige Constraints:
- KEIN `import` von `moloch_service`, `core_integrator`, oder anderen Core-Modulen
- NUR `from core.moloch_event_bus import get_event_bus, PRIO_BRIDGE, PRIO_INFO`
- Thread-safe: Lock um State-Zugriff
- try/except um jeden Tick — darf NIEMALS crashen
- Logging nur bei State-Aenderung > 0.05, nicht bei jedem Tick

---

### Schritt 3: Tension-Offset Consumer in CoreIntegrator

**Datei:** `core/core_integrator.py`
**Wo:** In `__init__()` — Event Bus Subscribe hinzufuegen
**Wo:** In `_tick()` — Offset nach Vision-Update addieren

#### Aenderung 1: Subscribe in `__init__` (~Zeile 170):
```python
# TAO Engine Tension-Offset
self._tao_tension_offset = 0.0
try:
    from core.moloch_event_bus import get_event_bus
    get_event_bus().subscribe("tao.tension_offset", self._on_tao_offset, priority=5)
except Exception:
    pass
```

#### Aenderung 2: Callback:
```python
def _on_tao_offset(self, event):
    """TAO Engine Offset empfangen (max ±0.02)."""
    offset = event.get("payload", {}).get("offset", 0.0)
    self._tao_tension_offset = max(-0.02, min(0.02, float(offset)))
```

#### Aenderung 3: In `_tick()`, NACH Zeile 495 (nach allen Tension-Berechnungen):
```python
# TAO Engine Offset (subtiler Unterbewusstseins-Drift)
if abs(self._tao_tension_offset) > 0.001:
    self._tension = _clamp(self._tension + self._tao_tension_offset, lo=0.05, hi=0.95)
    _logger.debug(f"[TAO] Offset {self._tao_tension_offset:+.3f} → T={self._tension:.3f}")
    self._tao_tension_offset = 0.0  # Einmal anwenden, dann reset
```

**Clamp 0.05-0.95:** TAO darf Tension nie auf 0.0 oder 1.0 treiben.
**Reset nach Anwendung:** Verhindert Akkumulation.

---

### Schritt 4: TaoEngine in moloch_service.py einbinden

**Datei:** `core/moloch_service.py`

#### Aenderung 1: Import oben (nach bestehenden Imports):
Nicht noetig — lazy Import im `start()` Block, wie alle anderen Module.

#### Aenderung 2: In `start()` (~nach Zeile 1226, nach CoreIntegrator.start()):
```python
# TaoEngine (Unterbewusstsein) — optional via Kill Switch
try:
    from core.unconscious_engine import TaoEngine
    from core.moloch_event_bus import get_event_bus
    _tao_enabled = self._settings.get("tao_engine", {}).get("enabled", False)
    if _tao_enabled:
        self._tao_engine = TaoEngine(get_event_bus())
        self._tao_engine.start()
        logger.info("[START] TaoEngine gestartet (500ms Tick)")
    else:
        self._tao_engine = None
        logger.info("[START] TaoEngine deaktiviert (settings.json)")
except Exception as e:
    self._tao_engine = None
    logger.warning(f"[START] TaoEngine nicht verfuegbar: {e}")
```

#### Aenderung 3: In `stop()` (~nach Zeile 1808, nach CoreIntegrator.stop()):
```python
# TaoEngine stoppen
if getattr(self, '_tao_engine', None):
    self._tao_engine.stop()
```

**Total: 8 Zeilen in moloch_service.py.**

---

### Schritt 5: ANIMA Mappings Config

**Datei:** `config/anima_mappings.json` (NEU)

```json
{
  "_comment": "ANIMA: TAO State → Behavior Mappings (gelesen von Behavior Engine)",
  "visual_intensity": "activity",
  "movement_style": {
    "yin_high": "slow_fluid",
    "yang_high": "fast_sharp"
  },
  "presence": "stability",
  "variation": "flow",
  "decision_bias": {
    "yin_high": "reduce_activity",
    "yang_high": "increase_activity",
    "wu_wei_high": "delay_decision",
    "ziran_high": "allow_variation"
  }
}
```

**Reine Config-Datei.** Wird von bestehender Behavior Engine gelesen wenn sie dafuer
erweitert wird (separater Task, NICHT in dieser Implementation).

---

### Verification

1. **Smoke Test:** Service starten mit `tao_engine.enabled: true`
   - Kein Crash nach 5 Minuten
   - `grep "TaoEngine" ~/moloch/logs/moloch.log` zeigt Start-Meldung
   - `grep "tao.state_update" ~/moloch/logs/events/` zeigt Events

2. **Kill Switch Test:** `tao_engine.enabled: false` setzen, Service neustarten
   - Keine TaoEngine-Logzeilen
   - Kein tao.* Event im Bus

3. **RAM Test:** `ps aux | grep moloch` vor und nach TaoEngine
   - Delta < 5MB (4 Floats + 1 Thread)

4. **Tension-Drift Test:** 5 Min ohne Person
   - Tension-Offset max ±0.1 vom Ausgangswert
   - Kein Wegdriften auf 0.0 oder 1.0

5. **Regressionstest:** `python3 ~/moloch/moloch_audit.py --auto`
   - Alle bestehenden Tests PASS

---

### NICHT implementieren
- Kein asyncio (Codebase hat keins)
- Kein MCP-Protokoll (Event Bus reicht)
- Keine Environment-Inputs (temperature, light_level — Gate 9)
- Kein GUI-Element fuer TAO
- Keine Aenderung an Vision/GStreamer Pipeline
- Kein separater Prozess
