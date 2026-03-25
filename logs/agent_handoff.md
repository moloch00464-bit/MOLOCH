# M.O.L.O.C.H. Übergabeprotokoll
**Datum:** 2026-03-25, 19:25
**Von:** Claude Opus 4.6 Session (Home-Fix + Tracker + Preview)
**Service-Status:** GESTOPPT (manuell, RAM-Analyse)
**USE_TAPPAS:** 1 (aktiv)

---

## Was wurde erledigt

### 1. Home-Position vereinheitlicht (4 Dateien)
EINE Quelle: `settings.json → ptz.home_pan / ptz.home_tilt`
- `popup_tracker.py`: Save-Pfad tracker.* → ptz.*, IPC sendet Pan/Tilt mit
- `moloch_service.py`: IPC-Handler liest Pan/Tilt aus Message, _save_settings(), Default tilt 0.0
- `camera.py`: goto_home() nutzt _home_position Dict + neue set_home_position() Methode
- `camera_manager.py` + `autonomous_tracker.py`: werden via Service korrekt versorgt
- **Datenkette:** GUI → settings.json ptz.* → Service → CameraManager + Camera + Tracker

### 2. Tracker-Stabilität (3 Bugs gefixt)
- **Orphan-Kill bei TAPPAS:** `camera_manager.py` — Bei USE_TAPPAS=1 wird Tracker NIE als orphaned gekillt. TAPPAS ist das Detektionssystem, Tracker muss permanent leben.
- **Coast-Bug:** `autonomous_tracker.py` Zeile 1143 — Coast-Aktivierung prüft jetzt `error_magnitude` statt `abs(error_y)`. Vorher fror Kamera auf -35.8° ein.
- **min_step_deg:** 2.0 → 0.5 — Residual-Korrektur funktioniert, Person bleibt zentriert.

### 3. Tracking-Punkt korrigiert
- `autonomous_tracker.py` Zeile 781 — Body-BBox Fallback: 15% → 8% von Oberkante. Bei großen BBoxen (nahe Person) lag der Punkt auf Schultern statt Kopf.
- **Ergebnis:** Error von 60-100px → 8px, Tilt von -35.8° → +13.8°

### 4. Preview-Performance (panel_preview.py)
- PhotoImage-Recycling via .paste() statt Neuerstellung pro Frame
- SHM File-Descriptor persistent offen halten
- Avatar-Intervall 110ms → 200ms (5fps statt 9fps)
- **Commit:** 6c08921

### 5. Service GUI-Unabhängigkeit: BESTÄTIGT
- Service = systemd headless, Panel = optionaler IPC-Client
- Panel-Close hat keinen Effekt auf Service

---

## Was ist OFFEN

### KRITISCH: RAM-Verbrauch / Memory Leak
- **Service verbraucht 3.2 GB RSS nach ~30s Laufzeit!** (Pi5 hat 4 GB)
- Nach 15s: 384 MB (normal), nach 30s: 3208 MB (LEAK!)
- System geht in Swap, wird unbenutzbar
- **Verdächtige Quelle:** TAPPAS/GStreamer Pipeline, möglicherweise Frame-Buffer-Leak
- **TODO:** `core/perception/tappas_pipeline.py` auf Buffer-Leaks prüfen
- **TODO:** Python-seitige Frame-Kopien suchen (numpy arrays die nie freigegeben werden)
- **TODO:** `tracemalloc` einschalten um Leak zu lokalisieren
- **TODO:** Prüfen ob alte Legacy-Module (InferenceEngine, HailoManager) bei USE_TAPPAS trotzdem importiert werden und RAM fressen

### Hintergrund-Services (unnötig?)
- InfluxDB: 46 MB — wird das von MOLOCH genutzt? Wenn nein: deaktivieren
- Docker: 49 MB — nur für Qdrant? Container prüfen
- pcmanfm Desktop: 29 MB — braucht man das?

### Tentakel-System (Smart Tracking ↔ MOLOCH)
- Aktuell: Bei TAPPAS ist Guardian-Mode AN, aber Orphan-Check ist deaktiviert
- Smart Tracking der Sonoff-Kamera wird NICHT genutzt (ST AUS)
- Markus' Wunsch: ST soll grob tracken, MOLOCH übernimmt bei Gesichtserkennung
- **TODO:** Tentakel-Flow mit TAPPAS integrieren (ST AN → Detection → Takeover → ST AUS → MOLOCH trackt → Release → ST AN)
- **ACHTUNG:** Zwei PTZ-Systeme gleichzeitig (ST + Tracker) = Konflikt!

### Coast-Schwellen (Tuning nach Test)
- coast_threshold: 50 → 40px
- coast_resume: 50 → 35px
- Könnte noch Feintuning brauchen nach Langzeit-Test

---

## Geänderte Dateien seit letztem stabilen Stand

```
core/gui/popups/popup_tracker.py   — Home Save-Pfad ptz.*
core/gui/panel_preview.py         — PhotoImage-Recycling, persistent FD
core/moloch_service.py            — IPC set_ptz_home, Defaults 0.0
core/hardware/camera.py           — goto_home() + set_home_position()
core/camera_manager.py            — Orphan-Kill TAPPAS-Guard
core/mpo/autonomous_tracker.py    — Coast-Bug, min_step_deg, track_y
config/settings.json              — ptz.home_tilt 0.0
```

---

## Wichtige Commits (chronologisch)

```
a13c3a9  BACKUP vor Home-Position Vereinheitlichung
caaab2f  BACKUP vor Tracker-Orphan-Fix
58670ab  Fix Tracker: Coast-Bug, Orphan-Kill, Tracking-Punkt
5d203d9  Fix Orphan-Kill: Bei TAPPAS Tracker nie als orphaned killen
225678b  BACKUP vor Preview-Performance Quick Wins
6c08921  Preview-Performance: PhotoImage-Recycling, persistent FD, Avatar 5fps
```

---

## Anweisungen für nächste Instanz

1. **Lies CLAUDE.md** (immer zuerst)
2. **Lies dieses Handoff** (logs/agent_handoff.md)
3. **KRITISCH:** RAM-Leak finden und fixen BEVOR irgendwas anderes
4. **Service ist GESTOPPT** — erst starten wenn RAM-Leak untersucht
5. **Agenten-Toolbox:** `MOLOCH_AGENT_TOOLBOX.json` für Domain-spezifische Agenten
6. **Regel 10 (Christian-Prinzip):** 1 Modul = 1 Aufgabe, Fail Isolation, Atomic Changes
