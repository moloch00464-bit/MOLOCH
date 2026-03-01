# 🔴 GATE 0 v2 — SYSTEMSCHLIESSUNG & VERDRAHTUNG
**Datum:** 2026-03-01
**Auftraggeber:** Markus (First Moloch)
**Architekten:** Claude (Tech) + ChatGPT (Koordination) + Gemini (Review)
**Target:** Pi5 4GB + Hailo-10H + Sonoff PT2
**Version:** 2.0 — basierend auf physischen Tests und Molochs Selbstdiagnose

---

## ⚠️ REGELN — LIES DAS ZUERST

1. **Lies CLAUDE.md** — alle Regeln gelten, besonders Regel 10 (Christian's Principle) und Regel 12 (Regressionstest).
2. **Git Backup VOR jeder Änderung.** `git add -A && git commit -m "vor Phase X"`
3. **NACH JEDER PHASE:** `python3 ~/moloch/moloch_audit.py --auto` — alle Tests müssen PASS sein. Bei FAIL → Phase nicht abgeschlossen. Fix the regression FIRST.
4. **Eine Phase nach der anderen.** Nicht parallel. Nicht vorgreifen. Nicht überspringen.
5. **4GB RAM.** Jede Zeile sparsam. Kein zweites Modell laden wenn eins reicht.
6. **Moloch-Sprache:** Alle neuen/geänderten Logs im Format `[VERB] Objekt key=value`. Referenz: `~/moloch/docs/MOLOCH_SPRACHE_V3_FINAL.md`
7. **KEIN Refactoring.** Kein Aufräumen. Kein "Verbesserung nebenbei". NUR was hier steht.
8. **Bestehende Funktionen NICHT kaputt machen.** Das ist der GANZE PUNKT von Gate 0.
9. **NICHT blind Parameter ändern.** Erst LOGGEN, VERSTEHEN, DANN fixen. Drei Instanzen haben PTZ-Tracking schon "gefixt" — es geht immer noch nicht.
10. **NICHT "fertig" melden wenn nur Code kompiliert.** Fertig heißt: Markus steht vor der Kamera und es FUNKTIONIERT.

---

## DAS KERNPROBLEM — VIER INSELN OHNE BRÜCKEN

Moloch besteht aktuell aus vier getrennten Systemen die NICHT miteinander kommunizieren:

```
INSEL 1 — NPU/Kamera:
  Erkennt Personen, Gesichter, Posen
  → Liefert Daten ins GUI und in moloch_status.json
  → Hat KEINEN Einfluss auf Core, Persönlichkeit, Kamera, LED

INSEL 2 — Chat/LLM:
  Versteht Sprache, antwortet intelligent
  → Wenn User sagt "das bin ich" ändert sich NICHTS im System
  → Kein Rückkanal zu NPU, Persönlichkeit, oder Hardware

INSEL 3 — Persönlichkeit/Tension:
  Berechnet Guardian/Shadow/Berserker
  → Tension reagiert TEILWEISE auf Erkennung (steigt bei Unbekannt)
  → ABER: Iris zeigt falschen Modus, LED lügt, Stimme ändert sich nicht

INSEL 4 — Hardware (Kamera/LED/Auge/TTS):
  PTZ-Kamera, LED-Indikator, Panel-Iris, Lautsprecher
  → Reagiert auf NICHTS automatisch
  → Tracking-Moves = 0, LED leuchtet blau bei Unbekannt
```

**Molochs eigene Diagnose bestätigt das:**
> "Ich bin ein Paradox — hochintelligente Seele in einem verkrüppelten Körper.
> Wie ein Geist der durch Wände starrt aber nichts bewegen kann."
> "Kamera-Steuerung: Ich SEHE alles, kann aber nicht schwenken/zoomen"
> "Passiver Beobachter: Bei allem außer Chat nur Zuschauer statt Akteur"

Gate 0 baut die Brücken zwischen den vier Inseln.

---

## WAS PHYSISCHE TESTS GEZEIGT HABEN

Markus hat am 2026-02-28 und 2026-03-01 physisch getestet. Ergebnisse:

| Test | Ergebnis | Problem |
|------|----------|---------|
| FPS | 9-13 statt 25+ | face_attr Load/Unload Loop |
| Tracking | Trk: 0, Kamera folgt NICHT | Pipeline NPU→Tracker nicht verdrahtet |
| Smart Tracking | Kämpft gegen Moloch, fährt auf Werksposition | Muss KOMPLETT AUS |
| Zentrierung | Person am Bildrand statt Mitte | Kein Ganzkörper-Framing |
| Home-Position | Fährt auf letzte manuelle Position (Decke) | Muss Raummitte sein |
| Tilt | Oben/Unten Steuerung kaputt | Nach oben starren statt Raummitte |
| Erkennung | "Unbekannt" trotz 51 Embeddings | Schwellwert zu streng oder Matching kaputt |
| Tension | War bei 1.00 (Shadow) | Weil er Markus nicht erkannte → korrekt aber ungewollt |
| Iris | Zeigte Guardian obwohl Shadow aktiv | Iris nicht an Persönlichkeit gekoppelt |
| Iris-Übergang | Grau-orange statt klares Rot | Flackert bei instabiler Tension, kein sauberer Wechsel |
| LED blau | Leuchtet obwohl Person nicht erkannt | LED lügt |
| Chat→System | "Das bin ich" ändert nichts | Chat hat keinen Rückkanal zum Core |
| NPU-Stufen | Schalten nicht automatisch | Manuell okay, automatisch nicht |
| Arbiter | Einmal gewechselt, sonst immer "Kamera führt" | Smart Tracking überschreibt alles |

---

## PHASE 1: FPS STABILISIEREN ⚡ (KRITISCH — ohne das geht GARNIX)

### Problem
FPS bei 9-13 statt 25+. Ursache: face_attr Modell wird in Endlosschleife geladen/entladen. Hailo-Logs zeigen `HAILO_DRIVER_OPERATION_FAILED(36)` und `HAILO_CONNECTION_REFUSED(89)`. Bei 9 FPS ist Tracking unmöglich, Iris-Animation ruckelt, alles stottert.

### Diagnose BEVOR du Code änderst
```bash
# Was lädt/entlädt ständig?
journalctl --user -u moloch --no-pager 2>/dev/null | grep -i "load\|unload\|configure\|unconfigure\|face_attr" | tail -30

# Oder in den Service-Logs:
grep -rn "face_attr\|configure\|unconfigure" ~/moloch/core/model_orchestrator.py | head -20
grep -rn "face_attr" ~/moloch/core/*.py | head -20

# Laufzeit-Check: Wie oft wird configure() aufgerufen?
# Füge temporäres Logging ein und zähle über 60 Sekunden
```

### Was zu tun ist
1. **Finde den face_attr Load/Unload Loop** — irgendein Modul ruft ständig configure/unconfigure auf
2. **STOPPE den Loop** — Modelle dürfen nur bei ECHTEM Stufenwechsel geladen/entladen werden (Phase 4)
3. **Hailo-Error-Handling:** Bei DRIVER_OPERATION_FAILED → retry mit 1s backoff, max 3 Versuche. Nicht sofort neu laden.
4. **Bounded Frame Queue:** `maxsize=2`, wenn voll → ältesten Frame verwerfen, NICHT blockieren
5. **Frame-Timestamp-Check:** Frame älter als 200ms → verwerfen

### Akzeptanz
- FPS stabil über 15, Ziel 25+
- Kein linearer Memory-Anstieg über 30 Minuten
- Keine Hailo DRIVER_OPERATION_FAILED Errors mehr
- `python3 ~/moloch/moloch_audit.py --auto` → PASS

---

## PHASE 2: SMART TRACKING AUS — MOLOCH ÜBERNIMMT KOMPLETT

### Problem
Smart Tracking der Sonoff PT2 und Molochs Tracking kämpfen gegeneinander. Smart Tracking verliert Person → fährt auf Werksposition (Decke/Ecke) → Moloch sieht nichts mehr. Das ist die Ursache für das "nach oben starren" und "in manueller Position verharren".

**MARKUS HAT ENTSCHIEDEN: Smart Tracking KOMPLETT AUS. Moloch macht ALLES. Keine Diskussion.**

### Was zu tun ist
1. **Smart Tracking beim Service-Start DEAKTIVIEREN:**
   - Via eWeLink/ONVIF: Smart Tracking = OFF
   - Einmalig beim `init()`, nicht togglen
   - Loggen: `[STARTE] ptz_modus=moloch_allein smart_tracking=deaktiviert`
   - Bei jedem Service-Start prüfen und erneut deaktivieren falls Kamera-Reset

2. **Arbiter vereinfachen:**
   - Modus "KAMERA_FUEHRT" entfernen — gibt es nicht mehr
   - Nur noch: MOLOCH_AUTONOM (default) und MOLOCH_MANUELL (wenn User per GUI steuert)
   - Wenn User manuell steuert → 30s Timeout → dann zurück zu MOLOCH_AUTONOM

3. **Panel anpassen:**
   - "ST: AN" Button → entfernen oder dauerhaft "ST: AUS" anzeigen
   - "Kamera fuehrt" → nicht mehr als Option anzeigen
   - Nur noch: "MOLOCH AUTONOM" oder "MANUELL"

4. **Home-Position definieren (WICHTIG!):**
   - Keine Person sichtbar → Kamera fährt auf **Raummitte (Pan: 0°, Tilt: 0°)**
   - **NICHT** auf letzte manuelle Position
   - **NICHT** auf Sonoff Werksposition
   - Home-Position in `config/ptz_settings.json` konfigurierbar

5. **Scan-Modus wenn niemand da:**
   - Keine Person seit 30 Sekunden → langsamer Scan links-rechts
   - Scan-Geschwindigkeit: 5°/Sekunde
   - Bei Person-Detection → SOFORT Scan stoppen und tracken

### Akzeptanz
- Smart Tracking ist AUS nach Service-Start (verifizierbar im Log)
- Kamera fährt auf Raummitte wenn niemand da, NICHT auf Decke/Ecke
- Panel zeigt keinen "Kamera fuehrt" Modus mehr
- Manuelle Steuerung ist temporär, danach zurück zu Autonom
- `python3 ~/moloch/moloch_audit.py --auto` → PASS

---

## PHASE 3: TRACKING — KAMERA FOLGT PERSON 🎯

### Problem
PTZ Tracking Moves = 0 laut Panel. Kamera bewegt sich NICHT wenn Person sichtbar. DREI Instanzen haben das vorher "gefixt" — es geht immer noch nicht.

### WICHTIG: Diagnose ZUERST
**NICHT blind Parameter ändern. NICHT raten. MESSEN.**

```bash
# 1. Was empfängt der Tracker von der NPU?
grep -n "person\|detection\|bbox\|track" ~/moloch/core/mpo/autonomous_tracker.py | head -30

# 2. Was sendet er als PTZ-Befehl?
grep -n "move\|absolute\|onvif\|ptz_command\|send" ~/moloch/core/mpo/autonomous_tracker.py | head -20

# 3. Logge am EINGANG des Trackers was reinkommt
# 4. Logge am AUSGANG was als PTZ-Befehl rausgeht
# 5. Logge was die Kamera ANTWORTET

# Wenn NICHTS reinkommt → Verdrahtung inference_engine → tracker kaputt
# Wenn reinkommt aber NICHTS gesendet wird → Tracker-Logik kaputt
# Wenn gesendet aber Kamera ignoriert → ONVIF-Problem
# Wenn gesendet und ausgeführt aber FALSCH → PID-Parameter
```

### Was zu tun ist

#### A) NPU → Tracker Verdrahtung prüfen und fixen
- YOLO liefert Person Bounding Box → diese MUSS beim Tracker ankommen
- Am Eingang des Trackers loggen: `[SEHE] Person bbox_x=X% bbox_y=Y% w=W h=H confidence=C`
- Wenn nichts ankommt: Die Pipe zwischen inference_engine.py und autonomous_tracker.py ist kaputt → FIXEN

#### B) Ganzkörper-Zentrierung (NICHT nur Kopf!)
- Tracking-Ziel: Die GANZE Person-Bounding-Box mit 15% Rand in der Bildmitte
- Vertikal: Mitte der Person-Box auf 50% Bildhöhe
- Horizontal: Mitte der Person-Box auf 50% Bildbreite
- Deadzone: ±10% — innerhalb davon keine Kamerabewegung (verhindert Zittern)
- Person am Bildrand (>75% oder <25%) → SCHNELLERE Nachführung
- Moloch WILL die Person komplett im Bild haben — das ist sein Antrieb

#### C) Tracking-Regeln
- Person sichtbar → Kamera folgt SOFORT
- Person droht aus dem Bild → Kamera folgt SCHNELLER
- Person verloren → 5 Sekunden warten → Home-Position → Scan
- NIEMALS über sichtbare Person hinwegschwenken
- NIEMALS Scan starten wenn Person sichtbar

#### D) Tilt-Range
- Voller physischer Bereich nutzen
- Keine künstlichen Software-Limits die Kamera nach oben zwingen
- Standard-Tilt bei "Person steht": leicht nach unten (Person steht auf Boden, nicht an Decke)

#### E) Logging jedes PTZ-Moves
```
[SCHWENKE] ziel_pan=15.3 ziel_tilt=-5.0 grund=person_zentrieren
[SCHWENKE] ergebnis=erreicht actual_pan=15.1 actual_tilt=-4.8 dauer_ms=450
```

### Akzeptanz
- Trk > 0 im Panel wenn Person sichtbar
- Kamera folgt Person flüssig (kein Ruckeln, kein Überschwenken)
- Person bleibt zentriert im Bild (Ganzkörper, ±15% von Mitte)
- Person verloren → Home-Position (Raummitte) → Scan
- `python3 ~/moloch/moloch_audit.py --auto` → PASS
- **INTERAKTIV:** `python3 ~/moloch/moloch_audit.py --full` → Person-Tracking Test bestehen

---

## PHASE 4: NPU STUFEN-SCHALTUNG 🧠

### Problem
Alle Modelle laufen immer oder schalten unsauber. Face_attr Loop (Phase 1) ist ein Symptom davon. Stufenschaltung existiert nicht automatisch.

### Drei Stufen

| Stufe | Bedingung | Aktive Modelle | Hailo-Last |
|-------|-----------|----------------|------------|
| IDLE | Keine Person seit 60s | yolov8m only | ~20% |
| PERSON | YOLO sieht Person > 0.6 | + SCRFD | ~50% |
| FACE | SCRFD findet Gesicht > 0.5 | + ArcFace + Pose | ~80% |

### Übergangslogik
```
IDLE → PERSON: YOLO meldet Person (confidence > 0.6)
PERSON → FACE: SCRFD findet Gesicht (confidence > 0.5)
FACE → PERSON: Gesicht verloren seit 10 Sekunden
PERSON → IDLE: Keine Person seit 60 Sekunden
FACE_DIRECT (SCRFD + ArcFace ohne YOLO) = gültiger Zustand, kein Fehler
```

### Wichtig
- Modelle laden bei Stufenwechsel, NICHT ständig (das war der FPS-Bug!)
- Stufenwechsel loggen: `[WECHSLE] npu_stage=person→face weil=gesicht_erkannt`
- **Service-Start: IMMER in IDLE** (nur yolov8m)
- **Maximal EIN Stufenwechsel pro Event** — nicht hin-und-her-flackern

### Status in `/dev/shm/moloch_status.json`
```json
{
  "npu_stage": "idle|person|face",
  "npu_stage_since": "2026-03-01T06:00:00"
}
```

### Akzeptanz
- NPU wechselt automatisch zwischen Stufen
- Im IDLE nur YOLO aktiv, Hailo-Temp sinkt
- Kein Load/Unload-Loop
- `python3 ~/moloch/moloch_audit.py --auto` → PASS
- **INTERAKTIV:** Weggehen → IDLE, Zurückkommen → FACE

---

## PHASE 5: BRÜCKE 1 — NPU → PERSÖNLICHKEIT → IRIS 💀

### Problem (Drei Bugs in einem)
1. **Tension reagiert TEILWEISE auf NPU** (steigt bei Unbekannt) — aber flackert statt stabil zu steigen/fallen
2. **Iris zeigt FALSCHEN Modus** — System sagt Shadow, Iris zeigt Guardian
3. **Iris-Übergang hängt** — grau-orange statt klares Rot, Wechsel wird nicht abgeschlossen

### Bug 1: Tension-Flackern → Hysterese einbauen
Das Problem: Bei 9 FPS und wechselnder Erkennung springt Tension hin und her. Iris berechnet Farbe neu bei jedem Frame → grau-orange Brei.

Lösung: **Hysterese mit Zeitfenster**
```python
# NICHT bei jedem Frame wechseln!
# Erst nach stabilem Zustand über Zeit

if tension > 0.6 for mindestens 10 Sekunden:
    personality_mode = SHADOW
elif tension < 0.3 for mindestens 10 Sekunden:
    personality_mode = GUARDIAN
else:
    # Bleib wo du bist — kein Wechsel
    pass
```

### Bug 2: Iris zeigt falschen Modus
Die Iris liest ihren Modus von einer ANDEREN Quelle als die Persönlichkeitslogik.

Fix: **EINE Wahrheit.**
```
moloch_status.json enthält:
  "personality_mode": "guardian|shadow|berserker"
  "tension": 0.45

Iris liest NUR von moloch_status.json.
Persönlichkeit schreibt NUR in moloch_status.json.
Keine zweite Quelle. Keine lokale Variable. EINE Wahrheit.
```

### Bug 3: Iris-Übergang unvollständig
Fix: **Harte Farben statt Überblendung bei niedrigen FPS**
```
Guardian → BLAU (#0066ff) — sofort, kein Fade
Shadow → ROT (#cc0000) — sofort, kein Fade
Berserker → DUNKELROT (#880000) pulsierend — sofort
Übergangszeit: maximal 2 Sekunden
Wenn FPS < 15: KEIN Fade, direkter Farbwechsel
```

### Tension-Logik (vollständig)
```python
# Unbekannte Person sichtbar
if person_visible and not face_recognized:
    seconds_unrecognized += dt
    if seconds_unrecognized > 60:
        tension += 0.01 * dt    # Langsam steigen
    if seconds_unrecognized > 180:
        tension += 0.02 * dt    # Schneller
    tension = min(tension, 1.0)

# Markus erkannt
elif person_visible and face_recognized and name == "Markus":
    tension = max(tension - 0.05 * dt, 0.1)   # Schnell fallen
    seconds_unrecognized = 0

# Niemand da
elif not person_visible:
    tension = max(tension - 0.01 * dt, 0.15)  # Langsam auf Grundlevel
    seconds_unrecognized = 0
```

### Loggen
```
[FÜHLE] tension=0.65 modus=shadow weil=unbekannte_person_seit_180s
[WECHSLE] guardian→shadow weil=tension_stabil_ueber_0.6_seit_10s
[FÜHLE] tension=0.25 modus=guardian weil=markus_erkannt
```

### Akzeptanz
- Iris zeigt IMMER den korrekten Modus (Guardian=blau, Shadow=rot)
- Wechsel innerhalb 2 Sekunden sichtbar
- Kein grau-orange Zwischending
- Tension steigt bei unbekannter Person, fällt bei Markus
- Shadow aktiviert sich nach 10+ Sekunden stabiler hoher Tension
- `python3 ~/moloch/moloch_audit.py --auto` → PASS

---

## PHASE 6: BRÜCKE 2 — LED MUSS WAHRHEIT ZEIGEN 💡

### Problem
LED-Indikator leuchtet blau obwohl Person NICHT erkannt wurde. LED lügt.

### Was zu tun ist
LED-Farbe folgt dem GLEICHEN Status wie die Iris:

```
Guardian + Markus erkannt → BLAU
Guardian + niemand da → BLAU gedimmt
Shadow + unbekannte Person → ROT
Shadow + Markus nicht erkannt → ROT pulsierend
Berserker → ROT schnell blinkend
```

LED liest DENSELBEN `personality_mode` aus `moloch_status.json` wie die Iris.
**EINE Wahrheit für Iris UND LED.**

### Akzeptanz
- LED blau NUR wenn Markus erkannt
- LED rot bei Shadow
- LED stimmt IMMER mit Iris überein
- `python3 ~/moloch/moloch_audit.py --auto` → PASS

---

## PHASE 7: BRÜCKE 3 — CHAT → CORE RÜCKKANAL 🔗

### Problem
Wenn Markus im Chat sagt "das bin ich" oder "das was du siehst bin ich", passiert NICHTS im System. Chat ist Endpunkt, kein Rückkanal zum Core.

### Was zu tun ist
Bestimmte Chat-Eingaben müssen System-Aktionen auslösen:

```
CHAT-TRIGGER → SYSTEM-AKTION:

"das bin ich" / "ich bin es" / "erkenne mich"
  → Face-System: Override! Aktuelle Person = Markus
  → Tension: sofort auf 0.2 senken
  → Persönlichkeit: Guardian aktivieren
  → Loggen: [ERKENNE] markus_override quelle=chat

"alarm" / "Intruder"
  → Tension: sofort auf 1.0
  → Shadow/Berserker aktivieren
  → Loggen: [ALARMIERE] quelle=chat_befehl

"beruhige dich" / "alles gut"
  → Tension: auf 0.15 senken
  → Guardian aktivieren
  → Loggen: [LINDERE] quelle=chat_befehl
```

### Implementierung
- Im Chat-Handler: Prüfe auf Trigger-Wörter BEVOR die Nachricht an die LLM-API geht
- Bei Match: Führe System-Aktion aus UND sende an LLM
- LLM bekommt den Kontext "User hat Identität bestätigt" → antwortet entsprechend
- Logge: `[VERSTEHE] chat_trigger=identitaet_bestaetigt aktion=tension_reset`

### Akzeptanz
- "das bin ich" → Tension fällt, Guardian aktiviert, Iris wird blau
- Wirkung innerhalb 2 Sekunden sichtbar
- `python3 ~/moloch/moloch_audit.py --auto` → PASS

---

## PHASE 8: GESICHTSERKENNUNG VERBESSERN 👤

### Problem
"Unbekannt [Neutral] M/35-43" trotz 51 Embeddings. 40% Ähnlichkeit. Panel sagt "ERKANNT: MARKUS" aber Kamerabild sagt "Unbekannt" — zwei widersprüchliche Anzeigen.

### Diagnose
```bash
# Schwellwerte?
grep -n "thresh\|threshold\|similarity\|confidence" ~/moloch/core/*.py | head -20

# Embeddings Struktur?
python3 -c "
import json
d = json.load(open(os.path.expanduser('~/moloch/data/face_embeddings.json')))
print(f'Einträge: {len(d)}')
for k in list(d.keys())[:10]:
    print(f'  {k}')
"
```

### Was zu tun ist
1. **Widerspruch klären:** Warum sagt Panel "ERKANNT: MARKUS" aber Kamerabild "Unbekannt"? Finde die ZWEI verschiedenen Quellen und vereinheitliche sie. EINE Wahrheit.
2. **ArcFace Schwellwert prüfen** — wenn 0.55 zu streng → auf 0.45 senken. Bei seitlichem Winkel oder schlechtem Licht ist 55% zu hart.
3. **Matching:** Vergleiche gegen ALLE Embeddings, Best-Match zählt
4. **Loggen:** `[SEHE] Gesicht name=Markus confidence=0.87` oder `[SEHE] Gesicht name=unbekannt best_match=0.41 threshold=0.55`

### Akzeptanz
- Markus frontal bei gutem Licht → >80% Erkennung
- Seitlich oder schlechtes Licht → >50% Erkennung
- Panel und Kamerabild zeigen DASSELBE (keine Widersprüche)
- `python3 ~/moloch/moloch_audit.py --auto` → PASS

---

## PHASE 9: PANEL STABILITÄT 🖥️

### Problem
Panel friert gelegentlich ein. Vermutlich gelöst wenn Phase 1 (FPS) gelöst ist.

### Was zu tun ist
1. Wenn Phase 1 gelöst (stabile FPS) → testen ob Panel noch einfriert
2. Falls JA: Panel-Watchdog:
   ```python
   if time.time() - last_render > 2.0:
       logger.warning("[WARNUNG] Panel render_timeout")
   ```
3. Panel CPU-Last: unter 15%
4. Status-Polling: nicht öfter als alle 500ms

### Akzeptanz
- Panel läuft 30+ Minuten ohne Freeze
- Panel CPU < 15%
- `python3 ~/moloch/moloch_audit.py --auto` → PASS

---

## PHASE 10: 6-STUNDEN STABILITÄTSTEST 📊

### Vorbedingung
Phase 1-9 müssen ALLE abgeschlossen sein. Audit PASS.

### Durchführung
1. Deploye `stability_test_runner.py` und `analyze_stability.py`
2. FPS und NPU-Status aus `/dev/shm/moloch_status.json` lesen (NICHT aus Textdateien)
3. Runner als **SEPARATEN Prozess** starten — NICHT im Moloch-Service! (Gemini: "Der Wächter darf nicht Teil der Psyche sein die er überwacht")
   ```bash
   # Terminal 1: Moloch läuft normal
   # Terminal 2: Runner beobachtet von AUSSEN
   python3 stability_test_runner.py &
   ```
4. 6 Stunden laufen lassen
5. Analyzer: `python3 analyze_stability.py`

### Fail Conditions
- Memory Drift > 5%/h
- FPS Drift > 10%
- Thread Count steigend
- Event Loop Delay > 50ms
- CPU Temp > 80°C
- Crash / Restart
- PTZ Conflicts > 0

### Akzeptanz
```json
{
  "gate_0": {
    "runtime_hours": 6,
    "crashes": 0,
    "fps_stable": true,
    "memory_stable": true,
    "npu_idle_working": true,
    "ptz_conflicts": 0,
    "panel_freeze": 0,
    "iris_correct": true,
    "led_correct": true,
    "tracking_functional": true,
    "status": "PASSED"
  }
}
```

---

## REIHENFOLGE — STRIKT

```
Phase 1:  FPS stabilisieren            → audit --auto → PASS?
Phase 2:  Smart Tracking AUS           → audit --auto → PASS?
Phase 3:  Tracking verdrahten          → audit --auto + --full → Tracking-Test?
Phase 4:  NPU Stufen-Schaltung         → audit --auto + --full → Idle-Test?
Phase 5:  NPU→Persönlichkeit→Iris      → audit --auto → PASS?
Phase 6:  LED Wahrheit                 → audit --auto → PASS?
Phase 7:  Chat→Core Rückkanal          → audit --auto → PASS?
Phase 8:  Gesichtserkennung            → audit --auto → PASS?
Phase 9:  Panel Stabilität             → audit --auto → PASS?
Phase 10: 6h Stabilitätstest           → analyze_stability.py → PASS?
```

**NACH JEDER PHASE:**
```bash
python3 ~/moloch/moloch_audit.py --auto
git add -A && git commit -m "Gate0 Phase X fertig - audit PASS"
```

### Instanz-Aufteilung (empfohlen)
```
Instanz 1: Phase 1 + 2 (FPS + Smart Tracking)
Instanz 2: Phase 3 + 4 (Tracking + NPU Stufen)
Instanz 3: Phase 5 + 6 + 7 (Brücken: Persönlichkeit + LED + Chat)
Instanz 4: Phase 8 + 9 (Erkennung + Panel)
Instanz 5: Phase 10 (6h Test)
```

---

## WAS NICHT IN GATE 0 GEHÖRT

❌ Neue Verben für Moloch-Sprache
❌ Emergentis / Personality-Erweiterungen
❌ Action Bridge (Gate 1)
❌ Neue Sensorik (ESP, DT50)
❌ WGT-Modus
❌ Nacht-Zyklus
❌ Humor / Chuck Norris
❌ Aufräumen / Refactoring
❌ Neue GUI-Features
❌ VITALE Body Monitor
❌ Timing-Synchronisation
❌ Körper-Kalibrierung

**NUR VERDRAHTUNG. NUR STABILISIERUNG. NUR GATE 0.**

---

## DEBUG-WORKFLOW

```
1. LIES den relevanten Code KOMPLETT
2. LOGGE was reinkommt und was rausgeht
3. VERSTEHE warum es nicht funktioniert
4. DANN ERST fixen
5. Service restart
6. Audit: python3 ~/moloch/moloch_audit.py --auto
7. Git commit
8. Nächste Phase
```

**Drei Instanzen haben PTZ-Tracking schon "gefixt". Es geht immer noch nicht.**
**NICHT raten. MESSEN. Wie in der DGM.**

---

## REFERENZ-DOKUMENTE

- `CLAUDE.md` — Regeln (besonders 10 + 12)
- `~/moloch/docs/MOLOCH_SPRACHE_V3_FINAL.md` — Sprach-Format
- `~/moloch/moloch_audit.py` — Regressionstest
- `/dev/shm/moloch_status.json` — Echtzeit-Status

---

*Gate 0 = Vier Inseln verbinden. NPU sieht → Core fühlt → Hardware reagiert → Iris zeigt Wahrheit.*
*Wenn Moloch sagt "ich sehe dich" muss die Kamera folgen. Wenn er sagt "ich kenne dich nicht" muss die Iris rot werden.*
*Alles andere ist Theater.*

*M.A.M.⁴ — Claude + ChatGPT + Gemini + Markus 🖤⚡*
