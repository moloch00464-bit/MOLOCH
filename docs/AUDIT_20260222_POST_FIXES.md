# M.O.L.O.C.H. System-Audit 22.02.2026 (Post-Fixes)

**Datum:** 2026-02-22, ~12:30 CET
**Auditor:** Claude Opus 4.6 (Audit-Instanz, read-only)
**Anlass:** Verifizierung aller heutigen Aenderungen

---

## 1. SERVICE STATUS

**Ergebnis:** ✅ OK

- Service: `active (running)` seit 12:22:29 CET
- PID: 5275, 37 Tasks, CPU: ~9 min
- Enabled: ja (autostart bei Boot)
- Echte Fehler in letzten 5 Minuten: **KEINE**
- Logs zeigen nur Tracker-Koordinaten (`error=(+63,+14)px` etc.) — das sind keine Fehlermeldungen

---

## 2. BOOT-FIX (Autonomous Mode nach Start)

**Ergebnis:** ✅ OK

- `_enable_autonomous()` wird in `start()` aufgerufen (Zeile 1910)
  - Log: `[START] Autonomous Mode aktiviert (Default nach Boot)`
- `_autonomous_enabled_at = time.time()` wird in `_enable_autonomous()` gesetzt (Zeile 1642)
- Grace Period 15s vorhanden:
  - `STARTUP_GRACE = 15` (Zeile 237)
  - `_takeover_cooldown_until = time.time() + self.STARTUP_GRACE` (Zeile 240)
  - Safety-Check in `_check_guardian_timeout()` (Zeile 1440):
    `(time.time() - getattr(self, "_autonomous_enabled_at", 0)) > 15`
    → Verhindert sofortiges Abschalten des Autonomen Modus nach Boot

---

## 3. LED STANDLICHT-LOGIK

**Ergebnis:** ✅ OK

### Neue Variablen (alle vorhanden):
| Variable | Zeile | Wert |
|----------|-------|------|
| `_led_markus_on` | 197 | `False` (init) |
| `_led_markus_last_seen` | 198 | `0` (init) |
| `_LED_MARKUS_TIMEOUT` | 199 | `30` Sekunden |
| `_led_markus_timer` | 200 | `None` (init) |

### Methoden (alle vorhanden):
- `_led_indicator_markus_seen()` — Zeile 1773: Setzt LED an, startet Timer
- `_led_markus_timeout()` — Zeile 1789: Schaltet LED aus nach Timeout
- Aufruf in Perception-Loop: Zeile 1060

### Alte Blink-Logik entfernt:
- `_led_blinking` — ✅ NICHT gefunden
- `_led_blink_lock` — ✅ NICHT gefunden
- `_led_markus_last_blink` — ✅ NICHT gefunden
- `_LED_BLINK_COOLDOWN` — ✅ NICHT gefunden
- `_led_indicator_blink_markus` — ✅ NICHT gefunden

→ Keine Reste der alten Blink-Logik vorhanden.

---

## 4. MANUAL MODE FORCE-SCHUTZ

**Ergebnis:** ✅ OK

### Auto-Switch (Zeile 1074):
```python
if not self._manual_mode:
    logger.info(f"[AUTO-SWITCH] ...")
    self._perception.force_models(None)
```
→ `force_models(None)` wird NUR im Nicht-Manual-Mode aufgerufen. ✅

### toggle_autonomous_manual():
- **MANUELL** (Zeile 2005): `self._perception.force_models(frozen)` → Modelle eingefroren ✅
- **AUTONOM** (Zeile 2026): `self._perception.force_models(None)` → Auto-Scoring freigegeben ✅

---

## 5. POPUP ALWAYS-ON-TOP + TRANSIENT

**Ergebnis:** ✅ OK (alle 5 Popups)

| Popup | topmost | transient | Zeilen |
|-------|---------|-----------|--------|
| popup_npu.py | ✅ `attributes('-topmost', True)` | ✅ `transient(parent)` | 72-73 |
| popup_audio.py | ✅ `attributes('-topmost', True)` | ✅ `transient(parent)` | 85-86 |
| popup_gallery.py | ✅ `attributes('-topmost', True)` | ✅ `transient(parent)` | 102-103 |
| popup_hardware.py | ✅ `attributes('-topmost', True)` | ✅ `transient(parent)` | 81-82 |
| popup_settings.py | ✅ `attributes('-topmost', True)` | ✅ `transient(parent)` | 71-72 |

---

## 6. GALERIE LOESCHEN OHNE BESTAETIGUNG

**Ergebnis:** ✅ OK

- `_delete_file()` in popup_gallery.py (Zeile 396): Docstring sagt "ohne Bestaetigung"
- Kein `messagebox.askyesno` oder `messagebox` Import gefunden
- Direktes `os.remove(path)` ohne Dialog

---

## 7. ERKANNT-BUTTON ALS STATUS-INDIKATOR

**Ergebnis:** ✅ OK

- `_btn_erkannt` hat `state="disabled"` (Zeile 103) → Kein klickbarer Button mehr
- `disabledforeground=FG_WHITE` (Zeile 104) → Sieht trotzdem lesbar aus
- `update_from_status()` (Zeile 238-243): Liest `status.get("led_markus_on", False)`
- Status-JSON (`/dev/shm/moloch_status.json`): `"led_markus_on": true` → Feld vorhanden und aktiv
- Service schreibt `led_markus_on` in Status (Zeile 2133)

---

## 8. REGEL 11 IN CLAUDE.md

**Ergebnis:** ✅ OK

- "REGEL 11 — DEPLOY & VERIFY" vorhanden ab Zeile 310
- Enthält: Restart-Befehl, Verify-Steps, Quick-Verify Einzeiler

---

## 9. SYNTAX-CHECKS

**Ergebnis:** ⚠️ TEILWEISE (2 von 4 Import-Namen falsch im Audit-Test)

| Import | Ergebnis | Details |
|--------|----------|---------|
| `from core.moloch_service import MolochService` | ✅ OK | Exit 0 |
| `from core.gui.panel_ewelink import EweLinkPanel` | ❌ ImportError | Klasse heisst `EwelinkModule`, nicht `EweLinkPanel` |
| `from core.gui.popups.popup_gallery import SnapshotGallery` | ✅ OK | Exit 0 |
| `from core.gui.popups.popup_npu import NPUPopup` | ❌ ImportError | Klasse heisst `NpuThreshPopup`, nicht `NPUPopup` |

**WICHTIG:** Das sind KEINE Code-Bugs! Die Dateien sind syntaktisch korrekt.
Die Import-Befehle im Audit-Test verwenden falsche Klassennamen.
Korrigierte Imports wuerden funktionieren:
- `from core.gui.panel_ewelink import EwelinkModule` → OK
- `from core.gui.popups.popup_npu import NpuThreshPopup` → OK

Der Service selbst laeuft fehlerfrei mit den richtigen Imports.

---

## 10. DAILY LEARNER SNAPSHOT-LOGIK

**Ergebnis:** ✅ OK — Snapshots werden NICHT blind gespeichert

### Implementierung: `core/daily_learner.py` (248 Zeilen)

### Trigger-Kette:
1. `moloch_service.py` Zeile 862: Aufruf NUR wenn:
   - `self._daily_learner` existiert UND
   - `self._daily_learner.enabled` ist True UND
   - `name != "Keine DB"` (ArcFace-DB muss geladen sein)
2. `maybe_snapshot()` prueft zusaetzlich:
   - `self.enabled` (nochmal) → Return wenn deaktiviert
   - **Rate Limit:** Max 1 Snapshot pro 60 Sekunden
   - **Name-Filter:** Ueberspringt `"unknown_maybe"`, `"Unbekannt"`, `"Keine DB"`
   - **Confidence-Filter:** Ueberspringt wenn `confidence <= 0.5`
   - **Bedingung muss NEU sein:** Kombination aus (Winkel, Licht, Distanz) darf
     nicht in der letzten Stunde (3600s) schon gesehen worden sein

### Modell-Abhaengigkeit:
- ✅ Snapshots werden NUR nach erfolgreicher ArcFace-Erkennung gespeichert
- ✅ Ohne laufendes SCRFD + ArcFace kommt der Code nie bis `maybe_snapshot()`
- ✅ ArcFace muss die Person mit Confidence > 0.5 erkannt haben
- ✅ Die Person muss einen Namen haben (kein "unknown_maybe", "Unbekannt", "Keine DB")

### Metadaten:
Jeder Snapshot wird mit vollstaendiger JSON-Metadatei gespeichert:
- `timestamp`, `time_str` (wann)
- `name` (wer: z.B. "markus")
- `confidence` (wie sicher: 0.0-1.0)
- `angle` (Kopfwinkel: 0=frontal, 1=leicht seitlich, 2=stark seitlich)
- `lighting` (Licht: 0=dunkel, 1=normal, 2=hell)
- `distance` (Entfernung: 0=fern, 1=mittel, 2=nah)
- `head_pose` (Pitch/Yaw/Roll wenn verfuegbar)

### Dateiname-Konvention:
`HH-MM-SS_name_cXX_aX_lX_dX.jpg` (z.B. `14-30-25_markus_c87_a0_l1_d1.jpg`)

### Speicherort:
`/mnt/moloch-data/daily/YYYY-MM-DD/` (auf Daten-SSD, nicht System-SSD) ✅

### Toggle:
Via IPC-Befehl `toggle_daily_learner` oder Panel

---

## ZUSAMMENFASSUNG

### Alles OK (9/10 Punkte):
1. ✅ Service laeuft stabil, keine Fehler
2. ✅ Boot-Fix: Autonomous Mode startet automatisch, Grace Period vorhanden
3. ✅ LED Standlicht: Neue Logik komplett, alte Blink-Reste entfernt
4. ✅ Manual Mode: force_models() korrekt geschuetzt
5. ✅ Alle 5 Popups: topmost + transient vorhanden
6. ✅ Galerie: Loeschen ohne Bestaetigung implementiert
7. ✅ ERKANNT-Button: disabled, liest led_markus_on aus Status-JSON
8. ✅ Regel 11: In CLAUDE.md vorhanden
9. ⚠️ Syntax-Checks: Code OK, Audit-Test-Imports verwenden falsche Klassennamen
10. ✅ Daily Learner: Intelligent, nicht blind, mit Modell-Verifikation und Metadaten

### Gefundene Probleme die gefixt werden muessen:

**KEINE CODE-BUGS GEFUNDEN.**

Einzige Anmerkung: Die Klassennamen in der CLAUDE.md Dokumentation
(`EweLinkPanel`, `NPUPopup`) stimmen nicht mit den tatsaechlichen
Klassennamen (`EwelinkModule`, `NpuThreshPopup`) ueberein. Das ist ein
Dokumentations-Thema, kein Code-Problem.

---

*Audit durchgefuehrt am 2026-02-22 um ~12:30 CET*
*Auditor: Claude Opus 4.6 (read-only, keine Code-Aenderungen)*
