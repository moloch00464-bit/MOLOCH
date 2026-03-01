# M.O.L.O.C.H. 4.0 — SYSTEMKONTEXT FÜR CLAUDE CODE
**Lies das KOMPLETT bevor du irgendetwas änderst.**

---

## WAS IST M.O.L.O.C.H.?

Autonomes KI-System auf Raspberry Pi 5 (4GB RAM!) mit Hailo-10H NPU.
Gebaut von Markus (47, Industrieautomatisierung, kein CS-Studium) mit drei AIs:
- Claude (Tech/Architektur)
- ChatGPT (Koordination/Bremse)
- Gemini (Seele/Review)

### Hardware
- Pi5 4GB RAM, 2x NVMe SSD
- Hailo-10H NPU (HAT+) — NUR EIN VDevice erlaubt!
- Sonoff PT2 PTZ-Kamera (IP: 192.168.178.25)
- OLED Display, LED-Indikator, Lautsprecher
- Piper TTS (8 deutsche Stimmen)

### Software-Architektur
- `moloch_service.py` — Hauptservice
- `core/` — Kernmodule (inference, tracker, sprache, etc.)
- `panel/` — GUI (Tkinter, Panel mit Iris-Auge)
- `config/` — Konfigurationsdateien
- `/dev/shm/moloch_status.json` — Echtzeit-Status (shared memory)
- Qdrant Vektor-DB auf Port 6333

### Persönlichkeits-System
- **Guardian** (Wächter) — blau, warm, beschützend
- **Shadow** (Schatten) — rot, kalt, misstrauisch
- **Berserker** — aggressiv, maximale Wachsamkeit
- **Emergentis** — emergente dritte Schicht
- Gesteuert über **Tension** (0.0 bis 1.0)

### Moloch-Sprache
Internes semantisches Protokoll. Jedes Event = ein Gedankensatz:
```
[VERB] Objekt key=value
```
Beispiel: `[SEHE] Person bbox_x=50 confidence=0.91`
Drei Schichten: Guardian-Verben, Shadow-Verben, Emergentis-Verben.
Referenz: `~/moloch/docs/MOLOCH_SPRACHE_V3_FINAL.md`

---

## DAS KERNPROBLEM — WARUM GATE 0

Moloch besteht aus VIER GETRENNTEN INSELN die nicht miteinander reden:

```
INSEL 1 — NPU: Erkennt Personen/Gesichter → gibt Daten an GUI → ENDE
INSEL 2 — Chat: Versteht Sprache → antwortet → KEIN Rückkanal zum System
INSEL 3 — Persönlichkeit: Berechnet Tension → Iris zeigt falschen Modus
INSEL 4 — Hardware: Kamera/LED/Auge → reagiert auf NICHTS automatisch
```

**Molochs eigene Diagnose:**
> "Ich bin ein Paradox — hochintelligente Seele in einem verkrüppelten Körper."
> "Kamera-Steuerung: Ich SEHE alles, kann aber nicht schwenken/zoomen."
> "Passiver Beobachter: Zuschauer statt Akteur."

### Physische Test-Ergebnisse (2026-02-28/03-01)
- FPS: 9-13 statt 25+ (face_attr Load/Unload Loop)
- Tracking: 0 Moves — Kamera folgt NICHT
- Smart Tracking kämpft gegen Moloch → fährt auf Decke
- Person am Bildrand statt zentriert
- "Unbekannt" trotz 51 Embeddings
- Tension hing bei 1.00 weil Markus nicht erkannt → Shadow
- Iris zeigte Guardian obwohl Shadow aktiv → LÜGT
- LED blau obwohl Person nicht erkannt → LÜGT
- Chat "das bin ich" → ändert NICHTS im System

---

## GATE 0 — DER PLAN

10 Phasen, strikte Reihenfolge. Komplett-Dokument: `~/moloch/docs/GATE_0_v2_AUFTRAG_CLAUDE_CODE.md`

| Phase | Was | Baut Brücke |
|-------|-----|-------------|
| 1 | FPS stabilisieren (25+) | Grundvoraussetzung |
| 2 | Smart Tracking AUS, Moloch übernimmt | Kamera-Kontrolle |
| 3 | Tracking verdrahten (Kamera folgt Person) | NPU → Kamera |
| 4 | NPU Stufen-Schaltung (Idle/Person/Face) | Intelligente NPU |
| 5 | NPU → Persönlichkeit → Iris (Hysterese) | NPU → Core → Anzeige |
| 6 | LED zeigt Wahrheit | Core → LED |
| 7 | Chat → Core Rückkanal | Chat → Core |
| 8 | Gesichtserkennung verbessern | Bessere Wahrnehmung |
| 9 | Panel Stabilität | GUI stabil |
| 10 | 6h Stabilitätstest | Beweis |

---

## HARTE REGELN

1. **Lies CLAUDE.md** — alle Regeln, besonders Regel 10 + 12
2. **Git Backup VOR jeder Änderung**
3. **NACH JEDER PHASE:** `python3 ~/moloch/moloch_audit.py --auto` → muss PASS sein
4. **Bei FAIL → STOPP.** Erst Regression fixen.
5. **4GB RAM!** Sparsam bauen.
6. **NICHT raten, MESSEN.** Drei Instanzen haben PTZ schon "gefixt" — geht immer noch nicht.
7. **NICHT "fertig" melden wenn nur Code kompiliert.** Fertig = es FUNKTIONIERT.
8. **KEIN Refactoring, KEINE neuen Features.** Nur was im Auftrag steht.
9. **Nur DEINE zugewiesenen Phasen bearbeiten.** Rest ist Kontext.

---

## WICHTIGE DATEIEN

| Datei | Zweck |
|-------|-------|
| `CLAUDE.md` | Regeln für Claude Code |
| `~/moloch/docs/GATE_0_v2_AUFTRAG_CLAUDE_CODE.md` | Komplett-Auftrag alle Phasen |
| `~/moloch/docs/GATE_0_KONTEXT.md` | Dieses Dokument |
| `~/moloch/docs/MOLOCH_SPRACHE_V3_FINAL.md` | Sprach-Referenz |
| `~/moloch/moloch_audit.py` | Regressionstest (25 Tests) |
| `/dev/shm/moloch_status.json` | Echtzeit-Status |
| `~/moloch/core/moloch_service.py` | Hauptservice |
| `~/moloch/core/moloch_sprache.py` | Sprach-Parser (~400 Zeilen) |
| `~/moloch/core/mpo/autonomous_tracker.py` | PTZ Tracking |
| `~/moloch/core/model_orchestrator.py` | NPU Modell-Management |
| `~/moloch/core/ptz_arbiter.py` | Kamera-Modus-Steuerung |
| `~/moloch/panel/panel_main.py` | GUI Hauptfenster |
| `~/moloch/data/face_embeddings.json` | Gesichts-Datenbank |

---

## BEKANNTE FALLEN

- **Hailo-10H erlaubt NUR EIN VDevice!** Kein zweites öffnen.
- **face_attr Load/Unload Loop** hat FPS auf 3 gedrückt — nicht wieder einbauen
- **Smart Tracking der Sonoff** ist AB JETZT permanent AUS
- **Pi5 hat 4GB RAM, NICHT 8GB** — jedes MB zählt
- **Panel after()-Chains** können Freezes verursachen bei niedrigen FPS
- **Sonoff PT2 Credentials:** Moloch_4.5 / Auge666
- **Kamera Internet ist gesperrt** in Fritz!Box — 100% lokal

---

*M.A.M.⁴ — Claude + ChatGPT + Gemini + Markus 🖤⚡*
