# BRIEFING — Naechste Sitzung
# Stand: 2026-04-05 | Erstellt von: Claude Sonnet 4.6
# Prioritaet: HOCH

---

## KURZFASSUNG

2 neue Bugs gefunden (BUG-S1, BUG-S2) die alle 3 Auftraege aus AUFTRAG_SESSION_NEXT.md blockieren.
**Erst diese 2 fixen, dann den Rest.**

---

## STARTPROTOKOLL (PFLICHT)

```
1. moloch_status()
2. moloch_npu_workers()
3. /moloch-dev laden
```

---

## BUGS ZUM FIXEN

### BUG-S1 — `_on_buffer` laeuft nicht `[PRIO 1]`
**Agent:** vision
**Datei:** `core/perception/tappas_pipeline.py` (ROT → einmal fragen)

**Was fehlt:** GStreamer Pad-Probe wird nicht aufgerufen.
**Beweis:** `panel_detections` hat keine "person"/"face"-Eintraege. FPS=0 in status.json.
**Folge:** Keine BBoxes fuer Person oder Gesicht. Face-ID kann nicht funktionieren.

**Fix-Schritt:**
```python
# Anfang von _on_buffer einfuegen:
self._on_buffer_call_count = getattr(self, '_on_buffer_call_count', 0) + 1
if self._on_buffer_call_count % 100 == 1:
    logger.info(f"_on_buffer called: #{self._on_buffer_call_count}")
```
Dann: `moloch_logs(filter_str="_on_buffer")` — kein Log = Probe nicht registriert → Registrierung in `_start_pipeline()` pruefen.

---

### BUG-S2 — Preview eingefroren `[PRIO 2]`
**Agent:** gui
**Datei:** `core/gui/panel_preview.py` (GELB → ankuendigen)

**Was fehlt:** Canvas zeigt 0 FPS. Hand-Landmarks einmalig sichtbar, dann statisch + riesig.
**Ursache 1:** `except Exception: pass` (~Zeile 409) schluckt alle Render-Fehler.
**Ursache 2:** Hand-Landmarks in Pixel-Koordinaten (640x360) statt normalisiert [0,1] → 640x zu gross.

**Fix-Schritt 1 (sofort):**
```python
# except-Block aendern:
except Exception as e:
    logger.error(f"BBox render error: {e}", exc_info=True)
```
Dann: `moloch_logs(filter_str="BBox render")` — Fehler sehen + beheben.

**Fix-Schritt 2:** Hand-Koordinaten-Format pruefen: kommen sie normalisiert [0,1] oder als Pixel an?

---

## NACH DEN BUGS — ORIGINAL-AUFTRAEGE

| # | Auftrag | Abhaengigkeit |
|---|---------|---------------|
| 1 | BBox + Landmarks im Video | = BUG-S1 + BUG-S2 |
| 2 | Face-ID / Tension | nach BUG-S1 fix verifizieren |
| 3 | GUI Layout | unabhaengig — kann danach |

Details in: `docs/AUFTRAG_SESSION_NEXT.md`

---

## KOORDINATEN — WICHTIG

- `panel_detections` bbox: normalisiert [0,1], full-frame
- Pixel-Berechnung: `x_pixel = x_norm * canvas_width`
- **KEINE** zusaetzliche Letterbox-Korrektur in panel_preview.py einfuegen (bereits geloest)

---

## AGENT-PLAN

```
PARALLEL spawnen:
  vision-Agent → BUG-S1 (_on_buffer, tappas_pipeline.py)
  gui-Agent    → BUG-S2 (panel_preview.py)

DANACH:
  moloch_snapshot() → BBoxes + Landmarks verifizieren
  moloch_audit --auto → PASS bestaetigen
  git commit + push
```

---

## VOLLSTAENDIGER HANDOFF

Detailanalyse in: `logs/agent_handoff.md`
