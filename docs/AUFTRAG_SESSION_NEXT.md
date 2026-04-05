# AUFTRAG: 3 Probleme — Naechste Session
# Erstellt: 2026-04-05 von Claude Opus (auf Anweisung Markus)
# Prioritaet: HOCH — alle 3 Probleme betreffen taegliche Nutzung

---

## UEBERSICHT

| # | Problem | Hauptagent | Nebenagent | Dateien (erwartet) |
|---|---------|------------|------------|--------------------|
| 1 | BBox + Landmarks fehlen im Videofenster | gui | vision | panel_preview.py |
| 2 | Face-ID erkennt Markus nicht, Tension bleibt -100% | personality | awareness, memory | core_integrator.py, personality_engine.py |
| 3 | GUI-Layout: Video zu gross, Konsole zu klein, Chat-Schrift zu klein | gui | — | panel_*.py, gui_main.py |

**Reihenfolge:** 1 → 2 → 3 (Rendering erst fixen, dann Erkennung, dann Layout)
**Regel:** 1 Auftrag = 1 Datei. Zwischen jedem Problem: Audit + Restart + Verify.

---

## PROBLEM 1: BBox + Landmarks fehlen im Videofenster

**Symptom:** Im Preview-Panel sind KEINE Overlays sichtbar:
- Keine Pose-Landmarks (Skelett)
- Keine BBox um Person (gruen)
- Keine BBox um Gesicht (cyan/gelb)

**Kontext:**
- hailooverlay wurde am 30.03. ENTFERNT aus GStreamer (blockierte SHM)
- BBox-Rendering laeuft seitdem via PIL in `panel_preview.py`
- `panel_detections` in `moloch_status.json` enthaelt alle Detektionen (normalisiert [0-1])
- Pose-Landmarks kommen vom PoseWorker (via `vision_workers.py`)

**Agent:** gui-Agent spawnen
**Skill:** `/moloch-dev` laden (NEVER-Regeln fuer GUI)
**MCP:** `moloch_status()` → pruefen ob `panel_detections` ueberhaupt Daten hat
**MCP:** `moloch_snapshot()` → Frame visuell pruefen

**Pruefen:**
1. Kommen Detektionen im Status-JSON an? (`panel_detections` nicht leer?)
2. Werden BBoxes in `panel_preview.py` gezeichnet? (PIL ImageDraw Code vorhanden?)
3. Werden Landmarks gezeichnet? (Pose-Daten muessen vom PoseWorker durchgereicht werden)
4. Koordinaten-Mapping: normalisiert [0-1] → Pixel korrekt? (Letterbox beachten!)

**Wenn Daten fehlen:** vision-Agent als Sub-Agent spawnen → Datenfluss PoseWorker → Status-JSON pruefen

**Verify:** `moloch_snapshot()` nach Fix — BBoxes + Landmarks muessen sichtbar sein

---

## PROBLEM 2: Face-ID erkennt Markus nicht + Tension bleibt bei -100%

**Symptom:**
- MOLOCH erkennt Markus NICHT (Face-ID: None, Face-Conf: 0.00)
- Tension bleibt bei -100% (komplett IDLE) obwohl Markus im Raum ist
- Erwartetes Verhalten: Markus erkannt → Tension steigt SOFORT auf Guardian-Zone (Hoechstwert)
- Ohne Person im Raum: Tension sollte bei Mittelwert/neutral sein, NICHT bei -100%

**Kontext:**
- ArcFace-Enrollment funktioniert (Similarity 1.00 frontal, seit 31.03.)
- FaceWorker laeuft (32k+ Inferences, 0 Errors)
- Face-DB: 308 Embeddings (Markus) — moeglicherweise alte DB inkompatibel?
- CoreIntegrator steuert Tension basierend auf Perception-Events
- PersonalityEngine reagiert auf Face-ID Events

**Agenten:** personality-Agent spawnen (Hauptproblem: Tension-Reaktion)
- awareness-Agent als Sub: Warum wird Markus nicht als bekannt erkannt?
- memory-Agent als Sub: Ist die Face-DB aktuell? Muss neu enrollt werden?

**Skill:** `/moloch-dev` laden
**MCP:** `moloch_status()` → Face-ID, Face-Conf, Zone pruefen
**MCP:** `moloch_conversation()` → Hat MOLOCH ueberhaupt Face-Events geloggt?
**MCP:** `moloch_logs(n=50, filter_str="FACE")` → Face-Match Logs pruefen

**Pruefen:**
1. Kommt ueberhaupt ein Face-Match? (ArcFace Similarity-Werte in Logs?)
2. Wenn ja: Wird das Event an CoreIntegrator weitergeleitet?
3. Wenn ja: Reagiert PersonalityEngine auf das Event?
4. Tension-Mapping: Was setzt Tension auf -100%? Fehlt der Trigger fuer Guardian?
5. Idle-Tension: Sollte NICHT -100% sein wenn niemand da ist — eher neutral/Mittelwert

**Verify:**
- Markus vor Kamera → `moloch_status()` zeigt Face-ID: "markus", Tension > 0
- Markus weg → Tension faellt auf Mittelwert (NICHT -100%)

---

## PROBLEM 3: GUI-Layout — Video zu gross, Konsole zu klein

**Symptom:** Das Gesamtfenster und die Proportionen stimmen nicht:
- Videofenster (Preview) ist zu gross
- Konsole (Mittelfenster) ist zu klein — Buttons werden abgeschnitten
- Chatfenster hat zu kleine Schrift

**Gewuenschtes Layout:**
- **Videofenster:** KLEINER machen, aber Aufloesung beibehalten (nur Darstellung skalieren)
- **Konsole (Mitte):** VERBREITERN + HOEHER — alle Buttons muessen sichtbar sein
- **Chatfenster:** KLEINER machen, aber Schriftgroesse GROESSER (besser lesbar)
- **Gesamtfenster:** Proportional verkleinern, alles passt zusammen

**Agent:** gui-Agent spawnen
**Skill:** `/moloch-dev` laden

**Pruefen:**
1. Welche panel_*.py steuert das Layout? (wahrscheinlich gui_main.py oder aehnlich)
2. Wie ist das Grid/Pack-Layout aufgebaut?
3. Wo wird die Preview-Groesse gesetzt?
4. Wo wird die Chat-Schriftgroesse gesetzt?
5. Wo werden Konsolen-Buttons platziert?

**WICHTIG:** Keine Funktionalitaet aendern, NUR Layout-Anpassungen!

**Verify:** GUI starten, visuell pruefen:
- Alle Buttons sichtbar?
- Schrift im Chat lesbar?
- Video nicht mehr ueberproportional gross?

---

## WORKFLOW-CHECKLISTE

Fuer JEDES der 3 Probleme diese Schritte:

```
[ ] 1. moloch_status() + moloch_npu_workers() — Startprotokoll
[ ] 2. Domain-Erkennung → richtigen Agent spawnen
[ ] 3. /moloch-dev laden — NEVER-Regeln pruefen
[ ] 4. Betroffene Dateien lesen (ERST lesen, DANN coden)
[ ] 5. Ampel-Check: ROT = einmal fragen, GELB = ankuendigen
[ ] 6. Code aendern (1 Datei pro Schritt)
[ ] 7. __pycache__ loeschen
[ ] 8. sudo systemctl restart moloch
[ ] 9. Verify via MCP (moloch_status, moloch_snapshot)
[ ] 10. moloch_audit --auto → muss PASS sein
[ ] 11. Git commit + push
[ ] 12. Naechstes Problem
```

---

## AGENTEN-UEBERSICHT (was geladen werden muss)

| Agent | Fuer Problem | Rolle |
|-------|-------------|-------|
| gui | 1, 3 | BBox/Landmark-Rendering, Layout-Anpassung |
| vision | 1 (Sub) | Datenfluss PoseWorker → Status-JSON |
| personality | 2 | Tension-Reaktion auf Face-ID |
| awareness | 2 (Sub) | Warum wird Person nicht erkannt? |
| memory | 2 (Sub) | Face-DB Status, ggf. Neu-Enrollment |

## MCP-TOOLS (die gebraucht werden)

| Tool | Wofuer |
|------|--------|
| `moloch_status()` | Startprotokoll + Face-ID + Tension pruefen |
| `moloch_npu_workers()` | Worker-Health |
| `moloch_snapshot()` | Visuell pruefen ob BBoxes/Landmarks sichtbar |
| `moloch_logs(filter="FACE")` | Face-Match Events |
| `moloch_logs(filter="ERROR")` | Fehler |
| `moloch_audit()` | Regressionstest nach jedem Fix |
| `moloch_say()` | MOLOCH direkt fragen ob er Markus sieht |

## SKILLS (die geladen werden muessen)

| Skill | Wann |
|-------|------|
| `/moloch-dev` | VOR jeder Code-Aenderung |
| `/moloch-agent` | Wenn Agent-Zuordnung unklar |
| `/moloch-status` | Fuer Status-Interpretation |
