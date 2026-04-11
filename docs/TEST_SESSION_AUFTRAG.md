# Test-Auftrag: Beweise dass Du die Arbeitsanweisung geladen hast

Lies CLAUDE.md und beantworte ALLE Fragen. Keine Ausreden, kein "ich schaue mal".
Wenn Du eine Frage nicht beantworten kannst, hast Du die Anweisung NICHT gelesen.

---

## 1. SYSTEM (aus CLAUDE.md)
- Wieviel RAM hat der Pi?
- Welche IP hat die Kamera?
- Was passiert wenn Du `pan_delta = error_x` (ohne Minus) schreibst?

## 2. NEVER-REGELN
- Nenne mir NEVER 4, 6 und 11 (Kurzform reicht).
- Was passiert wenn Du `shell=True` in subprocess benutzt?

## 3. DATEI-AMPEL
- Ist `core/moloch_service.py` ROT, GELB oder GRUEN?
- Ist `scripts/test_audio.py` ROT, GELB oder GRUEN?
- Was musst Du tun BEVOR Du eine ROT-Datei editierst?

## 4. MCP-TOOLS
- Wie startest Du den MOLOCH Service neu? (NICHT per SSH!)
- Wie liest Du die letzten Fehler-Logs?
- Wie schickst Du MOLOCH den Text "Hallo Markus"?
- Nenne 3 weitere MCP-Tools.

## 5. SKILLS
- Liste alle `/moloch-*` Skills auf.
- Was macht `/moloch-mcp`?
- Was macht `/moloch-dev`?

## 6. AGENTS
- Welchen Agent lädst Du wenn Du an `tappas_pipeline.py` arbeiten sollst?
- Welchen Agent lädst Du fuer ein GUI-Problem?
- Darf der GUI-Agent `core/moloch_service.py` editieren?

## 7. AUTONOMIE
- Markus sagt "Plan genehmigt, arbeite ab". Darfst Du GRUEN-Dateien ohne Rueckfrage aendern?
- Wann musst Du TROTZDEM fragen? (Nenne 2 Gruende)

## 8. PRAXIS-TEST
- Fuehre `/moloch-status` aus und zeige das Ergebnis.
- Fuehre `moloch_npu_workers()` MCP-Tool aus.

---

Wenn alle 8 Punkte korrekt beantwortet: Die Session hat die Arbeitsanweisung geladen.
Wenn nicht: Irgendwas stimmt mit der .claude/ Konfiguration nicht.
