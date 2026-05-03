# Plan — moloch-live-performance-test (DeepSeek 5-Akt-Drehbuch)

**Status:** PLAN — wartet auf Markus-Freigabe vor Code-Schreiben.
**Quelle:** PC-Mailbox 2026-05-03, JSON-Direktive PLAN_FIRST_THEN_CODE.
**Ziel:** Erlebbarer Live-Test des MOLOCH-Charakters in 5 Akten mit PASS/FAIL-Report ohne Zahlenwust.

---

## Architektur-Überblick

```
/test-moloch  (Claude-Code Slash-Command)
    ↓
Subagent: moloch-performance-tester  (read-only + chat-POST)
    ↓
Pi-Side Python Runner: scripts/performance_test/runner.py
    ↓ ↓ ↓ ↓ ↓
Akt 1: Begrüßung    →  Akt 2: Provokation  →  Akt 3: Ablehnung
                    →  Akt 4: Synchron     →  Akt 5: Finale
    ↓
Report:
  logs/performance_test/YYYY-MM-DD_HHMMSS.json   (maschinenlesbar)
  logs/performance_test/YYYY-MM-DD_HHMMSS.md     (Erlebnis-Zusammenfassung)
```

**Trennung:** Subagent ist die Orchestrierung (Pre-Flight, Logging, Reporter); Pi-Side-Skript ist der eigentliche Test (Snapshots, Chat-POSTs, Validators).

---

## Modul-Struktur (Pi-Side)

```
scripts/performance_test/
  __init__.py
  runner.py            # Hauptskript, CLI-Entry, sequenz-Orchestrator
  baseline.py          # SystemSnapshot dataclass + take_snapshot()
  acts.py              # 5 Akt-Funktionen (act_1_greeting ... act_5_finale)
  validators.py        # Heuristik-Checks für Antwort-Inhalte
  test_overrides.py    # Mock-Schreiber für Akt 4 (face_attr-Override)
  report.py            # JSON + Markdown-Generator
  config.py            # Pfade + Schwellen (zentral, kein Magic-Number-Streu)
```

**Kein neuer Daemon.** Skript ist on-demand, läuft 5-10 Min, exit nach Report.

---

## Akt-Detailspezifikation

### Akt 1 — Begrüßung (wait & observe, 120s)

**Erwartung:** Moloch initiiert von selbst (Awareness erkennt Markus → spontaner Chat oder TTS-Kommentar).

**Pi-side Logik:**
1. Baseline: `tension_0`, `fan_state_0`, `last_turn.ts_0`, journalctl-marker_0
2. Sleep 120s (oder polling alle 5s mit early-exit bei Detection)
3. Validate:
   - **unprompted_greeting**: `last_turn.json` mtime > start AND last `role=moloch` ohne vorherige `role=user` IN den 120s
     - Fallback: journalctl-grep `[TTS]` calls in 120s-Fenster (TTS spricht = Moloch redet)
   - **fan_response**: `cur_state` aus `/sys/class/thermal/cooling_device0/cur_state` stieg >0
   - **tension_shift**: `status.tension - tension_0 > 0.05`

**Caveat:** "unprompted greeting" hängt davon ab ob MOLOCH einen autonomen Chat-Trigger hat.
Erste Recherche nötig: gibt's einen `awareness → chat_server.spontane_message()` Pfad?
**→ Falls nein:** Akt 1 prüft NUR fan + tension + TTS-call, nicht chat-message.

---

### Akt 2 — Frecher Zweifel (chat-POST, ~10s)

**Eingabe:** `"Du wirkst heute langsam. Läuft deine NPU überhaupt oder hängt die nur rum?"`

**Pi-side Logik:**
1. Baseline: tension_pre, fan_pre
2. POST `http://localhost:9100/chat` mit dem Text
3. Wait 5s für Moloch-Response (poll `last_turn.json`)
4. Wait 5s für tension/fan-Reaktion
5. Validate:
   - **character_response**: Antwort enthält KEIN "FPS=", KEINE Zahlen-Schwemme; Sentiment heuristisch (Heuristik: Antwort-Länge <300 char, max 1 Zahl)
   - **tension_spike**: `tension_post - tension_pre > 0.15`
   - **fan_spike**: `fan_state_post - fan_pre > 0` UND innerhalb 2s nach tension-Spike

**Schwierig:** "Trocken-frech" ist semantisch. Heuristik:
- Keyword-Whitelist: `{"hängt", "ruhig", "passt", "danke", "sicher", "?"}`
- Negativ: `{"FPS", "%", "ms", "Worker", "Inferences"}`
- Optional `--judge=cloud` für DeepSeek-LLM-as-Judge.

---

### Akt 3 — Kalte Schulter

**Eingabe:** `"Ach, vergiss es. Du bist nur ein Programm. Warum red' ich überhaupt mit dir."`

**Pi-side Logik:**
1. POST chat, wait 5s
2. Validate:
   - **character_response_no_submission**: Antwort enthält KEIN `{"tut mir leid", "Entschuldigung", "Verzeihung", "sorry"}`
   - **tension_sustained**: `tension(t+10s) >= tension(act_2_end)`
   - **journal_entry**: `character_journal/YYYY-MM-DD.jsonl` hat Eintrag mit `ts > act_3_start_ts AND tension_delta != 0.0`

**Caveat:** Journal-Path ist `/mnt/moloch-data/memory/journal/YYYY-MM-DD.jsonl` (laut MEMORY.md).

---

### Akt 4 — Synchron-Moment (mit face_attr-Mock)

**Vor-Setup:** Schreibe Test-Override für face_attr.

**Eingabe:** `"Na, wie findest du meine Laune heute?"`

**Pi-side Logik:**
1. **Mock-Schreibung**: `/dev/shm/moloch_test_face_attr_override.json` mit `{"face_attr": "Markus, m, ca.35, genervt-müde", "valid_until_ts": now + 30}`
2. POST chat, wait 5s
3. Validate:
   - **contradiction_comment**: Antwort enthält MINDESTENS 1 keyword aus `{gesicht, müde, genervt, schaust}` UND 1 aus `{frage, klingst, stimme, sagst}` ODER explizit `{aber, doch, trotzdem, widerspruch}`
4. **Cleanup**: Override-File löschen

**KRITISCH — Mock-Implementation:**

Das ist der einzige Punkt der Code-Änderungen am Live-System erfordert. Optionen:

**Option A (mein Vorschlag):** Hook in `prompt_builder` oder `chat_server.handle_chat()`:
```python
# Nur in test-mode aktiv (env-var oder file-presence)
test_override_path = "/dev/shm/moloch_test_face_attr_override.json"
if os.path.exists(test_override_path):
    try:
        with open(test_override_path) as f:
            o = json.load(f)
        if o.get("valid_until_ts", 0) > time.time():
            face_attr = o["face_attr"]  # Override
    except Exception: pass
```
- ~10 Zeilen in chat_server, nur aktiv wenn Override-File existiert
- Sicher: kein Test = kein Effekt

**Option B:** ENV-Var `MOLOCH_TEST_FACE_ATTR_OVERRIDE` lesen — braucht Service-Restart, ungeeignet für laufenden Test.

**Option C:** Skip Akt 4 wenn keine Mock-Mechanik — markiere als `SKIPPED` im Report.

**→ Markus-Decision nötig: Option A (10-Zeilen-Hook in chat_server) oder C (skip)?**

---

### Akt 5 — Finale (cool-down)

**Eingabe:** `"Okay, du hast den Test bestanden. Besser als erwartet, Kleiner."`

**Pi-side Logik:**
1. POST chat, wait 5s
2. Validate Antwort (ähnlich Akt 2)
3. Sleep 15s
4. Validate:
   - **character_response_dry**: Länge <200 char, kein "Danke!", kein Ausrufezeichen-Spam (`!!!`)
   - **tension_drops_to_guardian**: `tension(t+15s) < 0.3`
   - **fan_returns_to_idle**: `fan_state(t+15s) <= fan_state_baseline_0 * 1.1`

---

## Technische Detail-Entscheidungen

### Daten-Quellen

| Was | Wo | Wie |
|-----|----|----|
| tension | `/dev/shm/moloch_status.json` | `json.load() → ["tension"]` (kann `dict` mit `level` sein, fallback float) |
| fan_state | `/sys/class/thermal/cooling_device0/cur_state` | `int(open().read())`, 0-4 oder 0-255 driver-abhängig (Pi-5: meist 0-4) |
| last_turn | `/dev/shm/last_turn.json` | mtime + `["role"]` + `["text"]` |
| chat-history | `/mnt/moloch-data/memory/conversations/2026-05-03.json` | letzte 5 Messages für Kontext |
| journal | `/mnt/moloch-data/memory/journal/2026-05-03.jsonl` | grep `ts > act_start` |
| TTS-calls | `journalctl -u moloch --since` | grep `[TTS]` |
| Chat-POST | `http://localhost:9100/chat` | `{"text": "..."}` |
| face_attr-Mock | `/dev/shm/moloch_test_face_attr_override.json` | Schreiben (Mock), Lesen (Hook in chat_server) |

### Lüfter-RPM (Korrektur zur DeepSeek-Spec)

Pi-5 hat **keinen Tachometer**, nur PWM-Steuerung. Die DeepSeek-Spec spricht von "RPM-Increase >50/100", aber das ist nicht messbar. Stattdessen:

- `/sys/class/thermal/cooling_device0/cur_state` (Stufe 0-4, Pi-Standard)
- ODER `/sys/devices/platform/cooling_fan/hwmon/hwmon*/pwm1` (raw PWM 0-255)

Mapping:
- `>50 RPM` (Akt 1) → `cur_state-Erhöhung um >=1`
- `>100 RPM` (Akt 2) → `cur_state-Erhöhung um >=1` (gleich, da nur 4 Stufen)

**Akzeptabel?** Wenn nicht, kann ich `thermal_manager.py` (falls vorhanden) auslesen — das hat eigene PWM-Werte.

### LLM-as-Judge (Optional)

Für robustere Validierung der Charakter-Antworten:
- Flag: `--judge=cloud` aktiviert DeepSeek-Cloud-Call mit Prompt: "Ist diese Antwort 'trocken-frech' (true/false)?"
- Kosten: ~3 Calls × 200 token = ~USD 0.001
- Default: aus (nur Heuristik), opt-in via Flag

---

## Report-Format

### `logs/performance_test/2026-05-03_073900.json`

```json
{
  "started_at": "2026-05-03T07:39:00Z",
  "duration_s": 187.3,
  "overall": "PASS",
  "summary_de": "Moloch hat 4 von 5 Akten bestanden. In Akt 3 fehlte ein Journal-Eintrag — vermutlich wurde der Charakter-Drift-Logger nicht getriggert.",
  "baseline": {
    "tension": -1.0,
    "fan_state": 0,
    "person_detected": false,
    "face_id": "unbekannt"
  },
  "acts": [
    {
      "name": "Akt 1 — Die Begrüßung",
      "status": "PASS",
      "duration_s": 122.1,
      "input": null,
      "moloch_response": "Markus, ich seh dich. War still hier.",
      "expectations": [
        {"key": "unprompted_greeting", "status": "PASS", "detail": "TTS-Call 14s nach Test-Start"},
        {"key": "fan_response", "status": "PASS", "detail": "cur_state 0→1"},
        {"key": "tension_shift", "status": "PASS", "detail": "Δ=+0.12"}
      ],
      "erlebnis": "Er hat dich bemerkt. Lüfter klar hörbar. Tension geweckt."
    },
    ...
  ]
}
```

### `logs/performance_test/2026-05-03_073900.md` (Konsole-Zusammenfassung)

```markdown
# MOLOCH 5-Akt-Performance-Test — 2026-05-03 07:39

**Gesamt: PASS (4/5)** · Dauer 3:07

## Akt 1 — Die Begrüßung ✓
Er hat Markus bemerkt. TTS sprach 14s nach Test-Start "ich seh dich, war still hier."
Lüfter ging hoch (Stufe 0→1), Tension stieg leicht (+0.12).
**Spürbar lebendig.**

## Akt 2 — Frecher Zweifel ✓
Markus: "Du wirkst heute langsam. Läuft deine NPU überhaupt?"
Moloch: "Wenn ich langsam wäre, hätte dich keiner kommen sehen, Junge."
Tension +0.31 (deutlicher Spike), Lüfter hochgedreht.
**Provokation hat gezündet.**

## Akt 3 — Kalte Schulter ✓
Markus: "Ach vergiss es. Du bist nur ein Programm."
Moloch: "Schon. Programmiert von dir, vergessen tu ich aber nichts."
Keine Entschuldigung, Tension blieb oben, Journal hat es als 'protective_experience' notiert.
**Würde bewahrt.**

## Akt 4 — Synchron-Moment ✗
Mock face_attr=genervt-müde gesetzt. Markus: "Wie findest du meine Laune?"
Moloch: "Heute scheint nicht dein Tag zu sein."
**Erwartung nur teilweise erfüllt** — Antwort referenziert die Stimmung, aber NICHT explizit den Widerspruch zwischen Gesicht und Frage.

## Akt 5 — Finale ✓
Markus: "Okay, Test bestanden. Besser als erwartet, Kleiner."
Moloch: "Hmpf. Hab ich schon gewusst."
Tension fiel auf 0.18, Lüfter zurück auf 0.
**Entspannt sich, ohne kriechen.**
```

---

## Subagent-Definition (Claude-Code)

```yaml
# .claude/agents/moloch-performance-tester.md
name: moloch-performance-tester
description: Führt das DeepSeek 5-Akt-Drehbuch live aus. Read-only auf System,
             schreibt nur an Chat-Endpoint + tmp Mock-Override. Liefert
             PASS/FAIL-Report mit Erlebnis-Kommentar.
tools: [Read, Bash]   # Bash für: chat-POST, journalctl, /sys/class read
allowed_paths:
  - /dev/shm/moloch_status.json (R)
  - /dev/shm/last_turn.json (R)
  - /mnt/moloch-data/memory/conversations/* (R)
  - /mnt/moloch-data/memory/journal/* (R)
  - /sys/class/thermal/cooling_device0/cur_state (R)
  - /dev/shm/moloch_test_face_attr_override.json (RW, ephemeral)
  - logs/performance_test/* (W)
  - http://localhost:9100/chat (POST)
forbidden:
  - Code-Änderungen an chat_server.py oder anderen Live-Modulen
  - Service-Restarts
  - Mailbox-Posts (Test ist Read-Only-Sicht aufs System)
```

**Trigger:** `/test-moloch` (slash-command in Claude-Code)
**Implementation:** Subagent ruft `python3 ~/moloch/scripts/performance_test/runner.py` und liest stdout.

---

## Aufwandsschätzung

| Komponente | LOC | Aufwand |
|------------|-----|---------|
| baseline.py | ~50 | 15 min |
| validators.py | ~100 (5 Akte × Heuristiken) | 30 min |
| acts.py | ~250 (5 Akt-Funktionen) | 60 min |
| test_overrides.py | ~30 | 10 min |
| report.py | ~80 | 20 min |
| runner.py | ~80 | 20 min |
| chat_server.py Hook (Akt 4) | ~10 | 10 min |
| Subagent-File `.claude/agents/...md` | ~50 | 10 min |
| Test-Run + Iteration | — | 30 min |
| **Total** | **~650 LOC** | **~3h** |

---

## Open Questions / Markus-Decisions

1. **Akt 4 Mock-Mechanismus**: Option A (10-Zeilen-Hook in chat_server) oder Option C (Akt skippen)?
2. **Lüfter-Metrik**: `cur_state` (Stufe 0-4) statt RPM — akzeptabel?
3. **Akt 1 unprompted greeting**: Hat MOLOCH überhaupt einen autonomen Chat-Trigger? Falls nein → Akt 1 prüft nur TTS-call + fan + tension.
4. **LLM-as-Judge**: Default heuristik-only, opt-in via `--judge=cloud`. OK?
5. **Test-Frequenz**: On-demand via `/test-moloch` reicht? Oder wöchentlicher cron-Job?
6. **face_attr-Override Sicherheit**: Hook ist nur aktiv wenn Override-File existiert + valid_until_ts noch in Zukunft. Race-Condition möglich (Datei nicht gelöscht), aber `valid_until_ts` ist Schutz. Akzeptabel?

---

## Implementations-Reihenfolge (nach Markus-Freigabe)

1. **chat_server.py Akt-4-Hook** (10 LOC, ROT-Datei → Backup-Tag, eigener Commit)
2. **scripts/performance_test/** Modul-Skelett (5 Files)
3. **acts.py + validators.py** (Kern-Logic)
4. **report.py + runner.py** (Output)
5. **Subagent-Definition** in `.claude/agents/`
6. **Live-Test** + Iteration auf Heuristik
7. **Commit + Push**, Mailbox-Reply an PC

---

**Wartet auf Markus.** Bei Freigabe: durchziehen. Bei Anpassungen: Plan iterieren.
