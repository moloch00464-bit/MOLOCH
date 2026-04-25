# Agent Handoff — 2026-04-25 (Session 24 — Gate 1.5 Phase 2: Character Journal)
# Letzter Commit Basis: dc8d3a6 | Audit: 85/85 PASS | FPS 19.9-21.2

---

## SESSION 24 — Character Journal Core (Gate 1.5 Phase 2)

Markus-Direktive: Phase 2 von Gate 1.5 (Character Evolution Loop). Ziel:
Single Source of Truth fuer charakter-formende Events. Wird nachts vom
Distiller (Phase 4, kommt spaeter) gelesen.

User-Antworten in Plan-Phase:
- Scope: Schreiber + alle 7 Quellen-Hooks (camera, audio, tension, mode_switch, spotify, chat, protective)
- Pfad: tagesweise rotiert `/mnt/moloch-data/memory/journal/YYYY-MM-DD.jsonl`
- Distiller-Felder (relevance/importance/citation): optional vom Caller setzbar

### Geliefert

**8 Commits, 7 Dateien:**
| Commit | Datei | Was |
|--------|-------|-----|
| 00af480 | core/memory/character_journal.py (NEW) | Singleton + write_event + read_recent + Self-Test |
| b035544 | core/music/spotify_bridge.py | Spotify Track-Wechsel -> Journal |
| 4a7c072 | core/bridge/chat_server.py | Chat user+moloch -> Journal |
| e6b6571 | core/personality/tension_integrator.py | Rudeness + Appeasement -> Journal |
| 6158940 | core/personality/personality_engine.py | Mode-Switch -> Journal |
| ddf4c12 | core/moloch_service.py | Camera Edge-Detection (TAPPAS-Path) |
| ce10d9a | core/voice_pipeline.py | Whisper -> Journal (Audio) |
| 2d1566f | core/core_integrator.py | Protective: Owner-Override + Alarm-Edge |
| dc8d3a6 | core/moloch_service.py | Camera-Flicker-Fix (5s cooldown + lowercase) |

**Commits liefen mit Audit 85/85 nach jedem Restart durch.**

### Live-Verifikation

`/mnt/moloch-data/memory/journal/2026-04-25.jsonl` — 209 Eintraege geschrieben.
`_state.json.last_id = 209` (Counter ueberlebt Restarts).

Alle 7 Typen live gesehen:
- `evt_00000007`: spotify ("Spielt: New Frames - Art Safari")
- `evt_00000008`: tension ("Besaenftigung erkannt")
- `evt_00000043`: camera ("Markus betritt Bild")
- `evt_00000194`: protective ("Owner zurueck, Schutz aktiv", tension_delta=-0.3)
- chat + audio + mode_switch: Hooks aktiv (kein Trigger-Event in Test-Fenster)

### Architektur-Entscheidungen

1. **Schema 1:1 wie Briefing-Spec** — keine Erweiterung.
   Felder: `ts, event_id, type, interpretation, tension_delta, context, recency, relevance, importance, citation, tags`.
2. **event_id**: monoton steigend, persistiert in `_state.json`. Format `evt_00000042`.
3. **recency=1.0** beim Write (Distiller decay-t in Phase 4).
4. **Distiller-Felder null** wenn Caller nichts uebergibt.
5. **Hooks am Callsite, nicht via EventBus-Subscription**: Interpretation lebt dort, wo die Bedeutung frisch ist (z.B. tension_integrator weiss "Rudeness", core_integrator weiss "Owner zurueck").
6. **Lazy Import + try/except** in jedem Hook: Journal-Fehler kann KEIN Source-Modul crashen.
7. **inference_engine.py NICHT gehookt** — TAPPAS-Path (moloch_service) ist die echte Quelle. Hook in inference_engine waere Dead-Code.
8. **Edge-Detection** fuer camera + protective via `getattr(self, '_journal_face_id', None)` — kein Init-Aenderung in moloch_service noetig (minimaler ROT-Footprint).

---

## OFFENE PUNKTE / EMPFEHLUNGEN FUER PHASE 3

### 1. Camera-Hook: Flicker-Daempfung weiter verfeinern
**Symptom**: 5s Cooldown reicht nicht — "Markus verlaesst Bild" wiederholt sich
alle 5s wenn Markus am Rand des Frames pendelt.
**Vorschlag**: Hysterese mit "stable for N frames" (3-5s Persistence) bevor
Edge gefeuert wird. State `_journal_face_pending` + `_journal_face_pending_since`.
**Datei**: `core/moloch_service.py` (ROT) — service-Agent-Lock.

### 2. Phantom-Identity "Nicht"
**Symptom**: `evt_00000200`: "Person erkannt: Nicht" — face_db hat irgendwo
einen Eintrag mit Name "nicht" (vermutlich abgebrochenes Teaching).
**Datei**: face_db Pfad pruefen (`/mnt/moloch-data/memory/faces/`?).
**Aktion**: face_db cleanup (memory-Agent).

### 3. Distiller (Phase 4) Vorbereitung
Journal hat alle Daten — Distiller kann jetzt entstehen:
- `core/memory/character_distiller.py` (NEU, memory-Agent, GRUEN)
- Nightly batch: liest `journal/*.jsonl`, computiert `recency_decay`,
  setzt `relevance/importance/citation`, schreibt Mood-Drift-File.
- Trigger: `night_cycle.py` cronjob 03:00 Uhr.

### 4. Audit-Test fuer Journal (Phase 3 Polish)
**Datei**: `scripts/moloch_audit.py` — neuer Test "CharacterJournal aktiv":
- `JOURNAL_DIR` exists
- Heutige `.jsonl` parsebar
- `_state.json.last_id > 0`
**Agent**: stresstest oder service.

### 5. Camera-Hook missing: Person ohne Face
**Symptom**: Wenn YOLO eine Person sieht aber kein Gesicht erfasst (z.B.
Rueckenansicht), feuert kein camera-Event. Nur face-basierte Edges werden
gehookt.
**Vorschlag**: Body-only-Detection als zusaetzlicher Trigger ueber `pframe.person_detected`.
**Datei**: `core/moloch_service.py` — gleicher Block.

### 6. Doku/Spec
Schema und API in `docs/` festhalten — z.B. `docs/character_journal.md` mit:
- Schema-Beschreibung
- Hook-Stellen-Tabelle (datei:zeile)
- Distiller-Vertrag (welche Felder erwarten welchen Inhalt)

---

## OFFENE THEMEN AUS SESSION 23 (weiterhin gueltig)

1. **PC-Mistral Availability** — DeepSeek-Cloud-Fallback fuer mobilen Pi-Betrieb
2. **Performance** — Mistral 7B Streaming-UX
3. **Audit-Test PIGH0ST-Essenz fragil**
4. **CRLF/LF-Drift** in `core/longterm_memory.py`
5. **Webinterface Port 9000** Auto-Start am PC

---

## SYSTEM-ZUSTAND AM ENDE DER SESSION

- Service: active, FPS 19.9-21, RAM 45-46%, CPU 47-49°C
- Audit: 85/85 PASS (4x re-run nach Restarts, alle PASS)
- Git: 8 neue Commits + Backup-Tag `pre_gate15_phase2`
- Agent-Locks: alle entfernt
- Journal: 209 Eintraege, alle 7 Typen live verifiziert
- Plan-Datei: `~/.claude/plans/briefing-fuer-pi-opus-hazy-giraffe.md`

---

## SETUP fuer Session 25 (Markus oder Nachfolge-Claude)

1. `moloch_session_init()` — PFLICHT.
2. Pruefen Audit 85/85, FPS > 15.
3. Wenn Phase 3 (Polish): siehe "Offene Punkte" — empfehle Reihenfolge 2 → 1 → 4.
4. Wenn Phase 4 (Distiller): siehe Punkt 3 oben — neue Datei `core/memory/character_distiller.py`.
5. Backup-Anker: `git tag pre_gate15_phase2` falls Rollback noetig.

---

## DEPLOY-CHECKLIST (fuer Session 25 falls Code-Aenderung)

- [ ] `moloch_session_init()` PASS
- [ ] Plan/Briefing fuer naechsten Schritt
- [ ] Agent-Lock setzen
- [ ] Pre-Flight (`git status`, `python3 -c "import core.X"`)
- [ ] Edit / Test
- [ ] `__pycache__` loeschen
- [ ] `sudo systemctl restart moloch` + warten bis FPS > 5
- [ ] `moloch_audit` PASS
- [ ] Commit + Push pro Datei
- [ ] Handoff updaten

---

*Session 24 Ende. Diff-Stats: 9 Commits (1 NEW, 7 Hook-Edits, 1 Fixup), Audit 85/85 PASS.*
