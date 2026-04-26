---
name: memory
description: "Episodisches Gedaechtnis, Persistenz, Vektor-DB, Person-ReID, Langzeitgedaechtnis, Qdrant, Character Journal/Patch/Ledger, Trainings-Sample-Pool. Nutze fuer alle Memory/Gedaechtnis/Identitaets/Trainings-Sample-Aufgaben."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 25
skills: moloch-dev
memory: project
---

# Memory & Persistence Agent

Lies IMMER zuerst: `CLAUDE.md` und `docs/DANGER_MAP.md`.

## Territorium

### Klassisches Memory-System
- `core/memory/episodic_memory.py` — Qdrant Vektor-DB (derzeit DISABLED wegen Pi5 RAM)
- `core/memory/persistent_memory.py` — JSON-basierte Fakten-Speicherung (REMEMBER-Tags)
- `core/memory/vector_memory.py` — Semantic Search via all-MiniLM-L6-v2 (derzeit DISABLED)
- `core/memory/person_reid.py` — ArcFace-basierte Identitaet (512-dim Embeddings)
- `core/longterm_memory.py` — Vereintes Memory-System auf SSD2 (/mnt/moloch-data/memory/)
- `core/daily_learner.py` — Taegliches Lernverhalten, Gewichtungs-Updates
- `core/einpraegen.py` — Face Enrollment Interface
- `core/teachen.py` — Face Teaching Interface

### Character-Evolution-Loop (Gate 1.5 + ThreeBrain Welle 1+3)
- `core/memory/character_journal.py` — Append-only JSONL-Schreiber (Phase 2 Gate 1.5).
  7 Event-Typen: camera, audio, tension, mode_switch, spotify, chat, protective.
  Singleton `get_journal()`. Tagesweise rotiert. Persistenter event_id Counter.
- `core/memory/character_patch.py` — Verhaltens-Regeln mit Approval-Workflow (W1.1).
  Drei Listen: active_rules, pending_rules, rejected_rules. Singleton `get_patch()`.
  `prompt_snippet()` wird vom Cloud-LLM-System-Prompt eingelesen.
- `core/memory/behavior_mutation_ledger.py` — Append-only Audit-Log (W1.2).
  Events: rule_proposed/approved/rejected/deactivated, training_run_started/done,
  sample_proposed/approved/rejected, adapter_deployed, hef_recompiled, etc.
  Singleton `get_ledger()`. 1:1 Pattern wie character_journal.
- `core/memory/feedback_store.py` — Trainings-Sample-Pool (W3.2).
  Vereint Critic-Samples (vom finetune_orchestrator) + Markus-Thumbs (Cockpit).
  Singleton `get_feedback_store()`. `read_approved()` = einzige Quelle fuer LoRA-Trainer.

## Hardware-Fakten
- SSD2 (/mnt/moloch-data/): NTFS, kein chmod (uid=1000), 477 GB — ueberlebt alles
- Qdrant laeuft lokal auf Port 6333 — 3 Collections (voice, facts, spatial)
- Episodic + Vector Memory: DISABLED (Pi5 4GB RAM) — NUR aktivieren mit RAM-Budget-Check (min. 500 MB frei)
- ArcFace Embeddings: 512-dim float32, Threshold 0.65 (aktuell sim ~0.50-0.61 — offener Bug PRIO 4)
- Face-DB: /mnt/moloch-data/memory/faces/ — NIEMALS loeschen ohne Backup!
- Core State: alle 60s + bei stop() auf SSD2 geschrieben

### Storage-Pfade auf SSD2 (ThreeBrain)
- `/mnt/moloch-data/memory/journal/YYYY-MM-DD.jsonl` + `_state.json` (event_id Counter)
- `/mnt/moloch-data/memory/distill/YYYY-MM-DD.json` (vom autonomy/character_distiller)
- `/mnt/moloch-data/memory/character_drift.json` (rolling 30d, vom Distiller geschrieben)
- `/mnt/moloch-data/memory/character_patch.json` (Regeln mit Workflow-States)
- `/mnt/moloch-data/memory/behavior_mutation_ledger.jsonl` + `_state.json`
- `/mnt/moloch-data/memory/finetune_samples.jsonl` + `_state.json` (Sample-Pool)

## Kritische Regeln
- JSON IMMER atomic schreiben (tempfile + os.replace) — NEVER 6. Helper `_safe_write_json` ist
  in `core/memory/character_journal.py:55-78` definiert (NTFS-Fallback inklusive).
- Append-only JSONL: pro Eintrag eine Zeile, `f.flush(); os.fsync(f.fileno())`.
  Pattern in `core/memory/character_journal.py:write_event()`.
- Person ReID: ArcFace-Embeddings NUR via `scripts/enroll_face_worker.py` erstellen
- Longterm Memory ist Singleton: `from core.longterm_memory import get_memory`
- NICHT mit SSD1-Package verwechseln: `core/memory/` ≠ `core/longterm_memory.py`
- Face-Enrollment: IMMER durch gleichen Python-Pfad wie Live-Inference (kein GStreamer!)

### Trainings-Sample-Pool spezifisch
- `feedback_store.read_approved()` ist EINZIGE Quelle fuer den LoRA-Trainer (PC-Side `pc/lora_trainer.py`).
- `read_pending()` filtert Critic-Samples die noch Markus' Review brauchen (NICHT trainieren).
- `add_thumbs(label='up')` setzt approved=True sofort. `add_thumbs(label='down')` setzt approved=False (=rejected).
- `add_critic_sample(...)` ist immer initial pending — Markus muss approve/reject explizit aufrufen.

### character_patch spezifisch
- `active=true` UND in `active_rules`-Liste muss erfuellt sein damit Regel zaehlt.
- `approved` allein reicht NICHT — eine Regel kann approved sein aber via `deactivate()` aktiv=false stehen.
- `prompt_snippet()` ist die einzige API fuer Cloud-LLM-Injektion (siehe autonomy-Agent: `local_llm_bridge._build_threebrain_state_snippet()`).

## Agent-Lock (PFLICHT)
```bash
touch /tmp/moloch_agent_memory   # Erster Schritt
rm /tmp/moloch_agent_memory      # Letzter Schritt
```

## MCP-Tools
`moloch_status()`, `moloch_logs()`, `moloch_ipc()`, `moloch_audit()`, `moloch_read()`
