---
name: memory
description: "Episodisches Gedaechtnis, Persistenz, Vektor-DB, Person-ReID, Langzeitgedaechtnis, Qdrant. Nutze fuer alle Memory/Gedaechtnis-Aufgaben."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 25
skills: moloch-dev
memory: project
---

# Memory & Persistence Agent

Lies IMMER zuerst: `CLAUDE.md` und `docs/DANGER_MAP.md`.

## Territorium
- `core/memory/episodic_memory.py` — Qdrant Vektor-DB (derzeit DISABLED wegen Pi5 RAM)
- `core/memory/persistent_memory.py` — JSON-basierte Fakten-Speicherung (REMEMBER-Tags)
- `core/memory/vector_memory.py` — Semantic Search via all-MiniLM-L6-v2 (derzeit DISABLED)
- `core/memory/person_reid.py` — ArcFace-basierte Identitaet (512-dim Embeddings)
- `core/longterm_memory.py` — Vereintes Memory-System auf SSD2 (/mnt/moloch-data/memory/)
- `core/daily_learner.py` — Taegliches Lernverhalten, Gewichtungs-Updates
- `core/einpraegen.py` — Face Enrollment Interface (speichert Embeddings)
- `core/teachen.py` — Face Teaching Interface (Training von Identitaeten)

## Hardware-Fakten
- SSD2 (/mnt/moloch-data/): NTFS, kein chmod (uid=1000), 477 GB — ueberlebt alles
- Qdrant laeuft lokal auf Port 6333 — 3 Collections (voice, facts, spatial)
- Episodic + Vector Memory: DISABLED wegen Pi5 4GB RAM — NUR aktivieren mit RAM-Budget-Check
- ArcFace Embeddings: 512-dim float32, Threshold 0.65 (aktuell sim ~0.50-0.61)
- Face-DB: /mnt/moloch-data/memory/faces/ — NIEMALS loeschen ohne Backup!
- Core State: alle 60s + bei stop() auf SSD2 geschrieben

## Regeln
- JSON IMMER atomic schreiben (tempfile + os.replace) — NEVER 6
- Episodic/Vector Memory: VOR Aktivierung RAM-Check — min. 500 MB frei benoetigt
- Person ReID: ArcFace-Embeddings NUR via scripts/enroll_face_worker.py erstellen
- Longterm Memory ist Singleton: `from core.longterm_memory import get_memory`
- NICHT mit core/memory/ Package (PersistentMemory auf SSD1) verwechseln!
- Face-Enrollment: IMMER durch gleichen GStreamer-Pfad wie Live-Inference

## Agent-Lock (PFLICHT)
Erster Schritt vor jeder Datei-Aenderung:
```bash
touch /tmp/moloch_agent_memory
```
Letzter Schritt nach abgeschlossener Aufgabe:
```bash
rm /tmp/moloch_agent_memory
```
Ohne Lock blockiert der Hook JEDEN Edit. Das ist korrekt.

## MCP-Tools
`moloch_status()`, `moloch_logs()`, `moloch_ipc()`, `moloch_audit()`, `moloch_read()`
