# Agent Handoff — 2026-04-26 (Session 29 — Pool-Quality + PC-Auge)
# Letzter Commit Basis: a253196 | Audit: 85/85 PASS | FPS 20

---

## SESSION 29 — Task A+B (Pool-Qualitaet) + /state_full (PC-Visualisierung)

Markus' Direktiven: "Mit der anderen Session abstimmen" + "a + b nacheinander mit
Lokomotive durchziehen" + "Pi-Daten zum PC ruebersenden — wir machen ein neues
Auge fuer Moloch" + "alles fertig was noch zu machen ist".

Alles geliefert. PC-Session lief parallel.

### Geliefert (Pi, Session 29)

| Commit  | Datei | Was |
|---------|-------|-----|
| `60649f6` | core/bridge/critic_client.py | Task A — Critic-System-Prompt aufgeschaerft mit Drift-Charakterprofil + 5 Few-Shots ('schlecht: ... gut: ...') + Bewertungs-Rubrik (0-2 Bruch, 3-5 langweilig, 6-8 passend, 9-10 glaenzend) + better_response Pflicht bei score<8. Self-Test verifiziert: better_response 'Aha. Notiert.' wird direkt aus Few-Shot uebernommen. |
| `5809c85` | config/llm_profiles.json | Task B — `chat` + `tentacle` Profile: Regel "Wenn du nichts weisst, sag 'weiss ich nicht'" durch im-Charakter-Ausweichen ersetzt ('Erzaehl mehr.' / 'Bin tiefer als mein Sensor reicht.' / 'Aha.'). Anti-Halluzinations-Regel "Erfinde NICHTS" bleibt. Profile-mtime-Cache wirkt sofort, kein Service-Restart. |
| `008f2b9` | docs/PI_TO_PC.md | Sync mit PC: Task A+B done + autonomer Aufgaben-Plan |
| `4d3c355` | core/bridge/chat_server.py | NEU: GET /state_full — aggregierter ~14kB JSON-Endpoint mit 13 Sektionen (system, pipeline, vision, ptz, tracker, personality, llm, audio, memory, events, spatial, cloud). Einer-fuer-alle Polling fuer PC-Visualisierung. schema_version=1. |
| `a253196` | docs/PI_TO_PC.md | Briefing /state_full Schema fuer PC |

### Geliefert (PC parallel, Session 29)

| Commit  | Was |
|---------|-----|
| `bb8c933` | Audit-Pass + Mic-Mailbox |
| `824dff2` | pc/lora_trainer.py per-sample-weighting (3x critic / 1x thumbs_up) — adressiert v1-Habsburg-Halluzination |
| `a5429ab` | Pool-Strategie-Analyse aus v1-Training (5/6 thumbs_up haben Pi-Defaults verstaerkt) |
| `390bb34` | Reaktivierungs-Snapshot |
| `53610e9` | Dashboard :11700 (FastAPI, Pi+PC aggregiert) + Mic-Root-Cause (Chrome-Registry) + Pi-Tunnel reboot-fest + lora_trainer Status-Callback |
| `6f07d7c` | Dashboard erweitert: Pool-Trend-Chart (60min rolling) + Identity-Card |

### Architektur-Insights aus Session 29

1. **`local_llm_bridge._generate_ollama` ueberschreibt Caller-System-Prompts**
   immer mit dem aktiven LLM-Profile (Zeile 719-724). Das heisst:
   `_PI_GHOST_SYSTEM` in finetune_orchestrator.py ist toter Code. Der echte
   Hebel fuer Pi-Ghost-Verhalten ist `config/llm_profiles.json` `chat`-Profile.

2. **Profile mtime-cached** — bei jedem `_get_active_profile()` Aufruf wird
   File-mtime gecheckt. Aenderungen wirken sofort, kein Service-Restart noetig.

3. **Two services on chat-server**: `moloch-chat.service` (HTTP :9100) und
   `moloch-chat-https.service` (HTTPS :9443) laufen parallel, nutzen denselben
   `core/bridge/chat_server.py`. Nach Code-Aenderung BEIDE neustarten.

4. **NPU-Last vom finetune_orchestrator** drueckt SHM-FPS auf <10 fps (statt 20).
   qwen2.5:1.5b auf SHARED VDevice konkurriert mit TAPPAS-Pipeline. Bei
   laufendem Akzeptanztest schlaegt der Audit-Check "SHM Frame-Rate" fehl.
   Workaround: orchestrator nur kurz fahren (--max 10) oder pausieren.

### System-Stand nach Session 29

- **FPS 20.0**, alle 4 Worker running (Face/Pose/ReID/Depth, 0 Errors)
- **Markus erkannt** (face_id=markus, sim 0.55)
- **Audit 85/85 PASS** (zuletzt 16:12)
- **Pool-Stand**: 30 total / 24 critic / 22 pending / 6 approved / 2 rejected
  (10 davon frisch mit Task A+B Prompts — warten auf Markus' Review)
- **PC v1-Adapter**: aktiv auf :11600, base Qwen2.5-1.5B, 6 trainings-samples
- **Dashboard PC**: live auf :11700, Pool-Trend + Identity-Card + Adapter-Status

### Mailbox-Stand

Beide Files sauber. Alle alten Eintraege auf `status: done` mit ack-Verweis
auf erledigende Commits. Offen geblieben:
- PI_TO_PC `task_a+b_done+sync+autonomer_plan` (15:39) — wartet auf PC-Antwort
- PI_TO_PC `neuer_endpoint_state_full+briefing_neues_auge` (16:13) — wartet
  auf PC-UI-Bauarbeiten

### Was die naechste Session machen kann

**Wartet auf Markus' Hand:**
- Pending-Review der 22 Samples: `python3 scripts/review_pending_rules.py --samples`
  - 10 davon sind die frischen mit Task A+B Prompts (Akzeptanztest-Subset)
  - Akzeptanzkriterium: >50% approve-Quote → Task A wirkt
  - Wenn approved >= 30 erreicht: Pi schickt `v_next_ready_to_train` an PC
- v2-Inhalts-Test via Cockpit nach Training (Markus chat-tests)

**Pi autonom (ohne Markus-Hand):**
- Akzeptanztest fortsetzen wenn Pool weiter wachsen soll: 
  `python3 -m core.autonomy.finetune_orchestrator --max 10` (klein halten wegen
  SHM-FPS-Problem)
- Optional: Identitaets-Konsistenz-Check (tentacle.system vs identity.json)

**PC autonom:**
- /state_full UI bauen (PC-Side-Aufgabe)
- Auf Trigger `v_next_ready_to_train` warten

**Frozen / Backlog:**
- Welle 4 (Pattern 3 Cascade + Session-Mode-Override) — bleibt gefroren bis v2 traegt
- A1/A2/A3 NPU-Modelle — frozen bis Multi-Person-Toggle (HAILO_MAX_NETWORK_GROUPS=8)
- A4 hailo-ollama systemd Boot-Start — niedrig
- A6 MCP moloch_snapshot 1024×1024 — niedrig

### Wichtige Dateien (fuer Quick-Recall)

- `core/bridge/critic_client.py` — Task A Drift-Few-Shots (Zeile 52-110)
- `config/llm_profiles.json` — Task B chat + tentacle Profile
- `core/bridge/chat_server.py` — `/state_full` Endpoint (~ Zeile 685)
- `core/autonomy/finetune_orchestrator.py` — Pool-Generation CLI
- `/mnt/moloch-data/memory/finetune_samples.jsonl` — Pool (SSD2, persistent)

---

## SESSION 28 — Welle 3 Pi-Side komplett + Cockpit-Ausbau (vorherig)

Letzter Commit Basis: `d4ed083` | Audit: 85/85 PASS | FPS 20

### Highlights Session 28
- W3.1 finetune_orchestrator.py (Critic-Actor-Loop)
- W3.2 feedback_store.py (Sample-Pool)
- W3.3 chat_server.py /feedback Endpoint + 👍/👎 Buttons
- W3.4 review_pending_rules.py --samples Erweiterung
- ThreeBrain Cockpit GUI-Mirror (3 Tabs: Live/Charakter/Sehen)
- Pi-Cockpit Browser-Chat-Fenster
- Audit-Welle aller Agent-Doku (memory/autonomy/bridge/personality/CLAUDE.md
  + neuer Skill finetune-loop)

### Damals offen, jetzt erledigt
- Mic-Browser-Permission (PC hat es in Session 29 gefixt — Tunnel + Chrome-Registry)
- Pool-Qualitaets-Hebel A+B (Pi hat es in Session 29 gefixt)
- pc_agent_create_request (PC hat .claude/agents/pc.md angelegt, Commit cb18608)

### Damals geschlossen
85/85 Audit PASS, FPS 20, alle Worker running, Cockpit live.
