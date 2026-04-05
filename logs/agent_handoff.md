# Agent Handoff — 2026-04-05
# Session: Claude Sonnet 4.6 (Agenten-Overhaul + Workflow-Audit + MCP-Plan)
# Branch: main
# GitHub: moloch00464-bit/MOLOCH
# Status: AUDIT PASS 54/54 ✅ | Git sauber | 16 Agenten aktiv

---

## SYSTEM-BASELINE (aktuell)

- FPS: 20.1 | CPU: 46.3°C | RAM: 43.7%
- Worker: 7/7 running, 0 Errors
- Audit: 54/54 PASS
- Zone: IDLE | Tracker: tracking
- Git: main, sauber, synchron mit GitHub

---

## WAS DIESE SESSION GEMACHT HAT

### 1. Hook Domain-Mapping: alle 16 Agenten (Commit ba679b1)
- pre-edit-check.sh: 9 neue Domains eingetragen (autonomy, awareness, personality,
  memory, watchdog, music, deepseek, tentacle, unconscious)
- Verzeichnis-Pattern personality/* → personality (war fälschlich voice)
- Datei-Zuordnungen bereinigt: spotify→music, wifi_mic→tentacle, longterm_memory→memory
- Fehlermeldung listet jetzt alle 16 Agenten

### 2. GUI-Agent: BBox + Landmark Rendering (Commit ba679b1)
- gui.md + AGENT_GUI.md: BBox/Landmark-Darstellung als GUI-Territorium
- Letterbox-Warnung: keine Doppelkorrektur
- Audit-Checkliste für BBoxen/Landmarks

### 3. moloch-agent Skill: alle 16 Agenten (Commit b6d22e9)
- Agent-Mapping auf 16 Agenten erweitert
- Territorium-Tabelle korrigiert: personality/memory aus voice/service raus
- BBox-Anzeige + Landmarks im GUI-Eintrag

### 4. Alle 16 Agenten-Prompts überarbeitet (Commit 2151d50)
- Territorium-Konflikte gelöst (voice/service/hardware/autonomy)
- unconscious: MCP-Tools hinzugefügt (fehlten komplett)
- stresstest: Erfolgskriterien + moloch_audit() in Tools
- Descriptions präziser mit Abgrenzungshinweisen

### 5. Stresstest E2E Display-Chain + MCP Konnektivität (Commit bc627e6)
- stresstest.md: Backend-Event → IPC → Status-JSON → Panel-Anzeige
- 7 Trigger-Tests mit Akzeptanzkriterien
- MCP-Konnektivitätstest: alle 17 Tools mit Erwartungswerten

### 6. Workflow-Audit durchgeführt
- Alle Komponenten geprüft: Session-Start, Hooks, Skills, Agenten, MCP
- Befund: 95% OK — 3 kleine Fixes nötig (→ diese Session erledigt)
- pre-bash-check.sh: jq → python3 (jq nicht auf Pi!)

### 7. MCP-Bidirektional-Plan geschrieben
- Plan: .claude/plans/delightful-hugging-mitten.md
- Moloch's NPU-LLM direkt via MCP befragbar (moloch_llm_query)
- Moloch kann Claude fragen via MCP Sampling (moloch_ask_claude)
- hailo-ollama systemd-Service ebenfalls geplant

---

## PFLICHT-STARTPROTOKOLL (Phase 1 — JEDE Session)

1. `logs/agent_handoff.md` lesen
2. `git status` — uncommitted changes?
3. `moloch_status()` — löst /tmp/moloch_session_lock
4. `moloch_npu_workers()` — Worker-Health
5. `moloch_audit()` — Baseline PASS/FAIL
6. Markus zeigen: "Service läuft, X FPS, Audit X/54"

---

## WORKFLOW (Kurzfassung)

```
Auftrag → /moloch-agent Skill → richtigen Agenten spawnen
         → touch /tmp/moloch_agent_[name]
         → Code editieren (Hook prüft Domain!)
         → sudo systemctl restart moloch
         → moloch_audit() → bei FAIL: STOPP
         → rm /tmp/moloch_agent_[name]
         → git commit (1 Datei = 1 Commit)
         → git push
```

---

## 16 AGENTEN (vollständige Liste)

| Agent | Territorium |
|-------|-------------|
| vision | core/perception/, tappas_pipeline, NPU, BBox-Inferenz |
| hardware | core/hardware/, camera.py, LED, Thermal |
| gui | core/gui/, panel_*.py, popups/, BBox-Zeichnen, Landmarks |
| tracking | core/mpo/, ptz_arbiter, ptz_tracker |
| voice | core/speech/, core/tts/, core/audio/, voice_pipeline |
| service | moloch_service, core_integrator, ipc_router |
| unconscious | unconscious_engine.py, TaoEngine, anima_mappings |
| autonomy | core/autonomy/ (decision, homeostasis, introspection, llm_bridge, night_cycle, atmosphere, preference_learner) |
| awareness | core/awareness/ (activity, context, motion, room_map), world_state |
| personality | core/personality/ (engine, mood, behavior, tension), event_bus, sprache, timeline |
| memory | core/memory/ (episodic, persistent, vector, reid), longterm_memory, daily_learner |
| watchdog | system_watchdog, diagnostics, capability_monitor, status |
| music | core/music/, spotify_controller |
| deepseek | local_llm_bridge, deepseek_client, llm_response, introspection |
| tentacle | wifi_mic, camera_cloud_bridge, firmware/respeaker_wifi_mic/ |
| stresstest | scripts/, Tests, Chaos Engineering, E2E Display-Chain |

---

## OFFENE AUFGABEN (priorisiert)

PRIO 1 — MCP Bidirektional (NÄCHSTE SESSION):
  - Plan: .claude/plans/delightful-hugging-mitten.md
  - 5 Commits: moloch_llm_query + moloch_llm_status → moloch_ask_claude (async Sampling)
    → IPC-Handler in moloch_service.py → hailo-ollama systemd → stresstest E2E
  - Agent: service
  - hailo-ollama API: /api/chat (NICHT /v1/chat/completions — gibt 404!)

PRIO 2 — TaoEngine implementieren: ✅ PRIO VERSCHOBEN (nach MCP-Plan)
  - Plan: docs/plans/tao_engine_plan.md (von Opus 4.6 erstellt)
  - Agent: unconscious

PRIO 3 — Tracker STUCK-AT-LIMIT:
  - core/mpo/autonomous_tracker.py → _track_tracking_target()
  - Nach 8s am mechanischen Anschlag → SEARCH starten
  - Agent: tracking

PRIO 4 — ArcFace Similarity 0.50-0.61 (Threshold 0.65):
  - Neu-Enrollment: scripts/enroll_face_worker.py
  - Agent: vision

---

## REFERENZ-DATEIEN

| Datei | Inhalt |
|-------|--------|
| CLAUDE.md | Master-Regeln, 12 NEVER, Datei-Ampel, 16 Agenten |
| .claude/plans/delightful-hugging-mitten.md | MCP-Bidirektional Implementierungsplan |
| docs/DANGER_MAP.md | ROT/GELB/GRUEN für alle Dateien mit NEVER-DOs |
| logs/system_contract_pipeline.md | 10 Pipeline-Sync-Regeln (bei Pipeline-Arbeit) |
| docs/plans/tao_engine_plan.md | TaoEngine Implementierungsplan (Opus 4.6) |
| .claude/hooks/pre-edit-check.sh | Agent-Lock + Domain-Check Hook (16 Agenten) |
| .claude/hooks/pre-bash-check.sh | Bash-Check Hook (python3, kein jq!) |
| .claude/agents/*.md | 16 Agenten-Definitionen |
