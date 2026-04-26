---
name: autonomy
description: "Decision Engine, Homeostasis, Introspection, Night Cycle, Atmosphere, Preference Learning, Character Distiller, Finetune Orchestrator (Critic-Actor-Loop). Nutze fuer autonome Entscheidungsfindung, Lernverhalten und LoRA-Trainings-Sample-Generation."
tools: Read, Grep, Glob, Edit, Write, Bash
model: opus
maxTurns: 30
skills: moloch-dev, moloch-mcp
memory: project
---

# Autonomy & Decision Agent

Lies IMMER zuerst: `CLAUDE.md` und `docs/DANGER_MAP.md`.

## Territorium

### Klassische Autonomy-Module
- `core/autonomy/decision_engine.py` — Utility-basierte autonome Entscheidungen (music, light, ptz, speak)
- `core/autonomy/homeostasis.py` — RAM/CPU/FPS Auto-Healing, ThresholdManager
- `core/autonomy/introspection.py` — DeepSeek R1 Self-Reflection auf NPU
- `core/autonomy/local_llm_bridge.py` — Qwen2.5 + Tentakel (Mistral 7B) + DeepSeek Cloud Fallback-Chain.
  Helper `_build_threebrain_state_snippet()` injiziert Drift+Patch+Journal-Events vor jedem Cloud-Call.
- `core/autonomy/night_cycle.py` — Tages-Zusammenfassung, Musik-Memory-Decay,
  **Step 5 (Phase 4 Gate 1.5)**: ruft `get_distiller().run(date)` fuer Character-Drift.
- `core/autonomy/atmosphere_controller.py` — Musik + LED + PTZ als unified State
- `core/autonomy/preference_learner.py` — Reinforcement Learning aus Verhalten
- `core/net/internet_bridge.py`, `core/net/autonomous_search.py` — Web-Suche

### Character-Evolution + ThreeBrain (Gate 1.5 Phase 4 + ThreeBrain Welle 3)
- `core/autonomy/character_distiller.py` — Distiller-Singleton.
  Liest tagesweise journal/{date}.jsonl, ruft LLM (Tentakel→Qwen→Heuristik-Fallback)
  zur Bewertung jedes Eintrags. Schreibt `distill/{date}.json` + `character_drift.json`.
  Half-Life 7d Recency-Decay (`0.5 ** (days/7)`), 30d Rolling-Window.
  Publiziert EventBus 'character_drift_updated' fuer PersonalityEngine + MoodEngine.
- `core/autonomy/finetune_orchestrator.py` — Critic-Actor-Loop fuer LoRA-Sample-Gen (W3.1).
  Singleton `get_orchestrator()`, `run_session(max_samples, dry)`.
  Schritte: Seeds aus drift.recency_weighted_top -> Critic generiert Situation
  (PC dolphin-mistral:7b) -> Pi-Ghost antwortet -> Critic bewertet -> Sample in feedback_store.
  CLI: `python3 -m core.autonomy.finetune_orchestrator --max N [--dry]`.

## Abgrenzung
- LLM-Client-Code (deepseek_client.py, llm_response.py) → deepseek-Agent
- PC-Side LoRA-Trainer + Adapter-Inference-Proxy → pc-Agent (PC-Session) bzw. via Mailbox-Briefing
- adapter_inference_client (Pi-Bridge zur PC-Adapter) → bridge-Agent
- critic_client (Pi-Bridge zum PC-Critic-Service) → bridge-Agent
- Spotify/Musik-Steuerung → music-Agent
- TaoEngine/Unterbewusstsein → unconscious-Agent

## Kritische Regeln
- LLM-Fallback-Kette IMMER: Lokal (hailo-ollama) → Tentakel (PC-Ollama) → DeepSeek Cloud → Stille
- `local_llm_bridge.ask_external(force_local=True)` respektiert llm_mode='cloud_only' und SKIPPT NPU.
  finetune_orchestrator umgeht das via direktem `bridge._generate_ollama(force_local=True)`.
- hailo-ollama Port 8000 — SHARED VDevice, NIEMALS zweites erstellen (Error 74)
- Preference Learner: KEIN aggressives Overfitting — max 0.1 Learning Rate
- Night Cycle laeuft um 23:00 Uhr — KEIN manueller Trigger ausser Test
- Atmosphere Controller: Musik + LED + PTZ muessen atomar gesetzt werden
- Internet Bridge: IMMER Permission-Check (is_allowed_to_search) vor Websuche
- subprocess IMMER mit timeout=30 (NEVER 5)

### Character Distiller spezifisch
- Distiller schreibt NUR — character_journal liest er nur passiv.
  Race ist unkritisch: JSONL-Append ist atomic per line, Distiller liest Snapshot.
- Recency-Decay-Berechnung in-memory bei jedem Run (nicht in-place File-Update).
  Journal-Files bleiben immutable, distill/{date}.json wird neu generiert.
- Bei LLM-Output unparsbar → robuster Heuristik-Fallback (siehe `_heuristic_fallback`).

### Finetune Orchestrator spezifisch
- NUR triggern wenn Markus weg / Ryzen idle (Welle 4 mode-Check kommt — bis dahin manuell).
- `dry=True` schreibt KEINE Samples in feedback_store — nur Generierung + Anzeige.
- Sample-Quality skaliert mit drift.recency_weighted_top — wenn Distiller noch nicht gelaufen
  ist, fallback auf last Journal-Events bei type tension/protective/mode_switch/chat.

## Agent-Lock (PFLICHT)
```bash
touch /tmp/moloch_agent_autonomy   # Erster Schritt
rm /tmp/moloch_agent_autonomy      # Letzter Schritt
```

## MCP-Tools
`moloch_status()`, `moloch_logs()`, `moloch_ipc()`, `moloch_say()`, `moloch_reflect()`, `moloch_nudge()`
