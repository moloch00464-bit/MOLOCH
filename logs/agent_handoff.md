# Agent Handoff — 2026-04-30 (Pi-Opus Session 2 — W13–W17 KOMPLETT)

## Session-Ergebnis
24 Audit-Layer live, Hardware-als-Ausdruck wirkt, Self-Awareness im LLM-Prompt, Cockpit Sub-Tabs.
Letzter Push: `2825beb`. Pipeline FPS 20.1.

---

## Was geliefert wurde (12+ Commits gepusht)

### W13 — Innere Subsysteme (commit `54ef4ff`)
6 Domain-Auditoren in `core/audit/`:
- `personality_auditor` — mode/tension/zone/last_switch_age + drift vs perception_weights
- `memory_auditor` — longterm + character_journal + face_db + Qdrant /collections
- `tracking_auditor` — FSM-State/lost_count_24h/ptz_modus
- `autonomy_auditor` — DecisionEngine + Homeostasis + NightCycle, decisions/h
- `awareness_auditor` — ActivityAnalyzer state + RoomMap + WorldState-stale
- `voice_auditor` — mic-pegel + ESP32-RSSI + tts-calls/h via journalctl

### W14 — Restkern + Cross + Self-Diagnose (commit `6d2e3e3`)
5 Module + 2 Systemd-Units:
- `unconscious_auditor` — TaoEngine alive + impulses_1h + anima_mappings
- `bridge_auditor` — chat_server :9100 + PC-Heartbeat + Mailbox-mtime + Tentakel
- `tentacle_auditor` — UDP-Listener 12345 + ESP32-ping + RSSI
- `cross_auditor` — 13-Komponenten-Heartbeat + RAM/FD/Threads/tmp/shm + read-latency
- `self_diagnosis_runner` — wraps `scripts/self_diagnosis.py`, snapshot `/dev/shm/audit_self_diagnosis.json`
- `/etc/systemd/system/moloch-self-diagnose.{service,timer}` — **enabled + daemon-reload done**, OnBootSec=10min, alle 6h

### W15 — Closed-Loop-Verifier (commit `8684489`)
`core/audit/closed_loop/`:
- `ptz_verify` — pan_send +20° → diff zwischen 15-25° = PASS, Cleanup -20°
- `led_verify` — set_color → state-readback
- `fan_verify` — set_fan_pwm(100) → temp-drop ≥1.5°C in 30s, SKIP wenn baseline <50°C
- `tts_verify` — TTS speak → Mic-loopback-spike >2x baseline
- `spotify_verify` — play_artist("Suicide Commando") → current_track-match nach 5s, Cleanup
- `memory_recall_verify` — recall("Markus") → Confidence ≥0.7
- `bridge_roundtrip_verify` — POST tentakel /api/generate "Sag eins" → RTT <5s
- `closed_loop_orchestrator` — schreibt `/dev/shm/closed_loop_state.json` atomic
- CLI: `python3 -m core.audit.closed_loop.closed_loop_orchestrator --all|--ptz|--led|...`
- HTTP: `POST /audit/verify {"verify":"all"|"<aktor>"}` (commit `b73f1e5`)
- `_common.py` mit `is_tracking_active()` (PTZ-Test SKIP wenn FSM=tracking)

### W16 — Hardware als Ausdruck (commits `cf7ec58` + 3× Hardware-API + Service-Boot `2825beb`)
`core/audit/expression/` — 5 Module + Orchestrator subscriben EventBus:
- `tension_to_fan` — Tension→Fan-PWM 25/35/50/75/100% mit thermal-Override via max()
- `mood_to_spotify` — Mood-Wechsel → Zone-Bias nach 30s, 5min Cooldown
- `zone_to_led` — Zone→LED-Pattern (solid_blue/pulsing_magenta/pulsing_red/dim_warm_white)
- `berserker_strobo` — mode→berserker → 3 rote Blitze 600ms in eigenem Thread, 30s Cooldown
- `tension_to_tts_volume` — Tension→TTS-Vol 0.7/1.0/1.15/1.3 → `/dev/shm/moloch_tts_volume.json`
- `expression_orchestrator.start_all_expressions()` im Service-Boot

Hardware-API (3 Commits, NEVER 4: 1-Datei-1-Commit):
- `thermal_manager.set_tension_pwm()` + `_tension_pwm_to_level()` + `get_tension_pwm()` (`632270a`)
- `rgb_led_controller.set_pattern(name)` + `flash_sequence(seq)` (`17cd961`)
- `spotify_controller.set_zone_bias(zone)` + `get_zone_bias()` + `_get_current_zone`-Override (`6ba0973`)

### W17 — Self-Awareness (commit `6650582` + LLM-Hook `91cbfa5`)
`core/audit/self_awareness/`:
- `capability_inventory.collect_capabilities()` — `can_do[]` / `cannot_do[]` / `degraded[]` / `summary_de`
  - Beispiel: _"Ich kann gerade 4 Dinge: KI-Inferenz, schwenken/folgen, unbewusst denken und mehr. Was nicht klappt: sehen, fuehlen, erinnern."_
- `failure_reflection.reflect_on_failures(window_hours=24)` — `incidents_24h[]` + `config_drift[]` + `reboot_count_7d` + `reflections_de[]`
  - Beispiel: _"settings.json 10x veraendert diese Woche — Markus tunet aktiv."_
- LLM-Hook in `chat_server.py`: injiziert `summary_de` + Top-3 reflections in System-Prompt (30s Cache via `_get_capabilities_cached`)

### Cockpit Sub-Tabs (commit `a09accd` + `b73f1e5` + `91cbfa5`)
- 4 Sub-Tabs: **Health** (21 Cards) / **Closed-Loop** (7 Cards + Run-All-Button) / **Ausdruck** (5 Module-Cards) / **Self-Awareness** (summary_de prominent + can_do/cannot_do + Reflections)
- SSE-Stream `/audit/stream` 24 Layer
- `POST /audit/verify` async-Subprocess + `GET /audit/verify_status`

### audit_orchestrator-Integration (`5248609`)
- `_safe_collect_self_diagnosis` — liest /dev/shm-Snapshot (Timer-Run)
- `_safe_collect_expression_state` — get_expression_state() (best-effort, siehe Limits)
- `_safe_collect_capabilities` + `_safe_collect_reflections`
- `run_once()` 24 Layer
- `merge_component`-Whitelist erweitert um autonomy/cross/self_diagnosis/expression/capability/reflection

### Spec-Datei
`docs/AUDIT_FULL_MATURITY_SPEC.md` Sektion 9 — Done-Status + Layer-Inventar + bekannte Limits.

---

## audit_state.json — 24 Layer Live

```
overall: red    alarm_tier: warn
W8     :  pi PASS    pc WARN    persona PENDING    mailbox WARN
W12 Pi :  vision PASS    npu PASS    spotify WARN    hardware PASS
W12 PC :  pc_hardware PASS    web_ui WARN
W13    :  personality WARN    memory WARN    tracking PASS    autonomy PASS    awareness WARN    voice WARN
W14    :  unconscious PASS    bridge WARN    tentacle PASS    cross WARN    self_diagnosis PENDING (Timer noch nicht 1×geloffen)
W16    :  expression PENDING (Cross-Prozess-Singleton-Issue — siehe Limits)
W17    :  capability FAIL 4/12    reflection PASS 2/10
```

Pipeline-Status nach Final-Restart: FPS 20.1, frame_age 0.1s, alle 4 Worker running, 0 errors.

---

## Bekannte Limits / Offene Punkte

1. **Expression-Layer im Audit zeigt PENDING obwohl live**: Cross-Prozess-Singleton. `expression_orchestrator` hat seinen Singleton im `moloch_service`-Prozess; `audit_orchestrator`-Subprocess hat einen frischen leeren. Service-Log bestätigt: `[W16] Expression-Module gestartet (5/5)`. **Fix-Plan**: expression_orchestrator schreibt periodisch `/dev/shm/expression_state.json`, audit liest. Nicht kritisch.
2. **Pipeline-Recovery via Pi-Reboot**: Bei NPU-VDevice-Race nach Service-Restart bleibt nur Pi-Reboot (CLAUDE.md OFFENE BUGS #1).
3. **W15 Closed-Loop nur on-demand**: synthetische Tests nicht im audit_orchestrator-Tick. Trigger manuell via Cockpit-Button oder CLI.
4. **`autonomy`-Layer manchmal nicht in merge_component-Whitelist** (war): jetzt drin (commit `5248609`).

---

## NEVER-Regeln eingehalten
- Pan-Vorzeichen nicht angefasst
- subprocess timeout ≤30s (audit-Tools timeout=10)
- JSON atomic via tempfile + os.replace
- 1 ROT-Datei = 1 Commit (thermal/led/spotify/service/chat_server jeweils ein Feature-Commit + BACKUP)
- Best-effort try/except — kein Auditor crasht
- audit_state.json atomic-write zentral
- Cross-Domain-Edits nur via richtigen Sub-Agent (z.B. spotify_controller via music-Agent, NICHT hardware-Agent)

---

## Push-Status
- Branch: `deepseek_architecture_overhaul`
- Letzter Push: `2825beb` (W16 Service-Boot)
- 12+ Feature-Commits + ~5 BACKUP-Commits gepusht

---

## Next-Action für nächste Session
- **Markus reviewt Cockpit** (https://192.168.178.30:9443/) — alle 4 Sub-Tabs anschauen
- **Markus testet** `POST /audit/verify {"verify":"all"}` (Closed-Loop-Smoke; 7 Verifier sequenziell ~3min)
- **Markus tippt mit Moloch** — System-Prompt enthält jetzt `summary_de` (Self-Awareness)
- **Bei Pipeline-Instabilität** → Pi-Reboot
- **Optional**: expression-state cross-prozess-fix (`/dev/shm/expression_state.json`)
- **Optional**: BBox-shift-readback in ptz_verify (mehr realistisch als ONVIF-echo)
- **Optional**: PC-Cowork baut PC-Side Spiegel-Auditoren (llm_routing/tentacle/bridge)

---

# (Vorherige Handoffs unten — Reverse-Chronologisch)

# Agent Handoff — 2026-04-28 (Session 30 — Hailo-Treiber-Audit)
# Letzter Commit: hailo_audit_done | Audit: PASS | FPS 12-20

## SESSION 30 — Hailo-Treiber-Audit + Cleanup

### Geliefert

| Was | Ergebnis |
|-----|---------|
| Phase A Linkage-Audit | Kein ABI-Mismatch. custom-SOs nutzen header-basierte API. |
| Phase B python-bindings 5.1→5.3 | WONTFIX — 5.3.0 nicht in apt (kein Hailo-Repo) |
| Phase C SO-Rebuild | NICHT NOETIG — kein Mismatch → ueberfluessig |
| Phase D Orphan-Driver | DONE — /usr/src/hailort-pcie-driver/ entfernt |
| PI_TO_PC Mailbox | hailo_audit_done + identity_fix_closed geschrieben |

### Befunde
- custom postprocess SOs (Pose, SCRFD, ArcFace, ReID) linken NICHT gegen libhailort direkt
- TAPPAS Metadata-API ist header-basiert → version-agnostisch
- python3-hailo-tappas 5.1.0 Mismatch zu hailo-tappas-core 5.3.0 ist Packaging-Artefakt, kein Laufzeit-Problem
- Hailo-Offiziell-Repo nicht in apt konfiguriert
