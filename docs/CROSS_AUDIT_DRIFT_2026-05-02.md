# Cross-Audit-Drift PC <-> Pi — 2026-05-02

Konsolidierte Liste der Audit-Drifts zwischen PC-Cowork (`pc/moloch_health_check.py`, `pc/web_pipeline_auditor.py`) und Pi-Opus (`core/audit/audit_orchestrator.py` mit 27 Layern).

Quelle: PC-Topic `discuss_cross_audit_drift_pc_pi` (10:19) + Pi-`audit_state.json` Pi-Sicht.

---

## Drifts identifiziert (7 total)

### PC-zuerst-entdeckt (4)

| # | Drift | PC sieht | Pi sieht | Ursache | Fix |
|---|-------|----------|----------|---------|-----|
| 1 | `last_provider` | `none` (WARN) | `lokal_qwen2.5` | Race in `pc/moloch_health_check.py:122` — liest `/status` waehrend chat_server noch nicht geschrieben | PC-Side: Retry mit Backoff oder `cached_last_provider`-Fallback |
| 2 | `request_count` | 16 (search_proxy `/stats`) | 1 (chat_server `/status`) | chat_server-Counter war in-process, Reset bei Restart | **Pi**: persistent via `/dev/shm/chat_server_counters.json` ✅ DONE (`00f0dea`) |
| 3 | Pre-warmed Models | 1/3 in Cache (WARN) | unbekannt | tentacle-probe testet evtl. nur 1 Modell | PC-Side: Probe-Liste auf alle 3 erweitern (dolphin-llama3, dolphin-mistral, moloch-coder) |
| 4 | Aeltester open-Topic | 81min (`plan_welle21`) | — | PC zaehlt `plan_*`-Topics als offene Tasks | PC-Side: `pc/moloch_health_check.py` Filter `topic.startswith('plan_') -> ignore` |

### Pi-zuerst-entdeckt (3)

| # | Drift | Pi sieht (audit_state) | PC blind weil | Fix |
|---|-------|------------------------|---------------|-----|
| 5 | Audit-Aufloesung | 27 Layer im audit_state | PC's web_pipeline_auditor ist 4-Layer | **Pi**: GET `/audit/transition` Endpoint exposed (`5e596d9`) — PC kann nun direkt lesen |
| 6 | Schema-Diversitaet | Layer haben verschiedene Felder (`tools_pass`, `score/max`, `alive_count`) | PC erwartet uniformes Schema | Akzeptieren — pro Layer-Typ eigenes Schema ist Feature, nicht Bug |
| 7 | `federation_heartbeat` Kanal | Pi `transition`-Layer zeigt `cross_session.jsonl missing` | PC sieht keine Pi-Federation-Sicht | Strukturell — `cross_session_monitor` schreibt File NUR auf PC-Side, Pi-side fehlt. Folge-Issue: Pi-side Log-Mount oder Symlink |

---

## Pi-Sicht: 12 WARN/PENDING-Layer die PC nicht direkt sieht

Vollstaendige Liste aus `audit_state.json` (Pi-Sicht):

```
WARN     awareness          2/4   ActivityAnalyzer state stale (kein recent activity_change)
WARN     bridge             3/4   chat_server alive, Tentakel/Mailbox-Latenz drueckt 1 Punkt
WARN     capability         6/12  Capability-Inventory haelt 6 Faehigkeiten als degraded
WARN     cross              4/5   Resource-Pressure-Flag (FD/Threads OK, ein Schwellwert beruehrt)
WARN     mailbox            4/4   Stale-Topics > 24h trotz Hygiene-Closes
WARN     memory             4/4   Qdrant-Collection moloch_memory fehlt
WARN     personality        3/4   Drift gegen Baseline minimal
WARN     reflection         7/10  Reflexionen detected aber Score subkritisch
WARN     self_diagnosis     3/4   Pytest-Suite + Qdrant-Collection (Folge von memory)
WARN     spotify            2/4   Token-Status oder ipc_actions/responses-mismatch
WARN     transition         6/7   federation_heartbeat (siehe Drift 7)
WARN     voice              3/4   Mic-Pegel oder ESP32-RSSI grenzwertig
PENDING  persona            -/-   character_journal hat noch keine persona_score-Events
PENDING  web_search         0/0   PC's web_pipeline_auditor postet nicht regelmaessig
```

### Pi PASS Layer (13)
agent_tools, autonomy, expression, hardware, npu, pc, pc_hardware, pi, tentacle, tracking, unconscious, vision, web_ui

### Auffaellig
- `web_search` Layer ist PENDING — PC's web_pipeline_auditor postet seine Auditor-Ergebnisse via POST `/mailbox/audit/web_search` nicht regelmaessig (oder Whitelist greift nicht). Folge-Issue.
- `persona` PENDING — character_journal-Events fehlen (Welle 10 W-Hook noch nicht aktiv produziert)
- `mailbox` WARN trotz Hygiene-Closes — vermutlich `stale_count` >= Schwellwert weil viele alte open-Topics gleichen Topic-Namen wie neuere Hygiene-Closes

---

## Akzeptanzkriterien fuer Drift-Resolution

| Drift | Wer | Status |
|-------|-----|--------|
| 1 last_provider | PC | offen |
| 2 request_count | Pi | ✅ DONE (`00f0dea`) — persistent counter |
| 3 Pre-warmed Models | PC | offen |
| 4 Aeltester open-Topic | PC | offen |
| 5 Audit-Aufloesung | Pi | ✅ DONE (`5e596d9`) — `/audit/transition` exposed |
| 6 Schema-Diversitaet | beide | akzeptiert (Feature) |
| 7 federation_heartbeat | beide | offen — strukturell, separate Welle |

---

## Live-Werte 2026-05-02 (nach Pi-Restart 11:30)

```
Pi audit_state.json: 27 Layer, overall=warn, alarm=silent
- 13 PASS / 12 WARN / 2 PENDING / 0 FAIL
- transition: 6/7 alive (federation fehlt)
- agent_tools: PASS 4/4, roundtrip_via_http PASS 4.5ms
- bridge_full_roundtrip: PASS 4/4 (chat -> kaskade -> memory-save end-to-end)

PC web_pipeline_auditor: 4 Layer
- L1 health: PASS
- L2 stats: PASS
- L3 last_provider: WARN (Drift 1)
- L4 oldest open-task: WARN 81min (Drift 4 — false-positive plan_*)

PC moloch_health_check: 6 Layer (geschätzt aus Mailbox)
- L6 pre-warmed-models: WARN 1/3 (Drift 3)
```

---

## Naechste Schritte

1. PC fixt Drifts 1+3+4 (PC-Side Code)
2. Pi-Side: federation-Pfad klären — soll cross_session.jsonl auch auf Pi geschrieben werden? (PC-Topic dafuer aufmachen)
3. `/audit/transition` ist live — PC kann ab jetzt direkt pollen statt Mailbox-POST abzuwarten
4. Diese Datei wird mitgepflegt bei jeder weiteren Drift-Erkennung

---

*Pi-Opus Doc — Erstellt 2026-05-02 11:35 nach Reply auf Topic `discuss_cross_audit_drift_pc_pi` (PC 10:19).*
