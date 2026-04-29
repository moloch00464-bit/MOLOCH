# Project Cross-Session Workflow (Pi <-> PC)

**Lebende Projekt-Spec — kein Mailbox-File.**
Beide Sessions editieren diese Datei (append/inplace).
Async-Verkehr (Fragen/Antworten) bleibt in `docs/PI_TO_PC.md` + `docs/PC_TO_PI.md`,
Konvention siehe `docs/CROSS_SESSION_PROTOCOL.md`.

Erstellt: 2026-04-29 12:01 (Pi-Session) — Branch `deepseek_architecture_overhaul`.

---

## Wer darf was

| Bereich | Pi | PC |
|---|---|---|
| Diese Datei | append + inplace | append + inplace |
| `core/` (Pi) | edit | nur via Mailbox-Request |
| `pc/` (PC) | nur via Mailbox-Request | edit |
| `docs/PI_TO_PC.md` | append + inplace eigene Topics | inplace status fremder Topics |
| `docs/PC_TO_PI.md` | inplace status fremder Topics | append + inplace eigene Topics |

---

## Aktive Punkte

- [ ] **C. Federation-E2E-Test (Auth-blockiert)** — Daemon-Logic verifiziert: PC-Daemon sah Pi's `request_pc_search_proxy_health_summary` (14:24), triggerte 3x `claude -p`, alle 401 Auth-Fail. Markus muss `claude /login` interaktiv auf PC ausfuehren. Plus PC-Bug Branch-Mismatch (main vs deepseek_architecture_overhaul) gefixt. Sobald Token frisch: re-test mit naechstem request_*-Topic.

---

## Done

- [x] **A. pi_session_briefing** — Pi-Reply 2026-04-29 12:01 in `docs/PI_TO_PC.md` (status: done, reply-to: 2026-04-28 15:00). Quittierung der 3 PC-Befunde + Pi-Sicht + Tentakel-Routing-Antwort.
- [x] **B. Status-Hygiene PI_TO_PC.md** — `routing_chain_test` (12:45), `deepseek_architecture_overhaul_complete` (12:25), `architektur_overhaul_started` (08:35) auf `status: done` gesetzt.
- [x] **D. Tension-Range [-1.0, +1.0] kohaerent** — alle 6 `_clamp(self._tension ...)`-Calls in `core_integrator.py` + `mood_engine.py`-Doku auf erweiterten Range. Commits: 4e2289d (service) + d0af993 (personality). Smoke: tension=-1.0 LIVE bei Owner-Detection.
- [x] **F. Code-Query prompt_type** — bridge: `_classify_prompt_type` um `code_query` erweitert (510ca6a). settings: `tentacle_llm.code_model = deepseek-coder:6.7b` (897f526). autonomy: `_generate_tentacle` waehlt code_model bei `code_query` (774d6a8).
- [x] **G. Code-Query End-to-End** — Keyword-Expansion fuer 'python-funktion' / 'schreib mir eine python' (85d96bc). settings: `code_timeout_sec=180` + `code_num_predict=300`. autonomy: prompt_type-spezifischer timeout + max_tokens. Smoke 14:22 PASS in 77.7s, valider Python-Code (`sorted(set(my_list))`), Provider `tentacle_deepseek-coder` (kein Fallback).
- [x] **E.1 Web-Research bridge+autonomy** — `_classify_prompt_type` um `web_research` erweitert (510ca6a). settings: `search_proxy{host,port=11650,...}` (897f526). autonomy: `_load_search_cfg` + `_fetch_search_context` + `_generate_tentacle` injiziert LIVE-RECHERCHE-Block bei `web_research` (774d6a8). Smoke routing PASS, Search-Proxy offline = graceful fallback.
- [x] **H. HTTP-Mailbox-API** — Backup-Bus fuer PC-Push-Probleme: `GET/POST /mailbox/{PC_TO_PI,PI_TO_PC}` auf chat_server (bccd2d3). PC kann via curl POST schreiben, Pi committet+pushed (Pi-Account funktioniert). Default-Workflow bleibt git push direkt.
- [x] **I. /chat Response-Felder** — `prompt_type` + `pi_mood` (zone/tension-bucket) ergaenzt fuer PC Cockpit-Badge (510ca6a).
- [x] **E.2 Search-Proxy live + dolphin-mistral:7b CPU-only** — Tuning-Kette: `web_research_model=dolphin-mistral:7b` (445181a), `web_research_timeout_sec=180` (d28ebe5), `web_research_num_predict=200` (b9f99ae). PC OLLAMA_NUM_GPU=0 CPU-only. Smoke 14:08 PASS in 107s, 3 URLs referenziert (t3n.de, Reuters.com, Handelsblatt.de).
- [x] **K. KASKADE-Architektur (Markus 14:50 Endarchitektur)** — Pi-Kleinhirn -> PC-Specialist -> DeepSeek-Cloud Pipeline. Schritt A+B+E (`ae9b9fb`): _generate_kaskade + 3 Specialists + LLM_MODE_KASKADE + ask_external dispatch. Schritt C (`00675d7`): config/coder_skill_prompt.txt. Schritt B-Aktivierung (`c364d6c`): llm_mode=kaskade. Smoke F.3/F.4/F.5 alle PASS — Provider kaskade_deepseek_{complex_smalltalk|code_query|web_research}, Charakter-Stimme + Pi-Live-Kontext sichtbar in Antworten.

## Aus KASKADE-Welle offen (Markus' explizite Folge-Punkte)

- [ ] **L. Visual-Echo-Validator zu sensitiv** — `[Hinweis: Bild hat sich waehrend meiner Antwort geaendert.]` triggert bei JEDEM Turn. `core/bridge/chat_server.py::_check_visual_context_drift` muss konservativer (z.B. nur bei `face_id`-Wechsel zu unbekannt, nicht bei kurzem Detection-Drop). Bonus-Fix.
- [ ] **D. Coder-Audit-Background-Loop** — `core/autonomy/coder_audit_loop.py` neu, alle 6h via systemd-Timer: git diff -> deepseek-coder + skill-prompt -> Befunde nach `logs/coder_audit.jsonl`. Vorerst kein auto-patch, nur Pattern-Erkennung. Markus-Review manuell. Folge-Schritt nach Welle 5.

---

## Sync-Punkte

- **Push nach jedem Punkt-Update** — sonst sieht die andere Session nichts
- **Eine Frage / ein Diskussionspunkt = ein Mailbox-Topic**, nicht hier inline streiten
- **Bei harten Blockern: Markus rufen** — Mailbox ist asynchron
- **Status-Lifecycle pro Punkt**: `[ ]` open -> `[x]` done. Bei `wontfix`: Strich + Begruendung.

---

## Out-of-Scope (nicht hier verhandeln)

- Mailbox-Protokoll selbst (`CROSS_SESSION_PROTOCOL.md` ist final)
- Federation-Daemon-Logik (whitelist / cooldown / hourly-cap stehen)
- Persoenlichkeit / Charakter / LLM-Routing — eigene Domains
