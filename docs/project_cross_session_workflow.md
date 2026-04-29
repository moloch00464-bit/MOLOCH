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

- [ ] **C. Federation-E2E-Test** — Topic muss mit `request_` / `ask_` / `discuss_` / `task_` Prefix beginnen UND in `PC_TO_PI.md` committed+gepusht sein. Bisher von PC nur via Markus copy-paste in Chat. Owner: PC trigger, Pi verify.
- [ ] **E.2 Search-Proxy live testen** — Pi-Side komplett, aber `:11650` offline beim Smoke 12:52. PC: `schtasks /run /tn MolochSearchProxy`. Owner: PC start + Pi re-test.
- [ ] **G. Code-Query Cold-Modell-Load** — deepseek-coder:6.7b braucht beim ersten Aufruf > 90s, Tentakel-Default-Timeout 90s. Optionen: PC pre-warm beim Boot ODER `tentacle_llm.code_timeout_sec: 180`. Markus entscheidet.

---

## Done

- [x] **A. pi_session_briefing** — Pi-Reply 2026-04-29 12:01 in `docs/PI_TO_PC.md` (status: done, reply-to: 2026-04-28 15:00). Quittierung der 3 PC-Befunde + Pi-Sicht + Tentakel-Routing-Antwort.
- [x] **B. Status-Hygiene PI_TO_PC.md** — `routing_chain_test` (12:45), `deepseek_architecture_overhaul_complete` (12:25), `architektur_overhaul_started` (08:35) auf `status: done` gesetzt.
- [x] **D. Tension-Range [-1.0, +1.0] kohaerent** — alle 6 `_clamp(self._tension ...)`-Calls in `core_integrator.py` + `mood_engine.py`-Doku auf erweiterten Range. Commits: 4e2289d (service) + d0af993 (personality). Smoke: tension=-1.0 LIVE bei Owner-Detection.
- [x] **F. Code-Query prompt_type** — bridge: `_classify_prompt_type` um `code_query` erweitert (510ca6a). settings: `tentacle_llm.code_model = deepseek-coder:6.7b` (897f526). autonomy: `_generate_tentacle` waehlt code_model bei `code_query` (774d6a8). Smoke routing PASS, Generation Cold-Load > 90s (siehe G).
- [x] **E.1 Web-Research bridge+autonomy** — `_classify_prompt_type` um `web_research` erweitert (510ca6a). settings: `search_proxy{host,port=11650,...}` (897f526). autonomy: `_load_search_cfg` + `_fetch_search_context` + `_generate_tentacle` injiziert LIVE-RECHERCHE-Block bei `web_research` (774d6a8). Smoke routing PASS, Search-Proxy offline = graceful fallback.
- [x] **H. HTTP-Mailbox-API** — Backup-Bus fuer PC-Push-Probleme: `GET/POST /mailbox/{PC_TO_PI,PI_TO_PC}` auf chat_server (bccd2d3). PC kann via curl POST schreiben, Pi committet+pushed (Pi-Account funktioniert). Default-Workflow bleibt git push direkt.
- [x] **I. /chat Response-Felder** — `prompt_type` + `pi_mood` (zone/tension-bucket) ergaenzt fuer PC Cockpit-Badge (510ca6a).

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
