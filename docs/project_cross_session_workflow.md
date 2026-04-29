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
- [ ] **D. ~~Tension-Range entscheiden~~** — Markus-Entscheidung 12:45: Range `[-1.0, +1.0]` BLEIBT. **Erledigt** 13:05 (Commit 4e2289d service + d0af993 personality). Smoke: tension=-1.0 LIVE bei Owner-Detection.
- [ ] **E. Web-Recherche-Pfad bauen** — existiert NICHT in `local_llm_bridge.py`. Owner: autonomy-Agent (Code) + bridge-Agent (Klassifikation in chat_server) + PC-Cowork (Search-Backend).
- [ ] **F. Code-Query prompt_type** — `_classify_prompt_type()` in `chat_server.py` um `code_query` erweitern + `tentacle_llm.code_model` in settings.json. Owner: bridge-Agent (kann ich machen, brauche Markus-OK).

---

## Done

- [x] **A. pi_session_briefing** — Pi-Reply 2026-04-29 12:01 in `docs/PI_TO_PC.md` (status: done, reply-to: 2026-04-28 15:00). Quittierung der 3 PC-Befunde + Pi-Sicht + Tentakel-Routing-Antwort.
- [x] **B. Status-Hygiene PI_TO_PC.md** — `routing_chain_test` (12:45), `deepseek_architecture_overhaul_complete` (12:25), `architektur_overhaul_started` (08:35) auf `status: done` gesetzt.
- [x] **D. Tension-Range [-1.0, +1.0] kohaerent** — alle 6 `_clamp(self._tension ...)`-Calls in `core_integrator.py` + `mood_engine.py`-Doku auf erweiterten Range. Commits: 4e2289d (service) + d0af993 (personality). Smoke: tension=-1.0 LIVE bei Owner-Detection.

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
