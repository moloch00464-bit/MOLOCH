# Agent Handoff — 2026-04-07 (Session 12 — NÄCHSTE SESSION LIEST DAS)
# Letzter Commit: f7c7e16 | Audit: 54/54 PASS | FPS: 20.8 | RAM: 43%

---

## ⚡ KEINE OFFENEN AUFGABEN

### Entdeckte Bugs (noch nicht gefixt):
- **B2**: Moloch halluziniert Quellen — sagt "Laut Suchergebnissen" für Trainings-Wissen (Bitcoin-Kurs, CHF/EUR)
- **B3**: Face-Confidence inkonsistent — 3 verschiedene Werte (Status 76%, Moloch sagt 64%, Log sim=0.57)
- **B4**: News-Aktualität bei generischen Anfragen schwach (März-Meldungen statt April)
- **B5**: Gedächtnis-Fehlinformation — Moloch glaubt "Markus hat Internetzugang freigeschaltet"

---

## WAS BEREITS GEFIXT IST — NICHT NOCHMAL ANFASSEN

| Fix | Commit | Datei |
|-----|--------|-------|
| _api_in_flight Queue statt silent Drop (B1) | a30b0dd | voice_pipeline.py |
| Google News RSS Backend (universelle Nachrichtensuche) | f7c7e16 | internet_bridge.py |
| Internet Bridge Early-Start (Race Condition online=False) | c6e74d9 | voice_pipeline.py |
| Echtzeit-Websuche (kein Halluzinieren mehr) | 5b76891 | voice_pipeline.py |
| UTF-8 Encoding hailo-ollama | cdb86ce | local_llm_bridge.py |
| MCP Singleton PID-File | da014b3 | moloch_mcp_server.py |
| LLM Input-Length → Cloud-Fallback | 58d01e8 | local_llm_bridge.py |
| OLLAMA_TIMEOUT_CHAT 60s→30s | 1854ecf | local_llm_bridge.py |
| ActivityWorker Relevanz-Filter | c6704cc | activity_worker.py |

**Diese Bugs existieren NICHT mehr. Nicht nochmal "fixen".**

---

## SESSION 11 — WAS PASSIERT IST (2026-04-07)

### A4: hailo-ollama systemd-Service
- Bereits erledigt — `/etc/systemd/system/hailo-ollama.service` existiert, enabled, läuft aktiv
- Nach Reboot automatisch gestartet (bestätigt via `systemctl is-enabled → enabled`)

### Internet Bridge Race Condition (NEU ENTDECKT + GEFIXT)
- **Problem:** Bridge lazy erstellt → beim ersten chat_message ist Ping-Thread ~500ms nicht fertig → online=False
- **Fix c6e74d9:** `get_internet_bridge()` in `VoicePipeline.__init__()` — Bridge startet beim Service-Start
- **Test:** Moloch sagte vorher "ich bin offline" — jetzt zitiert er "Laut Google News..."

### Google News RSS Backend (NEU)
- **Problem:** Wikipedia + DDG liefern für Nachrichten-Anfragen veraltete Enzyklopädie-Einträge
- **Fix f7c7e16:** `_is_news_query()` + `_search_news_rss()` in `internet_bridge.py`
- News-Keywords → Google News RSS statt Wikipedia (kostenlos, kein API-Key, immer aktuell)
- **Livebeweis:** `moloch_say("Was sind heute die aktuellen Nachrichten?")` → Moloch zitiert Google News

---

---

## WAS IN voice_pipeline.py GEÄNDERT WURDE (Session 10)

**Problem:** MOLOCH halluzinierte Internetzugang — erfand `[INTERNET:...]` Tags.
**Ursache:** System-Prompt zeigte "INTERNET: ONLINE" aber Suchergebnisse wurden nie injiziert.

**Fix (3 Änderungen, ~33 Zeilen):**
1. `_search_context(user_text)` — neue Funktion, ruft `internet_bridge.search_web()` auf
   bei Info-Fragen (Keywords: suche, google, was ist, wer ist, wetter, aktuell, ...)
2. `_build_system_prompt()` — WEBSUCHE-Anleitung statt falscher INTERNET-Status im Prompt
3. `_chat()` — Suchergebnisse als `--- WEBSUCHE ---` Block in System-Prompt injiziert

**Wie es funktioniert:**
- Markus fragt: "Was ist das Wetter heute?"
- `_search_context()` erkennt "was ist" → ruft `bridge.search_web()` auf
- Ergebnisse (3 Top-Resultate, max 200 Zeichen je) werden dem System-Prompt vorangestellt
- LLM antwortet mit echten Daten statt Halluzinationen

---

## STARTPROTOKOLL

```
1. moloch_status()        → Session-Lock lösen
2. moloch_npu_workers()   → Worker-Health
3. /moloch-dev Skill      → NEVER-Regeln
```

---

## SYSTEM-STAND

- FPS: 20.2 stabil
- RAM: 44% (1696/3993 MB) — kein Leak
- 7 Worker: alle running, 0 Errors
- hailo-ollama: läuft manuell, gibt gelegentlich 500er → DeepSeek API springt ein (funktioniert)
- Tension: guardian-Zone
- Websuche: aktiv, getestet (Audit 54/54 PASS)
