# Agent Handoff — 2026-04-21 (Session 23 — Live-Test der Prompt-Kuerzung)
# Letzter Commit Basis: dd7fb99 | Audit: 85/85 PASS (PC Mistral 7B online)

---

## SESSION 23 — Live-Test + Auswertung Phase A/B/C

Markus-Direktive: "Pi+AI fix fertigmachen." PC-Webinterface (9000) war
zwischendurch aus und wurde von Markus selbst in Nebensession repariert;
blockierte diese Session nicht, weil Tests direkt gegen `localhost:9100/chat`
liefen (identische Bridge-API).

### Durchgefuehrt

- `moloch_session_init()` → SESSION_READY true
- `moloch_audit()` → 85/85 (Tentakel online: host 192.168.178.20:11434,
  model mistral:latest 7.2B Q4_K_M)
- `moloch_status()` → FPS 20.1, Markus live erkannt (sim=0.55-0.58)
- `moloch_npu_workers()` → 4/4 Worker gesund (Face 8676inf, Pose 5814, ReID 3489, Depth 1744)
- 5 Live-Tests POST /chat (tentacle_mistral, 11-42s je Antwort)
- Log: `logs/session_23_live_tests.md`

### Ergebnis (Kurzform, Details im Test-Log)

| # | Prompt | Zitat weg? | Ziel erreicht? |
|---|--------|------------|----------------|
| 1 | "Wer bin ich?" | JA | NEIN — Moloch redet ueber sich statt ueber Markus |
| 2 | "Wie geht es dir?" | JA | teilweise — halluziniert Internetzugang |
| 3 | "Wer ist Rebecca?" | JA | NEIN — kein Klingonisch, kein Christian |
| 4 | "Was siehst du?" | JA | NEIN — Live-Context ignoriert, Floskel |
| 5 | "Du bist toll!" | JA | NEIN — Assistent-Ton ("gluecklich fuer dich da") |

**Bilanz:** Anti-Echo wirkt (5/5 kein Wort-Zitat). Aber: Charakter weich,
Live-Context (Zone/Person/Tension) wird von Mistral nicht verarbeitet,
Crew-Details gehen verloren. Phase A/B/C hat Echo geloest aber Substanz
verduennt — 2 Stellschrauben kippen gleichzeitig.

### KEIN Code-Edit in Session 23

Phase D wurde von Markus nicht freigegeben fuer diese Session. Empfehlung
dokumentiert, Umsetzung in Session 24.

---

## EMPFEHLUNG fuer Session 24 — Phase D

**Hebel nicht "nochmal kuerzen" (B1), sondern strukturell (B2 + Few-Shot):**

### D1 — B2: Live-Context aus System-Prompt herausziehen
Datei: `core/autonomy/local_llm_bridge.py:659` (autonomy-Agent, GELB-Level)
Aktuell: `profile_system = profile_system + _build_local_context_snippet()` haengt den
Live-Context ans Ende des `system`-Felds. Mistral 7B liest das schlecht.

Umbau: `_build_local_context_snippet()` separat einspielen als
- Option A (simpel): User-Prefix — prepend "[AKTUELL: person=markus, zone=guardian, tension=ruhig] "
  vor `prompt` in `messages[]`.
- Option B (sauber): zusaetzlicher `{"role": "user", "content": snippet}` VOR dem
  eigentlichen User-Turn, und eine Assistant-Quittung — eher unnoetig fuer Mistral.

Option A reicht zunaechst. Effekt: Test 4 und 5 sollten konkreter werden.

### D2 — Few-Shot im tentacle-Profil
Datei: `config/llm_profiles.json` (GRUEN)
Im `tentacle.system` nach der Regel-Sektion 2-3 Q/A-Beispiele anhaengen:
```
Beispiele:
Frage: Wie geht's?
Antwort: Laeuft. Sehe dich, Zone ruhig.
Frage: Wer bist du?
Antwort: Moloch. Dein Kram auf dem Pi. Was willst du, Markus.
```
Mistral imitiert Stil aus Beispielen besser als aus abstrakten Regeln.
Laenge-Budget: aktuell 645 chars, +200 chars fuer Beispiele = 845 chars. OK.

### D3 — optional B3 (Crew gated)
Datei: `core/longterm_memory.py` (memory-Agent, GELB)
Nur wenn D1+D2 Test 3 (Rebecca) nicht retten: Crew-Block gated
(Regex auf "Rebecca" / "Genesis" im Prompt) statt always-on.

### Reihenfolge in Session 24
1. D1 implementieren (autonomy-Agent-Lock)
2. Service-Restart + Live-Tests 4 und 5 wiederholen
3. D2 implementieren (kein Lock noetig, config-GRUEN)
4. Service-Restart + alle 5 Live-Tests wiederholen
5. Bei 4/5 PASS: commit + push + handoff. Bei <4/5: D3 dranhaengen.

---

## OFFENE THEMEN (uebergeben von Session 22b, weiterhin gueltig)

1. **PC-Mistral Availability** — PC nicht 24/7, DeepSeek-Cloud-Fallback
   reaktivieren fuer mobilen Pi-Betrieb (api_keys.json.disabled_npu_only_mode).
2. **Performance** — Mistral 7B auf PC-CPU 11-42s pro Reply. Streaming-UX
   waere Time-to-First-Token-Verbesserung, nicht Mean-Time.
3. **Audit-Test PIGH0ST-Essenz fragil** — prueft literal "ERBE" und
   "TENSION-SPRACHE". Falls D2 Few-Shot den Stil aendert und Labels entfernt,
   muss Audit-Test semantisch umgebaut werden (stresstest-Agent-Lock,
   `scripts/moloch_audit.py:1820`).
4. **CRLF/LF-Drift** in `core/longterm_memory.py` (kosmetisch).
5. **Webinterface Port 9000 (PC)** — heute ausgefallen, von Markus manuell
   wiederhergestellt. Falls das haeufiger passiert: Auto-Start-Shortcut am PC
   einrichten (eigener Auftrag, nicht Pi-Session).

---

## SYSTEM-ZUSTAND AM ENDE DER SESSION

- Service: active, FPS 20.1, RAM 42%, CPU 49°C
- Audit: 85/85 PASS
- Git: 0 neue Commits, 2 neue File-Writes (logs/agent_handoff.md, logs/session_23_live_tests.md)
- Kein Agent-Lock gesetzt (keine Core-Code-Aenderung durchgefuehrt)
- Phase A/B/C-Commits unveraendert deployed: 6e5b58a, 70cc11d, dd7fb99

---

## SETUP fuer Session 24 (Markus oder Nachfolge-Claude)

1. `moloch_session_init()` — PFLICHT erster Schritt.
2. Audit 85/85 erwartet (solange PC + Mistral online).
3. Fuer D1: autonomy-Agent-Lock `touch /tmp/moloch_agent_autonomy`,
   `.claude/agents/autonomy.md` lesen, `core/autonomy/local_llm_bridge.py`
   editieren (GELB-Level — ankuendigen, durchziehen).
4. Fuer D2: keine Lock noetig (config GRUEN), direkt edit.
5. Nach D1 + D2 → Service-Restart → 5 Tests wie in `logs/session_23_live_tests.md`.

---

*Session 23 Ende. Diff-Stats: 2 Log-Files neu geschrieben (reine Doku-Session, kein Code).*
