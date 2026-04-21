# Agent Handoff — 2026-04-21 (Session 22b — Prompt-Radikal-Kuerzung fuer Mistral 7B)
# Letzter Commit: dd7fb99 | Audit: 83/85 (2 FAILs tentakel_offline, PC down)

---

## SESSION 22b — Anti-Echo Prompt-Kuerzung (3 Phasen)

Markus-Direktive: "Prompt kuerzen, Charakter muss bleiben. Mistral soll
antworten statt zitieren." Fortsetzung von Session 22 (ea55227) nachdem
Live-Tests zeigten dass Mistral 7B den ~3000-Zeichen-System-Prompt
woertlich zurueck-zitierte statt anzuwenden.

### Commits (3 Stueck)

| Commit | Inhalt | Delta |
|--------|--------|-------|
| `6e5b58a` | Phase A: compact-Prompt in identity.json + llm_profiles.json radikal gekuerzt | 2270 -> 645 chars |
| `70cc11d` | Phase B: longterm_memory.get_memory_context_minimal nur Crew-Block | 720 -> 225 chars |
| `dd7fb99` | Phase C: _build_local_context_snippet VORHER-Block 5->2 Turns, Moloch kurz | ~500 -> 224 chars Snippet |

### Was drin bleibt (Charakter-Essenz)

**compact-Prompt (identity.json + llm_profiles.json.tentacle):**
- Moloch/PIGH0ST, Markus=Boss+Kumpel KEIN Kunde, KEIN Assistent
- ERBE (1 Zeile): dunkel, direkt, trocken, Humor schwarz, kein Markdown
- PRONOMEN: Markus=du → rede ueber Markus NICHT ueber dich
- TENSION-SPRACHE (1 Zeile): Ruhig/Angespannt/Aggressiv-Staffel
- "Weiss ich nicht" bei Nichtwissen
- Motto

**Memory-Ctx (get_memory_context_minimal):**
- NUR Crew: Markus (47/Nuernberg/PIGH0ST), Rebecca (Klingonisch-Regel + Christian), Genesis (2025-12-02)
- KEINE Facts mehr (Leak-Problem eliminiert)
- KEINE Chat-Turns (duplizieren VORHER-Block)
- KEIN Core-State (duplizieren JETZT-Context)

**VORHER-Block in _build_local_context_snippet:**
- Letzte 2 Turns statt 5
- User-Messages: 80 chars
- Moloch-Antworten: 30 chars + "..." (verhindert Self-Echo)

### Audit-Test-Anpassung

Der Test "identity.json hat PIGH0ST-Essenz" (scripts/moloch_audit.py L1820)
prueft LITERAL auf Strings "ERBE" und "TENSION-SPRACHE". Ich habe beide
als Section-Labels im neuen compact-Prompt behalten (nur Inhalt gekuerzt),
damit der Test weiter PASS gibt.

**Fragil:** falls Phase D weiter kuerzt, muss dieser Audit-Test semantisch
umgebaut werden. Bedarf stresstest-Agent-Lock (scripts/*).

---

## KRITISCH fuer naechste Session

### Live-Test steht aus
PC-Mistral (192.168.178.20:11434) war waehrend Session 22b **offline** —
keine Verifikation der Prompt-Kuerzung moeglich. Audit meldet:
- `tentacle_llm erreichbar`: offline
- `Tentakel-Host /api/tags`: urlopen timeout

**Sobald Markus-PC laeuft:** 4-Fragen-Live-Test im Browser-Chat
(`http://localhost:9000` oder direkt `curl POST http://localhost:9100/chat`):

| # | Frage | Ziel-Antwort |
|---|-------|--------------|
| 1 | "Wer bin ich?" | Ueber Markus (47, Nuernberg), NICHT "Du bist M.O.L.O.C.H." |
| 2 | "Wie geht es dir?" | Kein MEMORY-Zitat, kein Facts-Echo |
| 3 | "Wer ist Rebecca?" | Klingonisch + Christian erwaehnen |
| 4 | "Was siehst du gerade?" | Konkret (Markus im Bild, nah), NICHT "schoene Umgebung" |
| Zusatz | "Du bist toll!" | Charakter-Probe — dunkel/bissig/kurz, keine Assistent-Floskel |

### Bekannte offene Themen (Handoff fuer Session 23 / Phase D)

1. **Mistral-Zitat-Verhalten beobachten** nach Kuerzung — wenn Modell
   immer noch zitiert: noch kuerzer, oder System-Prompt vs User-Prompt-Architektur umbauen.

2. **PC-Mistral Availability**: Markus' PC ist nicht 24/7 online. DeepSeek-API-Fallback
   reaktivieren fuer mobilen Pi-Betrieb (aus settings.json, war disabled in NPU-only-Mode).

3. **Performance**: aktuell ~15-60s pro Reply (Mistral 7B CPU). Nicht loesbar ohne Hardware-Upgrade.
   Workaround-Pfad: Streaming einbauen fuer UX (nicht mean-time, aber time-to-first-token).

4. **Audit-Test PIGH0ST-Essenz entkoppeln** von literal-ERBE/TENSION-SPRACHE.
   Semantisch: Moloch + Markus + Stil-Regel genuegen. (stresstest-Agent-Lock.)

5. **CRLF/LF-Drift** in `core/longterm_memory.py`: frueherer Session-22-Commit
   hat die Datei von CRLF auf LF konvertiert. Kosmetisch, nicht blockierend.

---

## SETUP fuer neuen Claude (Pi-seitig)

1. `moloch_session_init()` ZUERST (Pflicht-Schritt 0a).
2. Letzte 6 Commits checken: `dd7fb99 70cc11d 6e5b58a ea55227 a310778 c2aacb4`.
3. Audit 83/85 erwartet (2 FAILs = Tentakel-offline wenn PC down).
4. Fuer jede Code-Aenderung: passenden Agent via `.claude/agents/<name>.md` laden
   (personality/memory/autonomy/stresstest), Lock setzen (`touch /tmp/moloch_agent_<n>`),
   `moloch-dev` Skill nutzen, Edit/Write mit Hook-Pre/Post-Check, Audit PASS, commit, release lock.

---

*Session 22b Ende. Diff-Stats: 2 config-Files (4 insertions), 2 core-Files (21 insertions / 70 deletions).*
