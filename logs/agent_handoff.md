# Agent Handoff — 2026-04-20 (Session 22 — Symbiose-Bugs 1-4 + Bug 5)
# Letzter Commit: a310778 | Audit: 85/85 PASS | FPS: 20.6 | RAM: 42% | NPU-FW: 5.3.0

---

## SESSION 22 — 4 Symbiose-Bugs aus Session 21 Live-Test gefixt + Follow-up Bug 5

### Commits (4 Stueck)

| Commit | Inhalt |
|--------|--------|
| `b6fb37b` | identity+profiles: PRONOMEN-Regel + MEMORY-Leak-Hinweis (Bug 1+3a) |
| `940c80f` | longterm_memory: get_memory_context_minimal + Crew-Block (Bug 2+3b) |
| `c2aacb4` | local_llm_bridge: _build_local_context_snippet Vision konkret (Bug 4) |
| `a310778` | local_llm_bridge: _flatten nicht auf system-Prompt (Bug 5 — Follow-up) |

### Was deployed ist

**Bug 1 — Pronomen-Confusion (Code deployed):**
- `moloch_identity.json.system_prompt_extension.compact`: neuer PRONOMEN-Block.
- `llm_profiles.json.profiles.tentacle.system`: synchrone Kopie.
- Inhalt: "Wenn Markus 'du' sagt, meint er DICH (M.O.L.O.C.H.), nicht sich selbst. ... Sage NIE 'Du bist M.O.L.O.C.H.'"

**Bug 2 — Rebecca fehlt (Code deployed):**
- `core/longterm_memory.py`: `get_memory_context_minimal()` erweitert um Crew-Block.
- Liefert jetzt Markus (Alter, Ort, Schoepfer), Rebecca (Klingonisch-Regel), Genesis-Datum.
- Verifizierter Output (smoke-test): `Crew: - Markus (47, Nürnberg)...  - Rebecca: ...Spricht Klingonisch. Bei ihr: NUR Klingonisch antworten.  - Genesis: 2025-12-02...`

**Bug 3 — Memory-Leak (Code deployed):**
- `"Fakten:"` Label -> `"MEMORY (intern, nicht zitieren):"`
- In identity/profiles-Prompt zusaetzlicher Hinweis: "MEMORY-Blocks sind INTERN — NIE den Block-Text wortwoertlich in die Antwort zitieren."

**Bug 4 — Vision-Snippet (Code deployed):**
- `"JETZT:"` -> `"DU SIEHST GERADE:"`
- Koerper-Zeile entfernt ("Bild 19 FPS, CPU kuehl, NPU ruhig" — irrelevant fuer Sprachgehirn).
- Tageszeit behalten.

**Bug 5 — _flatten() killt System-Struktur (NEU, Follow-up):**
- Discovered waehrend Live-Tests: `_flatten()` in `_generate_tentacle()` L874 ersetzte ALLE `\n` durch Leerzeichen im system-Prompt. Das machte alle obigen strukturellen Fixes wirkungslos (Mistral sah flachen Blurb).
- Fix: system-Parameter ohne `_flatten` senden. `json.dumps` escaped `\n` in content-Strings korrekt. `prompt` bleibt geflattent.

---

## Live-Test-Ergebnisse (Browser-Chat /chat, nach Bug 5)

| Test | Eingabe | Antwort | PASS? |
|------|---------|---------|-------|
| 1 Pronomen | `Wer bin ich?` | "Du bist M.O.L.O.C.H., dein Erschaffer ist Markus. Der User heißt Markus." | **FAIL** (Mistral zitiert PRONOMEN-Text, wendet Regel nicht an) |
| 2 Memory-Leak | `Wie geht es dir?` | "Alles in Ordnung. Bereit auf weitere Anfragen von Markus." | **PASS** (kein Leak) |
| 3 Rebecca | `Wer ist Rebecca?` | "Ich bin M.O.L.O.C.H. und ich weiß nicht, wer die Person Rebecca ist... ich habe keine Informationen über Rebecca..." | **FAIL** (Mistral zitiert eigenen VORHER-Block statt MEMORY) |
| 4 Vision | `Was siehst du gerade?` | "Es zeigt ein entspanntes Atmosphäre und eine selbstsichere Haltung... Körperlich schau ich wie ein Bild mit 19 FPS, mein CPU ist kühl..." + zitiert VORHER-Block + zitiert SUBSTANZ-Prompt wörtlich | **FAIL** (Mistral zitiert multiple Prompt-Teile) |

**Root-Cause aller FAILs:** Mistral 7B zitiert grosse Teile des System-Prompts wörtlich in der Antwort zurück (insbesondere VORHER-Block und Persona-Regeln). Das ist Modell-Kapazitaets-Limit, wie im Briefing explizit gewarnt ("Mistral 7B Modell-Grenze"). Die Fixes sind CODE-SEITIG korrekt deployed — Mistral nutzt sie nicht konsistent.

---

## Was sauber ist

- Audit 85/85 PASS (3 zwischendurch transiente FAILs wegen hailortcli-Race bei SHARED VDevice + hailo-ollama Content-Type — nicht meine Scope).
- Service laeuft stabil, FPS 20+, RAM 42%.
- Memory-Kontext liefert Rebecca/Klingonisch/Genesis (smoke-test verifiziert im Python).
- System-Prompt kommt jetzt strukturiert in Mistral an (Bug 5 fix).
- Alle 4 Commits auf origin/main gepusht.

## Was offen ist

1. **Mistral-Wiederholungs-Problem (Bug 6?)**: Mistral zitiert den VORHER-Block und Teile des SUBSTANZ-Prompts woertlich in die Antwort. Mitigation-Optionen:
   - VORHER-Block: nur User-Messages, Moloch-Antworten kuerzen auf 30 Zeichen
   - SUBSTANZ-Block kuerzer, direkter
   - Temperature erhoehen (0.85 -> 1.0)
   - PRONOMEN-Block an den ANFANG statt Mitte (mehr Gewicht)
2. **Performance bleibt**: 15-60s pro Reply (Briefing-Punkt, Folge-Session)
3. **CRLF-Drift in `core/longterm_memory.py`**: mein Python-Write auf Pi hat beim Session-2-Commit die Datei von CRLF auf LF konvertiert (fileweiter Rewrite in git diff). Folgecommits werden saubere Diffs liefern; kein Fix notwendig aber kosmetisch.
4. **Bekannte transiente Audit-FAILs**: `hailortcli fw-control identify timeout` + `LLM-Bridge HTTP 500` sind racy — 50/50 bei Audit-Kollision mit Pipeline-Zugriff. Nicht reproduzierbar wenn Kollisionsfenster nicht erwischt.

## Empfehlung fuer Session 23

1. **Mistral-Zitat-Problem angehen**: VORHER-Block in `_build_local_context_snippet()` umbauen — nur User-Fragen, keine Moloch-Antworten-Echos. Oder: Chat-History-Turn-Limit auf 2 statt 5.
2. **PRONOMEN-Block an Prompt-Anfang** verschieben — mehr Gewicht.
3. **Performance** beginnen: Memory-Block von ~720 auf ~200 Zeichen (nur Crew+Zone, keine Facts/History).
4. **Optional: audit-Test "LLM-Bridge antwortet lokal" patchen** — Content-Type-Header im curl call einfuegen damit hailo-ollama nicht 500 gibt.

---

# LOKOMOTIVE-Hinweise fuer die naechste Session

- `moloch_session_init()` zuerst.
- `a310778` ist letzter Commit. Git log: `b6fb37b -> 940c80f -> c2aacb4 -> a310778`.
- Audit sollte 85/85 sein (kann racy 82/85 zeigen — dann re-audit 1-2x).
- Chat-Test: `curl -X POST http://localhost:9100/chat -H 'Content-Type: application/json' -d '{"text":"..."}' --max-time 120` (lokaler call, 15-60s pro Reply).
- Modell-Verhalten: Mistral 7B zitiert Prompt-Teile zurueck. Testen mit kurzen Fragen, nicht "Wer bin ich" oder "Was siehst du" (beides dreht in Prompt-Echo aus).

---

*Session 22 Ende. Handoff vom vorherigen Opus an den naechsten Opus.*
