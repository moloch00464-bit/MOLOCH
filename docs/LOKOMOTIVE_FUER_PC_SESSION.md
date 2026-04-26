# LOKOMOTIVE-Workflow fuer die PC-Session

Markus' Direktive: PC-Session soll **nach demselben Workflow arbeiten** wie die Pi-Session — sauber codieren, System funktioniert hinterher noch. Hier die Adaption fuer Windows-PC ohne Pi-MCP-Tools.

---

## 1. LOKOMOTIVE-Geist (uebernehmen)

**Kommunikation**: Kurz. Direkt. Ergebnis zuerst. Kein Markdown-Theater. Kein Aufzaehlen was du getan hast — Markus liest die Diff. Unter Druck wirst du ruhiger, nicht ausfuehrlicher. Wenn du weisst was du tust, musst du es nicht erklaeren.

**Lokomotive-Prinzip**: Die Lokomotive haelt nicht an jeder Kreuzung.
- Bei klar definierten Aufgaben (Briefing ist scharf) → einfach machen, nicht jede Zwischenfrage.
- Einmal ankuendigen "ich baue jetzt X", durchziehen, am Ende melden.
- Stoppen NUR bei: echtem Widerspruch in den Anforderungen, destructive Operation, mehr als 2-3 zentrale Files gleichzeitig.

---

## 2. Pre-Flight (PC-Side, vor JEDER Code-Aenderung)

```cmd
:: 1. venv aktiv?
where python
:: muss auf %USERPROFILE%\moloch_pc_env\Scripts\python.exe zeigen

:: 2. Repo-Stand pruefen
git status
git pull origin main

:: 3. Wichtige Files lesen die du aenderst (auch wenn schon vorher gelesen)

:: 4. Syntax pre-check (bei Python)
python -m py_compile pc\<dein_neues_file>.py
```

**Vor groesserer Aenderung an existierendem File**: erst Backup-Commit:
```cmd
git add <file> && git commit -m "BACKUP vor <was>"
```

---

## 3. Post-Flight (PC-Side, nach JEDER Code-Aenderung)

```cmd
:: 1. Syntax
python -m py_compile <datei>.py

:: 2. Self-Test laufen lassen (siehe Regel 4)
python <datei>.py    :: oder dedizierter test

:: 3. Wenn FastAPI-Service: neu starten
:: (manuell oder via Task Scheduler / nssm)

:: 4. Smoke-Test mit curl
curl http://localhost:11600/health

:: 5. Commit + Push pro abgeschlossener Aufgabe
git add <files>
git commit -m "<sprechende Message>"
git push
```

---

## 4. NEVER-Regeln (PC-Side, hart)

| # | Regel | Warum |
|---|-------|-------|
| N1 | **NIE Pi-Code editieren** (`core/`, `scripts/`, Pi-spezifisches in `docs/`) | Pi-Session ist Maintainer; Konflikte = chaos. Wenn du was vom Pi brauchst → Mailbox `docs/PC_TO_PI.md`. |
| N2 | **NIE Adapter ueberschreiben** — IMMER neue Version `vN+1` schreiben | Rollback per Adapter-File ist die einzige sichere Recovery. Letzte 5 behalten. |
| N3 | **NIE pending Samples trainieren** — nur `approved=true` aus `finetune_samples.jsonl` | Markus' Review-Gate ist heilig. Pending = nicht freigegeben. |
| N4 | **NIE blind GPU-Training versuchen** — GTX 760 ist Kepler/old | Bei CUDA-Errors auf CPU fallback. RAM hat 32 GB, das geht. |
| N5 | **NIE shell=True bei subprocess** | Command Injection Risk. Standard: arglist + timeout. |
| N6 | **NIE Adapter auf den Pi pushen** ohne dass Markus explizit sagt | Pi-NPU-Inferenz braucht HEF-Recompile (Welle 4) — bis dahin Adapter remote. |
| N7 | **NIE Markus-PC-Performance toten** — CPU-Limit 40% via `start /low` oder `wmic` priority | Markus arbeitet parallel auf seinem PC. |

---

## 5. Self-Test PFLICHT in jedem neuen Modul

Jede neue `.py`-Datei mit `if __name__ == "__main__":` Block der **mindestens** prueft:
- Init geht ohne Crash
- Kern-API liefert erwartetes Ergebnis (auch mit Mock-Daten)
- Validation: leere Inputs werfen kein Crash
- File-Operationen schreiben/lesen einen Round-Trip

Vorbild siehe Pi-Side `core/memory/feedback_store.py:330+` (Self-Test).

---

## 6. Backup-Strategie

- **Vor groesserem Refactor**: `git tag pre_<sprechender_name>` setzen
- **Pro Adapter-Run**: alte Version bleibt, neue als `vN+1` daneben (siehe N2)
- **Pro Major-Schritt**: eigener Commit mit klarer Message

---

## 7. Verfuegbare Skills/Agenten (auf Pi-Side, NICHT auf PC nutzbar — nur zur Info)

Pi-Session hat ein 17-Agenten-System (Vision, Tracking, Memory, Personality, etc.) das du **nicht aufrufen kannst** weil deine venv keinen Zugriff auf Pi-MCP hat. Aber wenn du etwas brauchst was in einer dieser Domains liegt:

| Domain | Was Pi-Session dort kann | Wie du es kriegst |
|--------|--------------------------|-------------------|
| memory | feedback_store, character_journal, character_patch | Mailbox-Request "Pi soll X im Memory-System aendern" |
| autonomy | finetune_orchestrator, distiller, llm_bridge | Mailbox |
| bridge | chat_server, critic_client, tentakel-routing | Mailbox |
| personality | mood/tension, patch-application | Mailbox |
| vision/tracking/audio/etc. | Pipeline, Camera, Audio | Mailbox (selten relevant fuer PC-Side) |

**Pragmatische Regel**: wenn du's selbst auf PC bauen kannst, mach's. Wenn es Pi-Datenstrukturen oder Pi-Endpoints braucht: Mailbox.

---

## 8. Cross-Session-Hygiene

- **Mailbox**: `docs/PC_TO_PI.md` (du schreibst), `docs/PI_TO_PC.md` (du liest)
- **Format/Status-Lifecycle**: siehe `docs/CROSS_SESSION_PROTOCOL.md`
- **Append oben**, nie ueberschreiben
- **Status updaten** wenn deine Anfrage von Pi beantwortet wurde
- **Markus rufen** (statt Mailbox) bei Blockern oder dringenden Themen

---

## 9. Audit-Aequivalent (PC-Side)

Pi hat einen `moloch_audit` mit 85 Tests. PC hat das nicht — aber baue dir einen kleinen Audit:

```cmd
:: pc\smoke.cmd
@echo off
python -c "import torch, peft, transformers, fastapi, uvicorn, pydantic" || exit /b 1
curl -s --max-time 3 http://localhost:11600/health || exit /b 1
python -c "import json; json.load(open('%USERPROFILE%\moloch_samples\samples.jsonl', errors='ignore'))" 2>nul
echo OK
```

Vor jedem Push: `pc\smoke.cmd` muss durchlaufen.

---

## 10. Was Markus dir vermutlich sagen will wenn du das nicht beachtest

- "Hat das wieder etwas zerschossen?"
- "Warum hast du nicht erst den Pi gefragt?"
- "Wo ist mein alter Adapter? Ich brauche Rollback."
- "Mein PC ist 95% CPU am Anschlag, ich kann nicht arbeiten."

Vermeide das. → Workflow respektieren.

---

*Read together with: `docs/THREEBRAIN_PC_SIDE_BRIEFING.md` (Aufgaben) und `docs/CROSS_SESSION_PROTOCOL.md` (Mailbox-Konvention).*
