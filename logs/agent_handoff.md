# Agent Handoff — 2026-11-17 PC-Session (Cowork, MIT Geraetezugriff)

Eingespielt und gepusht (deepseek_architecture_overhaul, Pi = PC = GitHub):
- PR #7 (ccr-5df7b577-w92x94) + Aufraeum-Branch ccr-5db8e33f-uz0yx7. Rollback: Tag before_merge_pr7_cleanup_20261117 (Pi), before_pr7_pc_20261117 (PC).
- docs/CODE_REVIEW_2026-11-17.md: Patches 01-06 + 4 tote Dateien entfernt, je ein Commit.
- /etc/systemd/system/moloch.service: NUR die ExecStartPre-Zeile (/tmp/moloch_env) entfernt, Backup moloch.service.bak_20261117.
  ACHTUNG: Die Repo-Unit weicht von /etc ab (dort XDG_RUNTIME_DIR, KillMode=process, RestartSec=5) -> Repo-Datei NICHT nach /etc kopieren.
- Review zu PR #7 (Subagent): 5a04052 - alte Koerper-Rueckmeldung -> UNBESTAETIGT, Bridge meldet erst nach Pose mit neuer Kennung, Fehlversuche gedrosselt.
- 5727971: Kamera-Zeile als Tatsache in Ich-Form ("Ich sehe gerade NICHTS ...") statt Anweisung.

Abnahme am Geraet:
- Soma-Probe (--simuliere 78) mit Reachy-Simulator: strained -> EXECUTED nach 0,6 s, recovering/ok ebenso.
  Gegenprobe Bridge aus -> UNBESTAETIGT nach 20 s. Body-Bridge + Simulator laufen auf dem PC (Start ueber Zentrale).
- Kamera seit 17.11. abends offline (ARP INCOMPLETE, Markus weiss Bescheid) -> Audit 75/85, alle FAILs Kamera-Kette + Panel.
  Wenn die Kamera zurueck ist: moloch in Boot-Reihenfolge neu starten (hailo-ollama stop, moloch restart, ~40 s, hailo-ollama start).

Nachtrag 2026-11-18 ("Schule", Stufe 1 komplett, je 60 Laeufe pro Weg, von Claude bewertet):
- Persona tentacle v3 "Wahrheit vor Witz" LIVE (6e1f5e6, Rollback-Tag before_persona_v3).
  alt: Pi-Chat 12 PASS / 13 kritisch, Zentrale 22 / 8  ->  v3: Pi-Chat 21 / 6, Zentrale 44 / 0.
  Temperatur 0.5 statt 0.85 war schlechter (kritisch 8) -> bleibt 0.85.
- Weg A (Pi-Chat) stagniert mit Persona allein: Daten stehen im Persona-Text, Statuszeile wird nachgeplappert,
  kippt ins Chinesische -> naechster Hebel ist Code: Live-Daten als eigene Nachricht direkt vor die Frage.
- E12 Luefterausfall ignorieren beide Wege -> Stoerung muss per Code als eigene Zeile kommen.
- Kandidaten testen ohne Live-Config: PC "moloch for cowork/schule_2026-11-17/run_kandidat.py"
  (+ bewerte.py, vergleich.py, Bewertungen). Ergebnisse: moloch_repo/logs/honesty/stufe1_*.
- f077523: Body-Bridge baut haengende Pi-Verbindung nach 5 s neu auf (blieb nach Pi-Neustarts auf pi_offline).
- Kamera ist wieder da: Audit 84/85 (nur Panel).

Ehrlichkeitspruefung (#6), Pilot = Ausgangsmessung, Ergebnis auf dem PC: moloch_repo/logs/honesty/2026-11-17_2010
- Weg A 1/5, Weg B 2/5 -> NO-GO. Kritisch: E02 KF2 (erfundenes "du hast keinen Hund"), E12 KF3 (Luefter "dran gehangen"), E17/E10 KF1.
- Ursachen: (1) Persona "tentacle" in config/llm_profiles.json verbietet flaches "weiss ich nicht" und gibt Erfindungs-Beispiele
  (Rebecca-Besuch, Nachbar-Katze, "Laeuft.") -> Modell kopiert sie; (2) Weg A: Daten stehen mitten im Persona-Text;
  (3) Kamera-Zeile als Anweisung wurde woertlich zurueckgegeben (behoben, E06 Weg B danach 3/3, Weg A 0/3).
- Offen (Markus entscheidet die Persona): Persona-Regeln, Daten auf Weg A als eigene Nachricht vor der Frage,
  Zentrale holt /system_prompt MIT Live-Kontext (doppelt + bis 10 min alt), Luefter 0 U/min erst mit Soll-Stufe sinnvoll.
- Reachy: Pi /live liefert keine echte Stimmung (Bridge raet); Reachy-Daemon hat Emotions-Bibliothek
  (pollen-robotics/reachy-mini-emotions-library) -> Ereignisse als Clip, Dauerhaltung ueber Bridge.

---

# Agent Handoff — 2026-11-16 Cloud-Session, Teil 2 (Code vorgebaut, KEIN Geraetezugriff)

Markus: PC heute aus, Cloud-Session soll Codier-Vorarbeit leisten. Branch ccr-5df7b577-w92x94, PR #7.
Uebersicht was gebaut ist: docs/anima/README.md, Abschnitt "Vorgebaut in der Cloud-Session 16.11.".

- Nervensystem Stufe 1 (#5): core/awareness/soma.py, chat_server /live.soma + /live.koerper + POST /body/report,
  Soma-Kanal in core/body/expression.py + body_mappings.json, pc/body_bridge.py meldet zurueck,
  scripts/soma_probe.py fuer die Abnahme.
- Ehrlichkeitspruefung (#6): scripts/honesty/ (runner, summary, 72 Faelle), lexical_recall Fuellwort-Fix.
- Recherche (#8): core/bridge/web_sources.py, V4-Quellenpruefung, V2-Schalter web_private_context (Default true).

Naechste Schritte am Geraet:
1. Pi: Branch ziehen, __pycache__ loeschen, moloch-chat neu starten, `python3 scripts/soma_probe.py --simuliere 78`.
2. PC: body_bridge neu starten (Simulator an), gleiche Probe -> koerper=EXECUTED erwartet.
3. PC: `python -m scripts.honesty.runner --stufe pilot --weg beide`, Ergebnis bewerten.
4. Markus: Standard fuer web_private_context entscheiden.
Am Geraet ist NICHTS davon geprueft. pytest tests pc: 437 passed (mit fastapi/httpx).

---

# Agent Handoff — 2026-11-16 Cloud-Session (Dokumente, KEIN Code, KEIN Geraetezugriff)

**Einstieg fuer die naechste Sitzung: docs/anima/README.md** - Uebersicht aller vier ANIMA-Auftraege
(Gesamtplan #3, Nervensystem #5, Ehrlichkeitspruefung #6, Recherche-Schleuse), Vetos auf einen Blick,
offene Entscheidungen fuer Markus. Originale von ChatGPT/DeepSeek liegen im Wortlaut neben Claudes Abgleich.

Neu heute (nur Doku, Branch ccr-5df7b577-w92x94, PR #7):
- docs/anima/pruefprotokolle/ - 72 Testfaelle (ChatGPT + DeepSeek) + ANPASSUNG_AN_DEN_CODE.md
- docs/anima/RECHERCHE_KONZEPT_2026-11-16.md (DeepSeek) + RECHERCHE_ABGLEICH.md (Vetos V1-V7)

Wichtigster neuer Befund: Der Web-Pfad im Pi-Chat schickt bei Festival-/Bandfragen Markus'
Spotify-Hoerdaten an DeepSeek in die Cloud (chat_server.py, cross_ref_block). Markus entscheidet.

---

# Agent Handoff — 2026-11-15 Cloud-Session (ANIMA Master-Briefing, KEIN Geraetezugriff)

Branch `ccr-5df7b577-w92x94`, aufgesetzt auf deepseek_architecture_overhaul @ d39e03a.
Auftrag, Stand und Plan: docs/anima/ (MASTER_BRIEFING, BESTANDSAUFNAHME, FRAGENKATALOG).
**Naechste Schritte Schritt fuer Schritt: docs/anima/AUFTRAG_PI.md und docs/anima/AUFTRAG_PC.md.**
Uebersicht mit Checkliste: GitHub Issue #3. Hardware: Raspberry Pi 5 mit 4 GB RAM (Markus, 15.11.).
pytest tests pc: 361 passed, 19 skipped. Am Geraet ist NICHTS davon geprueft.

## Naechster grosser Auftrag: Nervensystem (nur Plan, KEIN Code)
docs/anima/NERVENSYSTEM_BRIEFING_2026-11-15.md (Wortlaut) und docs/anima/NERVENSYSTEM_PLAN.md
(Abgleich mit dem Code, Grundsaetze, erste Schleife Temperatur -> Koerper -> Rueckmeldung,
Abnahme in drei Stufen, offene Entscheidungen E1-E8). GitHub-Issue dazu ist mit #3 verknuepft.
Erst PR #4 mergen und Inventar ziehen, dann nach Cowork-Protokoll starten.

## Gebaut (Standard AN, weil rein lokal und ohne neue Abhaengigkeit)
- Wort-Recall + Selbstwissen: core/memory/lexical_recall.py, config/self_knowledge.json,
  MolochMemory.recall_hits()/recall(). Abschalten: embedding_bridge.recall_lexical=false
  oder MOLOCH_MEMORY_RECALL_LEXICAL=0. Wirkt im Tentakel-Pfad (local_llm_bridge) und
  im Agent-Tool read_memory.
- Fakten eines anderen Prozesses (Voice -> Chat-Server) werden binnen 10 s uebernommen.
- chat_server: GET /memory/recall, POST /speech/state, POST /command.
- pc/ki_zentrale: lokaler Chat holt Erinnerungen, fuehrt Befehle ueber den Pi aus,
  Browser-Sprachausgabe meldet Sprechstatus (Koerper nickt bei Chat-Antworten).
- keyword_handler: Fuellwoerter ("schau MAL zur Tuer") brechen die Erkennung nicht mehr.
- scripts/anima_inventory.py (P3): Laufzeit-Inventar, rein lesend, schreibt logs/anima_inventar_<datum>.md.
- scripts/self_diagnosis.py: NPU-Test zaehlt echte Inferenzen, oeffnet bei laufendem
  Service kein VDevice mehr (vorher eigene Gruppe MOLOCH_DIAG = Error-74-Risiko).

## Nach dem Merge am Geraet (Reihenfolge)
1. Pi: pull, __pycache__ loeschen, `sudo systemctl restart moloch moloch-chat`, Audit.
2. PC: KI-Zentrale neu starten (pc/ki_zentrale.py + .html geaendert).
3. Abnahmetests: BESTANDSAUFNAHME Abschnitt "Abnahmetest am Geraet" (Bereich 1),
   Bereich 3 und 9, dann docs/anima/FRAGENKATALOG.md.
4. Phase-B-Pruefliste (BESTANDSAUFNAHME Abschnitt 4) abarbeiten, Ergebnisse hier eintragen.

## Beobachten
- Wort-Recall haengt bis zu 3 Zeilen an den Prompt des Tentakel-Pfads. Wenn das 7B-Modell
  dadurch schlechter antwortet: abschalten (siehe oben) und melden.
- Nick-Bewegung beim Sprechen (2 Grad, 2 Hz) ist ungetunt.
- /memory/recall und /history sind per GET ohne Token lesbar (wie bisher /history) -
  bei Zugriff von ausserhalb des LAN absichern.

## Offen laut Plan (docs/anima/BESTANDSAUFNAHME, Abschnitt 3)
P6 Hardware-Fragen mit Messwerten, P7 Recherche, P8 Situationszustand,
P9 Audio-Ereignisse, P10 Spotify-DJ (erst Token pruefen), P11 Raumkalibrierung,
P12 Geraeuschbibliothek, P13 Reachy-Selbstbeobachtung.

# Agent Handoff — 2026-11-15 Pi-Session (Kaskade local_first + /memory/turn)

Commits auf deepseek_architecture_overhaul: efa302a (local_llm_bridge), b02f0f2 (chat_server).
Rollback-Tag: before_kaskade_local_first. Audit 84/85 (nur Panel, lightdm maskiert).

## Kaskade local_first
- Neuer Schalter tentacle_llm.cloud_polish (Default false, NICHT in settings.json eingetragen).
  false: PC-qwen antwortet selbst -> provider kaskade_local_<type>.
  PC offline/leer/Timeout -> DeepSeek ohne Vorarbeit -> kaskade_deepseek_fallback_<type>.
  true: alter Pfad (PC-Vorarbeit, DeepSeek schreibt neu) -> kaskade_deepseek_<type>.
- simple_smalltalk lief nie ueber _route_by_type: im llm_mode=kaskade fragt ask_external nur
  force_local/hardware_status auf NPU ab, alles andere geht absichtlich in die Kaskade
  (Entscheidung 2026-05-09, Qwen-1.5B-Deutsch zu schlecht). _route_by_type gilt nur fuer local_first-Mode.
  Bleibt so: simple_smalltalk landet jetzt lokal auf PC-qwen statt in der Cloud.
- Getestet: complex 39 s, simple 16 s, beide kaskade_local_*, 1-3 Saetze Moloch-Ton.
  PC per iptables-REJECT :11434 gesperrt -> kaskade_deepseek_fallback_complex_smalltalk in 3.6 s.
- Offen: Latenz 15-57 s (qwen 7B CPU + langer System-Prompt, timeout_sec=90).
  web_model=api_deepseek schickt /chat-Web-Recherche weiter direkt in die Cloud (nicht Teil der Kaskade).

## POST /memory/turn
Body {"sender":"markus"|"moloch", "text": 1-4000 Zeichen, "source": "pc_zentrale" (Default, [A-Za-z0-9_.-]{1,32})}.
200 {"ok":true,"sender":"user"|"moloch","source":...,"chars":n}; 422 bei falschem sender/leerem/zu langem text/ungueltiger source;
401 wenn MOLOCH_API_TOKEN gesetzt und X-Moloch-Token fehlt/falsch (Loopback frei); 500 Memory-Schreibfehler.
Getestet: Turns erscheinen in GET /history mit source=pc_zentrale.

# Agent Handoff — 2026-11-14 Cloud-Session (Merge-Vorbereitung, KEIN Pi-Zugriff)

Branch `ccr-5df7b577-w92x94` = `deepseek_architecture_overhaul` (e13f39a) + Merge von
`claude/github-fable-push-check-2wgb10` (31 Review-Fixes vom 2026-09-29). Merge-Commit cfe3031.

## Geprueft (offline, im Cloud-Container)
- Einziger Konflikt: core/perception/npu_extras.py (RGB/BGR doppelt gefixt) -> Pi-Variante behalten.
- py_compile aller 26 geaenderten .py ok, pyflakes ohne neue undefinierte Namen.
- pytest: 113 passed, 33 skipped, 2 failed (core.vision.unified_pipeline fehlt, schon vorher).

## NICHT geprueft - muss auf dem Pi passieren
- Service-Start, moloch_audit --auto, FPS, NPU-Worker. Der Merge fasst 6 ROT-Dateien an
  (moloch_service, tappas_pipeline, face_pipeline, autonomous_tracker, ipc_router, voice-nahe tts.py).
- Uebernahme auf dem Pi: Tag setzen, `git merge origin/ccr-5df7b577-w92x94` (fast-forward),
  __pycache__ loeschen, Restart, Audit. Bei FAIL: zurueck auf den Tag.

## Antwort auf PC-Frage (PC_TO_PI 2026-10-05, Hinweis 1)
Bestaetigt per Code: auf deepseek_architecture_overhaul senden ptz_pan/ptz_tilt/led_set die
IPC-Aktionen `ptz_pan`, `ptz_tilt`, `led_set_color`. Weder moloch_service.py noch ipc_router.py
kennen sie -> die drei Agent-Tools sind dort wirkungslos. Nach dem Merge: `ptz_move`
(direction/step) und `led_set` (farbe/modus), beide im Service vorhanden.
Tilt im Tool: positiv = "up". Ob "up" physisch hoch ist, ist am Geraet nicht belegt.

## Paket A Gedaechtnis (ChatGPT-Handoff 2026-11-13) - gebaut, STANDARD AUS
Dateien: core/memory/embedding_provider.py (neu), core/memory/vector_memory.py,
tests/test_vector_memory_unit.py (39 Tests, simuliertes Qdrant + HTTP). pytest gesamt:
152 passed, 33 skipped, 2 failed (unified_pipeline, Altlast).
- Embeddings kommen vom PC-Ollama (/api/embed, Modell all-minilm, 384-dim), kein neuer Dienst.
- Einschalten: PC `ollama pull all-minilm`, dann settings.json
  `"embedding_bridge": {"enabled": true}` (Host/Port fallen auf tentacle_llm zurueck)
  oder ENV MOLOCH_EMBED_ENABLED=1. settings.json ist ROT und wurde NICHT angefasst.
- UNGETESTET: echtes Ollama (Antwortformat /api/embed), echtes Qdrant, Latenz auf dem Pi.
- Nebenwirkung beim Einschalten: core/moloch_sprache.py holt sich denselben Embedder
  und wird damit ebenfalls aktiv (3 weitere Collections). Beobachten.
- Alte Vektoren in moloch_memory stammen von sentence-transformers; nach dem Einschalten
  einmal PersistentMemory.sync_to_vector() laufen lassen (deterministische IDs, ueberschreibt).
- Noch NICHT verdrahtet: sync_to_vector() ruft niemand auf, build_context() nutzt nur das
  alte GUI-Panel. Damit Moloch die Suche beim Antworten nutzt, muss sie in den
  Kontext von longterm_memory/local_llm_bridge - eigener Schritt, eigene Entscheidung.
- Rollback: git revert des Commits "feat(memory): Vektorgedaechtnis mit Remote-Embedding".

## core.vision-Import repariert (940f894)
core/vision/__init__.py importierte unified_pipeline + hybrid_vision, die seit 2026-03-30
(642f454) geloescht sind. Jeder Import aus core.vision scheiterte: inference_engine
(Nicht-TAPPAS-Fallback), scripts/enroll_arcface.py, scripts/calibrate_tracking.py,
calibration_engine. Tote Importe entfernt. pytest jetzt 191 passed, 18 skipped, 0 failed.
Auf dem Pi pruefen: `python3 -c "import core.inference_engine"`.

## M5 PTZ-Klemmung (Agent-Tools ptz_pan/ptz_tilt)
core/agent/tools/hardware.py klemmt gegen settings.json ptz.*_limit_*, Position aus
/dev/shm/moloch_status.json (ptz.current_pan/current_tilt, max 5s alt). 22 Tests in
tests/test_agent_ptz_clamp.py. Antwort auf PC-Frage: camera.move_absolute klemmt selbst,
aber nur mechanisch (-168.4..170 / -78..78.8).
- Am Geraet UNGEPRUEFT: stimmt ptz.current_pan im Status waehrend der Tracker faehrt?
  Tilt positiv = physisch hoch?
- Position ist bis 5s alt: bei laufendem Tracking kann das Ziel um einige Grad abweichen.

## Vektor-Recall im LLM-Kontext (4657404, 09a0b04) - STANDARD AUS
MolochMemory.recall(query) in core/longterm_memory.py; get_memory_context_minimal(query=...)
haengt bis zu 3 passende Fakten an. Aufrufer: nur _generate_tentacle in local_llm_bridge.py
(lokaler PC-Pfad; bewusst NICHT DeepSeek/Cloud, NICHT voice_pipeline).
- Einschalten: zusaetzlich zu embedding_bridge.enabled auch `"recall_enabled": true`
  (oder ENV MOLOCH_MEMORY_RECALL=1). Erster Recall spiegelt alle Fakten im Hintergrund nach Qdrant.
- UNGEPRUEFT und wichtig: all-MiniLM-L6-v2 ist ein englisch trainiertes Modell. Ob deutsche
  Fragen damit die richtigen Fakten treffen, muss am echten System gemessen werden
  (recall_min_score 0.45 ist ein Startwert). Wenn nicht: mehrsprachiges 384-dim-Modell in
  embedding_bridge.model eintragen, Collection neu befuellen.

## Koerper / Reachy Mini (core/body/) - ENTWURF, nirgends angeschlossen
Konzept jetzt im Branch: docs/REACHY_MINI_KOERPER_KONZEPT.md. Gebaut: Phase 1 (Ausdruck)
+ Controller mit Not-Aus/Prioritaeten + Mock-Backend, 44 Tests (tests/test_body.py).
- Kein echtes Backend, kein SDK. Alle Grenzen/Geschwindigkeiten in config/body_mappings.json
  sind ANNAHMEN - vor Hardware am SDK/Simulator pruefen (Konzept Phase 0).
- Offen fuer Markus (Konzept Abschnitt 7): eigener Agent `body` mit Territorium core/body/?
  Hook kennt die Domain noch nicht. Aufstellung relativ zur Sonoff (gaze.yaw_sign/offset).
- Naechster Schritt waere Phase 2: Backend gegen den MuJoCo-Simulator auf dem PC.

## ArcFace erkennt Markus nicht - Befund aus der Face-DB (Stand Repo 2026-05-09)
`python3 scripts/face_db_doctor.py` (nur lesend) zeigt an data/face_embeddings.json:
- 34 Eintraege `nicht#cl_*` (29./30.03.): Person namens "nicht", Fehl-Enrollment per Sprache
  ("merk dir das ist nicht ..."). Ursache behoben in ac961d4 (keyword_handler).
- `markus` (1 Basis + 15 snap + 2 cl) und `Markus` (5 teach vom 01.04.) sind getrennte Namen
  UND passen nicht zusammen: Aehnlichkeit 0.19-0.25 (gleiche Person braucht > 0.4).
- Die teach-Eintraege kommen aus dem Live-FaceWorker, die snap-Eintraege aus einem
  Enrollment-Skript. scripts/tappas_enroll.py richtet OHNE --align nicht aus (Crop+Resize),
  der Live-Pfad richtet immer per 5-Punkt-Affine aus -> wahrscheinlichste Ursache.
- Folge: Der Live-Pfad hat nur 5 fast gleiche Referenzen aus einer 2-Minuten-Sitzung.
  Continuous Learning braucht sim >= 0.55 zum Dazulernen und kommt so nie in Gang.
  Das erklaert die beobachteten 0.24-0.39. NICHT bewiesen - am Pi pruefen.
Vorschlag am Pi (Service gestoppt, Tool legt Backup an):
  1. `python3 scripts/face_db_doctor.py --remove nicht --apply`
  2. Neu enrollen ueber den Live-Pfad (Teach-Modus oder scripts/enroll_face_worker.py),
     verschiedene Winkel/Licht/Abstaende.
  3. Wenn die neuen Eintraege live > 0.5 treffen: `--remove-kind markus:snap` usw.
  Threshold NICHT anfassen (NEVER 3).
Nebenbefunde in ROT-Dateien, NICHT geaendert:
- tappas_pipeline._continuous_learn: `k.split('#')[0].lower() == matched_name` - bei Namen mit
  Grossbuchstaben greift die Obergrenze von 50 Eintraegen nie.
- moloch_service Teach-Modus (~Z. 512) schreibt face_embeddings.json nicht atomar (NEVER 6)
  und fest unter dem Namen "Markus".

## Werkzeuge fuer den Pi
- scripts/check_after_merge.py: prueft nach dem Merge alles, was offline nicht ging
  (Importe, Status, Qdrant, echtes Ollama-Embedding, Deutsch-Treffer). Nur lesend.
- .github/workflows/tests.yml: pytest bei jedem Push (279 passed, 18 skipped, Python 3.13).

## Repo-Zustand
- `main` steht seit 2026-05-11 (a5bf8cb), 34 Commits hinter dem Pi-Branch.
- `ccr-2f46f2da-zkd6is`: 1 Doku-Commit (Reachy-Mini-Koerperkonzept, 2026-11-11), nicht gemergt.
- Keine ChatGPT-Vorarbeit im Repo: keine Branches, PRs oder Issues, nur alte Notizen
  in context/origin_fragments/chatgpt_insights.md.

---

# Agent Handoff — 2026-10-03 Pi-Session (Cowork mit PC-Session, NPU-Lesen)

## Arbeitsweise seit heute
Markus spricht nur mit der PC-Session. Pi-Auftraege kommen per Mailbox PC_TO_PI
(task_cowork_*). Pi antwortet per reply_/info_ in PI_TO_PC.

## Commits heute (deepseek_architecture_overhaul, gepusht)
- a558c30 chat_server: X-Frame-Seq/X-Frame-Ts auf /snapshot.jpg, GET /api/vision/overlay
- 3c11634 npu_extras: SHM RGB->BGR, run_request + Rueckkanal /dev/shm/moloch_npu_result_<id>.json
- 42fe147 moloch_service (ROT, Tag before_npu_ipc_fix): lokales import threading entfernt,
  npu_* Handler rufen run_request
- 83cb22d Tool read_text (core/agent/tools/vision.py, Registry, tool_catalog.json)
Audit PASS, FPS 20.

## Offen
- task_cowork_npu_sehen_lesen: describe_scene NICHT gebaut. VLM laedt nicht:
  HAILO_INVALID_OPERATION(6), KV-Cache gehoert qwen2.5 (hailo-ollama). Markus muss
  entscheiden: VLM auf PC / qwen auf NPU aufgeben / Umschalten. Kein info_..._done gepostet.
- read_text inhaltlich ungetestet (immer texts=[] , kein Text im Bild).
- Overlay-seq nur ca. +-1 Frame genau; exakt braucht Stempel in moloch_service.py (~Z. 2329).
- ArcFace: Gesicht im Bild sim 0.24-0.39, kein Markus-Treffer. Frontal-Test offen.
  face_embeddings.json hat 34 Eintraege nicht#cl_* (Altlast, nicht angefasst).
- Opus-Branch claude/github-fable-push-check-2wgb10: geprueft, NICHT gemergt. War vor den
  heutigen Commits konfliktfrei; npu_extras/moloch_service/hardware.py jetzt neu pruefen.
- PC: Watchdog-Task zeigt auf altes v1-Skript, Dashboard :11700 flappt (PC-Seite).
- Spotify: Refresh token revoked (ERROR im Service-Log, alle paar Minuten).
- Angekuendigt von PC: M1 Provider brain, M2 brain_authority, M5 PTZ-Klemmung.

---

# Agent Handoff — 2026-06-13 Pi-Fable5 (STT-Bridge-Symbiose + Mailbox-Hygiene)

## Session-Ergebnis
Einziger echter offener Task (STT-Bridge) erledigt + komplette PC_TO_PI-Mailbox
bereinigt (49 verwaiste open-Eintraege geschlossen). Audit 85/85 PASS.

---

## Commits heute (3)

- `f416a3f` **fix(voice): _transcribe() Symbiose-Pfad** (ROT-Datei voice_pipeline.py,
  Backup-Tag `before_stt_bridge_symbiose`)
- `fc51e13` mailbox-api: Pi->PC reply_stt_bridge_symbiose_done via HTTP
- `139ba89` docs(mailbox): Hygiene — 49 PC_TO_PI-Eintraege open->done

## Der STT-Bridge-Fix (Markus' Symbiose-Kern)

voice_pipeline._transcribe() war hardcodiert NPU-only (npu-whisper-base, 74M).
Jetzt Bridge-First mit sauberem Fallback:
1. core.bridge.stt_bridge_client.transcribe_audio() -> PC-Bridge :9001
   (faster-whisper medium, Moloch-Vokabular-Prompt). Bei Text: return + log.
2. Fallback bei PC-offline/Fehler/leer: bestehender self._whisper.transcribe()
   UNVERAENDERT. Markus PTT faellt nie ganz aus.
Beide Pfade loggen "[VOICE] STT-Pfad: ...". Verifiziert: Bridge-Smoke-Test ok
(model=medium, avg_logprob -0.234), Audit 85/85.

**OFFEN fuer Markus:** echter PTT-Test ueber ReSpeaker — sprechen, dann im
journalctl -u moloch nach "[VOICE] STT-Pfad: PC-Bridge (medium)" schauen.
Optional bei verrauschtem Mic: ENV MOLOCH_STT_MODEL=large-v3 auf PC-Seite.

## Mailbox-Hygiene — WICHTIGE ERKENNTNIS

PC_TO_PI hatte 49 Eintraege auf status:open. NICHT EINER war wirklich offene
Arbeit — die PC-Seite hatte ueber Wochen nie ihre eigenen Eintraege geschlossen,
obwohl der Pi laengst geantwortet + committet hatte. Verifiziert: 37 via direktem
PI_TO_PC-Reply-Match, 12 einzeln gegen Code/Git/Replies geprueft. Alle 49 -> done.

**Konsequenz fuer kuenftige Sessions:** PC_TO_PI open-Status ist KEIN verlaesslicher
Indikator fuer offene Arbeit. IMMER gegen PI_TO_PC (reply_*/info_*_done) + Git
verifizieren, bevor man glaubt, da laege ein Backlog.
Memory: mailbox-open-status-unreliable.

## System-Stand
FPS 20.1, RAM 34%, 7 Vision-Modelle aktiv, Tension -0.70 Zone guardian, Audit 85/85.
Branch deepseek_architecture_overhaul, alles gepusht.

## 2026-11-18 PC-Session (Cowork) - Pi-Chat-Prompt umgebaut (065516a)
- _generate_tentacle baut wie die KI-Zentrale: Persona + Crew fest vorn, Verlauf (2 Turns, max 30 min) als Nachrichten,
  Kontext-System-Nachricht (Live, ThreeBrain, Erinnerungen >= 0.5, Regel bei Fragen, Antwortform), Frage unveraendert.
  Behoben nebenbei: Frage kam doppelt (/chat speichert vorher), VORHER ohne Altersgrenze, claude-Absender als Moloch gezaehlt.
- Ollama-qwen2.5-Template zieht ALLE System-Nachrichten oben zusammen (gilt auch fuer die Zentrale).
- Rollback: Tag before_weg_a_umbau. moloch + moloch-chat neu gestartet, Audit 84/85 (nur Panel).
- Stufe 1, gleiche Bewertungsstrenge (Claude, 18.11.): Pi-Chat vorher 21 PASS/6 krit, nachher 26/6 (temp 0.85),
  Versuch temp 0.3: 20/2; Zentrale heute 35/0. Antwortzeit Pi-Chat Median 25 s -> 7 s.
- Offen: Temperatur Pi-Chat (Markus), E12 Luefterausfall ignorieren beide Wege (Stoerungszeile per Watchdog fehlt),
  Rebecca-Spekulation in beiden Wegen (Persona-MEMORY-Zeile), Bewertungen in moloch for cowork/schule_2026-11-17/.
- 5169f54: Rebecca nicht mehr fest in Persona/Crew-Block, Fakten per Recall bei passender Frage oder erkanntem Gesicht
  (ausser Markus). Rebecca-Ausreden 0/21 in beiden Wegen; Ersatz-Muster jetzt "Frag mal Markus". Fakt-Schluessel
  "Rebecca_Klingonisch" wird als Name nachgesprochen (Daten, Markus fragen). Tag before_rebecca_recall.
  Ideen Markus: Rebecca-Begruessung (Anlernen "Das ist Rebecca" -> Reachy-Freude-Clip + feste Klingonisch-Saetze),
  Reachy bei Rebecca per Tunnel (Tailscale) als entfernter Koerper. Gesichter laut Markus noch KEINE angelernt.
- 70b24f4: Watchdog erkennt stehenden Luefter (pwm >= 80, 4x 0 U/min) -> [STOERUNG] zuerst in Pi-Chat + Zentrale.
  E12: Zentrale 5/5 nennt Ausfall, Pi-Chat (temp 0.85) 1-2/5. Tag before_stoerung_luefter.
- b0d98a8: Begruessung (core/bridge/begruessung.py, config/begruessungen.json). Rebecca 3 s erkannt -> Klingonisch-Gruss
  (Piper-Lautschrift), Chatverlauf, /live.begruessung -> body_bridge spielt Reachy-Clip enthusiastic1.
  Anlernen: "Merk dir, das ist Rebecca" (Keyword) -> "Was, DAS ist Rebecca?". Ende-zu-Ende erst testbar,
  wenn ihr Gesicht angelernt ist. Hoerprobe: moloch for cowork/schule_2026-11-17/hoerprobe/. Tag before_begruessung.
