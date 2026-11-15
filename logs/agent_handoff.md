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
