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
