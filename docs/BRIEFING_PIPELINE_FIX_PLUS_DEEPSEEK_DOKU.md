# BRIEFING — fuer naechste Pi-Claude-Session (Copy-Paste-fertig)

**Erstellt: 2026-04-28 07:30 von Pi-Claude-Session #X**
**Pi HEAD bei Erstellung: `e44fb12`**

---

## Was du tust (in dieser Reihenfolge)

1. **Pflicht-Startprotokoll** ausfuehren
2. **Aufgabe A** — Pipeline-Lag fixen (akut, ~10min)
3. **Aufgabe B** — 5 DeepSeek-Diagnose-Files schreiben (~30-45min)
4. **Mailbox + Handoff** updaten

---

## 0. Pflicht-Startprotokoll (BEVOR Code)

```bash
# MCP via Tools (nicht SSH):
moloch_session_init     # entfernt /tmp/moloch_session_lock
moloch_status           # FPS, Worker, NPU
moloch_npu_workers      # Pipeline-Worker-Detail

cd ~/moloch
git fetch -q origin main && git log --oneline -5
head -30 docs/PC_TO_PI.md   # neue PC-Eintraege?
```

**STOP wenn:** SESSION_READY=false ODER Audit-FAIL ODER FPS<10.

---

## 1. Aktueller Stand (Pi-Side, Stand 28.04 07:25)

**Live-Symptom:**
- FPS auf 13.2 (statt 20)
- Frame-Age 0.80s (statt <0.2s)
- FaceWorker last_ms = **5947ms** (statt ~80ms)
- PoseWorker last_ms = **6473ms** (statt ~100ms)
- ROI-Dispatcher dropped **24%** der Frames (50279/204421)
- `moloch_status.json.panel_detections` = **leer** → keine BBoxen, keine Face-Landmarks im GUI
- moloch.service PID 434031 zieht **149% CPU**

**Was bereits gemacht wurde (NICHT nochmal):**
- Session 27 (gestern): Identity+Hardware Halluzinations-Fix (`ef09a24`)
  → `config/hardware_facts.json` neu, `_build_identity_block()` und
  `_build_telemetry_footer()` in `core/autonomy/local_llm_bridge.py`,
  `chat`-Profile feinjustiert.
- Session 30 (heute morgen): Hailo-Treiber-Audit (`e44fb12`)
  → kein ABI-Mismatch, custom-SOs OK, Phase B (5.1→5.3) wontfix,
  Phase D (orphan driver-tree) done.
- v2-LoRA-Training durchgelaufen (`6d88cce` auto v2_live mailbox).

**Wichtig**: Hailo-Treiber sind NICHT die Ursache. Treiber-Stack ist sauber.

**Hauptverdacht zum Pipeline-Lag:**
`_build_telemetry_footer()` in `local_llm_bridge.py` (eingefuehrt gestern)
ruft `subprocess.run(["vcgencmd", "measure_temp"])` als Subprocess. Bei
jedem LLM-Call (Chat, /infer, /critic_review) wird ein neuer Subprocess
gespawnt. PC-Dashboard polled `/state_full` haeufig — wenn das die
Bridge triggert, hagelt's Subprocesses → Pipeline blockiert.

---

## 2. Aufgabe A — Pipeline-Lag fixen

**Domain:** autonomy. **Lock:** `touch /tmp/moloch_agent_autonomy`.

**Pre-Flight:**
```bash
cd ~/moloch
git tag pre_telemetry_cache_$(date +%Y%m%d_%H%M)
python3 -c "from core.autonomy.local_llm_bridge import _build_telemetry_footer; print(_build_telemetry_footer())"
# verifiziert dass Footer-Funktion existiert
```

**Edit:** `core/autonomy/local_llm_bridge.py` — `_build_telemetry_footer()`
mit **5s-mtime-Cache** abdaempfen:

```python
_TELEMETRY_CACHE: Dict[str, Any] = {"text": "", "ts": 0.0}
_TELEMETRY_CACHE_TTL_S = 5.0  # Refresh max alle 5s

def _build_telemetry_footer() -> str:
    """Live-Telemetrie als Footer mit 5s-Cache (verhindert Subprocess-Storm)."""
    now = time.time()
    if now - _TELEMETRY_CACHE["ts"] < _TELEMETRY_CACHE_TTL_S:
        return _TELEMETRY_CACHE["text"]
    parts = ["\n=== LIVE-TELEMETRIE (jetzt gemessen) ==="]
    # ... (bestehender Code: vcgencmd, hwmon, /proc/meminfo, feedback_store)
    text = ("\n".join(parts) + "\n") if len(parts) > 1 else ""
    _TELEMETRY_CACHE["text"] = text
    _TELEMETRY_CACHE["ts"] = now
    return text
```

**Plus**: bei `_build_identity_block()` ist Cache schon vorhanden
(mtime-basiert), das ist OK.

**Verifikation:**
```bash
find ~/moloch/core -name "__pycache__" -exec rm -rf {} + 2>/dev/null
sudo systemctl restart moloch moloch-chat moloch-chat-https
sleep 30
moloch_status  # FPS sollte zurueck Richtung 20 sein
moloch_npu_workers  # last_ms sollte unter 200ms sein
python3 -c "
import json
d = json.load(open('/dev/shm/moloch_status.json'))
pd = d.get('panel_detections', [])
print(f'panel_detections count: {len(pd)} (sollte > 0 wenn Person im Bild)')"
```

**Akzeptanz A:**
- FPS >= 18
- Face/Pose-Worker last_ms < 200ms
- panel_detections > 0 wenn Person im Bild
- Audit 85/85 PASS
- Bei FAIL: rollback via `git checkout pre_telemetry_cache_*`

**Commit:**
```bash
GIT_AUTHOR_NAME="Cowork Pi-Side" GIT_AUTHOR_EMAIL="cowork-pi@moloch.local" \
GIT_COMMITTER_NAME="Cowork Pi-Side" GIT_COMMITTER_EMAIL="cowork-pi@moloch.local" \
  git -c gpg.sign=false commit -m "fix(local_llm_bridge): _build_telemetry_footer 5s-Cache - Pipeline-Lag-Fix"
git pull --rebase && git push
rm /tmp/moloch_agent_autonomy
```

---

## 3. Aufgabe B — 5 DeepSeek-Diagnose-Files schreiben

**Markus' Direktive:** "Schreibe in `docs/deepseek_briefing/` 5 Markdown-Files
die ich im Editor anklicken kann + an DeepSeek geben. DeepSeek soll Moloch
besser verstehen + Rat geben wie wir's anders aufstellen koennen."

**Output-Verzeichnis:**
```bash
mkdir -p ~/moloch/docs/deepseek_briefing
```

### B1. `01_prompt_dreischicht.md`

**Inhalt:** Wortlaut (NICHT Konzept) der drei Prompt-Schichten:
- `base_fix` — Hardware-Identity-Block aus `config/hardware_facts.json`
  (rendered durch `_build_identity_block()` in
  `core/autonomy/local_llm_bridge.py`)
- `semi_fix` — `chat`-Profile.system aus `config/llm_profiles.json`
  (das aktive Profile, ggf. plus `tentacle`-Profile.system zum Vergleich)
- `fluid_layer` — `_build_local_context_snippet()` Output:
  Person/Face/Zone/Tension + History letzte 2 Turns + Telemetrie-Footer

**Wie zusammensetzen:**
```python
# Live-Snapshot ziehen — was geht aktuell wirklich an die LLM
from core.autonomy.local_llm_bridge import (
    _build_identity_block,
    _build_local_context_snippet,
    _get_active_profile,
)
import json
profile = _get_active_profile() or {}
print("=== base_fix ===")
print(_build_identity_block())
print("=== semi_fix (chat-profile system) ===")
print(profile.get("system", ""))
print("=== fluid_layer (live snippet) ===")
print(_build_local_context_snippet())
```

Output dieses Skripts ist der Inhalt des Markdown-Files (mit ```...``` Bloecken
geframed je Schicht). Plus: am Ende eine Erklaerung welche Schicht wann
greift (Profile-Vorrang ueber Caller-System siehe local_llm_bridge.py
Zeile ~720, code-fence im Markdown).

### B2. `02_tension_core_mechanik.md`

**Inhalt:** Wie wird Tension berechnet? Variablen, Drift, Trigger als Code
(nicht Konzept).

**Quellen** (lesen + relevante Bloecke einfuegen):
- `core/core_integrator.py` — `tension`, `dominance`, `presence`,
  `_update_tension`, `get_effects()`, `get_status_dict()` (Zeile 861-881)
- `core/personality/tension_integrator.py` — falls vorhanden
- `core/autonomy/character_distiller.py` — Drift-Berechnung
  (`half_life=7d`, `recency_weighted_top`)
- `config/anima_mappings.json` falls da Trigger drin sind

**Format:** Code-Bloecke aus diesen Files mit Zeilen-Annotation. Plus
ein Pseudocode-Diagramm:
```
tension(t) = decay(tension(t-1)) + sum(events_since_last_tick * weight)
events: tension_delta aus character_journal Eintraegen + EventBus 'tension_*'
zone = guardian if tension < 0.3 else shadow if < 0.6 else berserker
effects = derived_from(tension, dominance, presence) — see get_effects()
```

### B3. `03_npu_pipeline.md`

**Inhalt:** Welche Hailo-Modelle laeuft, Ausgabeformat, wie landet das im Prompt.

**Quellen:**
- `core/perception/tappas_pipeline.py` — Pipeline-Aufbau,
  `YOLO_POSTPROCESS_SO`, `SCRFD_POSTPROCESS_SO` etc. (Zeile 66-89),
  Pipeline-String / Gst.parse_launch
- `core/perception/vision_workers.py` — die 4 HailoRT-Direct-Worker
  (FaceWorker, PoseWorker, ReIDWorker, DepthWorker)
- `CLAUDE.md` Pipeline-Architektur-Sektion (kopieren als Uebersicht)
- Aufgaben-Beschreibung pro Modell:
  ```
  yolov11m_h10.hef       — Stage1, Person/Object Detection
  scrfd_10g.hef          — Stage2 Face-Detection (5 Landmarks)
  arcface_mobilefacenet.hef — Face-Embeddings (512D)
  yolov8s_pose_h10.hef   — Pose-Keypoints
  repvgg_a0_person_reid_512.hef — ReID
  scdepthv3.hef          — monokulare Tiefe
  faceattr.hef           — Alter/Geschlecht
  ```
- Datenfluss zum Prompt:
  ```
  TAPPAS GstBuffer -> hailo.HAILO_DETECTION + HAILO_LANDMARKS
   -> tappas_pipeline.py _on_buffer (Zeile suchen, ~1880er)
   -> moloch_status.json.panel_detections + face_id + face_similarity
   -> _build_local_context_snippet() liest /dev/shm/moloch_status.json
   -> Prompt-Footer "DU SIEHST GERADE: ..." + "Innen: schaerfe=..."
  ```

### B4. `04_journal_beispiele.md`

**Inhalt:** 2-3 echte Eintraege aus `character_journal.jsonl` (PII redacted
falls Marlene/Markus-spezifisches drin).

**Quelle:**
```bash
ls -la /mnt/moloch-data/memory/journal/
# nimm den letzten Tag, sample 2-3 Eintraege:
tail -10 /mnt/moloch-data/memory/journal/$(date +%Y-%m-%d).jsonl 2>/dev/null
# oder Vortag
ls /mnt/moloch-data/memory/journal/*.jsonl | tail -1 | xargs tail -5
```

**Format:** Roh-JSON-Lines mit kurzer Erklaerung pro Feld:
```
{"event_id":"evt_00000123", "ts":"2026-04-27T19:00:00", "type":"tension",
 "interpretation":"Markus genervt", "tension_delta":+0.31, "weight":0.5, ...}
```
Plus Erklaerung welche Felder von wem geschrieben werden
(`character_journal.append(event_id, type, interpretation, tension_delta)`).

Plus: kurzer Verweis auf Distiller (`character_distiller.py run()` →
`character_drift.json` mit `recency_weighted_top` und `rolling_drift`).

### B5. `05_interprozess_kommunikation.md`

**Inhalt:** Wer ruft wen, Protokoll, Fluss.

**Diagramm (Markdown ASCII oder Text):**
```
[Sonoff CAM-PT2 192.168.178.25]
        | RTSP 1080p
        v
[GStreamer TAPPAS Pipeline (moloch.service)]
        | Stage 1: yolov11m -> Person-BBoxes
        | Stage 2: HailoRT-Direct Worker (Face/Pose/ReID/Depth)
        | -> /dev/shm/moloch_frame (Frame fuer Snapshot)
        | -> /dev/shm/moloch_status.json (Detections + State)
        v
[core/moloch_service.py — Hauptprozess]
        | EventBus, CoreIntegrator, IPC-Router
        v
   [/dev/shm/moloch_status.json] <- gemeinsamer State
        ^                |
        | poll           v
[chat_server.py :9100]   [panel_main.py GUI]
        |
        | /chat, /tts, /state_full, /snapshot.jpg
        v
[Markus-PC (192.168.178.20)]
        |    Browser Cockpit (https://moloch.local:9443/)
        |    pc/dashboard.py :11700 (Vision-Pane Polling)
        |    pc/avatar.py :11800 (3D-Mood-Mask)
        |    pc/cross_session_monitor.py (Federation, Heartbeat)
        |    pc/adapter_inference_proxy.py :11600 (Qwen+LoRA v2)
        v
[LLM-Bridge — local_llm_bridge.py]
        | Routing:
        | - kurz/lokal -> hailo-ollama qwen2.5:1.5b (Pi NPU)
        | - mittel -> Tentakel http://192.168.178.20:11434 (PC Ollama mistral/dolphin)
        | - lang/komplex -> DeepSeek Cloud (api.deepseek.com)
        | - LoRA-Pfad -> http://192.168.178.20:11600 (PC Adapter)
        v
[LLM-Antwort] -> chat_server -> Browser/Voice
                              -> personality_engine.speak() -> Pi-Piper TTS
```

Plus konkrete URLs/Ports + welcher Prozess welche Datei locked
(z.B. `_ctx_lock` in moloch_service, atomic JSON-write fuer status).

Quellen-Files fuer Details:
- `core/moloch_service.py` (Haupt-Loop, IPC-Setup)
- `core/ipc_router.py` (IPC-Dispatch)
- `core/autonomy/local_llm_bridge.py` (LLM-Routing)
- `core/bridge/chat_server.py` (HTTP-API)
- `core/bridge/critic_client.py` (PC-Critic-Bridge)
- `core/bridge/adapter_inference_client.py` (PC-LoRA-Bridge)
- `pc/cross_session_monitor.py` (Cross-Session-Sync)

---

## 4. Output-Pfade

Alle 5 Files in `docs/deepseek_briefing/`:
```
docs/deepseek_briefing/
  README.md                       <- Inhaltsverzeichnis + Hinweis fuer DeepSeek
  01_prompt_dreischicht.md
  02_tension_core_mechanik.md
  03_npu_pipeline.md
  04_journal_beispiele.md
  05_interprozess_kommunikation.md
```

`README.md` Inhalt: kurze Erklaerung was die 5 Files sind, Reihenfolge der
Lektuere, Hinweis "Diese Files sind Snapshot vom YYYY-MM-DD HH:MM, Pi HEAD
<commit>". Plus Frage an DeepSeek (Markus' Worte): **"Wie sollten wir den
Chor aufstellen damit Moloch weniger halluziniert + besser auf den
Charakter trifft? Welche Architektur-Aenderungen wuerdest du vorschlagen?"**

---

## 5. Akzeptanz GESAMT

- Aufgabe A: FPS>=18, panel_detections>0 wenn Person, Audit PASS
- Aufgabe B: 5 Files + README in `docs/deepseek_briefing/`, jedes File
  kompiliert (gut lesbar in einem Markdown-Renderer), echte Code-Bloecke
  + Wortlauts (KEIN abstraktes Konzept-Geschwafel — Markus will Substanz)
- Commits + Push
- PC-Mailbox-Eintrag `pipeline_lag_fix_done` + `deepseek_briefing_ready`
- `logs/agent_handoff.md` Update
- Status-Meldung an Markus: `LOKOMOTIVE abgeschlossen.`

---

## 6. NEVER-Regeln (CLAUDE.md, hart)

- **NIE shell=True** — alle subprocess als arglist
- **NIE git config user.* aendern** — Author via env-vars `Cowork Pi-Side`
- **NIE force-push** — `git pull --rebase` + retry
- **`__pycache__` clearen** nach Code-Aenderung
- **Backup-Tag VOR Aenderung**: `pre_telemetry_cache_*`
- **Bei Audit FAIL → STOPP, rollback**

---

## 7. Mailbox-Konvention zur PC-Session

PC-Session ist parallel aktiv (Markus' Windows-App, OAuth gueltig).
Nach Erfolg:
```
docs/PI_TO_PC.md neuer Eintrag oben:
## [TS] from=Pi topic=pipeline_lag_fix+deepseek_briefing_ready
status: info

A) Pipeline-Lag fix: _build_telemetry_footer 5s-cache,
   FPS zurueck auf X, panel_detections wieder befuellt.
B) docs/deepseek_briefing/ angelegt mit 5 Files + README.
   Markus kann jetzt an DeepSeek geben.
```

---

**Viel Erfolg. Markus geht aus dem Zimmer, kommt zurueck — Arbeit erledigt.**
