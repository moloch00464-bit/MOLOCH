# Agent Handoff — 2026-04-01
# Session: ecstatic-banach (Claude Sonnet 4.6)
# Status: VOLLSTAENDIG ABGESCHLOSSEN

---

## WAS DIESE SESSION ERLEDIGT HAT

### 1. Two-Stage Hybrid Pipeline (aus vorheriger Session fortgesetzt)

Architektur: GStreamer nur noch YOLO, alle anderen Modelle als HailoRT-Direct Worker
- core/perception/vision_workers.py — BaseWorker + ResultCollector
- core/perception/roi_dispatcher.py — Frame-Verteilung
- core/perception/face_pipeline.py — SCRFD + ArcFace + FaceAttr (Similarity 1.00)
- core/perception/pose_worker.py — Pose, ReID, Hand
- core/perception/tappas_pipeline.py — GStreamer 330 Zeilen → 50 Zeilen
- scripts/enroll_face_worker.py — Enrollment identisch wie Live-Inference
Commits: b1a5e7f, 31ff6cc, 6d8a5d1

### 2. Real-ESRGAN x2 Super Resolution

HEF: /mnt/moloch-data/hailo/models/real_esrgan_x2.hef (27MB)
Datei: core/perception/super_res_worker.py (Singleton, SHARED VDevice)
Integration: MCP moloch_snapshot() + daily_learner.py Face-Crops
Format: uint8 Input 512x512x3 → float32 Output 1024x1024x3
Commits: 694602c, 793f37a

### 3. Low Light Enhancement (zero_dce)

HEF: /mnt/moloch-data/hailo/models/zero_dce.hef (856KB, 200 FPS)
Datei: core/perception/low_light_processor.py (Singleton, SHARED VDevice)
Integration: tappas_pipeline._on_appsink_sample() vor SHM-Write
Logik: CPU-Brightness-Check < 80/255 → NPU Enhancement aktiv
Commits: 08cf594, 5c0246b

### 4. MCP-Server — 3 neue Tools

Datei: mcp/moloch_mcp_server.py
- moloch_npu_workers() — Health aller NPU-Worker
- moloch_npu_models() — Roadmap integriert vs. ausstehend
- moloch_low_light() — Helligkeit + Enhancement-Status
Commit: bc9019f

### 5. NPU-Modell-Roadmap + Skills verbessert

Roadmap: logs/npu_model_roadmap.md (alle 26 H10H-Kategorien dokumentiert)
Skills:
- .claude/skills/moloch-npu.md (NEU) — Worker-Architektur + Anleitung
- .claude/skills/moloch-dev.md — 4 neue NEVER-Regeln + HailoRT-Template + RGB/BGR
- .claude/skills/moloch-snapshot.md — SuperRes + LowLight erwaehnt
Commit: 08cf594

---

## SERVICE-STATUS

moloch.service: active (running)
USE_TAPPAS: 1 (in /etc/systemd/system/moloch.service)
GStreamer: YOLO-only, ~20 FPS
Worker: FaceWorker, PoseWorker, ReIDWorker, HandWorker (alle aktiv)
On-Demand: SuperResProcessor (lazy), LowLightProcessor (lazy, ab Dunkelheit)

---

## ALLE COMMITS DIESER SESSION

bc9019f feat: MCP NPU-Tools + LowLight stop()-Cleanup
5c0246b feat: LowLight in tappas_pipeline._on_appsink_sample()
08cf594 feat: LowLightProcessor + NPU Roadmap + Skills
793f37a fix: SuperRes Input uint8 statt float32
694602c feat: Real-ESRGAN x2 Super Resolution via Hailo-10H NPU
6d8a5d1 feat: Phase 4+5 — Alle Worker + Blaustich-Fix + FaceAttr
31ff6cc fix: ResultCollector.get_latest() direkt vom Worker
b1a5e7f feat: Phase 2+3 — GStreamer YOLO-Only + FaceWorker live

---

## NAECHSTE PRIORITAETEN

1. person_attr_resnet_v1_18.hef — Kleidung/Alter/Rucksack (Aufwand: gering)
   Download-URL in logs/npu_model_roadmap.md
2. r3d_18.hef — Aktivitaetserkennung (sitzt/geht/laeuft)
3. yolo_world_v2s.hef — Zero-Shot Objektsuche per Sprache

---

## BEKANNTE BUGS (unveraendert)

- Kamera Hot-Plug: nur Reboot hilft (kein RTSP-Reconnect)
- hailo-ollama: kein systemd-Service, laeuft nicht beim Boot
- MCP moloch_snapshot() gibt erst 1024x1024 nach MCP-Server-Neustart

---

## GELERNTE LEKTIONEN (in moloch-dev.md ergaenzt)

1. HailoRT Input dtype: Vision-Modelle = uint8, NICHT float32
2. np.ndarray Type-Hints nicht in moloch_service.py Signaturen (np nicht importiert)
3. __pycache__ immer loeschen nach Code-Aenderungen
4. Service laeuft von ~/moloch/, NIE vom Worktree
5. GStreamer = RGB, cv2.imwrite() = BGR → COLOR_RGB2BGR vor imwrite()
6. Stable Diffusion: KEIN H10H-HEF verfuegbar (definitiv verifiziert)
