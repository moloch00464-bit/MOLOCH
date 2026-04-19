---
name: moloch-npu
description: NPU-Modelle verwalten, neue Modelle integrieren, Worker debuggen, Roadmap pruefen. Nutze fuer Hailo-10H Modell-Arbeit.
allowed-tools: Read Grep Glob Bash
---

# M.O.L.O.C.H. NPU Skill

## WORKER-ARCHITEKTUR (Stand Session 19, 2026-04-19)

```
GStreamer (Schicht 1):      yolov11m — Person Detection, 20 FPS
HailoRT-Direct (Schicht 2): 4 Worker AKTIV — Face, Pose, ReID, Depth
                            4 Worker DEAKTIVIERT — Hand, Activity, PersonAttr, YOLOWorld
On-Demand (Schicht 3):      SuperResProcessor, LowLightProcessor
LLM (Schicht 4):            hailo-ollama qwen2.5:1.5b (SHARED VDevice mit TAPPAS)
```

**Hartes Limit:** `HAILO_MAX_NETWORK_GROUPS=8` (SDK-Header `hailort.h:52`).
4 Worker + TAPPAS (1 Group) + Whisper on-demand + Qwen LLM = 7 Groups, sicher
unter dem Limit. **Alle nutzen SHARED VDevice** (vdevice-group-id=SHARED).

---

## AKTIVE MODELLE (4 Worker + LLM)

| Worker | Modell / HEF | Trigger | Network-Group |
|--------|-------------|---------|---------------|
| GStreamer | yolov11m_h10.hef | jeder Frame (20 FPS) | TAPPAS (shared) |
| FaceWorker | scrfd_10g.hef + arcface_mobilefacenet.hef + face_attr | alle 2 Frames | 1 |
| PoseWorker | yolov8s_pose_h10.hef | alle 3 Frames | 1 |
| ReIDWorker | repvgg_a0_person_reid_512.hef | alle 5 Frames | 1 |
| DepthWorker | scdepthv3.hef | alle 10 Frames | 1 |
| **hailo-ollama** | qwen2.5:1.5b (Q4_0, ~2.3 GB) | bei LLM-Call | 1 |
| SuperResProcessor | real_esrgan_x2.hef | bei Snapshot | on-demand |
| LowLightProcessor | zero_dce.hef | bei Dunkelheit < 80/255 | on-demand |

## DEAKTIVIERTE MODELLE (Session 19, in tappas_pipeline.py auskommentiert)
- HandWorker (hand_landmark_lite.hef) — SEGV-Race-History
- ActivityWorker (r3d_18.hef) — Bug A2, nicht voll integriert
- PersonAttrWorker (person_attr_resnet_v1_18.hef) — Bug A1
- YOLOWorldWorker (yolo_world_v2s.hef) — Bug A3, every 60 frames

MCP: `moloch_npu_models()` | `moloch_npu_workers()` | `moloch_low_light()`

---

## NPU RAM

**~1.5 GB / 8 GB genutzt (~80% frei)** — meiste Last durch Qwen2.5:1.5b LLM.
Pipeline-HEFs nur ~65 MB. Alle Modelle permanent geladen — kein Valve-Switching.

---

## NEUES MODELL INTEGRIEREN

Fuer Details siehe [integration-steps.md](integration-steps.md).

1. HEF herunterladen nach `/mnt/moloch-data/hailo/models/`
2. Input/Output pruefen (Shape + dtype: uint8 oder float32!)
3. Worker erstellen (Vorlage: `super_res_worker.py` oder `face_pipeline.py`)
4. Integration in `roi_dispatcher.py` oder `tappas_pipeline.py`
5. stop()-Cleanup in `moloch_service.py`
6. MCP-Tool in `moloch_mcp_server.py` (optional)
7. Roadmap in `moloch_npu_models()` updaten

---

## DEBUGGING

| Problem | Loesung |
|---------|---------|
| Error 74 | Kein zweites VDevice — `moloch_service(action="restart")` |
| Worker errors > 0 | `moloch_npu_workers()` + `moloch_dmesg()` |
| FPS < 18 | Queue-Stau? `moloch_npu_workers()` → queue-Wert pruefen |
| uint8/float32 Mismatch | NEVER 9 — dtype VOR Inferenz pruefen |
