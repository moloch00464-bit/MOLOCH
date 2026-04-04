---
name: moloch-npu
description: NPU-Modelle verwalten, neue Modelle integrieren, Worker debuggen, Roadmap pruefen. Nutze fuer Hailo-10H Modell-Arbeit.
allowed-tools: Read Grep Glob Bash
---

# M.O.L.O.C.H. NPU Skill

## WORKER-ARCHITEKTUR

```
GStreamer (Schicht 1):     yolov11m — Person Detection, 20 FPS
HailoRT-Direct (Schicht 2): FaceWorker, PoseWorker, ReIDWorker, HandWorker
On-Demand (Schicht 3):    SuperResProcessor, LowLightProcessor
```

**Alle nutzen SHARED VDevice** — NIEMALS zweites VDevice erstellen!

---

## AKTIVE MODELLE

| Modell | HEF | Worker | Trigger |
|--------|-----|--------|---------|
| yolov11m | yolov11m_h10.hef | GStreamer | jeder Frame |
| SCRFD | scrfd_10g.hef | FaceWorker | alle 2 Frames |
| ArcFace | arcface_mobilefacenet.hef | FaceWorker | bei Gesicht |
| FaceAttr | face_attr_resnet_v1_18.hef | FaceWorker | bei Gesicht |
| YOLOv8s-Pose | yolov8s_pose_h10.hef | PoseWorker | alle 3 Frames |
| RepVGG ReID | repvgg_a0_person_reid_512.hef | ReIDWorker | alle 5 Frames |
| Hand | hand_landmark_lite.hef | HandWorker | alle 4 Frames |
| Real-ESRGAN x2 | real_esrgan_x2.hef | SuperResProcessor | bei Snapshot |
| zero_dce | zero_dce.hef | LowLightProcessor | bei Dunkelheit |

MCP: `moloch_npu_models()` | `moloch_npu_workers()` | `moloch_low_light()`

---

## NEUES MODELL INTEGRIEREN

Fuer Details siehe [integration-steps.md](integration-steps.md).

1. HEF herunterladen nach `/mnt/moloch-data/hailo/models/`
2. Input/Output pruefen (Shape + dtype)
3. Worker erstellen (Vorlage: `super_res_worker.py` oder `face_pipeline.py`)
4. Integration in `tappas_pipeline.py`
5. stop()-Cleanup in `moloch_service.py`
6. MCP-Tool in `moloch_mcp_server.py`
7. Roadmap updaten

---

## NPU RAM: 120 MB / 8192 MB (98% frei)

Kein Valve-Switching noetig — alle Modelle permanent geladen.
