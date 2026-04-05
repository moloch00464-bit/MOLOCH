---
name: moloch-npu
description: NPU-Modelle verwalten, neue Modelle integrieren, Worker debuggen, Roadmap pruefen. Nutze fuer Hailo-10H Modell-Arbeit.
allowed-tools: Read Grep Glob Bash
---

# M.O.L.O.C.H. NPU Skill

## WORKER-ARCHITEKTUR

```
GStreamer (Schicht 1):      yolov11m — Person Detection, 20 FPS
HailoRT-Direct (Schicht 2): 7 Worker — Face, Pose, ReID, Hand, Activity, PersonAttr, YOLOWorld
On-Demand (Schicht 3):      SuperResProcessor, LowLightProcessor
```

**Alle nutzen SHARED VDevice** — NIEMALS zweites VDevice erstellen! (Error 74)

---

## AKTIVE MODELLE (7 Worker)

| Worker | Modell / HEF | Trigger |
|--------|-------------|---------|
| GStreamer | yolov11m_h10.hef | jeder Frame (20 FPS) |
| FaceWorker | scrfd_10g.hef + arcface_mobilefacenet.hef + face_attr | alle 2 Frames |
| PoseWorker | yolov8s_pose_h10.hef | alle 3 Frames |
| ReIDWorker | repvgg_a0_person_reid_512.hef | alle 5 Frames |
| HandWorker | hand_landmark_lite.hef | alle 4 Frames |
| ActivityWorker | r3d_18.hef | alle 10 Frames |
| PersonAttrWorker | person_attr_resnet_v1_18.hef | alle 5 Frames |
| YOLOWorldWorker | yolo_world_v2s.hef | on-demand |
| SuperResProcessor | real_esrgan_x2.hef | bei Snapshot |
| LowLightProcessor | zero_dce.hef | bei Dunkelheit < 80/255 |

MCP: `moloch_npu_models()` | `moloch_npu_workers()` | `moloch_low_light()`

---

## NPU RAM

**~65 MB / 8192 MB genutzt (99% frei)**
Alle Modelle permanent geladen — kein Valve-Switching noetig.

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
