---
name: moloch-npu
description: NPU-Modelle verwalten, neue Modelle integrieren, Worker debuggen, Roadmap pruefen. Nutze wenn Du neue Hailo-10H Modelle hinzufuegen oder bestehende Worker debuggen willst.
---

# M.O.L.O.C.H. NPU Skill
# Stand: 2026-04-01

---

## WORKER-ARCHITEKTUR

```
GStreamer (Schicht 1):     yolov11m — Person Detection, 20 FPS
HailoRT-Direct (Schicht 2): FaceWorker, PoseWorker, ReIDWorker, HandWorker
On-Demand (Schicht 3):    SuperResProcessor, LowLightProcessor
```

**Alle nutzen SHARED VDevice** (`vdevice-group-id=SHARED`) — kein zweites VDevice erstellen!

**Pattern fuer neue Worker**: `core/perception/super_res_worker.py` (On-Demand)
oder `core/perception/face_pipeline.py` (Stream-Worker via BaseWorker)

---

## AKTIVE MODELLE (2026-04-01)

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
| zero_dce | zero_dce.hef | LowLightProcessor | bei Dunkelheit (<80/255) |

---

## NEUES MODELL INTEGRIEREN (Schritt fuer Schritt)

1. **HEF herunterladen**
   ```bash
   wget -O /mnt/moloch-data/hailo/models/MODELL.hef \
     "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/MODELL.hef"
   ```

2. **Input/Output pruefen**
   ```bash
   python3 -c "
   import hailo_platform as hp
   params = hp.VDevice.create_params(); params.group_id = 'SHARED'
   vdev = hp.VDevice(params)
   m = vdev.create_infer_model('/mnt/moloch-data/hailo/models/MODELL.hef')
   print('Inputs:', [(n, list(m.input(n).shape)) for n in m.input_names])
   print('Outputs:', [(n, list(m.output(n).shape)) for n in m.output_names])
   "
   ```

3. **Worker erstellen** — Vorlage: `core/perception/super_res_worker.py`
   - Synchron/On-Demand → Pattern wie SuperResProcessor
   - Stream (jeder Frame) → Pattern wie FaceWorker (BaseWorker Thread)

4. **Integration in tappas_pipeline.py**
   - On-Demand: in `_on_appsink_sample()` einfuegen
   - Stream-Worker: in `start()` registrieren via `_roi_dispatcher.register_worker()`

5. **stop()-Cleanup in moloch_service.py** ergaenzen

6. **MCP-Tool in moloch_mcp_server.py** ergaenzen

7. **Roadmap updaten**: `~/moloch/logs/npu_model_roadmap.md`

---

## ROADMAP PRUEFEN

```
MCP: moloch_npu_models()
```

Oder direkt:
```bash
cat ~/moloch/logs/npu_model_roadmap.md
```

Naechste empfohlene Modelle:
- `person_attr_resnet_v1_18.hef` — Kleidung/Alter/Rucksack erkennen
- `r3d_18.hef` — Aktivitaetserkennung (sitzt/geht/laeuft)
- `yolo_world_v2s.hef` — Zero-Shot Objektsuche per Sprachbefehl

---

## WORKER DEBUGGEN

```
MCP: moloch_npu_workers()
```

Oder:
```bash
journalctl -u moloch.service -n 50 | grep -E "Worker|SuperRes|LowLight|Face|Pose"
```

Haeufige Fehler:
- `HAILO_OUT_OF_PHYSICAL_DEVICES(74)` → zweites VDevice erstellt → `group_id=SHARED` setzen!
- `Input buffer size mismatch` → falscher dtype (uint8 vs float32) oder falsche Shape
- Worker gibt Original zurueck → Model-Load fehlgeschlagen → Logs pruefen

---

## NPU RAM BUDGET

```
Gesamt: 8192 MB (Hailo-10H LPDDR4)
Genutzt: ~120 MB (alle aktiven Modelle)
Frei:   ~8072 MB (98%)
```

Kein Valve-Switching noetig — alle Modelle permanent geladen.
