# NPU Modell-Integration (Schritt fuer Schritt)

## 1. HEF herunterladen

```bash
wget -O /mnt/moloch-data/hailo/models/MODELL.hef \
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/MODELL.hef"
```

## 2. Input/Output pruefen

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

## 3. Worker erstellen

- Synchron/On-Demand → Pattern wie `core/perception/super_res_worker.py`
- Stream (jeder Frame) → Pattern wie `core/perception/face_pipeline.py` (BaseWorker)

## 4. Integration in tappas_pipeline.py

- On-Demand: in `_on_appsink_sample()` einfuegen
- Stream-Worker: in `start()` via `_roi_dispatcher.register_worker()`

## 5. stop()-Cleanup in moloch_service.py

## 6. MCP-Tool in moloch_mcp_server.py ergaenzen

## 7. Roadmap updaten: `~/moloch/logs/npu_model_roadmap.md`

## Naechste empfohlene Modelle

- `person_attr_resnet_v1_18.hef` — Kleidung/Alter/Rucksack
- `r3d_18.hef` — Aktivitaetserkennung (sitzt/geht/laeuft)
- `yolo_world_v2s.hef` — Zero-Shot Objektsuche per Sprache
