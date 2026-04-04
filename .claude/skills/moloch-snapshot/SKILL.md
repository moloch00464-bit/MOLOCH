---
name: moloch-snapshot
description: Kamera-Snapshot von MOLOCH holen und analysieren. Zeigt was MOLOCH gerade sieht. Real-ESRGAN Upscaling + Low-Light Enhancement.
---

Nutze das MCP-Tool:
```
moloch_snapshot()
```

Analysiere:
- Welche BBoxen sind sichtbar (Person, Face, Pose)?
- Sind Pose-Landmarks korrekt auf dem Koerper?
- Stimmt die Face-BBox Groesse?
- Gibt es Artefakte oder Probleme?
- Ist das Bild klar oder noch dunkel (Low-Light aktiv)?

NPU-Verarbeitung im Snapshot:
- Real-ESRGAN x2: 640x360 → 1024x1024 (schaerfer)
- Low-Light (zero_dce): automatisch wenn Helligkeit < 80/255
