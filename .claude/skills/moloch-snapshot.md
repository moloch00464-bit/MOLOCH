---
name: moloch-snapshot
description: Kamera-Snapshot von MOLOCH holen und analysieren. Zeigt was MOLOCH gerade sieht. Nutzt automatisch Real-ESRGAN x2 Upscaling und Low-Light Enhancement.
---

Nutze das MCP-Tool — macht automatisch Upscaling + Low-Light:

```
moloch_snapshot()
```

Dann lies `/tmp/moloch_snapshot.jpg` und analysiere:
- Welche BBoxen sind sichtbar (Person, Face, Pose)?
- Sind Pose-Landmarks korrekt auf dem Koerper?
- Stimmt die Face-BBox Groesse?
- Gibt es Artefakte oder Probleme?
- Ist das Bild klar oder noch dunkel (Low-Light aktiv)?

Alternativ direkt aus SHM (ohne NPU-Processing):

```python
import struct, mmap, os, cv2, numpy as np
SHM = '/dev/shm/moloch_frame'
fd = os.open(SHM, os.O_RDONLY)
size = os.fstat(fd).st_size
mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
h, w, c, seq, ts = struct.unpack('<IIIId', mm[:24])
data = np.frombuffer(mm[24:24+h*w*c], dtype=np.uint8).reshape(h,w,c)
frame = cv2.resize(data, (1280, 720), interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite('/tmp/moloch_snapshot.png', frame)
mm.close(); os.close(fd)
```

NPU-Verarbeitung im Snapshot:
- Real-ESRGAN x2: 640x360 → 1024x1024 (schärfer)
- Low-Light (zero_dce): automatisch wenn Helligkeit < 80/255
