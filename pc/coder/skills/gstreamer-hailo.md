# GStreamer + Hailo-Inferenz

## NEVER 1: Pipeline-String unveraendert

GStreamer-Pipeline ist als gst-launch-style String definiert.
Aenderungen fuehren zu **SEGV** (segfault) — der Service crashed silent.

Lage: `core/perception/<pipeline>.py`

## NEVER 9: HailoRT-Tensor-Typ pruefen

Vor jeder Inferenz Tensor-dtype verifizieren:
- **SCRFD, YOLO**: erwarten `uint8` (0-255)
- **ArcFace, ReID Embeddings**: erwarten `float32` (normalisiert)
- Mismatch = silent garbage output, KEIN Crash, schwer zu debuggen

```python
assert tensor.dtype == np.uint8, f"SCRFD braucht uint8, got {tensor.dtype}"
```

## ROI-Dispatch

Frame mit bbox-tile vor Inferenz fuettern.
- Pose-Model: 256x256 Crops
- ArcFace: 112x112 Crops (aligned)
- ReID: 128x256 Crops

Lage: `core/vision/<model>.py`

## __pycache__ (NEVER 11)

Nach Code-Aenderung in vision/perception:
```bash
find core/ -name __pycache__ -exec rm -rf {} +
```

Sonst laufen alte Bytecode-Reste weiter.
