#!/usr/bin/env python3
"""
Test: FaceWorker via HailoRT-Direct — Standalone-Validierung.

Testet SCRFD + ArcFace + FaceAttr auf dem SHARED VDevice,
WAEHREND der MOLOCH-Service laeuft.

Nutzt den aktuellen SHM-Frame (Live-Kamera) oder ein Bild-Argument.

Usage:
    python3 scripts/test_face_worker.py                  # SHM Frame
    python3 scripts/test_face_worker.py --image test.jpg # Bild-Datei
    python3 scripts/test_face_worker.py --snapshot       # Neuen Snapshot nehmen
"""

import os
import sys
import time
import struct
import argparse
import numpy as np

# Projekt-Root
PROJECT_ROOT = os.path.expanduser("~/moloch")
sys.path.insert(0, PROJECT_ROOT)

SHM_FRAME_PATH = "/dev/shm/moloch_frame"
SHM_HEADER_SIZE = 24


def grab_shm_frame():
    """Aktuellen Frame aus SHM lesen."""
    with open(SHM_FRAME_PATH, "rb") as f:
        header = f.read(SHM_HEADER_SIZE)
        h, w, c = struct.unpack("<III", header[:12])
        data = f.read(h * w * c)
        frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w, c))
        return frame  # BGR


def main():
    parser = argparse.ArgumentParser(description="FaceWorker HailoRT-Direct Test")
    parser.add_argument("--image", type=str, help="Bild-Datei statt SHM")
    parser.add_argument("--snapshot", action="store_true", help="Frischen Snapshot nehmen")
    args = parser.parse_args()

    import cv2

    print("=" * 60)
    print("M.O.L.O.C.H. FaceWorker Test (HailoRT-Direct)")
    print("=" * 60)

    # Frame laden
    if args.image:
        frame_bgr = cv2.imread(args.image)
        if frame_bgr is None:
            print(f"FEHLER: Bild nicht lesbar: {args.image}")
            sys.exit(1)
        print(f"  Quelle: {args.image}")
    else:
        print("  Quelle: SHM (Live-Kamera)")
        frame_bgr = grab_shm_frame()

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fh, fw = frame_rgb.shape[:2]
    print(f"  Frame: {fw}x{fh}")

    # Face Pipeline importieren
    from core.perception.face_pipeline import (
        letterbox_resize, unletterbox_coords, align_face,
        SCRFD_CONF_THRESH, SCRFD_NMS_THRESH, ARCFACE_MATCH_THRESH,
        SCRFD_HEF, ARCFACE_HEF, FACE_DB_PATH,
    )
    from core.perception.hailo_postprocess import (
        decode_scrfd, normalize_arcface, match_face
    )
    from core.perception.vision_workers import create_configured_model, VDEVICE_GROUP_ID

    # VDevice joinen
    print("\n[1/5] VDevice joinen...")
    import hailo_platform as hp
    params = hp.VDevice.create_params()
    params.group_id = VDEVICE_GROUP_ID
    vdevice = hp.VDevice(params)
    print(f"  VDevice OK (group={VDEVICE_GROUP_ID})")

    # SCRFD laden
    print("\n[2/5] SCRFD laden...")
    _, scrfd_cfg, _, scrfd_outs, scrfd_shapes = create_configured_model(vdevice, SCRFD_HEF)
    print(f"  SCRFD OK — Outputs: {scrfd_outs}")

    # ArcFace laden
    print("\n[3/5] ArcFace laden...")
    _, arcface_cfg, _, arcface_outs, arcface_shapes = create_configured_model(vdevice, ARCFACE_HEF)
    print(f"  ArcFace OK — Outputs: {arcface_outs}")

    # Letterbox + SCRFD
    print("\n[4/5] SCRFD Inference...")
    padded, _scale, pad_x, pad_y, rw, rh = letterbox_resize(frame_rgb, 640)

    scrfd_bindings = scrfd_cfg.create_bindings()
    scrfd_bindings.input().set_buffer(np.ascontiguousarray(padded))
    scrfd_bufs = {}
    for name in scrfd_outs:
        buf = np.empty(scrfd_shapes[name], dtype=np.float32)
        scrfd_bindings.output(name).set_buffer(buf)
        scrfd_bufs[name] = buf

    t0 = time.monotonic()
    scrfd_cfg.run([scrfd_bindings], 10000)
    dt_scrfd = (time.monotonic() - t0) * 1000
    outputs = {n: scrfd_bufs[n].copy() for n in scrfd_outs}

    boxes, scores, landmarks = decode_scrfd(outputs, 640, SCRFD_CONF_THRESH, SCRFD_NMS_THRESH)
    print(f"  SCRFD: {len(boxes)} Gesichter in {dt_scrfd:.0f}ms")

    if len(boxes) == 0:
        print("\n  KEIN GESICHT ERKANNT — dreh dich zur Kamera!")
        print("  (Test mit --image nutzen fuer Offline-Bilder)")
        del vdevice
        return

    # Unletterbox
    boxes_norm, landmarks_norm = unletterbox_coords(boxes, landmarks, pad_x, pad_y, rw, rh)

    for i in range(len(boxes_norm)):
        box = boxes_norm[i]
        score = scores[i]
        lm = landmarks_norm[i]
        bw_px = int((box[2] - box[0]) * fw)
        bh_px = int((box[3] - box[1]) * fh)
        print(f"  Face #{i}: score={score:.3f} bbox=({box[0]:.3f},{box[1]:.3f},"
              f"{box[2]:.3f},{box[3]:.3f}) size={bw_px}x{bh_px}px")

    # ArcFace fuer bestes Gesicht
    print(f"\n[5/5] ArcFace Inference (bestes Gesicht)...")
    best_idx = np.argmax(scores)
    box = boxes_norm[best_idx]
    lm = landmarks_norm[best_idx]

    # Pixel-Koordinaten fuer Landmarks
    landmarks_px = []
    for p in range(5):
        lx = lm[p * 2] * fw
        ly = lm[p * 2 + 1] * fh
        landmarks_px.append([lx, ly])

    # Align
    aligned = align_face(frame_rgb, landmarks_px)
    if aligned is None:
        print("  WARNUNG: Alignment fehlgeschlagen, nutze Crop+Resize")
        px1 = max(0, int(box[0] * fw))
        py1 = max(0, int(box[1] * fh))
        px2 = min(fw, int(box[2] * fw))
        py2 = min(fh, int(box[3] * fh))
        crop = frame_rgb[py1:py2, px1:px2]
        aligned = cv2.resize(crop, (112, 112))

    # ArcFace Inference
    arcface_bindings = arcface_cfg.create_bindings()
    arcface_bindings.input().set_buffer(np.ascontiguousarray(aligned))
    arcface_bufs = {}
    for name in arcface_outs:
        buf = np.empty(arcface_shapes[name], dtype=np.float32)
        arcface_bindings.output(name).set_buffer(buf)
        arcface_bufs[name] = buf

    t0 = time.monotonic()
    arcface_cfg.run([arcface_bindings], 10000)
    dt_arcface = (time.monotonic() - t0) * 1000

    emb_raw = arcface_bufs[arcface_outs[0]].flatten().copy()
    embedding = normalize_arcface(emb_raw)
    print(f"  ArcFace: {len(embedding)}d embedding in {dt_arcface:.0f}ms, norm={np.linalg.norm(embedding):.4f}")

    # Face-DB laden und matchen
    import json
    if os.path.exists(FACE_DB_PATH):
        with open(FACE_DB_PATH, "r") as f:
            raw_db = json.load(f)
        face_db = {name: np.array(emb, dtype=np.float32) for name, emb in raw_db.items()}
        print(f"  Face-DB: {len(face_db)} Eintraege geladen")

        # Match gegen ALLE Eintraege
        print(f"\n  === SIMILARITY-ERGEBNIS ===")
        best_name, best_sim = match_face(embedding, face_db, ARCFACE_MATCH_THRESH)
        print(f"  BEST MATCH: {best_name} (sim={best_sim:.4f}, threshold={ARCFACE_MATCH_THRESH})")

        if best_sim >= ARCFACE_MATCH_THRESH:
            print(f"  ✅ ERKANNT: {best_name}")
        else:
            print(f"  ❌ NICHT ERKANNT (sim={best_sim:.4f} < {ARCFACE_MATCH_THRESH})")

        # Top-5 Matches anzeigen
        print(f"\n  Top-5 Matches:")
        sims = []
        for name, ref_emb in face_db.items():
            sim = float(np.dot(embedding, ref_emb / max(np.linalg.norm(ref_emb), 1e-8)))
            sims.append((sim, name))
        sims.sort(reverse=True)
        for sim, name in sims[:5]:
            marker = "✅" if sim >= ARCFACE_MATCH_THRESH else "  "
            print(f"    {marker} {sim:.4f}  {name}")

        # Vergleich: alte GStreamer-Pipeline Similarity vs. neue HailoRT-Direct
        print(f"\n  === VERGLEICH ===")
        print(f"  Alte Pipeline (GStreamer): Similarity 0.14-0.46 (INKOMPATIBEL)")
        print(f"  Neue Pipeline (HailoRT):  Similarity {best_sim:.4f}")
        if best_sim > 0.46:
            print(f"  → VERBESSERUNG: +{(best_sim - 0.46)*100:.0f}% gegenueber alter Pipeline")
    else:
        print(f"  WARNUNG: Keine Face-DB unter {FACE_DB_PATH}")

    # Aligned Face speichern (zur visuellen Kontrolle)
    aligned_path = "/tmp/face_worker_aligned.jpg"
    aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
    cv2.imwrite(aligned_path, aligned_bgr)
    print(f"\n  Aligned Face gespeichert: {aligned_path}")

    # Aufraeumen
    del vdevice
    print("\n" + "=" * 60)
    print("TEST ABGESCHLOSSEN")
    print("=" * 60)


if __name__ == "__main__":
    main()
