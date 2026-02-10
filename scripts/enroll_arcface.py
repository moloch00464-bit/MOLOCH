#!/usr/bin/env python3
"""
ArcFace Enrollment auf Hailo-10H NPU.

Laedt Bilder, findet Gesichter via SCRFD, erzeugt ArcFace Embeddings,
speichert Durchschnitts-Embedding in face_embeddings.json.

Usage:
    python3 scripts/enroll_arcface.py --name Markus
    python3 scripts/enroll_arcface.py --name Markus --source snapshots
    python3 scripts/enroll_arcface.py --list
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.expanduser("~/moloch"))

from hailo_platform import HEF, VDevice, FormatType
from core.perception.hailo_postprocess import decode_scrfd, normalize_arcface

MODEL_DIR = "/mnt/moloch-data/hailo/models"
SCRFD_PATH = f"{MODEL_DIR}/scrfd_10g.hef"
ARCFACE_PATH = f"{MODEL_DIR}/arcface_mobilefacenet.hef"
FACE_DB_PATH = os.path.expanduser("~/moloch/data/face_embeddings.json")

# Bild-Quellen
SOURCE_DIRS = {
    "snapshots": os.path.expanduser("~/moloch/snapshots"),
    "train_bart": os.path.expanduser("~/moloch/data/faces/train/Markus_Bart"),
    "train_glatt": os.path.expanduser("~/moloch/data/faces/train/Markus_Glatt"),
    "train_markus": os.path.expanduser("~/moloch/data/faces/train/markus"),
    "train_normal": os.path.expanduser("~/moloch/data/faces/train/normal"),
}


def collect_images(source_filter=None):
    """Sammle alle Bildpfade aus den Quellen."""
    images = []
    for key, path in SOURCE_DIRS.items():
        if source_filter and key != source_filter:
            continue
        if not os.path.isdir(path):
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            found = sorted(glob.glob(os.path.join(path, ext)))
            images.extend(found)
    return images


def run_enrollment(name, source_filter=None):
    """Enrollment: SCRFD + ArcFace auf allen Bildern."""
    images = collect_images(source_filter)
    if not images:
        print("FEHLER: Keine Bilder gefunden!")
        return

    print(f"Enrollment fuer '{name}': {len(images)} Bilder gefunden")

    params = VDevice.create_params()
    with VDevice(params) as vdevice:
        # SCRFD laden
        scrfd_hef = HEF(SCRFD_PATH)
        scrfd_model = vdevice.create_infer_model(SCRFD_PATH)
        scrfd_model.input().set_format_type(FormatType.UINT8)
        scrfd_outs = [o.name for o in scrfd_hef.get_output_vstream_infos()]
        for oname in scrfd_outs:
            scrfd_model.output(oname).set_format_type(FormatType.FLOAT32)

        # ArcFace laden
        arc_hef = HEF(ARCFACE_PATH)
        arc_model = vdevice.create_infer_model(ARCFACE_PATH)
        arc_model.input().set_format_type(FormatType.UINT8)
        arc_outs = [o.name for o in arc_hef.get_output_vstream_infos()]
        for oname in arc_outs:
            arc_model.output(oname).set_format_type(FormatType.FLOAT32)

        # Persistent configure fuer beide Modelle
        scrfd_ctx = scrfd_model.configure().__enter__()
        scrfd_bufs = {o: np.empty(scrfd_model.output(o).shape, dtype=np.float32) for o in scrfd_outs}
        scrfd_bind = scrfd_ctx.create_bindings(output_buffers=scrfd_bufs)

        arc_ctx = arc_model.configure().__enter__()
        arc_bufs = {o: np.empty(arc_model.output(o).shape, dtype=np.float32) for o in arc_outs}
        arc_bind = arc_ctx.create_bindings(output_buffers=arc_bufs)

        embeddings = []
        faces_found = 0

        for idx, img_path in enumerate(images):
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [{idx+1}/{len(images)}] SKIP (nicht lesbar): {os.path.basename(img_path)}")
                continue

            # Resize auf 640x640 fuer SCRFD
            input_640 = cv2.resize(img, (640, 640))
            input_rgb = cv2.cvtColor(input_640, cv2.COLOR_BGR2RGB)

            # SCRFD laufen lassen
            scrfd_bind.input().set_buffer(np.ascontiguousarray(input_rgb))
            scrfd_ctx.run([scrfd_bind], timeout=10000)
            scrfd_out = {o: scrfd_bufs[o].copy() for o in scrfd_outs}

            boxes, scores, landmarks = decode_scrfd(scrfd_out, conf_thresh=0.3, iou_thresh=0.4)

            if len(boxes) == 0:
                print(f"  [{idx+1}/{len(images)}] Kein Gesicht: {os.path.basename(img_path)}")
                continue

            # Bestes Gesicht (hoechster Score)
            best = np.argmax(scores)
            box = boxes[best]
            fh, fw = img.shape[:2]

            # Box in Pixel-Koordinaten
            x1 = max(0, int(box[0] * fw))
            y1 = max(0, int(box[1] * fh))
            x2 = min(fw, int(box[2] * fw))
            y2 = min(fh, int(box[3] * fh))

            # 20% Margin
            bw, bh = x2 - x1, y2 - y1
            mx, my = int(bw * 0.2), int(bh * 0.2)
            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(fw, x2 + mx)
            y2 = min(fh, y2 + my)

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop, resize, RGB
            crop = img[y1:y2, x1:x2]
            crop_112 = cv2.resize(crop, (112, 112))
            crop_rgb = cv2.cvtColor(crop_112, cv2.COLOR_BGR2RGB)

            # ArcFace Embedding
            arc_bind.input().set_buffer(np.ascontiguousarray(crop_rgb))
            arc_ctx.run([arc_bind], timeout=10000)
            emb = arc_bufs[arc_outs[0]].flatten().copy()
            emb = normalize_arcface(emb)

            embeddings.append(emb)
            faces_found += 1
            print(f"  [{idx+1}/{len(images)}] OK (score={scores[best]:.3f}): {os.path.basename(img_path)}")

    if not embeddings:
        print("\nFEHLER: Kein einziges Gesicht in allen Bildern gefunden!")
        return

    # Durchschnitts-Embedding, L2-normalisieren
    avg_emb = np.mean(embeddings, axis=0)
    avg_emb = normalize_arcface(avg_emb)

    print(f"\n{faces_found}/{len(images)} Bilder hatten Gesichter")
    print(f"Embedding-Dimension: {avg_emb.shape[0]}")
    print(f"Embedding-Norm: {np.linalg.norm(avg_emb):.4f}")

    # Bestehende DB laden oder neu erstellen
    db = {}
    if os.path.exists(FACE_DB_PATH):
        with open(FACE_DB_PATH, "r") as f:
            db = json.load(f)
        print(f"Bestehende DB geladen: {list(db.keys())}")

    db[name] = avg_emb.tolist()

    os.makedirs(os.path.dirname(FACE_DB_PATH), exist_ok=True)
    with open(FACE_DB_PATH, "w") as f:
        json.dump(db, f)

    print(f"\nGespeichert: {FACE_DB_PATH}")
    print(f"Personen in DB: {list(db.keys())}")
    print("FERTIG!")


def list_db():
    """Zeige bestehende Face-DB."""
    if not os.path.exists(FACE_DB_PATH):
        print(f"Keine DB vorhanden: {FACE_DB_PATH}")
        return
    with open(FACE_DB_PATH, "r") as f:
        db = json.load(f)
    print(f"Face-DB: {FACE_DB_PATH}")
    for name, emb in db.items():
        arr = np.array(emb)
        print(f"  {name}: {len(emb)}-dim, norm={np.linalg.norm(arr):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArcFace Enrollment auf Hailo-10H")
    parser.add_argument("--name", type=str, help="Name der Person")
    parser.add_argument("--source", type=str, default=None,
                        choices=list(SOURCE_DIRS.keys()),
                        help="Nur bestimmte Quelle verwenden")
    parser.add_argument("--list", action="store_true", help="DB anzeigen")
    args = parser.parse_args()

    if args.list:
        list_db()
    elif args.name:
        run_enrollment(args.name, args.source)
    else:
        parser.print_help()
