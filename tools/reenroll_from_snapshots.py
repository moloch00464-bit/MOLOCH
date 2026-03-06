#!/usr/bin/env python3
"""
ArcFace Re-Enrollment aus vorhandenen Snapshots (Full-HD Crops).

NUTZUNG:
  sudo systemctl stop moloch
  cd ~/moloch && python3 tools/reenroll_from_snapshots.py
  sudo systemctl start moloch

ACHTUNG: Service vorher stoppen — Hailo erlaubt nur EIN VDevice!

Ablauf:
  1. Alte face_embeddings.json sichern
  2. Alle Snapshots mit SCRFD+Letterboxing durchlaufen
  3. Beste 20 nach SCRFD-Score + Embedding-Diversitaet auswaehlen
  4. ArcFace-Embeddings als "markus" + "markus#snap_N" speichern
"""

import os
import sys
import json
import time
import glob
import shutil

import cv2
import numpy as np

# Projekt-Root zum Path (fuer core.perception Import)
PROJECT_ROOT = os.path.expanduser("~/moloch")
sys.path.insert(0, PROJECT_ROOT)

from core.perception.hailo_postprocess import decode_scrfd

# ArcFace Standard-Referenz-Landmarks fuer 112x112
# (gleiche Positionen wie TAPPAS libvms_face_align.so)
ARCFACE_REF_LANDMARKS = np.array([
    [38.2946, 51.6963],  # linkes Auge
    [73.5318, 51.5014],  # rechtes Auge
    [56.0252, 71.7366],  # Nase
    [41.5493, 92.3655],  # linker Mundwinkel
    [70.7299, 92.2041],  # rechter Mundwinkel
], dtype=np.float32)

# Pfade
SNAPSHOTS_DIR = os.path.join(PROJECT_ROOT, "snapshots")
FACE_DB_PATH = os.path.join(PROJECT_ROOT, "data", "face_embeddings.json")
MODEL_DIR = "/mnt/moloch-data/hailo/models"
SCRFD_HEF = os.path.join(MODEL_DIR, "scrfd_10g.hef")
ARCFACE_HEF = os.path.join(MODEL_DIR, "arcface_mobilefacenet.hef")

# Parameter
SCRFD_CONF = 0.70       # Gute Gesichter, select_diverse bevorzugt >0.85
SCRFD_NMS = 0.40
MIN_FACE_PIX = 40       # Minimale Gesichtsgroesse in Pixel (auf Full-HD)
TARGET_COUNT = 20        # Ziel-Anzahl Embeddings
DIVERSITY_THRESHOLD = 0.85  # Cosine-Sim unter der ein Embedding "divers" genug ist


def letterbox_resize(img, target_size=640):
    """Letterbox-Resize identisch zur Live-Pipeline."""
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return padded, scale, pad_x, pad_y, new_w, new_h


def unletterbox_scrfd(boxes, landmarks, pad_x, pad_y, rw, rh, target=640):
    """Letterbox-Space -> normalisierte [0,1] Koordinaten relativ zum Content."""
    bc = boxes.copy()
    lc = landmarks.copy()
    if pad_x == 0 and pad_y == 0 and rw == target and rh == target:
        return bc, lc
    bc[:, [0, 2]] = np.clip((boxes[:, [0, 2]] * target - pad_x) / rw, 0, 1)
    bc[:, [1, 3]] = np.clip((boxes[:, [1, 3]] * target - pad_y) / rh, 0, 1)
    for i in range(5):
        lc[:, i * 2] = np.clip((landmarks[:, i * 2] * target - pad_x) / rw, 0, 1)
        lc[:, i * 2 + 1] = np.clip((landmarks[:, i * 2 + 1] * target - pad_y) / rh, 0, 1)
    return bc, lc


def align_face(img, landmarks_5pt):
    """Face Alignment via 5-Point Affine Transform (wie TAPPAS libvms_face_align.so).

    Args:
        img: Original-Frame (BGR)
        landmarks_5pt: 5 Landmark-Punkte als Pixel-Koordinaten [(x,y), ...]

    Returns:
        aligned: 112x112 BGR Bild oder None bei Fehler
    """
    src_pts = np.array(landmarks_5pt, dtype=np.float32)
    tform, _ = cv2.estimateAffinePartial2D(src_pts, ARCFACE_REF_LANDMARKS)
    if tform is None:
        return None
    return cv2.warpAffine(img, tform, (112, 112), borderValue=(0, 0, 0))


def backup_face_db():
    """Bestehende Face-DB sichern."""
    if not os.path.exists(FACE_DB_PATH):
        print("  Keine bestehende Face-DB gefunden.")
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = f"{FACE_DB_PATH}.backup_{ts}"
    shutil.copy2(FACE_DB_PATH, backup)
    size_kb = os.path.getsize(backup) / 1024
    print(f"  Backup: {backup} ({size_kb:.0f} KB)")


def load_face_db():
    """Face-DB laden."""
    if not os.path.exists(FACE_DB_PATH):
        return {}
    with open(FACE_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_face_db(db):
    """Face-DB atomar speichern."""
    os.makedirs(os.path.dirname(FACE_DB_PATH), exist_ok=True)
    tmp = FACE_DB_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=1, ensure_ascii=False)
    os.replace(tmp, FACE_DB_PATH)


def select_diverse(candidates, target_count, sim_threshold):
    """Aus Kandidaten die diversesten auswaehlen.

    candidates: Liste von (score, embedding, filepath, bbox_info)
    Sortiert nach Score (absteigend), dann greedy diverse selection.
    """
    # Nach Score sortieren (hoechster zuerst)
    candidates.sort(key=lambda x: x[0], reverse=True)

    selected = []
    for score, emb, fpath, info in candidates:
        if len(selected) >= target_count:
            break
        # Pruefen ob divers genug zu allen bisherigen
        is_diverse = True
        for _, sel_emb, _, _ in selected:
            sim = float(np.dot(emb, sel_emb))
            if sim > sim_threshold:
                is_diverse = False
                break
        if is_diverse:
            selected.append((score, emb, fpath, info))

    # Falls nicht genug diverse: Rest nach Score auffuellen
    if len(selected) < target_count:
        for score, emb, fpath, info in candidates:
            if len(selected) >= target_count:
                break
            already = any(f == fpath for _, _, f, _ in selected)
            if not already:
                selected.append((score, emb, fpath, info))

    return selected


def main():
    print("=" * 60)
    print("M.O.L.O.C.H. ArcFace Re-Enrollment")
    print("=" * 60)

    # 1. Snapshots sammeln
    images = sorted(glob.glob(os.path.join(SNAPSHOTS_DIR, "*.jpg")))
    if not images:
        print(f"FEHLER: Keine JPGs in {SNAPSHOTS_DIR}")
        sys.exit(1)
    print(f"\n[1/5] {len(images)} Snapshots gefunden in {SNAPSHOTS_DIR}")

    # 2. Face-DB sichern
    print("\n[2/5] Face-DB Backup...")
    backup_face_db()

    # 3. Hailo VDevice + Modelle laden
    print("\n[3/5] Hailo NPU initialisieren...")
    try:
        from hailo_platform import HEF, VDevice, FormatType
    except ImportError:
        print("FEHLER: hailo_platform nicht installiert!")
        sys.exit(1)

    if not os.path.exists(SCRFD_HEF):
        print(f"FEHLER: SCRFD Modell nicht gefunden: {SCRFD_HEF}")
        sys.exit(1)
    if not os.path.exists(ARCFACE_HEF):
        print(f"FEHLER: ArcFace Modell nicht gefunden: {ARCFACE_HEF}")
        sys.exit(1)

    params = VDevice.create_params()
    vdevice = VDevice(params)

    # SCRFD laden
    scrfd_model = vdevice.create_infer_model(SCRFD_HEF)
    scrfd_model.input().set_format_type(FormatType.UINT8)
    scrfd_hef = HEF(SCRFD_HEF)
    scrfd_out_names = [o.name for o in scrfd_hef.get_output_vstream_infos()]
    for oname in scrfd_out_names:
        scrfd_model.output(oname).set_format_type(FormatType.FLOAT32)

    # ArcFace laden
    arcface_model = vdevice.create_infer_model(ARCFACE_HEF)
    arcface_model.input().set_format_type(FormatType.UINT8)
    arcface_hef = HEF(ARCFACE_HEF)
    arcface_out_names = [o.name for o in arcface_hef.get_output_vstream_infos()]
    for oname in arcface_out_names:
        arcface_model.output(oname).set_format_type(FormatType.FLOAT32)

    print(f"  SCRFD geladen: {SCRFD_HEF}")
    print(f"  ArcFace geladen: {ARCFACE_HEF}")

    # Modelle konfigurieren
    scrfd_ctx = scrfd_model.configure().__enter__()
    scrfd_bufs = {n: np.empty(scrfd_model.output(n).shape, dtype=np.float32)
                  for n in scrfd_out_names}
    scrfd_bindings = scrfd_ctx.create_bindings(output_buffers=scrfd_bufs)

    arcface_ctx = arcface_model.configure().__enter__()
    arcface_bufs = {n: np.empty(arcface_model.output(n).shape, dtype=np.float32)
                    for n in arcface_out_names}
    arcface_bindings = arcface_ctx.create_bindings(output_buffers=arcface_bufs)

    print("  Modelle konfiguriert.")

    # 4. Alle Snapshots durch SCRFD + ArcFace jagen
    print(f"\n[4/5] Verarbeite {len(images)} Snapshots...")
    candidates = []  # (score, embedding, filepath, info_dict)
    skipped_no_face = 0
    skipped_too_small = 0

    for idx, img_path in enumerate(images):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  {idx + 1}/{len(images)}...")

        frame = cv2.imread(img_path)
        if frame is None:
            continue
        fh, fw = frame.shape[:2]

        # Nur Full-HD verwenden (640x480 hat zu niedrige SCRFD-Scores)
        if fw < 1280:
            continue

        # Letterbox auf 640x640 (identisch zur Live-Pipeline)
        input_640, _scale, pad_x, pad_y, rw, rh = letterbox_resize(frame, 640)
        input_rgb = cv2.cvtColor(input_640, cv2.COLOR_BGR2RGB)

        # SCRFD Inference
        scrfd_bindings.input().set_buffer(np.ascontiguousarray(input_rgb))
        scrfd_ctx.run([scrfd_bindings], timeout=10000)
        outputs = {n: scrfd_bufs[n].copy() for n in scrfd_out_names}

        boxes, scores, landmarks = decode_scrfd(outputs, 640, SCRFD_CONF, SCRFD_NMS)
        if len(boxes) == 0:
            skipped_no_face += 1
            continue

        # Unletterbox: Model-Space -> Frame-Space (Boxes + Landmarks)
        boxes_c, lms_c = unletterbox_scrfd(boxes, landmarks, pad_x, pad_y, rw, rh)

        # Bestes Gesicht pro Bild (hoechster Score)
        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])
        box = boxes_c[best_idx]
        lm = lms_c[best_idx]  # (10,) = 5 Punkte x (x, y) normalisiert

        # Pixel-Koordinaten auf Original-Frame
        px1 = max(0, int(box[0] * fw))
        py1 = max(0, int(box[1] * fh))
        px2 = min(fw, int(box[2] * fw))
        py2 = min(fh, int(box[3] * fh))
        bw, bh = px2 - px1, py2 - py1

        if bw < MIN_FACE_PIX or bh < MIN_FACE_PIX:
            skipped_too_small += 1
            continue

        # Landmarks -> Pixel-Koordinaten fuer Alignment
        lm_pts = []
        for p in range(5):
            lm_pts.append([lm[p * 2] * fw, lm[p * 2 + 1] * fh])

        # Face Alignment: 5-Point Affine → 112x112 (wie TAPPAS)
        aligned = align_face(frame, lm_pts)
        if aligned is None:
            continue
        crop_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

        arcface_bindings.input().set_buffer(np.ascontiguousarray(crop_rgb))
        arcface_ctx.run([arcface_bindings], timeout=10000)
        emb_raw = arcface_bufs[arcface_out_names[0]].flatten().copy()
        norm = np.linalg.norm(emb_raw)
        if norm > 0:
            emb_raw = emb_raw / norm

        info = {
            "file": os.path.basename(img_path),
            "score": best_score,
            "bbox_px": [px1, py1, px2, py2],
            "face_size": [bw, bh],
        }
        candidates.append((best_score, emb_raw, img_path, info))

    print(f"\n  Ergebnis: {len(candidates)} Gesichter erkannt")
    print(f"  Kein Gesicht: {skipped_no_face}")
    print(f"  Zu klein: {skipped_too_small}")

    if not candidates:
        print("FEHLER: Kein einziges Gesicht erkannt!")
        sys.exit(1)

    # 5. Beste 20 auswaehlen (Score + Diversitaet)
    print(f"\n[5/5] Beste {TARGET_COUNT} auswaehlen (Score + Diversitaet)...")
    selected = select_diverse(candidates, TARGET_COUNT, DIVERSITY_THRESHOLD)
    print(f"  Ausgewaehlt: {len(selected)} Embeddings")

    # Infos anzeigen
    for i, (score, emb, fpath, info) in enumerate(selected):
        print(f"    #{i:2d}  score={score:.3f}  size={info['face_size']}  "
              f"file={info['file']}")

    # Face-DB aktualisieren
    db = load_face_db()

    # Alte markus-Eintraege entfernen
    old_keys = [k for k in db if k.lower().startswith("markus")]
    for k in old_keys:
        del db[k]
    print(f"\n  {len(old_keys)} alte Markus-Eintraege entfernt")

    # Bestes Embedding als Haupt-Referenz "markus"
    best_emb = selected[0][1]
    db["markus"] = best_emb.tolist()

    # Restliche als "markus#snap_N"
    for i, (score, emb, fpath, info) in enumerate(selected[1:]):
        key = f"markus#snap_{i}"
        db[key] = emb.tolist()

    save_face_db(db)
    print(f"  Face-DB gespeichert: {len(selected)} Markus-Embeddings")
    print(f"  Datei: {FACE_DB_PATH}")

    # Aufraeuemen
    del vdevice
    print("\n" + "=" * 60)
    print("FERTIG. Service starten mit: sudo systemctl start moloch")
    print("=" * 60)


if __name__ == "__main__":
    main()
