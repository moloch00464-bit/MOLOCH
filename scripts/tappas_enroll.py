#!/usr/bin/env python3
"""
TAPPAS-kompatibles ArcFace Enrollment auf Hailo-10H NPU.

Nutzt das GLEICHE Preprocessing wie die Live-TAPPAS-Pipeline:
  1. Letterbox 640x640 (Aspektverhaeltnis)
  2. SCRFD Face Detection (Landmarks)
  3. Face Alignment via 5-Point Affine Transform (wie libvms_face_align.so)
  4. ArcFace 112x112 Embedding

ACHTUNG: sudo systemctl stop moloch VORHER — NPU-Konflikt!

Usage:
    sudo systemctl stop moloch
    python3 scripts/tappas_enroll.py
    sudo systemctl start moloch

    # Optionen:
    python3 scripts/tappas_enroll.py --name markus --conf 0.50 --top 20
    python3 scripts/tappas_enroll.py --list
"""

import os
import sys
import json
import time
import glob
import shutil
import argparse

import cv2
import numpy as np

# Projekt-Root zum Path
PROJECT_ROOT = os.path.expanduser("~/moloch")
sys.path.insert(0, PROJECT_ROOT)

from core.perception.hailo_postprocess import decode_scrfd, normalize_arcface

# Pfade
SNAPSHOTS_DIR = os.path.join(PROJECT_ROOT, "snapshots")
FACE_DB_PATH = os.path.join(PROJECT_ROOT, "data", "face_embeddings.json")
MODEL_DIR = "/mnt/moloch-data/hailo/models"
SCRFD_HEF = os.path.join(MODEL_DIR, "scrfd_10g.hef")
ARCFACE_HEF = os.path.join(MODEL_DIR, "arcface_mobilefacenet.hef")

# Parameter
SCRFD_CONF = 0.50
SCRFD_NMS = 0.40
MIN_FACE_PIX = 40
TARGET_COUNT = 20
DIVERSITY_THRESHOLD = 0.85

# ArcFace Standard-Referenz-Landmarks fuer 112x112
# (gleiche Positionen wie libvms_face_align.so verwendet)
ARCFACE_REF_LANDMARKS = np.array([
    [38.2946, 51.6963],  # linkes Auge
    [73.5318, 51.5014],  # rechtes Auge
    [56.0252, 71.7366],  # Nase
    [41.5493, 92.3655],  # linker Mundwinkel
    [70.7299, 92.2041],  # rechter Mundwinkel
], dtype=np.float32)


def letterbox_resize(img, target_size=640):
    """Letterbox-Resize identisch zur TAPPAS hailocropper Pipeline."""
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


def unletterbox_coords(boxes, landmarks, pad_x, pad_y, rw, rh, target=640):
    """Letterbox-Space -> normalisierte [0,1] Koordinaten relativ zum Content."""
    bc = boxes.copy()
    bc[:, [0, 2]] = np.clip((boxes[:, [0, 2]] * target - pad_x) / rw, 0, 1)
    bc[:, [1, 3]] = np.clip((boxes[:, [1, 3]] * target - pad_y) / rh, 0, 1)

    lc = landmarks.copy()
    # Landmarks: (N, 10) = 5 Punkte x (x, y), normalisiert auf [0,1]
    for i in range(5):
        lc[:, i * 2] = np.clip((landmarks[:, i * 2] * target - pad_x) / rw, 0, 1)
        lc[:, i * 2 + 1] = np.clip((landmarks[:, i * 2 + 1] * target - pad_y) / rh, 0, 1)
    return bc, lc


def align_face(img, landmarks_5pt):
    """Face Alignment via 5-Point Affine Transform.

    Gleiche Logik wie TAPPAS libvms_face_align.so:
    Berechnet Similarity-Transform von erkannten 5 Landmarks
    auf die ArcFace-Referenz-Landmarks (112x112).

    Args:
        img: Original-Frame (BGR)
        landmarks_5pt: 5 Landmark-Punkte als Pixel-Koordinaten [(x,y), ...]

    Returns:
        aligned: 112x112 BGR Bild (face-aligned)
    """
    src_pts = np.array(landmarks_5pt, dtype=np.float32)
    dst_pts = ARCFACE_REF_LANDMARKS

    # Similarity-Transform (Rotation + Scale + Translation, KEIN Shear)
    tform, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if tform is None:
        # Fallback: einfaches Crop+Resize wenn Alignment fehlschlaegt
        return None

    aligned = cv2.warpAffine(img, tform, (112, 112), borderValue=(0, 0, 0))
    return aligned


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
    """Aus Kandidaten die diversesten auswaehlen (Score + Diversitaet)."""
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for score, emb, fpath, info in candidates:
        if len(selected) >= target_count:
            break
        is_diverse = True
        for _, sel_emb, _, _ in selected:
            sim = float(np.dot(emb, sel_emb))
            if sim > sim_threshold:
                is_diverse = False
                break
        if is_diverse:
            selected.append((score, emb, fpath, info))

    # Rest nach Score auffuellen falls noetig
    if len(selected) < target_count:
        for score, emb, fpath, info in candidates:
            if len(selected) >= target_count:
                break
            already = any(f == fpath for _, _, f, _ in selected)
            if not already:
                selected.append((score, emb, fpath, info))

    return selected


def list_db():
    """Bestehende Face-DB anzeigen."""
    if not os.path.exists(FACE_DB_PATH):
        print(f"Keine DB vorhanden: {FACE_DB_PATH}")
        return
    with open(FACE_DB_PATH, "r") as f:
        db = json.load(f)
    print(f"Face-DB: {FACE_DB_PATH}")
    base_names = set()
    for name, emb in db.items():
        arr = np.array(emb)
        base = name.split('#')[0]
        base_names.add(base)
        print(f"  {name}: {len(emb)}-dim, norm={np.linalg.norm(arr):.4f}")
    print(f"\nPersonen: {sorted(base_names)}")


def main():
    parser = argparse.ArgumentParser(description="TAPPAS-kompatibles ArcFace Enrollment")
    parser.add_argument("--name", type=str, default="markus", help="Name der Person (default: markus)")
    parser.add_argument("--conf", type=float, default=SCRFD_CONF, help=f"SCRFD Confidence (default: {SCRFD_CONF})")
    parser.add_argument("--top", type=int, default=TARGET_COUNT, help=f"Maximale Embedding-Anzahl (default: {TARGET_COUNT})")
    parser.add_argument("--align", action="store_true",
                        help="Face Alignment aktivieren (nur fuer TAPPAS-Pipeline, NICHT fuer InferenceEngine!)")
    parser.add_argument("--list", action="store_true", help="Face-DB anzeigen")
    args = parser.parse_args()

    if args.list:
        list_db()
        return

    name = args.name.lower()
    conf_thresh = args.conf
    target_count = args.top
    use_alignment = args.align

    print("=" * 60)
    print("M.O.L.O.C.H. TAPPAS-kompatibles ArcFace Enrollment")
    print("=" * 60)
    print(f"  Name:       {name}")
    print(f"  Conf:       {conf_thresh}")
    print(f"  Max Embs:   {target_count}")
    print(f"  Snapshots:  {SNAPSHOTS_DIR}")
    if use_alignment:
        print(f"  ALIGNMENT:  5-Point Affine (wie TAPPAS libvms_face_align.so)")
    else:
        print(f"  ALIGNMENT:  AUS (Crop+Resize, kompatibel mit InferenceEngine)")
    print()

    # 1. Snapshots sammeln
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(sorted(glob.glob(os.path.join(SNAPSHOTS_DIR, ext))))
    if not images:
        print(f"FEHLER: Keine Bilder in {SNAPSHOTS_DIR}")
        sys.exit(1)
    print(f"[1/5] {len(images)} Bilder gefunden")

    # 2. Face-DB Backup
    print("\n[2/5] Face-DB Backup...")
    backup_face_db()

    # 3. Hailo NPU initialisieren
    print("\n[3/5] Hailo NPU initialisieren...")
    try:
        from hailo_platform import HEF, VDevice, FormatType
    except ImportError:
        print("FEHLER: hailo_platform nicht installiert!")
        sys.exit(1)

    for hef_path in (SCRFD_HEF, ARCFACE_HEF):
        if not os.path.exists(hef_path):
            print(f"FEHLER: Modell nicht gefunden: {hef_path}")
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

    print(f"  SCRFD:   {SCRFD_HEF}")
    print(f"  ArcFace: {ARCFACE_HEF}")

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

    # 4. Alle Snapshots verarbeiten: SCRFD -> ArcFace
    preprocess_mode = "TAPPAS Letterbox" if use_alignment else "Squish (wie InferenceEngine)"
    print(f"\n[4/5] Verarbeite {len(images)} Bilder ({preprocess_mode} + SCRFD + ArcFace)...")
    candidates = []
    stats = {"total": len(images), "unlesbar": 0, "kein_gesicht": 0,
             "low_conf": 0, "zu_klein": 0, "align_fail": 0, "ok": 0}

    for idx, img_path in enumerate(images):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  {idx + 1}/{len(images)} ({stats['ok']} Gesichter bisher)...")

        frame = cv2.imread(img_path)
        if frame is None:
            stats["unlesbar"] += 1
            continue
        fh, fw = frame.shape[:2]

        if use_alignment:
            # TAPPAS-Modus: Letterbox auf 640x640 (Aspektverhaeltnis beibehalten)
            input_640, _scale, pad_x, pad_y, rw, rh = letterbox_resize(frame, 640)
            input_rgb = cv2.cvtColor(input_640, cv2.COLOR_BGR2RGB)
        else:
            # InferenceEngine-Modus: Squish auf 640x640 (identisch zur Live-Pipeline)
            input_640 = cv2.resize(frame, (640, 640))
            input_rgb = cv2.cvtColor(input_640, cv2.COLOR_BGR2RGB)

        # SCRFD Inference
        scrfd_bindings.input().set_buffer(np.ascontiguousarray(input_rgb))
        scrfd_ctx.run([scrfd_bindings], timeout=10000)
        outputs = {n: scrfd_bufs[n].copy() for n in scrfd_out_names}

        boxes, scores, landmarks = decode_scrfd(outputs, 640, 0.3, SCRFD_NMS)
        if len(boxes) == 0:
            stats["kein_gesicht"] += 1
            continue

        if use_alignment:
            # Letterbox: Unletterbox-Korrektur
            boxes_c, landmarks_c = unletterbox_coords(boxes, landmarks, pad_x, pad_y, rw, rh)
        else:
            # Squish: Boxes sind schon in normalisiertem [0,1] Space
            boxes_c = boxes
            landmarks_c = landmarks

        # Bestes Gesicht (hoechster Score)
        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])

        if best_score < conf_thresh:
            stats["low_conf"] += 1
            continue

        box = boxes_c[best_idx]
        lm = landmarks_c[best_idx]  # 10 Werte: x0,y0,x1,y1,...,x4,y4

        # Pixel-Koordinaten auf Original-Frame
        px1 = max(0, int(box[0] * fw))
        py1 = max(0, int(box[1] * fh))
        px2 = min(fw, int(box[2] * fw))
        py2 = min(fh, int(box[3] * fh))
        bw, bh = px2 - px1, py2 - py1

        if bw < MIN_FACE_PIX or bh < MIN_FACE_PIX:
            stats["zu_klein"] += 1
            continue

        if use_alignment:
            # 5 Landmarks in Pixel-Koordinaten
            landmarks_px = []
            for i in range(5):
                lx = lm[i * 2] * fw
                ly = lm[i * 2 + 1] * fh
                landmarks_px.append([lx, ly])

            # Face Alignment (wie TAPPAS libvms_face_align.so)
            aligned = align_face(frame, landmarks_px)
            if aligned is None:
                stats["align_fail"] += 1
                continue
            crop_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        else:
            # Crop + Resize (identisch zur InferenceEngine Live-Pipeline)
            mx, my = int(bw * 0.2), int(bh * 0.2)
            cx1 = max(0, px1 - mx)
            cy1 = max(0, py1 - my)
            cx2 = min(fw, px2 + mx)
            cy2 = min(fh, py2 + my)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            crop_112 = cv2.resize(crop, (112, 112))
            crop_rgb = cv2.cvtColor(crop_112, cv2.COLOR_BGR2RGB)
        arcface_bindings.input().set_buffer(np.ascontiguousarray(crop_rgb))
        arcface_ctx.run([arcface_bindings], timeout=10000)
        emb_raw = arcface_bufs[arcface_out_names[0]].flatten().copy()
        emb = normalize_arcface(emb_raw)

        info = {
            "file": os.path.basename(img_path),
            "score": best_score,
            "bbox_px": [px1, py1, px2, py2],
            "face_size": [bw, bh],
            "aligned": True,
        }
        candidates.append((best_score, emb, img_path, info))
        stats["ok"] += 1

    print(f"\n=== Statistik ===")
    print(f"  Bilder total:    {stats['total']}")
    print(f"  Unlesbar:        {stats['unlesbar']}")
    print(f"  Kein Gesicht:    {stats['kein_gesicht']}")
    print(f"  Low Confidence:  {stats['low_conf']} (< {conf_thresh})")
    print(f"  Zu klein:        {stats['zu_klein']} (< {MIN_FACE_PIX}px)")
    print(f"  Align fehlgesch: {stats['align_fail']}")
    print(f"  Gesichter OK:    {stats['ok']}")

    if not candidates:
        print("\nFEHLER: Kein einziges Gesicht ueber dem Threshold!")
        del vdevice
        sys.exit(1)

    # 5. Beste auswaehlen (Score + Diversitaet) und speichern
    print(f"\n[5/5] Beste {target_count} auswaehlen (Score + Diversitaet)...")
    selected = select_diverse(candidates, target_count, DIVERSITY_THRESHOLD)
    print(f"  Ausgewaehlt: {len(selected)} Embeddings")

    for i, (score, emb, fpath, info) in enumerate(selected):
        print(f"    #{i:2d}  score={score:.3f}  size={info['face_size']}  file={info['file']}")

    # Face-DB laden, alte Eintraege entfernen
    db = load_face_db()
    old_keys = [k for k in db if k.lower().startswith(name)]
    for k in old_keys:
        del db[k]
    if old_keys:
        print(f"\n  {len(old_keys)} alte '{name}'-Eintraege entfernt")

    # Durchschnitts-Embedding als Haupt-Referenz
    all_embs = [emb for _, emb, _, _ in selected]
    avg_emb = np.mean(all_embs, axis=0)
    avg_emb = normalize_arcface(avg_emb)
    db[name] = avg_emb.tolist()

    # Individuelle Embeddings als name#snap_N
    for i, (score, emb, fpath, info) in enumerate(selected):
        key = f"{name}#snap_{i}"
        db[key] = emb.tolist()

    save_face_db(db)
    snap_count = sum(1 for k in db if k.startswith(f"{name}#snap_"))
    print(f"\n=== Gespeichert ===")
    print(f"  DB:         {FACE_DB_PATH}")
    print(f"  '{name}':   1 Durchschnitt + {snap_count} individuelle = {snap_count + 1} total")
    print(f"  Keys total: {len(db)}")

    # Aufraeumen
    del vdevice
    print("\n" + "=" * 60)
    print("FERTIG. Service starten mit: sudo systemctl start moloch")
    print("=" * 60)


if __name__ == "__main__":
    main()
