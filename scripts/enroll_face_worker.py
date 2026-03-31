#!/usr/bin/env python3
"""
Enrollment mit dem IDENTISCHEN Code wie FaceWorker.

Nutzt face_pipeline.py: letterbox_resize + decode_scrfd +
unletterbox_coords + align_face + ArcFace via HailoRT-Direct.

Laeuft PARALLEL zum MOLOCH Service (SHARED VDevice).

Usage:
    python3 scripts/enroll_face_worker.py                     # Default: markus
    python3 scripts/enroll_face_worker.py --name sven
    python3 scripts/enroll_face_worker.py --dir snapshots/markus --name markus
"""

import os
import sys
import json
import time
import glob
import shutil
import argparse
import numpy as np
import cv2

PROJECT_ROOT = os.path.expanduser("~/moloch")
sys.path.insert(0, PROJECT_ROOT)

from core.perception.face_pipeline import (
    letterbox_resize, unletterbox_coords, align_face,
    SCRFD_HEF, ARCFACE_HEF, FACE_DB_PATH,
    SCRFD_CONF_THRESH, SCRFD_NMS_THRESH, MIN_FACE_PIX,
)
from core.perception.hailo_postprocess import decode_scrfd, normalize_arcface
from core.perception.vision_workers import create_configured_model, VDEVICE_GROUP_ID


def main():
    parser = argparse.ArgumentParser(description="FaceWorker-kompatibles Enrollment")
    parser.add_argument("--name", type=str, default="markus")
    parser.add_argument("--dir", type=str, default=None,
                        help="Snapshot-Verzeichnis (default: snapshots/<name>)")
    parser.add_argument("--conf", type=float, default=SCRFD_CONF_THRESH)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        if os.path.exists(FACE_DB_PATH):
            db = json.load(open(FACE_DB_PATH))
            for k in sorted(db.keys()):
                arr = np.array(db[k])
                print(f"  {k}: {len(arr)}d, norm={np.linalg.norm(arr):.4f}")
            print(f"\nTotal: {len(db)} Eintraege")
        else:
            print("Keine Face-DB vorhanden.")
        return

    name = args.name.lower()
    snap_dir = args.dir or os.path.join(PROJECT_ROOT, "snapshots", name)

    print("=" * 60)
    print("FaceWorker-kompatibles Enrollment (HailoRT-Direct)")
    print("=" * 60)
    print(f"  Name:      {name}")
    print(f"  Snapshots: {snap_dir}")
    print(f"  Conf:      {args.conf}")
    print(f"  Max Embs:  {args.top}")

    # Bilder sammeln
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(sorted(glob.glob(os.path.join(snap_dir, ext))))
    if not images:
        print(f"\nFEHLER: Keine Bilder in {snap_dir}")
        sys.exit(1)
    print(f"\n[1/4] {len(images)} Bilder gefunden")

    # Backup
    if os.path.exists(FACE_DB_PATH):
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = f"{FACE_DB_PATH}.backup_{ts}"
        shutil.copy2(FACE_DB_PATH, backup)
        print(f"[2/4] Backup: {backup}")
    else:
        print("[2/4] Keine bestehende DB — neu anlegen")

    # NPU initialisieren (SHARED VDevice)
    print("\n[3/4] NPU initialisieren (SHARED VDevice)...")
    import hailo_platform as hp
    params = hp.VDevice.create_params()
    params.group_id = VDEVICE_GROUP_ID
    vdevice = hp.VDevice(params)

    _, scrfd_cfg, _, scrfd_outs, scrfd_shapes = create_configured_model(vdevice, SCRFD_HEF)
    _, arcface_cfg, _, arcface_outs, arcface_shapes = create_configured_model(vdevice, ARCFACE_HEF)
    print("  SCRFD + ArcFace geladen")

    # Verarbeiten
    print(f"\n[4/4] Verarbeite {len(images)} Bilder...")
    candidates = []
    stats = {"total": 0, "kein_gesicht": 0, "low_conf": 0, "zu_klein": 0,
             "align_fail": 0, "ok": 0}

    for idx, img_path in enumerate(images):
        stats["total"] += 1
        frame_bgr = cv2.imread(img_path)
        if frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        fh, fw = frame_rgb.shape[:2]

        # Letterbox (IDENTISCH zum FaceWorker!)
        padded, _scale, pad_x, pad_y, rw, rh = letterbox_resize(frame_rgb, 640)

        # SCRFD
        scrfd_bindings = scrfd_cfg.create_bindings()
        scrfd_bindings.input().set_buffer(np.ascontiguousarray(padded))
        scrfd_bufs = {}
        for n in scrfd_outs:
            buf = np.empty(scrfd_shapes[n], dtype=np.float32)
            scrfd_bindings.output(n).set_buffer(buf)
            scrfd_bufs[n] = buf
        scrfd_cfg.run([scrfd_bindings], 10000)
        outputs = {n: scrfd_bufs[n].copy() for n in scrfd_outs}

        boxes, scores, landmarks = decode_scrfd(outputs, 640, 0.3, SCRFD_NMS_THRESH)
        if len(boxes) == 0:
            stats["kein_gesicht"] += 1
            continue

        # Unletterbox (IDENTISCH zum FaceWorker!)
        boxes_norm, landmarks_norm = unletterbox_coords(
            boxes, landmarks, pad_x, pad_y, rw, rh)

        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])
        if best_score < args.conf:
            stats["low_conf"] += 1
            continue

        box = boxes_norm[best_idx]
        lm = landmarks_norm[best_idx]

        px1 = max(0, int(box[0] * fw))
        py1 = max(0, int(box[1] * fh))
        px2 = min(fw, int(box[2] * fw))
        py2 = min(fh, int(box[3] * fh))
        bw, bh = px2 - px1, py2 - py1

        if bw < MIN_FACE_PIX or bh < MIN_FACE_PIX:
            stats["zu_klein"] += 1
            continue

        # Alignment (IDENTISCH zum FaceWorker!)
        landmarks_px = []
        for p in range(5):
            landmarks_px.append([lm[p * 2] * fw, lm[p * 2 + 1] * fh])

        aligned = align_face(frame_rgb, landmarks_px)
        if aligned is None:
            stats["align_fail"] += 1
            continue

        # ArcFace (IDENTISCH zum FaceWorker!)
        arcface_bindings = arcface_cfg.create_bindings()
        arcface_bindings.input().set_buffer(np.ascontiguousarray(aligned))
        arcface_bufs = {}
        for n in arcface_outs:
            buf = np.empty(arcface_shapes[n], dtype=np.float32)
            arcface_bindings.output(n).set_buffer(buf)
            arcface_bufs[n] = buf
        arcface_cfg.run([arcface_bindings], 10000)

        emb_raw = arcface_bufs[arcface_outs[0]].flatten().copy()
        emb = normalize_arcface(emb_raw)

        candidates.append((best_score, emb, os.path.basename(img_path), [bw, bh]))
        stats["ok"] += 1

        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"  {idx+1}/{len(images)} — score={best_score:.3f} size={bw}x{bh}")

    print(f"\n=== Statistik ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if not candidates:
        print("\nFEHLER: Kein Gesicht gefunden!")
        del vdevice
        sys.exit(1)

    # Beste auswaehlen (Score-basiert, Diversitaet via Cosine-Pruefung)
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for score, emb, fname, size in candidates:
        if len(selected) >= args.top:
            break
        is_diverse = True
        for _, sel_emb, _, _ in selected:
            if float(np.dot(emb, sel_emb)) > 0.85:
                is_diverse = False
                break
        if is_diverse:
            selected.append((score, emb, fname, size))

    # Rest auffuellen
    for score, emb, fname, size in candidates:
        if len(selected) >= args.top:
            break
        if not any(f == fname for _, _, f, _ in selected):
            selected.append((score, emb, fname, size))

    print(f"\nAusgewaehlt: {len(selected)} Embeddings")
    for i, (score, emb, fname, size) in enumerate(selected):
        print(f"  #{i}: score={score:.3f} size={size} file={fname}")

    # DB speichern
    db = {}
    if os.path.exists(FACE_DB_PATH):
        db = json.load(open(FACE_DB_PATH))

    # Alte Eintraege fuer diesen Namen entfernen
    old_keys = [k for k in db if k.lower().startswith(name)]
    for k in old_keys:
        del db[k]
    if old_keys:
        print(f"\n  {len(old_keys)} alte '{name}'-Eintraege entfernt")

    # Durchschnitts-Embedding
    all_embs = [emb for _, emb, _, _ in selected]
    avg_emb = np.mean(all_embs, axis=0)
    avg_emb = normalize_arcface(avg_emb)
    db[name] = avg_emb.tolist()

    # Individuelle
    for i, (score, emb, fname, size) in enumerate(selected):
        db[f"{name}#snap_{i}"] = emb.tolist()

    # Atomar speichern
    os.makedirs(os.path.dirname(FACE_DB_PATH), exist_ok=True)
    tmp = FACE_DB_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=1, ensure_ascii=False)
    os.replace(tmp, FACE_DB_PATH)

    snap_count = sum(1 for k in db if k.startswith(f"{name}#"))
    print(f"\n=== Gespeichert ===")
    print(f"  DB:       {FACE_DB_PATH}")
    print(f"  '{name}': 1 Durchschnitt + {snap_count} individuelle")
    print(f"  Total:    {len(db)} Eintraege")

    # Verifikation: Match gegen neue DB
    print(f"\n=== Verifikation ===")
    from core.perception.hailo_postprocess import match_face
    face_db = {k: np.array(v, dtype=np.float32) for k, v in db.items()}
    test_emb = selected[0][1]  # Erstes Embedding testen
    matched_name, sim = match_face(test_emb, face_db, 0.60)
    print(f"  Self-Match: {matched_name} (sim={sim:.4f})")
    if sim >= 0.60:
        print(f"  ✅ ENROLLMENT ERFOLGREICH")
    else:
        print(f"  ❌ Self-Match zu niedrig — Problem!")

    del vdevice
    print("\n" + "=" * 60)
    print("FERTIG")
    print("=" * 60)


if __name__ == "__main__":
    main()
