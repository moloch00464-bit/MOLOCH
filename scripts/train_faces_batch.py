#!/usr/bin/env python3
"""
!! DEPRECATED — NICHT VERWENDEN !!

Dieses Script erzeugt HailoRT-direkt Embeddings die INKOMPATIBEL mit der
TAPPAS GStreamer-Pipeline sind (anderes Face-Alignment → cosine sim ~0).

Stattdessen: Enrollment NUR ueber Live-Pipeline (IPC enrollment_start):
    echo '{"cmd":"enrollment_start","name":"markus","n":20}' | nc -U /tmp/moloch.sock

Oder per Chat-Keyword: "enrollment_start markus"

Siehe: CLAUDE.md Regel 11 — ArcFace Enrollment NUR ueber Live-Pipeline.

--- Original-Beschreibung (historisch) ---
M.O.L.O.C.H. Face Training — Batch-Enrollment aus Snapshot-Galerie.
Jagt alle JPGs aus snapshots/ durch SCRFD + ArcFace auf der Hailo-10H NPU.
"""
import sys
print("\\n!! DEPRECATED — Dieses Script erzeugt inkompatible Embeddings!")
print("   Nutze stattdessen IPC: enrollment_start markus")
print("   Siehe CLAUDE.md Regel 11\\n")
sys.exit(1)

import os
import sys
import json
import time
import glob
import argparse
import subprocess

import cv2
import numpy as np

# HailoRT
from hailo_platform import VDevice, HEF, FormatType

# --- Pfade ---
SNAPSHOTS_DIR = os.path.expanduser("~/moloch/snapshots")
FACES_DIR = os.path.expanduser("~/moloch/faces")
DAILY_DIR = "/mnt/moloch-data/Teachen"
FACE_DB_PATH = os.path.expanduser("~/moloch/data/face_embeddings.json")
MODELS_DIR = "/mnt/moloch-data/hailo/models"
SCRFD_HEF = os.path.join(MODELS_DIR, "scrfd_10g.hef")
ARCFACE_HEF = os.path.join(MODELS_DIR, "arcface_mobilefacenet.hef")

# --- Thresholds ---
SCRFD_CONF = 0.35
SCRFD_NMS = 0.45
MIN_FACE_SIZE = 30  # Pixel
ARCFACE_MIN_SIM = 0.35  # Mindest-Similarity zum Referenz-Embedding

# SCRFD Postprocess (gleiche Logik wie in einpraegen.py)
sys.path.insert(0, os.path.expanduser("~/moloch"))
from core.perception.hailo_postprocess import decode_scrfd


def letterbox_resize(frame, target_size=640):
    """Letterbox-Resize: Aspect-Ratio erhalten, schwarz auffuellen."""
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas_rgb, scale, pad_x, pad_y


def unletterbox_coords(x1n, y1n, x2n, y2n, target_size, scale, pad_x, pad_y, orig_w, orig_h):
    """Normalisierte BBox aus Letterbox zurueck auf Originalbild."""
    px1 = (x1n * target_size - pad_x) / scale
    py1 = (y1n * target_size - pad_y) / scale
    px2 = (x2n * target_size - pad_x) / scale
    py2 = (y2n * target_size - pad_y) / scale
    return max(0, int(px1)), max(0, int(py1)), min(orig_w, int(px2)), min(orig_h, int(py2))


def collect_images():
    """Alle JPGs aus snapshots/ und faces/ sammeln."""
    images = []
    # Snapshots-Galerie
    if os.path.isdir(SNAPSHOTS_DIR):
        for p in sorted(glob.glob(os.path.join(SNAPSHOTS_DIR, "*.jpg"))):
            images.append(p)
    # faces/ Ordner (pro Person)
    if os.path.isdir(FACES_DIR):
        for p in sorted(glob.glob(os.path.join(FACES_DIR, "**", "*.jpg"), recursive=True)):
            images.append(p)
    # daily/ Teaching-Fotos auf SSD2
    if os.path.isdir(DAILY_DIR):
        for p in sorted(glob.glob(os.path.join(DAILY_DIR, "**", "*.jpg"), recursive=True)):
            images.append(p)
    return images


def load_face_db():
    """Bestehende Face-DB laden."""
    if not os.path.exists(FACE_DB_PATH):
        return {}
    with open(FACE_DB_PATH, "r") as f:
        return json.load(f)


def save_face_db(db):
    """Face-DB atomar speichern."""
    os.makedirs(os.path.dirname(FACE_DB_PATH), exist_ok=True)
    tmp = FACE_DB_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(db, f, indent=1, ensure_ascii=False)
    os.replace(tmp, FACE_DB_PATH)


def setup_hailo():
    """VDevice + SCRFD + ArcFace konfigurieren (gleiche API wie ModelOrchestrator)."""
    print("[NPU] VDevice erstellen...")
    params = VDevice.create_params()
    vdevice = VDevice(params)

    # SCRFD laden + konfigurieren
    print(f"[NPU] SCRFD laden: {SCRFD_HEF}")
    scrfd_hef = HEF(SCRFD_HEF)
    scrfd_model = vdevice.create_infer_model(SCRFD_HEF)
    scrfd_model.input().set_format_type(FormatType.UINT8)
    scrfd_out_names = [o.name for o in scrfd_hef.get_output_vstream_infos()]
    for oname in scrfd_out_names:
        scrfd_model.output(oname).set_format_type(FormatType.FLOAT32)
    scrfd_ctx = scrfd_model.configure()
    scrfd_configured = scrfd_ctx.__enter__()
    scrfd_out_bufs = {n: np.empty(scrfd_model.output(n).shape, dtype=np.float32) for n in scrfd_out_names}
    scrfd_bindings = scrfd_configured.create_bindings(output_buffers=scrfd_out_bufs)

    # ArcFace laden + konfigurieren
    print(f"[NPU] ArcFace laden: {ARCFACE_HEF}")
    arcface_hef = HEF(ARCFACE_HEF)
    arcface_model = vdevice.create_infer_model(ARCFACE_HEF)
    arcface_model.input().set_format_type(FormatType.UINT8)
    arcface_out_names = [o.name for o in arcface_hef.get_output_vstream_infos()]
    for oname in arcface_out_names:
        arcface_model.output(oname).set_format_type(FormatType.FLOAT32)
    arcface_ctx = arcface_model.configure()
    arcface_configured = arcface_ctx.__enter__()
    arcface_out_bufs = {n: np.empty(arcface_model.output(n).shape, dtype=np.float32) for n in arcface_out_names}
    arcface_bindings = arcface_configured.create_bindings(output_buffers=arcface_out_bufs)

    return {
        "vdevice": vdevice,
        "scrfd": {"configured": scrfd_configured, "bindings": scrfd_bindings,
                  "out_bufs": scrfd_out_bufs, "out_names": scrfd_out_names},
        "arcface": {"configured": arcface_configured, "bindings": arcface_bindings,
                    "out_bufs": arcface_out_bufs, "out_names": arcface_out_names},
    }


def run_scrfd(ctx, input_rgb):
    """SCRFD Inference — gibt decodierte Faces zurueck."""
    ctx["bindings"].input().set_buffer(np.ascontiguousarray(input_rgb))
    ctx["configured"].run([ctx["bindings"]], timeout=10000)
    outputs = {n: ctx["out_bufs"][n].copy() for n in ctx["out_names"]}
    return decode_scrfd(outputs, 640, SCRFD_CONF, SCRFD_NMS)


def run_arcface(ctx, crop_rgb_112):
    """ArcFace Inference — gibt 512-dim Embedding zurueck."""
    ctx["bindings"].input().set_buffer(np.ascontiguousarray(crop_rgb_112))
    ctx["configured"].run([ctx["bindings"]], timeout=10000)
    # Erstes Output (512-dim embedding)
    emb = ctx["out_bufs"][ctx["out_names"][0]].flatten().copy()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def main():
    parser = argparse.ArgumentParser(description="M.O.L.O.C.H. Face Training Batch")
    parser.add_argument("--dry-run", action="store_true", help="Nur zaehlen, nichts speichern")
    parser.add_argument("--person", default="markus", help="Name der Person (default: markus)")
    parser.add_argument("--no-restart", action="store_true", help="Service nicht stoppen/starten")
    args = parser.parse_args()

    person_name = args.person.capitalize()

    # Bilder sammeln
    images = collect_images()
    print(f"\n=== M.O.L.O.C.H. Face Training ===")
    print(f"Bilder gefunden: {len(images)}")
    print(f"  snapshots/: {len([i for i in images if 'snapshots' in i])}")
    print(f"  faces/:     {len([i for i in images if '/faces/' in i])}")
    print(f"  daily/:     {len([i for i in images if '/daily/' in i])}")
    print(f"Person: {person_name}")
    print(f"Dry-Run: {args.dry_run}")
    print()

    if not images:
        print("Keine Bilder gefunden!")
        return

    # Service stoppen (NPU freigeben)
    if not args.no_restart:
        print("[SERVICE] Stoppe moloch...")
        subprocess.run(["sudo", "systemctl", "stop", "moloch"], check=True)
        time.sleep(3)
        print("[SERVICE] Gestoppt.")

    try:
        # NPU Setup
        hailo = setup_hailo()
        print("[NPU] SCRFD + ArcFace bereit.\n")

        # Face-DB laden
        face_db = load_face_db()
        old_count = len(face_db)

        # Referenz-Embedding (Durchschnitt aller existierenden Markus-Embeddings)
        ref_embeddings = []
        for key, emb_list in face_db.items():
            if key.lower().startswith(person_name.lower()):
                emb = np.array(emb_list, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                ref_embeddings.append(emb)

        if ref_embeddings:
            # Durchschnitt aller bisherigen Embeddings als Referenz
            ref_mean = np.mean(ref_embeddings, axis=0)
            ref_mean = ref_mean / np.linalg.norm(ref_mean)
            print(f"[REF] {person_name} Referenz: {len(ref_embeddings)} Embeddings (Mittelwert)")
        else:
            ref_mean = None
            print(f"[REF] Keine {person_name}-Referenz — alle Gesichter werden als {person_name} gespeichert!")

        # Statistik
        stats = {
            "total": len(images),
            "faces_found": 0,
            "saved": 0,
            "skipped_small": 0,
            "skipped_low_sim": 0,
            "skipped_no_face": 0,
        }

        t_start = time.time()

        for idx, img_path in enumerate(images):
            fname = os.path.basename(img_path)
            progress = f"[{idx+1}/{len(images)}]"

            frame = cv2.imread(img_path)
            if frame is None:
                continue
            fh, fw = frame.shape[:2]

            # SCRFD: Face Detection
            input_rgb, lb_scale, lb_pad_x, lb_pad_y = letterbox_resize(frame, 640)
            boxes, scores, landmarks = run_scrfd(hailo["scrfd"], input_rgb)

            if len(boxes) == 0:
                stats["skipped_no_face"] += 1
                if (idx + 1) % 50 == 0:
                    print(f"{progress} {fname} — kein Gesicht")
                continue

            for face_idx in range(len(boxes)):
                x1n, y1n, x2n, y2n = boxes[face_idx]
                conf = scores[face_idx]
                x1, y1, x2, y2 = unletterbox_coords(
                    x1n, y1n, x2n, y2n, 640, lb_scale, lb_pad_x, lb_pad_y, fw, fh)
                bw, bh = x2 - x1, y2 - y1

                if bw < MIN_FACE_SIZE or bh < MIN_FACE_SIZE:
                    stats["skipped_small"] += 1
                    continue

                stats["faces_found"] += 1

                # Face Crop mit 20% Margin
                mx, my = int(bw * 0.2), int(bh * 0.2)
                cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
                cx2, cy2 = min(fw, x2 + mx), min(fh, y2 + my)
                crop = frame[cy1:cy2, cx1:cx2]
                crop_112 = cv2.resize(crop, (112, 112))
                crop_rgb = cv2.cvtColor(crop_112, cv2.COLOR_BGR2RGB)

                # ArcFace Embedding
                embedding = run_arcface(hailo["arcface"], crop_rgb)
                if embedding is None:
                    continue

                # Similarity-Check gegen Referenz
                sim = 0.0
                if ref_mean is not None:
                    sim = float(np.dot(embedding, ref_mean))
                    if sim < ARCFACE_MIN_SIM:
                        stats["skipped_low_sim"] += 1
                        continue

                # Speichern
                key = f"{person_name}#train_{idx}_{face_idx}"
                if not args.dry_run:
                    face_db[key] = embedding.tolist()
                stats["saved"] += 1

                if stats["saved"] % 10 == 0 or stats["saved"] == 1:
                    print(f"{progress} {fname} — Gesicht {face_idx}: conf={conf:.2f} sim={sim:.3f} -> GESPEICHERT ({stats['saved']})")

        duration = time.time() - t_start

        # Speichern
        if not args.dry_run and stats["saved"] > 0:
            save_face_db(face_db)
            print(f"\n[DB] Gespeichert: {old_count} -> {len(face_db)} Embeddings")

        # Statistik
        print(f"\n=== ERGEBNIS ===")
        print(f"Bilder:            {stats['total']}")
        print(f"Gesichter gefunden: {stats['faces_found']}")
        print(f"Gespeichert:       {stats['saved']}")
        print(f"Kein Gesicht:      {stats['skipped_no_face']}")
        print(f"Zu klein:          {stats['skipped_small']}")
        print(f"Zu niedrige Sim:   {stats['skipped_low_sim']}")
        print(f"Dauer:             {duration:.1f}s ({len(images)/max(duration,0.1):.1f} Bilder/s)")

        # VDevice freigeben
        del hailo

    finally:
        # Service wieder starten
        if not args.no_restart:
            print("\n[SERVICE] Starte moloch...")
            subprocess.run(["sudo", "systemctl", "start", "moloch"], check=True)
            time.sleep(5)
            active = subprocess.run(["systemctl", "is-active", "moloch"],
                                   capture_output=True, text=True).stdout.strip()
            print(f"[SERVICE] Status: {active}")


if __name__ == "__main__":
    main()
