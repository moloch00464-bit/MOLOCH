#!/usr/bin/env python3
"""
Interaktives PTZ-Tracking Kalibrier-Script.

Prueft ob Pan/Tilt-Vorzeichen korrekt sind:
  - Markus steht vor der Kamera
  - Geht nach links/rechts
  - Script vergleicht Face-Position im Bild mit Kamera-Pan-Richtung

Standalone — braucht KEINEN laufenden moloch Service.
Nutzt HailoRT direkt fuer SCRFD Face Detection.

Usage: python3 scripts/calibrate_tracking.py
"""

import os
import sys
import time
import json
import logging

import cv2
import numpy as np

# Projekt-Root in sys.path damit Imports funktionieren
sys.path.insert(0, os.path.expanduser("~/moloch"))

from hailo_platform import HEF, VDevice
from core.inference_engine import letterbox_resize, _unletterbox_scrfd
from core.perception.hailo_postprocess import decode_scrfd
from core.hardware.camera import get_camera_controller

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("calibrate_tracking")

# Hailo SCRFD Modell
SCRFD_HEF = "/mnt/moloch-data/hailo/models/scrfd_10g.hef"

# Anzahl Messungen pro Phase
SAMPLES = 6
SAMPLE_INTERVAL = 0.5  # Sekunden zwischen Messungen


def setup_npu():
    """SCRFD auf Hailo-10H laden und konfigurieren."""
    print("[NPU] SCRFD laden...")
    hef = HEF(SCRFD_HEF)
    params = VDevice.create_params()
    vdevice = VDevice(params)
    infer_model = vdevice.create_infer_model(SCRFD_HEF)
    infer_model.input().set_format_type(0)  # UINT8
    out_names = [infer_model.output(i).name for i in range(infer_model.outputs_count)]
    ctx_mgr = infer_model.configure()
    configured = ctx_mgr.__enter__()
    output_buffers = {
        oname: np.empty(infer_model.output(oname).shape, dtype=np.float32)
        for oname in out_names
    }
    bindings = configured.create_bindings(output_buffers=output_buffers)
    print(f"[NPU] SCRFD geladen. {len(out_names)} Output-Layer.")
    return vdevice, configured, bindings, output_buffers, out_names, ctx_mgr


def setup_camera():
    """RTSP-Stream oeffnen und ONVIF-Verbindung herstellen."""
    cam = get_camera_controller()
    cam.connect()
    rtsp_url = cam.get_rtsp_url("main")
    print(f"[CAM] RTSP: {rtsp_url}")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("FEHLER: RTSP-Stream nicht erreichbar!")
        sys.exit(1)
    print("[CAM] RTSP-Stream offen. ONVIF verbunden.")
    return cam, cap


def detect_face(cap, configured, bindings, output_buffers, out_names):
    """Ein Frame grabben, SCRFD laufen lassen, groesstes Gesicht zurueckgeben.

    Returns: (center_x, center_y, confidence) oder None wenn kein Gesicht.
    Koordinaten normalisiert 0.0-1.0 relativ zum Frame.
    """
    ret, frame = cap.read()
    if not ret or frame is None:
        return None

    fh, fw = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    padded, scale, pad_x, pad_y, new_w, new_h = letterbox_resize(rgb, 640)
    input_data = np.expand_dims(padded, axis=0).astype(np.uint8)

    bindings.input().set_buffer(np.ascontiguousarray(input_data))
    configured.run([bindings], timeout=10000)
    outputs = {name: output_buffers[name].copy() for name in out_names}

    boxes, scores, landmarks = decode_scrfd(outputs, img_size=640,
                                             conf_thresh=0.4, iou_thresh=0.4)
    if len(boxes) == 0:
        return None

    # Letterbox-Korrektur
    boxes_c, _ = _unletterbox_scrfd(boxes, landmarks, pad_x, pad_y, new_w, new_h)

    # Groesstes Gesicht nehmen (nach Flaeche)
    areas = (boxes_c[:, 2] - boxes_c[:, 0]) * (boxes_c[:, 3] - boxes_c[:, 1])
    idx = np.argmax(areas)
    b = boxes_c[idx]
    cx = (b[0] + b[2]) / 2.0
    cy = (b[1] + b[3]) / 2.0
    return cx, cy, float(scores[idx])


def measure_phase(cap, cam, configured, bindings, output_buffers, out_names,
                  label=""):
    """Mehrere Messungen nehmen, Mittelwert von center_x und Pan zurueckgeben."""
    samples_cx = []
    samples_pan = []

    for i in range(SAMPLES):
        result = detect_face(cap, configured, bindings, output_buffers, out_names)
        pos = cam.get_position()

        if result is not None and pos is not None:
            cx, cy, conf = result
            samples_cx.append(cx)
            samples_pan.append(pos.pan)
            print(f"  [{i+1}/{SAMPLES}] center_x={cx:.3f}  pan={pos.pan:+.1f}°  conf={conf:.2f}")
        else:
            status = "kein Gesicht" if result is None else "kein PTZ"
            print(f"  [{i+1}/{SAMPLES}] {status} — uebersprungen")

        time.sleep(SAMPLE_INTERVAL)

    if len(samples_cx) < 2:
        print(f"  WARNUNG: Nur {len(samples_cx)} gueltige Messungen fuer '{label}'!")
        return None, None

    avg_cx = sum(samples_cx) / len(samples_cx)
    avg_pan = sum(samples_pan) / len(samples_pan)
    print(f"  => {label}: center_x={avg_cx:.3f}  pan={avg_pan:+.1f}°  ({len(samples_cx)} Samples)")
    return avg_cx, avg_pan


def main():
    print("=" * 60)
    print("  M.O.L.O.C.H. PTZ-Tracking Kalibrierung")
    print("=" * 60)
    print()

    # Setup
    vdevice, configured, bindings, output_buffers, out_names, ctx_mgr = setup_npu()
    cam, cap = setup_camera()

    try:
        # === Phase 1: MITTE ===
        print()
        print("─" * 40)
        input("Stell dich in die MITTE vor die Kamera. [Enter wenn bereit] ")
        print("Messe Baseline (Mitte)...")
        cx_mitte, pan_mitte = measure_phase(
            cap, cam, configured, bindings, output_buffers, out_names, "MITTE")
        if cx_mitte is None:
            print("ABBRUCH: Konnte Baseline nicht messen!")
            return

        # === Phase 2: LINKS ===
        print()
        print("─" * 40)
        input("Geh langsam NACH LINKS (aus Kamera-Sicht). [Enter wenn am Rand] ")
        print("Messe Position (Links)...")
        cx_links, pan_links = measure_phase(
            cap, cam, configured, bindings, output_buffers, out_names, "LINKS")
        if cx_links is None:
            print("ABBRUCH: Konnte Links-Position nicht messen!")
            return

        # === Phase 3: Zurueck zur Mitte ===
        print()
        print("─" * 40)
        input("Geh zurueck zur MITTE. [Enter wenn bereit] ")

        # === Phase 4: RECHTS ===
        print()
        print("─" * 40)
        input("Geh langsam NACH RECHTS (aus Kamera-Sicht). [Enter wenn am Rand] ")
        print("Messe Position (Rechts)...")
        cx_rechts, pan_rechts = measure_phase(
            cap, cam, configured, bindings, output_buffers, out_names, "RECHTS")
        if cx_rechts is None:
            print("ABBRUCH: Konnte Rechts-Position nicht messen!")
            return

        # === Analyse ===
        print()
        print("=" * 60)
        print("  ANALYSE")
        print("=" * 60)
        print()
        print(f"  MITTE:   center_x={cx_mitte:.3f}   pan={pan_mitte:+.1f}°")
        print(f"  LINKS:   center_x={cx_links:.3f}   pan={pan_links:+.1f}°")
        print(f"  RECHTS:  center_x={cx_rechts:.3f}   pan={pan_rechts:+.1f}°")
        print()

        # Wenn Person nach links geht: center_x SINKT (0.5 → ~0.2)
        # Kamera sollte nach links folgen: Pan SINKT (wenn korrekt)
        # Wenn Pan STEIGT stattdessen → invertiert

        dx_links = cx_links - cx_mitte    # sollte negativ sein (links = kleiner x)
        dpan_links = pan_links - pan_mitte

        dx_rechts = cx_rechts - cx_mitte  # sollte positiv sein (rechts = groesser x)
        dpan_rechts = pan_rechts - pan_mitte

        print(f"  Links-Delta:  center_x={dx_links:+.3f}  pan={dpan_links:+.1f}°")
        print(f"  Rechts-Delta: center_x={dx_rechts:+.3f}  pan={dpan_rechts:+.1f}°")
        print()

        # Vorzeichen-Check: center_x und pan sollten gleiche Richtung haben
        # (Person geht links → center_x sinkt → Kamera folgt → Pan sinkt)
        pan_ok_links = (dx_links * dpan_links > 0) if abs(dx_links) > 0.05 else None
        pan_ok_rechts = (dx_rechts * dpan_rechts > 0) if abs(dx_rechts) > 0.05 else None

        # Tilt (Y-Achse) nicht interaktiv getestet, nur Pan
        pan_invertiert = False

        if pan_ok_links is not None and pan_ok_rechts is not None:
            pan_invertiert = not pan_ok_links or not pan_ok_rechts
        elif pan_ok_links is not None:
            pan_invertiert = not pan_ok_links
        elif pan_ok_rechts is not None:
            pan_invertiert = not pan_ok_rechts
        else:
            print("  WARNUNG: Zu wenig Bewegung gemessen — Ergebnis unklar!")

        print("─" * 40)
        status = "JA ⚠️" if pan_invertiert else "NEIN ✓"
        print(f"  PAN INVERTIERT: {status}")
        print("─" * 40)

        if pan_invertiert:
            print()
            print("  EMPFOHLENER FIX:")
            print("  In der pan_error / pan_delta Berechnung das")
            print("  Vorzeichen von error_x umdrehen:")
            print("    pan_delta = -error_x * ...  →  pan_delta = error_x * ...")
            print()

        # Ergebnisse speichern
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mitte": {"center_x": cx_mitte, "pan_deg": pan_mitte},
            "links": {"center_x": cx_links, "pan_deg": pan_links},
            "rechts": {"center_x": cx_rechts, "pan_deg": pan_rechts},
            "delta_links": {"center_x": dx_links, "pan_deg": dpan_links},
            "delta_rechts": {"center_x": dx_rechts, "pan_deg": dpan_rechts},
            "pan_invertiert": pan_invertiert,
        }

        out_path = os.path.expanduser("~/moloch/config/tracking_calibration.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Ergebnis gespeichert: {out_path}")

    finally:
        cap.release()
        try:
            ctx_mgr.__exit__(None, None, None)
        except Exception:
            pass
        print()
        print("Fertig.")


if __name__ == "__main__":
    main()
