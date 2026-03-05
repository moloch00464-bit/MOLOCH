#!/usr/bin/env python3
"""
Gate 0.5 — Phase 2.4: GStreamer + SCRFD Face Detection auf Hailo-10H

Testet die TAPPAS-Pipeline mit SCRFD Face Detection auf dem RTSP-Stream.
Basiert auf der TAPPAS face_recognition_pipeline.py Referenz (nur Detection-Teil).

Pipeline:
  rtspsrc → INFERENCE_WRAPPER(letterbox → hailonet SCRFD → scrfd_10g_letterbox) → overlay → appsink

Speichert einen Frame mit BBoxen als Beweis-Bild.

Nutzung:
    sudo systemctl stop moloch
    python3 scripts/test_gstreamer_scrfd.py
"""

import os
import sys
import time
import threading

import numpy as np
from PIL import Image

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import hailo

# --- Konfiguration ---
RTSP_URL = "rtsp://Moloch_4.5:Auge666@192.168.178.25:554/av_stream/ch0"
HEF_PATH = "/mnt/moloch-data/hailo/models/scrfd_10g.hef"
POSTPROCESS_SO = "/usr/local/hailo/resources/so/libscrfd.so"
POSTPROCESS_FUNC = "scrfd_10g_letterbox"
CONFIG_JSON = "/usr/local/hailo/resources/json/scrfd.json"
WHOLE_BUFFER_SO = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so"

SNAPSHOT_PATH = os.path.expanduser("~/moloch/logs/tappas_face_test.jpg")
DURATION_SEC = 30
VDEVICE_GROUP_ID = "SHARED"

# --- Statistiken ---
stats = {
    "frame_count": 0,
    "face_detections": 0,
    "fps_samples": [],
    "start_time": 0,
    "last_fps_time": 0,
    "last_fps_count": 0,
    "ram_samples": [],
    "bbox_log": [],       # Erste 10 Detections mit Koordinaten loggen
    "snapshot_saved": False,
}
stats_lock = threading.Lock()


def get_ram_mb():
    """Aktueller RSS in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0


def save_snapshot_with_bboxes(frame_rgb, width, height, detections):
    """Speichert Frame mit eingezeichneten BBoxen als JPEG."""
    from PIL import ImageDraw, ImageFont
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)

    for det in detections:
        if det.get_label() != "face":
            continue
        bbox = det.get_bbox()
        # BBox ist relativ [0,1] — umrechnen auf Pixel
        x1 = int(bbox.xmin() * width)
        y1 = int(bbox.ymin() * height)
        x2 = int(bbox.xmax() * width)
        y2 = int(bbox.ymax() * height)
        conf = det.get_confidence()

        # Rotes Rechteck, 2px
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        # Label
        label = f"face {conf:.2f}"
        draw.text((x1, max(0, y1 - 12)), label, fill="red")

    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    img.save(SNAPSHOT_PATH, quality=90)
    print(f"  SNAPSHOT gespeichert: {SNAPSHOT_PATH}")
    print(f"    Aufloesung: {width}x{height}, Faces: {len([d for d in detections if d.get_label() == 'face'])}")


def build_pipeline_string():
    """
    Baut die GStreamer Pipeline analog zur TAPPAS face_recognition_pipeline.py:
    SOURCE → INFERENCE_WRAPPER(letterbox → SCRFD) → identity_callback → overlay → appsink
    """
    # Source: RTSP → H264 depay → decode → scale → convert → RGB 1280x720
    # Explizit nur Video-Stream nehmen (decodebin hat Probleme mit Audio+Video von Sonoff)
    source = (
        f'rtspsrc location="{RTSP_URL}" name=source latency=300 protocols=tcp ! '
        f'rtph264depay name=source_depay ! '
        f'queue name=source_queue_decode leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! '
        f'avdec_h264 name=source_decode ! '
        f'queue name=source_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'videoscale name=source_videoscale n-threads=2 ! '
        f'queue name=source_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'videoconvert n-threads=3 name=source_convert qos=false ! '
        f'video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width=1280, height=720 '
    )

    # SCRFD Inferenz (inner pipeline)
    inference_inner = (
        f'queue name=inf_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'videoscale name=inf_videoscale n-threads=2 qos=false ! '
        f'queue name=inf_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'video/x-raw, pixel-aspect-ratio=1/1 ! '
        f'videoconvert name=inf_videoconvert n-threads=2 ! '
        f'queue name=inf_hailonet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'hailonet name=inf_hailonet hef-path={HEF_PATH} batch-size=1 '
        f'vdevice-group-id={VDEVICE_GROUP_ID} '
        f'force-writable=true ! '
        f'queue name=inf_hailofilter_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'hailofilter name=inf_hailofilter so-path={POSTPROCESS_SO} '
        f'function-name={POSTPROCESS_FUNC} config-path={CONFIG_JSON} qos=false ! '
        f'queue name=inf_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
    )

    # Wrapper: hailocropper(whole_buffer, letterbox) → inference → hailoaggregator
    wrapper = (
        f'queue name=wrapper_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'hailocropper name=wrapper_crop so-path={WHOLE_BUFFER_SO} function-name=create_crops '
        f'use-letterbox=true resize-method=inter-area internal-offset=true '
        f'hailoaggregator name=wrapper_agg '
        f'wrapper_crop. ! queue name=wrapper_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0 ! wrapper_agg.sink_0 '
        f'wrapper_crop. ! {inference_inner} ! wrapper_agg.sink_1 '
        f'wrapper_agg. ! queue name=wrapper_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
    )

    # Identity Callback → Overlay → appsink
    callback_and_sink = (
        f'queue name=cb_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'identity name=identity_callback ! '
        f'queue name=overlay_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'hailooverlay name=hailo_overlay ! '
        f'queue name=sink_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'videoconvert n-threads=2 qos=false ! '
        f'video/x-raw, format=RGB ! '
        f'appsink name=sink emit-signals=true drop=true max-buffers=2 sync=false'
    )

    return f'{source} ! {wrapper} ! {callback_and_sink}'


def get_frame_from_buffer(buffer, pad):
    """Extrahiert numpy Frame aus GStreamer Buffer."""
    caps = pad.get_current_caps()
    if caps is None:
        return None, 0, 0
    struct = caps.get_structure(0)
    width = struct.get_value("width")
    height = struct.get_value("height")

    success, mapinfo = buffer.map(Gst.MapFlags.READ)
    if not success:
        return None, width, height
    frame = np.frombuffer(mapinfo.data, dtype=np.uint8).copy()
    buffer.unmap(mapinfo)

    expected = width * height * 3
    if len(frame) != expected:
        return None, width, height
    return frame.reshape(height, width, 3), width, height


def detection_callback(pad, info, user_data):
    """Pad-Probe auf identity element — zaehlt Face-Detections und speichert Snapshot."""
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    faces = 0
    for det in detections:
        if det.get_label() == "face":
            faces += 1

    now = time.time()
    with stats_lock:
        stats["frame_count"] += 1
        stats["face_detections"] += faces

        # BBox-Koordinaten der ersten Detections loggen
        if len(stats["bbox_log"]) < 10 and faces > 0:
            for det in detections:
                if det.get_label() != "face":
                    continue
                bbox = det.get_bbox()
                conf = det.get_confidence()
                stats["bbox_log"].append({
                    "frame": stats["frame_count"],
                    "xmin": bbox.xmin(),
                    "ymin": bbox.ymin(),
                    "xmax": bbox.xmax(),
                    "ymax": bbox.ymax(),
                    "conf": conf,
                })
                if len(stats["bbox_log"]) >= 10:
                    break

        # Snapshot: einmal speichern wenn Gesicht erkannt (nach Warmup)
        if not stats["snapshot_saved"] and faces > 0 and stats["frame_count"] > 30:
            frame_rgb, width, height = get_frame_from_buffer(buffer, pad)
            if frame_rgb is not None:
                save_snapshot_with_bboxes(frame_rgb, width, height, detections)
                stats["snapshot_saved"] = True

        # FPS alle 2 Sekunden
        elapsed_since_fps = now - stats["last_fps_time"]
        if elapsed_since_fps >= 2.0:
            frames_since = stats["frame_count"] - stats["last_fps_count"]
            fps = frames_since / elapsed_since_fps
            stats["fps_samples"].append(fps)
            stats["last_fps_time"] = now
            stats["last_fps_count"] = stats["frame_count"]

            ram = get_ram_mb()
            stats["ram_samples"].append(ram)

            elapsed_total = now - stats["start_time"]
            print(f"  [{elapsed_total:5.1f}s] FPS: {fps:5.1f} | Frames: {stats['frame_count']:4d} | "
                  f"Faces: {faces} (total: {stats['face_detections']}) | RAM: {ram:.0f} MB")

    return Gst.PadProbeReturn.OK


def on_appsink_sample(appsink):
    """appsink braucht diesen Callback damit Frames abgeholt werden."""
    appsink.emit("pull-sample")
    return Gst.FlowReturn.OK


def on_bus_message(bus, message, loop):
    """GStreamer Bus Messages."""
    t = message.type
    if t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"\n  FEHLER: {err}")
        print(f"  Debug: {debug}")
        loop.quit()
    elif t == Gst.MessageType.EOS:
        print("\n  End-of-Stream")
        loop.quit()
    return True


# --- Main ---
print("=" * 65)
print("Gate 0.5 — Phase 2.4: GStreamer + SCRFD Face Detection")
print("=" * 65)

# Voraussetzungen pruefen
for path, name in [
    (HEF_PATH, "SCRFD HEF"),
    (POSTPROCESS_SO, "SCRFD Postprocess SO"),
    (CONFIG_JSON, "SCRFD Config JSON"),
    (WHOLE_BUFFER_SO, "Whole Buffer SO"),
]:
    if not os.path.exists(path):
        print(f"  FEHLER: {name} nicht gefunden: {path}")
        sys.exit(1)

print(f"\n  RTSP:      {RTSP_URL.replace('Moloch_4.5:Auge666', '***:***')}")
print(f"  HEF:       {os.path.basename(HEF_PATH)} (640x640, Multi-Scale)")
print(f"  Postproc:  {os.path.basename(POSTPROCESS_SO)} → {POSTPROCESS_FUNC}")
print(f"  Config:    {os.path.basename(CONFIG_JSON)}")
print(f"  Letterbox: ja (hailocropper + whole_buffer + internal-offset)")
print(f"  Snapshot:  {SNAPSHOT_PATH}")
print(f"  Dauer:     {DURATION_SEC}s")
print()

# GStreamer initialisieren
Gst.init(None)

pipeline_str = build_pipeline_string()
print("  Pipeline: rtspsrc → WRAPPER(letterbox→SCRFD→scrfd_10g_letterbox) → overlay → appsink")
print()

try:
    pipeline = Gst.parse_launch(pipeline_str)
except GLib.Error as e:
    print(f"  FEHLER beim Pipeline-Erstellen: {e}")
    sys.exit(1)

# Identity Callback (Pad-Probe fuer Detection-Zaehlung + Snapshot)
identity = pipeline.get_by_name("identity_callback")
if identity is None:
    print("  FEHLER: identity_callback nicht gefunden!")
    sys.exit(1)
identity_pad = identity.get_static_pad("src")
identity_pad.add_probe(Gst.PadProbeType.BUFFER, detection_callback, None)

# appsink Callback damit Frames verbraucht werden
appsink = pipeline.get_by_name("sink")
appsink.connect("new-sample", on_appsink_sample)

# Bus
loop = GLib.MainLoop()
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", on_bus_message, loop)

# Timer
def on_timeout():
    print(f"\n  {DURATION_SEC}s abgelaufen — stoppe Pipeline...")
    loop.quit()
    return False

GLib.timeout_add_seconds(DURATION_SEC, on_timeout)

# Starten
print("  Starte Pipeline (NPU Warmup kann 2-3s dauern)...")
stats["start_time"] = time.time()
stats["last_fps_time"] = stats["start_time"]
stats["last_fps_count"] = 0

ret = pipeline.set_state(Gst.State.PLAYING)
if ret == Gst.StateChangeReturn.FAILURE:
    print("  FEHLER: Pipeline konnte nicht gestartet werden!")
    pipeline.set_state(Gst.State.NULL)
    sys.exit(1)

print()

try:
    loop.run()
except KeyboardInterrupt:
    print("\n  Abgebrochen (Ctrl+C)")

pipeline.set_state(Gst.State.NULL)
time.sleep(0.5)

# --- Ergebnis ---
total_time = time.time() - stats["start_time"]

print()
print("=" * 65)
print("ERGEBNIS")
print("=" * 65)

if stats["frame_count"] == 0:
    print("  FAIL: Keine Frames verarbeitet!")
    sys.exit(1)

avg_fps = stats["frame_count"] / total_time if total_time > 0 else 0
min_fps = min(stats["fps_samples"]) if stats["fps_samples"] else 0
max_fps = max(stats["fps_samples"]) if stats["fps_samples"] else 0
avg_ram = sum(stats["ram_samples"]) / len(stats["ram_samples"]) if stats["ram_samples"] else 0
max_ram = max(stats["ram_samples"]) if stats["ram_samples"] else 0

print(f"  Laufzeit:          {total_time:.1f}s")
print(f"  Frames gesamt:     {stats['frame_count']}")
print(f"  FPS (avg):         {avg_fps:.1f}")
print(f"  FPS (min/max):     {min_fps:.1f} / {max_fps:.1f}")
print(f"  Face-Detections:   {stats['face_detections']}")
print(f"  RAM avg/max:       {avg_ram:.0f} / {max_ram:.0f} MB")
print(f"  Snapshot:          {'gespeichert' if stats['snapshot_saved'] else 'NICHT gespeichert (kein Gesicht?)'}")
print()

# BBox-Koordinaten ausgeben
if stats["bbox_log"]:
    print("  BBox-Koordinaten (relativ 0.0-1.0, erste Detections):")
    print(f"  {'Frame':>5s}  {'xmin':>6s}  {'ymin':>6s}  {'xmax':>6s}  {'ymax':>6s}  {'conf':>5s}  {'Pixel (1280x720)':>20s}")
    print(f"  {'-'*5:>5s}  {'-'*6:>6s}  {'-'*6:>6s}  {'-'*6:>6s}  {'-'*6:>6s}  {'-'*5:>5s}  {'-'*20:>20s}")
    for b in stats["bbox_log"]:
        px = f"{int(b['xmin']*1280)},{int(b['ymin']*720)}-{int(b['xmax']*1280)},{int(b['ymax']*720)}"
        print(f"  {b['frame']:5d}  {b['xmin']:6.3f}  {b['ymin']:6.3f}  {b['xmax']:6.3f}  {b['ymax']:6.3f}  {b['conf']:5.2f}  {px:>20s}")
    print()

# Bewertung
issues = []
if avg_fps < 15:
    issues.append(f"FPS zu niedrig ({avg_fps:.1f} < 15)")
if max_ram > 3500:
    issues.append(f"RAM zu hoch ({max_ram:.0f} > 3500 MB)")
if not stats["snapshot_saved"]:
    issues.append("Kein Snapshot gespeichert — steh vor die Kamera!")

if not issues:
    print("  OK — SCRFD Face Detection laeuft stabil auf NPU via TAPPAS!")
    print("  → Snapshot pruefen: BBox muss EXAKT auf dem Gesicht sitzen")
    print(f"  → Bild: {SNAPSHOT_PATH}")
    print("  → Bereit fuer Phase 2.5 (Multi-Modell: SCRFD + ArcFace)")
else:
    print("  PROBLEME:")
    for issue in issues:
        print(f"    - {issue}")

print()
