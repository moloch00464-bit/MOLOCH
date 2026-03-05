#!/usr/bin/env python3
"""
Gate 0.5 — Phase 2.3: GStreamer + YOLOv8m auf Hailo-10H NPU

Testet die TAPPAS-Pipeline mit YOLOv8m Person-Detection auf dem RTSP-Stream.
Basiert auf der TAPPAS detection_pipeline.py Referenz.

Pipeline:
  rtspsrc → INFERENCE_WRAPPER(letterbox → hailonet yolov8m → postprocess) → appsink

Loggt: FPS, Person-Detections, RAM-Verbrauch ueber 30 Sekunden.

Nutzung:
    sudo systemctl stop moloch   # NPU freigeben
    python3 scripts/test_gstreamer_yolo.py
"""

import os
import sys
import time
import threading
import resource

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import hailo

# --- Konfiguration ---
RTSP_URL = "rtsp://Moloch_4.5:Auge666@192.168.178.25:554/av_stream/ch0"
HEF_PATH = "/mnt/moloch-data/hailo/models/yolov8m_h10.hef"
POSTPROCESS_SO = "/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so"
POSTPROCESS_FUNC = "filter_letterbox"
WHOLE_BUFFER_SO = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so"

DURATION_SEC = 30
VDEVICE_GROUP_ID = "SHARED"

# --- Statistiken ---
stats = {
    "frame_count": 0,
    "person_detections": 0,
    "total_detections": 0,
    "fps_samples": [],
    "start_time": 0,
    "last_fps_time": 0,
    "last_fps_count": 0,
    "ram_samples": [],
}
stats_lock = threading.Lock()


def get_ram_mb():
    """Aktueller RAM-Verbrauch des Prozesses in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except Exception:
        pass
    return 0


def build_pipeline_string():
    """
    Baut die GStreamer Pipeline analog zur TAPPAS detection_pipeline.py:
    SOURCE → INFERENCE_WRAPPER(letterbox → hailonet → postprocess) → identity_callback → appsink
    """
    # Source: RTSP → decode → scale → convert → RGB 1280x720
    # protocols=tcp verhindert UDP not-linked Fehler bei Audio+Video Streams
    source = (
        f'rtspsrc location="{RTSP_URL}" name=source latency=300 protocols=tcp ! '
        f'queue name=source_queue_decode leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! '
        f'decodebin name=source_decodebin ! '
        f'queue name=source_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'videoscale name=source_videoscale n-threads=2 ! '
        f'queue name=source_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'videoconvert n-threads=3 name=source_convert qos=false ! '
        f'video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width=1280, height=720 '
    )

    # Inferenz (inner pipeline): scale → convert → hailonet → hailofilter postprocess
    inference_inner = (
        f'queue name=inf_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'videoscale name=inf_videoscale n-threads=2 qos=false ! '
        f'queue name=inf_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'video/x-raw, pixel-aspect-ratio=1/1 ! '
        f'videoconvert name=inf_videoconvert n-threads=2 ! '
        f'queue name=inf_hailonet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'hailonet name=inf_hailonet hef-path={HEF_PATH} batch-size=1 '
        f'vdevice-group-id={VDEVICE_GROUP_ID} '
        f'nms-score-threshold=0.3 nms-iou-threshold=0.45 '
        f'output-format-type=HAILO_FORMAT_TYPE_FLOAT32 '
        f'force-writable=true ! '
        f'queue name=inf_hailofilter_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'hailofilter name=inf_hailofilter so-path={POSTPROCESS_SO} function-name={POSTPROCESS_FUNC} qos=false ! '
        f'queue name=inf_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
    )

    # Wrapper: hailocropper(whole_buffer, letterbox) → inference → hailoaggregator
    # Haelt Original-Aufloesung, skaliert intern auf 640x640 mit Letterbox
    wrapper = (
        f'queue name=wrapper_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'hailocropper name=wrapper_crop so-path={WHOLE_BUFFER_SO} function-name=create_crops '
        f'use-letterbox=true resize-method=inter-area internal-offset=true '
        f'hailoaggregator name=wrapper_agg '
        f'wrapper_crop. ! queue name=wrapper_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0 ! wrapper_agg.sink_0 '
        f'wrapper_crop. ! {inference_inner} ! wrapper_agg.sink_1 '
        f'wrapper_agg. ! queue name=wrapper_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
    )

    # Identity Callback (Python Pad-Probe) → Overlay → appsink
    callback_and_sink = (
        f'queue name=cb_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'identity name=identity_callback ! '
        f'queue name=overlay_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'hailooverlay name=hailo_overlay ! '
        f'queue name=sink_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'videoconvert n-threads=2 qos=false ! '
        f'video/x-raw, format=RGB ! '
        f'appsink name=sink emit-signals=false drop=true max-buffers=2 sync=false'
    )

    return f'{source} ! {wrapper} ! {callback_and_sink}'


def detection_callback(pad, info, user_data):
    """Pad-Probe Callback auf identity element — zaehlt Detections."""
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    persons = 0
    for det in detections:
        if det.get_label() == "person":
            persons += 1

    now = time.time()
    with stats_lock:
        stats["frame_count"] += 1
        stats["person_detections"] += persons
        stats["total_detections"] += len(detections)

        # FPS alle 2 Sekunden berechnen
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
                  f"Persons: {persons} (total: {stats['person_detections']}) | RAM: {ram:.0f} MB")

    return Gst.PadProbeReturn.OK


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
print("Gate 0.5 — Phase 2.3: GStreamer + YOLOv8m auf Hailo-10H")
print("=" * 65)

# Voraussetzungen pruefen
for path, name in [(HEF_PATH, "HEF"), (POSTPROCESS_SO, "Postprocess SO"), (WHOLE_BUFFER_SO, "Whole Buffer SO")]:
    if not os.path.exists(path):
        print(f"  FEHLER: {name} nicht gefunden: {path}")
        sys.exit(1)

print(f"\n  RTSP:     {RTSP_URL.replace('Moloch_4.5:Auge666', '***:***')}")
print(f"  HEF:      {os.path.basename(HEF_PATH)}")
print(f"  Postproc: {os.path.basename(POSTPROCESS_SO)} → {POSTPROCESS_FUNC}")
print(f"  Dauer:    {DURATION_SEC}s")
print(f"  Letterbox: ja (hailocropper + whole_buffer)")
print()

# GStreamer initialisieren
Gst.init(None)

pipeline_str = build_pipeline_string()

print("  Pipeline: rtspsrc → INFERENCE_WRAPPER(letterbox→hailonet→postproc) → overlay → appsink")
print()

try:
    pipeline = Gst.parse_launch(pipeline_str)
except GLib.Error as e:
    print(f"  FEHLER beim Pipeline-Erstellen: {e}")
    sys.exit(1)

# Identity Callback verbinden (Pad-Probe wie in TAPPAS Referenz)
identity = pipeline.get_by_name("identity_callback")
if identity is None:
    print("  FEHLER: identity_callback Element nicht gefunden!")
    sys.exit(1)

identity_pad = identity.get_static_pad("src")
identity_pad.add_probe(Gst.PadProbeType.BUFFER, detection_callback, None)

# Bus
loop = GLib.MainLoop()
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", on_bus_message, loop)

# Timer: nach DURATION_SEC stoppen
def on_timeout():
    print(f"\n  {DURATION_SEC}s abgelaufen — stoppe Pipeline...")
    loop.quit()
    return False

GLib.timeout_add_seconds(DURATION_SEC, on_timeout)

# Pipeline starten
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

# Main Loop
try:
    loop.run()
except KeyboardInterrupt:
    print("\n  Abgebrochen (Ctrl+C)")

# Aufraeumen
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
print(f"  Person-Detections: {stats['person_detections']}")
print(f"  Alle Detections:   {stats['total_detections']}")
print(f"  RAM avg/max:       {avg_ram:.0f} / {max_ram:.0f} MB")
print()

# Bewertung
issues = []
if avg_fps < 15:
    issues.append(f"FPS zu niedrig ({avg_fps:.1f} < 15)")
if max_ram > 3500:
    issues.append(f"RAM zu hoch ({max_ram:.0f} > 3500 MB)")
if stats["frame_count"] < DURATION_SEC * 10:
    issues.append(f"Zu wenige Frames ({stats['frame_count']} < {DURATION_SEC * 10})")

if not issues:
    print("  OK — YOLOv8m laeuft stabil auf NPU via TAPPAS!")
    print("  → Bereit fuer Phase 2.4 (SCRFD + ArcFace)")
else:
    print("  PROBLEME:")
    for issue in issues:
        print(f"    - {issue}")

print()
