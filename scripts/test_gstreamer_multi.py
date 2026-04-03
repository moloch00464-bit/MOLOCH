#!/usr/bin/env python3
"""
Gate 0.5 — Phase 2.5: GStreamer Multi-Model Pipeline auf Hailo-10H

Komplette Pipeline analog zur TAPPAS face_recognition_pipeline.py Referenz:
  rtspsrc → YOLO (Person Detection) → SCRFD (Face Detection) → Tracker → ArcFace (Face Recognition) → appsink

Alle Modelle teilen sich den NPU via vdevice-group-id=SHARED (Model Scheduler).
30 Sekunden laufen. Loggt: FPS, Person+Face Detections, Embeddings, RAM.
Speichert ein Bild mit allen BBoxen (Persons gruen, Faces rot).

Nutzung:
    sudo systemctl stop moloch
    python3 scripts/test_gstreamer_multi.py
"""

import os
import sys
import time
import threading

import numpy as np
from PIL import Image, ImageDraw

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import hailo

# --- Konfiguration ---
RTSP_URL = "rtsp://Moloch_4.5:Auge666@192.168.178.25:554/av_stream/ch0"

# Modelle (alle H10-nativ, vdevice-group-id=SHARED fuer Model Scheduler)
YOLO_HEF = "/mnt/moloch-data/hailo/models/yolov8m_h10.hef"
SCRFD_HEF = "/mnt/moloch-data/hailo/models/scrfd_10g.hef"
ARCFACE_HEF = "/mnt/moloch-data/hailo/models/arcface_mobilefacenet.hef"

# Postprocess SOs
YOLO_POSTPROCESS_SO = "/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so"
YOLO_POSTPROCESS_FUNC = "filter_letterbox"
SCRFD_POSTPROCESS_SO = "/usr/local/hailo/resources/so/libscrfd.so"
SCRFD_POSTPROCESS_FUNC = "scrfd_10g_letterbox"
SCRFD_CONFIG_JSON = "/usr/local/hailo/resources/json/scrfd.json"
ARCFACE_POSTPROCESS_SO = "/usr/local/hailo/resources/so/libface_recognition_post.so"
ARCFACE_POSTPROCESS_FUNC = "filter"
FACE_ALIGN_SO = "/usr/local/hailo/resources/so/libvms_face_align.so"
FACE_CROP_SO = "/usr/local/hailo/resources/so/libvms_croppers.so"
FACE_CROP_FUNC = "face_recognition"
WHOLE_BUFFER_SO = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so"

SNAPSHOT_PATH = os.path.expanduser("~/moloch/logs/tappas_multi_test.jpg")
DURATION_SEC = 30
VDEVICE_GROUP_ID = "SHARED"

# --- Statistiken ---
stats = {
    "frame_count": 0,
    "person_detections": 0,
    "face_detections": 0,
    "embeddings_count": 0,
    "fps_samples": [],
    "start_time": 0,
    "last_fps_time": 0,
    "last_fps_count": 0,
    "ram_samples": [],
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
    """Speichert Frame mit Person-BBoxen (gruen) und Face-BBoxen (rot)."""
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)

    for det in detections:
        bbox = det.get_bbox()
        x1 = int(bbox.xmin() * width)
        y1 = int(bbox.ymin() * height)
        x2 = int(bbox.xmax() * width)
        y2 = int(bbox.ymax() * height)
        conf = det.get_confidence()
        label = det.get_label()

        if label == "person":
            color = "lime"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, max(0, y1 - 12)), f"person {conf:.2f}", fill=color)
        elif label == "face":
            color = "red"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            # ArcFace Embedding vorhanden?
            embeddings = det.get_objects_typed(hailo.HAILO_MATRIX)
            classifications = det.get_objects_typed(hailo.HAILO_CLASSIFICATION)
            tag = f"face {conf:.2f}"
            if embeddings:
                tag += f" emb:{len(embeddings[0].get_data())}"
            if classifications:
                tag += f" [{classifications[0].get_label()}]"
            draw.text((x1, max(0, y1 - 12)), tag, fill=color)

    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    img.save(SNAPSHOT_PATH, quality=90)

    persons = len([d for d in detections if d.get_label() == "person"])
    faces = len([d for d in detections if d.get_label() == "face"])
    print(f"  SNAPSHOT gespeichert: {SNAPSHOT_PATH}")
    print(f"    Aufloesung: {width}x{height}, Persons: {persons}, Faces: {faces}")


def build_pipeline_string():
    """
    Baut die komplette Multi-Model Pipeline analog zur TAPPAS face_recognition Referenz:

    SOURCE → YOLO_WRAPPER(letterbox→hailonet→yolo_postproc) → SCRFD_WRAPPER(letterbox→hailonet→scrfd_postproc)
           → TRACKER → FACE_CROPPER(face_align→hailonet_arcface→arcface_postproc) → identity → overlay → appsink

    Alle hailonet Elemente teilen sich vdevice-group-id=SHARED → Hailo Model Scheduler.
    """

    # --- Source: RTSP → H264 depay → decode → scale → RGB 1280x720 ---
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

    # --- Stage 1: YOLO Person Detection (inference_wrapper mit letterbox) ---
    yolo_inner = (
        f'queue name=yolo_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'videoscale name=yolo_videoscale n-threads=2 qos=false ! '
        f'queue name=yolo_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'video/x-raw, pixel-aspect-ratio=1/1 ! '
        f'videoconvert name=yolo_videoconvert n-threads=2 ! '
        f'queue name=yolo_hailonet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'hailonet name=yolo_hailonet hef-path={YOLO_HEF} batch-size=1 '
        f'vdevice-group-id={VDEVICE_GROUP_ID} '
        f'nms-score-threshold=0.3 nms-iou-threshold=0.45 '
        f'output-format-type=HAILO_FORMAT_TYPE_FLOAT32 '
        f'force-writable=true ! '
        f'queue name=yolo_hailofilter_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'hailofilter name=yolo_hailofilter so-path={YOLO_POSTPROCESS_SO} '
        f'function-name={YOLO_POSTPROCESS_FUNC} qos=false ! '
        f'queue name=yolo_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
    )

    yolo_wrapper = (
        f'queue name=yolo_wrapper_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'hailocropper name=yolo_wrapper_crop so-path={WHOLE_BUFFER_SO} function-name=create_crops '
        f'use-letterbox=true resize-method=inter-area internal-offset=true '
        f'hailoaggregator name=yolo_wrapper_agg '
        f'yolo_wrapper_crop. ! queue name=yolo_wrapper_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0 ! yolo_wrapper_agg.sink_0 '
        f'yolo_wrapper_crop. ! {yolo_inner} ! yolo_wrapper_agg.sink_1 '
        f'yolo_wrapper_agg. ! queue name=yolo_wrapper_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
    )

    # --- Stage 2: SCRFD Face Detection (inference_wrapper mit letterbox) ---
    scrfd_inner = (
        f'queue name=scrfd_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'videoscale name=scrfd_videoscale n-threads=2 qos=false ! '
        f'queue name=scrfd_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'video/x-raw, pixel-aspect-ratio=1/1 ! '
        f'videoconvert name=scrfd_videoconvert n-threads=2 ! '
        f'queue name=scrfd_hailonet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'hailonet name=scrfd_hailonet hef-path={SCRFD_HEF} batch-size=1 '
        f'vdevice-group-id={VDEVICE_GROUP_ID} '
        f'force-writable=true ! '
        f'queue name=scrfd_hailofilter_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'hailofilter name=scrfd_hailofilter so-path={SCRFD_POSTPROCESS_SO} '
        f'function-name={SCRFD_POSTPROCESS_FUNC} config-path={SCRFD_CONFIG_JSON} qos=false ! '
        f'queue name=scrfd_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
    )

    scrfd_wrapper = (
        f'queue name=scrfd_wrapper_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'hailocropper name=scrfd_wrapper_crop so-path={WHOLE_BUFFER_SO} function-name=create_crops '
        f'use-letterbox=true resize-method=inter-area internal-offset=true '
        f'hailoaggregator name=scrfd_wrapper_agg '
        f'scrfd_wrapper_crop. ! queue name=scrfd_wrapper_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0 ! scrfd_wrapper_agg.sink_0 '
        f'scrfd_wrapper_crop. ! {scrfd_inner} ! scrfd_wrapper_agg.sink_1 '
        f'scrfd_wrapper_agg. ! queue name=scrfd_wrapper_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
    )

    # --- Stage 3: Tracker → Face Cropper (face_align + ArcFace) ---
    # Tracker haelt Face-IDs persistent ueber Frames
    tracker = (
        f'hailotracker name=hailo_face_tracker class-id=-1 '
        f'kalman-dist-thr=0.7 iou-thr=0.8 init-iou-thr=0.9 '
        f'keep-new-frames=2 keep-tracked-frames=6 keep-lost-frames=8 '
        f'keep-past-metadata=true qos=false ! '
        f'queue name=tracker_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
    )

    # ArcFace inner pipeline: face_align → hailonet arcface → arcface_postprocess
    arcface_inner = (
        f'hailofilter so-path={FACE_ALIGN_SO} name=face_align_hailofilter use-gst-buffer=true qos=false ! '
        f'queue name=face_align_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'queue name=arcface_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'videoscale name=arcface_videoscale n-threads=2 qos=false ! '
        f'queue name=arcface_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'video/x-raw, pixel-aspect-ratio=1/1 ! '
        f'videoconvert name=arcface_videoconvert n-threads=2 ! '
        f'queue name=arcface_hailonet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'hailonet name=arcface_hailonet hef-path={ARCFACE_HEF} batch-size=1 '
        f'vdevice-group-id={VDEVICE_GROUP_ID} '
        f'force-writable=true ! '
        f'queue name=arcface_hailofilter_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'hailofilter name=arcface_hailofilter so-path={ARCFACE_POSTPROCESS_SO} '
        f'function-name={ARCFACE_POSTPROCESS_FUNC} qos=false ! '
        f'queue name=arcface_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
    )

    # Face Cropper: schneidet erkannte Faces aus und schickt sie durch ArcFace
    face_cropper = (
        f'queue name=face_crop_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
        f'hailocropper name=face_cropper so-path={FACE_CROP_SO} function-name={FACE_CROP_FUNC} '
        f'use-letterbox=true no-scaling-bbox=true internal-offset=true resize-method=bilinear '
        f'hailoaggregator name=face_crop_agg '
        f'face_cropper. ! queue name=face_crop_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0 ! face_crop_agg.sink_0 '
        f'face_cropper. ! {arcface_inner} ! face_crop_agg.sink_1 '
        f'face_crop_agg. ! queue name=face_crop_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
    )

    # --- Callback + Overlay + appsink ---
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

    return (
        f'{source} ! '
        f'{yolo_wrapper} ! '
        f'{scrfd_wrapper} ! '
        f'{tracker} ! '
        f'{face_cropper} ! '
        f'{callback_and_sink}'
    )


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
    """Pad-Probe auf identity element — zaehlt Persons, Faces, Embeddings und speichert Snapshot."""
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    persons = 0
    faces = 0
    embeddings = 0

    for det in detections:
        label = det.get_label()
        if label == "person":
            persons += 1
        elif label == "face":
            faces += 1
            # ArcFace Embedding pruefen
            emb = det.get_objects_typed(hailo.HAILO_MATRIX)
            if emb:
                embeddings += 1

    now = time.time()
    with stats_lock:
        stats["frame_count"] += 1
        stats["person_detections"] += persons
        stats["face_detections"] += faces
        stats["embeddings_count"] += embeddings

        # Snapshot: einmal speichern wenn mindestens 1 Person + 1 Face (nach Warmup)
        if not stats["snapshot_saved"] and persons > 0 and faces > 0 and stats["frame_count"] > 30:
            frame_rgb, width, height = get_frame_from_buffer(buffer, pad)
            if frame_rgb is not None:
                save_snapshot_with_bboxes(frame_rgb, width, height, detections)
                stats["snapshot_saved"] = True

        # Fallback Snapshot: nur Persons (ohne Face, z.B. Gesicht nicht sichtbar)
        if not stats["snapshot_saved"] and (persons > 0 or faces > 0) and stats["frame_count"] > 100:
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
                  f"P:{persons} F:{faces} E:{embeddings} "
                  f"(total P:{stats['person_detections']} F:{stats['face_detections']} E:{stats['embeddings_count']}) | "
                  f"RAM: {ram:.0f} MB")

    return Gst.PadProbeReturn.OK


def on_appsink_sample(appsink):
    """appsink Callback damit Frames abgeholt werden."""
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
print("=" * 70)
print("Gate 0.5 — Phase 2.5: GStreamer Multi-Model Pipeline (YOLO+SCRFD+ArcFace)")
print("=" * 70)

# Voraussetzungen pruefen
required_files = [
    (YOLO_HEF, "YOLO HEF"),
    (SCRFD_HEF, "SCRFD HEF"),
    (ARCFACE_HEF, "ArcFace HEF"),
    (YOLO_POSTPROCESS_SO, "YOLO Postprocess SO"),
    (SCRFD_POSTPROCESS_SO, "SCRFD Postprocess SO"),
    (SCRFD_CONFIG_JSON, "SCRFD Config JSON"),
    (ARCFACE_POSTPROCESS_SO, "ArcFace Postprocess SO"),
    (FACE_ALIGN_SO, "Face Align SO"),
    (FACE_CROP_SO, "Face Crop SO"),
    (WHOLE_BUFFER_SO, "Whole Buffer SO"),
]

missing = False
for path, name in required_files:
    if not os.path.exists(path):
        print(f"  FEHLER: {name} nicht gefunden: {path}")
        missing = True
if missing:
    sys.exit(1)

print(f"\n  RTSP:       {RTSP_URL.replace('Moloch_4.5:Auge666', '***:***')}")
print(f"  Modelle:")
print(f"    YOLO:     {os.path.basename(YOLO_HEF)} (Person Detection)")
print(f"    SCRFD:    {os.path.basename(SCRFD_HEF)} (Face Detection)")
print(f"    ArcFace:  {os.path.basename(ARCFACE_HEF)} (Face Recognition/Embedding)")
print(f"  Scheduler:  vdevice-group-id={VDEVICE_GROUP_ID}")
print(f"  Letterbox:  ja (beide Detection-Wrapper)")
print(f"  Tracker:    hailotracker (Face-IDs ueber Frames)")
print(f"  Snapshot:   {SNAPSHOT_PATH}")
print(f"  Dauer:      {DURATION_SEC}s")
print()

# GStreamer initialisieren
Gst.init(None)

pipeline_str = build_pipeline_string()
print("  Pipeline: rtspsrc → YOLO_WRAPPER → SCRFD_WRAPPER → Tracker → FACE_CROPPER(align+ArcFace) → overlay → appsink")
print()

try:
    pipeline = Gst.parse_launch(pipeline_str)
except GLib.Error as e:
    print(f"  FEHLER beim Pipeline-Erstellen: {e}")
    sys.exit(1)

# Identity Callback (Pad-Probe)
identity = pipeline.get_by_name("identity_callback")
if identity is None:
    print("  FEHLER: identity_callback nicht gefunden!")
    sys.exit(1)
identity_pad = identity.get_static_pad("src")
identity_pad.add_probe(Gst.PadProbeType.BUFFER, detection_callback, None)

# appsink Callback
appsink = pipeline.get_by_name("sink")
appsink.connect("new-sample", on_appsink_sample)

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
print("  Starte Pipeline (3 Modelle laden, NPU Warmup kann 3-5s dauern)...")
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
print("=" * 70)
print("ERGEBNIS — Multi-Model Pipeline (YOLO + SCRFD + ArcFace)")
print("=" * 70)

if stats["frame_count"] == 0:
    print("  FAIL: Keine Frames verarbeitet!")
    sys.exit(1)

avg_fps = stats["frame_count"] / total_time if total_time > 0 else 0
min_fps = min(stats["fps_samples"]) if stats["fps_samples"] else 0
max_fps = max(stats["fps_samples"]) if stats["fps_samples"] else 0
avg_ram = sum(stats["ram_samples"]) / len(stats["ram_samples"]) if stats["ram_samples"] else 0
max_ram = max(stats["ram_samples"]) if stats["ram_samples"] else 0

print(f"  Laufzeit:           {total_time:.1f}s")
print(f"  Frames gesamt:      {stats['frame_count']}")
print(f"  FPS (avg):          {avg_fps:.1f}")
print(f"  FPS (min/max):      {min_fps:.1f} / {max_fps:.1f}")
print(f"  Person-Detections:  {stats['person_detections']}")
print(f"  Face-Detections:    {stats['face_detections']}")
print(f"  ArcFace Embeddings: {stats['embeddings_count']}")
print(f"  RAM avg/max:        {avg_ram:.0f} / {max_ram:.0f} MB")
print(f"  Snapshot:           {'gespeichert' if stats['snapshot_saved'] else 'NICHT gespeichert (niemand vor Kamera?)'}")
print()

# Modell-Uebersicht
print("  Modelle auf NPU (Model Scheduler):")
print(f"    YOLO     {os.path.basename(YOLO_HEF):30s}  vdevice-group-id={VDEVICE_GROUP_ID}")
print(f"    SCRFD    {os.path.basename(SCRFD_HEF):30s}  vdevice-group-id={VDEVICE_GROUP_ID}")
print(f"    ArcFace  {os.path.basename(ARCFACE_HEF):30s}  vdevice-group-id={VDEVICE_GROUP_ID}")
print()

# Bewertung
issues = []
if avg_fps < 10:
    issues.append(f"FPS zu niedrig ({avg_fps:.1f} < 10) — 3 Modelle sind anspruchsvoll")
if max_ram > 3500:
    issues.append(f"RAM zu hoch ({max_ram:.0f} > 3500 MB)")
if stats["frame_count"] < DURATION_SEC * 5:
    issues.append(f"Zu wenige Frames ({stats['frame_count']} < {DURATION_SEC * 5})")
if stats["person_detections"] == 0 and stats["face_detections"] == 0:
    issues.append("Keine Detections — steht jemand vor der Kamera?")

if not issues:
    print("  OK — Multi-Model Pipeline laeuft stabil auf NPU via Model Scheduler!")
    if stats["embeddings_count"] > 0:
        print(f"  → ArcFace erzeugt Embeddings ({stats['embeddings_count']}x) — Face Recognition READY!")
    else:
        print("  → ArcFace ohne Embeddings — pruefen ob Gesichter sichtbar waren")
    if stats["snapshot_saved"]:
        print(f"  → Snapshot: {SNAPSHOT_PATH}")
    print("  → Bereit fuer Phase 3 (Integration in tappas_pipeline.py)")
else:
    print("  PROBLEME:")
    for issue in issues:
        print(f"    - {issue}")

print()
