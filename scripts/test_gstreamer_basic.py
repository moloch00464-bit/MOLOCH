#!/usr/bin/env python3
"""
Gate 0.5 — Phase 2.2: GStreamer RTSP Basis-Test

Testet ob GStreamer den RTSP-Stream der Sonoff CAM-PT2 lesen kann.
Kein NPU, kein Modell. Nur GStreamer + RTSP.

Nutzung:
    sudo systemctl stop moloch   # NPU freigeben
    python3 scripts/test_gstreamer_basic.py
"""

import time
import sys
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# RTSP-Quelle
RTSP_URL = "rtsp://Moloch_4.5:Auge666@192.168.178.25:554/av_stream/ch0"
TARGET_FRAMES = 10
TIMEOUT_SEC = 15

# Ergebnisse sammeln
results = {
    "frames": [],
    "width": 0,
    "height": 0,
    "format": "",
    "start_time": 0,
    "first_frame_time": 0,
}


def on_new_sample(appsink):
    """Callback fuer jeden neuen Frame vom appsink."""
    sample = appsink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.OK

    buf = sample.get_buffer()
    caps = sample.get_caps()
    struct = caps.get_structure(0)

    width = struct.get_value("width")
    height = struct.get_value("height")
    fmt = struct.get_value("format")

    now = time.time()
    frame_num = len(results["frames"]) + 1

    if frame_num == 1:
        results["first_frame_time"] = now
        results["width"] = width
        results["height"] = height
        results["format"] = fmt
        print(f"  Erster Frame empfangen!")
        print(f"  Aufloesung: {width}x{height}")
        print(f"  Format: {fmt}")
        print(f"  Latenz bis erster Frame: {now - results['start_time']:.2f}s")

    # Frame-Daten lesen (optional, beweist dass Buffer valide ist)
    success, mapinfo = buf.map(Gst.MapFlags.READ)
    if success:
        data = np.frombuffer(mapinfo.data, dtype=np.uint8)
        expected_size = width * height * 3  # RGB
        buf.unmap(mapinfo)

        results["frames"].append({
            "num": frame_num,
            "time": now,
            "size": len(data),
            "valid": len(data) == expected_size,
        })

        print(f"  Frame {frame_num:2d}/{TARGET_FRAMES}: {width}x{height} "
              f"size={len(data)} bytes valid={len(data) == expected_size}")
    else:
        print(f"  Frame {frame_num:2d}/{TARGET_FRAMES}: Buffer-Map FEHLGESCHLAGEN")
        results["frames"].append({
            "num": frame_num,
            "time": now,
            "size": 0,
            "valid": False,
        })

    if frame_num >= TARGET_FRAMES:
        GLib.idle_add(loop.quit)

    return Gst.FlowReturn.OK


def on_bus_message(bus, message, loop):
    """GStreamer Bus Messages verarbeiten."""
    t = message.type
    if t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"\n  FEHLER: {err}")
        print(f"  Debug: {debug}")
        loop.quit()
    elif t == Gst.MessageType.EOS:
        print("\n  End-of-Stream (unerwartet bei Live-Quelle)")
        loop.quit()
    elif t == Gst.MessageType.STATE_CHANGED:
        if message.src.get_name() == "pipeline":
            old, new, pending = message.parse_state_changed()
            if new == Gst.State.PLAYING:
                print(f"  Pipeline laeuft (PLAYING)")
    return True


# --- Main ---
print("=" * 60)
print("Gate 0.5 — Phase 2.2: GStreamer RTSP Basis-Test")
print("=" * 60)
print(f"\n  RTSP: {RTSP_URL.replace('Moloch_4.5:Auge666', '***:***')}")
print(f"  Ziel: {TARGET_FRAMES} Frames lesen")
print(f"  Timeout: {TIMEOUT_SEC}s")
print()

# GStreamer initialisieren
Gst.init(None)

# Pipeline bauen — angelehnt an TAPPAS SOURCE_PIPELINE fuer RTSP
pipeline_str = (
    f'rtspsrc location="{RTSP_URL}" name=source latency=300 ! '
    f'queue name=source_queue_decode leaky=downstream max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
    f'decodebin name=source_decodebin ! '
    f'queue name=convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
    f'videoconvert n-threads=2 ! '
    f'video/x-raw, format=RGB ! '
    f'appsink name=sink emit-signals=true drop=true max-buffers=3 sync=false'
)

print(f"  Pipeline:")
# Zensierte Ausgabe
print(f"  rtspsrc → decodebin → videoconvert → RGB → appsink")
print()

try:
    pipeline = Gst.parse_launch(pipeline_str)
except GLib.Error as e:
    print(f"  FEHLER beim Pipeline-Erstellen: {e}")
    sys.exit(1)

pipeline.set_name("pipeline")

# appsink Callback verbinden
appsink = pipeline.get_by_name("sink")
appsink.connect("new-sample", on_new_sample)

# Bus fuer Fehler/EOS
loop = GLib.MainLoop()
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", on_bus_message, loop)

# Timeout
def on_timeout():
    """Bricht ab wenn zu lange keine Frames kommen."""
    print(f"\n  TIMEOUT nach {TIMEOUT_SEC}s!")
    loop.quit()
    return False

GLib.timeout_add_seconds(TIMEOUT_SEC, on_timeout)

# Pipeline starten
print("  Starte Pipeline...")
results["start_time"] = time.time()
ret = pipeline.set_state(Gst.State.PLAYING)
if ret == Gst.StateChangeReturn.FAILURE:
    print("  FEHLER: Pipeline konnte nicht gestartet werden!")
    pipeline.set_state(Gst.State.NULL)
    sys.exit(1)

# Main Loop
try:
    loop.run()
except KeyboardInterrupt:
    print("\n  Abgebrochen (Ctrl+C)")

# Aufräumen
pipeline.set_state(Gst.State.NULL)

# --- Ergebnis ---
print()
print("=" * 60)
print("ERGEBNIS")
print("=" * 60)

total_frames = len(results["frames"])
valid_frames = sum(1 for f in results["frames"] if f["valid"])

if total_frames == 0:
    print("  FAIL: Keine Frames empfangen!")
    sys.exit(1)

elapsed = results["frames"][-1]["time"] - results["first_frame_time"]
fps = (total_frames - 1) / elapsed if elapsed > 0 and total_frames > 1 else 0

print(f"  Frames empfangen: {total_frames}/{TARGET_FRAMES}")
print(f"  Davon valide:     {valid_frames}/{total_frames}")
print(f"  Aufloesung:       {results['width']}x{results['height']}")
print(f"  Format:           {results['format']}")
print(f"  Latenz (1. Frame): {results['first_frame_time'] - results['start_time']:.2f}s")
print(f"  FPS (gemessen):   {fps:.1f}")
print(f"  Dauer gesamt:     {elapsed:.2f}s")
print()

if valid_frames == TARGET_FRAMES:
    print("  ✓ GStreamer RTSP funktioniert einwandfrei!")
    print("  → Bereit fuer Phase 2.3 (YOLO Inferenz)")
else:
    print(f"  ⚠ Nur {valid_frames}/{TARGET_FRAMES} valide Frames")
    print("  → Debugging noetig bevor weiter")

print()
