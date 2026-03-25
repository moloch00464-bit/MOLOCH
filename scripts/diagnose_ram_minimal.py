#!/usr/bin/env python3
"""
RAM-Leak Diagnose Stufe 4: Minimaler Service-Simulator.

Testet Pipeline + Consumer-Threads stufenweise:
  A) Pipeline allein (Baseline — KEIN Leak erwartet)
  B) Pipeline + Tracker-Feed-Loop (15 Hz get_detections)
  C) Pipeline + Perception-Loop (5 Hz get_pframe + get_annotated_frame)
  D) Pipeline + Beide Loops zusammen

Identifiziert welcher Consumer-Loop den Leak triggert.
"""

import os
import sys
import time
import gc
import threading
import ctypes

sys.path.insert(0, os.path.expanduser("~/moloch"))
os.environ["MOLOCH_USE_TAPPAS"] = "1"

LOG = "/tmp/ram_minimal.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

def rss():
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return 0.0

def malloc_trim():
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

def tracker_feed_loop(pipeline, stop_event):
    """Simuliert _tappas_tracker_feed_loop: 15 Hz get_detections."""
    while not stop_event.is_set():
        if not pipeline.is_running():
            time.sleep(1)
            continue
        dets = pipeline.get_detections()
        # Simuliert Tracker-Verarbeitung
        for d in dets:
            _ = d.get("bbox", [0,0,0,0])
            _ = d.get("class", "")
        time.sleep(0.066)

def perception_loop(pipeline, stop_event):
    """Simuliert _tappas_perception_loop: 5 Hz get_pframe + get_annotated_frame."""
    while not stop_event.is_set():
        if not pipeline.is_running():
            time.sleep(1)
            continue
        pframe = pipeline.get_current_pframe()
        if pframe is None:
            time.sleep(0.2)
            continue
        # Simuliert Perception-Verarbeitung
        _ = getattr(pframe, 'face_detected', False)
        _ = getattr(pframe, 'person_detected', False)
        # get_annotated_frame (wie Teachen)
        frame = pipeline.get_annotated_frame()
        if frame is not None:
            _ = frame.shape  # Zugriff simulieren
        # get_detections (wie ReID + Episodic)
        dets = pipeline.get_detections()
        for d in dets:
            emb = d.get("embedding")
            if emb is not None:
                _ = len(emb)  # Embedding-Zugriff simulieren
        time.sleep(0.2)

def run_test(name, duration, use_tracker=False, use_perception=False):
    """Einzeltest: Pipeline + optionale Consumer-Loops."""
    from core.perception.tappas_pipeline import TappasPipeline

    log(f"\n{'='*50}")
    log(f"TEST: {name}")
    log(f"  tracker_feed={use_tracker}, perception_loop={use_perception}")
    log(f"  Dauer: {duration}s")
    log(f"{'='*50}")

    gc.collect()
    malloc_trim()
    rss_before = rss()
    log(f"Vor Pipeline-Init: {rss_before:.0f} MB")

    pipeline = TappasPipeline()
    pipeline.start()
    time.sleep(2)  # Pipeline warmlaufen lassen
    rss_start = rss()
    log(f"Pipeline laeuft: {rss_start:.0f} MB")

    stop = threading.Event()
    threads = []

    if use_tracker:
        t = threading.Thread(target=tracker_feed_loop, args=(pipeline, stop), daemon=True)
        t.start()
        threads.append(t)

    if use_perception:
        t = threading.Thread(target=perception_loop, args=(pipeline, stop), daemon=True)
        t.start()
        threads.append(t)

    # Monitor
    t_start = time.time()
    try:
        while time.time() - t_start < duration:
            time.sleep(3)
            gc.collect()
            malloc_trim()
            r = rss()
            elapsed = time.time() - t_start
            delta = r - rss_start
            fps = pipeline.get_fps().get("current", 0)
            thr = threading.active_count()
            log(f"  {elapsed:5.1f}s | RSS: {r:7.0f} MB | Δ: {delta:+6.0f} MB | "
                f"FPS: {fps:4.1f} | Threads: {thr}")

            if r > 1200:
                log("*** NOTBREMSE ***")
                break
    except KeyboardInterrupt:
        log("CTRL-C")

    stop.set()
    for t in threads:
        t.join(timeout=2)

    pipeline.stop()
    time.sleep(1)
    gc.collect()
    malloc_trim()
    rss_end = rss()
    delta_total = rss_end - rss_start
    log(f"ERGEBNIS {name}: Start={rss_start:.0f} MB, End={rss_end:.0f} MB, Δ={delta_total:+.0f} MB")
    return delta_total

if __name__ == "__main__":
    with open(LOG, "w") as f:
        f.write("")

    log(f"PID: {os.getpid()}")
    log(f"MALLOC_ARENA_MAX: {os.environ.get('MALLOC_ARENA_MAX', 'nicht gesetzt')}")

    DURATION = 30  # 30s pro Test

    # Test A: Pipeline allein (Baseline)
    delta_a = run_test("A: Pipeline allein", DURATION,
                       use_tracker=False, use_perception=False)

    time.sleep(3)
    gc.collect()

    # Test B: Pipeline + Tracker Feed
    delta_b = run_test("B: Pipeline + Tracker-Feed", DURATION,
                       use_tracker=True, use_perception=False)

    time.sleep(3)
    gc.collect()

    # Test C: Pipeline + Perception Loop
    delta_c = run_test("C: Pipeline + Perception-Loop", DURATION,
                       use_tracker=False, use_perception=True)

    time.sleep(3)
    gc.collect()

    # Test D: Pipeline + Beide
    delta_d = run_test("D: Pipeline + Tracker + Perception", DURATION,
                       use_tracker=True, use_perception=True)

    # Zusammenfassung
    log(f"\n{'='*50}")
    log("ZUSAMMENFASSUNG")
    log(f"{'='*50}")
    log(f"  A) Pipeline allein:              Δ {delta_a:+.0f} MB")
    log(f"  B) + Tracker-Feed (15Hz):        Δ {delta_b:+.0f} MB")
    log(f"  C) + Perception-Loop (5Hz):      Δ {delta_c:+.0f} MB")
    log(f"  D) + Tracker + Perception:       Δ {delta_d:+.0f} MB")
    log(f"\nWenn B oder C deutlich > A: Consumer-Loop ist Ursache.")
    log(f"Wenn D ~ A: Leak nur im vollen Service (Module-Interaktion).")
