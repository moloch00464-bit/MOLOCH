#!/usr/bin/env python3
"""
RAM-Leak Diagnose Stufe 2: Voller MolochService.

Startet MolochService.init() + start() und misst RSS alle 2 Sekunden.
Tracemalloc mit 25 Frames Tiefe.
Notbremse bei 1.5 GB (Service + Pipeline).

Usage:
  source ~/.profile && python3 scripts/diagnose_ram_service.py
"""

import os
import sys
import time
import gc
import tracemalloc
import threading

sys.path.insert(0, os.path.expanduser("~/moloch"))

# TAPPAS erzwingen
os.environ["MOLOCH_USE_TAPPAS"] = "1"

# Tracemalloc VOR allen Imports
tracemalloc.start(25)

def get_rss_mb():
    """RSS in MB."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0

def snapshot_diff(snap_start, label=""):
    """Tracemalloc Diff zum Start-Snapshot."""
    snap = tracemalloc.take_snapshot()
    stats = snap.compare_to(snap_start, 'lineno')
    print(f"\n--- Top Memory-Wachstum {label} ---")
    for s in stats[:20]:
        if s.size_diff > 100_000:  # Nur >100 KB
            print(f"  {s}")
    return snap

if __name__ == "__main__":
    print("=" * 60)
    print(" M.O.L.O.C.H. Service RAM-Leak Diagnose")
    print(f" PID: {os.getpid()}")
    print("=" * 60)

    snap_start = tracemalloc.take_snapshot()
    rss_baseline = get_rss_mb()
    print(f"[BASELINE] RSS: {rss_baseline:.0f} MB (vor Import)")

    # Phase 1: Import
    print("\n[PHASE 1] MolochService importieren...")
    t0 = time.time()
    from core.moloch_service import MolochService
    rss_import = get_rss_mb()
    print(f"[IMPORT]  RSS: {rss_import:.0f} MB (+{rss_import - rss_baseline:.0f} MB, {time.time()-t0:.1f}s)")

    # Phase 2: __init__
    print("\n[PHASE 2] MolochService.__init__()...")
    t0 = time.time()
    service = MolochService()
    rss_init = get_rss_mb()
    print(f"[INIT]    RSS: {rss_init:.0f} MB (+{rss_init - rss_import:.0f} MB, {time.time()-t0:.1f}s)")

    snap_after_init = snapshot_diff(snap_start, "nach __init__")

    # Phase 3: init() — Hardware-Init (VDevice, RTSP, Cloud)
    print("\n[PHASE 3] service.init()...")
    t0 = time.time()
    service.init()
    rss_hw = get_rss_mb()
    print(f"[HW-INIT] RSS: {rss_hw:.0f} MB (+{rss_hw - rss_init:.0f} MB, {time.time()-t0:.1f}s)")

    snap_after_hw = snapshot_diff(snap_after_init, "nach init()")

    # Phase 4: start() — Pipeline + Threads
    print("\n[PHASE 4] service.start()...")
    t0 = time.time()
    service.start()
    rss_start = get_rss_mb()
    print(f"[START]   RSS: {rss_start:.0f} MB (+{rss_start - rss_hw:.0f} MB, {time.time()-t0:.1f}s)")

    # Phase 5: Monitor RSS alle 2 Sekunden
    print(f"\n[PHASE 5] RSS-Monitor (2s Intervall, Notbremse 1500 MB)...")
    MAX_RSS = 1500
    DURATION = 60
    INTERVAL = 2

    rss_prev = rss_start
    t_start = time.time()

    try:
        while time.time() - t_start < DURATION:
            time.sleep(INTERVAL)
            gc.collect()
            rss = get_rss_mb()
            elapsed = time.time() - t_start
            delta_total = rss - rss_start
            delta_step = rss - rss_prev
            rate = delta_total / elapsed if elapsed > 0 else 0

            # Pipeline-FPS
            fps = 0.0
            try:
                fps_data = service._inference.get_fps()
                fps = fps_data.get("current", 0) if isinstance(fps_data, dict) else fps_data
            except Exception:
                pass

            print(f"  [{elapsed:5.1f}s] RSS: {rss:7.0f} MB | Δtotal: {delta_total:+7.0f} MB | "
                  f"Δstep: {delta_step:+5.0f} MB | Rate: {rate:5.1f} MB/s | FPS: {fps:.1f}")

            rss_prev = rss

            if rss > MAX_RSS:
                print(f"\n  *** NOTBREMSE: RSS > {MAX_RSS} MB ***")
                snap_leak = snapshot_diff(snap_after_hw, "LEAK-Snapshot")

                # Zusaetzlich: nach Datei gruppieren
                snap = tracemalloc.take_snapshot()
                stats = snap.statistics('filename')
                print(f"\n--- Top Memory nach Datei ---")
                for s in stats[:20]:
                    if s.size > 100_000:
                        print(f"  {s}")

                # Thread-Liste
                print(f"\n--- Aktive Threads ({threading.active_count()}) ---")
                for t in threading.enumerate():
                    print(f"  {t.name} (daemon={t.daemon})")
                break
    except KeyboardInterrupt:
        print("\n[CTRL-C]")
    finally:
        print("\n[STOP] Service wird gestoppt...")
        service.running = False
        try:
            service._inference.stop()
        except Exception:
            pass
        time.sleep(2)
        gc.collect()
        rss_end = get_rss_mb()
        print(f"[NACH STOP] RSS: {rss_end:.0f} MB")

    print("\nFertig.")
