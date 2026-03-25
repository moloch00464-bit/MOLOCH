#!/usr/bin/env python3
"""
RAM-Leak Diagnose für TAPPAS Pipeline.

Testet 3 Szenarien stufenweise:
  1. NUR TappasPipeline (ohne Service) → Leak in GStreamer?
  2. TappasPipeline + appsink DEAKTIVIERT → Leak in Frame-Copy?
  3. TappasPipeline + _on_buffer DEAKTIVIERT → Leak in Detection-Extraction?

Misst RSS alle 5 Sekunden für 60s. Stoppt sofort bei >2 GB.
Ergebnis: Welche Komponente leakt.

Usage:
  MOLOCH_USE_TAPPAS=1 python3 scripts/diagnose_ram_leak.py
"""

import os
import sys
import time
import resource
import gc
import tracemalloc

sys.path.insert(0, os.path.expanduser("~/moloch"))

# Tracemalloc SOFORT starten (vor allen Imports)
tracemalloc.start(25)  # 25 Frames deep

def get_rss_mb():
    """RSS in MB (echtes RAM, kein Swap)."""
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024  # kB → MB
    return 0.0

def print_tracemalloc_top(snapshot, key_type='lineno', limit=15):
    """Top Memory-Allokationen nach Tracemalloc."""
    stats = snapshot.statistics(key_type)
    print(f"\n--- Top {limit} Memory-Allokationen ({key_type}) ---")
    for i, stat in enumerate(stats[:limit], 1):
        print(f"  #{i}: {stat}")
    total = sum(s.size for s in stats)
    print(f"  TOTAL (tracemalloc tracked): {total / 1024 / 1024:.1f} MB")

def run_test(test_name, duration=60, interval=5, max_mb=2000):
    """Generischer Test-Runner mit RSS-Monitoring."""
    print(f"\n{'='*60}")
    print(f" TEST: {test_name}")
    print(f"{'='*60}")

    # Baseline
    gc.collect()
    rss_start = get_rss_mb()
    snap_start = tracemalloc.take_snapshot()
    print(f"[START] RSS: {rss_start:.0f} MB")

    from core.perception.tappas_pipeline import TappasPipeline
    pipeline = TappasPipeline()

    rss_after_init = get_rss_mb()
    print(f"[INIT]  RSS: {rss_after_init:.0f} MB (Delta: +{rss_after_init - rss_start:.0f} MB)")

    # Pipeline starten
    pipeline.start()
    print(f"[PLAY]  Pipeline gestartet, messe {duration}s...")

    rss_values = []
    t_start = time.time()

    try:
        while time.time() - t_start < duration:
            time.sleep(interval)
            gc.collect()  # GC erzwingen
            rss = get_rss_mb()
            elapsed = time.time() - t_start
            fps = pipeline.get_fps().get("current", 0)
            rss_values.append((elapsed, rss))

            delta = rss - rss_start
            rate = delta / elapsed if elapsed > 0 else 0
            print(f"  [{elapsed:5.1f}s] RSS: {rss:7.0f} MB | Delta: +{delta:7.0f} MB | "
                  f"Rate: {rate:5.1f} MB/s | FPS: {fps:.1f}")

            # Notbremse
            if rss > max_mb:
                print(f"\n  *** NOTBREMSE: RSS > {max_mb} MB! ***")
                # Tracemalloc Snapshot VOR dem Stopp
                snap_leak = tracemalloc.take_snapshot()
                print_tracemalloc_top(snap_leak, 'lineno', 20)
                # Diff zum Start
                stats = snap_leak.compare_to(snap_start, 'lineno')
                print(f"\n--- Top Differences seit Start ---")
                for s in stats[:15]:
                    print(f"  {s}")
                break
    except KeyboardInterrupt:
        print("\n[CTRL-C]")
    finally:
        print("\n[STOP] Pipeline wird gestoppt...")
        pipeline.stop()
        time.sleep(1)
        gc.collect()
        rss_end = get_rss_mb()
        print(f"[NACH STOP] RSS: {rss_end:.0f} MB")

    # Endauswertung
    if len(rss_values) >= 2:
        _, rss_first = rss_values[0]
        _, rss_last = rss_values[-1]
        total_delta = rss_last - rss_first
        total_time = rss_values[-1][0] - rss_values[0][0]
        rate = total_delta / total_time if total_time > 0 else 0
        print(f"\n[ERGEBNIS] {test_name}")
        print(f"  Start RSS: {rss_first:.0f} MB")
        print(f"  End RSS:   {rss_last:.0f} MB")
        print(f"  Delta:     {total_delta:+.0f} MB in {total_time:.0f}s")
        print(f"  Rate:      {rate:.1f} MB/s")
        if rate > 5.0:
            print(f"  VERDICT:   *** LEAK GEFUNDEN ({rate:.1f} MB/s) ***")
        elif rate > 1.0:
            print(f"  VERDICT:   Verdaechtig ({rate:.1f} MB/s)")
        else:
            print(f"  VERDICT:   OK (< 1 MB/s)")

    # Final Tracemalloc
    snap_end = tracemalloc.take_snapshot()
    print_tracemalloc_top(snap_end, 'lineno', 20)
    stats = snap_end.compare_to(snap_start, 'lineno')
    print(f"\n--- Top Differences seit Programmstart ---")
    for s in stats[:15]:
        print(f"  {s}")

    return rss_values

if __name__ == "__main__":
    print("=" * 60)
    print(" M.O.L.O.C.H. RAM-Leak Diagnose")
    print(f" PID: {os.getpid()}")
    print(f" Pi5 RAM: 4 GB, Notbremse: 2 GB")
    print("=" * 60)

    # Sicherstellen dass TAPPAS-Flag gesetzt
    if os.environ.get("MOLOCH_USE_TAPPAS", "0") != "1":
        print("WARNUNG: MOLOCH_USE_TAPPAS nicht gesetzt, setze auf 1")
        os.environ["MOLOCH_USE_TAPPAS"] = "1"

    rss_values = run_test("TAPPAS Pipeline isoliert (ohne MolochService)",
                          duration=60, interval=5, max_mb=2000)

    print("\n\nFertig. Ergebnis oben auswerten.")
