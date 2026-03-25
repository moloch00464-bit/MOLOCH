#!/usr/bin/env python3
"""
RAM-Leak Diagnose Stufe 5: Modul-Isolation.

Importiert MolochService Module einzeln und misst RSS nach jedem Import.
Dann: init() und start() stufenweise mit RSS-Messung.

Identifiziert welches Modul den Speicher frisst.
"""

import os
import sys
import time
import gc
import ctypes
import threading

sys.path.insert(0, os.path.expanduser("~/moloch"))
os.environ["MOLOCH_USE_TAPPAS"] = "1"

LOG = "/tmp/ram_modules.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")

def rss():
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return 0.0

def threads():
    return threading.active_count()

def trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def measure(label):
    gc.collect()
    trim()
    r = rss()
    t = threads()
    log(f"  {label:<45} RSS: {r:7.0f} MB  Threads: {t:3d}")
    return r

if __name__ == "__main__":
    with open(LOG, "w") as f:
        f.write("")

    log(f"PID: {os.getpid()}")
    log("=" * 70)
    log("PHASE 1: Imports messen")
    log("=" * 70)

    r0 = measure("Baseline")

    # Schwergewicht-Imports einzeln
    imports = [
        ("hailo_platform", "import hailo_platform"),
        ("gi + Gst", "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst"),
        ("cv2", "import cv2"),
        ("numpy", "import numpy"),
        ("core.moloch_service (USE_TAPPAS=1)", "from core.moloch_service import MolochService"),
    ]
    for name, stmt in imports:
        try:
            exec(stmt)
            measure(f"nach import {name}")
        except Exception as e:
            log(f"  FEHLER bei {name}: {e}")

    log("")
    log("=" * 70)
    log("PHASE 2: MolochService.__init__()")
    log("=" * 70)

    r_pre = measure("vor __init__")
    from core.moloch_service import MolochService
    service = MolochService()
    r_init = measure("nach __init__")
    log(f"  __init__ Kosten: {r_init - r_pre:+.0f} MB")

    log("")
    log("=" * 70)
    log("PHASE 3: service.init() (Hardware)")
    log("=" * 70)

    r_pre = measure("vor init()")
    service.init()
    r_hw = measure("nach init()")
    log(f"  init() Kosten: {r_hw - r_pre:+.0f} MB")

    log("")
    log("=" * 70)
    log("PHASE 4: service.start() + 45s Monitor")
    log("=" * 70)

    r_pre = measure("vor start()")
    service.start(blocking=False)  # NICHT blockieren!
    time.sleep(2)
    r_start = measure("nach start() + 2s")
    log(f"  start() Kosten: {r_start - r_pre:+.0f} MB")

    # Thread-Inventar
    log("")
    log(f"  --- Thread-Inventar ({threads()} Threads) ---")
    for t in sorted(threading.enumerate(), key=lambda t: t.name):
        log(f"    {t.name:<40} daemon={t.daemon}")

    # Monitor 45 Sekunden
    log("")
    log("  --- RSS Monitor (alle 3s, 45s) ---")
    t_start = time.time()
    try:
        for i in range(15):
            time.sleep(3)
            gc.collect()
            trim()
            r = rss()
            elapsed = time.time() - t_start
            delta = r - r_start
            t_count = threads()
            log(f"    {elapsed:5.1f}s | RSS: {r:7.0f} MB | Δ: {delta:+7.0f} MB | Threads: {t_count}")
            if r > 2000:
                log(f"    *** NOTBREMSE bei {r:.0f} MB ***")
                break
    except KeyboardInterrupt:
        log("    CTRL-C")

    log("")
    log("=" * 70)
    log("PHASE 5: Cleanup")
    log("=" * 70)
    service.running = False
    try:
        service._inference.stop()
    except Exception:
        pass
    time.sleep(2)
    gc.collect()
    trim()
    measure("nach Stop")
    log("Fertig.")
