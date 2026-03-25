#!/usr/bin/env python3
"""
RAM-Leak Diagnose Stufe 3: MALLOC_ARENA_MAX Test.

Testet ob glibc malloc-Arena-Explosion die Ursache ist.
Startet vollen Service mit MALLOC_ARENA_MAX=2.

Ergebnis wird in /tmp/ram_diagnosis.log geschrieben (NICHT stdout).
"""

import os
import sys
import time
import gc
import ctypes

sys.path.insert(0, os.path.expanduser("~/moloch"))
os.environ["MOLOCH_USE_TAPPAS"] = "1"

LOG_FILE = "/tmp/ram_diagnosis.log"

def log(msg):
    """Schreibt direkt in Log-Datei (umgeht stdout-Buffer)."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(LOG_FILE, "a") as f:
        f.write(line)
    print(line, end="", flush=True)

def get_rss_mb():
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0

def get_thread_count():
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0

def malloc_trim():
    """glibc malloc_trim(0) aufrufen — gibt freie Pages ans OS zurueck."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        result = libc.malloc_trim(0)
        return result
    except Exception:
        return -1

if __name__ == "__main__":
    # Log-Datei frisch starten
    with open(LOG_FILE, "w") as f:
        f.write("M.O.L.O.C.H. RAM Arena Diagnose\n")

    arena_max = os.environ.get("MALLOC_ARENA_MAX", "nicht gesetzt")
    log(f"PID: {os.getpid()}")
    log(f"MALLOC_ARENA_MAX: {arena_max}")
    log(f"Baseline RSS: {get_rss_mb():.0f} MB")

    # Service importieren und starten
    log("Importiere MolochService...")
    from core.moloch_service import MolochService
    log(f"Nach Import: RSS={get_rss_mb():.0f} MB")

    log("MolochService.__init__()...")
    service = MolochService()
    log(f"Nach __init__: RSS={get_rss_mb():.0f} MB")

    log("service.init()...")
    service.init()
    log(f"Nach init(): RSS={get_rss_mb():.0f} MB")

    log("service.start()...")
    service.start()
    log(f"Nach start(): RSS={get_rss_mb():.0f} MB")

    # Monitor
    log("=== Monitor Start (60s, alle 3s) ===")
    MAX_RSS = 1500
    t_start = time.time()
    rss_baseline = get_rss_mb()

    try:
        for i in range(20):  # 20 × 3s = 60s
            time.sleep(3)
            gc.collect()
            trimmed = malloc_trim()
            rss = get_rss_mb()
            threads = get_thread_count()
            elapsed = time.time() - t_start
            delta = rss - rss_baseline

            log(f"  {elapsed:5.1f}s | RSS: {rss:7.0f} MB | Δ: {delta:+7.0f} MB | "
                f"Threads: {threads:3d} | trim={trimmed}")

            if rss > MAX_RSS:
                log(f"*** NOTBREMSE bei {rss:.0f} MB ***")
                break
    except KeyboardInterrupt:
        log("CTRL-C")
    finally:
        log("Stoppe Service...")
        service.running = False
        try:
            service._inference.stop()
        except Exception:
            pass
        time.sleep(2)
        gc.collect()
        malloc_trim()
        log(f"Nach Stop: RSS={get_rss_mb():.0f} MB")
        log("Fertig.")
