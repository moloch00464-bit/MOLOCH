#!/usr/bin/env python3
"""
M.O.L.O.C.H. Event Logger — Gate 3/4 Perception Debug
=======================================================

Leichtgewichtiges Trace-Logging fuer Event Bus Events.
Schreibt in /home/molochzuhause/moloch/logs/event_trace.log

Rotation: 10MB max, 3 Backups (event_trace.log.1/.2/.3)
"""

import os
import time

_LOG_PATH = "/home/molochzuhause/moloch/logs/event_trace.log"
_MAX_SIZE = 10 * 1024 * 1024   # 10 MB
_MAX_BACKUPS = 3
_check_counter = 0
_CHECK_INTERVAL = 500           # alle 500 Events auf Groesse pruefen


def _rotate():
    """Aelteste Backups wegschieben, aktuelle Datei -> .1 umbenennen."""
    try:
        # .3 loeschen, .2 -> .3, .1 -> .2, aktuell -> .1
        for i in range(_MAX_BACKUPS - 1, 0, -1):
            src = f"{_LOG_PATH}.{i}"
            dst = f"{_LOG_PATH}.{i + 1}"
            if os.path.exists(src):
                os.rename(src, dst)
        if os.path.exists(_LOG_PATH):
            os.rename(_LOG_PATH, f"{_LOG_PATH}.1")
    except Exception:
        pass


def log_event(event_type, data):
    """Event in Trace-Log schreiben. Rotiert automatisch bei 10 MB."""
    global _check_counter
    try:
        _check_counter += 1
        if _check_counter % _CHECK_INTERVAL == 0:
            if os.path.exists(_LOG_PATH) and os.path.getsize(_LOG_PATH) > _MAX_SIZE:
                _rotate()
        with open(_LOG_PATH, "a") as f:
            f.write(f"{time.time()} | {event_type} | {data}\n")
    except Exception:
        pass


# Beim Import: einmalig rotieren wenn Datei schon zu gross ist
try:
    if os.path.exists(_LOG_PATH) and os.path.getsize(_LOG_PATH) > _MAX_SIZE:
        _rotate()
except Exception:
    pass
