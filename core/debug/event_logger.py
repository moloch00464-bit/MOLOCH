#!/usr/bin/env python3
"""
M.O.L.O.C.H. Event Logger — Gate 3/4 Perception Debug
=======================================================

Leichtgewichtiges Trace-Logging fuer Event Bus Events.
Schreibt in /home/molochzuhause/moloch/logs/event_trace.log
"""

import time

_LOG_PATH = "/home/molochzuhause/moloch/logs/event_trace.log"


def log_event(event_type, data):
    """Event in Trace-Log schreiben."""
    try:
        with open(_LOG_PATH, "a") as f:
            f.write(f"{time.time()} | {event_type} | {data}\n")
    except Exception:
        pass
