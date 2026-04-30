"""Unconscious-Engine-Auditor (Welle 14).

Pullt:
- L0: core.unconscious_engine import + get_unconscious() callable
- L1: mood-impulse-rate (Anzahl Impulse letzte Stunde aus journalctl|grep "[UNCONSCIOUS]"
      oder via API falls vorhanden)
- L2: anima-Aktivierung (config/anima_mappings.json existiert + Modtime)

Schreibt audit_state.layers.unconscious Schema:
  {module_alive, impulses_1h, anima_mappings_loaded, last_impulse_age_s,
   score, max, status, detail}

Status-Logik:
- PASS: Modul alive, impulses_1h >= 1
- WARN: Modul alive aber impulses=0 letzte 6h
- FAIL: Modul-Import-Crash
"""
from __future__ import annotations

import os
import re
import time
import subprocess
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("unconscious_auditor")

_ANIMA_PATH = "/home/molochzuhause/moloch/config/anima_mappings.json"
_TS_RE = re.compile(r"^([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})")


def _journal_impulses(window: str) -> int:
    """Zaehlt [UNCONSCIOUS] Eintraege aus journalctl im Window (z.B. '1 hour ago')."""
    try:
        r = subprocess.run(
            ["sudo", "journalctl", "-u", "moloch", "--since", window,
             "--no-pager", "-n", "5000"],
            capture_output=True, text=True, timeout=10,
        )
        count = 0
        for ln in r.stdout.splitlines():
            if "[UNCONSCIOUS]" in ln or "unconscious_impulse" in ln.lower():
                count += 1
        return count
    except Exception:
        return 0


def _last_impulse_age_seconds() -> Optional[float]:
    """Liest letzte [UNCONSCIOUS]-Zeile aus journalctl 24h und schaetzt Alter."""
    try:
        r = subprocess.run(
            ["sudo", "journalctl", "-u", "moloch", "--since", "24 hours ago",
             "--no-pager", "-n", "5000"],
            capture_output=True, text=True, timeout=10,
        )
        last_line = None
        for ln in r.stdout.splitlines():
            if "[UNCONSCIOUS]" in ln or "unconscious_impulse" in ln.lower():
                last_line = ln
        if not last_line:
            return None
        # journalctl-Format: "Apr 30 12:34:56 host moloch[123]: ..."
        m = _TS_RE.match(last_line)
        if not m:
            return None
        try:
            ts = time.strptime(f"{time.strftime('%Y')} {m.group(1)}", "%Y %b %d %H:%M:%S")
            secs = time.mktime(ts)
            return max(0.0, time.time() - secs)
        except Exception:
            return None
    except Exception:
        return None


def collect() -> Dict[str, Any]:
    """Sammelt Unconscious-Layer-Daten."""
    detail: Dict[str, Any] = {}
    module_alive = False
    api_impulses_1h: Optional[int] = None

    # L0: Import (Symbol kann je nach Version verschieden heissen)
    try:
        eng = None
        try:
            from core.unconscious_engine import get_unconscious_engine  # type: ignore
            eng = get_unconscious_engine()
        except Exception:
            try:
                from core.unconscious_engine import get_unconscious  # type: ignore
                eng = get_unconscious()
            except Exception:
                pass
        module_alive = eng is not None
        # L1 best-effort via API
        if module_alive:
            for attr in ("get_status", "stats", "status"):
                if hasattr(eng, attr):
                    try:
                        st = getattr(eng, attr)
                        st = st() if callable(st) else st
                        if isinstance(st, dict):
                            for key in ("impulses_1h", "impulses_last_hour", "impulse_count_1h"):
                                if key in st:
                                    api_impulses_1h = int(st[key])
                                    break
                            detail["api_status"] = {
                                k: v for k, v in st.items()
                                if k in ("running", "rate_per_min", "last_impulse",
                                         "impulses_total")
                            }
                    except Exception as e:
                        detail["api_error"] = str(e)[:80]
                    break
    except Exception as e:
        detail["import_error"] = str(e)[:120]
        module_alive = False

    # L1: Journal-Counter
    impulses_1h = api_impulses_1h if api_impulses_1h is not None else _journal_impulses("1 hour ago")
    impulses_6h = _journal_impulses("6 hours ago")
    detail["impulses_6h"] = impulses_6h

    # L2: anima_mappings
    anima_loaded = False
    anima_mtime_age = None
    if os.path.exists(_ANIMA_PATH):
        anima_loaded = True
        try:
            anima_mtime_age = round(time.time() - os.path.getmtime(_ANIMA_PATH), 1)
        except Exception:
            pass
    detail["anima_path"] = _ANIMA_PATH
    detail["anima_mtime_age_s"] = anima_mtime_age

    # Letzter Impuls-Age
    last_age = _last_impulse_age_seconds()

    # Status-Berechnung
    score = 0
    max_score = 4
    if module_alive:
        score += 1
    if impulses_1h >= 1:
        score += 1
    if impulses_6h >= 1:
        score += 1
    if anima_loaded:
        score += 1

    if not module_alive:
        status = "FAIL"
    elif impulses_6h == 0:
        status = "WARN"
    elif impulses_1h == 0:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "score": score,
        "max": max_score,
        "status": status,
        "module_alive": module_alive,
        "impulses_1h": impulses_1h,
        "anima_mappings_loaded": anima_loaded,
        "last_impulse_age_s": last_age,
        "detail": detail,
    }
