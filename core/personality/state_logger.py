#!/usr/bin/env python3
"""
M.O.L.O.C.H. State Logger — Phase 1 (Drei-Hirn-Synthese, ChatGPT-Position)
===========================================================================

Zeitbasierte Aufzeichnung aller State-Transitions als JSONL.
ChatGPT-Synthese-Punkt: "Debugging, Training, Simulation".

Pfad: /mnt/moloch-data/memory/state_log/YYYY-MM-DD.jsonl  (SSD2 - ueberlebt Reboots)
Format: 1 JSON pro Zeile mit ts, from_state, to_state, vector, tension, reason
Rotation: 7 Tage werden gehalten, aeltere Files automatisch geloescht.

Singleton: get_state_logger()
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("MolochStateLogger")

LOG_DIR = Path("/mnt/moloch-data/memory/state_log")
RETENTION_DAYS = 7


class StateLogger:
    """JSONL-Logger fuer State-Transitions mit 7d-Rotation."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_rotate_ts: float = 0.0
        self._rotate_interval_s: float = 3600.0  # einmal pro Stunde rotieren
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"LOG_DIR mkdir fehlgeschlagen: {e}")

    def _path_today(self) -> Path:
        return LOG_DIR / f"{datetime.now():%Y-%m-%d}.jsonl"

    def log_transition(self, from_state: str, to_state: str,
                       vector: Optional[Dict[str, float]] = None,
                       tension: float = 0.0,
                       reason: str = "") -> bool:
        """Append eine Transition als JSON-Line.

        Returnt True bei Erfolg, False bei IO-Error (kein Crash).
        """
        entry = {
            "ts": time.time(),
            "iso": datetime.now().isoformat(timespec="seconds"),
            "from": from_state,
            "to": to_state,
            "tension": float(tension) if tension is not None else 0.0,
            "vector": dict(vector) if isinstance(vector, dict) else {},
            "reason": (reason or "")[:120],
        }
        line = json.dumps(entry, ensure_ascii=False)
        path = self._path_today()
        try:
            with self._lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                self._maybe_rotate()
            return True
        except Exception as e:
            logger.warning(f"log_transition fail ({path}): {e}")
            return False

    def _maybe_rotate(self) -> None:
        now = time.time()
        if (now - self._last_rotate_ts) < self._rotate_interval_s:
            return
        self._last_rotate_ts = now
        try:
            cutoff = datetime.now() - timedelta(days=RETENTION_DAYS)
            for f in LOG_DIR.glob("*.jsonl"):
                try:
                    name = f.stem  # YYYY-MM-DD
                    file_date = datetime.strptime(name, "%Y-%m-%d")
                    if file_date < cutoff:
                        f.unlink()
                        logger.info(f"rotated out: {f.name}")
                except (ValueError, OSError):
                    continue
        except Exception as e:
            logger.debug(f"rotate fail: {e}")

    def today_size_bytes(self) -> int:
        path = self._path_today()
        try:
            return path.stat().st_size if path.exists() else 0
        except OSError:
            return 0

    def today_count(self) -> int:
        path = self._path_today()
        try:
            if not path.exists():
                return 0
            with open(path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0


_instance: Optional[StateLogger] = None
_singleton_lock = threading.Lock()


def get_state_logger() -> StateLogger:
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = StateLogger()
        return _instance


if __name__ == "__main__":
    sl = get_state_logger()
    ok = sl.log_transition("idle", "observing", vector={"observing": 0.8, "idle": 0.2}, tension=0.1, reason="self-test")
    print(f"log_transition ok={ok}")
    print(f"today size={sl.today_size_bytes()} count={sl.today_count()}")
