"""DH-6 State-Logger (PC-Side).

ChatGPT-Synthese: Zeitbasierte Aufzeichnung aller Zustande + Uebergaenge.
Format JSONL fuer einfaches replay/training.

Pfad: %LOCALAPPDATA%/moloch_pc_state/state_log/YYYY-MM-DD.jsonl
Rotation taeglich, 7 Tage Retention.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

_LOCAL_APPDATA = os.environ.get("LOCALAPPDATA")
if _LOCAL_APPDATA:
    _LOG_BASE = Path(_LOCAL_APPDATA) / "moloch_pc_state" / "state_log"
else:
    _LOG_BASE = Path.home() / "moloch_pc_state" / "state_log"
_LOG_BASE.mkdir(parents=True, exist_ok=True)

RETENTION_DAYS = 7


def _today_log_path() -> Path:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return _LOG_BASE / f"{today}.jsonl"


def log_state(
    vector: Dict[str, float],
    primary: str,
    tension_meta: float,
    authority: str = "pc",
    extra: Dict[str, Any] | None = None,
) -> None:
    """Append state-Snapshot ans tagesaktuelle Log-File.

    Stiller Fehler-Mode: bei Disk-Voll/Permission gibts log-loss, kein crash.
    """
    entry = {
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "vector": {k: round(v, 4) for k, v in vector.items()},
        "primary": primary,
        "tension_meta": round(tension_meta, 4),
        "authority": authority,
    }
    if extra:
        entry.update(extra)
    try:
        path = _today_log_path()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def cleanup_old_logs(retention_days: int = RETENTION_DAYS) -> int:
    """Loescht Log-Files aelter als retention_days. Returnt count deleted.

    Soll periodisch (z.B. taeglich) aufgerufen werden.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    deleted = 0
    try:
        for f in _LOG_BASE.glob("*.jsonl"):
            try:
                stem = f.stem  # YYYY-MM-DD
                file_date = datetime.strptime(stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if file_date < cutoff:
                    f.unlink()
                    deleted += 1
            except Exception:
                continue
    except Exception:
        pass
    return deleted


def log_path() -> Path:
    return _today_log_path()


def all_logs() -> list[Path]:
    """Returnt alle JSONL-Files sortiert nach Datum."""
    try:
        return sorted(_LOG_BASE.glob("*.jsonl"))
    except Exception:
        return []
