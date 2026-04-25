#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Behavior Mutation Ledger
=======================================

Welle 1 / W1.2 von ThreeBrain FineTune Loop.

Append-only Audit-Log fuer alle Charakter-/Verhaltens-Aenderungen:
  - rule_proposed / rule_approved / rule_rejected / rule_deactivated
  - training_run_started / training_run_complete / training_run_failed
  - sample_proposed / sample_approved / sample_rejected
  - adapter_deployed / hef_recompiled / adapter_rolled_back

Storage:
  /mnt/moloch-data/memory/behavior_mutation_ledger.jsonl
  /mnt/moloch-data/memory/behavior_mutation_ledger_state.json   (event_counter)

Singleton: get_ledger()

API:
  ledger.log(event, **meta) -> ledger_id
  ledger.read_recent(n=50) -> List[Dict]
  ledger.read_filtered(event_type=None, since_ts=None) -> List[Dict]
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("BehaviorLedger")

LEDGER_PATH = "/mnt/moloch-data/memory/behavior_mutation_ledger.jsonl"
STATE_PATH = "/mnt/moloch-data/memory/behavior_mutation_ledger_state.json"

MAX_META_VALUE_LEN = 500


def _utc_iso_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_write_json(path: str, data: Any) -> None:
    """Atomar + NTFS-Fallback (Pattern aus character_journal.py)."""
    tmp_path = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp_path, path)
        except OSError:
            with open(tmp_path, "r", encoding="utf-8") as f_src:
                content = f_src.read()
            with open(path, "w", encoding="utf-8") as f_dst:
                f_dst.write(content)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        logger.error(f"[LEDGER] State-Write fehlgeschlagen: {e}")
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _truncate_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Lange String-Werte truncen damit Ledger-File handhabbar bleibt."""
    out = {}
    for k, v in meta.items():
        if isinstance(v, str) and len(v) > MAX_META_VALUE_LEN:
            out[k] = v[:MAX_META_VALUE_LEN] + "...[truncated]"
        else:
            out[k] = v
    return out


# =============================================================================
# BehaviorMutationLedger
# =============================================================================

class BehaviorMutationLedger:
    """Append-only Ledger fuer Charakter-/Training-/Adapter-Events."""

    def __init__(self):
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
        self._last_id = self._load_last_id()
        logger.info(f"[LEDGER] Initialisiert: file={LEDGER_PATH} last_id={self._last_id}")

    def _load_last_id(self) -> int:
        if not os.path.exists(STATE_PATH):
            return 0
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return int(json.load(f).get("last_id", 0))
        except Exception as e:
            logger.warning(f"[LEDGER] State kaputt, starte bei 0: {e}")
            return 0

    def _save_last_id(self, last_id: int) -> None:
        _safe_write_json(STATE_PATH, {"last_id": last_id, "updated": _utc_iso_ms()})

    # ---------------------------------------------------------------- WRITE

    def log(self, event: str, **meta) -> Optional[str]:
        """Append-Eintrag.

        Args:
            event: kurzer Event-Name (z.B. "rule_approved", "training_run_started")
            **meta: beliebige Key-Value Daten (lange Strings werden getruncht)

        Returns:
            ledger_id (z.B. "led_00000042") oder None bei Fehler.
        """
        if not event or not event.strip():
            logger.warning("[LEDGER] Leerer event — verworfen")
            return None
        meta = _truncate_meta(meta or {})

        with self._lock:
            new_id = self._last_id + 1
            ledger_id = f"led_{new_id:08d}"
            entry = {
                "ts": _utc_iso_ms(),
                "ledger_id": ledger_id,
                "event": event.strip()[:80],
                "meta": meta,
            }
            try:
                with open(LEDGER_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                logger.error(f"[LEDGER] Append fehlgeschlagen: {e}")
                return None
            self._last_id = new_id
            self._save_last_id(new_id)
            return ledger_id

    # ----------------------------------------------------------------- READ

    def read_recent(self, n: int = 50) -> List[Dict]:
        """Letzte N Eintraege (chronologisch, neueste zuletzt)."""
        if not os.path.exists(LEDGER_PATH):
            return []
        try:
            entries: List[Dict] = []
            with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return entries[-n:]
        except Exception as e:
            logger.error(f"[LEDGER] Read fehlgeschlagen: {e}")
            return []

    def read_filtered(
        self,
        event_type: Optional[str] = None,
        since_ts: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict]:
        """Filter nach event-Name oder Zeitstempel."""
        results: List[Dict] = []
        for entry in self.read_recent(limit * 2):
            if event_type and entry.get("event") != event_type:
                continue
            if since_ts and entry.get("ts", "") < since_ts:
                continue
            results.append(entry)
        return results[-limit:]

    def get_state(self) -> Dict[str, Any]:
        """Status fuer IPC/Panel."""
        return {
            "last_id": self._last_id,
            "file": LEDGER_PATH,
            "exists": os.path.exists(LEDGER_PATH),
        }


# =============================================================================
# Singleton
# =============================================================================

_instance: Optional[BehaviorMutationLedger] = None
_instance_lock = threading.Lock()


def get_ledger() -> BehaviorMutationLedger:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = BehaviorMutationLedger()
    return _instance


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    L = get_ledger()
    start_id = L._last_id

    samples = [
        ("rule_proposed", {"rule_id": "rule_00000001", "by": "distiller", "trigger": "test"}),
        ("rule_approved", {"rule_id": "rule_00000001", "by": "markus"}),
        ("training_run_started", {"samples": 142, "adapter_version": "v3"}),
        ("training_run_complete", {"adapter_version": "v3", "loss": 0.42, "duration_min": 23}),
        ("adapter_deployed", {"version": "v3", "scope": "remote_ryzen"}),
        ("hef_recompiled", {"version": "v3", "scope": "pi_local", "size_mb": 720}),
    ]

    written = []
    for ev, meta in samples:
        lid = L.log(ev, **meta)
        assert lid is not None
        written.append(lid)

    expected = [f"led_{start_id + i + 1:08d}" for i in range(len(samples))]
    assert written == expected, f"Sequenz falsch:\n got {written}\n exp {expected}"

    # Validation
    assert L.log("") is None

    # Lange meta-strings truncen
    long_str = "x" * 1000
    L.log("test_truncation", payload=long_str)
    last = L.read_recent(1)[-1]
    assert "...[truncated]" in last["meta"]["payload"]

    # Filter
    filtered = L.read_filtered(event_type="rule_approved")
    assert len(filtered) >= 1
    assert all(e["event"] == "rule_approved" for e in filtered)

    print(f"\nSelf-Test PASS")
    print(f"  Geschrieben: {len(written) + 1} ({start_id + 1}..{L._last_id})")
    print(f"  File: {LEDGER_PATH}")
    print(f"  Letzter Eintrag: {L.read_recent(1)[-1]}")
