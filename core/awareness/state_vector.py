#!/usr/bin/env python3
"""
M.O.L.O.C.H. State Vector — 6-State Lightweight Reflector (Welle DH-1)
========================================================================

Pi-Side State-Reflector aus der Drei-Hirn-Synthese (Gemini/DeepSeek/ChatGPT).
Pi haelt einen LIGHTWEIGHT State-Vector (4GB-RAM-Constraint). Die volle
Transition-Engine + Safety Layer + Logger lebt auf dem PC (Welle DH-6, separat).

6 States (gewichtet, nicht binaer):
  1. idle             — kein Mensch, niedrige Tension
  2. observing        — Mensch da, keine Interaktion
  3. engaged          — aktive Interaktion (Sprache/Chat in letzten 10s)
  4. overloaded       — System-Stress (RAM>85% / Temp>75C / lange hohe Tension)
  5. withdrawing      — nach Konflikt-Zone (berserker recent)
  6. offline_anchor   — PC weg, Hardware-only

Tension ist Meta-Parameter (ChatGPT-Synthese-Entscheidung):
- KEIN direkter State-Trigger
- beeinflusst Uebergangsgeschwindigkeit + Lueftermodul
- gespeichert als tension_meta, separat vom State-Vector

Singleton: get_state_vector()
Read-Only fuer andere Module via vector() / primary() / snapshot()
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("MolochStateVector")

STATUS_PATH = Path("/dev/shm/moloch_status.json")

STATES = ("idle", "observing", "engaged", "overloaded", "withdrawing", "offline_anchor")

ENGAGED_WINDOW_SEC = 10.0
WITHDRAWING_WINDOW_SEC = 30.0
SMOOTHING_ALPHA = 0.35

RAM_OVERLOAD = 85.0
TEMP_OVERLOAD = 75.0
TENSION_HIGH_OVERLOAD = 0.70
TENSION_HIGH_DURATION_SEC = 60.0


class StateVector:
    """Lightweight 6-State-Vector mit gewichteter Aktivierung."""

    def __init__(self):
        self._lock = threading.Lock()
        self._vector: Dict[str, float] = {s: 0.0 for s in STATES}
        self._vector["idle"] = 1.0
        self._tension_meta: float = 0.0
        self._last_engaged_ts: float = 0.0
        self._last_berserker_ts: float = 0.0
        self._tension_high_since: float = 0.0
        self._authority: str = "pi_heuristic"
        self._last_update: float = 0.0

    def mark_engaged(self) -> None:
        """Externer Trigger - chat_server / voice_pipeline meldet Interaktion."""
        with self._lock:
            self._last_engaged_ts = time.time()

    def mark_berserker(self) -> None:
        """Externer Trigger - zone_changed event auf berserker."""
        with self._lock:
            self._last_berserker_ts = time.time()

    def apply_pc_authority(self, vector: Dict[str, float]) -> None:
        """Override durch PC-State-Authority (Welle DH-6).

        Vector wird normalisiert; unbekannte Keys ignoriert.
        """
        clean = {s: max(0.0, float(vector.get(s, 0.0))) for s in STATES}
        total = sum(clean.values()) or 1.0
        with self._lock:
            self._vector = {s: v / total for s, v in clean.items()}
            self._authority = "pc_remote"
            self._last_update = time.time()

    def tick(self) -> None:
        """Heuristische Neuberechnung aus moloch_status.json.

        Wird alle ~1s vom Service aufgerufen. Bei aktiver PC-Authority
        bleibt der externe Vector erhalten - hier nur tension_meta + zone.
        """
        status = self._read_status()
        if status is None:
            self._mark_offline()
            return

        person_count = self._person_count(status)
        ram_pct = float(status.get("ram_percent", 0.0) or 0.0)
        temp = float(status.get("cpu_temp", 0.0) or 0.0)
        zone = (status.get("zone") or "").lower()
        tension = self._tension(status)

        now = time.time()

        if zone == "berserker":
            self._last_berserker_ts = now
        if tension >= TENSION_HIGH_OVERLOAD:
            if self._tension_high_since == 0.0:
                self._tension_high_since = now
        else:
            self._tension_high_since = 0.0

        with self._lock:
            self._tension_meta = tension

            if self._authority == "pc_remote":
                # PC haelt den State autoritativ - nicht ueberschreiben
                self._last_update = now
                return

            target = self._heuristic_vector(
                person_count=person_count,
                ram_pct=ram_pct,
                temp=temp,
                zone=zone,
                now=now,
            )
            self._vector = self._smooth(self._vector, target)
            self._last_update = now

    def _heuristic_vector(self, person_count: int, ram_pct: float,
                          temp: float, zone: str, now: float) -> Dict[str, float]:
        v = {s: 0.0 for s in STATES}

        if person_count <= 0:
            v["idle"] = 1.0
        else:
            since_engaged = now - self._last_engaged_ts
            if since_engaged <= ENGAGED_WINDOW_SEC:
                v["engaged"] = 0.8
                v["observing"] = 0.2
            else:
                v["observing"] = 0.85
                v["idle"] = 0.15

        if ram_pct >= RAM_OVERLOAD or temp >= TEMP_OVERLOAD:
            v = {s: x * 0.4 for s, x in v.items()}
            v["overloaded"] = max(v["overloaded"], 0.6)

        if self._tension_high_since > 0.0 and (now - self._tension_high_since) >= TENSION_HIGH_DURATION_SEC:
            v = {s: x * 0.5 for s, x in v.items()}
            v["overloaded"] = max(v["overloaded"], 0.5)

        since_berserker = now - self._last_berserker_ts
        if 0.0 < since_berserker <= WITHDRAWING_WINDOW_SEC:
            v = {s: x * 0.6 for s, x in v.items()}
            v["withdrawing"] = max(v["withdrawing"], 0.4)

        total = sum(v.values()) or 1.0
        return {s: x / total for s, x in v.items()}

    def _smooth(self, prev: Dict[str, float], target: Dict[str, float]) -> Dict[str, float]:
        a = SMOOTHING_ALPHA
        out = {s: (1 - a) * prev.get(s, 0.0) + a * target.get(s, 0.0) for s in STATES}
        total = sum(out.values()) or 1.0
        return {s: v / total for s, v in out.items()}

    def _mark_offline(self) -> None:
        with self._lock:
            self._vector = {s: 0.0 for s in STATES}
            self._vector["offline_anchor"] = 1.0
            self._authority = "pi_offline"
            self._last_update = time.time()

    def _read_status(self) -> Optional[dict]:
        try:
            if not STATUS_PATH.exists():
                return None
            with open(STATUS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"status read fail: {e}")
            return None

    def _person_count(self, status: dict) -> int:
        dets = status.get("panel_detections") or []
        return sum(1 for d in dets if (d.get("class") or d.get("label") or "").lower() == "person")

    def _tension(self, status: dict) -> float:
        try:
            return float(status.get("tension", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def vector(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._vector)

    def primary(self) -> str:
        with self._lock:
            return max(self._vector.items(), key=lambda kv: kv[1])[0]

    def tension_meta(self) -> float:
        with self._lock:
            return self._tension_meta

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "vector": dict(self._vector),
                "primary": max(self._vector.items(), key=lambda kv: kv[1])[0],
                "tension_meta": self._tension_meta,
                "authority": self._authority,
                "last_update": self._last_update,
            }


_instance: Optional[StateVector] = None
_singleton_lock = threading.Lock()


def get_state_vector() -> StateVector:
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = StateVector()
        return _instance


if __name__ == "__main__":
    import pprint
    sv = get_state_vector()
    sv.tick()
    pprint.pprint(sv.snapshot())
