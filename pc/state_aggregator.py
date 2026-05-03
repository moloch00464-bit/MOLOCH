"""MOLOCH State-Aggregator (PC-Side, Phase 1).

Liest Pi /api/state/current (kommt mit Pi-Side state_engine.py von Pi-Opus)
oder Fallback auf /state_full. Aggregiert zu State-Vector mit
Historien-Gewichtung — fuer Avatar 2.0, Cockpit-State-Visualisierung,
und PC-Side State-Aware-Logic.

Konzept:
- Pi schreibt Single-State + State-Vector pro Tick
- PC bildet GEWICHTETEN Vector mit Historien-Decay (z.B. EMA mit alpha=0.3)
- Avatar liest /api/state_vector (oder direkt aus diesem Service)

Endpoints:
  GET /api/state_vector  -> {weighted_vector, current_pi_state, tension, history_n}
  GET /health

Optional CLI: python -m pc.state_aggregator --once -> einmalig pollen + JSON ausgeben

NEVER 5: timeout=Pflicht. NEVER 6: atomic write. NEVER 8: kein shell.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import httpx
import uvicorn
from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("state-aggregator")

HOST = os.environ.get("MOLOCH_STATE_AGG_HOST", "0.0.0.0")
PORT = int(os.environ.get("MOLOCH_STATE_AGG_PORT", "11652"))
PI_BASE = os.environ.get("MOLOCH_PI_BASE", "http://192.168.178.30:9100")
POLL_INTERVAL_SEC = float(os.environ.get("MOLOCH_STATE_POLL_SEC", "1.0"))
HISTORY_SIZE = int(os.environ.get("MOLOCH_STATE_HISTORY", "30"))
EMA_ALPHA = float(os.environ.get("MOLOCH_STATE_EMA_ALPHA", "0.3"))

# 6-State-Vector Order (matches Pi-Side)
STATE_KEYS = (
    "idle",
    "observing",
    "engaged",
    "overloaded",
    "withdrawing",
    "offline_anchor",
)

# State-File fuer Avatar/Cockpit-Read (atomic write)
_STATE_FILE = Path(
    os.environ.get(
        "MOLOCH_STATE_AGG_FILE",
        os.path.expandvars("%LOCALAPPDATA%/moloch_pc_state/state_vector.json"),
    )
)
_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


class StateAggregator:
    def __init__(self) -> None:
        self.history: Deque[Dict[str, Any]] = deque(maxlen=HISTORY_SIZE)
        # EMA-Vector
        self.ema_vector: Dict[str, float] = {k: 0.0 for k in STATE_KEYS}
        self.ema_vector["idle"] = 1.0  # initial idle
        self.last_pi_state: Optional[str] = None
        self.last_tension: float = 0.0
        self.last_zone: Optional[str] = None
        self.last_update_ts: float = 0.0

    def update_from_pi(self, pi_state: Dict[str, Any]) -> Dict[str, Any]:
        """Akzeptiert dict von /api/state/current oder /state_full.
        Aktualisiert EMA + Historie. Returnt aggregierten Vector.
        """
        now = time.time()

        # Extract Pi state
        current_state = pi_state.get("current_state") or pi_state.get("zone") or "idle"
        tension = float(pi_state.get("tension", 0.0))
        zone = pi_state.get("zone")

        # Single-State -> One-Hot Probe
        observed = {k: 0.0 for k in STATE_KEYS}
        if current_state in observed:
            observed[current_state] = 1.0
        else:
            # Fallback: zone-Mapping (legacy)
            if zone == "guardian":
                observed["idle"] = 0.5
                observed["observing"] = 0.5
            elif zone == "shadow":
                observed["withdrawing"] = 0.7
                observed["observing"] = 0.3
            elif zone == "berserker":
                observed["overloaded"] = 0.7
                observed["engaged"] = 0.3
            else:
                observed["idle"] = 1.0

        # EMA-Update
        for k in STATE_KEYS:
            self.ema_vector[k] = (
                EMA_ALPHA * observed[k] + (1.0 - EMA_ALPHA) * self.ema_vector[k]
            )

        # Append history snapshot
        self.history.append(
            {
                "ts": now,
                "current_state": current_state,
                "tension": tension,
                "zone": zone,
            }
        )

        self.last_pi_state = current_state
        self.last_tension = tension
        self.last_zone = zone
        self.last_update_ts = now

        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "ts": self.last_update_ts,
            "current_pi_state": self.last_pi_state,
            "current_zone": self.last_zone,
            "tension": self.last_tension,
            "weighted_vector": {k: round(v, 4) for k, v in self.ema_vector.items()},
            "history_n": len(self.history),
            "history_size_max": HISTORY_SIZE,
            "ema_alpha": EMA_ALPHA,
        }


_aggregator = StateAggregator()


def _atomic_write_state(snapshot: Dict[str, Any]) -> None:
    """NEVER 6: tempfile + os.replace."""
    import tempfile

    try:
        fd, tmp = tempfile.mkstemp(
            dir=str(_STATE_FILE.parent),
            prefix=_STATE_FILE.name + ".",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
            os.replace(tmp, str(_STATE_FILE))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except Exception as e:
        logger.warning(f"atomic_write failed: {e}")


async def _poll_pi(client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """Versuche mehrere Pi-Endpoints in Reihenfolge, normalisiere Response.

    Akzeptiert:
    - /api/state/current  (mein Spec-Vorschlag, current_state/tension/...)
    - /state/vector       (Pi-Opus Welle DH-1, primary/tension_meta/state_vector)
    - /state_full         (legacy fallback, zone-only)

    Normalisiert auf einheitliches Format fuer update_from_pi():
    {current_state, state_vector, tension, zone, identity_phrase}
    """
    for url in (
        f"{PI_BASE}/api/state/current",
        f"{PI_BASE}/state/vector",
        f"{PI_BASE}/state_full",
    ):
        try:
            r = await client.get(url, timeout=2.0)
            if r.status_code == 200:
                try:
                    raw = r.json()
                    return _normalize_pi_response(raw, source_url=url)
                except Exception:
                    pass
        except Exception:
            pass
    return None


def _normalize_pi_response(raw: Dict[str, Any], source_url: str) -> Dict[str, Any]:
    """Normalisiert Pi-Response auf einheitliches Format.

    Toleriert beide Spec-Varianten:
    - Mein Vorschlag: current_state, tension, state_vector, ...
    - Pi-Opus DH-1:   primary, tension_meta, state_vector, ...
    """
    # Direct passthrough wenn bereits in Mein-Format
    if "current_state" in raw:
        return raw

    # Pi-Opus DH-1 Format -> normalisieren
    if "primary" in raw:
        return {
            "current_state": raw.get("primary", "idle"),
            "state_vector": raw.get("state_vector", {}),
            "tension": raw.get("tension_meta", raw.get("tension", 0.0)),
            "zone": raw.get("zone"),
            "identity_phrase": raw.get("identity_phrase"),
            "timestamp": raw.get("timestamp"),
            "_source_endpoint": source_url,
        }

    # /state_full legacy -> minimal mappen (zone -> Pseudo-State)
    return {
        "current_state": "idle",  # Default da kein State-Engine
        "tension": float(raw.get("tension", 0.0)) if isinstance(raw.get("tension"), (int, float)) else 0.0,
        "zone": raw.get("zone"),
        "_source_endpoint": source_url,
        "_legacy_state_full": True,
    }


async def _poll_loop():
    """Background-Task: pollt Pi + updated EMA + schreibt State-File.

    IMPORTANT FIX (Code-Review #3): Default-Timeout auf Client-Level.
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
        while True:
            pi_state = await _poll_pi(client)
            if pi_state is not None:
                snapshot = _aggregator.update_from_pi(pi_state)
                _atomic_write_state(snapshot)
            try:
                import asyncio

                await asyncio.sleep(POLL_INTERVAL_SEC)
            except Exception:
                break


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import asyncio

    task = asyncio.create_task(_poll_loop())
    logger.info(
        f"MOLOCH State-Aggregator startet auf {HOST}:{PORT} "
        f"(Pi-Source: {PI_BASE}, Poll {POLL_INTERVAL_SEC}s)"
    )
    yield
    task.cancel()
    try:
        await task
    except Exception:
        pass


app = FastAPI(title="MOLOCH State-Aggregator", version="1.0", lifespan=lifespan)


@app.get("/api/state_vector")
def api_state_vector() -> Dict[str, Any]:
    return _aggregator.snapshot()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "moloch-state-aggregator",
        "version": "1.0",
        "port": PORT,
        "pi_base": PI_BASE,
        "history_n": len(_aggregator.history),
        "last_update_ts": _aggregator.last_update_ts,
    }


def main() -> None:
    logger.info(f"MOLOCH State-Aggregator startet auf {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


def cli_once() -> None:
    """CLI: einmalig pollen + JSON-Snapshot ausgeben (debug)."""
    import asyncio
    import sys

    async def _once():
        async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
            pi_state = await _poll_pi(client)
            if pi_state is None:
                print(json.dumps({"error": "Pi nicht erreichbar"}, indent=2))
                sys.exit(1)
            snap = _aggregator.update_from_pi(pi_state)
            print(json.dumps(snap, indent=2, ensure_ascii=False))

    asyncio.run(_once())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        cli_once()
    else:
        main()
