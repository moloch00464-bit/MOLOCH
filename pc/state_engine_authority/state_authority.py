"""DH-6 State-Authority (PC-Side).

POSTet authoritative State-Vector zurueck an Pi via /state/authority Endpoint
(falls Pi-Opus den Endpoint baut). Bis dahin: nur logging + state-file write.

Pi state_vector.py hat apply_pc_authority(vector) Methode - via HTTP-POST
oder /dev/shm-File-Write erreicht.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import httpx

logger = logging.getLogger("state-authority")

PI_BASE = os.environ.get("MOLOCH_PI_BASE", "http://192.168.178.30:9100")
AUTHORITY_ENDPOINTS = (
    f"{PI_BASE}/state/authority",
    f"{PI_BASE}/api/state/authority",
)

# State-File fuer Avatar/Cockpit-Direct-Read
_LOCAL_APPDATA = os.environ.get("LOCALAPPDATA")
if _LOCAL_APPDATA:
    _STATE_DIR = Path(_LOCAL_APPDATA) / "moloch_pc_state"
else:
    _STATE_DIR = Path.home() / "moloch_pc_state"
_STATE_DIR.mkdir(parents=True, exist_ok=True)
AUTHORITY_FILE = _STATE_DIR / "state_authority.json"


async def push_authority(
    vector: Dict[str, float],
    primary: str,
    tension_meta: float,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, object]:
    """POSTet authoritative Vector an Pi. Returnt {posted, endpoint, error}.

    Wenn Pi-Endpoint nicht existiert: lokale State-File reicht (Avatar liest da).
    """
    # FIX: Pi-Opus' DH-6-Counterpart-Endpoint erwartet state_vector + current_state
    # (nicht vector + primary wie urspruenglicher Spec). Pi-Spec final laut commit ae771af.
    payload = {
        "state_vector": vector,
        "current_state": primary,
        "tension_meta": tension_meta,
    }

    # Atomic write to local state-file (immer)
    _atomic_write(AUTHORITY_FILE, payload)

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=httpx.Timeout(2.0))

    posted_endpoint = None
    last_error = None
    try:
        for url in AUTHORITY_ENDPOINTS:
            try:
                r = await client.post(url, json=payload, timeout=2.0)
                if r.status_code in (200, 201, 204):
                    posted_endpoint = url
                    break
                else:
                    last_error = f"HTTP {r.status_code}"
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
    finally:
        if own_client:
            await client.aclose()

    return {
        "posted": posted_endpoint is not None,
        "endpoint": posted_endpoint,
        "error": last_error,
        "local_file": str(AUTHORITY_FILE),
    }


def _atomic_write(path: Path, data: Dict[str, object]) -> None:
    """NEVER 6: tempfile + os.replace."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, str(path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except Exception as e:
        logger.warning(f"atomic_write fail: {e}")
