"""W21 Tools — Spotify (Top-Artists + Play)."""
from __future__ import annotations
import json
import logging
import os
import tempfile
import time
from typing import Any, Dict

logger = logging.getLogger("agent.tools.spotify")
SPOTIFY_PROFILE_PATH = "/mnt/moloch-data/memory/spotify/spotify_profile.json"
CMD_DIR = "/tmp"


def spotify_top_artists(n: int = 20) -> Dict[str, Any]:
    try:
        with open(SPOTIFY_PROFILE_PATH) as f:
            prof = json.load(f)
        artists = prof.get("top_artists") or prof.get("artists") or []
        return {"artists": artists[:n], "total": len(artists)}
    except Exception as e:
        return {"error": str(e)[:200], "artists": []}


def spotify_play(query_or_uri: str) -> Dict[str, Any]:
    try:
        cmd = {
            "action": "spotify_play_query",
            "query": query_or_uri,
            "ts": time.time(),
        }
        # atomic IPC-cmd write (NEVER 6)
        fd, tmp = tempfile.mkstemp(
            dir=CMD_DIR, prefix="moloch_cmd_play_", suffix=".json.tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(cmd, f)
            target = tmp.replace(".tmp", "")
            os.replace(tmp, target)
            return {"ok": True, "queued": query_or_uri}
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        return {"error": str(e)[:200]}
