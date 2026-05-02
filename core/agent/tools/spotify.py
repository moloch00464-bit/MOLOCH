"""W21 Tools — Spotify (Top-Artists + Play + Phase 3 #3 9 Tools)."""
from __future__ import annotations
import json
import logging
import os
import tempfile
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("agent.tools.spotify")
SPOTIFY_PROFILE_PATH = "/mnt/moloch-data/memory/spotify/spotify_profile.json"
SPOTIFY_STATE_PATH = "/dev/shm/moloch_spotify_state.json"
CMD_DIR = "/tmp"


def _atomic_ipc_cmd(action: str, params: Optional[Dict[str, Any]] = None) -> bool:
    """Atomic IPC-Cmd-Write (NEVER 6: tempfile + os.replace)."""
    cmd: Dict[str, Any] = {"action": action, "ts": time.time()}
    if params:
        cmd.update(params)
    fd, tmp = tempfile.mkstemp(
        dir=CMD_DIR, prefix=f"moloch_cmd_{action}_", suffix=".json.tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cmd, f)
        target = tmp.replace(".tmp", "")
        os.replace(tmp, target)
        return True
    except Exception as e:
        logger.warning(f"_atomic_ipc_cmd {action} fail: {e}")
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


def spotify_top_artists(n: int = 20) -> Dict[str, Any]:
    try:
        with open(SPOTIFY_PROFILE_PATH) as f:
            prof = json.load(f)
        artists = prof.get("top_artists") or prof.get("artists") or []
        return {"artists": artists[:n], "total": len(artists)}
    except Exception as e:
        return {"error": str(e)[:200], "artists": []}


def spotify_top_tracks(n: int = 20) -> Dict[str, Any]:
    """Markus' Top-Tracks aus Spotify-Profil."""
    try:
        with open(SPOTIFY_PROFILE_PATH) as f:
            prof = json.load(f)
        tracks = prof.get("top_tracks") or prof.get("tracks") or []
        return {"tracks": tracks[:n], "total": len(tracks)}
    except Exception as e:
        return {"error": str(e)[:200], "tracks": []}


def spotify_play(query_or_uri: str) -> Dict[str, Any]:
    try:
        ok = _atomic_ipc_cmd("spotify_play_query", {"query": query_or_uri})
        return {"ok": ok, "queued": query_or_uri}
    except Exception as e:
        return {"error": str(e)[:200]}


def spotify_pause() -> Dict[str, Any]:
    try:
        return {"ok": _atomic_ipc_cmd("spotify_pause", {})}
    except Exception as e:
        return {"error": str(e)[:200]}


def spotify_next() -> Dict[str, Any]:
    try:
        return {"ok": _atomic_ipc_cmd("spotify_skip", {})}
    except Exception as e:
        return {"error": str(e)[:200]}


def spotify_prev() -> Dict[str, Any]:
    try:
        return {"ok": _atomic_ipc_cmd("spotify_previous", {})}
    except Exception as e:
        return {"error": str(e)[:200]}


def spotify_volume(percent: int) -> Dict[str, Any]:
    try:
        ok = _atomic_ipc_cmd("spotify_volume", {"volume": int(percent)})
        return {"ok": ok, "percent": int(percent)}
    except Exception as e:
        return {"error": str(e)[:200]}


def spotify_search(query: str) -> Dict[str, Any]:
    try:
        ok = _atomic_ipc_cmd("spotify_search", {"query": query})
        return {"ok": ok, "query": query}
    except Exception as e:
        return {"error": str(e)[:200]}


def spotify_now_playing() -> Dict[str, Any]:
    """Aktueller Track aus state-file (W18 cross-prozess)."""
    try:
        with open(SPOTIFY_STATE_PATH) as f:
            d = json.load(f)
        ct = d.get("current_track") or {}
        return {
            "playing": d.get("playing"),
            "track": ct.get("name"),
            "artist": ct.get("artist"),
            "uri": ct.get("uri"),
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def spotify_recommend(seed_genre: Optional[str] = None,
                      target_energy: Optional[float] = None) -> Dict[str, Any]:
    """Empfehlung — best-effort: nutzt zonen-basierte Recommendation aus controller.
    KEIN Spotify-API-Call (laut music-Agent: lokaler Index ist EINZIGE Quelle).
    """
    try:
        from core.spotify_controller import get_spotify  # type: ignore
        sp = get_spotify()
        if hasattr(sp, "get_zone_recommendation"):
            rec = sp.get_zone_recommendation(seed_genre)
            return {"recommendation": rec, "seed_genre": seed_genre}
        return {
            "hint": "use spotify_play_genre instead",
            "seed_genre": seed_genre,
            "target_energy": target_energy,
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def spotify_play_genre(genre: str) -> Dict[str, Any]:
    """Spielt zonen-passendes Genre (mapping ueber zone_bias)."""
    try:
        ok = _atomic_ipc_cmd("spotify_play_genre", {"genre": genre})
        return {"ok": ok, "genre": genre}
    except Exception as e:
        return {"error": str(e)[:200]}
