"""Spotify Closed-Loop-Verifier — play_artist -> Track-Wechsel.

PASS  : Artist-Match nach 5 s
WARN  : Track wechselte aber falscher Artist
FAIL  : kein Track-Wechsel (Bug B Verifier)
SKIP  : Spotify nicht initialisiert / kein Auth
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from ._common import fail_result, now, skip_result, write_ipc_cmd

logger = logging.getLogger("spotify_verify")

_TEST_ARTIST = "Suicide Commando"
_SLEEP_AFTER_PLAY = 5.0


def _get_spotify():
    try:
        from core.spotify_controller import get_spotify  # type: ignore
        return get_spotify()
    except Exception as e:
        logger.debug("spotify import failed: %s", e)
        return None


def _track_summary(status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(status, dict):
        return {}
    ct = status.get("current_track") or {}
    if not isinstance(ct, dict):
        ct = {}
    return {
        "uri": ct.get("uri"),
        "name": ct.get("name"),
        "artist": ct.get("artist") or ct.get("artists"),
        "track_str": status.get("current_track_str"),
    }


def _trigger_play_artist(sp, artist: str) -> str:
    """Versucht play_artist API -> faellt zurueck auf IPC. Returns command-string."""
    try:
        if hasattr(sp, "play_artist") and callable(sp.play_artist):
            ok = sp.play_artist(artist)
            if ok:
                return f"play_artist('{artist}')"
    except Exception as e:
        logger.debug("play_artist failed: %s", e)
    if write_ipc_cmd("spotify_play_artist", {"artist": artist}):
        return f"ipc_spotify_play_artist('{artist}')"
    return ""


def verify(timeout_s: int = 15) -> Dict[str, Any]:
    sp = _get_spotify()
    if sp is None:
        return skip_result("spotify_unavailable")

    try:
        baseline_status = sp.get_status()
    except Exception as e:
        return skip_result("get_status_failed", error=str(e)[:120])

    if not isinstance(baseline_status, dict) or not baseline_status.get("initialized"):
        return skip_result("spotify_not_initialized")

    t_start = now()
    baseline = _track_summary(baseline_status)

    cmd = _trigger_play_artist(sp, _TEST_ARTIST)
    if not cmd:
        return fail_result("play_command_failed", baseline=baseline)

    time.sleep(_SLEEP_AFTER_PLAY)

    try:
        after_status = sp.get_status()
    except Exception as e:
        return fail_result(
            "after_get_status_failed",
            error=str(e)[:120],
            baseline=baseline,
            command_sent=cmd,
        )

    after = _track_summary(after_status)

    artist_after = str(after.get("artist") or "").lower()
    track_changed = (
        baseline.get("uri") != after.get("uri")
        or baseline.get("name") != after.get("name")
    )
    artist_match = _TEST_ARTIST.lower() in artist_after

    # Cleanup: vorherigen Track resumen wenn moeglich
    try:
        prev_uri = baseline.get("uri")
        if prev_uri and hasattr(sp, "_sp") and sp._sp is not None:
            try:
                sp._sp.start_playback(uris=[prev_uri])
            except Exception:
                pass
    except Exception:
        pass

    if artist_match:
        status, score = "PASS", 2
    elif track_changed:
        status, score = "WARN", 1
    else:
        status, score = "FAIL", 0

    return {
        "score": score,
        "max": 2,
        "status": status,
        "command_sent": cmd,
        "baseline": baseline,
        "after": after,
        "delta": {
            "track_changed": track_changed,
            "artist_match": artist_match,
        },
        "duration_s": round(now() - t_start, 2),
        "detail": {"target_artist": _TEST_ARTIST, "sleep_s": _SLEEP_AFTER_PLAY},
    }
