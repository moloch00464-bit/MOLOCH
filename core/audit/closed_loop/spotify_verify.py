"""Spotify Closed-Loop-Verifier — IPC play_artist -> Track-Wechsel via state-file.

W18 Cross-Prozess-Fix: liest /dev/shm/moloch_spotify_state.json (vom Service
geschriebener Track-Snapshot) statt SpotifyController-Singleton im Audit-
Subprozess zu instanziieren. Triggert Wechsel via IPC-Cmd 'spotify_artist'.

PASS  : Artist-Match nach Sleep
WARN  : Track wechselte aber falscher Artist
FAIL  : kein Track-Wechsel (Bug B Verifier!)
SKIP  : state-file fehlt / initialized=false / IPC-write fehlgeschlagen
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from ._common import fail_result, now, skip_result, write_ipc_cmd

logger = logging.getLogger("spotify_verify")

_STATE_PATH = "/dev/shm/moloch_spotify_state.json"
_TEST_ARTIST = "Suicide Commando"
_SLEEP_AFTER_PLAY = 5.0


def _read_state() -> Optional[Dict[str, Any]]:
    """Liest Spotify-State-File, None wenn fehlt/kaputt."""
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
        logger.debug("spotify state read failed: %s", e)
        return None


def _track_summary(st: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(st, dict):
        return {}
    ct = st.get("current_track") or {}
    if not isinstance(ct, dict):
        ct = {}
    return {
        "uri": ct.get("uri"),
        "name": ct.get("name"),
        "artist": ct.get("artist"),
        "album": ct.get("album"),
    }


def verify(timeout_s: int = 15) -> Dict[str, Any]:
    t_start = now()

    baseline_st = _read_state()
    if baseline_st is None:
        return skip_result("spotify_state_file_missing", path=_STATE_PATH)
    if not baseline_st.get("initialized", False):
        return skip_result("spotify_not_initialized")

    baseline = _track_summary(baseline_st)

    cmd_payload = {"action": "spotify_artist", "artist": _TEST_ARTIST}
    cmd_str = f"ipc spotify_artist(artist='{_TEST_ARTIST}')"
    sent_ok = write_ipc_cmd("spotify_artist", cmd_payload)
    if not sent_ok:
        return fail_result("ipc_write_failed", command_attempted=cmd_str, baseline=baseline)

    time.sleep(_SLEEP_AFTER_PLAY)

    after_st = _read_state()
    if after_st is None:
        return fail_result(
            "state_file_missing_after_trigger",
            command_sent=cmd_str,
            baseline=baseline,
        )

    after = _track_summary(after_st)

    artist_after = str(after.get("artist") or "").lower()
    track_changed = (
        baseline.get("uri") != after.get("uri")
        or baseline.get("name") != after.get("name")
    )
    artist_match = _TEST_ARTIST.lower() in artist_after

    # Cleanup: vorigen Track best-effort restoren wenn URI bekannt.
    # Cmd-Name 'spotify_play_uri' nicht garantiert -> als best-effort
    # 'spotify_play' mit uri-Param (siehe moloch_service.py:3123).
    try:
        prev_uri = baseline.get("uri")
        if prev_uri:
            write_ipc_cmd("spotify_play", {"action": "spotify_play", "uri": prev_uri})
    except Exception as e:
        logger.debug("spotify cleanup failed: %s", e)

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
        "command_sent": cmd_str,
        "baseline": baseline,
        "after": after,
        "delta": {
            "track_changed": track_changed,
            "artist_match": artist_match,
        },
        "duration_s": round(now() - t_start, 2),
        "detail": {
            "target_artist": _TEST_ARTIST,
            "sleep_s": _SLEEP_AFTER_PLAY,
            "state_path": _STATE_PATH,
            "note": "W18: state-file-read statt Singleton + IPC spotify_artist",
        },
    }


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(verify(), indent=2, ensure_ascii=False))
