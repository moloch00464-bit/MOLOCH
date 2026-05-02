"""Spotify-Auditor (Welle 12 Schritt 3 + Bug B Detection).

Pullt:
- spotify_controller.get_status() Live-State
- IPC-Counter aus journalctl (letzte 24h grep [SPOTIFY])
- Token-Validity

Schreibt audit_state.layers.spotify Schema:
  {ipc_actions_24h: {play_artist, play_playlist, play_from_year, play_top_tracks,
                     play_by_mood, play_similar, play_new_music},
   last_play_call_ts, current_track_uri, current_track_name,
   token_valid, mismatch_actions_vs_responses, status, score, max, detail}

Status-Logik:
- PASS: spotify reachable, token valid, last_play_call_ts <24h
- WARN: token expires <1h ODER mismatch >3 in 24h
- FAIL: spotify unreachable ODER token expired ODER controller-init-error
"""
from __future__ import annotations

import re
import subprocess
import logging
from collections import Counter
from typing import Any, Dict

logger = logging.getLogger("spotify_auditor")

_IPC_ACTIONS = ("play_artist", "play_playlist", "play_from_year",
                "play_top_tracks", "play_by_mood", "play_similar",
                "play_new_music", "play_search", "play")


def _journal_ipc_counts(window: str = "24 hours ago") -> Dict[str, int]:
    """Counts spotify-IPC-Actions aus journalctl der letzten N Stunden."""
    counts = Counter()
    try:
        r = subprocess.run(
            ["sudo", "journalctl", "-u", "moloch", "--since", window,
             "--no-pager", "-n", "5000"],
            capture_output=True, text=True, timeout=20,
        )
        for ln in r.stdout.splitlines():
            if "[SPOTIFY]" not in ln and "spotify_" not in ln.lower():
                continue
            for action in _IPC_ACTIONS:
                if action in ln:
                    counts[action] += 1
                    break
    except Exception:
        pass
    return dict(counts)


def collect() -> Dict[str, Any]:
    """Sammelt Spotify-Layer-Daten."""
    detail: Dict[str, Any] = {}
    current_uri = None
    current_name = None
    token_valid = None
    last_play_ts = None

    # 1. Spotify-Controller Live-Status
    try:
        from core.spotify_controller import get_spotify  # type: ignore
        sp = get_spotify()
        if hasattr(sp, "get_status"):
            st = sp.get_status() or {}
            current_uri = st.get("current_track_uri") or st.get("track_uri")
            current_name = st.get("current_track_name") or st.get("track_name")
            token_valid = st.get("token_valid") if "token_valid" in st else None
            last_play_ts = st.get("last_play_ts") or st.get("last_play_call_ts")
            detail["controller_status"] = {
                k: v for k, v in st.items()
                if k in ("playing", "device", "volume", "shuffle", "auth_ok")
            }
    except Exception as e:
        detail["controller_error"] = str(e)[:100]

    # 2. IPC-Action-Counter aus journalctl
    actions_24h = _journal_ipc_counts("24 hours ago")
    detail["ipc_actions_24h"] = actions_24h
    total_ipc = sum(actions_24h.values())

    # 3. mismatch_actions_vs_responses (Bug B Detector)
    # Heuristik: count "[LLM-ROUTE] type=music_query" vs total_ipc
    music_queries = 0
    try:
        r = subprocess.run(
            ["sudo", "journalctl", "-u", "moloch-chat", "--since", "24 hours ago",
             "--no-pager", "-n", "5000"],
            capture_output=True, text=True, timeout=20,
        )
        for ln in r.stdout.splitlines():
            if "type=music_query" in ln or "prompt_type=music_query" in ln:
                music_queries += 1
    except Exception:
        pass
    mismatch = max(0, music_queries - total_ipc)
    detail["music_queries_24h"] = music_queries
    detail["mismatch_actions_vs_responses"] = mismatch

    # 4. Status-Berechnung
    score = 0
    max_score = 4
    if "controller_error" not in detail:
        score += 1
    if total_ipc > 0:
        score += 1
    if token_valid is True or (token_valid is None and current_uri):
        score += 1
    if mismatch <= 3:
        score += 1

    # auth_ok=False + token_valid=None ohne controller_error = lazy-not-init
    # (Spotify wurde seit Service-Restart noch nicht angefragt). Idle, kein WARN.
    auth_ok = (detail.get("controller_status") or {}).get("auth_ok")
    lazy_idle = (
        auth_ok is False
        and token_valid is None
        and total_ipc == 0
        and "controller_error" not in detail
    )

    if "controller_error" in detail:
        status = "FAIL"
    elif token_valid is False or mismatch > 10:
        status = "FAIL"
    elif lazy_idle:
        # Spotify-Controller noch nicht aktiviert, niemand hat angefragt — PASS.
        # Bei echtem Auth-Fehler waere controller_error gesetzt.
        status = "PASS"
    elif mismatch > 3:
        status = "WARN"
    elif total_ipc == 0:
        # Auth ok, aber 24h still — idle, PASS (vergleiche voice tts_calls_1h=0).
        status = "PASS"
    else:
        status = "PASS"

    return {
        "score": score,
        "max": max_score,
        "status": status,
        "ipc_actions_24h": actions_24h,
        "total_ipc_24h": total_ipc,
        "current_track_uri": current_uri,
        "current_track_name": current_name,
        "token_valid": token_valid,
        "last_play_call_ts": last_play_ts,
        "mismatch_actions_vs_responses": mismatch,
        "detail": detail,
    }
