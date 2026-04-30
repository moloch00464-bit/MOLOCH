"""TTS Closed-Loop-Verifier — IPC speak() -> Mic-Pegel-Spike via state-file.

W18 Cross-Prozess-Fix: liest /dev/shm/moloch_audio_pegel.json (vom Service
geschriebener Pegel-Snapshot) statt UDP-Mic im Audit-Subprozess zu binden.
Triggert TTS via IPC-Cmd statt personality_engine.speak()-Singleton-Import.

PASS  : Spike >+6 dB ueber Baseline
WARN  : Spike >+2.3 dB ueber Baseline
FAIL  : kein Spike (TTS oder Pegel-Stream kaputt)
SKIP  : state-file fehlt / available=false / Pegel-Stream stale (>5s)
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from ._common import fail_result, now, skip_result, write_ipc_cmd

logger = logging.getLogger("tts_verify")

_STATE_PATH = "/dev/shm/moloch_audio_pegel.json"
_TTS_TEXT = "Audio-Test eins zwei drei"
_BASELINE_SAMPLES = 3
_BASELINE_INTERVAL = 1.0
_SPIKE_WINDOW_S = 3.0
_SPIKE_POLL_S = 0.3
_MAX_PACKET_AGE_S = 5.0


def _read_state() -> Optional[Dict[str, Any]]:
    """Liest Audio-Pegel-State-File, None wenn fehlt/kaputt."""
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
        logger.debug("audio_pegel state read failed: %s", e)
        return None


def _state_usable(st: Optional[Dict[str, Any]]) -> bool:
    """Prueft ob state-file Pegel-Daten in Echtzeit liefert."""
    if not isinstance(st, dict):
        return False
    if not st.get("available", False):
        return False
    age = st.get("last_packet_age_s")
    try:
        if age is not None and float(age) > _MAX_PACKET_AGE_S:
            return False
    except (TypeError, ValueError):
        return False
    return True


def _read_rms_db(st: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(st, dict):
        return None
    val = st.get("rms_db")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _measure_baseline() -> Optional[float]:
    """3 Samples im 1s-Abstand, gibt Mittelwert von rms_db zurueck."""
    samples: List[float] = []
    for i in range(_BASELINE_SAMPLES):
        st = _read_state()
        if not _state_usable(st):
            return None
        v = _read_rms_db(st)
        if v is not None:
            samples.append(v)
        if i < _BASELINE_SAMPLES - 1:
            time.sleep(_BASELINE_INTERVAL)
    if not samples:
        return None
    return sum(samples) / len(samples)


def _measure_peak(duration_s: float) -> Optional[float]:
    """Pollt rms_db alle 0.3s und gibt Maximum waehrend Sprech-Fenster."""
    end = time.time() + duration_s
    peak: Optional[float] = None
    while time.time() < end:
        st = _read_state()
        v = _read_rms_db(st)
        if v is not None:
            if peak is None or v > peak:
                peak = v
        time.sleep(_SPIKE_POLL_S)
    return peak


def verify(timeout_s: int = 15) -> Dict[str, Any]:
    t_start = now()

    # Pre-Check: state-file ueberhaupt brauchbar?
    pre = _read_state()
    if pre is None:
        return skip_result("audio_pegel_state_file_missing", path=_STATE_PATH)
    if not pre.get("available", False):
        return skip_result("audio_pegel_unavailable", available=False)
    age_pre = pre.get("last_packet_age_s")
    try:
        if age_pre is not None and float(age_pre) > _MAX_PACKET_AGE_S:
            return skip_result(
                "audio_pegel_stream_stale",
                last_packet_age_s=round(float(age_pre), 2),
                threshold_s=_MAX_PACKET_AGE_S,
            )
    except (TypeError, ValueError):
        return skip_result("audio_pegel_invalid_age", raw_age=age_pre)

    baseline_db = _measure_baseline()
    if baseline_db is None:
        return skip_result("baseline_unreachable_or_silent_stream")

    cmd_payload = {"action": "speak", "text": _TTS_TEXT}
    cmd_str = f"ipc speak(text='{_TTS_TEXT}')"
    sent_ok = write_ipc_cmd("speak", cmd_payload)
    if not sent_ok:
        return fail_result(
            "ipc_write_failed",
            command_attempted=cmd_str,
            baseline={"rms_db_avg": round(baseline_db, 2)},
        )

    peak_db = _measure_peak(_SPIKE_WINDOW_S)
    if peak_db is None:
        return fail_result(
            "no_peak_during_speak",
            command_sent=cmd_str,
            baseline={"rms_db_avg": round(baseline_db, 2)},
        )

    spike_db = peak_db - baseline_db

    if spike_db >= 6.0:
        status, score = "PASS", 2
    elif spike_db >= 2.3:
        status, score = "WARN", 1
    else:
        status, score = "FAIL", 0

    return {
        "score": score,
        "max": 2,
        "status": status,
        "command_sent": cmd_str,
        "baseline": {"rms_db_avg": round(baseline_db, 2)},
        "after": {"rms_db_peak": round(peak_db, 2)},
        "delta": {"spike_db": round(spike_db, 2)},
        "duration_s": round(now() - t_start, 2),
        "detail": {
            "text": _TTS_TEXT,
            "spike_window_s": _SPIKE_WINDOW_S,
            "state_path": _STATE_PATH,
            "thresholds_db": {"pass": 6.0, "warn": 2.3},
            "note": "W18: state-file-read statt UDP-Bind + IPC speak statt Singleton",
        },
    }


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(verify(), indent=2, ensure_ascii=False))
