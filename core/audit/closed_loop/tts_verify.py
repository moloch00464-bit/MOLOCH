"""TTS Closed-Loop-Verifier — Sprechen -> Mic-Pegel-Spike.

ESP32-Mic-Pegel via UDP-Sample-Stream gemessen (best-effort).
PASS  : Spike >2x Baseline
WARN  : Spike 1.3-2x Baseline
FAIL  : kein Spike (TTS oder Mic kaputt)
SKIP  : keine speak()-API verfuegbar ODER Mic nicht erreichbar
"""
from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from typing import Any, Dict, List, Optional

from ._common import fail_result, now, skip_result

logger = logging.getLogger("tts_verify")

_MIC_UDP_PORT = 12345
_BASELINE_SAMPLES = 10
_BASELINE_INTERVAL = 0.05
_TTS_TEXT = "Audio-Test eins zwei drei"
_SPIKE_WINDOW_S = 3.0


def _sample_amplitude(timeout: float = 0.3) -> Optional[float]:
    """Liest 1 UDP-Paket vom ESP32-Mic und rechnet RMS-Amplitude.

    Format: 16-bit PCM mono @16kHz, ~320 byte/Paket.
    """
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        sock.bind(("0.0.0.0", _MIC_UDP_PORT))
        data, _addr = sock.recvfrom(4096)
        if not data:
            return None
        n = len(data) // 2
        if n == 0:
            return None
        samples = struct.unpack(f"<{n}h", data[: n * 2])
        # RMS
        sq = sum(s * s for s in samples)
        rms = (sq / n) ** 0.5
        return float(rms)
    except (socket.timeout, OSError):
        return None
    except Exception as e:
        logger.debug("amplitude sample failed: %s", e)
        return None
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass


def _measure_baseline() -> Optional[float]:
    samples: List[float] = []
    for _ in range(_BASELINE_SAMPLES):
        a = _sample_amplitude(0.3)
        if a is not None:
            samples.append(a)
        time.sleep(_BASELINE_INTERVAL)
    if not samples:
        return None
    return sum(samples) / len(samples)


def _measure_peak(duration_s: float) -> Optional[float]:
    end = time.time() + duration_s
    peak: Optional[float] = None
    while time.time() < end:
        a = _sample_amplitude(0.3)
        if a is not None:
            if peak is None or a > peak:
                peak = a
    return peak


def _trigger_speak(text: str) -> str:
    """Versucht personality_engine.speak() -> voice_pipeline._speak. Returns command-Beschreibung."""
    try:
        from core.personality.personality_engine import get_personality_engine  # type: ignore
        pe = get_personality_engine()
        ok = pe.speak(text)
        if ok:
            return "personality_engine.speak"
    except Exception as e:
        logger.debug("personality_engine.speak failed: %s", e)
    try:
        from core.voice_pipeline import VoicePipeline  # type: ignore
        # Module-Singleton — versuche __dict__ Lookup
        import core.voice_pipeline as vp_mod  # type: ignore
        for cand in ("get_voice_pipeline", "_voice_pipeline", "voice_pipeline"):
            obj = getattr(vp_mod, cand, None)
            if obj and hasattr(obj, "_speak"):
                obj._speak(text)
                return f"voice_pipeline._speak (via {cand})"
            if callable(obj):
                try:
                    inst = obj()
                    if hasattr(inst, "_speak"):
                        inst._speak(text)
                        return f"voice_pipeline._speak (via {cand}())"
                except Exception:
                    pass
    except Exception as e:
        logger.debug("voice_pipeline path failed: %s", e)
    return ""


def verify(timeout_s: int = 15) -> Dict[str, Any]:
    t_start = now()

    baseline = _measure_baseline()
    if baseline is None:
        return skip_result("mic_unreachable_or_silent_stream")

    cmd = _trigger_speak(_TTS_TEXT)
    if not cmd:
        return skip_result("no_speak_api_available", baseline_rms=round(baseline, 1))

    # Peak im Sprech-Fenster messen (parallel zur TTS-Wiedergabe)
    peak_holder = {"peak": None}

    def _bg():
        peak_holder["peak"] = _measure_peak(_SPIKE_WINDOW_S)

    t = threading.Thread(target=_bg, daemon=True)
    t.start()
    t.join(timeout=_SPIKE_WINDOW_S + 1.0)

    peak = peak_holder.get("peak")
    if peak is None:
        return fail_result(
            "no_peak_during_speak",
            baseline_rms=round(baseline, 1),
            command_sent=cmd,
        )

    ratio = peak / max(baseline, 1.0)

    if ratio >= 2.0:
        status, score = "PASS", 2
    elif ratio >= 1.3:
        status, score = "WARN", 1
    else:
        status, score = "FAIL", 0

    return {
        "score": score,
        "max": 2,
        "status": status,
        "command_sent": cmd,
        "baseline": {"rms_avg": round(baseline, 1)},
        "after": {"rms_peak": round(peak, 1)},
        "delta": {"ratio": round(ratio, 2)},
        "duration_s": round(now() - t_start, 2),
        "detail": {"text": _TTS_TEXT, "spike_window_s": _SPIKE_WINDOW_S},
    }
