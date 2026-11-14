"""W21 Phase 3 #4 Hardware-Tools (ptz_pan, led_set, camera_snapshot)."""
from __future__ import annotations
import json
import logging
import os
import tempfile
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("agent.tools.hardware")

CMD_DIR = "/tmp"
FRAME_PATH = "/dev/shm/moloch_frame"


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


_LED_COLORS = {
    "red": "rot", "green": "gruen", "blue": "blau", "yellow": "gelb",
    "magenta": "magenta", "cyan": "cyan", "white": "weiss", "off": "aus",
}


STATUS_PATH = "/dev/shm/moloch_status.json"
SETTINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "config", "settings.json",
)
STATUS_MAX_AGE_S = 5.0   # aeltere Positionsangabe gilt als unbekannt
MIN_MOVE_DEG = 0.5       # kleinere Reste werden nicht gefahren

# Mechanische Grenzen der Sonoff CAM-PT2 (core/hardware/camera.py PAN_MIN/...).
# camera.move_absolute klemmt darauf ohnehin; hier geht es um die engeren
# Tracker-Grenzen aus settings.json["ptz"] (Popup "Tracker").
_MECH_LIMITS = {"pan": (-168.4, 170.0), "tilt": (-78.0, 78.8)}


def _axis_limits(axis: str) -> tuple:
    """Konfigurierte Grenzen einer Achse, nie weiter als die mechanischen."""
    lo, hi = _MECH_LIMITS[axis]
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            ptz = json.load(f).get("ptz", {}) or {}
        cfg_lo = float(ptz.get(f"{axis}_limit_min", lo))
        cfg_hi = float(ptz.get(f"{axis}_limit_max", hi))
        if cfg_lo < cfg_hi:
            lo, hi = max(lo, cfg_lo), min(hi, cfg_hi)
    except Exception as e:
        logger.debug(f"PTZ-Grenzen nicht lesbar, nutze mechanische: {e}")
    return lo, hi


def _current_position(axis: str) -> Optional[float]:
    """Aktuelle Achsposition in Grad aus dem Service-Status (None = unbekannt)."""
    try:
        if time.time() - os.stat(STATUS_PATH).st_mtime > STATUS_MAX_AGE_S:
            return None
        with open(STATUS_PATH, "r", encoding="utf-8") as f:
            val = (json.load(f).get("ptz", {}) or {}).get(f"current_{axis}")
        return float(val) if val is not None else None
    except Exception:
        return None


def _clamp_delta(axis: str, angle: float) -> Dict[str, Any]:
    """Relativen Winkel so kuerzen, dass das Ziel innerhalb der Grenzen bleibt.

    Positiver Winkel erhoeht den Achswert (Pan: links, Tilt: hoch) - genau wie
    camera.move_manual. Steht die Kamera schon ausserhalb, wird nur noch die
    Bewegung nach innen zugelassen.
    """
    lo, hi = _axis_limits(axis)
    cur = _current_position(axis)
    if cur is None:
        if (lo, hi) == _MECH_LIMITS[axis]:
            # Keine engeren Grenzen konfiguriert: die Kamera klemmt selbst.
            return {"angle": angle, "clamped": False, "position_known": False}
        return {"angle": 0.0, "clamped": True, "position_known": False,
                "note": "Position unbekannt - keine Bewegung, weil engere Grenzen gelten"}
    target = max(lo, min(hi, cur + angle))
    eff = target - cur
    if eff * angle <= 0:
        eff = 0.0    # Grenze erreicht oder Kamera steht ausserhalb und soll weiter raus
    elif abs(eff) > abs(angle):
        eff = angle  # von ausserhalb nach innen: nie weiter als verlangt
    elif abs(eff - angle) > 0.05 and abs(eff) < MIN_MOVE_DEG:
        eff = 0.0    # gekuerzter Rest zu klein zum Fahren
    out: Dict[str, Any] = {"angle": round(eff, 1), "clamped": abs(eff - angle) > 0.05,
                           "position_known": True, "target": round(cur + eff, 1),
                           "limits": [lo, hi]}
    if eff == 0.0:
        out["note"] = "Grenze erreicht"
    return out


def _relative_move(axis: str, pos_dir: str, neg_dir: str, angle: float, limit: float) -> Dict[str, Any]:
    """Relative PTZ-Bewegung ueber die Service-Aktion ptz_move, absolut geklemmt."""
    requested = max(-limit, min(limit, float(angle)))
    if requested == 0:
        return {"ok": True, "angle": 0.0, "note": "keine Bewegung"}
    res = _clamp_delta(axis, requested)
    res["requested"] = requested
    eff = res["angle"]
    if eff == 0:
        res["ok"] = False
        return res
    res["direction"] = pos_dir if eff > 0 else neg_dir
    res["ok"] = _atomic_ipc_cmd("ptz_move", {"direction": res["direction"], "step": abs(eff)})
    return res


def ptz_pan(angle: float) -> Dict[str, Any]:
    """Relative Pan-Bewegung in Grad. NEVER 2: positive=LINKS (camera.move_manual 'left' = pan+)."""
    try:
        return _relative_move("pan", "left", "right", angle, 180.0)
    except Exception as e:
        return {"error": str(e)[:200]}


def led_set(color: str = "blue") -> Dict[str, Any]:
    """LED-Farbe (red/green/blue/yellow/magenta/cyan/white/off)."""
    try:
        farbe = _LED_COLORS.get(str(color).lower())
        if farbe is None:
            return {"error": f"unbekannte Farbe {color!r}", "allowed": sorted(_LED_COLORS)}
        ok = _atomic_ipc_cmd("led_set", {"farbe": farbe, "modus": "statisch"})
        return {"ok": ok, "color": str(color)}
    except Exception as e:
        return {"error": str(e)[:200]}


def camera_snapshot() -> Dict[str, Any]:
    """Snapshot-Info — Pfad + Groesse + Alter (visueller Frame in /dev/shm)."""
    try:
        st = os.stat(FRAME_PATH)
        return {
            "path": FRAME_PATH,
            "size_kb": round(st.st_size / 1024.0, 1),
            "age_s": round(time.time() - st.st_mtime, 1),
            "exists": True,
        }
    except Exception as e:
        return {"exists": False, "error": str(e)[:200]}


def ptz_tilt(angle: float) -> Dict[str, Any]:
    """Relative Tilt-Bewegung (-90..+90 Grad), positive=hoch."""
    try:
        return _relative_move("tilt", "up", "down", angle, 90.0)
    except Exception as e:
        return {"error": str(e)[:200]}


def thermal_set_tension_pwm(percent: int) -> Dict[str, Any]:
    """W16: Tension-PWM 0..100 (Luefter-Boost). Best-effort via thermal_manager."""
    try:
        from core.hardware.thermal_manager import get_thermal_manager
        get_thermal_manager().set_tension_pwm(int(percent))
        return {"ok": True, "pwm_pct": int(percent)}
    except Exception as e:
        return {"error": str(e)[:200]}


def get_face_id() -> Dict[str, Any]:
    """Aktuell erkannte Person aus moloch_status.json."""
    try:
        with open("/dev/shm/moloch_status.json") as f:
            st = json.load(f)
        return {
            "face_id": st.get("face_id"),
            "person": bool(st.get("person")),
            "face_match": st.get("face_match"),
            "face_detect": st.get("face_detect"),
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def get_npu_status() -> Dict[str, Any]:
    """NPU-Worker-Health aus moloch_status.json."""
    try:
        with open("/dev/shm/moloch_status.json") as f:
            st = json.load(f)
        wh = st.get("worker_health", {})
        active = st.get("active_models", [])
        out: Dict[str, Any] = {"active_models": active, "workers": {}}
        for name, w in (wh or {}).items():
            if isinstance(w, dict):
                out["workers"][name] = {
                    "running": w.get("running"),
                    "loaded": w.get("models_loaded"),
                    "inferences": w.get("total_inferences"),
                    "errors": w.get("total_errors"),
                    "last_ms": w.get("last_inference_ms"),
                }
        return out
    except Exception as e:
        return {"error": str(e)[:200]}
