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


def ptz_pan(angle: float) -> Dict[str, Any]:
    """Pan-Bewegung in Grad. NEVER 2: Sonoff invertiert, positive=LINKS (Vorzeichen vom Service erwartet)."""
    try:
        ok = _atomic_ipc_cmd("ptz_pan", {"angle": float(angle)})
        return {"ok": ok, "angle": float(angle)}
    except Exception as e:
        return {"error": str(e)[:200]}


def led_set(color: str = "blue") -> Dict[str, Any]:
    """LED-Farbe (red/green/blue/yellow/magenta/cyan/white/off)."""
    try:
        ok = _atomic_ipc_cmd("led_set_color", {"color": str(color)})
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
    """Tilt-Bewegung (-90..+90 Grad)."""
    try:
        ok = _atomic_ipc_cmd("ptz_tilt", {"angle": float(angle)})
        return {"ok": ok, "angle": float(angle)}
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
