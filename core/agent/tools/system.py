"""W21 System/Audit-Tools fuer Cloud-Orchestrator."""
from __future__ import annotations
import json
import logging
import os
import tempfile
import time
from typing import Any, Dict

logger = logging.getLogger("agent.tools.system")
CMD_DIR = "/tmp"


def _atomic_ipc_cmd(action: str, params: dict) -> bool:
    """Atomic IPC-Cmd-Write (NEVER 6: tempfile + os.replace)."""
    cmd: Dict[str, Any] = {"action": action, **params, "ts": time.time()}
    fd, tmp = tempfile.mkstemp(
        dir=CMD_DIR, prefix=f"moloch_cmd_{action}_", suffix=".json.tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cmd, f)
        os.replace(tmp, tmp.replace(".tmp", ""))
        return True
    except Exception as e:
        logger.warning(f"_atomic_ipc_cmd {action} fail: {e}")
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


def get_audit_state() -> Dict[str, Any]:
    """Liefert audit_state.json compact (nur status+overall+layer_count)."""
    try:
        with open("/dev/shm/audit_state.json") as f:
            s = json.load(f)
        layers = s.get("layers", {})
        st_count: Dict[str, int] = {}
        for v in layers.values():
            if isinstance(v, dict):
                st = v.get("status", "-")
                st_count[st] = st_count.get(st, 0) + 1
        return {
            "overall": s.get("overall"),
            "alarm_tier": s.get("alarm_tier"),
            "layer_count": len(layers),
            "status_counts": st_count,
            "updated_at": s.get("updated_at"),
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def moloch_status_summary() -> Dict[str, Any]:
    """Compact MOLOCH-Status: FPS, Person, Face-ID, RAM, CPU-Temp."""
    try:
        with open("/dev/shm/moloch_status.json") as f:
            st = json.load(f)
        wd = st.get("watchdog", {})
        fps = st.get("fps", {})
        return {
            "fps": fps.get("total"),
            "person": bool(st.get("person")),
            "face_id": st.get("face_id"),
            "ram_pct": wd.get("ram_percent"),
            "cpu_temp_c": wd.get("cpu_temp"),
            "frame_age_s": st.get("frame_age"),
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def read_memory(query: str, limit: int = 5) -> Dict[str, Any]:
    """Memory-Recall-Best-Effort via longterm_memory."""
    try:
        from core.longterm_memory import get_memory
        m = get_memory()
        for fn_name in ["recall", "search", "query"]:
            fn = getattr(m, fn_name, None)
            if callable(fn):
                try:
                    res = fn(query, limit=limit)
                    return {
                        "results": res if isinstance(res, list) else [res],
                        "method": fn_name,
                    }
                except TypeError:
                    try:
                        res = fn(query)
                        return {
                            "results": res if isinstance(res, list) else [res],
                            "method": fn_name,
                        }
                    except Exception:
                        continue
        return {"results": [], "method": "no_compatible_api"}
    except Exception as e:
        return {"error": str(e)[:200]}


def tts_say(text: str) -> Dict[str, Any]:
    """Triggert TTS via IPC speak-Cmd (W18.1)."""
    try:
        ok = _atomic_ipc_cmd("speak", {"text": str(text)[:500]})
        return {"ok": ok, "text_chars": len(str(text))}
    except Exception as e:
        return {"error": str(e)[:200]}
