"""NPU-Sehen-Tools: read_text (OCR auf dem aktuellen Kamerabild).

Ruft die Service-Aktion npu_ocr per IPC auf und liest das Ergebnis aus
/dev/shm/moloch_npu_result_<request_id>.json (npu_extras.run_request).
Die NPU gehoert dem Service-Prozess - hier kein eigenes VDevice.
"""
from __future__ import annotations
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Dict

from .hardware import _atomic_ipc_cmd

logger = logging.getLogger("agent.tools.vision")

RESULT_PREFIX = "/dev/shm/moloch_npu_result_"
READ_TEXT_TIMEOUT_S = 15.0  # erster Aufruf laedt die OCR-Modelle (~1 s), danach ~60 ms

_lock = threading.Lock()  # nur ein NPU-Abruf gleichzeitig


def _npu_request(action: str, timeout_s: float) -> Dict[str, Any]:
    """IPC-Aktion ausloesen und auf die Ergebnisdatei warten."""
    if not _lock.acquire(blocking=False):
        return {"error": "NPU-Abruf laeuft bereits, gleich nochmal versuchen"}
    rid = uuid.uuid4().hex[:12]
    path = f"{RESULT_PREFIX}{rid}.json"
    try:
        if not _atomic_ipc_cmd(action, {"request_id": rid}):
            return {"error": "IPC-Kommando konnte nicht geschrieben werden"}
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
            time.sleep(0.05)
        return {"error": f"keine Antwort vom Service nach {timeout_s:.0f}s"}
    except Exception as e:
        return {"error": str(e)[:200]}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
        _lock.release()


def read_text() -> Dict[str, Any]:
    """Text im aktuellen Kamerabild lesen (OCR auf der NPU)."""
    res = _npu_request("npu_ocr", READ_TEXT_TIMEOUT_S)
    if res.get("error"):
        return {"error": f"NPU kann gerade nicht lesen: {res['error']}"}
    return {
        "texts": res.get("texts", []),
        "duration_ms": res.get("duration_ms"),
        "frame_seq": res.get("frame_seq"),
    }
