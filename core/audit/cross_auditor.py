"""Cross-Cutting-Auditor (Welle 14).

Aggregiert system-weite Indikatoren:
- Heartbeat-Inventar: 13 Komponenten (vision, npu, personality, memory, tracking,
  autonomy, awareness, voice, unconscious, bridge, tentacle, music, hardware)
  pruefen ob Layer in audit_state.json vorhanden ODER Modul importierbar
- Resource-Pressure: RAM%, FD-Count, Thread-Count, /tmp + /dev/shm Fuellung
- Latency-Layer: Read time von /dev/shm/moloch_status.json + audit_state.json

Schreibt audit_state.layers.cross Schema:
  {heartbeat_inventory: {component: bool}, ram_pct, fd_count, thread_count,
   tmp_used_mb, shm_used_mb, status_read_ms, score, max, status, detail}

Status-Logik:
- PASS: alle 13 Komponenten alive, RAM <85%, FD <500, threads <100, tmp <50MB
- WARN: ein Schwellwert ueberschritten
- FAIL: RAM >90% ODER >3 Komponenten dead
"""
from __future__ import annotations

import json
import os
import time
import shutil
import subprocess
import importlib
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("cross_auditor")

_STATUS_PATH = "/dev/shm/moloch_status.json"
_AUDIT_STATE = "/dev/shm/audit_state.json"

# Komponenten + Modul-Probes (best-effort import)
_COMPONENTS: Dict[str, str] = {
    "vision": "core.perception.vision_workers",
    "npu": "core.hardware.hailo_manager",
    "personality": "core.personality.personality_engine",
    "memory": "core.longterm_memory",
    "tracking": "core.mpo.autonomous_tracker",
    "autonomy": "core.autonomy.decision_engine",
    "awareness": "core.awareness.activity_analyzer",
    "voice": "core.voice_pipeline",
    "unconscious": "core.unconscious_engine",
    "bridge": "core.bridge.chat_server",
    "tentacle": "core.audio.wifi_mic",
    "music": "core.spotify_controller",
    "hardware": "core.hardware.camera",
}


def _moloch_pid() -> Optional[int]:
    """Findet pid von moloch_service via pgrep."""
    try:
        r = subprocess.run(
            ["pgrep", "-f", "moloch_service"],
            capture_output=True, text=True, timeout=5,
        )
        for ln in r.stdout.splitlines():
            try:
                return int(ln.strip())
            except Exception:
                continue
    except Exception:
        pass
    return None


def _read_audit_state_layers() -> Dict[str, Dict[str, Any]]:
    try:
        with open(_AUDIT_STATE, "r", encoding="utf-8") as f:
            st = json.load(f)
        return st.get("layers", {}) or {}
    except Exception:
        return {}


def _heartbeat_inventory() -> Dict[str, bool]:
    """Pruefe pro Komponente: Layer in audit_state ODER Modul importierbar."""
    layers = _read_audit_state_layers()
    inv: Dict[str, bool] = {}
    for comp, modname in _COMPONENTS.items():
        # 1. Layer in audit_state.json (mit status != FAIL)
        alive = False
        if comp in layers:
            st = layers.get(comp, {}).get("status")
            alive = st in ("PASS", "WARN")
        # 2. Fallback: Modul importierbar
        if not alive:
            try:
                importlib.import_module(modname)
                alive = True
            except Exception:
                alive = False
        inv[comp] = alive
    return inv


def _ram_pct() -> Optional[float]:
    try:
        with open("/proc/meminfo", "r") as f:
            mem = {}
            for ln in f:
                k, _, v = ln.partition(":")
                mem[k.strip()] = v.strip()
        total = float(mem.get("MemTotal", "0 kB").split()[0])
        avail = float(mem.get("MemAvailable", "0 kB").split()[0])
        if total > 0:
            return round((1 - avail / total) * 100, 1)
    except Exception:
        pass
    return None


def _fd_count(pid: int) -> Optional[int]:
    try:
        return len(os.listdir(f"/proc/{pid}/fd"))
    except Exception:
        return None


def _thread_count(pid: int) -> Optional[int]:
    try:
        return len(os.listdir(f"/proc/{pid}/task"))
    except Exception:
        return None


def _dir_used_mb(path: str) -> Optional[float]:
    """du -sb path - returns MB used."""
    try:
        r = subprocess.run(
            ["du", "-sb", path],
            capture_output=True, text=True, timeout=10,
        )
        size = int(r.stdout.split()[0])
        return round(size / (1024 * 1024), 1)
    except Exception:
        return None


def _read_ms(path: str) -> Optional[float]:
    if not os.path.exists(path):
        return None
    try:
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            f.read()
        return round((time.perf_counter() - t0) * 1000, 2)
    except Exception:
        return None


def collect() -> Dict[str, Any]:
    """Sammelt Cross-Cutting-Daten."""
    detail: Dict[str, Any] = {}

    # 1. Heartbeat-Inventar
    inv = _heartbeat_inventory()
    dead: List[str] = [k for k, v in inv.items() if not v]
    detail["dead_components"] = dead

    # 2. Resource-Pressure
    pid = _moloch_pid()
    ram = _ram_pct()
    fd = _fd_count(pid) if pid else None
    threads = _thread_count(pid) if pid else None
    tmp_mb = _dir_used_mb("/tmp")
    shm_mb = _dir_used_mb("/dev/shm")
    detail["moloch_pid"] = pid

    # 3. Latency-Layer
    status_read_ms = _read_ms(_STATUS_PATH)
    audit_read_ms = _read_ms(_AUDIT_STATE)
    detail["status_read_ms"] = status_read_ms
    detail["audit_state_read_ms"] = audit_read_ms

    # Status-Berechnung
    score = 0
    max_score = 5
    alive_count = sum(1 for v in inv.values() if v)
    if alive_count == len(inv):
        score += 1
    if ram is not None and ram < 85:
        score += 1
    if fd is not None and fd < 500:
        score += 1
    if threads is not None and threads < 100:
        score += 1
    if tmp_mb is not None and tmp_mb < 50:
        score += 1

    if (ram is not None and ram > 90) or len(dead) > 3:
        status = "FAIL"
    elif (
        len(dead) > 0
        or (ram is not None and ram >= 85)
        or (fd is not None and fd >= 500)
        or (threads is not None and threads >= 100)
        or (tmp_mb is not None and tmp_mb >= 50)
        or (status_read_ms is not None and status_read_ms > 5)
    ):
        status = "WARN"
    else:
        status = "PASS"

    return {
        "score": score,
        "max": max_score,
        "status": status,
        "heartbeat_inventory": inv,
        "ram_pct": ram,
        "fd_count": fd,
        "thread_count": threads,
        "tmp_used_mb": tmp_mb,
        "shm_used_mb": shm_mb,
        "status_read_ms": status_read_ms,
        "detail": detail,
    }
