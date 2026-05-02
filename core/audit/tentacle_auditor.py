"""Tentacle-Auditor (Welle 14) — ESP32 ReSpeaker Mic.

Pullt:
- L0: ss -ulnp | grep 12345 (UDP-Listener fuer Mic-Audio aktiv)
- L1: ping ESP32 (Default 10.42.0.2)
- L2: RSSI + last-frame aus /dev/shm/moloch_status.json.audio.wifi_mic
      ODER aus core.audio.wifi_mic Singleton

Schreibt audit_state.layers.tentacle Schema:
  {udp_listener_active, esp32_ping_ms, esp32_rssi, last_frame_age_s,
   score, max, status, detail}

Status-Logik:
- PASS: UDP listener + ESP32 ping <50ms + RSSI >-70dBm
- WARN: ping ok aber RSSI <-75 ODER last_frame stale
- FAIL: UDP listener weg ODER ping timeout
"""
from __future__ import annotations

import json
import os
import time
import socket
import subprocess
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("tentacle_auditor")

_STATUS_PATH = "/dev/shm/moloch_status.json"
_DEFAULT_ESP32_IP = "10.42.0.2"
_UDP_PORT = 12345


def _udp_listener_active(port: int = _UDP_PORT) -> bool:
    """Prueft via ss ob UDP-Port aktiv lauscht."""
    try:
        r = subprocess.run(
            ["ss", "-ulnp"],
            capture_output=True, text=True, timeout=5,
        )
        for ln in r.stdout.splitlines():
            # Format: "UNCONN 0  0  0.0.0.0:12345  ..."
            if f":{port}" in ln:
                return True
        return False
    except Exception:
        return False


def _ping_ms(host: str, count: int = 1, timeout: int = 5) -> Optional[float]:
    try:
        r = subprocess.run(
            ["ping", "-c", str(count), "-W", "1", host],
            capture_output=True, text=True, timeout=timeout,
        )
        for ln in r.stdout.splitlines():
            if "time=" in ln:
                try:
                    return float(ln.split("time=")[-1].split()[0])
                except Exception:
                    continue
            if "avg" in ln and "/" in ln:
                try:
                    parts = ln.split("=")[-1].strip().split()[0].split("/")
                    return float(parts[1]) if len(parts) >= 2 else None
                except Exception:
                    continue
    except Exception:
        return None
    return None


def _esp32_ip_from_settings() -> str:
    """Liest wifi_mic.ip aus settings.json, default 10.42.0.2."""
    try:
        with open("/home/molochzuhause/moloch/config/settings.json", "r", encoding="utf-8") as f:
            s = json.load(f)
        ip = s.get("audio", {}).get("wifi_mic_ip") or s.get("wifi_mic", {}).get("ip")
        if ip:
            return str(ip)
    except Exception:
        pass
    return _DEFAULT_ESP32_IP


def _wifi_mic_status_from_shm() -> Dict[str, Any]:
    """Liest audio.wifi_mic-Block aus moloch_status.json (best-effort)."""
    try:
        with open(_STATUS_PATH, "r", encoding="utf-8") as f:
            st = json.load(f)
        audio = st.get("audio", {}) or {}
        mic = audio.get("wifi_mic") or audio.get("respeaker") or {}
        return mic if isinstance(mic, dict) else {}
    except Exception:
        return {}


def _wifi_mic_status_from_singleton() -> Dict[str, Any]:
    try:
        from core.audio.wifi_mic import get_wifi_mic  # type: ignore
        m = get_wifi_mic()
        if m is None:
            return {}
        out: Dict[str, Any] = {}
        for attr in ("rssi", "last_frame_ts", "last_packet_ts",
                     "frames_received", "running"):
            if hasattr(m, attr):
                try:
                    v = getattr(m, attr)
                    out[attr] = v() if callable(v) else v
                except Exception:
                    pass
        if hasattr(m, "get_status"):
            try:
                st = m.get_status()
                if isinstance(st, dict):
                    out.update(st)
            except Exception:
                pass
        return out
    except Exception:
        return {}


def collect() -> Dict[str, Any]:
    """Sammelt Tentacle-Layer-Daten."""
    detail: Dict[str, Any] = {}

    # L0: UDP-Listener
    udp_ok = _udp_listener_active(_UDP_PORT)
    detail["udp_port"] = _UDP_PORT

    # L1: Ping ESP32
    esp32_ip = _esp32_ip_from_settings()
    detail["esp32_ip"] = esp32_ip
    ping_ms = _ping_ms(esp32_ip)

    # L2: RSSI + last_frame
    rssi: Optional[float] = None
    last_frame_age: Optional[float] = None
    src = "shm"
    mic_status = _wifi_mic_status_from_shm()
    if not mic_status:
        mic_status = _wifi_mic_status_from_singleton()
        src = "singleton" if mic_status else "none"
    detail["status_source"] = src

    if mic_status:
        try:
            r = mic_status.get("rssi")
            rssi = float(r) if r is not None else None
        except Exception:
            rssi = None
        for key in ("last_frame_age_s", "last_packet_age_s"):
            if key in mic_status and mic_status[key] is not None:
                try:
                    last_frame_age = float(mic_status[key])
                    break
                except Exception:
                    continue
        if last_frame_age is None:
            for key in ("last_frame_ts", "last_packet_ts"):
                if key in mic_status and mic_status[key]:
                    try:
                        last_frame_age = max(0.0, time.time() - float(mic_status[key]))
                        break
                    except Exception:
                        continue
        detail["mic_running"] = mic_status.get("running")
        detail["frames_received"] = mic_status.get("frames_received")

    # Status-Berechnung
    score = 0
    max_score = 4
    if udp_ok:
        score += 1
    if ping_ms is not None and ping_ms < 50:
        score += 1
    if rssi is not None and rssi > -70:
        score += 1
    if last_frame_age is not None and last_frame_age < 5:
        score += 1

    # Pi-Side vs ESP32-Side trennen:
    # - Pi-Side ready (udp_listener) ist die Pflicht. udp_listener weg = FAIL.
    # - ESP32-Outage (ping fail) ist external — WARN, nicht FAIL. Pi kann
    #   ESP32-Reboot nicht erzwingen, aber sobald ESP32 wieder online ist,
    #   wird Audio empfangen.
    if not udp_ok:
        status = "FAIL"
    elif ping_ms is None:
        # ESP32 nicht erreichbar — externe Outage, Pi-Side OK.
        status = "WARN"
    elif rssi is not None and rssi < -75:
        status = "WARN"
    elif last_frame_age is not None and last_frame_age > 10:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "score": score,
        "max": max_score,
        "status": status,
        "udp_listener_active": udp_ok,
        "esp32_ping_ms": round(ping_ms, 1) if ping_ms is not None else None,
        "esp32_rssi": rssi,
        "last_frame_age_s": round(last_frame_age, 2) if last_frame_age is not None else None,
        "detail": detail,
    }
