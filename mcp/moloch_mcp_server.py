#!/usr/bin/env python3
"""
M.O.L.O.C.H. MCP Server
========================
Gibt Claude Code direkten Zugriff auf MOLOCH Live-Daten.

Tools:
  moloch_status()         — Live System-Status (FPS, Temp, Face-ID, NPU)
  moloch_logs(n, filter)  — Letzte N Zeilen journalctl
  moloch_snapshot()       — Kamera-Frame aus SHM als Base64-PNG
  moloch_service(action)  — start/stop/restart/status
  moloch_audit()          — Vollständiger Audit-Lauf
  moloch_read(path)       — Config/Log-Datei lesen (nur erlaubte Pfade)
  moloch_git_log(n)       — Letzte N Commits
  moloch_dmesg()          — Letzte dmesg Zeilen (NPU/GStreamer Fehler)

Start: python3 ~/moloch/mcp/moloch_mcp_server.py
Config: .mcp.json im Moloch-Verzeichnis
"""

import json
import os
import subprocess
import struct
import mmap
import tempfile
from pathlib import Path
from mcp.server.fastmcp import FastMCP

MOLOCH_DIR = Path("/home/molochzuhause/moloch")
STATUS_SHM = "/dev/shm/moloch_status.json"
FRAME_SHM = "/dev/shm/moloch_frame"

# Erlaubte Pfade fuer moloch_read (Sicherheit)
ALLOWED_READ_PREFIXES = [
    "/home/molochzuhause/moloch/",
    "/mnt/moloch-data/memory/",
    "/etc/systemd/system/moloch",
    "/dev/shm/moloch",
]

mcp = FastMCP("moloch")


@mcp.tool()
def moloch_status() -> str:
    """Live MOLOCH System-Status: FPS, CPU-Temp, RAM, Face-ID, NPU-Szenario, Tracking."""
    try:
        with open(STATUS_SHM, "r") as f:
            data = json.load(f)
    except Exception as e:
        return f"FEHLER: Status-JSON nicht lesbar: {e}\nService läuft wahrscheinlich nicht."

    fps = data.get("fps", {})
    ptz = data.get("ptz", {})
    lines = [
        "=== MOLOCH LIVE STATUS ===",
        f"FPS total:    {fps.get('total', 0):.1f}",
        f"FPS yolov8m:  {fps.get('yolov8m', 0):.1f}",
        f"FPS scrfd:    {fps.get('scrfd', 0):.1f}",
        f"FPS arcface:  {fps.get('arcface', 0):.1f}",
        f"FPS pose:     {fps.get('pose', 0):.1f}",
        "",
        f"CPU Temp:     {data.get('cpu_temp', 0):.1f}°C",
        f"RAM:          {data.get('ram_used_mb', 0):.0f} MB",
        f"Frame Age:    {data.get('frame_age', 0):.2f}s",
        "",
        f"Person:       {data.get('person_detected', False)}",
        f"Face-ID:      {data.get('face_id', 'none')}",
        f"Face-Conf:    {data.get('face_confidence', 0):.2f}",
        "",
        f"NPU Szenario: {data.get('npu_sched_mode', 'UNBEKANNT')}",
        f"NPU Stage:    {data.get('npu_stage', '?')}",
        f"Tracker:      {ptz.get('tracker_state', '?')}",
        f"PTZ Modus:    {ptz.get('mode', '?')}",
        "",
        f"Active Models: {', '.join(data.get('active_models', []))}",
    ]
    return "\n".join(lines)


@mcp.tool()
def moloch_logs(n: int = 50, filter_str: str = "") -> str:
    """Letzte N Zeilen MOLOCH Service-Logs aus journalctl.

    Args:
        n: Anzahl Zeilen (default 50)
        filter_str: Optionaler Grep-Filter (z.B. 'FACE-MATCH', 'ERROR', 'SEGV')
    """
    cmd = ["journalctl", "-u", "moloch", "--no-pager", "-n", str(min(n, 500))]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = result.stdout
        if filter_str:
            lines = [l for l in output.splitlines() if filter_str.lower() in l.lower()]
            output = "\n".join(lines) if lines else f"Keine Zeilen mit Filter '{filter_str}'"
        return output or "Keine Logs gefunden."
    except Exception as e:
        return f"FEHLER: {e}"


@mcp.tool()
def moloch_snapshot() -> str:
    """Aktueller Kamera-Frame als Base64-PNG.

    Liest Frame aus /dev/shm/moloch_frame (SHM).
    Gibt Base64-enkodiertes PNG zurück.
    Nur verfügbar wenn MOLOCH Service läuft.
    """
    try:
        import numpy as np
        import cv2

        fd = os.open(FRAME_SHM, os.O_RDONLY)
        size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)

        h, w, c, seq, ts = struct.unpack('<IIIId', mm[:24])
        if h == 0 or w == 0:
            mm.close()
            os.close(fd)
            return "FEHLER: Frame leer (Service gestartet? Kamera verbunden?)"

        data = np.frombuffer(mm[24:24 + h * w * c], dtype=np.uint8).reshape(h, w, c)
        frame = cv2.resize(data, (640, 360), interpolation=cv2.INTER_AREA)
        mm.close()
        os.close(fd)

        out_path = "/tmp/moloch_snapshot.jpg"
        ok = cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return "FEHLER: JPEG-Encoding fehlgeschlagen"

        return f"Snapshot gespeichert: {out_path} [{w}x{h}px, seq={seq}, ts={ts:.2f}] — Lies die Datei mit Read-Tool um das Bild zu sehen."

    except ImportError:
        return "FEHLER: numpy/cv2 nicht verfügbar"
    except FileNotFoundError:
        return f"FEHLER: {FRAME_SHM} nicht gefunden — Service läuft nicht"
    except Exception as e:
        return f"FEHLER: {e}"


@mcp.tool()
def moloch_service(action: str) -> str:
    """MOLOCH Service steuern.

    Args:
        action: 'status', 'start', 'stop', 'restart'
    """
    action = action.lower().strip()
    if action not in ("status", "start", "stop", "restart"):
        return f"Ungültige Aktion '{action}'. Erlaubt: status, start, stop, restart"

    try:
        if action == "status":
            r = subprocess.run(
                ["systemctl", "status", "moloch", "--no-pager", "-l"],
                capture_output=True, text=True, timeout=10
            )
            return r.stdout + r.stderr
        else:
            r = subprocess.run(
                ["sudo", "systemctl", action, "moloch"],
                capture_output=True, text=True, timeout=30
            )
            if r.returncode == 0:
                return f"OK: Service {action} erfolgreich"
            else:
                return f"FEHLER: {r.stderr or r.stdout}"
    except Exception as e:
        return f"FEHLER: {e}"


@mcp.tool()
def moloch_audit() -> str:
    """Vollständiger MOLOCH System-Audit (39 Tests).

    Dauert ~30 Sekunden. Gibt PASS/FAIL/WARN für alle Subsysteme.
    """
    audit_script = MOLOCH_DIR / "moloch_audit.py"
    if not audit_script.exists():
        return f"FEHLER: {audit_script} nicht gefunden"

    try:
        r = subprocess.run(
            ["python3", str(audit_script), "--auto"],
            capture_output=True, text=True, timeout=120,
            cwd=str(MOLOCH_DIR)
        )
        output = r.stdout
        if r.stderr:
            output += "\n--- STDERR ---\n" + r.stderr
        return output or "Kein Output"
    except subprocess.TimeoutExpired:
        return "TIMEOUT: Audit dauerte >120s"
    except Exception as e:
        return f"FEHLER: {e}"


@mcp.tool()
def moloch_read(path: str, lines: int = 200) -> str:
    """Config/Log-Datei lesen (nur erlaubte MOLOCH-Pfade).

    Args:
        path: Absoluter Pfad (z.B. /home/molochzuhause/moloch/config/system_capabilities.json)
        lines: Max. Zeilen (default 200)
    """
    # Sicherheits-Check
    allowed = any(path.startswith(p) for p in ALLOWED_READ_PREFIXES)
    if not allowed:
        return f"VERWEIGERT: Pfad '{path}' nicht in erlaubten Verzeichnissen.\nErlaubt: {ALLOWED_READ_PREFIXES}"

    try:
        with open(path, "r") as f:
            content = f.readlines()
        if len(content) > lines:
            content = content[:lines]
            content.append(f"\n[... {len(content)} von {lines} Zeilen gezeigt ...]")
        return "".join(content)
    except FileNotFoundError:
        return f"FEHLER: Datei nicht gefunden: {path}"
    except Exception as e:
        return f"FEHLER: {e}"


@mcp.tool()
def moloch_git_log(n: int = 10) -> str:
    """Letzte N Git-Commits im MOLOCH Repository.

    Args:
        n: Anzahl Commits (default 10)
    """
    try:
        r = subprocess.run(
            ["git", "log", "--oneline", f"-{min(n, 50)}"],
            capture_output=True, text=True, timeout=10,
            cwd=str(MOLOCH_DIR)
        )
        return r.stdout or "Keine Commits gefunden"
    except Exception as e:
        return f"FEHLER: {e}"


@mcp.tool()
def moloch_dmesg() -> str:
    """Letzte dmesg-Einträge — fokussiert auf NPU/Hailo, GStreamer, SEGV, GPU.

    Zeigt Fehler der letzten 10 Minuten.
    """
    try:
        r = subprocess.run(
            ["dmesg", "--since", "-10min", "--level", "err,warn,crit"],
            capture_output=True, text=True, timeout=10
        )
        output = r.stdout.strip()

        # Zusätzlich: Hailo/GStreamer Einträge
        r2 = subprocess.run(
            ["dmesg", "--since", "-10min"],
            capture_output=True, text=True, timeout=10
        )
        hailo_lines = [
            l for l in r2.stdout.splitlines()
            if any(k in l.lower() for k in ["hailo", "gstreamer", "segfault", "oom", "killed"])
        ]

        result = []
        if output:
            result.append("=== ERR/WARN ===\n" + output)
        if hailo_lines:
            result.append("=== HAILO/GST ===\n" + "\n".join(hailo_lines))
        return "\n\n".join(result) if result else "Keine relevanten Einträge in den letzten 10 Minuten."
    except Exception as e:
        return f"FEHLER: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
