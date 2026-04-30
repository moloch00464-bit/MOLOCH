"""MOLOCH PC-Hardware-Auditor (Welle 12, PC-Side).

Probt PC-eigene Hardware + Services + Resource-Pressure.
POSTet Befund als layer.pc_hardware an Pi audit-Orchestrator.

Geprueft:
  webcam_devices    USB-Kamera-Devices (Markus' Webcam falls vorhanden)
  audio_devices     Input/Output-Devices (Mikrofon, Lautsprecher)
  disk_free_gb      C:/ freier Speicher
  ollama_ram_mb     Ollama-Process Memory-Footprint
  cpu_load_pct      Aktuelle CPU-Last (5s avg)
  ram_pct           PC RAM-Auslastung
  gpu_status        GTX 760 - via wmic Win32_VideoController
  pi_reachable      Pi-Ping (192.168.178.30) latency_ms
  ollama_models     count + size in MB

POSTet alle 5 Min. Reboot-Persistent via Startup-Folder.

CLI:
  python pc/hardware_auditor.py --once
  python pc/hardware_auditor.py --interval-s 300
  python pc/hardware_auditor.py --json

NEVER-Regeln:
- subprocess timeout=30 (NEVER 5)
- atomic state-write (NEVER 6)
- KEIN shell=True (NEVER 8)
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pc-hardware-auditor")

PI_BASE = os.environ.get("MOLOCH_PI_CHAT", "http://192.168.178.30:9100")
PI_HOST = os.environ.get("MOLOCH_PI_HOST", "192.168.178.30")
DEFAULT_INTERVAL_S = 300
TIMEOUT_S = 8
HEADERS = {"Content-Type": "application/json"}
STATE_DIR = Path.home() / "moloch_logs" / "audit"
STATE_DIR.mkdir(parents=True, exist_ok=True)
OLLAMA_EXE = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"


def safe_json_write(path: Path, data: dict) -> None:
    """Atomic write — NEVER-Regel 6."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, str(path))
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def run_ps(cmd: str, timeout: int = 15) -> str:
    """PowerShell ohne shell=True. Robust gegen Threading-Bugs in Python 3.13."""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=timeout, encoding="utf-8", errors="replace",
        )
        return (r.stdout or "").strip()
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug(f"[ps] {e}")
        return ""


def probe_webcam_devices() -> dict:
    """USB-Kamera via WMI."""
    out = run_ps("Get-PnpDevice -PresentOnly -Class Camera,Image | Select -ExpandProperty FriendlyName")
    devices = [l.strip() for l in out.splitlines() if l.strip()]
    return {"count": len(devices), "names": devices[:5]}


def probe_audio_devices() -> dict:
    """Audio-Input/Output via WMI."""
    out = run_ps("Get-CimInstance Win32_SoundDevice | Where-Object {$_.StatusInfo -ne 5} | Select -ExpandProperty Name")
    devices = [l.strip() for l in out.splitlines() if l.strip()]
    return {"count": len(devices), "names": devices[:5]}


def probe_disk() -> dict:
    """C:/ free GB."""
    try:
        total, used, free = shutil.disk_usage("C:/")
        return {
            "free_gb": round(free / (1024**3), 1),
            "total_gb": round(total / (1024**3), 1),
            "pct_used": round(100 * used / total, 1),
        }
    except OSError as e:
        return {"error": str(e)[:60]}


def probe_ollama() -> dict:
    """Ollama process + RAM + Modelle."""
    info: dict = {}
    # Process RAM
    out = run_ps("(Get-Process ollama* -ErrorAction SilentlyContinue | Measure WorkingSet -Sum).Sum")
    try:
        info["ram_mb"] = round(int(out) / (1024**2), 1) if out else 0
    except (ValueError, TypeError):
        info["ram_mb"] = 0
    # Modelle
    if OLLAMA_EXE.exists():
        try:
            r = subprocess.run(
                [str(OLLAMA_EXE), "list"],
                capture_output=True, text=True, timeout=10,
            )
            lines = r.stdout.strip().split("\n")[1:]  # skip header
            info["models_count"] = len([l for l in lines if l.strip()])
            info["models"] = [l.split()[0] for l in lines if l.strip()][:10]
        except (subprocess.TimeoutExpired, OSError) as e:
            info["error"] = str(e)[:60]
    else:
        info["error"] = "ollama.exe nicht gefunden"
    return info


def probe_cpu_ram() -> dict:
    """CPU-Load + RAM-Pct."""
    cpu_out = run_ps("(Get-Counter '\\Processor(_Total)\\% Processor Time' -SampleInterval 1 -MaxSamples 1).CounterSamples.CookedValue")
    ram_out = run_ps("(Get-CimInstance Win32_OperatingSystem | ForEach-Object {[math]::Round((($_.TotalVisibleMemorySize - $_.FreePhysicalMemory) / $_.TotalVisibleMemorySize) * 100, 1)})")
    try:
        cpu_pct = round(float(cpu_out), 1) if cpu_out else 0.0
    except (ValueError, TypeError):
        cpu_pct = 0.0
    try:
        ram_pct = float(ram_out) if ram_out else 0.0
    except (ValueError, TypeError):
        ram_pct = 0.0
    return {"cpu_load_pct": cpu_pct, "ram_pct": ram_pct}


def probe_gpu() -> dict:
    """GPU via wmic Win32_VideoController."""
    out = run_ps("Get-CimInstance Win32_VideoController | Where-Object {$_.AdapterRAM -gt 0} | Select-Object -First 1 -ExpandProperty Name")
    return {"name": out or "unknown"}


def probe_pi_reachable() -> dict:
    """Ping Pi + HTTP-Health."""
    info = {}
    # Ping (defensiv: stdout kann None sein)
    try:
        r = subprocess.run(
            ["ping", "-n", "1", "-w", "2000", PI_HOST],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=5, encoding="utf-8", errors="replace",
        )
        info["icmp_ok"] = r.returncode == 0
        ping_out = r.stdout or ""
        for line in ping_out.splitlines():
            if "Zeit=" in line or "time=" in line:
                try:
                    ms = line.split("=")[1].split("ms")[0].strip()
                    info["latency_ms"] = int(ms)
                    break
                except (IndexError, ValueError):
                    pass
    except (subprocess.TimeoutExpired, OSError):
        info["icmp_ok"] = False
    # HTTP
    try:
        r = requests.get(f"{PI_BASE}/health", timeout=3)
        info["http_ok"] = r.status_code == 200
    except requests.RequestException:
        info["http_ok"] = False
    return info


def collect_pc_hardware() -> dict:
    """Alle PC-Hardware-Probes."""
    started = time.time()
    data = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "webcam": probe_webcam_devices(),
        "audio": probe_audio_devices(),
        "disk": probe_disk(),
        "ollama": probe_ollama(),
        "cpu_ram": probe_cpu_ram(),
        "gpu": probe_gpu(),
        "pi_reachable": probe_pi_reachable(),
    }
    # Status-Aggregation
    status = "PASS"
    issues = []
    if data["disk"].get("free_gb", 100) < 10:
        status = "WARN"
        issues.append(f"disk_free<10gb ({data['disk'].get('free_gb')}gb)")
    if data["cpu_ram"]["ram_pct"] > 90:
        status = "WARN"
        issues.append(f"ram_pct={data['cpu_ram']['ram_pct']}")
    if not data["pi_reachable"].get("http_ok"):
        status = "FAIL"
        issues.append("pi_chat_unreachable")
    if not data["audio"]["count"]:
        status = "WARN"
        issues.append("no_audio_devices")
    data["status"] = status
    data["issues"] = issues
    data["duration_s"] = round(time.time() - started, 2)
    score = max(0, 7 - len(issues))
    return {"score": score, "max": 7, "status": status, "detail": data}


def post_layer(payload: dict) -> bool:
    try:
        r = requests.post(
            f"{PI_BASE}/mailbox/audit/pc_hardware",
            headers=HEADERS, json=payload, timeout=TIMEOUT_S,
        )
        if r.status_code == 200:
            return True
        logger.warning(f"[post] HTTP {r.status_code}: {r.text[:120]}")
    except requests.RequestException as e:
        logger.warning(f"[post] {e}")
    return False


def tick() -> dict:
    started = time.time()
    payload = collect_pc_hardware()
    posted = post_layer(payload)
    state = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(time.time() - started, 2),
        "payload": payload,
        "posted": posted,
    }
    safe_json_write(STATE_DIR / "pc_hardware_auditor_last.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description="MOLOCH PC-Hardware-Auditor (Welle 12)")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-s", type=int, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.json:
        last = STATE_DIR / "pc_hardware_auditor_last.json"
        print(last.read_text(encoding="utf-8") if last.exists() else "{}")
        return

    if args.once:
        state = tick()
        print(f"[once] status={state['payload']['status']} score={state['payload']['score']}/{state['payload']['max']} posted={state['posted']}")
        if state['payload']['detail'].get('issues'):
            print(f"       issues: {state['payload']['detail']['issues']}")
        return

    logger.info(f"PC-Hardware-Auditor: Loop alle {args.interval_s}s")
    while True:
        try:
            state = tick()
            logger.info(f"tick status={state['payload']['status']} posted={state['posted']}")
        except Exception as e:
            logger.exception(f"tick fail: {e}")
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
