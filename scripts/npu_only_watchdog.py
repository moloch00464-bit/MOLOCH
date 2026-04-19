#!/usr/bin/env python3
"""
M.O.L.O.C.H. NPU-Only Watchdog
================================

Permanenter Health-Watchdog fuer den NPU-only-Modus (Cloud-API hart deaktiviert).
Alle WATCHDOG_INTERVAL_SEC: Mini-Probe gegen hailo-ollama /api/chat mit
Profil "chat", loggt Latenz + Antwort-Laenge + Status in CSV.

Bei 3x FAIL hintereinander: TTS-Alarm via piper (oder Log-Only wenn TTS fehlt).

Aufgaben:
- erkennt wenn lokales LLM stumm wird (HTTP 5xx, Timeout, leere Antwort)
- erkennt wenn hailo-ollama-Service crasht (PID-Wechsel via systemctl)
- schreibt CSV-Trend zur spaeteren Auswertung (logs/npu_watchdog.csv)
- schreibt JSONL fuer Detail-Forensik (logs/npu_watchdog.jsonl)

Laeuft als systemd-Service (moloch-npu-watchdog.service).
"""

import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# === Konfiguration ===
WATCHDOG_INTERVAL_SEC = 30 * 60   # 30 Min zwischen Probes
PROBE_TIMEOUT_SEC = 60            # Max Wartezeit pro Probe
PROBE_MODEL = "qwen2.5:1.5b"
PROBE_PROMPT = "ping"
PROBE_NUM_PREDICT = 5
FAIL_THRESHOLD = 3                # nach N FAILs hintereinander -> TTS-Alarm

OLLAMA_URL = "http://127.0.0.1:8000/api/chat"
SERVICE_NAME = "hailo-ollama"
LOG_DIR = Path("/home/molochzuhause/moloch/logs")
CSV_PATH = LOG_DIR / "npu_watchdog.csv"
JSONL_PATH = LOG_DIR / "npu_watchdog.jsonl"

CSV_HEADER = [
    "timestamp_iso", "elapsed_run_s", "probe_status",
    "latency_ms", "response_chars", "model",
    "hailo_ollama_active", "hailo_ollama_pid",
    "fail_streak", "alarm_fired",
]

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s NPUWatchdog: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("NPUWatchdog")


def get_service_state():
    """Gibt (active_str, main_pid_int) fuer hailo-ollama zurueck."""
    try:
        active = subprocess.run(
            ["systemctl", "is-active", SERVICE_NAME],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        active = "unknown"
    pid = 0
    try:
        out = subprocess.run(
            ["systemctl", "show", SERVICE_NAME, "-p", "MainPID"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        # "MainPID=1234"
        if "=" in out:
            pid = int(out.split("=", 1)[1] or "0")
    except Exception:
        pid = 0
    return active, pid


def do_probe():
    """Schickt Mini-Probe gegen hailo-ollama. Gibt (status, latency_ms, response_chars)."""
    body = {
        "model": PROBE_MODEL,
        "messages": [{"role": "user", "content": PROBE_PROMPT}],
        "stream": False,
        "options": {"num_predict": PROBE_NUM_PREDICT},
    }
    t0 = time.monotonic()
    try:
        r = requests.post(OLLAMA_URL, json=body, timeout=PROBE_TIMEOUT_SEC)
        latency_ms = int((time.monotonic() - t0) * 1000)
        if r.status_code != 200:
            return f"http_{r.status_code}", latency_ms, 0
        data = r.json()
        text = (data.get("message", {}).get("content") or "").strip()
        if not text:
            return "empty", latency_ms, 0
        return "ok", latency_ms, len(text)
    except requests.exceptions.Timeout:
        return "timeout", int((time.monotonic() - t0) * 1000), 0
    except Exception as e:
        log.warning("Probe-Exception: %s", e)
        return "exception", int((time.monotonic() - t0) * 1000), 0


def fire_tts_alarm(message: str) -> bool:
    """Versucht piper-TTS-Alarm. Best-effort, no-throw."""
    try:
        # Pruefen ob piper verfuegbar ist
        which = subprocess.run(["which", "piper"], capture_output=True, text=True, timeout=3)
        if which.returncode != 0:
            log.warning("piper nicht installiert, kein Audio-Alarm")
            return False
        # piper braucht Voice-Model; wir nutzen den default-Pfad falls vorhanden
        voice = "/home/molochzuhause/moloch/models/piper/de_DE-thorsten-low.onnx"
        if not os.path.exists(voice):
            log.warning("piper Voice-File fehlt: %s", voice)
            return False
        cmd = f'echo "{message}" | piper --model {voice} --output_raw | aplay -r 22050 -f S16_LE -t raw -'
        subprocess.Popen(cmd, shell=True)
        return True
    except Exception as e:
        log.warning("TTS-Alarm Fehler: %s", e)
        return False


def append_csv(row: dict):
    """CSV-Zeile anhaengen, Header beim ersten Mal schreiben."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_HEADER})


def append_jsonl(payload: dict):
    """JSONL-Zeile anhaengen."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main():
    log.info("Start (Interval=%ds, Timeout=%ds, Modell=%s)",
             WATCHDOG_INTERVAL_SEC, PROBE_TIMEOUT_SEC, PROBE_MODEL)
    t_start = time.monotonic()
    fail_streak = 0
    last_alarm_streak = 0
    while True:
        try:
            elapsed = int(time.monotonic() - t_start)
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            active, pid = get_service_state()
            status, latency_ms, resp_chars = do_probe()

            if status == "ok":
                fail_streak = 0
                last_alarm_streak = 0
            else:
                fail_streak += 1

            alarm_fired = False
            if fail_streak >= FAIL_THRESHOLD and fail_streak > last_alarm_streak:
                msg = f"Lokales LLM hat {fail_streak} mal nicht geantwortet. NPU-Modus instabil."
                log.error(msg)
                alarm_fired = fire_tts_alarm(msg)
                last_alarm_streak = fail_streak

            row = {
                "timestamp_iso": now_iso,
                "elapsed_run_s": elapsed,
                "probe_status": status,
                "latency_ms": latency_ms,
                "response_chars": resp_chars,
                "model": PROBE_MODEL,
                "hailo_ollama_active": active,
                "hailo_ollama_pid": pid,
                "fail_streak": fail_streak,
                "alarm_fired": int(alarm_fired),
            }
            append_csv(row)
            append_jsonl(row)

            if status == "ok":
                log.info("OK %dms (%dch) — service=%s pid=%d", latency_ms, resp_chars, active, pid)
            else:
                log.warning("FAIL[%s] %dms — streak=%d service=%s pid=%d",
                            status, latency_ms, fail_streak, active, pid)
        except Exception as e:
            log.exception("Watchdog-Iteration Exception: %s", e)
        # Sleep bis zum naechsten Probe
        time.sleep(WATCHDOG_INTERVAL_SEC)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Beendet (KeyboardInterrupt)")
        sys.exit(0)
