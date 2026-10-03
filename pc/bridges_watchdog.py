"""MOLOCH Bridges Watchdog v2 — prueft alle PC-Side-Dienste, heilt via Scheduled Task.

Ersetzt watchdog_bridges.ps1 (moloch-bridge/, Task warf seit Anlage Result 1,
hat nie geloggt — Bridges starben unbemerkt, siehe Audit-FAIL 2026-06-11).

Laeuft als Scheduled Task "MOLOCH Bridges Watchdog v2" alle 5 Minuten via
pythonw.exe (kein Fenster, keine ExecutionPolicy-Abhaengigkeit).
Stdlib only — kein pip-Paket noetig.
"""

import os
import subprocess
import urllib.request
from datetime import datetime

LOG = os.path.join(os.environ.get("TEMP", r"C:\Windows\Temp"),
                   "moloch_bridges_watchdog_v2.log")
LOG_MAX_ZEILEN = 300

# Dienst -> (Port, Health-Pfad, Scheduled-Task-Name zum Heilen)
# Ollama heilt ueber MolochOllama (nicht das Duplikat "Ollama LAN-Server",
# das den Boot-Race verursacht hat).
DIENSTE = [
    ("Ollama",        11434, "/api/tags", "MolochOllama"),
    ("STT-Bridge",     9001, "/health",   "MOLOCH STT-Bridge"),
    ("TTS-Bridge",     9002, "/health",   "MOLOCH TTS-Bridge"),
    # :9000 ist der SSH-Tunnel zum Pi (NICHT die alte chat_ui) — das Dashboard haengt
    # dran, also pflegen wir den Tunnel statt der deprecated Chat-UI.
    ("Pi-Tunnel",      9000, "/live",     "MolochPiTunnel"),
    ("Dashboard",     11700, "/",         "MolochDashboard"),
    ("Search-Proxy",  11650, "/health",   "MolochSearchProxy"),
    ("Adapter-Proxy", 11600, "/health",   "MolochAdapterProxy"),
]


def port_ok(port: int, pfad: str) -> bool:
    """HTTP-Check auf 127.0.0.1 — True bei Status 200."""
    url = f"http://127.0.0.1:{port}{pfad}"
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            return r.status == 200
    except Exception:
        return False


def task_starten(task_name: str) -> str:
    """Scheduled Task anstossen. Gibt 'OK' oder Fehlertext zurueck."""
    try:
        r = subprocess.run(
            ["schtasks", "/run", "/tn", task_name],
            capture_output=True, text=True, timeout=30,
        )
        return "OK" if r.returncode == 0 else (r.stderr or r.stdout).strip()
    except Exception as e:
        return str(e)


def log_schreiben(zeilen: list) -> None:
    """Anhaengen + Rotation auf die letzten LOG_MAX_ZEILEN Zeilen."""
    alt = []
    if os.path.exists(LOG):
        try:
            with open(LOG, encoding="utf-8", errors="replace") as f:
                alt = f.read().splitlines()
        except OSError:
            alt = []
    alle = (alt + zeilen)[-LOG_MAX_ZEILEN:]
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(alle) + "\n")


def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    zeilen = [f"{ts} [RUN] Watchdog v2 Lauf"]  # Start-Marker: beweist, dass der Task laeuft
    for name, port, pfad, task in DIENSTE:
        if port_ok(port, pfad):
            zeilen.append(f"{ts} [OK]   {name} (:{port})")
            continue
        ergebnis = task_starten(task)
        if ergebnis == "OK":
            zeilen.append(f"{ts} [HEAL] {name} (:{port}) down — Task '{task}' gestartet")
        else:
            zeilen.append(f"{ts} [FAIL] {name} (:{port}) down — Task '{task}': {ergebnis}")
    log_schreiben(zeilen)


if __name__ == "__main__":
    main()
