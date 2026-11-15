#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANIMA Laufzeit-Inventar (Master-Briefing Phase B) - auf dem Pi ausfuehren.

Stellt fest, was am laufenden System WIRKLICH da ist, und schreibt einen
Bericht nach logs/anima_inventar_<datum>.md. Rein lesend:
  - keine Aenderung an Konfiguration, Diensten, Treibern oder Daten
  - kein eigener Zugriff auf die NPU (nur Status des laufenden Service
    und hailortcli-Abfragen)
  - nichts wird bewegt, nichts aufgenommen
Was nicht messbar ist, steht als "unbekannt" im Bericht - nie als Schaetzung.

    cd ~/moloch && python3 scripts/anima_inventory.py
    python3 scripts/anima_inventory.py --stdout      # nur ausgeben, keine Datei

Danach den Bericht committen, damit PC-Session und Cloud-Session ihn sehen.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STATUS_JSON = "/dev/shm/moloch_status.json"
SETTINGS_JSON = os.path.join(ROOT, "config", "settings.json")
UNKNOWN = "unbekannt"
TIMEOUT_S = 30


def run(cmd: List[str], timeout: int = TIMEOUT_S, any_exit: bool = False) -> Optional[str]:
    """Befehl ausfuehren. Ausgabe als Text, None wenn nicht vorhanden oder fehlgeschlagen.

    any_exit=True: Ausgabe auch bei Exit-Code != 0 (systemctl is-active meldet
    "inactive" mit Code 3 - das ist eine Antwort, kein Fehler).
    """
    if shutil.which(cmd[0]) is None:
        return None
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    out = (p.stdout or "").strip()
    return out if out and (any_exit or p.returncode == 0) else None


def read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", errors="replace") as f:
            return f.read().strip("\x00\n ")
    except OSError:
        return None


def read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def http_json(url: str, timeout: float = 4.0) -> Optional[dict]:
    try:
        import requests
        r = requests.get(url, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def val(x) -> str:
    return UNKNOWN if x is None or x == "" else str(x)


# === Einzelne Erhebungen (jede liefert Zeilen "Name | Wert | Quelle") ===

def parse_meminfo(text: Optional[str]) -> Dict[str, float]:
    """/proc/meminfo -> {'total_gb', 'available_gb'}; leer wenn nicht lesbar."""
    out: Dict[str, float] = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            if parts[0] == "MemTotal:":
                out["total_gb"] = round(int(parts[1]) / 1048576, 2)
            elif parts[0] == "MemAvailable:":
                out["available_gb"] = round(int(parts[1]) / 1048576, 2)
    return out


def parse_throttled(text: Optional[str]) -> Optional[Dict[str, bool]]:
    """vcgencmd get_throttled -> Flags. None wenn nicht lesbar."""
    if not text or "=" not in text:
        return None
    try:
        bits = int(text.split("=", 1)[1], 16)
    except ValueError:
        return None
    return {
        "unterspannung_jetzt": bool(bits & 0x1),
        "gedrosselt_jetzt": bool(bits & 0x4),
        "unterspannung_seit_start": bool(bits & 0x10000),
        "gedrosselt_seit_start": bool(bits & 0x40000),
    }


def section_system() -> List[Tuple[str, str, str]]:
    mem = parse_meminfo(read_text("/proc/meminfo"))
    thr = parse_throttled(run(["vcgencmd", "get_throttled"]))
    rows = [
        ("Modell", val(read_text("/proc/device-tree/model")), "/proc/device-tree/model"),
        ("RAM gesamt (GB)", val(mem.get("total_gb")), "/proc/meminfo"),
        ("RAM verfuegbar (GB)", val(mem.get("available_gb")), "/proc/meminfo"),
        ("Python", sys.version.split()[0], "sys.version"),
        ("Kernel", val(run(["uname", "-r"])), "uname -r"),
        ("Laufzeit", val(run(["uptime", "-p"])), "uptime -p"),
        ("CPU-Temperatur", val(run(["vcgencmd", "measure_temp"])), "vcgencmd measure_temp"),
    ]
    if thr is None:
        rows.append(("Unterspannung/Drosselung", UNKNOWN, "vcgencmd get_throttled"))
    else:
        for k, v in thr.items():
            rows.append((k, "JA" if v else "nein", "vcgencmd get_throttled"))
    fan = read_text("/sys/devices/platform/cooling_fan/hwmon/hwmon0/fan1_input")
    if fan is None:
        for base in ("/sys/devices/platform/cooling_fan/hwmon",):
            try:
                for d in sorted(os.listdir(base)):
                    fan = fan or read_text(os.path.join(base, d, "fan1_input"))
            except OSError:
                pass
    rows.append(("Luefter-Drehzahl (U/min)", val(fan), "hwmon fan1_input (nur mit Tacho-Signal)"))
    return rows


def section_storage() -> List[Tuple[str, str, str]]:
    rows = []
    lsblk = run(["lsblk", "-o", "NAME,SIZE,FSTYPE,MOUNTPOINT", "-n"])
    rows.append(("Datentraeger", val(lsblk).replace("\n", "<br>"), "lsblk"))
    for mount in ("/", "/mnt/moloch-data"):
        fstype = run(["findmnt", "-n", "-o", "FSTYPE", mount])
        try:
            du = shutil.disk_usage(mount)
            free = f"{du.free / 1e9:.1f} GB frei von {du.total / 1e9:.1f} GB ({100 * du.used / du.total:.0f} % belegt)"
        except OSError:
            free = UNKNOWN
        rows.append((f"{mount}", f"{val(fstype)}, {free}", "findmnt / disk_usage"))
    return rows


def section_npu() -> List[Tuple[str, str, str]]:
    rows = [
        ("Geraeteknoten /dev/hailo0", "vorhanden" if os.path.exists("/dev/hailo0") else "fehlt",
         "Dateisystem (belegt keine Inferenz)"),
        ("Kernelmodul", val(next((ln for ln in (run(["lsmod"]) or "").splitlines()
                                  if "hailo" in ln.lower()), None)), "lsmod"),
        ("Identitaet", val(run(["hailortcli", "fw-control", "identify"])).replace("\n", "<br>"),
         "hailortcli fw-control identify"),
    ]
    st = read_json(STATUS_JSON)
    if st is None:
        rows.append(("Service-Status", UNKNOWN, STATUS_JSON))
        return rows
    age = time.time() - os.stat(STATUS_JSON).st_mtime
    rows.append(("Service-Status", f"{age:.1f} s alt", STATUS_JSON))
    fps = st.get("fps") or {}
    rows.append(("FPS", json.dumps(fps) if fps else UNKNOWN, STATUS_JSON))
    rows.append(("Aktive Modelle", ", ".join(map(str, st.get("active_models") or [])) or UNKNOWN, STATUS_JSON))
    for name, w in sorted((st.get("worker_health") or {}).items()):
        if isinstance(w, dict):
            rows.append((f"Worker {name}",
                         f"laeuft={w.get('running')} inferenzen={w.get('total_inferences')} "
                         f"fehler={w.get('total_errors')} letzte_ms={w.get('last_inference_ms')}",
                         STATUS_JSON))
    return rows


def section_services() -> List[Tuple[str, str, str]]:
    rows = []
    for unit in ("moloch", "moloch-chat", "moloch-chat-https", "moloch-cross-monitor", "qdrant",
                 "hailo-ollama", "lightdm"):
        rows.append((f"Dienst {unit}",
                     f"{val(run(['systemctl', 'is-active', unit], any_exit=True))} / "
                     f"{val(run(['systemctl', 'is-enabled', unit], any_exit=True))}",
                     "systemctl is-active / is-enabled"))
    ports = run(["ss", "-ltn"]) or ""
    for port, what in ((9100, "Chat-Server"), (9443, "Chat HTTPS"), (6333, "Qdrant"), (8000, "hailo-ollama")):
        rows.append((f"Port {port} ({what})", "offen" if f":{port} " in ports else "zu", "ss -ltn"))
    return rows


def section_llm() -> List[Tuple[str, str, str]]:
    s = read_json(SETTINGS_JSON) or {}
    t = s.get("tentacle_llm") or {}
    rows = [
        ("llm_mode", val(s.get("llm_mode")), "settings.json"),
        ("llm_profile", val(s.get("llm_profile")), "settings.json"),
        ("tentacle_llm.model (konfiguriert)", val(t.get("model")), "settings.json"),
        ("embedding_bridge", json.dumps(s.get("embedding_bridge")) if s.get("embedding_bridge") else "nicht gesetzt (Vektorsuche aus)",
         "settings.json"),
    ]
    host, port = t.get("host"), t.get("port", 11434)
    if host:
        ps = http_json(f"http://{host}:{port}/api/ps")
        tags = http_json(f"http://{host}:{port}/api/tags")
        loaded = ", ".join(m.get("name", "?") for m in (ps or {}).get("models", [])) if ps else None
        avail = ", ".join(m.get("name", "?") for m in (tags or {}).get("models", [])) if tags else None
        rows.append(("PC-Ollama erreichbar", "ja" if tags is not None else "nein", f"{host}:{port}/api/tags"))
        rows.append(("PC-Ollama geladen (antwortet gerade)", val(loaded) if ps is not None else UNKNOWN, "/api/ps"))
        rows.append(("PC-Ollama vorhandene Modelle", val(avail), "/api/tags"))
    npu_tags = http_json("http://localhost:8000/api/tags")
    rows.append(("NPU-Modell (hailo-ollama)",
                 ", ".join(m.get("name", "?") for m in npu_tags.get("models", [])) if npu_tags else UNKNOWN,
                 "localhost:8000/api/tags"))
    return rows


def section_memory() -> List[Tuple[str, str, str]]:
    rows = []
    base = "/mnt/moloch-data/memory"
    for name in ("facts.json", "identity.json", "core_state.json"):
        p = os.path.join(base, name)
        try:
            size = os.path.getsize(p)
            data = read_json(p)
            n = len(data) if isinstance(data, dict) else UNKNOWN
            rows.append((name, f"{size} Bytes, {n} Eintraege", p))
        except OSError:
            rows.append((name, "fehlt", p))
    try:
        days = len([f for f in os.listdir(os.path.join(base, "conversations")) if f.endswith(".json")])
        rows.append(("Gespraechstage", str(days), base + "/conversations"))
    except OSError:
        rows.append(("Gespraechstage", UNKNOWN, base + "/conversations"))
    q = http_json("http://localhost:6333/collections")
    if q is None:
        rows.append(("Qdrant", "nicht erreichbar", "localhost:6333"))
    else:
        cols = [c.get("name") for c in (q.get("result") or {}).get("collections", [])]
        rows.append(("Qdrant-Collections", ", ".join(cols) or "keine", "localhost:6333/collections"))
    rec = http_json("http://localhost:9100/memory/recall?q=Wer%20bist%20du%20eigentlich")
    if rec is None:
        rows.append(("GET /memory/recall", "nicht erreichbar (Chat-Server alt oder aus?)", "localhost:9100"))
    else:
        rows.append(("GET /memory/recall", f"{rec.get('count')} Treffer, Quellen "
                     f"{sorted({h.get('source') for h in rec.get('hits', [])})}", "localhost:9100"))
    return rows


def section_peripherals() -> List[Tuple[str, str, str]]:
    rows = []
    s = read_json(SETTINGS_JSON) or {}
    cards = read_text("/proc/asound/cards")
    rows.append(("Soundkarten", val(cards).replace("\n", "<br>"), "/proc/asound/cards"))
    rows.append(("audio_source", val((s.get("audio") or {}).get("audio_source")), "settings.json"))
    live = http_json("http://localhost:9100/live")
    if live is None:
        rows.append(("/live", "nicht erreichbar", "localhost:9100"))
    else:
        ptz = live.get("ptz") or {}
        rows.append(("Kamera-Position", f"pan={val(ptz.get('pan'))} tilt={val(ptz.get('tilt'))} "
                     f"tracker={val(ptz.get('tracker_state'))}", "/live"))
        rows.append(("Erkannt", f"person={live.get('person_detected')} gesicht={val(live.get('face_id'))}", "/live"))
    return rows


SECTIONS = [
    ("System", section_system),
    ("Massenspeicher", section_storage),
    ("NPU (Hailo)", section_npu),
    ("Dienste und Ports", section_services),
    ("Sprachmodelle", section_llm),
    ("Gedaechtnis", section_memory),
    ("Peripherie", section_peripherals),
]


def build_report() -> str:
    lines = [f"# ANIMA Laufzeit-Inventar – {time.strftime('%Y-%m-%d %H:%M')}", "",
             "Erzeugt von `scripts/anima_inventory.py` (rein lesend). `unbekannt` = nicht messbar, keine Schaetzung.", ""]
    for title, fn in SECTIONS:
        lines += [f"## {title}", "", "| Was | Wert | Quelle |", "|---|---|---|"]
        try:
            rows = fn()
        except Exception as e:  # ein kaputter Abschnitt darf den Bericht nicht verhindern
            rows = [("Abschnitt fehlgeschlagen", f"{type(e).__name__}: {str(e)[:120]}", "-")]
        for name, value, source in rows:
            lines.append(f"| {name} | {str(value).replace('|', '/')} | {source} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ANIMA Laufzeit-Inventar (rein lesend)")
    ap.add_argument("--stdout", action="store_true", help="nur ausgeben, keine Datei schreiben")
    args = ap.parse_args(argv)
    report = build_report()
    print(report)
    if not args.stdout:
        path = os.path.join(ROOT, "logs", f"anima_inventar_{time.strftime('%Y-%m-%d')}.md")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        os.replace(tmp, path)
        print(f"\nBericht: {path}\nBitte committen: git add -f {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
