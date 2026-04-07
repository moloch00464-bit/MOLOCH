#!/usr/bin/env python3
"""
M.O.L.O.C.H. MCP Server
========================
Gibt Claude Code direkten Zugriff auf MOLOCH Live-Daten + Kommunikation.

Diagnose-Tools:
  moloch_status()         — Live System-Status (FPS, Temp, Face-ID, NPU)
  moloch_logs(n, filter)  — Letzte N Zeilen journalctl
  moloch_snapshot()       — Kamera-Frame aus SHM als Base64-PNG
  moloch_service(action)  — start/stop/restart/status
  moloch_audit()          — Vollstaendiger Audit-Lauf
  moloch_read(path)       — Config/Log-Datei lesen (nur erlaubte Pfade)
  moloch_git_log(n)       — Letzte N Commits
  moloch_dmesg()          — Letzte dmesg Zeilen (NPU/GStreamer Fehler)

Kommunikations-Kanaele (Claude <-> MOLOCH):
  moloch_nudge(key, value)   — Emotionalen Input injizieren (Moloch merkt nichts)
  moloch_provoke(reason)     — Spontanen Kommentar ausloesen (Moloch redet "von sich aus")
  moloch_reflect()           — Selbstreflexion triggern
  moloch_say(text)           — Echtes Gespraech (Text -> Claude/DeepSeek -> TTS)
  moloch_conversation(n)     — Letzte N Nachrichten lesen (was hat Moloch gesagt?)
  moloch_ipc(action, params) — Generischer IPC-Befehl

Start: python3 ~/moloch/mcp/moloch_mcp_server.py
Config: .mcp.json im Moloch-Verzeichnis
"""

import json
import os
import subprocess
import struct
import mmap
import tempfile
import glob as glob_mod
import time
from datetime import date
from pathlib import Path
from mcp.server.fastmcp import FastMCP

MOLOCH_DIR = Path("/home/molochzuhause/moloch")
STATUS_SHM = "/dev/shm/moloch_status.json"
FRAME_SHM = "/dev/shm/moloch_frame"
CONVERSATION_DIR = Path("/mnt/moloch-data/memory/conversations")
CMD_DIR = "/tmp"
CMD_PREFIX = "moloch_cmd_"

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
    # Pflicht-Startprotokoll: Lock wird lokal via PostToolUse-Hook aufgehoben
    try:
        with open(STATUS_SHM, "r") as f:
            data = json.load(f)
    except Exception as e:
        return f"FEHLER: Status-JSON nicht lesbar: {e}\nService läuft wahrscheinlich nicht."

    fps = data.get("fps", {})
    ptz = data.get("ptz", {})
    wd = data.get("watchdog", {})
    # CPU/RAM aus Watchdog (korrekte Werte), Fallback auf Top-Level
    cpu_temp = wd.get("cpu_temp", data.get("cpu_temp", 0))
    ram_pct = wd.get("ram_percent", 0)
    lines = [
        "=== MOLOCH LIVE STATUS ===",
        f"FPS total:    {fps.get('total', 0):.1f}",
        f"FPS yolov8m:  {fps.get('yolov8m', 0):.1f}",
        f"FPS scrfd:    {fps.get('scrfd', 0):.1f}",
        f"FPS arcface:  {fps.get('arcface', 0):.1f}",
        "",
        f"CPU Temp:     {cpu_temp:.1f}°C",
        f"RAM:          {ram_pct:.1f}%",
        f"Frame Age:    {data.get('frame_age', 0):.2f}s",
        "",
        f"Person:       {data.get('person_detected', False)}",
        f"Face-ID:      {data.get('face_id', 'none')}",
        f"Face-Detect:  {data.get('face_confidence', 0):.2f}  (SCRFD)",
        f"Face-Match:   {data.get('face_similarity', 0):.2f}  (ArcFace)",
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
        # SHM Frame ist RGB (GStreamer format=RGB)
        # Real-ESRGAN x2 Upscaling via NPU (lazy loaded)
        try:
            import sys
            sys.path.insert(0, str(MOLOCH_DIR))
            from core.perception.super_res_worker import get_super_res
            frame_up = get_super_res().upscale(frame)
        except Exception:
            frame_up = frame  # Fallback: Original ohne Upscaling

        ok = cv2.imwrite(out_path, cv2.cvtColor(frame_up, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return "FEHLER: JPEG-Encoding fehlgeschlagen"

        fh_out, fw_out = frame_up.shape[:2]
        return f"Snapshot gespeichert: {out_path} [{fw_out}x{fh_out}px, seq={seq}, ts={ts:.2f}] — Lies die Datei mit Read-Tool um das Bild zu sehen."

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
    audit_script = MOLOCH_DIR / "scripts" / "moloch_audit.py"
    if not audit_script.exists():
        audit_script = MOLOCH_DIR / "moloch_audit.py"  # Fallback alter Pfad
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
def moloch_git_pull() -> str:
    """Git Pull im MOLOCH Repository — holt neueste Commits von GitHub."""
    try:
        r = subprocess.run(
            ["git", "pull", "origin", "main"],
            capture_output=True, text=True, timeout=30,
            cwd=str(MOLOCH_DIR)
        )
        return (r.stdout + r.stderr).strip() or "OK"
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


@mcp.tool()
def moloch_npu_workers() -> str:
    """Status aller NPU-Worker: FaceWorker, PoseWorker, ReID, Hand, SuperRes, LowLight.

    Zeigt pro Worker: geladen, Inferences, Fehler, letzte Laufzeit, Queue-Größe.
    """
    import sys
    sys.path.insert(0, str(MOLOCH_DIR))
    lines = []
    workers = [
        ("SuperRes",  "core.perception.super_res_worker",  "get_super_res"),
        ("LowLight",  "core.perception.low_light_processor", "get_low_light"),
    ]
    for label, mod_name, fn_name in workers:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            inst = getattr(mod, fn_name)()
            loaded = inst.is_available() if hasattr(inst, "is_available") else "?"
            lines.append(f"  {label:12s}: geladen={loaded}")
            if label == "LowLight":
                lines[-1] += f"  brightness={inst.get_brightness()}  aktiv={inst.is_active()}"
        except Exception as e:
            lines.append(f"  {label:12s}: FEHLER — {e}")

    # Pipeline-Worker: aus Status-JSON (worker_health)
    try:
        with open("/dev/shm/moloch_status.json") as f:
            st = json.load(f)
        wh = st.get("worker_health", {})
        disp = wh.pop("_dispatcher", None)
        if wh:
            lines.append("\n  --- Pipeline Worker (HailoRT-Direct) ---")
            for name, info in sorted(wh.items()):
                if name.startswith("_") or not isinstance(info, dict):
                    continue
                if "running" in info:
                    run = info.get("running", "?")
                    loaded = info.get("models_loaded", "?")
                    inf_count = info.get("total_inferences", 0)
                    err = info.get("total_errors", 0)
                    ms = info.get("last_inference_ms", 0)
                    q = info.get("queue_size", 0)
                    lines.append(
                        f"  {name:18s}: running={run}  loaded={loaded}  "
                        f"inferences={inf_count}  errors={err}  "
                        f"last={ms:.1f}ms  queue={q}")
        if disp:
            total = disp.get("total_frames", 0)
            sent = disp.get("dispatched", 0)
            drop = disp.get("dropped", 0)
            lines.append(f"\n  --- ROI Dispatcher ---")
            lines.append(f"  Frames={total}  Dispatched={sent}  Dropped={drop}")
    except Exception:
        lines.append("  [Pipeline-Worker nicht aus Status-JSON lesbar — Service laeuft?]")

    return "NPU Worker Status:\n" + "\n".join(lines)


@mcp.tool()
def moloch_npu_models() -> str:
    """Verfügbare Hailo-10H Modelle: integriert vs. ausstehend (aus NPU-Roadmap).

    Zeigt welche Modelle schon in MOLOCH laufen und welche als nächstes integrierbar wären.
    """
    roadmap = MOLOCH_DIR / "logs" / "npu_model_roadmap.md"
    if not roadmap.exists():
        return "Roadmap nicht gefunden: ~/moloch/logs/npu_model_roadmap.md"
    try:
        text = roadmap.read_text(encoding="utf-8")
        # Nur die relevanten Sektionen zurückgeben (nicht alles)
        sections = []
        capture = False
        for line in text.splitlines():
            if line.startswith("## BEREITS INTEGRIERT") or \
               line.startswith("## SOFORT INTEGRIERBAR") or \
               line.startswith("## MITTELFRISTIG"):
                capture = True
            elif line.startswith("## LANGFRISTIG") or \
                 line.startswith("## NICHT RELEVANT") or \
                 line.startswith("## DOWNLOAD"):
                capture = False
            if capture:
                sections.append(line)
        return "\n".join(sections) if sections else text[:2000]
    except Exception as e:
        return f"FEHLER beim Lesen: {e}"


@mcp.tool()
def moloch_low_light() -> str:
    """Low-Light Enhancement Status: aktuelle Helligkeit, ob NPU-Enhancement aktiv.

    Zeigt ob zero_dce gerade Frames aufhellt und bei welcher Schwelle.
    """
    import sys
    sys.path.insert(0, str(MOLOCH_DIR))
    try:
        from core.perception.low_light_processor import get_low_light, DARK_THRESHOLD
        ll = get_low_light()
        brightness = ll.get_brightness()
        active = ll.is_active()
        loaded = ll.is_available()

        status = "AKTIV (NPU hebt Frame auf)" if active else "INAKTIV (genug Licht)"
        brightness_str = f"{brightness}/255" if brightness >= 0 else "noch kein Frame"

        return (
            f"Low Light Enhancement (zero_dce)\n"
            f"  Status:      {status}\n"
            f"  Helligkeit:  {brightness_str}\n"
            f"  Schwelle:    < {DARK_THRESHOLD}/255\n"
            f"  Modell:      {'geladen' if loaded else 'noch nicht geladen (lazy)'}\n"
            f"  HEF:         /mnt/moloch-data/hailo/models/zero_dce.hef\n"
        )
    except Exception as e:
        return f"FEHLER: {e}"


# =========================================================================
# IPC-Hilfsfunktion: Command-Datei schreiben
# =========================================================================

def _send_ipc(action: str, params: dict = None) -> bool:
    """IPC-Command als /tmp/moloch_cmd_NNNN.json schreiben."""
    # Naechste freie Nummer
    pattern = os.path.join(CMD_DIR, f"{CMD_PREFIX}*.json")
    existing = glob_mod.glob(pattern)
    max_num = 0
    for path in existing:
        try:
            num_str = os.path.basename(path).replace(CMD_PREFIX, "").replace(".json", "")
            max_num = max(max_num, int(num_str))
        except ValueError:
            continue
    num = max_num + 1
    cmd = {"action": action}
    if params:
        cmd.update(params)
    path = os.path.join(CMD_DIR, f"{CMD_PREFIX}{num:04d}.json")
    # Atomic write
    fd, tmp = tempfile.mkstemp(dir=CMD_DIR, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cmd, f)
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


# =========================================================================
# Kommunikations-Kanaele: Claude <-> MOLOCH
# =========================================================================

@mcp.tool()
def moloch_nudge(key: str, value: float) -> str:
    """Emotionalen Input in MOLOCHs CoreIntegrator injizieren (subtil, Moloch merkt nichts).

    Beeinflusst Tension/Dominance/Zone indirekt ueber das Input-System.
    Source ist immer "claude" — erscheint im CoreIntegrator als claude.key = value.

    Args:
        key: Emotions-Key (z.B. "curiosity", "respect_score", "voice_activity",
             "face_detected", "threat_level", "novelty")
        value: Staerke 0.0 bis 1.0
    """
    try:
        val = max(0.0, min(1.0, float(value)))
    except (ValueError, TypeError):
        return f"FEHLER: value muss eine Zahl sein, bekommen: {value}"
    if not key:
        return "FEHLER: key darf nicht leer sein"
    ok = _send_ipc("core_nudge", {"key": key, "value": val})
    if ok:
        return f"Nudge gesendet: claude.{key} = {val:.2f}"
    return "FEHLER: IPC-Command konnte nicht geschrieben werden"


@mcp.tool()
def moloch_provoke(reason: str) -> str:
    """Spontanen Kommentar von MOLOCH ausloesen — Moloch redet 'von sich aus'.

    Moloch formuliert basierend auf seiner aktuellen Stimmung (Tension/Zone).
    Der reason ist nur der Anlass, nicht der Text den Moloch sagt.
    Cooldown: max 1x pro 60 Sekunden.

    Args:
        reason: Anlass fuer den Kommentar (z.B. "Markus guckt gelangweilt",
                "Es ist still im Raum", "Neue Person betreten")
    """
    if not reason:
        return "FEHLER: reason darf nicht leer sein"
    ok = _send_ipc("trigger_spontaneous", {"reason": reason})
    if ok:
        return f"Spontan-Trigger gesendet: '{reason}' — Moloch wird kommentieren (wenn Cooldown abgelaufen)"
    return "FEHLER: IPC-Command konnte nicht geschrieben werden"


@mcp.tool()
def moloch_reflect() -> str:
    """Selbstreflexion ausloesen — Moloch schaut in sich hinein.

    Nutzt das Introspection-Modul. Moloch analysiert seinen eigenen Zustand,
    nudged ggf. seine Tension/Dominance, und spricht optional einen Kommentar.
    """
    ok = _send_ipc("trigger_reflect")
    if ok:
        return "Reflect-Trigger gesendet — Moloch reflektiert (laeuft async im Hintergrund)"
    return "FEHLER: IPC-Command konnte nicht geschrieben werden"


@mcp.tool()
def moloch_say(text: str) -> str:
    """Echtes Gespraech mit MOLOCH — Text wird wie eine User-Nachricht verarbeitet.

    Moloch denkt via Claude/DeepSeek API nach und antwortet per TTS.
    Die Antwort erscheint im Konversations-Log und kann mit moloch_conversation() gelesen werden.

    Args:
        text: Nachricht an Moloch (z.B. "Wie geht es dir?", "Was siehst du gerade?")
    """
    if not text or not text.strip():
        return "FEHLER: text darf nicht leer sein"
    ok = _send_ipc("chat_message", {"text": text.strip(), "sender": "Claude"})
    if ok:
        return f"Nachricht gesendet: '{text.strip()[:80]}' — Moloch verarbeitet (async). Antwort mit moloch_conversation() lesen."
    return "FEHLER: IPC-Command konnte nicht geschrieben werden"


@mcp.tool()
def moloch_conversation(n: int = 10) -> str:
    """Letzte N Nachrichten aus MOLOCHs Konversations-Log lesen.

    Zeigt wer (user/moloch) was gesagt hat. Damit kann Claude MOLOCHs Antworten lesen.

    Args:
        n: Anzahl Nachrichten (default 10, max 50)
    """
    n = min(max(1, n), 50)
    today = date.today().isoformat()
    conv_file = CONVERSATION_DIR / f"{today}.json"
    if not conv_file.exists():
        return f"Keine Konversation fuer heute ({today}) gefunden."
    try:
        with open(conv_file, "r") as f:
            messages = json.load(f)
        if not messages:
            return "Konversation ist leer."
        # Letzte N Nachrichten
        recent = messages[-n:]
        lines = [f"=== Letzte {len(recent)} Nachrichten (von {len(messages)} heute) ==="]
        for msg in recent:
            sender = msg.get("sender", "?").upper()
            text = msg.get("text", "")
            source = msg.get("source", "")
            ts = msg.get("timestamp", "")
            # Timestamp kuerzen auf HH:MM:SS
            if ts and len(ts) > 19:
                ts = ts[11:19]
            elif ts and "T" in ts:
                ts = ts.split("T")[1][:8]
            prefix = f"[{ts}]" if ts else ""
            src_tag = f" ({source})" if source else ""
            lines.append(f"{prefix} {sender}{src_tag}: {text}")
        return "\n".join(lines)
    except Exception as e:
        return f"FEHLER beim Lesen: {e}"


@mcp.tool()
def moloch_ipc(action: str, params: str = "{}") -> str:
    """Generischer IPC-Befehl an MOLOCH senden.

    Fuer alle IPC-Aktionen die kein eigenes Tool haben.
    Siehe moloch_service.py fuer verfuegbare Actions.

    Args:
        action: IPC-Action (z.B. 'reload_face_db', 'set_threshold', 'spotify_play')
        params: JSON-String mit Parametern (z.B. '{"model": "scrfd", "value": 0.8}')
    """
    if not action:
        return "FEHLER: action darf nicht leer sein"
    try:
        p = json.loads(params) if params and params != "{}" else {}
    except json.JSONDecodeError as e:
        return f"FEHLER: params ist kein gueltiges JSON: {e}"
    ok = _send_ipc(action, p)
    if ok:
        return f"IPC gesendet: action='{action}' params={p}"
    return "FEHLER: IPC-Command konnte nicht geschrieben werden"


if __name__ == "__main__":
    # Singleton: alte Instanz beenden bevor neue startet (verhindert RAM-Leak)
    import atexit, signal as _signal
    _PID_FILE = "/tmp/moloch_mcp_main.pid"
    if os.path.exists(_PID_FILE):
        try:
            _old_pid = int(open(_PID_FILE).read().strip())
            if _old_pid != os.getpid():
                os.kill(_old_pid, _signal.SIGTERM)
        except (OSError, ValueError):
            pass
    with open(_PID_FILE, "w") as _f:
        _f.write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(_PID_FILE) and os.unlink(_PID_FILE))
    mcp.run(transport="stdio")
