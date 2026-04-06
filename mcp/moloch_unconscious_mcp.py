#!/usr/bin/env python3
"""
M.O.L.O.C.H. Unterbewusstsein MCP Server
==========================================
Zweiter MCP-Server: Exponiert MOLOCHs inneren Zustand fuer Claude.

Unterschied zum Haupt-Server (moloch_mcp_server.py):
  Haupt-Server:  Claude steuert MOLOCH (Status, Service, Logs)
  Dieser Server: Claude liest/beeinflusst MOLOCHs Psyche direkt

Tools:
  uc_get_state()               — Aktueller Unterbewusstsein-Zustand
  uc_get_mood()                — Letzter Mood-Impuls (aus moloch_impulse.json)
  uc_get_history(n)            — Letzte N Concern-Eintraege aus self_tune.log
  uc_inject_impulse(type, str) — Mood-Impuls direkt injizieren (via IPC)
  uc_reflect(question)         — Selbstreflexion ausloesen mit Frage

Start: python3 ~/moloch/mcp/moloch_unconscious_mcp.py
Config: zweiter Eintrag in .mcp.json (moloch-unconscious)
"""

import json
import os
import glob as glob_mod
import tempfile
import time
from pathlib import Path
from mcp.server.fastmcp import FastMCP

MOLOCH_DIR = Path("/home/molochzuhause/moloch")
STATUS_PATH  = "/dev/shm/moloch_status.json"
IMPULSE_PATH = "/dev/shm/moloch_impulse.json"
SELF_TUNE_LOG = MOLOCH_DIR / "logs" / "self_tune.log"
CMD_DIR = "/tmp"
CMD_PREFIX = "moloch_cmd_"

mcp = FastMCP("moloch-unconscious")


# ============================================================
# IPC-Hilfsfunktion (Kopie aus moloch_mcp_server.py)
# ============================================================

def _send_ipc(action: str, params: dict = None) -> bool:
    """IPC-Command als /tmp/moloch_cmd_NNNN.json schreiben (atomic)."""
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


# ============================================================
# TOOLS
# ============================================================

@mcp.tool()
def uc_get_state() -> str:
    """Aktueller Unterbewusstsein-Zustand: Tension, Mood-Schichten, NPU-Szenario.

    Liest aus /dev/shm/moloch_status.json — die gleiche Quelle wie UnconsciousEngine.
    Zeigt was das Unterbewusstsein gerade 'sieht' und bewertet.
    """
    try:
        with open(STATUS_PATH, "r") as f:
            st = json.load(f)
    except Exception as e:
        return f"FEHLER: Status nicht lesbar: {e}"

    tension   = st.get("tension", 0.0)
    fps_raw   = st.get("fps", {})
    fps       = fps_raw.get("total", 0.0) if isinstance(fps_raw, dict) else float(fps_raw)
    watchdog  = st.get("watchdog", {})
    temp      = watchdog.get("cpu_temp", 0.0) if isinstance(watchdog, dict) else 0.0
    ram_pct   = watchdog.get("ram_percent", 0.0) if isinstance(watchdog, dict) else 0.0
    ram_mb    = ram_pct * 4096.0 / 100.0
    face_id   = st.get("face_id", "none")
    face_sim  = st.get("face_similarity", 0.0)
    face_act  = st.get("face_detected", False)
    scenario  = st.get("npu_sched_mode", st.get("npu_stage", "?"))
    bridge    = st.get("bridge", {})
    track_st  = bridge.get("state", "?") if isinstance(bridge, dict) else "?"
    track_mv  = bridge.get("moves_per_minute", 0.0) if isinstance(bridge, dict) else 0.0

    # Schicht-1-Bewertung (Mood-Regel) nachbilden
    mood_eval = "neutral"
    if tension > 0.7 and not face_act:
        mood_eval = "shadow (Tension hoch, niemand da)"
    elif tension < 0.3 and face_act:
        mood_eval = "guardian (Ruhig, Markus ist da)"

    # Schicht-2-Bewertung (Pipeline)
    concerns = []
    if temp > 70.0:
        concerns.append(f"TEMP KRITISCH: {temp:.0f}C")
    elif temp > 65.0:
        concerns.append(f"Temp Warnung: {temp:.0f}C")
    if fps < 10.0:
        concerns.append(f"FPS KRITISCH: {fps:.1f}")
    elif fps < 15.0:
        concerns.append(f"FPS Warnung: {fps:.1f}")
    if ram_mb > 3500:
        concerns.append(f"RAM KRITISCH: {ram_mb:.0f}MB")
    elif ram_mb > 3200:
        concerns.append(f"RAM Warnung: {ram_mb:.0f}MB")
    if face_act and face_id and 0 < face_sim < 0.50:
        concerns.append(f"Gesicht-Sim niedrig: {face_id} sim={face_sim:.2f}")

    lines = [
        "=== UNTERBEWUSSTSEIN — AKTUELLER ZUSTAND ===",
        f"",
        f"[SCHICHT 1 — MOOD]",
        f"  Tension:       {tension:.2f}",
        f"  Gesicht aktiv: {face_act}  (ID: {face_id}, sim={face_sim:.2f})",
        f"  Mood-Eval:     {mood_eval}",
        f"",
        f"[SCHICHT 2 — PIPELINE]",
        f"  FPS:           {fps:.1f}",
        f"  CPU-Temp:      {temp:.1f}C",
        f"  RAM:           {ram_mb:.0f}MB ({ram_pct:.1f}%)",
        f"  NPU-Szenario:  {scenario}",
        f"  Tracking:      {track_st} ({track_mv:.0f} moves/min)",
    ]
    if concerns:
        lines.append("")
        lines.append("[CONCERNS]")
        for c in concerns:
            lines.append(f"  ! {c}")
    else:
        lines.append("")
        lines.append("[CONCERNS] keine — System ruhig")

    return "\n".join(lines)


@mcp.tool()
def uc_get_mood() -> str:
    """Letzter Mood-Impuls aus /dev/shm/moloch_impulse.json.

    Zeigt was das Unterbewusstsein zuletzt 'gefuehlt' und als Impuls
    nach aussen gesendet hat (an PersonalityEngine / MoodEngine).
    """
    try:
        with open(IMPULSE_PATH, "r") as f:
            imp = json.load(f)
    except FileNotFoundError:
        return "Kein Impuls vorhanden (moloch_impulse.json existiert nicht)."
    except Exception as e:
        return f"FEHLER: {e}"

    impulse_type = imp.get("type", "?")
    impulse_val  = imp.get("impulse", imp.get("key", "?"))
    reason       = imp.get("reason", "")
    source       = imp.get("source", "?")
    ts           = imp.get("timestamp", 0)
    age_s        = time.time() - ts if ts else 0

    lines = [
        "=== LETZTER UNTERBEWUSSTSEIN-IMPULS ===",
        f"  Typ:     {impulse_type}",
        f"  Impuls:  {impulse_val}",
        f"  Quelle:  {source}",
        f"  Grund:   {reason}",
        f"  Alter:   {age_s:.0f}s",
    ]
    # Self-Tune hat zusaetzliche Felder
    if impulse_type == "self_tune":
        section  = imp.get("section", "?")
        key      = imp.get("key", "?")
        old_val  = imp.get("old_value", "?")
        new_val  = imp.get("new_value", "?")
        lines.append(f"  Aenderung: {section}.{key}: {old_val} → {new_val}")

    return "\n".join(lines)


@mcp.tool()
def uc_get_history(n: int = 20) -> str:
    """Letzte N Concern-Eintraege aus self_tune.log.

    Zeigt was das Unterbewusstsein in der Vergangenheit beobachtet hat:
    RAM-Trends, FPS-Drops, Temp-Warnungen, Gesicht-Erkennungsprobleme.

    Args:
        n: Anzahl Eintraege (default 20, max 100)
    """
    n = min(max(1, n), 100)
    if not SELF_TUNE_LOG.exists():
        return f"Log-Datei nicht gefunden: {SELF_TUNE_LOG}"
    try:
        lines = SELF_TUNE_LOG.read_text(encoding="utf-8").splitlines()
        recent = lines[-n:] if len(lines) >= n else lines
        if not recent:
            return "Concern-Log ist leer — alles ruhig."
        header = f"=== LETZTE {len(recent)} CONCERNS (von {len(lines)} gesamt) ==="
        return header + "\n" + "\n".join(recent)
    except Exception as e:
        return f"FEHLER beim Lesen: {e}"


@mcp.tool()
def uc_inject_impulse(impulse_type: str, reason: str = "") -> str:
    """Mood-Impuls direkt in MOLOCHs Unterbewusstsein injizieren.

    Schreibt einen Impuls in die IPC-Queue — wird vom MolochService
    als 'core_nudge' verarbeitet. Umgeht den Cooldown der UnconsciousEngine.

    Erlaubte impulse_type Werte:
      shadow   — Spannung, Rueckzug, Misstrauen
      guardian — Ruhe, Schutz, Vertrauen
      reduce   — System unter Stress, Aktivitaet reduzieren
      berserker — Hohe Energie, Aktion, Chaos

    Args:
        impulse_type: Mood-Typ (shadow / guardian / reduce / berserker)
        reason: Optionaler Grund fuer den Impuls
    """
    erlaubt = {"shadow", "guardian", "reduce", "berserker"}
    if impulse_type not in erlaubt:
        return (f"FEHLER: impulse_type '{impulse_type}' ungueltig.\n"
                f"Erlaubt: {', '.join(sorted(erlaubt))}")

    ok = _send_ipc("unconscious_impulse", {
        "impulse": impulse_type,
        "reason": reason or f"Manuell injiziert via MCP (uc_inject_impulse)",
        "source": "claude_mcp",
        "timestamp": time.time(),
    })
    if ok:
        return f"Impuls gesendet: {impulse_type} (Grund: '{reason}')\nMOLOCH verarbeitet via IPC."
    return "FEHLER: IPC-Command konnte nicht geschrieben werden"


@mcp.tool()
def uc_reflect(question: str = "") -> str:
    """Selbstreflexion in MOLOCH ausloesen — mit optionaler Frage.

    Triggert das Introspection-Modul. MOLOCH analysiert seinen eigenen Zustand,
    bewertet Tension/Dominance und spricht ggf. einen Kommentar.
    Die Antwort erscheint im Konversations-Log (via moloch_conversation).

    Args:
        question: Optionale Frage die MOLOCH beantworten soll
                  (z.B. "Wie fuehlt sich dein aktueller Zustand an?")
    """
    params: dict = {}
    if question and question.strip():
        params["question"] = question.strip()

    ok = _send_ipc("trigger_reflect", params)
    if ok:
        msg = "Reflect-Trigger gesendet"
        if question:
            msg += f" mit Frage: '{question.strip()[:80]}'"
        msg += "\n→ MOLOCH reflektiert async. Antwort via moloch_conversation() lesen."
        return msg
    return "FEHLER: IPC-Command konnte nicht geschrieben werden"


if __name__ == "__main__":
    mcp.run(transport="stdio")
