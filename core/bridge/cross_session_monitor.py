#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Cross-Session Monitor (Pi-Side)

Haelt die Pi <-> PC Verbindung autonom synchron.

Loop alle 30s:
  1) git fetch im ~/moloch-repo, log neue Commits + Topic-Liste
  2) Pi self-check (chat_server :9100/health)
  3) PC endpoints pingen (Adapter :11600, Dashboard :11700, Avatar :11800)
  4) Heartbeat in /mnt/moloch-data/memory/cross_session_log.jsonl persistieren
  5) State-Transitions (UP->DOWN, DOWN->UP) ins Journal loggen
  6) Auto-Reaction auf Trigger-Topics aus PC_TO_PI.md (Hooks pluggable)

Run als systemd-Service moloch-cross-monitor.service.
Stop via SIGTERM (sauberer Loop-Exit).

Outage-Detection:
  - Wenn PC :11600 fuer >120s nicht antwortet -> Mailbox-Note schreiben
    (markiert die Zeit ab wann Outage begann)
  - Bei Pi-Reboot setzt persistent journal jetzt die Logs ueber Boots fort
    (gefixt heute via /etc/systemd/journald.conf.d/persistent.conf)

Trigger-Topics (Pi reagiert wenn PC sowas committet):
  - v2_live / v3_live -> ping :11600/list, log Adapter-Inventur
  - lora_training_started -> erhoehter Poll auf /list im naechsten Loop
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(os.path.expanduser("~/moloch"))
LOG_PATH = Path("/mnt/moloch-data/memory/cross_session_log.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

PC_HOST = "192.168.178.20"
PC_ENDPOINTS = {
    "tentakel_ollama": f"http://{PC_HOST}:11434/api/tags",  # PC-Ollama (Mistral/Tentakel + Critic)
    "adapter": f"http://{PC_HOST}:11600/health",            # LoRA-Adapter-Inference (Qwen+v1)
    "dashboard": f"http://{PC_HOST}:11700/api/state",       # Trainings-Dashboard
    "avatar": f"http://{PC_HOST}:11800/api/state",          # 3D-Avatar (Three.js)
}

PI_SELF = "http://localhost:9100/health"
LOOP_INTERVAL_S = 30
HTTP_TIMEOUT_S = 4
OUTAGE_NOTE_THRESHOLD_S = 120  # nach 2 min Outage ein Mailbox-Hint

# Topics auf die Pi reagieren soll (sobald PC sie committet)
PC_TRIGGER_TOPICS = {
    "v2_live", "v3_live", "v_next_live",
    "lora_training_started", "lora_training_done",
    "adapter_reload",
}

logger = logging.getLogger("cross-monitor")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
logger.addHandler(_h)

_running = True


def _sigterm(*_):
    global _running
    _running = False
    logger.info("SIGTERM/SIGINT empfangen — beende loop")


signal.signal(signal.SIGTERM, _sigterm)
signal.signal(signal.SIGINT, _sigterm)


# =============================================================================
# Helpers
# =============================================================================

def _git_fetch_and_diff(last_sha: Optional[str]) -> Dict:
    """git fetch -q + return new commits seit last_sha."""
    try:
        subprocess.run(
            ["git", "fetch", "-q", "origin", "main"],
            cwd=REPO, timeout=15, check=True,
        )
        cur_sha = subprocess.run(
            ["git", "rev-parse", "origin/main"],
            cwd=REPO, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        new_commits: List[str] = []
        if last_sha and cur_sha != last_sha:
            res = subprocess.run(
                ["git", "log", "--oneline", f"{last_sha}..{cur_sha}"],
                cwd=REPO, capture_output=True, text=True, timeout=5,
            )
            if res.returncode == 0:
                new_commits = [ln for ln in res.stdout.strip().splitlines() if ln]
        return {"head": cur_sha, "new_commits": new_commits}
    except subprocess.TimeoutExpired:
        return {"head": last_sha, "new_commits": [], "error": "fetch_timeout"}
    except Exception as e:
        return {"head": last_sha, "new_commits": [], "error": str(e)[:80]}


def _ping(url: str) -> Dict:
    """HTTP-GET, return {ok, status_code, latency_ms, error?}."""
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "moloch-cross-monitor"}
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            resp.read(1024)  # drain
            return {
                "ok": True,
                "status_code": resp.status,
                "latency_ms": int((time.monotonic() - t0) * 1000),
            }
    except urllib.error.URLError as e:
        return {
            "ok": False,
            "error": str(getattr(e, "reason", e))[:80],
            "latency_ms": int((time.monotonic() - t0) * 1000),
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)[:80],
            "latency_ms": int((time.monotonic() - t0) * 1000),
        }


def _append_log(entry: Dict) -> None:
    """JSONL-Append, atomic-genug fuer Single-Writer-Monitor."""
    entry["ts"] = time.time()
    entry["iso"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"log write fail: {e}")


def _parse_mailbox_topics(file: str, n: int = 4) -> List[Dict]:
    """Top-N Eintraege aus Mailbox: liefert {ts, from, topic, status} pro Header."""
    try:
        path = REPO / "docs" / file
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8")
        entries: List[Dict] = []
        cur: Optional[Dict] = None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("## ["):
                # Naechster Header gestartet — alten pushen
                if cur:
                    entries.append(cur)
                    if len(entries) >= n:
                        break
                cur = {"raw_header": stripped}
                # parse "## [TS] from=X topic=Y..."
                if "from=" in stripped:
                    cur["from"] = stripped.split("from=", 1)[1].split(" ", 1)[0]
                if "topic=" in stripped:
                    cur["topic"] = stripped.split("topic=", 1)[1].split(" ", 1)[0]
                if stripped.startswith("## ["):
                    end = stripped.find("]")
                    if end > 0:
                        cur["ts"] = stripped[3:end]
            elif cur is not None and stripped.startswith("status:"):
                cur["status"] = stripped.split(":", 1)[1].strip().split(" ", 1)[0]
        if cur and len(entries) < n:
            entries.append(cur)
        return entries
    except Exception as e:
        return [{"parse_error": str(e)[:80]}]


def _maybe_write_outage_note(endpoint: str, outage_s: int) -> None:
    """Bei langem PC-Outage einen Mailbox-Hint in PI_TO_PC.md schreiben.

    Best-effort: bei Push-Fehler nur ins Journal loggen.
    Idempotent: dieselbe Note nicht 2x schreiben (kurze Cooldown via env-state).
    """
    try:
        marker = REPO / ".cross_session_outage_marker"
        # Cooldown 1h zwischen Notes
        if marker.exists() and (time.time() - marker.stat().st_mtime) < 3600:
            return
        path = REPO / "docs" / "PI_TO_PC.md"
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8")
        ts = time.strftime("%Y-%m-%d %H:%M")
        note = (
            f"\n---\n"
            f"## [{ts}] from=Pi topic=cross_session_outage_detected\n"
            f"status: info\n"
            f"\n"
            f"Pi cross_session_monitor hat detektiert: PC `{endpoint}` "
            f"war fuer ~{outage_s}s nicht erreichbar. Pi laeuft, Verbindung war "
            f"weg. Falls du was Anhaengiges hattest (samples_pulled, /reload, ...), "
            f"hat es vermutlich gefehlt. Schau in dein Heartbeat-Log "
            f"(falls schon implementiert) fuer Details.\n"
        )
        # Insert nach erstem '---' Marker (Konvention: append-on-top)
        idx = text.find("---")
        if idx < 0:
            return
        new_text = text[: idx + 3] + note + text[idx + 3 :]
        path.write_text(new_text, encoding="utf-8")
        marker.touch()
        # Nicht selber commiten — Markus oder die naechste aktive Session committen
        logger.info(f"Outage-Note in PI_TO_PC.md geschrieben ({endpoint}, {outage_s}s)")
    except Exception as e:
        logger.warning(f"outage note fail: {e}")


# =============================================================================
# Main loop
# =============================================================================

def main():
    logger.info("Cross-Session Monitor START — interval=%ss log=%s",
                LOOP_INTERVAL_S, LOG_PATH)
    last_head: Optional[str] = None
    last_pc_ok: Dict[str, bool] = {ep: True for ep in PC_ENDPOINTS}
    outage_start: Dict[str, Optional[float]] = {ep: None for ep in PC_ENDPOINTS}
    last_outage_note_at: Dict[str, float] = {ep: 0.0 for ep in PC_ENDPOINTS}
    iteration = 0

    # Boot-Marker im Log
    _append_log({
        "type": "monitor_start",
        "boot_id": _read_boot_id(),
        "pid": os.getpid(),
    })

    while _running:
        iteration += 1
        cycle_start = time.time()
        entry: Dict = {"type": "heartbeat", "iter": iteration}

        # 1. Git
        git_info = _git_fetch_and_diff(last_head)
        entry["git"] = git_info
        new_commits = git_info.get("new_commits") or []
        if new_commits:
            logger.info("Neue Commits (%d): %s", len(new_commits), new_commits[:3])
            entry["pc_to_pi_top"] = _parse_mailbox_topics("PC_TO_PI.md")
            entry["pi_to_pc_top"] = _parse_mailbox_topics("PI_TO_PC.md")
            # Trigger-Topic-Erkennung (best-effort)
            for tp in entry.get("pc_to_pi_top", []):
                if tp.get("from") == "PC" and tp.get("topic") in PC_TRIGGER_TOPICS:
                    logger.info("TRIGGER-Topic erkannt: %s", tp.get("topic"))
                    entry.setdefault("triggers", []).append(tp.get("topic"))
        last_head = git_info.get("head") or last_head

        # 2. Pi-Self-check (Chat-Server)
        entry["pi_self"] = _ping(PI_SELF)

        # 3. PC-Endpoints
        entry["pc"] = {}
        for name, url in PC_ENDPOINTS.items():
            r = _ping(url)
            entry["pc"][name] = r
            ok_now = r.get("ok", False)
            if last_pc_ok[name] and not ok_now:
                outage_start[name] = time.time()
                logger.warning("PC %s DOWN: %s", name, r.get("error"))
            elif not last_pc_ok[name] and ok_now:
                start = outage_start[name] or time.time()
                outage_s = int(time.time() - start)
                logger.info("PC %s UP wieder nach %ss outage", name, outage_s)
                r["recovered_after_s"] = outage_s
                outage_start[name] = None
            elif not ok_now and outage_start[name]:
                # Laufender Outage — bei adapter-Endpoint Note schreiben wenn lange
                outage_s = int(time.time() - outage_start[name])
                if (
                    name == "adapter"
                    and outage_s >= OUTAGE_NOTE_THRESHOLD_S
                    and (time.time() - last_outage_note_at[name]) > 3600
                ):
                    _maybe_write_outage_note(name, outage_s)
                    last_outage_note_at[name] = time.time()
            last_pc_ok[name] = ok_now

        # 4. Log
        _append_log(entry)

        # 5. Sleep signal-aware
        elapsed = time.time() - cycle_start
        sleep_s = max(0.0, LOOP_INTERVAL_S - elapsed)
        end_at = time.time() + sleep_s
        while time.time() < end_at and _running:
            time.sleep(0.5)

    logger.info("Cross-Session Monitor STOP nach %d Iterationen", iteration)
    _append_log({"type": "monitor_stop", "iters": iteration})
    return 0


def _read_boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except Exception:
        return ""


if __name__ == "__main__":
    sys.exit(main())
