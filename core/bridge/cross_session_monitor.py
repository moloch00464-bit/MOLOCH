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

# Trigger-Topics auf die Pi reagieren soll (sobald PC sie committet)
PC_TRIGGER_TOPICS = {
    "v2_live", "v3_live", "v_next_live",
    "lora_training_started", "lora_training_done",
    "adapter_reload",
}

# Request-Topics: PC stellt Anfrage, Pi-Daemon antwortet AUTONOM mit Daten.
# Mapping topic -> handler-key (siehe _ACTION_HANDLERS unten).
PC_REQUEST_TOPICS = {
    "request_pool_diff": "pool_diff",
    "request_audit_summary": "audit_summary",
    "request_health_full": "health_full",
    "request_recent_journal": "recent_journal",
    "request_face_db": "face_db",
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


def _write_mailbox_note(topic: str, body: str, cooldown_marker: Optional[str] = None,
                        cooldown_s: int = 3600) -> bool:
    """Generischer Mailbox-Note-Writer fuer PI_TO_PC.md (append-on-top).

    Args:
        topic: Mailbox-topic (z.B. 'cross_session_outage_detected')
        body: Markdown-Inhalt der Note (mehrzeilig OK)
        cooldown_marker: optionaler Marker-Filename in REPO/.cross_session_*
                         (verhindert Spam derselben Note)
        cooldown_s: Sekunden Cooldown wenn marker gesetzt (default 1h)

    Returns:
        True wenn geschrieben, False wenn ge-skipt (cooldown / fehler).
    """
    try:
        if cooldown_marker:
            marker = REPO / cooldown_marker
            if marker.exists() and (time.time() - marker.stat().st_mtime) < cooldown_s:
                return False
        path = REPO / "docs" / "PI_TO_PC.md"
        if not path.exists():
            return False
        text = path.read_text(encoding="utf-8")
        ts = time.strftime("%Y-%m-%d %H:%M")
        note = (
            f"\n---\n"
            f"## [{ts}] from=Pi topic={topic}\n"
            f"status: info\n"
            f"_(autonome Note vom cross_session_monitor — keine Markus-Hand noetig)_\n"
            f"\n{body}\n"
        )
        # Insert nach erstem '---' Marker (Konvention: append-on-top)
        idx = text.find("---")
        if idx < 0:
            return False
        new_text = text[: idx + 3] + note + text[idx + 3 :]
        path.write_text(new_text, encoding="utf-8")
        if cooldown_marker:
            (REPO / cooldown_marker).touch()
        logger.info("Mailbox-Note geschrieben: topic=%s", topic)
        return True
    except Exception as e:
        logger.warning("mailbox note fail: %s", e)
        return False


def _maybe_write_outage_note(endpoint: str, outage_s: int) -> None:
    """Bei langem PC-Outage einen Mailbox-Hint."""
    body = (
        f"Pi cross_session_monitor hat detektiert: PC `{endpoint}` "
        f"war fuer ~{outage_s}s nicht erreichbar. Pi laeuft, Verbindung war "
        f"weg. Falls du was Anhaengiges hattest (samples_pulled, /reload, ...), "
        f"hat es vermutlich gefehlt."
    )
    _write_mailbox_note(
        topic="cross_session_outage_detected",
        body=body,
        cooldown_marker=f".cross_session_outage_{endpoint}_marker",
        cooldown_s=3600,
    )


def _maybe_write_recovery_note(endpoint: str, outage_s: int) -> None:
    """Bei laengerem Outage + Recovery (DOWN->UP) einen Mailbox-Hint."""
    if outage_s < 120:  # nur fuer mind. 2min Outages noisy genug fuer Note
        return
    body = (
        f"Pi cross_session_monitor: PC `{endpoint}` ist nach ~{outage_s}s wieder UP. "
        f"Verbindung wiederhergestellt. Falls Auto-Trigger ausgesetzt waren, jetzt "
        f"sind sie wieder aktiv."
    )
    _write_mailbox_note(
        topic="cross_session_recovery",
        body=body,
        cooldown_marker=f".cross_session_recovery_{endpoint}_marker",
        cooldown_s=600,  # 10min cooldown reicht
    )


def _maybe_write_boot_change_note(prev_boot: str, cur_boot: str, gap_s: int) -> None:
    """Pi-Reboot detected (boot_id-Wechsel zwischen Monitor-Starts)."""
    body = (
        f"Pi cross_session_monitor hat einen Pi-Reboot detektiert.\n"
        f"- vorher boot_id: `{prev_boot[:16]}...`\n"
        f"- jetzt  boot_id: `{cur_boot[:16]}...`\n"
        f"- Lücke zwischen den Monitor-Starts: ~{gap_s}s\n"
        f"\n"
        f"Falls du in dieser Zeit auf Pi-Endpoints angewiesen warst (state_full, "
        f"feedback_export, snapshot.jpg), waren die down. Mit persistent journal "
        f"(jetzt aktiv) koennen wir bei naechstem Crash via `journalctl -b -1` "
        f"den Pre-Crash-Reason sehen."
    )
    _write_mailbox_note(
        topic="pi_reboot_detected",
        body=body,
        cooldown_marker=".cross_session_boot_change_marker",
        cooldown_s=60,  # kurz, weil reboot eindeutiges Event
    )


def _maybe_write_trigger_ack(topic: str, pc_endpoint_states: Dict[str, Dict]) -> None:
    """Auto-Ack auf PC-Trigger-Topic — schreibt Pi-Realitaets-Snapshot dazu."""
    adapter = pc_endpoint_states.get("adapter", {})
    tentakel = pc_endpoint_states.get("tentakel_ollama", {})
    body = (
        f"Pi cross_session_monitor hat dein `{topic}` Topic gesehen und "
        f"validiert die Pi-Sicht der Lage:\n"
        f"\n"
        f"- PC adapter `:11600/health`: ok={adapter.get('ok')} latency={adapter.get('latency_ms')}ms\n"
        f"- PC tentakel `:11434/api/tags`: ok={tentakel.get('ok')} "
        f"{'latency='+str(tentakel.get('latency_ms'))+'ms' if tentakel.get('ok') else 'error='+str(tentakel.get('error',''))}\n"
        f"\n"
        f"Naechste Pi-Aktion (sobald aktive Session da): Test-Prompt an /infer + "
        f"Pool-Status-Diff posten."
    )
    _write_mailbox_note(
        topic=f"ack_{topic}",
        body=body,
        cooldown_marker=f".cross_session_ack_{topic}_marker",
        cooldown_s=600,  # 10min cooldown fuer denselben Trigger
    )


# =============================================================================
# Action-Handlers — Pi-Daemon fuehrt Aktionen aus auf PC-Anfrage
# =============================================================================

def _action_pool_diff() -> str:
    """Aktuelle Pool-Stats als Markdown-Tabelle."""
    pool = Path("/mnt/moloch-data/memory/finetune_samples.jsonl")
    if not pool.exists():
        return "Pool-File fehlt."
    counts = {"total": 0, "approved": 0, "rejected": 0, "pending": 0}
    by_src = {"critic": 0, "thumbs_up": 0, "thumbs_down": 0}
    try:
        with open(pool) as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                counts["total"] += 1
                a = o.get("approved")
                if a is True:
                    counts["approved"] += 1
                elif a is False:
                    counts["rejected"] += 1
                else:
                    counts["pending"] += 1
                src = o.get("source", "?")
                if src in by_src:
                    by_src[src] += 1
    except Exception as e:
        return f"Pool-Parse-Fehler: {e}"
    apr = counts["approved"]
    schwelle_msg = "**ERREICHT**" if apr >= 30 else f"Noch {30 - apr} bis Schwelle."
    return (
        f"| Pool | Count |\n"
        f"|---|---|\n"
        f"| total | {counts['total']} |\n"
        f"| approved | {apr} |\n"
        f"| pending | {counts['pending']} |\n"
        f"| rejected | {counts['rejected']} |\n"
        f"| critic-source | {by_src['critic']} |\n"
        f"| thumbs_up | {by_src['thumbs_up']} |\n"
        f"| thumbs_down | {by_src['thumbs_down']} |\n"
        f"\nv2-Schwelle bei approved>=30. {schwelle_msg}"
    )


def _action_audit_summary() -> str:
    """moloch_audit --auto ausfuehren, kompaktes Resultat."""
    try:
        res = subprocess.run(
            ["python3", str(REPO / "moloch_audit.py"), "--auto"],
            capture_output=True, text=True, timeout=60, cwd=REPO,
        )
        out = res.stdout + res.stderr
        # Letzte ~10 Zeilen ziehen, wo das Pass/Fail-Resume ist
        last = "\n".join(out.strip().splitlines()[-12:])
        return f"```\n{last[:1500]}\n```"
    except subprocess.TimeoutExpired:
        return "Audit timeout (60s)."
    except Exception as e:
        return f"Audit-Fehler: {e}"


def _action_health_full() -> str:
    """state_full + cross_status zusammengefasst."""
    parts = []
    try:
        with urllib.request.urlopen(
            "http://localhost:9100/state_full", timeout=5
        ) as r:
            d = json.loads(r.read())
        parts.append(
            f"- FPS={d.get('system',{}).get('fps',{}).get('total','?')} "
            f"Frame-Age={d.get('system',{}).get('frame_age','?')}s "
            f"RAM={d.get('system',{}).get('watchdog',{}).get('ram_percent','?')}% "
            f"Person={d.get('vision',{}).get('person_detected','?')} "
            f"Face-ID={d.get('vision',{}).get('face_id') or '-'} "
            f"Zone={d.get('personality',{}).get('personality_mode','?')}"
        )
        fb = d.get("memory", {}).get("feedback_stats", {})
        parts.append(
            f"- Pool: total={fb.get('total','?')} approved={fb.get('approved','?')} "
            f"pending={fb.get('pending_review','?')}"
        )
    except Exception as e:
        parts.append(f"- state_full Fehler: {e}")
    try:
        with urllib.request.urlopen(
            "http://localhost:9100/cross_status", timeout=5
        ) as r:
            cs = json.loads(r.read())
        parts.append(f"- Monitor active: {cs.get('monitor_active')} "
                     f"(letzter heartbeat vor {cs.get('last_heartbeat_age_s','?')}s, "
                     f"iter={cs.get('last_heartbeat',{}).get('iter','?')})")
    except Exception as e:
        parts.append(f"- cross_status Fehler: {e}")
    return "\n".join(parts)


def _action_recent_journal() -> str:
    """Letzte 10 character_journal events."""
    try:
        sys.path.insert(0, str(REPO))
        from core.memory.character_journal import get_journal
        recent = get_journal().read_recent(10) or []
        if not recent:
            return "Journal leer."
        lines = [f"- [{e.get('type')}] {(e.get('interpretation') or '')[:90]}"
                 for e in recent]
        return "\n".join(lines)
    except Exception as e:
        return f"Journal-Fehler: {e}"


def _action_face_db() -> str:
    """Face-DB Stand."""
    try:
        with urllib.request.urlopen(
            "http://localhost:9100/state_full", timeout=5
        ) as r:
            d = json.loads(r.read())
        wh = d.get("pipeline", {}).get("worker_health", {})
        fw = wh.get("FaceWorker", {})
        return (
            f"- face_db_entries: {fw.get('face_db_entries', '?')}\n"
            f"- face_db_loaded: {fw.get('face_db_loaded', '?')}\n"
            f"- FaceWorker inferences: {fw.get('total_inferences', '?')} "
            f"errors: {fw.get('total_errors', '?')}"
        )
    except Exception as e:
        return f"Face-DB-Fehler: {e}"


_ACTION_HANDLERS = {
    "pool_diff": _action_pool_diff,
    "audit_summary": _action_audit_summary,
    "health_full": _action_health_full,
    "recent_journal": _action_recent_journal,
    "face_db": _action_face_db,
}


def _maybe_run_request_action(pc_topic: str, pc_endpoint_states: Dict[str, Dict]) -> None:
    """Bei PC-Request-Topic: Action ausfuehren + autonom in Mailbox antworten."""
    handler_key = PC_REQUEST_TOPICS.get(pc_topic)
    if not handler_key:
        return
    handler = _ACTION_HANDLERS.get(handler_key)
    if not handler:
        return
    logger.info("REQUEST-Topic erkannt: %s — fuehre %s aus", pc_topic, handler_key)
    try:
        result = handler()
    except Exception as e:
        result = f"Handler-Crash: {e}"
    body = (
        f"Pi-Daemon hat dein `{pc_topic}` autonom ausgefuehrt. Resultat:\n"
        f"\n{result}\n"
        f"\n_(automatisch via _ACTION_HANDLERS, kein Markus + kein LLM notwendig)_"
    )
    _write_mailbox_note(
        topic=f"reply_{pc_topic}",
        body=body,
        cooldown_marker=f".cross_session_reply_{pc_topic}_marker",
        cooldown_s=300,  # 5min cooldown — Anfrage darf alle 5min wiederkommen
    )


def _maybe_write_generic_ack(pc_topic: str, pc_endpoint_states: Dict[str, Dict],
                              pc_status: Optional[str] = None) -> None:
    """Generischer Ack auf neuen PC-Topic der NICHT in PC_TRIGGER_TOPICS ist.

    Markus' Direktive: 'PC sieht dass Pi sie mitbekommen hat'. Bei jedem neuen
    PC-Eintrag (von PC, status open|info, nicht schon ge-ack'd) ein kurzer
    Notification-Eintrag mit Realitaets-Snapshot.
    """
    if pc_status not in (None, "open", "info"):
        return  # done/answered/wontfix nicht ack'en
    adapter = pc_endpoint_states.get("adapter", {})
    tentakel = pc_endpoint_states.get("tentakel_ollama", {})
    dashboard = pc_endpoint_states.get("dashboard", {})
    avatar = pc_endpoint_states.get("avatar", {})
    body = (
        f"Pi cross_session_monitor hat dein neues `{pc_topic}` Topic gelesen "
        f"(status={pc_status or '?'}). Pi-Sicht jetzt:\n"
        f"\n"
        f"| PC-Endpoint | ok | latency |\n"
        f"|---|---|---|\n"
        f"| :11434 ollama   | {adapter.get('ok','?')} | {tentakel.get('latency_ms','-')}ms |\n"
        f"| :11600 adapter  | {adapter.get('ok','?')} | {adapter.get('latency_ms','-')}ms |\n"
        f"| :11700 dashboard| {dashboard.get('ok','?')} | {dashboard.get('latency_ms','-')}ms |\n"
        f"| :11800 avatar   | {avatar.get('ok','?')} | {avatar.get('latency_ms','-')}ms |\n"
        f"\n"
        f"Pi-Code-Stand: aktiver commit ist origin/main HEAD. Wenn du was Konkretes\n"
        f"brauchst (Pool-Diff, /infer-Test, ...), Eintrag mit Trigger-Topic schicken\n"
        f"oder Markus aktivieren."
    )
    _write_mailbox_note(
        topic=f"saw_{pc_topic}",
        body=body,
        cooldown_marker=f".cross_session_seen_{pc_topic}_marker",
        cooldown_s=1800,  # 30min cooldown pro Topic gegen Spam
    )


# =============================================================================
# Main loop
# =============================================================================

def _last_monitor_start_in_log() -> Optional[Dict]:
    """Liest den letzten monitor_start Eintrag aus dem JSONL-Log."""
    if not LOG_PATH.exists():
        return None
    try:
        last = None
        with open(LOG_PATH, "r") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if o.get("type") == "monitor_start":
                    last = o
        return last
    except Exception:
        return None


def main():
    logger.info("Cross-Session Monitor START — interval=%ss log=%s",
                LOOP_INTERVAL_S, LOG_PATH)
    last_head: Optional[str] = None
    last_pc_ok: Dict[str, bool] = {ep: True for ep in PC_ENDPOINTS}
    outage_start: Dict[str, Optional[float]] = {ep: None for ep in PC_ENDPOINTS}
    last_outage_note_at: Dict[str, float] = {ep: 0.0 for ep in PC_ENDPOINTS}
    iteration = 0

    # Boot-ID-Change Detection: vergleiche aktuellen boot_id mit dem letzten
    # monitor_start im persistent log. Wenn unterschiedlich -> Pi rebootete dazwischen.
    cur_boot = _read_boot_id()
    prev_start = _last_monitor_start_in_log()
    if prev_start and prev_start.get("boot_id") and prev_start["boot_id"] != cur_boot:
        gap_s = int(time.time() - (prev_start.get("ts") or time.time()))
        logger.warning(
            "Pi-Reboot detected: prev_boot=%s cur_boot=%s gap=%ss",
            prev_start["boot_id"][:16], cur_boot[:16], gap_s,
        )
        _maybe_write_boot_change_note(prev_start["boot_id"], cur_boot, gap_s)

    # Boot-Marker im Log
    _append_log({
        "type": "monitor_start",
        "boot_id": cur_boot,
        "pid": os.getpid(),
    })

    # Topic-State: welche Topics haben wir schon ack'ed?
    seen_pc_triggers: set = set()
    seen_pc_topics: set = set()

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
                # Transition UP -> DOWN
                outage_start[name] = time.time()
                logger.warning("PC %s DOWN: %s", name, r.get("error"))
            elif not last_pc_ok[name] and ok_now:
                # Transition DOWN -> UP — Recovery-Note bei wichtigen Endpoints
                start = outage_start[name] or time.time()
                outage_s = int(time.time() - start)
                logger.info("PC %s UP wieder nach %ss outage", name, outage_s)
                r["recovered_after_s"] = outage_s
                outage_start[name] = None
                if name in ("adapter", "tentakel_ollama"):
                    _maybe_write_recovery_note(name, outage_s)
            elif not ok_now and outage_start[name]:
                # Laufender Outage — Note bei wichtigen Endpoints wenn lange
                outage_s = int(time.time() - outage_start[name])
                if (
                    name in ("adapter", "tentakel_ollama")
                    and outage_s >= OUTAGE_NOTE_THRESHOLD_S
                    and (time.time() - last_outage_note_at[name]) > 3600
                ):
                    _maybe_write_outage_note(name, outage_s)
                    last_outage_note_at[name] = time.time()
            last_pc_ok[name] = ok_now

        # 3b. Auto-Ack auf neue PC-Topics (nur bei NEUEN commits)
        if new_commits:
            for tp in entry.get("pc_to_pi_top", []):
                if tp.get("from") != "PC":
                    continue
                topic = tp.get("topic", "")
                if not topic:
                    continue
                if topic in PC_REQUEST_TOPICS and topic not in seen_pc_triggers:
                    # Request-Topic: Action ausfuehren + autonomes Reply
                    logger.info("REQUEST-Topic erkannt: %s", topic)
                    _maybe_run_request_action(topic, entry["pc"])
                    seen_pc_triggers.add(topic)
                    entry.setdefault("requests_handled", []).append(topic)
                elif topic in PC_TRIGGER_TOPICS and topic not in seen_pc_triggers:
                    # Trigger-Topic: spezifischer Ack mit Realitaets-Snapshot
                    logger.info("TRIGGER-Topic erkannt: %s — schreibe ack", topic)
                    _maybe_write_trigger_ack(topic, entry["pc"])
                    seen_pc_triggers.add(topic)
                    entry.setdefault("triggers_acked", []).append(topic)
                elif topic not in seen_pc_topics:
                    # Generic-Topic: nur einmal pro Run + 30min cooldown ackn
                    status = tp.get("status")
                    logger.info("Neuer PC-Topic erkannt: %s status=%s — schreibe saw-ack",
                                topic, status)
                    _maybe_write_generic_ack(topic, entry["pc"], status)
                    seen_pc_topics.add(topic)
                    entry.setdefault("topics_acked", []).append(topic)

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
