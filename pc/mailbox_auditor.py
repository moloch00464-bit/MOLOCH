"""MOLOCH Mailbox-Auditor + PC-Health-Reporter (Welle 9 + W8-Beitrag).

Zwei Aufgaben in einem 5-min-Loop:

1. PC-Health-Snapshot
   - Ruft pc/moloch_health_check.py --json auf
   - POSTet Ergebnis als layer.pc an Pi audit-Orchestrator
     -> POST http://192.168.178.30:9100/mailbox/audit/pc_health

2. Mailbox-Hygiene
   - Pullt PC_TO_PI.md + PI_TO_PC.md via HTTP-API
   - Regeln:
     * stale: status=open AND age > 24h
     * dup: gleicher Topic-Name innerhalb der letzten 50 Eintraege
     * backlog: > 200 Eintraege gesamt
     * archive_trigger: > 500 Eintraege
   - POSTet Befund als layer.mailbox an Pi
     -> POST http://192.168.178.30:9100/mailbox/audit/hygiene

Reboot-persistent via Startup-Folder-Shortcut MolochMailboxAuditor.lnk.
Aufruf direkt: python pc/mailbox_auditor.py [--once] [--interval-s N]

NEVER-Regeln respektiert:
- subprocess timeout=30
- atomic state-Write via tempfile + os.replace (Cache-File)
- KEIN shell=True
- API-Keys nicht in Logs
"""
import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mailbox-auditor")

PI_BASE = os.environ.get("MOLOCH_PI_CHAT", "http://192.168.178.30:9100")
HEALTH_CHECK = Path(__file__).parent / "moloch_health_check.py"
PYTHON_EXE = os.environ.get(
    "MOLOCH_PYTHON",
    str(Path.home() / "moloch_pc_env" / "Scripts" / "python.exe"),
)
STATE_DIR = Path.home() / "moloch_logs" / "audit"
STATE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INTERVAL_S = 300  # 5 min
HEADERS = {"Content-Type": "application/json"}
TIMEOUT_S = 8

STALE_HOURS = 24
DUP_WINDOW = 50
BACKLOG_WARN = 200
ARCHIVE_TRIGGER = 500

TOPIC_RE = re.compile(
    r"^## \[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] from=(\w+) topic=(\S+)", re.M
)
STATUS_RE = re.compile(r"^status:\s*(\w+)", re.M)


def safe_json_write(path: Path, data: dict) -> None:
    """Atomic JSON-Write — NEVER-Regel 6."""
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


def parse_mailbox(text: str, n: int = 100) -> list[dict]:
    """Parse top-N entries aus Mailbox-Markdown."""
    entries: list[dict] = []
    for sec in text.split("\n## [")[1 : n + 1]:
        head = "## [" + sec.split("\n", 1)[0]
        body = sec.split("\n", 1)[1] if "\n" in sec else ""
        tm = TOPIC_RE.match(head)
        sm = STATUS_RE.search(body[:300])
        if tm:
            entries.append({
                "ts": tm.group(1),
                "from": tm.group(2),
                "topic": tm.group(3),
                "status": sm.group(1) if sm else "?",
                "body_hash": hashlib.sha256(body[:500].encode("utf-8")).hexdigest()[:16],
            })
    return entries


def audit_mailbox(box_name: str) -> dict:
    """Audit eine Mailbox: stale, dups, backlog."""
    try:
        r = requests.get(f"{PI_BASE}/mailbox/{box_name}", timeout=TIMEOUT_S)
        r.raise_for_status()
        text = r.text
    except requests.RequestException as e:
        logger.warning(f"[mailbox-fetch] {box_name} FAIL: {e}")
        return {"error": str(e)[:80]}

    entries = parse_mailbox(text, n=100)
    total = entries.__len__()  # Top-100 only fuer perf
    stale_cutoff = datetime.now() - timedelta(hours=STALE_HOURS)
    stale = []
    for e in entries[:DUP_WINDOW]:
        if e["status"] != "open":
            continue
        try:
            ts = datetime.strptime(e["ts"], "%Y-%m-%d %H:%M")
            if ts < stale_cutoff:
                stale.append({"topic": e["topic"], "age_h": int((datetime.now() - ts).total_seconds() / 3600)})
        except ValueError:
            continue

    # Duplikate: gleicher topic-Name UND gleicher body-hash in window
    seen_hashes: dict[str, list[str]] = {}
    for e in entries[:DUP_WINDOW]:
        key = f"{e['topic']}|{e['body_hash']}"
        seen_hashes.setdefault(key, []).append(e["ts"])
    dups = [
        {"topic": k.split("|")[0], "ts_list": v}
        for k, v in seen_hashes.items()
        if len(v) > 1
    ]

    # Total-Count via grep statt parse_mailbox (perf)
    total_actual = text.count("\n## [")

    return {
        "box": box_name,
        "total": total_actual,
        "open_count": sum(1 for e in entries if e["status"] == "open"),
        "stale_count": len(stale),
        "stale_topics": stale[:5],
        "dup_count": len(dups),
        "dup_topics": dups[:5],
        "backlog_warn": total_actual > BACKLOG_WARN,
        "archive_needed": total_actual > ARCHIVE_TRIGGER,
    }


def collect_pc_health() -> dict:
    """PC-Health via moloch_health_check.py --json."""
    if not HEALTH_CHECK.exists():
        return {"error": "moloch_health_check.py fehlt", "score": 0, "max": 0, "status": "FAIL"}
    try:
        r = subprocess.run(
            [PYTHON_EXE, str(HEALTH_CHECK), "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode > 1:  # 0=PASS, 1=FAIL, 2=WARN-only
            logger.warning(f"[health] non-zero exit {r.returncode}")
        data = json.loads(r.stdout) if r.stdout.strip() else {}
        summary = data.get("summary", {})
        passed = summary.get("PASS", 0)
        warned = summary.get("WARN", 0)
        failed = summary.get("FAIL", 0)
        total = passed + warned + failed
        if failed > 0:
            status = "FAIL"
        elif warned > 3:
            status = "WARN"
        else:
            status = "PASS"
        return {
            "score": passed,
            "max": total,
            "status": status,
            "detail": {"PASS": passed, "WARN": warned, "FAIL": failed, "duration_s": data.get("duration_s")},
        }
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        return {"error": str(e)[:80], "score": 0, "max": 0, "status": "FAIL"}


def post_audit(component: str, payload: dict) -> bool:
    """POST an Pi audit-Orchestrator."""
    url = f"{PI_BASE}/mailbox/audit/{component}"
    try:
        r = requests.post(url, headers=HEADERS, json=payload, timeout=TIMEOUT_S)
        if r.status_code == 200:
            logger.info(f"[post] {component} -> ok")
            return True
        logger.warning(f"[post] {component} HTTP {r.status_code}: {r.text[:120]}")
    except requests.RequestException as e:
        logger.warning(f"[post] {component} FAIL: {e}")
    return False


def tick() -> dict:
    """Ein Audit-Tick: PC-Health + beide Mailboxen + POST an Pi."""
    started = time.time()
    pc = collect_pc_health()
    mb_pc_to_pi = audit_mailbox("PC_TO_PI")
    mb_pi_to_pc = audit_mailbox("PI_TO_PC")

    # Hygiene-Aggregate
    hyg_status = "PASS"
    if any(e.get("error") for e in (mb_pc_to_pi, mb_pi_to_pc)):
        hyg_status = "FAIL"
    elif (mb_pc_to_pi.get("stale_count", 0) + mb_pi_to_pc.get("stale_count", 0)) > 5:
        hyg_status = "WARN"
    elif mb_pc_to_pi.get("backlog_warn") or mb_pi_to_pc.get("backlog_warn"):
        hyg_status = "WARN"

    hygiene_payload = {
        "score": 4 - sum(1 for e in (mb_pc_to_pi, mb_pi_to_pc) if e.get("error")),
        "max": 4,
        "status": hyg_status,
        "detail": {
            "pc_to_pi": mb_pc_to_pi,
            "pi_to_pc": mb_pi_to_pc,
            "backlog_pc": mb_pc_to_pi.get("total", -1),
            "backlog_pi": mb_pi_to_pc.get("total", -1),
            "stale": mb_pc_to_pi.get("stale_count", 0) + mb_pi_to_pc.get("stale_count", 0),
            "dups": mb_pc_to_pi.get("dup_count", 0) + mb_pi_to_pc.get("dup_count", 0),
        },
    }

    posted_pc = post_audit("pc_health", pc)
    posted_hyg = post_audit("hygiene", hygiene_payload)

    state = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(time.time() - started, 2),
        "pc_health": pc,
        "hygiene": hygiene_payload,
        "posted": {"pc_health": posted_pc, "hygiene": posted_hyg},
    }
    safe_json_write(STATE_DIR / "mailbox_auditor_last.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description="MOLOCH Mailbox-Auditor")
    parser.add_argument("--once", action="store_true", help="Einmal-Tick + exit")
    parser.add_argument(
        "--interval-s", type=int, default=DEFAULT_INTERVAL_S, help="Loop-Intervall (Default 300)"
    )
    parser.add_argument("--json", action="store_true", help="Letztes State als JSON ausgeben")
    args = parser.parse_args()

    if args.json:
        last = STATE_DIR / "mailbox_auditor_last.json"
        if last.exists():
            print(last.read_text(encoding="utf-8"))
        else:
            print("{}")
        return

    if args.once:
        state = tick()
        print(f"[once] PC={state['pc_health'].get('status','?')}  HYG={state['hygiene'].get('status','?')}")
        print(f"       posted pc_health={state['posted']['pc_health']}  hygiene={state['posted']['hygiene']}")
        return

    logger.info(f"MOLOCH Mailbox-Auditor: Loop alle {args.interval_s}s, Pi={PI_BASE}")
    while True:
        try:
            state = tick()
            logger.info(
                f"tick PC={state['pc_health'].get('status','?')} "
                f"HYG={state['hygiene'].get('status','?')} "
                f"posted={state['posted']}"
            )
        except Exception as e:
            logger.exception(f"tick fail: {e}")
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
