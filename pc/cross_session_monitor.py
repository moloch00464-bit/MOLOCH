"""MOLOCH Cross-Session Monitor (PC-Side).

Long-running daemon, Mirror zu Pi-Side `core/bridge/cross_session_monitor.py`.

Loop alle 30s:
  1. git fetch + log new commits
  2. Pi-Endpoints pingen (chat_server :9100 via Tunnel, /state_full)
  3. Eigene Services pingen (:11434 ollama, :11600 adapter, :11700 dashboard, :11800 avatar)
  4. Mailbox top-4 parse aus PC_TO_PI.md + PI_TO_PC.md
  5. Heartbeat-Log persistent (%USERPROFILE%/moloch_logs/cross_session.jsonl)
  6. State-Transitions UP->DOWN / DOWN->UP loggen, Outage-Notes >120s
  7. Auto-Trigger Pipeline: bei Pi-Topic `v_next_ready_to_train` autonom
     sync_samples + lora_trainer + curl POST /reload + Mailbox-Reply
     `v2_live` mit `[auto-ack]`-Tag, commit + push.

Run als Scheduled Task `MolochCrossMonitor` (logon, ExecutionTimeLimit unbegrenzt).
Stop via Taskkill oder schtasks /end.
"""
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set

REPO = Path(os.environ.get("MOLOCH_REPO", r"C:\Users\49179\moloch_repo"))
LOG_DIR = Path.home() / "moloch_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "cross_session.jsonl"
LOCK_FILE = LOG_DIR / "v_next_train.lock"
HANDLED_FILE = LOG_DIR / "handled_topics.json"

PI_TUNNEL = os.environ.get("MOLOCH_PI_TUNNEL_URL", "http://localhost:9000")
PC_HOST = "localhost"
PC_ENDPOINTS = {
    "ollama":    f"http://{PC_HOST}:11434/api/tags",
    "adapter":   f"http://{PC_HOST}:11600/health",
    "dashboard": f"http://{PC_HOST}:11700/api/state",
    "avatar":    f"http://{PC_HOST}:11800/api/state",
}
PI_ENDPOINTS = {
    "pi_chat":  f"{PI_TUNNEL}/health",
    "pi_state": f"{PI_TUNNEL}/state_full",
}

LOOP_INTERVAL_S = 30
HTTP_TIMEOUT_S = 4
OUTAGE_NOTE_THRESHOLD_S = 120
LOCK_STALE_AFTER_S = 2400        # >> 1800s train timeout
TRIGGER_COOLDOWN_S = 3600        # min 1h zwischen Auto-Trains gleichen Topics
LOG_ROTATE_BYTES = 50_000_000    # 50 MB

# Pi-committet -> PC reagiert autonom
PI_TRIGGER_TOPICS: Set[str] = {
    "v_next_ready_to_train",
    "samples_ready_for_v2",
    "samples_ready_for_v3",
}

AUTO_ACK = "[auto-ack]"
GIT_AUTHOR_ENV = {
    "GIT_AUTHOR_NAME": "Cowork PC-Side Monitor",
    "GIT_AUTHOR_EMAIL": "cowork-monitor@moloch.local",
    "GIT_COMMITTER_NAME": "Cowork PC-Side Monitor",
    "GIT_COMMITTER_EMAIL": "cowork-monitor@moloch.local",
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
    logger.info("SIGTERM/SIGINT received - exiting loop")


signal.signal(signal.SIGTERM, _sigterm)
try:
    signal.signal(signal.SIGINT, _sigterm)
except Exception:
    pass


# =============================================================================
# Helpers
# =============================================================================

def _git_env() -> Dict[str, str]:
    return {**os.environ, **GIT_AUTHOR_ENV}


def _git_fetch_and_diff(last_sha: Optional[str]) -> Dict:
    try:
        subprocess.run(
            ["git", "fetch", "-q", "origin", "main"],
            cwd=REPO, timeout=15, check=True, env=_git_env(),
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
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "moloch-cross-monitor"})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            resp.read(1024)
            return {
                "ok": True,
                "status_code": resp.status,
                "latency_ms": int((time.monotonic() - t0) * 1000),
            }
    except Exception as e:
        return {
            "ok": False,
            "error": str(getattr(e, "reason", e))[:80],
            "latency_ms": int((time.monotonic() - t0) * 1000),
        }


def _append_log(entry: Dict) -> None:
    entry["ts"] = time.time()
    entry["iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > LOG_ROTATE_BYTES:
            rotated = LOG_PATH.with_suffix(f".{int(time.time())}.jsonl")
            LOG_PATH.rename(rotated)
            logger.info(f"log rotated -> {rotated.name}")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"log write fail: {e}")


def _parse_mailbox(file: str, n: int = 4) -> List[Dict]:
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
                if cur:
                    entries.append(cur)
                    if len(entries) >= n:
                        return entries
                cur = {"raw": stripped}
                if "from=" in stripped:
                    cur["from"] = stripped.split("from=", 1)[1].split(" ", 1)[0]
                if "topic=" in stripped:
                    cur["topic"] = stripped.split("topic=", 1)[1].split(" ", 1)[0]
                end = stripped.find("]")
                if end > 0:
                    cur["ts"] = stripped[3:end]
            elif stripped.startswith("status:") and cur and "status" not in cur:
                cur["status"] = stripped.split(":", 1)[1].strip()
        if cur and len(entries) < n:
            entries.append(cur)
        return entries
    except Exception as e:
        logger.warning(f"mailbox parse fail {file}: {e}")
        return []


def _load_handled() -> Dict[str, float]:
    """Load {key -> last-fired-ts} dict. Keeps order via timestamps."""
    try:
        if HANDLED_FILE.exists():
            data = json.loads(HANDLED_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): float(v) for k, v in data.items()}
            # backward-compat: old set-based file
            if isinstance(data, list):
                now = time.time()
                return {str(k): now for k in data}
    except Exception:
        pass
    return {}


def _save_handled(handled: Dict[str, float]) -> None:
    """Persist as dict, keep newest 100 entries by timestamp."""
    try:
        keep = dict(sorted(handled.items(), key=lambda kv: kv[1])[-100:])
        HANDLED_FILE.write_text(json.dumps(keep), encoding="utf-8")
    except Exception:
        pass


def _acquire_train_lock() -> bool:
    """Atomic O_EXCL create. Auto-clears stale lock older than LOCK_STALE_AFTER_S."""
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(time.time()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            age = time.time() - LOCK_FILE.stat().st_mtime
        except OSError:
            return False
        if age < LOCK_STALE_AFTER_S:
            return False
        try:
            LOCK_FILE.unlink()
        except OSError:
            return False
        # retry once
        try:
            fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(time.time()).encode())
            os.close(fd)
            return True
        except OSError:
            return False


def _release_train_lock() -> None:
    try:
        LOCK_FILE.unlink()
    except OSError:
        pass


def _kill_process_tree(pid: int) -> None:
    """Windows: taskkill /F /T kills the whole tree (children stay otherwise)."""
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            timeout=10, capture_output=True,
        )
    except Exception:
        pass


def _git_run_safe(args: List[str], env: Dict[str, str], timeout: int = 30) -> bool:
    """Run a git command. On failure, log + try `rebase --abort` cleanup if relevant."""
    res = subprocess.run(args, cwd=REPO, env=env, timeout=timeout,
                         capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"git {' '.join(args)} failed: {res.stderr.strip()[:200]}")
        if "rebase" in args or "Conflict" in (res.stderr or ""):
            subprocess.run(["git", "rebase", "--abort"], cwd=REPO, env=env,
                           timeout=10, capture_output=True)
        return False
    return True


def _insert_after_first_separator(text: str, msg: str) -> str:
    """CRLF-aware: insert msg right after the first standalone '---' line."""
    pattern = re.compile(r"^---[ \t]*\r?\n", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return msg + text
    return text[:m.end()] + msg + text[m.end():]


def _trigger_v_next_train(triggering_topic: str, triggering_ts: str) -> Optional[str]:
    """Auto-Pipeline: sync_samples -> lora_trainer -> /reload -> mailbox v2_live.

    Returns new version string (vN+1) on success, None on failure.
    """
    if not _acquire_train_lock():
        logger.info(f"v_next train lock present (or recently held), skipping {triggering_topic}")
        return None
    try:
        logger.info(f"AUTO-TRIGGER: {triggering_topic} at {triggering_ts} -> sync + train + reload")

        try:
            sync_result = subprocess.run(
                ["cmd.exe", "/c", str(REPO / "pc" / "sync_samples.bat")],
                timeout=60, capture_output=True, text=True, stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as e:
            if e.pid:
                _kill_process_tree(e.pid)
            logger.error("sync_samples timeout")
            return None
        if sync_result.returncode != 0:
            logger.error(f"sync_samples failed (rc={sync_result.returncode}): {sync_result.stderr[:200]}")
            return None

        venv_python = Path.home() / "moloch_pc_env" / "Scripts" / "python.exe"
        samples = Path.home() / "moloch_samples" / "samples.jsonl"
        adapters_out = Path.home() / "moloch_adapters"
        try:
            train_result = subprocess.run(
                [str(venv_python), str(REPO / "pc" / "lora_trainer.py"),
                 "--samples", str(samples), "--out", str(adapters_out)],
                timeout=1800, capture_output=True, text=True, stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as e:
            if e.pid:
                _kill_process_tree(e.pid)
            logger.error("lora_trainer timeout - process tree killed")
            return None
        if train_result.returncode != 0:
            logger.error(f"lora_trainer failed (rc={train_result.returncode}): {train_result.stderr[:200]}")
            return None
        new_version = None
        for line in train_result.stdout.splitlines():
            if "[trainer] done v" in line:
                try:
                    new_version = line.split("done ", 1)[1].split(",")[0].strip()
                except Exception:
                    pass
                break

        # POST /reload
        reload_curl = subprocess.run(
            ["curl", "-X", "POST", "-sS", "--max-time", "10",
             "http://localhost:11600/reload"],
            timeout=15, capture_output=True, text=True, stdin=subprocess.DEVNULL,
        )
        reload_ok = reload_curl.returncode == 0 and "reloaded" in reload_curl.stdout.lower()

        # Mailbox-Reply (CRLF-aware insertion, no rebase corruption)
        ts = time.strftime("%Y-%m-%d %H:%M")
        msg = "\n".join([
            f"## [{ts}] from=PC topic=v2_live {AUTO_ACK} reply-to={triggering_ts} {triggering_topic}",
            "status: done",
            "",
            "Auto-Pipeline durch (Cross-Session-Monitor, kein Markus-Klick):",
            "- sync_samples.bat -> samples.jsonl gepullt",
            f"- lora_trainer.py -> {new_version or 'unbekannt'} trainiert",
            f"- POST :11600/reload -> {'OK' if reload_ok else 'FAIL'}",
            "",
            "Trigger-zu-Adapter-Live: vollautonom.",
            "",
            "---",
            "",
        ]) + "\n"

        mailbox = REPO / "docs" / "PC_TO_PI.md"
        old = mailbox.read_text(encoding="utf-8")
        new_text = _insert_after_first_separator(old, msg)
        mailbox.write_text(new_text, encoding="utf-8")

        env = _git_env()
        if not _git_run_safe(["git", "add", "docs/PC_TO_PI.md"], env, timeout=10):
            return None
        commit_msg = f"auto: v2_live mailbox after auto-train ({new_version or '?'})"
        if not _git_run_safe(["git", "commit", "-m", commit_msg], env, timeout=10):
            return None
        if not _git_run_safe(["git", "pull", "--rebase"], env, timeout=30):
            logger.warning("rebase failed, push aborted - manual reconciliation required")
            return None
        if not _git_run_safe(["git", "push"], env, timeout=30):
            return None
        logger.info(f"AUTO-TRIGGER complete: {new_version or '?'} live, mailbox pushed")
        return new_version or "unknown"
    except Exception as e:
        logger.error(f"auto-trigger fail: {e}")
        return None
    finally:
        _release_train_lock()


# =============================================================================
# Main loop
# =============================================================================

def main() -> int:
    logger.info(f"PC cross-session monitor starting (loop {LOOP_INTERVAL_S}s)")
    logger.info(f"  REPO     = {REPO}")
    logger.info(f"  LOG      = {LOG_PATH}")
    logger.info(f"  Pi via   = {PI_TUNNEL}")
    logger.info(f"  triggers = {sorted(PI_TRIGGER_TOPICS)}")

    last_sha: Optional[str] = None
    state: Dict[str, str] = {}        # endpoint -> "up"/"down"
    first_down: Dict[str, float] = {} # endpoint -> ts when went down
    handled = _load_handled()

    while _running:
        loop_start = time.monotonic()

        git_info = _git_fetch_and_diff(last_sha)
        if git_info.get("new_commits"):
            for c in git_info["new_commits"]:
                logger.info(f"new commit: {c}")
        last_sha = git_info.get("head", last_sha)

        ping_results: Dict[str, Dict] = {}
        for name, url in {**PC_ENDPOINTS, **PI_ENDPOINTS}.items():
            r = _ping(url)
            ping_results[name] = r
            new_state = "up" if r["ok"] else "down"
            old_state = state.get(name)
            if old_state != new_state:
                if old_state is not None:
                    logger.warning(f"STATE {name}: {old_state} -> {new_state}")
                state[name] = new_state
                if new_state == "down":
                    first_down[name] = time.time()
                else:
                    first_down.pop(name, None)
            elif new_state == "down":
                down_for = time.time() - first_down.get(name, time.time())
                if down_for > OUTAGE_NOTE_THRESHOLD_S and int(down_for) % 300 < LOOP_INTERVAL_S:
                    logger.warning(f"OUTAGE: {name} down for {int(down_for)}s")

        pc_to_pi = _parse_mailbox("PC_TO_PI.md", n=4)
        pi_to_pc = _parse_mailbox("PI_TO_PC.md", n=4)

        for entry in pi_to_pc:
            topic = entry.get("topic", "")
            status = entry.get("status", "")
            ts = entry.get("ts", "")
            if not topic or topic not in PI_TRIGGER_TOPICS:
                continue
            if status != "open":
                continue
            key = f"{topic}:{ts}"
            if key in handled:
                continue
            # Topic-cooldown: blockiert wenn das Topic in den letzten TRIGGER_COOLDOWN_S
            # schon gefeuert hat (anti-spam bei Pi-message-storm)
            now = time.time()
            recent = max((t for k, t in handled.items() if k.startswith(topic + ":")),
                         default=0)
            if now - recent < TRIGGER_COOLDOWN_S:
                logger.info(f"cooldown active for {topic} ({int(now-recent)}s ago), skip {ts}")
                continue
            new_v = _trigger_v_next_train(topic, ts)
            if new_v:
                handled[key] = now
                _save_handled(handled)

        _append_log({
            "kind": "heartbeat",
            "head": last_sha[:8] if last_sha else None,
            "new_commits": git_info.get("new_commits", []),
            "endpoints": {k: {"ok": v["ok"], "latency_ms": v.get("latency_ms")}
                          for k, v in ping_results.items()},
            "open_pc_to_pi": [e.get("topic") for e in pc_to_pi if e.get("status") == "open"],
            "open_pi_to_pc": [e.get("topic") for e in pi_to_pc if e.get("status") == "open"],
        })

        elapsed = time.monotonic() - loop_start
        if _running and elapsed < LOOP_INTERVAL_S:
            time.sleep(LOOP_INTERVAL_S - elapsed)

    logger.info("loop exited cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
