"""MOLOCH Health-Check (PC-Side, Self-Test).

Zweck: in 30s pruefen ob die Symbiose-Pipeline rund laeuft.
Markus' Wunsch: System um Fehler zu finden.

Layers (sequentiell, je mit PASS/WARN/FAIL):
  L1 PC-Services    (search_proxy, ollama, adapter, pi-tunnel)
  L2 Pi-Services    (chat_server, /status, /history)
  L3 Routing-State  (llm_mode=kaskade, last_provider plausibel)
  L4 Mailbox        (offene Topics, Drift-Risiko)
  L5 Federation     (fed_kill marker, daemon-heartbeat-Alter)
  L6 PC-ENV         (OLLAMA_NUM_GPU=0, Pre-Warm-Liste)
  L7 Memory-Drift   (Anthropic-Erwaehnung, Doppel-Updates)
  L8 Git-State      (uncommitted, branch, untracked)

Output: kompakte Tabelle + Final-Score X/Y PASS.

Aufruf:
  python pc/moloch_health_check.py
  python pc/moloch_health_check.py --json   (fuer Monitoring-Tool-Integration)
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

PI_HOST = "http://192.168.178.30:9100"
PC_HOST = "http://localhost"
REPO = Path(os.environ.get("MOLOCH_REPO", str(Path.home() / "moloch_repo")))
LOG_DIR = Path.home() / "moloch_logs"
MEMORY_DIR = Path(
    r"C:\Users\49179\.claude\projects\C--Users-49179-Desktop-Kleine-Moloch-PIGHOST-moloch-for-cowork\memory"
)


class Result:
    """Single check result with status + message."""

    def __init__(self, layer: str, name: str, status: str, msg: str = ""):
        self.layer = layer
        self.name = name
        self.status = status  # PASS / WARN / FAIL
        self.msg = msg


def _ok(layer: str, name: str, msg: str = "") -> Result:
    return Result(layer, name, "PASS", msg)


def _warn(layer: str, name: str, msg: str) -> Result:
    return Result(layer, name, "WARN", msg)


def _fail(layer: str, name: str, msg: str) -> Result:
    return Result(layer, name, "FAIL", msg)


def http_ok(url: str, timeout: int = 4) -> tuple[bool, str]:
    """Returns (ok, status_or_error)."""
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200, str(r.status_code)
    except requests.RequestException as e:
        return False, str(e)[:60]


def check_l1_pc_services() -> list[Result]:
    """L1: PC-Services."""
    out: list[Result] = []
    services = [
        ("search_proxy :11650", f"{PC_HOST}:11650/health"),
        ("ollama :11434", f"{PC_HOST}:11434/api/tags"),
        ("adapter_proxy :11600", f"{PC_HOST}:11600/health"),
        ("pi-tunnel :9000", f"{PC_HOST}:9000/health"),
    ]
    for name, url in services:
        ok, info = http_ok(url)
        out.append(_ok("L1", name, info) if ok else _fail("L1", name, info))
    return out


def check_l2_pi_services() -> list[Result]:
    """L2: Pi-Side direkter Chat-Server."""
    out: list[Result] = []
    ok, info = http_ok(f"{PI_HOST}/health")
    out.append(_ok("L2", "pi_chat_server :9100/health", info) if ok else _fail("L2", "pi_chat_server :9100/health", info))
    try:
        r = requests.get(f"{PI_HOST}/status", timeout=5)
        if r.status_code == 200:
            data = r.json()
            llm_mode = data.get("llm_mode", "")
            out.append(_ok("L2", "/status reachable", f"llm_mode={llm_mode}"))
            return out
        out.append(_fail("L2", "/status", f"HTTP {r.status_code}"))
    except Exception as e:
        out.append(_fail("L2", "/status", str(e)[:60]))
    return out


def check_l3_routing() -> list[Result]:
    """L3: Routing-State."""
    out: list[Result] = []
    try:
        data = requests.get(f"{PI_HOST}/status", timeout=5).json()
    except Exception as e:
        return [_fail("L3", "routing-state", str(e)[:60])]

    llm_mode = data.get("llm_mode", "")
    if llm_mode == "kaskade":
        out.append(_ok("L3", "llm_mode=kaskade", "Anatomie aktiv"))
    elif llm_mode in ("local_first", "cloud_only"):
        out.append(_warn("L3", f"llm_mode={llm_mode}", "Anatomie sollte 'kaskade' sein"))
    else:
        out.append(_fail("L3", "llm_mode", f"unbekannt: {llm_mode}"))

    last_provider = data.get("last_provider", "")
    if last_provider.startswith("kaskade_"):
        out.append(_ok("L3", "last_provider", last_provider))
    elif last_provider in ("lokal_qwen2.5", "stille", ""):
        out.append(_warn("L3", "last_provider", f"{last_provider} (war evtl. Hardware/Intent)"))
    else:
        out.append(_warn("L3", "last_provider", last_provider))

    tentacle = data.get("tentacle", {})
    if tentacle.get("fail_count", 0) >= 3:
        out.append(_fail("L3", "tentacle circuit-breaker", f"fail_count={tentacle.get('fail_count')}"))
    else:
        out.append(_ok("L3", "tentacle reachable", f"fail_count={tentacle.get('fail_count', 0)}"))

    return out


def check_l4_mailbox() -> list[Result]:
    """L4: Mailbox open topics + age."""
    out: list[Result] = []
    try:
        pc_to_pi = requests.get(f"{PI_HOST}/mailbox/PC_TO_PI", timeout=5).text
        pi_to_pc = requests.get(f"{PI_HOST}/mailbox/PI_TO_PC", timeout=5).text
    except Exception as e:
        return [_fail("L4", "mailbox", str(e)[:60])]

    topic_re = re.compile(r"^## \[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] from=(\w+) topic=(\S+)", re.M)
    status_re = re.compile(r"^status:\s*(\w+)", re.M)

    def parse_top(text: str, n: int = 8):
        entries = []
        sections = text.split("\n## [")
        for i, sec in enumerate(sections[1:n + 1]):
            head = "## [" + sec.split("\n", 1)[0]
            body = sec.split("\n", 1)[1] if "\n" in sec else ""
            tm = topic_re.match(head)
            sm = status_re.search(body[:200])
            if tm:
                entries.append({
                    "ts": tm.group(1),
                    "from": tm.group(2),
                    "topic": tm.group(3),
                    "status": sm.group(1) if sm else "?",
                })
        return entries

    pc_open = [e for e in parse_top(pc_to_pi) if e["status"] == "open"]
    pi_open = [e for e in parse_top(pi_to_pc) if e["status"] == "open"]

    if len(pc_open) <= 2:
        out.append(_ok("L4", "PC->Pi open topics", f"{len(pc_open)} offen"))
    else:
        out.append(_warn("L4", "PC->Pi open topics", f"{len(pc_open)} offen — Backlog?"))

    if len(pi_open) <= 2:
        out.append(_ok("L4", "Pi->PC open topics", f"{len(pi_open)} offen"))
    else:
        out.append(_warn("L4", "Pi->PC open topics", f"{len(pi_open)} offen"))

    # Alter aeltestes open Topic auf PC-Seite
    if pc_open:
        oldest = min(pc_open, key=lambda e: e["ts"])
        try:
            ts = datetime.strptime(oldest["ts"], "%Y-%m-%d %H:%M")
            age_min = (datetime.now() - ts).total_seconds() / 60
            if age_min > 60:
                out.append(_warn("L4", "aeltestes PC-open topic", f"{oldest['topic']} ({int(age_min)} min alt)"))
            else:
                out.append(_ok("L4", "aeltestes PC-open topic", f"{oldest['topic']} ({int(age_min)} min)"))
        except ValueError:
            pass

    return out


def check_l5_federation() -> list[Result]:
    """L5: Federation-Daemon-Status."""
    out: list[Result] = []
    fed_kill = LOG_DIR / "fed_kill"
    if fed_kill.exists():
        out.append(_warn("L5", "fed_kill marker", "Federation deaktiviert"))
    else:
        out.append(_ok("L5", "Federation aktiv", "kein fed_kill"))

    heartbeat_log = LOG_DIR / "cross_session.jsonl"
    if heartbeat_log.exists():
        try:
            tail = heartbeat_log.read_text(encoding="utf-8").strip().split("\n")[-1]
            ts = datetime.fromisoformat(json.loads(tail)["iso"])
            if ts.tzinfo is None:
                age_s = (datetime.now() - ts).total_seconds()
            else:
                age_s = (datetime.now(tz=ts.tzinfo) - ts).total_seconds()
            if age_s < 90:
                out.append(_ok("L5", "Daemon-Heartbeat", f"{int(age_s)}s alt"))
            elif age_s < 300:
                out.append(_warn("L5", "Daemon-Heartbeat", f"{int(age_s)}s alt — verzoegert"))
            else:
                out.append(_fail("L5", "Daemon-Heartbeat", f"{int(age_s)}s alt — Daemon haengt?"))
        except Exception as e:
            out.append(_warn("L5", "Heartbeat-Parse", str(e)[:60]))
    else:
        out.append(_fail("L5", "cross_session.jsonl", "fehlt"))

    return out


def check_l6_env() -> list[Result]:
    """L6: PC-ENV + Pre-Warm."""
    out: list[Result] = []
    # OLLAMA_NUM_GPU registry-check (User-scope persistent)
    try:
        ng = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "[Environment]::GetEnvironmentVariable('OLLAMA_NUM_GPU','User')"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if ng == "0":
            out.append(_ok("L6", "OLLAMA_NUM_GPU=0 user-env", "CPU-only persistent"))
        else:
            out.append(_warn("L6", "OLLAMA_NUM_GPU", f"='{ng}' (sollte '0' sein)"))
    except Exception as e:
        out.append(_warn("L6", "OLLAMA_NUM_GPU check", str(e)[:60]))

    # Pre-warm-Status
    try:
        ps_out = subprocess.run(
            [str(Path.home() / "AppData/Local/Programs/Ollama/ollama.exe"), "ps"],
            capture_output=True, text=True, timeout=8,
        ).stdout
        loaded = sum(1 for m in ("deepseek-coder", "dolphin-mistral", "dolphin-llama3") if m in ps_out)
        if loaded >= 2:
            out.append(_ok("L6", "Pre-warmed Modelle", f"{loaded}/3 in Cache"))
        elif loaded == 1:
            out.append(_warn("L6", "Pre-warmed Modelle", "1/3 — Cold-Load-Risk"))
        else:
            out.append(_warn("L6", "Pre-warmed Modelle", "0/3 — alle cold"))
    except Exception as e:
        out.append(_warn("L6", "ollama ps", str(e)[:60]))

    return out


def check_l7_memory_drift() -> list[Result]:
    """L7: Memory-File-Drift."""
    out: list[Result] = []
    if not MEMORY_DIR.exists():
        return [_warn("L7", "Memory-Dir", "fehlt")]

    # Anthropic-API-Key-Erwaehnung (sollte raus)
    bad_patterns = re.compile(r"ANTHROPIC_API_KEY|sk-ant-api", re.I)
    bad_files = []
    for f in MEMORY_DIR.glob("*.md"):
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if bad_patterns.search(text):
            bad_files.append(f.name)

    if not bad_files:
        out.append(_ok("L7", "kein ANTHROPIC_API_KEY in Memory", "sauber"))
    else:
        out.append(_warn("L7", "ANTHROPIC_API_KEY-Erwaehnung", ", ".join(bad_files)))

    # MEMORY.md Index existiert
    idx = MEMORY_DIR / "MEMORY.md"
    if idx.exists() and idx.stat().st_size > 200:
        out.append(_ok("L7", "MEMORY.md Index", f"{idx.stat().st_size} bytes"))
    else:
        out.append(_warn("L7", "MEMORY.md Index", "fehlt oder leer"))

    return out


def check_l8_git_state() -> list[Result]:
    """L8: Git-State (PC-Repo)."""
    out: list[Result] = []
    if not (REPO / ".git").exists():
        return [_fail("L8", "moloch_repo", f"kein git-Repo in {REPO}")]

    try:
        branch = subprocess.run(
            ["git", "-C", str(REPO), "branch", "--show-current"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if branch == "deepseek_architecture_overhaul":
            out.append(_ok("L8", "Branch", branch))
        else:
            out.append(_warn("L8", "Branch", f"{branch} (erwartet deepseek_architecture_overhaul)"))

        st = subprocess.run(
            ["git", "-C", str(REPO), "status", "-s"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        modified = [l for l in st if l.startswith(" M") or l.startswith("M ")]
        untracked = [l for l in st if l.startswith("??")]
        if not modified:
            out.append(_ok("L8", "modified files", "0"))
        else:
            out.append(_warn("L8", "modified files", f"{len(modified)}"))
        if not untracked:
            out.append(_ok("L8", "untracked files", "0"))
        else:
            out.append(_warn("L8", "untracked files", f"{len(untracked)}"))
    except Exception as e:
        out.append(_fail("L8", "git", str(e)[:80]))

    return out


def main():
    json_mode = "--json" in sys.argv
    t0 = time.time()
    all_checks: list[Result] = []
    all_checks.extend(check_l1_pc_services())
    all_checks.extend(check_l2_pi_services())
    all_checks.extend(check_l3_routing())
    all_checks.extend(check_l4_mailbox())
    all_checks.extend(check_l5_federation())
    all_checks.extend(check_l6_env())
    all_checks.extend(check_l7_memory_drift())
    all_checks.extend(check_l8_git_state())
    duration_s = time.time() - t0

    if json_mode:
        out = {
            "duration_s": round(duration_s, 2),
            "checks": [
                {"layer": r.layer, "name": r.name, "status": r.status, "msg": r.msg}
                for r in all_checks
            ],
            "summary": {
                "PASS": sum(1 for r in all_checks if r.status == "PASS"),
                "WARN": sum(1 for r in all_checks if r.status == "WARN"),
                "FAIL": sum(1 for r in all_checks if r.status == "FAIL"),
            },
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print(f"=== MOLOCH HEALTH CHECK ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
    layer_now = ""
    for r in all_checks:
        if r.layer != layer_now:
            print(f"\n[{r.layer}]")
            layer_now = r.layer
        marker = {"PASS": "  PASS ", "WARN": "  WARN ", "FAIL": "  FAIL "}[r.status]
        msg = f" -- {r.msg}" if r.msg else ""
        print(f"{marker} {r.name}{msg}")

    pass_count = sum(1 for r in all_checks if r.status == "PASS")
    warn_count = sum(1 for r in all_checks if r.status == "WARN")
    fail_count = sum(1 for r in all_checks if r.status == "FAIL")
    print(f"\n=== ZUSAMMENFASSUNG ({duration_s:.1f}s) ===")
    print(f"PASS: {pass_count}  WARN: {warn_count}  FAIL: {fail_count}")
    if fail_count > 0:
        sys.exit(1)
    if warn_count > 3:
        sys.exit(2)


if __name__ == "__main__":
    main()
