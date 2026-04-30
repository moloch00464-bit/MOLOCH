"""Self-Diagnosis-Runner (Welle 14).

Wrapper um scripts/self_diagnosis.py:
- Fuehrt periodisch alle Diagnose-Tests aus (NPU, Kamera, Mic, Whisper, Claude)
- Parst data/last_diagnosis.json (vom Skript geschrieben)

Schreibt audit_state.layers.self_diagnosis Schema:
  {tests_total, tests_passed, tests_failed, last_run_iso,
   score, max, status, detail (mit failed_tests-Liste)}

Status-Logik:
- PASS: alle Tests pass
- WARN: 1-2 fail
- FAIL: 3+ fail ODER subprocess-timeout

CLI: python3 -m core.audit.self_diagnosis_runner
  -> Fuehrt subprocess aus + writes Result
"""
from __future__ import annotations

import json
import os
import sys
import time
import subprocess
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("self_diagnosis_runner")

_DIAGNOSIS_SCRIPT = "/home/molochzuhause/moloch/scripts/self_diagnosis.py"
_DIAGNOSIS_REPORT = "/home/molochzuhause/moloch/data/last_diagnosis.json"
_TIMEOUT = 60


def _run_diagnosis_subprocess(mode: str = "quick") -> Dict[str, Any]:
    """Ruft self_diagnosis.py auf, returnt Run-Metadaten."""
    out: Dict[str, Any] = {"timeout": False, "returncode": None, "stderr_tail": ""}
    try:
        r = subprocess.run(
            [sys.executable, _DIAGNOSIS_SCRIPT, mode],
            capture_output=True, text=True, timeout=_TIMEOUT,
            cwd="/home/molochzuhause/moloch",
        )
        out["returncode"] = r.returncode
        out["stderr_tail"] = (r.stderr or "")[-200:]
    except subprocess.TimeoutExpired:
        out["timeout"] = True
    except Exception as e:
        out["error"] = str(e)[:120]
    return out


def _parse_last_report() -> Optional[Dict[str, Any]]:
    if not os.path.exists(_DIAGNOSIS_REPORT):
        return None
    try:
        with open(_DIAGNOSIS_REPORT, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def collect(run_now: bool = False, mode: str = "quick") -> Dict[str, Any]:
    """Sammelt Self-Diagnosis-Daten.

    Args:
        run_now: Wenn True, fuehrt self_diagnosis.py aus VOR dem Parsen.
                 (Sonst nur letzten Report parsen — billiger.)
        mode: 'quick' (kein Hailo/Whisper) oder 'all'.
    """
    detail: Dict[str, Any] = {}

    if run_now:
        run_result = _run_diagnosis_subprocess(mode)
        detail["run_result"] = run_result
        if run_result.get("timeout"):
            return {
                "score": 0,
                "max": 4,
                "status": "FAIL",
                "tests_total": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "last_run_iso": None,
                "detail": {**detail, "reason": "subprocess_timeout"},
            }

    report = _parse_last_report()
    if not report:
        return {
            "score": 0,
            "max": 4,
            "status": "WARN",
            "tests_total": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "last_run_iso": None,
            "detail": {**detail, "reason": "no_report_yet"},
        }

    total = int(report.get("passed", 0)) + int(report.get("failed", 0))
    passed = int(report.get("passed", 0))
    failed = int(report.get("failed", 0))
    tests = report.get("tests", []) or []
    failed_tests: List[str] = [
        t.get("name", "?") for t in tests if not t.get("ok", False)
    ]

    last_iso = report.get("iso_time")
    last_ts = report.get("timestamp")
    age_s: Optional[float] = None
    if last_ts:
        try:
            age_s = max(0.0, time.time() - float(last_ts))
        except Exception:
            pass
    detail["last_run_age_s"] = age_s
    detail["failed_tests"] = failed_tests
    detail["mode"] = report.get("mode")

    # Status
    score = 0
    max_score = 4
    if total > 0:
        score += 1
    if failed == 0:
        score += 2
    elif failed <= 2:
        score += 1
    if age_s is not None and age_s < 24 * 3600:
        score += 1

    if failed >= 3:
        status = "FAIL"
    elif failed >= 1:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "score": score,
        "max": max_score,
        "status": status,
        "tests_total": total,
        "tests_passed": passed,
        "tests_failed": failed,
        "last_run_iso": last_iso,
        "detail": detail,
    }


def main() -> int:
    """CLI-Entry: laeuft self_diagnosis.py + schreibt Audit-Layer-Snapshot
    nach /dev/shm/audit_self_diagnosis.json (atomic)."""
    import tempfile
    out_path = "/dev/shm/audit_self_diagnosis.json"
    result = collect(run_now=True, mode="quick")
    payload = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "result": result,
    }
    try:
        fd, tmp = tempfile.mkstemp(dir="/dev/shm", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp, out_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("status") in ("PASS", "WARN") else 1
    except Exception as e:
        print(f"ERROR writing snapshot: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
