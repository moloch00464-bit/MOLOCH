"""Token-Budget-Tracking fuer Cloud-LLM-Calls (Welle 21 Phase 5).

State in %LOCALAPPDATA%/moloch_pc_state/token_budget.json (Windows) oder
$HOME/moloch_pc_state/token_budget.json (POSIX). Atomic-write (NEVER 6).

Limits:
  Per-Turn:  4000 tokens default, 10000 hard
  Per-Loop:  15000 tokens default
  Per-Hour:  100000 tokens
  Per-Day:   1500000 tokens (~$1.50 DeepSeek)

Pricing-Estimate (2026-05-02, $/1M tokens):
  deepseek-chat       in $0.14  out $0.28
  deepseek-reasoner   in $0.55  out $2.19
  claude-haiku-4.5    in $1.00  out $5.00
  claude-sonnet-4.5   in $3.00  out $15.00

Aufruf:
  from pc.agent.token_budget import record_call, get_state, is_over_daily_cap
  record_call("deepseek-chat", input_tokens=500, output_tokens=200)
  if is_over_daily_cap(): ...
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# State-Pfad (Windows %LOCALAPPDATA%, sonst $HOME)
_LOCAL_APPDATA = os.environ.get("LOCALAPPDATA")
if _LOCAL_APPDATA:
    _STATE_DIR = Path(_LOCAL_APPDATA) / "moloch_pc_state"
else:
    _STATE_DIR = Path.home() / "moloch_pc_state"
_STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = Path(os.environ.get("MOLOCH_TOKEN_STATE_PATH", str(_STATE_DIR / "token_budget.json")))

# Limits (env-overridable)
PER_TURN_DEFAULT = int(os.environ.get("MOLOCH_TURN_TOKEN_BUDGET", "4000"))
PER_TURN_HARD = int(os.environ.get("MOLOCH_TURN_TOKEN_HARD_CAP", "10000"))
PER_LOOP_DEFAULT = int(os.environ.get("MOLOCH_LOOP_TOKEN_BUDGET", "15000"))
HOURLY_CAP = int(os.environ.get("MOLOCH_HOURLY_TOKEN_CAP", "100000"))
DAILY_HARD_CAP = int(os.environ.get("MOLOCH_DAILY_TOKEN_HARD_CAP", "1500000"))

# Pricing (USD per 1M tokens)
PRICING: Dict[str, Dict[str, float]] = {
    "deepseek-chat":     {"input": 0.14,  "output": 0.28},
    "deepseek-reasoner": {"input": 0.55,  "output": 2.19},
    "claude-haiku-4.5":  {"input": 1.00,  "output": 5.00},
    "claude-sonnet-4.5": {"input": 3.00,  "output": 15.00},
}

_LOCK = threading.Lock()


def _empty_state() -> Dict[str, Any]:
    return {
        "version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "totals": {},
        "daily_buckets": {},
        "alerts_today": [],
    }


def _atomic_write(path: Path, data: Dict[str, Any]) -> bool:
    """NEVER 6: atomic via tempfile + os.replace."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, str(path))
            return True
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return False
    except Exception:
        return False


def _read_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return _empty_state()
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _empty_state()


def _today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def estimate_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Geschaetzte Kosten in USD."""
    p = PRICING.get(model)
    if not p:
        # generic fallback
        p = {"input": 0.50, "output": 1.50}
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def record_call(model: str, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
    """Trackt einen LLM-Call. Returns aktuellen State.

    Atomic-write, Lock-protected.
    """
    with _LOCK:
        state = _read_state()
        # totals (kumuliert ueber alle Tage)
        totals = state.setdefault("totals", {})
        m_total = totals.setdefault(model, {"input": 0, "output": 0, "calls": 0})
        m_total["input"] += int(input_tokens)
        m_total["output"] += int(output_tokens)
        m_total["calls"] += 1
        # daily buckets
        day = _today_key()
        daily = state.setdefault("daily_buckets", {})
        d = daily.setdefault(day, {"input": 0, "output": 0, "calls": 0, "usd": 0.0, "models": {}})
        d["input"] += int(input_tokens)
        d["output"] += int(output_tokens)
        d["calls"] += 1
        d["usd"] += estimate_usd(model, input_tokens, output_tokens)
        m_d = d["models"].setdefault(model, {"input": 0, "output": 0, "calls": 0})
        m_d["input"] += int(input_tokens)
        m_d["output"] += int(output_tokens)
        m_d["calls"] += 1
        # alerts
        total_today = d["input"] + d["output"]
        if total_today > DAILY_HARD_CAP * 0.5 and "50pct" not in state.get("alerts_today", []):
            state.setdefault("alerts_today", []).append("50pct")
        if total_today > DAILY_HARD_CAP * 0.9 and "90pct" not in state.get("alerts_today", []):
            state.setdefault("alerts_today", []).append("90pct")
        # purge old daily_buckets (nur 7 Tage)
        cutoff = (datetime.now(timezone.utc).timestamp() - 7 * 86400)
        for k in list(daily.keys()):
            try:
                if datetime.strptime(k, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() < cutoff:
                    del daily[k]
            except Exception:
                pass
        state["last_call"] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model,
            "input": int(input_tokens),
            "output": int(output_tokens),
        }
        _atomic_write(STATE_PATH, state)
        return state


def get_state() -> Dict[str, Any]:
    """Current state."""
    return _read_state()


def get_daily_total_tokens(day: Optional[str] = None) -> int:
    """Tokens heute (input + output)."""
    state = _read_state()
    d = state.get("daily_buckets", {}).get(day or _today_key(), {})
    return int(d.get("input", 0) + d.get("output", 0))


def get_daily_usd(day: Optional[str] = None) -> float:
    state = _read_state()
    return float(state.get("daily_buckets", {}).get(day or _today_key(), {}).get("usd", 0.0))


def is_over_daily_cap() -> bool:
    return get_daily_total_tokens() >= DAILY_HARD_CAP


def is_over_hourly_cap() -> bool:
    """Estimate: letzte Stunde aus daily_bucket-Anteil (heuristik, nicht exakt).
    Fuer exakte Stunden-Werte muesste man minute-Buckets fuehren — fuer jetzt
    nutzen wir die approximation 1/24 des Tages-Werts als Proxy.
    """
    return get_daily_total_tokens() >= HOURLY_CAP * 24


def per_turn_budget_remaining(used_in_turn: int) -> int:
    """Verbleibendes Per-Turn-Budget."""
    return max(0, PER_TURN_DEFAULT - used_in_turn)


def per_loop_budget_remaining(used_in_loop: int) -> int:
    """Verbleibendes Per-Loop-Budget (5 iter ~ 6000 tokens each)."""
    return max(0, PER_LOOP_DEFAULT - used_in_loop)


def report() -> str:
    """Human-readable Tagesbericht."""
    state = _read_state()
    day = _today_key()
    d = state.get("daily_buckets", {}).get(day, {})
    if not d:
        return f"=== TOKEN-BUDGET {day} ===\n(keine Calls heute)\n"
    total = d.get("input", 0) + d.get("output", 0)
    usd = d.get("usd", 0.0)
    cap_pct = (total / DAILY_HARD_CAP) * 100
    bar = "#" * int(cap_pct / 5) + "." * (20 - int(cap_pct / 5))
    lines = [f"=== TOKEN-BUDGET {day} ==="]
    for model, m in d.get("models", {}).items():
        lines.append(f"  {model:25} {m['input']:>7} in / {m['output']:>7} out  ({m['calls']} calls)")
    lines.append(f"  {'-' * 60}")
    lines.append(f"  {'Total':25} {d['input']:>7} in / {d['output']:>7} out  = ${usd:.4f}")
    lines.append(f"  Daily-Cap: [{bar}] {cap_pct:.1f}%")
    if state.get("alerts_today"):
        lines.append(f"  Alerts: {', '.join(state['alerts_today'])}")
    return "\n".join(lines)
