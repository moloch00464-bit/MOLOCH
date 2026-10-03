"""Sicherheits-Schicht vor jedem Tool-Dispatch des lokalen PC-Hirns.

Jeder Tool-Aufruf geht durch check(): Whitelist (tool_policy.json), Wertebereiche
klemmen, Rate-Limits, Nachtruhe, Kill-Datei. Was nicht in der Policy steht, ist gesperrt —
neue Pi-Tools werden also nicht automatisch freigeschaltet.

Ergebnis ist eine Decision: allow / clamp (mit korrigierten args) / deny / confirm.

Pan-Konvention: ptz_pan(angle) ist RELATIV, positiv = physisch LINKS (NEVER-Regel 2,
core/agent/tools/hardware.py). Hier wird nur der Betrag begrenzt, nie das Vorzeichen gedreht.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Optional

POLICY_PATH = Path(__file__).with_name("tool_policy.json")


@dataclass
class Decision:
    action: str                      # allow | clamp | deny | confirm
    args: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.action in ("allow", "clamp")


@dataclass
class Context:
    """Lage beim Aufruf. Unbekanntes bleibt None — dann gilt die vorsichtige Grenze."""
    user_turn: bool = True           # False = proaktiv (vom Hirn selbst ausgeloest)
    current_pan: Optional[float] = None
    current_tilt: Optional[float] = None
    tracker_state: Optional[str] = None
    now: Optional[float] = None      # fuer Tests


class Safety:
    def __init__(self, policy_path: Path = POLICY_PATH) -> None:
        self.policy = json.loads(policy_path.read_text(encoding="utf-8"))
        self.tools: Dict[str, Dict[str, Any]] = self.policy.get("tools", {})
        self._calls: Dict[str, Deque[float]] = defaultdict(deque)
        self.kill_file = Path(os.path.expandvars(self.policy.get("kill_file", "")))

    # --- Kill-Switch ---------------------------------------------------------
    def killed(self) -> bool:
        try:
            return bool(str(self.kill_file)) and self.kill_file.exists()
        except OSError:
            return True              # im Zweifel: gesperrt

    # --- Rate-Limit ----------------------------------------------------------
    def _count(self, key: str, window_s: float, now: float) -> int:
        q = self._calls[key]
        while q and q[0] < now - window_s:
            q.popleft()
        return len(q)

    def _record(self, key: str, now: float) -> None:
        self._calls[key].append(now)

    # --- Hauptpruefung -------------------------------------------------------
    def check(self, tool: str, args: Optional[Dict[str, Any]], ctx: Optional[Context] = None) -> Decision:
        ctx = ctx or Context()
        now = ctx.now if ctx.now is not None else time.time()
        args = dict(args or {})

        if self.killed():
            return Decision("deny", reason="Kill-Switch aktiv (brain_kill)")
        rule = self.tools.get(tool)
        if rule is None:
            return Decision("deny", reason=f"'{tool}' steht nicht in der Whitelist")
        tier = rule.get("tier")
        if tier == "gesperrt":
            return Decision("deny", reason=f"'{tool}' ist gesperrt")
        if tier == "bestaetigung":
            return Decision("confirm", args, f"'{tool}' braucht Markus' Bestaetigung")
        if rule.get("user_turn_only") and not ctx.user_turn:
            return Decision("deny", reason=f"'{tool}' nur auf Markus' Wunsch, nicht proaktiv")

        clamped = False

        # Zahlenbereiche klemmen
        for name, (lo, hi) in (rule.get("clamp") or {}).items():
            if name in args:
                try:
                    v = float(args[name])
                except (TypeError, ValueError):
                    return Decision("deny", reason=f"{tool}.{name} ist keine Zahl")
                nv = min(max(v, lo), hi)
                if nv != v:
                    clamped = True
                args[name] = int(nv) if isinstance(lo, int) and isinstance(hi, int) else nv

        # Textlaengen kappen
        for name, limit in (rule.get("max_chars") or {}).items():
            if isinstance(args.get(name), str) and len(args[name]) > limit:
                args[name] = args[name][:limit]
                clamped = True

        # PTZ: relativen Schritt begrenzen, Ziel in den Tracker-Grenzen halten
        if tool in ("ptz_pan", "ptz_tilt"):
            d = self._check_ptz(tool, args, ctx)
            if not d.ok:
                return d
            clamped = clamped or d.action == "clamp"
            args = d.args

        # Proaktive Sprache: Nachtruhe + Obergrenze pro 10 Minuten
        if tool == "tts_say" and not ctx.user_turn:
            qh = self.policy.get("quiet_hours") or {}
            hour = datetime.fromtimestamp(now).hour
            if rule.get("quiet_hours_proactive") and _in_quiet(hour, qh.get("from", 23), qh.get("to", 7)):
                return Decision("deny", reason="Nachtruhe — proaktiv stumm")
            limit = rule.get("proactive_per_10min")
            if limit is not None and self._count("tts_say:proaktiv", 600, now) >= limit:
                return Decision("deny", reason=f"proaktives Limit ({limit} pro 10 min) erreicht")

        # Rate-Limits
        if tier == "frei":
            key, limit = "frei:gesamt", self.policy.get("read_rate_per_min", 30)
        else:
            key, limit = tool, rule.get("rate_per_min")
        if limit is not None and self._count(key, 60, now) >= limit:
            return Decision("deny", reason=f"Rate-Limit fuer '{tool}' ({limit}/min) erreicht")

        self._record(key, now)
        if tool == "tts_say" and not ctx.user_turn:
            self._record("tts_say:proaktiv", now)
        return Decision("clamp" if clamped else "allow", args)

    def _check_ptz(self, tool: str, args: Dict[str, Any], ctx: Context) -> Decision:
        p = self.policy.get("ptz", {})
        try:
            angle = float(args.get("angle"))
        except (TypeError, ValueError):
            return Decision("deny", reason=f"{tool}: angle fehlt oder ist keine Zahl")

        state = (ctx.tracker_state or "").lower()
        if state in p.get("blocked_tracker_states", []) and not ctx.user_turn:
            return Decision("deny", reason="Tracker folgt gerade einer Person — Reflex hat Vorrang")

        pan = tool == "ptz_pan"
        step_max = p["max_step_pan"] if pan else p["max_step_tilt"]
        lo, hi = (p["pan_min"], p["pan_max"]) if pan else (p["tilt_min"], p["tilt_max"])
        cur = ctx.current_pan if pan else ctx.current_tilt
        if cur is None:
            step_max = min(step_max, p["unknown_pos_step"])   # Position unbekannt: kleine Schritte

        new = min(max(angle, -step_max), step_max)
        if cur is not None:
            # Ziel = aktuelle Position + Schritt, in den Grenzen halten.
            # Steht die Kamera schon ausserhalb, sind nur Schritte zurueck in den Bereich erlaubt.
            target = min(max(cur + new, min(lo, cur)), max(hi, cur))
            new = target - cur
        if abs(new) < 0.5:
            return Decision("deny", reason=f"{tool}: Grenze erreicht, keine Bewegung moeglich")
        args = dict(args, angle=round(new, 1))
        return Decision("clamp" if abs(new - angle) > 1e-6 else "allow", args)


def _in_quiet(hour: int, start: int, end: int) -> bool:
    return (hour >= start or hour < end) if start > end else (start <= hour < end)
