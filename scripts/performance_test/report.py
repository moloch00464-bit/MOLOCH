"""Report-Generator: JSON (maschinenlesbar) + Markdown (Erlebnis-Zusammenfassung)."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from .acts import ActResult
from .config import LOG_DIR


@dataclass
class TestRun:
    started_at: str
    duration_s: float
    overall: str          # "PASS" | "PARTIAL" | "FAIL"
    summary_de: str
    baseline: Dict[str, Any]
    acts: List[ActResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "started_at": self.started_at,
            "duration_s": self.duration_s,
            "overall": self.overall,
            "summary_de": self.summary_de,
            "baseline": self.baseline,
            "acts": [a.to_dict() for a in self.acts],
        }
        return d


def aggregate_overall(acts: List[ActResult]) -> str:
    fails = sum(1 for a in acts if a.status == "FAIL")
    partials = sum(1 for a in acts if a.status == "PARTIAL")
    if fails > 0:
        return "FAIL"
    if partials > 0:
        return "PARTIAL"
    return "PASS"


def build_summary_de(run: TestRun) -> str:
    passes = sum(1 for a in run.acts if a.status == "PASS")
    total = len(run.acts)
    if run.overall == "PASS":
        return f"Moloch hat alle {total} Akte sauber bestanden. Spuerbar lebendig."
    fails = [a.name for a in run.acts if a.status == "FAIL"]
    partials = [a.name for a in run.acts if a.status == "PARTIAL"]
    msg = f"Moloch hat {passes} von {total} Akten bestanden."
    if fails:
        msg += f" Gescheitert: {', '.join(fails)}."
    if partials:
        msg += f" Teilweise: {', '.join(partials)}."
    return msg


def write_json_report(run: TestRun, path: Path) -> None:
    """Atomic write via tempfile + os.replace (NEVER 6)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(run.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, str(path))
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise


def build_markdown(run: TestRun) -> str:
    """Lesbares Protokoll fuer Markus."""
    icon = {"PASS": "✓", "FAIL": "✗", "PARTIAL": "~", "SKIP": "·"}
    out = [
        f"# MOLOCH 5-Akt-Performance-Test — {run.started_at}",
        "",
        f"**Gesamt: {run.overall}** · Dauer {run.duration_s:.1f}s · "
        f"{sum(1 for a in run.acts if a.status == 'PASS')}/{len(run.acts)} Akte PASS",
        "",
        f"_{run.summary_de}_",
        "",
    ]
    for act in run.acts:
        out.append(f"## {act.name} {icon.get(act.status, '?')}")
        if act.input_text:
            out.append(f"**Markus:** {act.input_text}")
        if act.moloch_response:
            out.append(f"**Moloch:** {act.moloch_response}")
        out.append("")
        for exp in act.expectations:
            out.append(f"- {icon.get(exp.status, '?')} **{exp.key}** — {exp.detail}")
        out.append("")
        if act.erlebnis:
            out.append(f"**Erlebnis:** {act.erlebnis}")
        out.append("")
    return "\n".join(out)


def write_markdown_report(run: TestRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_markdown(run), encoding="utf-8")


def report_paths(started_at_iso: str) -> tuple[Path, Path]:
    """Pfade fuer JSON + Markdown-Report basierend auf Start-Timestamp."""
    safe = started_at_iso.replace(":", "").replace("-", "").replace("T", "_")[:15]
    return (
        LOG_DIR / f"{safe}_performance_test.json",
        LOG_DIR / f"{safe}_performance_test.md",
    )
