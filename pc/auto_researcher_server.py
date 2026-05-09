"""MOLOCH Auto-Researcher Server (PC-Side, Phase 2 Synthese-Plan).

FastAPI auf :11653 — exposiert staging/research_proposals/ als HTTP.
Komplementaer zu pc/auto_researcher.py (CLI-One-Shot).

Endpoints:
  GET  /health
  GET  /proposals         {proposals: [...], auto_deploy_until: iso|null}
  POST /approve/<pid>     Markiert Proposal als approved
  POST /reject/<pid>      Markiert Proposal als rejected
  POST /auto_deploy       Body {days:N}, setzt Stufe-2-Toggle fuer N Tage

State persistent in %LOCALAPPDATA%/moloch_pc_state/auto_researcher.json.
Pi-Proxy nimmt 'research_' Prefix weg: /research_proposals -> /proposals etc.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

REPO = Path(os.environ.get("MOLOCH_REPO", r"C:\Users\49179\moloch_repo"))
STAGING = REPO / "staging" / "research_proposals"

_LOCAL_APPDATA = os.environ.get("LOCALAPPDATA")
_STATE_DIR = Path(_LOCAL_APPDATA) / "moloch_pc_state" if _LOCAL_APPDATA else Path.home() / "moloch_pc_state"
STATE_FILE = _STATE_DIR / "auto_researcher.json"

app = FastAPI(title="MOLOCH AutoResearcher Server", version="0.1")


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(STATE_FILE))


def _parse_findings(md_text: str, date: str) -> list[dict]:
    findings = []
    sections = md_text.split("\n### ")
    for i, section in enumerate(sections[1:]):
        lines = section.split("\n", 1)
        title = lines[0].strip()
        body = lines[1] if len(lines) > 1 else ""
        summary = ""
        for ln in body.splitlines():
            if ln.startswith("- **Description:**"):
                summary = ln.split(":**", 1)[1].strip()
                break
        if not summary:
            summary = body.split("\n", 1)[0].strip()[:200]
        findings.append({
            "id": f"{date}-{i:02d}",
            "date": date,
            "title": title,
            "summary": summary,
            "source": "auto_researcher",
        })
    return findings


def _list_proposals() -> list[dict]:
    if not STAGING.exists():
        return []
    state = _load_state()
    decisions = state.get("decisions", {})
    proposals: list[dict] = []
    for f in sorted(STAGING.glob("*.md"), reverse=True):
        date = f.stem
        try:
            md = f.read_text(encoding="utf-8")
        except Exception:
            continue
        for finding in _parse_findings(md, date):
            finding["status"] = decisions.get(finding["id"], "open")
            proposals.append(finding)
    return [p for p in proposals if p["status"] == "open"]


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "staging_dir": str(STAGING),
        "state_file": str(STATE_FILE),
        "n_open": len(_list_proposals()),
    }


@app.get("/proposals")
def proposals() -> dict:
    state = _load_state()
    until_ts = state.get("auto_deploy_until_ts")
    until_iso: Any = None
    if isinstance(until_ts, (int, float)) and until_ts > time.time():
        until_iso = datetime.fromtimestamp(until_ts, tz=timezone.utc).isoformat(timespec="seconds")
    return {
        "proposals": _list_proposals(),
        "auto_deploy_until": until_iso,
    }


@app.post("/approve/{pid}")
def approve(pid: str) -> dict:
    state = _load_state()
    decisions = state.setdefault("decisions", {})
    decisions[pid] = {"status": "approved", "ts": time.time()}
    _save_state(state)
    return {"ok": True, "id": pid, "status": "approved"}


@app.post("/reject/{pid}")
def reject(pid: str) -> dict:
    state = _load_state()
    decisions = state.setdefault("decisions", {})
    decisions[pid] = {"status": "rejected", "ts": time.time()}
    _save_state(state)
    return {"ok": True, "id": pid, "status": "rejected"}


@app.post("/auto_deploy")
def auto_deploy(body: dict) -> dict:
    try:
        days = int(body.get("days", 7))
    except Exception:
        raise HTTPException(status_code=400, detail="days must be int")
    if days < 1 or days > 30:
        raise HTTPException(status_code=400, detail="days must be in [1, 30]")
    state = _load_state()
    until_ts = time.time() + days * 86400
    state["auto_deploy_until_ts"] = until_ts
    state["auto_deploy_until_iso"] = datetime.fromtimestamp(until_ts, tz=timezone.utc).isoformat(timespec="seconds")
    _save_state(state)
    return {"ok": True, "days": days, "until": state["auto_deploy_until_iso"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "11653")))
