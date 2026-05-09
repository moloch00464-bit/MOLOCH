"""FastAPI-Server fuer pc/simulation/ (Phase 2 Plan).

Exposiert Scenario-Liste + Run-Steuerung an Cockpit-Sub-Tab Simulation.
Default-Port :11654 (kann via PORT env-var ueberschrieben werden).

Skeleton-Stage: /scenarios + /scenarios/<name>/run + /runs/<run_id>.
Echte Replay-Logik (Event-Stream + State-Pipeline-Feed) folgt in iterativem Ausbau.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException

from pc.simulation import ScenarioRunner

SCENARIOS_DIR = Path(
    os.environ.get(
        "MOLOCH_SIM_SCENARIOS_DIR",
        str(Path(__file__).parent / "simulation" / "scenarios"),
    )
)
MAX_RUNS_KEPT = 100  # FIFO-Cap gegen Memory-Leak bei langer Laufzeit / Spam

app = FastAPI(title="MOLOCH Simulation Server", version="0.2")
_runs: "OrderedDict[str, dict]" = OrderedDict()
_runs_lock = threading.Lock()


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "scenarios_dir": str(SCENARIOS_DIR),
        "runs_active": len(_runs),
    }


@app.get("/scenarios")
def list_scenarios() -> dict:
    if not SCENARIOS_DIR.exists():
        return {"scenarios": []}
    names = sorted(p.stem for p in SCENARIOS_DIR.glob("*.json"))
    return {"scenarios": names}


@app.post("/scenarios/{name}/run")
def run_scenario(name: str) -> dict:
    runner = ScenarioRunner(SCENARIOS_DIR)
    try:
        n_events = runner.load_scenario(name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    run_id = uuid.uuid4().hex[:12]
    entry = {
        "run_id": run_id,
        "scenario": name,
        "n_events": n_events,
        "expected_state_path": runner.expected_state_path(),
        "started_ts": time.time(),
        "status": "skeleton-not-yet-executing",
    }
    with _runs_lock:
        _runs[run_id] = entry
        while len(_runs) > MAX_RUNS_KEPT:
            _runs.popitem(last=False)
    return entry


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    with _runs_lock:
        entry = _runs.get(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return entry


@app.get("/runs")
def list_runs() -> dict:
    with _runs_lock:
        return {"runs": list(_runs.values())}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "11654")))
