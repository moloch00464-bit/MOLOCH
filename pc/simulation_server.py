"""FastAPI-Server fuer pc/simulation/ (Phase 2 Plan).

Exposiert Scenario-Liste + Run-Steuerung an Cockpit-Sub-Tab Simulation.
Default-Port :11654 (kann via PORT env-var ueberschrieben werden).

Skeleton-Stage: /scenarios + /scenarios/<name>/run + /runs/<run_id>.
Echte Replay-Logik (Event-Stream + State-Pipeline-Feed) folgt in iterativem Ausbau.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException

from pc.simulation import ScenarioRunner

SCENARIOS_DIR = Path(
    os.environ.get(
        "MOLOCH_SIM_SCENARIOS_DIR",
        str(Path(__file__).parent / "simulation" / "scenarios"),
    )
)

app = FastAPI(title="MOLOCH Simulation Server", version="0.1")
_runs: dict[str, dict] = {}


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
    _runs[run_id] = {
        "run_id": run_id,
        "scenario": name,
        "n_events": n_events,
        "expected_state_path": runner.expected_state_path(),
        "started_ts": time.time(),
        "status": "skeleton-not-yet-executing",
    }
    return _runs[run_id]


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="run_id not found")
    return _runs[run_id]


@app.get("/runs")
def list_runs() -> dict:
    return {"runs": list(_runs.values())}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "11654")))
