"""FastAPI-Server fuer pc/simulation/ (Phase 3 - echte Replay-Logik).

Exposiert Scenario-Liste + Run-Steuerung an Cockpit-Sub-Tab Simulation.
Default-Port :11654 (kann via PORT env-var ueberschrieben werden).

Phase 3 (commit nach 9050f13):
- Background-Thread per Run dispatcht ScenarioEvents zeitlich gestaffelt
- Status-Lifecycle: pending -> running -> completed | failed
- Per-Event-Update in _runs[run_id] (events_dispatched + last_event)
- Optional speed_factor query-param zur Beschleunigung
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from pc.simulation import ScenarioRunner

SCENARIOS_DIR = Path(
    os.environ.get(
        "MOLOCH_SIM_SCENARIOS_DIR",
        str(Path(__file__).parent / "simulation" / "scenarios"),
    )
)
MAX_RUNS_KEPT = 100  # FIFO-Cap gegen Memory-Leak bei langer Laufzeit / Spam
MAX_REPLAY_DURATION_S = 600  # 10min Hard-Cap pro Replay (Watchdog gegen runaway threads)

app = FastAPI(title="MOLOCH Simulation Server", version="0.3")
_runs: "OrderedDict[str, dict]" = OrderedDict()
_runs_lock = threading.Lock()


def _replay_worker(run_id: str, scenario_name: str, speed_factor: float = 1.0) -> None:
    """Background-Thread: dispatcht ScenarioEvents zeitlich gestaffelt + updatet _runs.

    Lifecycle: pending (set by run_scenario) -> running -> completed | failed.
    Hartes 10min-Cap gegen runaway threads bei broken scenario data.
    """
    try:
        runner = ScenarioRunner(SCENARIOS_DIR)
        runner.load_scenario(scenario_name)
        events = list(runner)
    except Exception as e:
        with _runs_lock:
            entry = _runs.get(run_id)
            if entry is not None:
                entry["status"] = "failed"
                entry["error"] = str(e)[:200]
        return

    with _runs_lock:
        entry = _runs.get(run_id)
        if entry is None:
            return  # evicted by FIFO cap
        entry["status"] = "running"
        entry["started_at_ts"] = time.time()

    if not events:
        with _runs_lock:
            entry = _runs.get(run_id)
            if entry is not None:
                entry["status"] = "completed"
                entry["completed_ts"] = time.time()
                entry["duration_s"] = 0.0
        return

    replay_start = time.time()
    base_ev_ts = events[0].ts
    sf = max(speed_factor, 0.01)

    for i, ev in enumerate(events):
        # Watchdog: harten Cap pruefen bevor wir weiter sleepen
        if time.time() - replay_start > MAX_REPLAY_DURATION_S:
            with _runs_lock:
                entry = _runs.get(run_id)
                if entry is not None:
                    entry["status"] = "failed"
                    entry["error"] = f"max_duration_exceeded ({MAX_REPLAY_DURATION_S}s cap)"
            return

        offset = (ev.ts - base_ev_ts) / sf if i > 0 else 0.0
        target = replay_start + offset
        sleep_s = max(0.0, target - time.time())
        if sleep_s > 0:
            time.sleep(sleep_s)

        with _runs_lock:
            entry = _runs.get(run_id)
            if entry is None:
                return  # evicted while running
            entry["events_dispatched"] = i + 1
            entry["last_event"] = {
                "kind": ev.kind,
                "rel_ts": round(ev.ts - base_ev_ts, 3),
                "payload": ev.payload,
            }

    with _runs_lock:
        entry = _runs.get(run_id)
        if entry is not None:
            entry["status"] = "completed"
            entry["completed_ts"] = time.time()
            entry["duration_s"] = round(time.time() - replay_start, 3)


@app.get("/health")
def health() -> dict:
    with _runs_lock:
        active = sum(1 for r in _runs.values() if r.get("status") in ("pending", "running"))
        total = len(_runs)
    return {
        "ok": True,
        "scenarios_dir": str(SCENARIOS_DIR),
        "runs_total": total,
        "runs_active": active,
    }


@app.get("/scenarios")
def list_scenarios() -> dict:
    if not SCENARIOS_DIR.exists():
        return {"scenarios": []}
    names = sorted(p.stem for p in SCENARIOS_DIR.glob("*.json"))
    return {"scenarios": names}


@app.post("/scenarios/{name}/run")
def run_scenario(name: str, speed_factor: float = Query(1.0, ge=0.01, le=100.0)) -> dict:
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
        "status": "pending",
        "events_dispatched": 0,
        "speed_factor": speed_factor,
    }
    with _runs_lock:
        _runs[run_id] = entry
        while len(_runs) > MAX_RUNS_KEPT:
            _runs.popitem(last=False)
    threading.Thread(
        target=_replay_worker,
        args=(run_id, name, speed_factor),
        daemon=True,
        name=f"sim-replay-{run_id}",
    ).start()
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
