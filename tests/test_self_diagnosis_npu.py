#!/usr/bin/env python3
"""
Selbstdiagnose NPU: echter Inferenz-Nachweis ueber die Worker-Zaehler, ohne Hardware.

Run:
    python3 -m pytest tests/test_self_diagnosis_npu.py -v
"""

import importlib.util
import json
import os
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "self_diagnosis", os.path.join(ROOT, "scripts", "self_diagnosis.py"))
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)


@pytest.fixture
def status(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    monkeypatch.setattr(diag, "STATUS_JSON", str(path))
    state = {"n": 100, "step": 40, "fps": 20.0}

    def write(age_s=0.0):
        path.write_text(json.dumps({
            "worker_health": {"FaceWorker": {"total_inferences": state["n"]},
                              "PoseWorker": {"total_inferences": state["n"] // 2}},
            "fps": {"pipeline": state["fps"]},
        }))
        t = time.time() - age_s
        os.utime(path, (t, t))

    def fake_sleep(_s):          # waehrend der Wartezeit "rechnet" der Service weiter
        state["n"] += state["step"]
        if state.get("vanish"):
            path.unlink()
        else:
            write()

    monkeypatch.setattr(diag.time, "sleep", fake_sleep)
    write()
    state["write"] = write
    return state


def test_running_service_with_rising_counters_passes(status):
    ok, detail = diag.test_hailo_npu()
    assert ok is True and "NPU rechnet" in detail and "2 Worker" in detail


def test_frozen_counters_and_zero_fps_fail(status):
    status["step"] = 0
    status["fps"] = 0.0
    status["write"]()
    ok, detail = diag.test_hailo_npu()
    assert ok is False and "rechnet nicht" in detail


def test_idle_workers_with_running_pipeline_pass_with_hint(status):
    status["step"] = 0
    ok, detail = diag.test_hailo_npu()
    assert ok is True and "ohne Arbeit" in detail


def test_status_vanishing_during_measurement_fails(status):
    status["vanish"] = True
    ok, detail = diag.test_hailo_npu()
    assert ok is False and "verschwunden" in detail


def test_stale_status_counts_as_service_down(status):
    status["write"](age_s=60)
    assert diag._npu_inference_counts() is None


def test_never_opens_second_vdevice_while_service_runs(status, monkeypatch):
    """Solange der Service laeuft, darf hailo_platform gar nicht importiert werden."""
    import builtins
    real_import = builtins.__import__

    def guard(name, *a, **k):
        assert not name.startswith("hailo_platform"), "VDevice neben laufendem Service!"
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", guard)
    assert diag.test_hailo_npu()[0] is True
