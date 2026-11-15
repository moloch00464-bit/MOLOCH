#!/usr/bin/env python3
"""
Agent-Tools ptz_pan / ptz_tilt: absolute Klemmung gegen die Tracker-Grenzen (M5).

Ohne Hardware: Status- und Settings-Datei liegen in tmp_path, das
IPC-Kommando wird abgefangen statt nach /tmp geschrieben.

Run:
    python3 -m pytest tests/test_agent_ptz_clamp.py -v
"""

import json
import os
import time

import pytest

import core.agent.tools.hardware as hw


@pytest.fixture
def env(tmp_path, monkeypatch):
    status = tmp_path / "status.json"
    settings = tmp_path / "settings.json"
    sent = []
    monkeypatch.setattr(hw, "STATUS_PATH", str(status))
    monkeypatch.setattr(hw, "SETTINGS_PATH", str(settings))
    monkeypatch.setattr(hw, "_atomic_ipc_cmd",
                        lambda action, params=None: sent.append((action, params)) or True)

    class Env:
        def position(self, pan=None, tilt=None, age_s=0.0):
            ptz = {}
            if pan is not None:
                ptz["current_pan"] = pan
            if tilt is not None:
                ptz["current_tilt"] = tilt
            status.write_text(json.dumps({"ptz": ptz}))
            t = time.time() - age_s
            os.utime(status, (t, t))

        def limits(self, **kw):
            settings.write_text(json.dumps({"ptz": kw}))

    e = Env()
    e.sent = sent
    return e


class TestPanClamp:

    def test_move_inside_limits_is_unchanged(self, env):
        env.limits(pan_limit_min=-90, pan_limit_max=90)
        env.position(pan=10)
        r = hw.ptz_pan(30)
        assert r["ok"] is True and r["angle"] == 30 and r["clamped"] is False
        assert env.sent == [("ptz_move", {"direction": "left", "step": 30})]

    def test_positive_angle_is_left_negative_is_right(self, env):
        """NEVER 2: positiver Pan = physisch links."""
        env.position(pan=0)
        assert hw.ptz_pan(20)["direction"] == "left"
        assert hw.ptz_pan(-20)["direction"] == "right"

    def test_move_is_shortened_at_upper_limit(self, env):
        env.limits(pan_limit_min=-90, pan_limit_max=90)
        env.position(pan=70)
        r = hw.ptz_pan(50)
        assert r["angle"] == 20 and r["clamped"] is True and r["target"] == 90
        assert r["requested"] == 50
        assert env.sent == [("ptz_move", {"direction": "left", "step": 20})]

    def test_move_is_shortened_at_lower_limit(self, env):
        env.limits(pan_limit_min=-90, pan_limit_max=90)
        env.position(pan=-80)
        r = hw.ptz_pan(-45)
        assert r["angle"] == -10 and r["direction"] == "right"
        assert env.sent == [("ptz_move", {"direction": "right", "step": 10})]

    def test_at_limit_nothing_is_sent(self, env):
        env.limits(pan_limit_min=-90, pan_limit_max=90)
        env.position(pan=90)
        r = hw.ptz_pan(15)
        assert r["ok"] is False and r["angle"] == 0 and "Grenze" in r["note"]
        assert env.sent == []

    def test_outside_limits_only_inward_allowed(self, env):
        env.limits(pan_limit_min=-90, pan_limit_max=90)
        env.position(pan=120)
        assert hw.ptz_pan(10)["ok"] is False          # weiter raus: nein
        assert env.sent == []
        r = hw.ptz_pan(-10)                           # nach innen: ja, wie verlangt
        assert r["ok"] is True and r["angle"] == -10

    def test_never_moves_against_requested_direction(self, env):
        env.limits(pan_limit_min=-90, pan_limit_max=90)
        env.position(pan=120)
        hw.ptz_pan(10)
        assert all(p["direction"] != "right" for _, p in env.sent)

    def test_small_move_inside_limits_still_goes_through(self, env):
        env.position(pan=0)
        r = hw.ptz_pan(0.3)
        assert r["ok"] is True and r["angle"] == 0.3

    def test_zero_angle(self, env):
        env.position(pan=0)
        assert hw.ptz_pan(0)["note"] == "keine Bewegung"
        assert env.sent == []

    def test_request_is_capped_to_180(self, env):
        env.position(pan=-160)
        r = hw.ptz_pan(999)
        assert r["requested"] == 180
        assert r["target"] <= 170.0


class TestLimitsSource:

    def test_defaults_are_mechanical_limits(self, env):
        env.position(pan=160)
        r = hw.ptz_pan(50)
        assert r["target"] == 170.0 and r["angle"] == 10

    def test_configured_limits_cannot_exceed_mechanical(self, env):
        env.limits(pan_limit_min=-999, pan_limit_max=999)
        assert hw._axis_limits("pan") == (-168.4, 170.0)

    def test_broken_limits_fall_back_to_mechanical(self, env):
        env.limits(pan_limit_min=50, pan_limit_max=-50)
        assert hw._axis_limits("pan") == (-168.4, 170.0)

    def test_real_settings_json_is_readable(self, monkeypatch):
        monkeypatch.undo()
        lo, hi = hw._axis_limits("pan")
        assert -168.4 <= lo < hi <= 170.0


class TestUnknownPosition:

    def test_unknown_position_passes_with_mechanical_limits(self, env):
        """Keine engeren Grenzen: Verhalten wie bisher, Kamera klemmt selbst."""
        r = hw.ptz_pan(30)
        assert r["ok"] is True and r["position_known"] is False
        assert env.sent == [("ptz_move", {"direction": "left", "step": 30})]

    def test_unknown_position_blocks_with_tighter_limits(self, env):
        env.limits(pan_limit_min=-90, pan_limit_max=90)
        r = hw.ptz_pan(30)
        assert r["ok"] is False and r["position_known"] is False
        assert env.sent == []

    def test_stale_status_counts_as_unknown(self, env):
        env.limits(pan_limit_min=-90, pan_limit_max=90)
        env.position(pan=0, age_s=60)
        assert hw.ptz_pan(30)["ok"] is False
        assert env.sent == []

    def test_status_without_position_counts_as_unknown(self, env):
        env.limits(pan_limit_min=-90, pan_limit_max=90)
        env.position()
        assert hw.ptz_pan(30)["ok"] is False


class TestTiltClamp:

    def test_positive_is_up(self, env):
        env.position(tilt=0)
        r = hw.ptz_tilt(10)
        assert r["direction"] == "up"
        assert env.sent == [("ptz_move", {"direction": "up", "step": 10})]

    def test_tilt_is_shortened(self, env):
        env.limits(tilt_limit_min=-15, tilt_limit_max=30)
        env.position(tilt=25)
        r = hw.ptz_tilt(40)
        assert r["angle"] == 5 and r["target"] == 30

    def test_tilt_down_limit(self, env):
        env.limits(tilt_limit_min=-15, tilt_limit_max=30)
        env.position(tilt=-15)
        assert hw.ptz_tilt(-5)["ok"] is False
        assert env.sent == []

    def test_pan_limits_do_not_affect_tilt(self, env):
        env.limits(pan_limit_min=-10, pan_limit_max=10)
        env.position(pan=0, tilt=0)
        assert hw.ptz_tilt(40)["angle"] == 40
