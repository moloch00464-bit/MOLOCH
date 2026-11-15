#!/usr/bin/env python3
"""
Koerper (Reachy-Mini-Entwurf): Ausdruck, Grenzen, Prioritaeten, Not-Aus.

Alles gegen MockMotionBackend und eine Test-Uhr - kein Roboter, kein SDK.

Run:
    python3 -m pytest tests/test_body.py -v
"""

import math
from dataclasses import replace

import pytest

from core.body import (BodyController, BodyPose, BodyState, MockMotionBackend,
                       Priority, clamp_pose, compute_pose, load_mappings)
from core.body.pose import AXES, NEUTRAL


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


@pytest.fixture
def maps():
    return load_mappings()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def backend():
    return MockMotionBackend()


@pytest.fixture
def ctl(backend, maps, clock):
    return BodyController(backend, maps, clock=clock)


def _settle(ctl, clock, backend):
    """Erste (langsame) Bewegung hinter sich bringen: Koerper steht in Ruhepose."""
    assert ctl.tick() == "neutral"
    clock.t += 1.0
    backend.log.clear()


POSE_A = BodyPose(head_yaw=20.0, duration_s=0.5)
POSE_B = BodyPose(head_yaw=-20.0, duration_s=0.5)


# === Konfiguration ===

class TestMappings:

    def test_file_is_complete(self, maps):
        for axis in AXES:
            lo, hi = maps["limits"][axis]
            assert lo < 0 < hi
        for name in ("idle", "observing", "engaged", "overloaded", "withdrawing", "offline_anchor"):
            assert name in maps["states"]
        for name in ("calm", "focused", "alert", "agitated", "euphoric", "dark"):
            assert name in maps["moods"]
        for name in ("guardian", "shadow", "berserker"):
            assert name in maps["zones"]

    def test_state_names_match_state_engine(self, maps):
        from core.personality.state_engine import VALID_STATES
        assert set(VALID_STATES) == set(maps["states"])


# === Grenzen ===

class TestClamp:

    def test_values_are_clamped(self, maps):
        p = clamp_pose(BodyPose(head_pitch=999, head_yaw=-999, antenna_l=500), maps["limits"])
        assert p.head_pitch == maps["limits"]["head_pitch"][1]
        assert p.head_yaw == maps["limits"]["head_yaw"][0]
        assert p.antenna_l == maps["limits"]["antenna_l"][1]

    def test_nan_and_inf_become_zero(self, maps):
        p = clamp_pose(BodyPose(head_pitch=float("nan"), head_yaw=float("inf"),
                                duration_s=float("nan")), maps["limits"])
        assert p.head_pitch == 0.0 and p.head_yaw == 0.0 and p.duration_s == 0.0

    def test_missing_limit_means_no_motion_on_that_axis(self, maps):
        limits = dict(maps["limits"])
        del limits["body_yaw"]
        assert clamp_pose(BodyPose(body_yaw=45.0), limits).body_yaw == 0.0


# === Ausdruck ===

class TestExpression:

    def test_is_deterministic(self, maps):
        s = BodyState(being_state="engaged", mood="agitated", tension=0.8)
        assert compute_pose(s, maps, 3.3) == compute_pose(s, maps, 3.3)

    def test_dominance_raises_head(self, maps):
        low = compute_pose(BodyState(dominance=-1.0), maps, 0.0)
        high = compute_pose(BodyState(dominance=1.0), maps, 0.0)
        assert high.head_pitch > low.head_pitch

    def test_tension_adds_restlessness(self, maps):
        calm = [abs(compute_pose(BodyState(tension=0.0), maps, t / 10).head_roll) for t in range(50)]
        tense = [abs(compute_pose(BodyState(tension=1.0), maps, t / 10).head_roll) for t in range(50)]
        assert max(calm) == 0.0 and max(tense) > 1.0

    def test_mood_moves_antennas(self, maps):
        alert = compute_pose(BodyState(mood="alert"), maps, 0.0)
        dark = compute_pose(BodyState(mood="dark"), maps, 0.0)
        assert alert.antenna_l > dark.antenna_l

    def test_gaze_is_split_between_head_and_body(self, maps):
        p = compute_pose(BodyState(being_state="observing", gaze_yaw=30.0), maps, 0.0)
        assert p.head_yaw + p.body_yaw == pytest.approx(30.0)
        assert p.body_yaw > 0 and p.head_yaw > 0

    def test_gaze_sign_is_configurable(self, maps):
        maps["gaze"]["yaw_sign"] = -1.0
        p = compute_pose(BodyState(being_state="observing", gaze_yaw=30.0), maps, 0.0)
        assert p.head_yaw + p.body_yaw == pytest.approx(-30.0)

    def test_withdrawing_ignores_gaze(self, maps):
        a = compute_pose(BodyState(being_state="withdrawing", gaze_yaw=40.0), maps, 0.0)
        b = compute_pose(BodyState(being_state="withdrawing"), maps, 0.0)
        assert a == b

    def test_unknown_names_fall_back(self, maps):
        weird = compute_pose(BodyState(being_state="???", zone="???", mood="???"), maps, 1.0)
        assert weird == compute_pose(BodyState(), maps, 1.0)

    def test_garbage_numbers_do_not_propagate(self, maps):
        p = compute_pose(BodyState(tension=float("nan"), dominance=float("inf"),
                                   gaze_yaw=float("nan")), maps, 1.0)
        assert all(math.isfinite(getattr(p, a)) for a in AXES)

    @pytest.mark.parametrize("zone", ["guardian", "shadow", "berserker"])
    def test_every_combination_stays_inside_limits_after_clamp(self, maps, zone):
        for st in maps["states"]:
            for mood in maps["moods"]:
                for t in (0.0, 0.7, 3.1):
                    s = BodyState(st, zone, mood, tension=1.0, dominance=1.0,
                                  gaze_yaw=170.0, gaze_pitch=80.0, speaking=True)
                    p = clamp_pose(compute_pose(s, maps, t), maps["limits"])
                    for a in AXES:
                        lo, hi = maps["limits"][a]
                        assert lo <= getattr(p, a) <= hi


# === Controller: Prioritaeten ===

class TestPriorities:

    def test_manual_beats_dialogue_beats_idle(self, ctl, clock, backend):
        _settle(ctl, clock, backend)
        ctl.submit(Priority.IDLE, BodyPose(head_yaw=5.0))
        ctl.submit(Priority.DIALOGUE, BodyPose(head_yaw=10.0))
        ctl.submit(Priority.MANUAL, BodyPose(head_yaw=15.0))
        assert ctl.tick() == "sent"
        assert backend.poses[-1].head_yaw == 15.0

    def test_lower_priority_takes_over_when_higher_expires(self, ctl, clock, backend):
        _settle(ctl, clock, backend)
        ctl.submit(Priority.IDLE, BodyPose(head_yaw=5.0), ttl_s=5.0)
        ctl.submit(Priority.MANUAL, BodyPose(head_yaw=15.0), ttl_s=1.0)
        ctl.tick()
        clock.t += 1.5
        assert ctl.tick() == "sent"
        assert backend.poses[-1].head_yaw == 5.0

    def test_quiet_mode_blocks_everything_but_manual(self, ctl, clock, backend):
        _settle(ctl, clock, backend)
        ctl.set_quiet(True)
        ctl.submit(Priority.DIALOGUE, POSE_A)
        assert ctl.tick() == "idle"
        assert backend.poses == []
        ctl.submit(Priority.MANUAL, POSE_A)
        assert ctl.tick() == "sent"


# === Controller: veraltete Befehle, Ruhepose ===

class TestStaleAndNeutral:

    def test_expired_command_is_not_executed(self, ctl, clock, backend):
        _settle(ctl, clock, backend)
        ctl.submit(Priority.DIALOGUE, POSE_A, ttl_s=0.5)
        clock.t += 0.6
        assert ctl.tick() == "idle"
        assert backend.poses == []

    def test_command_already_stale_on_arrival_is_rejected(self, ctl, clock):
        assert ctl.submit(Priority.DIALOGUE, POSE_A, ttl_s=1.0, created_at=clock.t - 2.0) is False

    def test_command_from_the_future_is_rejected(self, ctl, clock):
        assert ctl.submit(Priority.DIALOGUE, POSE_A, ttl_s=1.0, created_at=clock.t + 60) is False

    @pytest.mark.parametrize("ttl", [0.0, -1.0, 11.0, 3600.0])
    def test_invalid_ttl_is_rejected(self, ctl, ttl):
        assert ctl.submit(Priority.DIALOGUE, POSE_A, ttl_s=ttl) is False

    def test_returns_to_neutral_once_after_input_stops(self, ctl, clock, backend, maps):
        _settle(ctl, clock, backend)
        ctl.submit(Priority.DIALOGUE, POSE_A, ttl_s=0.5)
        assert ctl.tick() == "sent"
        clock.t += 1.0
        assert ctl.tick() == "idle"            # Befehl weg, Wartezeit laeuft
        clock.t += maps["motion"]["idle_timeout_s"]
        assert ctl.tick() == "neutral"
        assert backend.poses[-1].max_delta(NEUTRAL) == 0.0
        clock.t += 5.0
        assert ctl.tick() == "idle"            # nur EINMAL
        assert len(backend.poses) == 2


# === Controller: Grenzen, Tempo, Drosselung ===

class TestShaping:

    def test_pose_is_clamped_before_sending(self, ctl, clock, backend, maps):
        _settle(ctl, clock, backend)
        ctl.submit(Priority.MANUAL, BodyPose(head_yaw=9999.0))
        ctl.tick()
        assert backend.poses[-1].head_yaw == maps["limits"]["head_yaw"][1]

    def test_speed_limit_stretches_duration(self, ctl, clock, backend, maps):
        _settle(ctl, clock, backend)
        ctl.submit(Priority.MANUAL, BodyPose(head_yaw=60.0, duration_s=0.01))
        ctl.tick()
        sent = backend.poses[-1]
        assert 60.0 / sent.duration_s <= maps["motion"]["max_speed_deg_s"] + 1e-6

    def test_first_move_after_connect_is_slow(self, ctl, backend, maps):
        ctl.submit(Priority.MANUAL, BodyPose(head_yaw=30.0, duration_s=0.1))
        assert ctl.tick() == "sent"
        assert backend.poses[-1].duration_s == maps["motion"]["max_duration_s"]

    def test_rate_limit(self, ctl, clock, backend, maps):
        _settle(ctl, clock, backend)
        ctl.submit(Priority.MANUAL, POSE_A)
        assert ctl.tick() == "sent"
        ctl.submit(Priority.MANUAL, POSE_B)
        clock.t += maps["motion"]["min_interval_s"] / 2
        assert ctl.tick() == "rate_limited"
        clock.t += maps["motion"]["min_interval_s"]
        assert ctl.tick() == "sent"
        assert len(backend.poses) == 2

    def test_tiny_change_is_not_sent(self, ctl, clock, backend):
        _settle(ctl, clock, backend)
        ctl.submit(Priority.MANUAL, POSE_A)
        ctl.tick()
        clock.t += 1.0
        ctl.submit(Priority.MANUAL, replace(POSE_A, head_yaw=POSE_A.head_yaw + 0.1))
        assert ctl.tick() == "unchanged"
        assert len(backend.poses) == 1


# === Controller: Not-Aus ===

class TestEstop:

    def test_estop_sends_stop_once_and_blocks_motion(self, ctl, clock, backend):
        _settle(ctl, clock, backend)
        ctl.submit(Priority.MANUAL, POSE_A)
        ctl.estop()
        for _ in range(5):
            assert ctl.tick() == "estop"
            clock.t += 1.0
        assert backend.stops == 1 and backend.poses == []

    def test_submit_is_rejected_during_estop(self, ctl):
        ctl.estop()
        assert ctl.submit(Priority.MANUAL, POSE_A) is False

    def test_estop_beats_manual(self, ctl, clock, backend):
        _settle(ctl, clock, backend)
        ctl.submit(Priority.MANUAL, POSE_A, ttl_s=10.0)
        ctl.tick()
        ctl.estop()
        clock.t += 1.0
        ctl.tick()
        assert backend.log[-1][0] == "stop"

    def test_old_commands_do_not_run_after_release(self, ctl, clock, backend):
        _settle(ctl, clock, backend)
        ctl.submit(Priority.MANUAL, POSE_A, ttl_s=10.0)
        ctl.estop()
        ctl.tick()
        ctl.release_estop()
        clock.t += 0.5
        ctl.tick()
        assert all(p.head_yaw == 0.0 for p in backend.poses)

    def test_first_move_after_release_is_slow(self, ctl, clock, backend, maps):
        _settle(ctl, clock, backend)
        ctl.estop()
        ctl.tick()
        ctl.release_estop()
        clock.t += 1.0
        ctl.submit(Priority.MANUAL, POSE_A)
        assert ctl.tick() == "sent"
        assert backend.poses[-1].duration_s == maps["motion"]["max_duration_s"]

    def test_failed_stop_is_retried(self, ctl, clock, backend):
        _settle(ctl, clock, backend)
        backend.fail_stop = True
        ctl.estop()
        ctl.tick()
        assert backend.stops == 0
        backend.fail_stop = False
        ctl.tick()
        assert backend.stops == 1


# === Controller: Verbindung ===

class TestConnection:

    def test_no_body_is_not_an_error(self, maps, clock):
        backend = MockMotionBackend(connected=False)
        ctl = BodyController(backend, maps, clock=clock)
        ctl.submit(Priority.MANUAL, POSE_A)
        assert ctl.tick() == "offline"
        assert backend.log == []
        assert ctl.status()["connected"] is False

    def test_reconnect_uses_backoff(self, maps, clock):
        backend = MockMotionBackend(connected=False)
        ctl = BodyController(backend, maps, clock=clock)
        for _ in range(20):
            ctl.tick()
            clock.t += 0.2
        assert backend.connect_calls == 1
        clock.t += maps["motion"]["reconnect_backoff_s"]
        ctl.tick()
        assert backend.connect_calls == 2

    def test_commands_queued_while_offline_do_not_run_later(self, maps, clock):
        backend = MockMotionBackend(connected=False)
        ctl = BodyController(backend, maps, clock=clock)
        ctl.submit(Priority.MANUAL, POSE_A, ttl_s=10.0)
        ctl.tick()
        backend.reachable = True
        clock.t += maps["motion"]["reconnect_backoff_s"] + 0.1
        ctl.tick()
        assert all(p.head_yaw == 0.0 for p in backend.poses)

    def test_connection_loss_during_goto(self, ctl, clock, backend, maps):
        _settle(ctl, clock, backend)
        backend.fail_goto = True
        ctl.submit(Priority.MANUAL, POSE_A)
        assert ctl.tick() == "offline"
        backend.fail_goto = False
        clock.t += maps["motion"]["reconnect_backoff_s"] + 0.1
        ctl.submit(Priority.MANUAL, POSE_A)
        assert ctl.tick() == "sent"
        assert backend.poses[-1].duration_s == maps["motion"]["max_duration_s"]


class TestEndToEnd:

    def test_expression_drives_mock_body(self, ctl, clock, backend, maps):
        """5-Hz-Schleife: Zustand -> Pose -> Controller -> Mock. Alles in Grenzen."""
        state = BodyState("engaged", "berserker", "agitated", tension=0.9,
                          dominance=0.8, gaze_yaw=45.0)
        for _ in range(50):
            ctl.submit(Priority.IDLE, compute_pose(state, maps, clock.t), ttl_s=1.0)
            ctl.tick()
            clock.t += 0.2
        assert len(backend.poses) > 10
        for p in backend.poses:
            for a in AXES:
                lo, hi = maps["limits"][a]
                assert lo <= getattr(p, a) <= hi
        assert ctl.status()["last_result"] in ("sent", "unchanged")
