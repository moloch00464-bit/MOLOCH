"""Tests fuer core/autonomy/presence_triggers.py - reine Logik mit falscher Uhr."""

import json

import pytest

from core.autonomy.presence_triggers import PresenceTriggers, _is_quiet


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    hour = {"h": 12}
    path = tmp_path / "proactive.json"
    pt = PresenceTriggers(config_path=str(path), clock=clock, hour=lambda: hour["h"])
    return pt, clock, hour, path


def run(pt, clock, seconds, face_id=None, face=False, person=False, step=2.0):
    """`seconds` lang dieselbe Beobachtung fuettern; alle Ereignisse einsammeln."""
    events = []
    for _ in range(int(seconds / step)):
        clock.t += step
        ev = pt.update(face_id, face, person)
        if ev:
            events.append(ev["event"])
    return events


def test_no_event_right_after_start(env):
    pt, clock, _, _ = env
    # Markus sitzt beim Start schon da: keine Begruessung (min_absence nicht erfuellt)
    assert run(pt, clock, 60, "markus", True, True) == []
    assert pt.state()["owner_present"] is True


def test_arrival_after_absence(env):
    pt, clock, _, _ = env
    assert run(pt, clock, 200) == []                       # niemand da
    assert run(pt, clock, 20, "markus", True, True) == ["markus_arrived"]
    assert run(pt, clock, 120, "markus", True, True) == []  # kein zweites Mal


def test_single_frame_does_not_count_as_arrival(env):
    pt, clock, _, _ = env
    run(pt, clock, 200)
    assert run(pt, clock, 2, "markus", True, True) == []    # ein Treffer reicht nicht
    assert run(pt, clock, 10) == []
    assert pt.state()["owner_present"] is False


def test_flicker_does_not_trigger_left_or_unknown(env):
    pt, clock, _, _ = env
    run(pt, clock, 90, "markus", True, True)
    events = []
    for _ in range(20):                                     # 10 s unbekannt, 10 s Markus, ...
        events += run(pt, clock, 10, "unknown", True, True)
        events += run(pt, clock, 10, "markus", True, True)
    assert events == []


def test_left_after_long_presence(env):
    pt, clock, _, _ = env
    run(pt, clock, 120, "markus", True, True)
    assert run(pt, clock, 44) == []
    assert run(pt, clock, 6) == ["markus_left"]


def test_left_not_fired_after_short_visit(env):
    pt, clock, _, _ = env
    run(pt, clock, 200)
    run(pt, clock, 20, "markus", True, True)                # nur 20 s da
    assert run(pt, clock, 120) == []


def test_unknown_person_needs_owner_absent_and_sustain(env):
    pt, clock, _, _ = env
    assert run(pt, clock, 6, "unknown", True, True) == []
    assert run(pt, clock, 6, "unknown", True, True) == ["unknown_person"]
    assert run(pt, clock, 100, "unknown", True, True) == []          # Sperrzeit 180 s
    assert run(pt, clock, 100, "unknown", True, True) == ["unknown_person"]


def test_unknown_face_without_name_counts(env):
    pt, clock, _, _ = env
    assert run(pt, clock, 14, None, True, True) == ["unknown_person"]


def test_person_without_face_is_not_unknown(env):
    pt, clock, _, _ = env
    assert run(pt, clock, 60, None, False, True) == []


def test_quiet_hours(env):
    pt, clock, hour, _ = env
    hour["h"] = 2
    run(pt, clock, 200)
    assert run(pt, clock, 20, "markus", True, True) == []
    assert _is_quiet(23, 23, 7) and _is_quiet(6, 23, 7) and not _is_quiet(7, 23, 7)
    assert not _is_quiet(12, 0, 0)


def test_rate_limit_and_disable(env):
    pt, clock, _, path = env
    path.write_text(json.dumps({"max_per_10min": 1,
                                "events": {"unknown_person": {"cooldown_s": 10}}}))
    assert run(pt, clock, 14, None, True, True) == ["unknown_person"]
    assert run(pt, clock, 120, None, True, True) == []      # Obergrenze 1 pro 10 min
    path.write_text(json.dumps({"enabled": False}))
    clock.t += 700
    assert run(pt, clock, 60, None, True, True) == []


def test_reason_text(env):
    pt, clock, _, _ = env
    run(pt, clock, 600)
    clock.t += 2
    pt.update("markus", True, True)
    clock.t += 5
    ev = pt.update("markus", True, True)
    assert ev and ev["event"] == "markus_arrived"
    assert "10 Minuten" in ev["reason"] and "Kamerabild" in ev["situation"]
