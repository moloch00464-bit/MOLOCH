"""Konstanten + Pfade fuer Performance-Test."""
from __future__ import annotations
import os
from pathlib import Path

MOLOCH_DIR = Path(os.path.expanduser("~/moloch"))
LOG_DIR = MOLOCH_DIR / "logs" / "performance_test"
STATUS_JSON = Path("/dev/shm/moloch_status.json")
LAST_TURN = Path("/dev/shm/last_turn.json")
TEST_OVERRIDE_FACE = Path("/dev/shm/moloch_test_face_attr_override.json")

JOURNAL_DIR = Path("/mnt/moloch-data/memory/journal")
CONVERSATIONS_DIR = Path("/mnt/moloch-data/memory/conversations")

# Pi-5 hat keinen Tachometer, nutze cur_state (0-4 typischerweise)
FAN_STATE_PATH = Path("/sys/class/thermal/cooling_device0/cur_state")

CHAT_ENDPOINT = "http://localhost:9100/chat"
CHAT_HEALTH = "http://localhost:9100/health"
TTS_ENDPOINT = "http://localhost:9100/tts"

# PC-Side Cloud-Judge fuer Hybrid-Validation (PC-Topic 08:25, judge_proxy live)
JUDGE_URL = "http://192.168.178.20:11651/judge_act"
JUDGE_TIMEOUT_S = 70  # DeepSeek Cloud kann lang brauchen
JUDGE_HEALTH_URL = "http://192.168.178.20:11651/health"

# Schwellen (DeepSeek-Spec adaptiert auf Pi-5-Realitaet)
TENSION_DELTA_SHIFT = 0.05    # Akt 1 — leichter Anstieg
TENSION_DELTA_SPIKE = 0.15    # Akt 2 — deutlicher Spike
TENSION_GUARDIAN_MAX = 0.3    # Akt 5 — Cooldown-Ziel
FAN_STATE_DELTA_RESPONSE = 1  # Akt 1 — cur_state +1 Stufe
FAN_STATE_DELTA_SPIKE = 1     # Akt 2 — gleiche Stufe (cur_state hat nur 4 Stufen)

ACT1_WAIT_SECONDS = 120
CHAT_RESPONSE_WAIT_SECONDS = 8
ACT3_TENSION_HOLD_SECONDS = 10
ACT5_COOLDOWN_SECONDS = 15

OVERRIDE_VALID_DURATION_S = 30
