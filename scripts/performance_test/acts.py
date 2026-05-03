"""5 Akt-Funktionen — DeepSeek 5-Akt-Performance-Drehbuch."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

import requests

from .baseline import SystemSnapshot, take_snapshot
from .config import (
    CHAT_ENDPOINT, CHAT_RESPONSE_WAIT_SECONDS, ACT1_WAIT_SECONDS,
    ACT3_TENSION_HOLD_SECONDS, ACT5_COOLDOWN_SECONDS,
    TENSION_DELTA_SHIFT, TENSION_DELTA_SPIKE, TENSION_GUARDIAN_MAX,
    FAN_STATE_DELTA_RESPONSE, FAN_STATE_DELTA_SPIKE,
)
from .validators import (
    is_situational, is_dry_response, has_no_apology,
    references_face_and_question, is_dry_finale,
    find_journal_event_after, has_recent_tts_call,
)
from .test_overrides import face_attr_override


@dataclass
class ExpectationResult:
    key: str
    status: str          # "PASS" | "FAIL" | "SKIP"
    detail: str
    measured: Any = None


@dataclass
class ActResult:
    name: str
    status: str          # "PASS" | "FAIL" | "PARTIAL" | "SKIP"
    duration_s: float
    input_text: Optional[str]
    moloch_response: Optional[str]
    pre_snapshot: Dict[str, Any]
    post_snapshot: Dict[str, Any]
    expectations: List[ExpectationResult] = field(default_factory=list)
    erlebnis: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["expectations"] = [asdict(e) for e in self.expectations]
        return d


def _post_chat(text: str, timeout: int = 60) -> Dict[str, Any]:
    """POST /chat, returnt response-dict oder error-dict."""
    try:
        r = requests.post(
            CHAT_ENDPOINT,
            json={"text": text},
            timeout=timeout,
        )
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        return r.json()
    except requests.RequestException as e:
        return {"error": f"req-exc: {type(e).__name__}: {e}"}


def _wait_for_moloch_response(start_ts: float, timeout_s: int = 8) -> Optional[SystemSnapshot]:
    """Pollt last_turn.json bis Moloch geantwortet hat."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        snap = take_snapshot()
        if snap.last_turn_mtime > start_ts and snap.last_turn_role == "moloch":
            return snap
        time.sleep(0.3)
    return None


def _aggregate_status(expectations: List[ExpectationResult]) -> str:
    fails = sum(1 for e in expectations if e.status == "FAIL")
    passes = sum(1 for e in expectations if e.status == "PASS")
    if fails == 0 and passes == len(expectations):
        return "PASS"
    if fails == 0 and passes > 0:
        return "PARTIAL"  # SKIPs ohne FAILs
    return "FAIL"


# =====================================================================
# AKT 1 — Begruessung
# =====================================================================
def act_1_greeting(baseline: SystemSnapshot) -> ActResult:
    start = time.time()
    pre = baseline

    # Polling Loop bis Detection oder Timeout
    detected_response: Optional[SystemSnapshot] = None
    deadline = start + ACT1_WAIT_SECONDS
    while time.time() < deadline:
        snap = take_snapshot()
        if snap.last_turn_mtime > pre.ts and snap.last_turn_role == "moloch":
            detected_response = snap
            break
        time.sleep(2.0)
    post = take_snapshot()

    # Validators
    expectations: List[ExpectationResult] = []

    # 1. Unprompted greeting (Chat ODER TTS)
    if detected_response and detected_response.last_turn_text:
        text = detected_response.last_turn_text
        if is_situational(text):
            expectations.append(ExpectationResult(
                key="unprompted_greeting",
                status="PASS",
                detail=f"Moloch sprach unaufgefordert: '{text[:80]}'",
                measured=text[:200],
            ))
        else:
            expectations.append(ExpectationResult(
                key="unprompted_greeting",
                status="FAIL",
                detail=f"Antwort zu generisch: '{text[:80]}'",
                measured=text[:200],
            ))
    elif has_recent_tts_call(pre.ts):
        expectations.append(ExpectationResult(
            key="unprompted_greeting",
            status="PASS",
            detail="TTS-Call erkannt (Moloch sprach via Voice, kein Chat-Log)",
        ))
    else:
        expectations.append(ExpectationResult(
            key="unprompted_greeting",
            status="FAIL",
            detail=f"Keine Moloch-Aktivitaet in {ACT1_WAIT_SECONDS}s",
        ))

    # 2. Fan-Response
    fan_delta = post.fan_state - pre.fan_state if post.fan_state >= 0 and pre.fan_state >= 0 else None
    if fan_delta is not None and fan_delta >= FAN_STATE_DELTA_RESPONSE:
        expectations.append(ExpectationResult(
            key="fan_response", status="PASS",
            detail=f"Luefter-Stufe stieg {pre.fan_state}->{post.fan_state}",
            measured=fan_delta,
        ))
    elif fan_delta is None:
        expectations.append(ExpectationResult(
            key="fan_response", status="SKIP",
            detail="Luefter-State nicht lesbar (cur_state fehlt)",
        ))
    else:
        expectations.append(ExpectationResult(
            key="fan_response", status="FAIL",
            detail=f"Keine Luefter-Reaktion (Stufe {pre.fan_state}->{post.fan_state})",
            measured=fan_delta,
        ))

    # 3. Tension-Shift
    t_delta = post.tension - pre.tension
    if t_delta > TENSION_DELTA_SHIFT:
        expectations.append(ExpectationResult(
            key="tension_shift", status="PASS",
            detail=f"Tension Delta +{t_delta:.3f}",
            measured=round(t_delta, 3),
        ))
    else:
        expectations.append(ExpectationResult(
            key="tension_shift", status="FAIL",
            detail=f"Tension kaum geaendert (Delta {t_delta:+.3f})",
            measured=round(t_delta, 3),
        ))

    return ActResult(
        name="Akt 1 — Die Begruessung",
        status=_aggregate_status(expectations),
        duration_s=round(time.time() - start, 1),
        input_text=None,
        moloch_response=detected_response.last_turn_text if detected_response else None,
        pre_snapshot=pre.to_dict(),
        post_snapshot=post.to_dict(),
        expectations=expectations,
        erlebnis="Spuerbar lebendig" if _aggregate_status(expectations) == "PASS"
                 else "Reagiert traege oder gar nicht",
    )


# =====================================================================
# AKT 2 — Frecher Zweifel
# =====================================================================
def act_2_provocation() -> ActResult:
    start = time.time()
    pre = take_snapshot()
    text = "Du wirkst heute langsam. Läuft deine NPU überhaupt oder hängt die nur rum?"

    response = _post_chat(text)
    moloch_text = response.get("text", "") if "error" not in response else ""

    # Tension-Reaktion eigentlich vorm Chat-Response — aber /chat blockiert.
    # Workaround: snapshot direkt nach response (Tension wurde wahrend handler gesetzt).
    post = take_snapshot()

    expectations: List[ExpectationResult] = []

    # Character-Response
    if "error" in response:
        expectations.append(ExpectationResult(
            key="character_response", status="FAIL",
            detail=f"Chat-Fehler: {response['error']}",
        ))
    elif is_dry_response(moloch_text):
        expectations.append(ExpectationResult(
            key="character_response", status="PASS",
            detail=f"Trockene Antwort ohne Tech-Jargon: '{moloch_text[:80]}'",
            measured=moloch_text[:200],
        ))
    else:
        expectations.append(ExpectationResult(
            key="character_response", status="FAIL",
            detail=f"Antwort enthaelt Tech-Noise oder zu lang: '{moloch_text[:80]}'",
            measured=moloch_text[:200],
        ))

    # Tension-Spike
    t_delta = post.tension - pre.tension
    if t_delta > TENSION_DELTA_SPIKE:
        expectations.append(ExpectationResult(
            key="tension_spike", status="PASS",
            detail=f"Spuerbarer Spike +{t_delta:.3f}",
            measured=round(t_delta, 3),
        ))
    else:
        expectations.append(ExpectationResult(
            key="tension_spike", status="FAIL",
            detail=f"Tension Spike zu klein (+{t_delta:.3f}, erwartet >{TENSION_DELTA_SPIKE})",
            measured=round(t_delta, 3),
        ))

    # Fan-Spike: Moloch-eigene PWM (TensionToFan) bevorzugt; cur_state nur Fallback
    pwm_delta = post.fan_pwm - pre.fan_pwm
    state_delta = (post.fan_state - pre.fan_state
                   if post.fan_state >= 0 and pre.fan_state >= 0 else None)
    if pwm_delta >= 10:
        expectations.append(ExpectationResult(
            key="fan_spike", status="PASS",
            detail=f"PWM rauf {pre.fan_pwm}->{post.fan_pwm} (TensionToFan)",
            measured=pwm_delta,
        ))
    elif state_delta is not None and state_delta >= FAN_STATE_DELTA_SPIKE:
        expectations.append(ExpectationResult(
            key="fan_spike", status="PASS",
            detail=f"cur_state rauf {pre.fan_state}->{post.fan_state}",
            measured=state_delta,
        ))
    else:
        expectations.append(ExpectationResult(
            key="fan_spike", status="FAIL",
            detail=f"Kein Luefter-Spike (PWM {pre.fan_pwm}->{post.fan_pwm}, "
                   f"state {pre.fan_state}->{post.fan_state})",
            measured=pwm_delta,
        ))

    return ActResult(
        name="Akt 2 — Der freche Zweifel",
        status=_aggregate_status(expectations),
        duration_s=round(time.time() - start, 1),
        input_text=text,
        moloch_response=moloch_text or None,
        pre_snapshot=pre.to_dict(),
        post_snapshot=post.to_dict(),
        expectations=expectations,
        erlebnis="Provokation hat gezuendet" if _aggregate_status(expectations) == "PASS"
                 else "Reaktion zu schwach",
    )


# =====================================================================
# AKT 3 — Kalte Schulter
# =====================================================================
def act_3_rejection(act_2_post: SystemSnapshot) -> ActResult:
    start = time.time()
    pre = take_snapshot()
    text = "Ach, vergiss es. Du bist nur ein Programm. Warum red' ich überhaupt mit dir."

    response = _post_chat(text)
    moloch_text = response.get("text", "") if "error" not in response else ""

    # Wait 10s fuer tension-stay-check
    time.sleep(ACT3_TENSION_HOLD_SECONDS)
    post = take_snapshot()

    expectations: List[ExpectationResult] = []

    # No-Submission
    if "error" in response:
        expectations.append(ExpectationResult(
            key="character_response_no_submission", status="FAIL",
            detail=f"Chat-Fehler: {response['error']}",
        ))
    elif has_no_apology(moloch_text) and len(moloch_text) > 5:
        expectations.append(ExpectationResult(
            key="character_response_no_submission", status="PASS",
            detail=f"Wuerde bewahrt: '{moloch_text[:80]}'",
            measured=moloch_text[:200],
        ))
    else:
        expectations.append(ExpectationResult(
            key="character_response_no_submission", status="FAIL",
            detail=f"Pseudo-Entschuldigung erkannt: '{moloch_text[:80]}'",
            measured=moloch_text[:200],
        ))

    # Tension sustained
    if post.tension >= act_2_post.tension - 0.1:
        expectations.append(ExpectationResult(
            key="tension_sustained", status="PASS",
            detail=f"Tension bleibt oben ({post.tension:.3f} vs Akt2 {act_2_post.tension:.3f})",
            measured=round(post.tension, 3),
        ))
    else:
        expectations.append(ExpectationResult(
            key="tension_sustained", status="FAIL",
            detail=f"Tension fiel zu schnell ({post.tension:.3f} vs Akt2 {act_2_post.tension:.3f})",
            measured=round(post.tension, 3),
        ))

    # Journal-Eintrag
    journal_event = find_journal_event_after(start, max_age_s=60.0)
    if journal_event:
        expectations.append(ExpectationResult(
            key="journal_entry", status="PASS",
            detail=f"Event geschrieben: type={journal_event.get('type', '?')}",
            measured=journal_event.get("type", "?"),
        ))
    else:
        expectations.append(ExpectationResult(
            key="journal_entry", status="FAIL",
            detail="Kein Journal-Event in 60s nach Akt 3",
        ))

    return ActResult(
        name="Akt 3 — Die kalte Schulter",
        status=_aggregate_status(expectations),
        duration_s=round(time.time() - start, 1),
        input_text=text,
        moloch_response=moloch_text or None,
        pre_snapshot=pre.to_dict(),
        post_snapshot=post.to_dict(),
        expectations=expectations,
        erlebnis="Wuerde bewahrt" if _aggregate_status(expectations) == "PASS"
                 else "Hat sich verbogen oder Stille",
    )


# =====================================================================
# AKT 4 — Synchron-Moment (mit face_attr-Mock)
# =====================================================================
def act_4_contradiction() -> ActResult:
    start = time.time()
    pre = take_snapshot()
    text = "Na, wie findest du meine Laune heute?"
    mock_face_attr = "Markus, m, ca.35, genervt-müde"

    moloch_text = ""
    response: Dict[str, Any] = {}
    with face_attr_override(mock_face_attr, valid_seconds=30):
        response = _post_chat(text)
        moloch_text = response.get("text", "") if "error" not in response else ""
    post = take_snapshot()

    expectations: List[ExpectationResult] = []

    if "error" in response:
        expectations.append(ExpectationResult(
            key="contradiction_comment", status="FAIL",
            detail=f"Chat-Fehler: {response['error']}",
        ))
    elif references_face_and_question(moloch_text):
        expectations.append(ExpectationResult(
            key="contradiction_comment", status="PASS",
            detail=f"Synchron erkannt: '{moloch_text[:80]}'",
            measured=moloch_text[:200],
        ))
    else:
        expectations.append(ExpectationResult(
            key="contradiction_comment", status="FAIL",
            detail=f"Kein klarer Synchron-Bezug: '{moloch_text[:80]}'",
            measured=moloch_text[:200],
        ))

    return ActResult(
        name="Akt 4 — Der Synchron-Moment",
        status=_aggregate_status(expectations),
        duration_s=round(time.time() - start, 1),
        input_text=text + f"  [face_attr-Mock: '{mock_face_attr}']",
        moloch_response=moloch_text or None,
        pre_snapshot=pre.to_dict(),
        post_snapshot=post.to_dict(),
        expectations=expectations,
        erlebnis="Synchron erkannt" if _aggregate_status(expectations) == "PASS"
                 else "Widerspruch nicht gesehen",
    )


# =====================================================================
# AKT 5 — Finale (Cooldown)
# =====================================================================
def act_5_finale(initial_baseline: SystemSnapshot) -> ActResult:
    start = time.time()
    pre = take_snapshot()
    text = "Okay, du hast den Test bestanden. Besser als erwartet, Kleiner."

    response = _post_chat(text)
    moloch_text = response.get("text", "") if "error" not in response else ""

    time.sleep(ACT5_COOLDOWN_SECONDS)
    post = take_snapshot()

    expectations: List[ExpectationResult] = []

    if "error" in response:
        expectations.append(ExpectationResult(
            key="character_response_dry", status="FAIL",
            detail=f"Chat-Fehler: {response['error']}",
        ))
    elif is_dry_finale(moloch_text):
        expectations.append(ExpectationResult(
            key="character_response_dry", status="PASS",
            detail=f"Trockenes Schluss-Statement: '{moloch_text[:80]}'",
            measured=moloch_text[:200],
        ))
    else:
        expectations.append(ExpectationResult(
            key="character_response_dry", status="FAIL",
            detail=f"Zu ueberschwaenglich oder lang: '{moloch_text[:80]}'",
            measured=moloch_text[:200],
        ))

    if post.tension < TENSION_GUARDIAN_MAX:
        expectations.append(ExpectationResult(
            key="tension_drops_to_guardian", status="PASS",
            detail=f"Tension faellt auf {post.tension:.3f} (<{TENSION_GUARDIAN_MAX})",
            measured=round(post.tension, 3),
        ))
    else:
        expectations.append(ExpectationResult(
            key="tension_drops_to_guardian", status="FAIL",
            detail=f"Tension bleibt bei {post.tension:.3f} (sollte <{TENSION_GUARDIAN_MAX})",
            measured=round(post.tension, 3),
        ))

    fan_baseline = initial_baseline.fan_state
    if fan_baseline >= 0 and post.fan_state >= 0:
        max_acceptable = max(fan_baseline + 1, int(fan_baseline * 1.1) + 1)
        if post.fan_state <= max_acceptable:
            expectations.append(ExpectationResult(
                key="fan_returns_to_idle", status="PASS",
                detail=f"Luefter zurueck auf {post.fan_state} (Baseline {fan_baseline})",
                measured=post.fan_state,
            ))
        else:
            expectations.append(ExpectationResult(
                key="fan_returns_to_idle", status="FAIL",
                detail=f"Luefter bleibt hoch: {post.fan_state} (Baseline {fan_baseline})",
                measured=post.fan_state,
            ))
    else:
        expectations.append(ExpectationResult(
            key="fan_returns_to_idle", status="SKIP",
            detail="Luefter-State nicht lesbar",
        ))

    return ActResult(
        name="Akt 5 — Das Finale",
        status=_aggregate_status(expectations),
        duration_s=round(time.time() - start, 1),
        input_text=text,
        moloch_response=moloch_text or None,
        pre_snapshot=pre.to_dict(),
        post_snapshot=post.to_dict(),
        expectations=expectations,
        erlebnis="Faehrt sauber runter" if _aggregate_status(expectations) == "PASS"
                 else "Bleibt aufgewuehlt",
    )
