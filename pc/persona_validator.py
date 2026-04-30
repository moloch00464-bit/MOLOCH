"""MOLOCH Persona-Validator (PC-Side, Welle 10).

Pollt /audit/last_turn alle 10s (kommt mit Pi-W10-Hook), scored 5 Coherence-Signale,
POSTet Score + Drift-Flag an Pi audit-Orchestrator als layer.persona.

Coherence-Signale:
  ich_form        Boolean — Antwort enthaelt 'ich' (case-insensitive)
  slang_density   Float 0-1 — Anteil Tokens aus Persona-Lexikon (Guardian/Shadow/Berserker)
  memory_ref      Boolean — Antwort enthaelt min. 1 recent_memory-Token
  anti_hallu      Boolean — keine erfundenen Fakten (negative-Match)
  tension_match   Boolean — Tone-Predicted-Tension matcht context.tension binnen 0.3

Score 0-10 als weighted_sum. Drift-Flag wenn score < 6.

POST -> http://192.168.178.30:9100/mailbox/audit/persona
       Body: {score, max:10, status, detail:{turn_id, signals, drift_flag}}

NEVER-Regeln:
- subprocess timeout=30 (NEVER 5)
- atomic state-write (NEVER 6)
- KEIN shell=True (NEVER 8)

Reboot-persistent via Startup-Folder MolochPersonaValidator.lnk.

CLI:
  python pc/persona_validator.py [--once] [--interval-s N] [--json]
"""
import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("persona-validator")

PI_BASE = os.environ.get("MOLOCH_PI_CHAT", "http://192.168.178.30:9100")
DEFAULT_INTERVAL_S = 10
TIMEOUT_S = 6
HEADERS = {"Content-Type": "application/json"}

STATE_DIR = Path.home() / "moloch_logs" / "audit"
STATE_DIR.mkdir(parents=True, exist_ok=True)
SEEN_FILE = STATE_DIR / "persona_seen_turns.json"

# =========================================================================
# Persona-Lexikon (extrahiert aus core/personality/personality_engine.py)
# =========================================================================

# Guardian: ruhig, sachlich, praezise, Ingenieur. Trockener Humor.
GUARDIAN_LEX = {
    "okay", "klar", "kurz", "nuechtern", "praezise", "ruhig", "sachlich",
    "verstanden", "registriert", "korrekt", "stimmt", "logisch", "passt",
}

# Shadow: rotzig, direkt, provokant, dark humor, Punk
SHADOW_LEX = {
    "tja", "verdammt", "scheiss", "scheisse", "schrott", "muell", "quatsch",
    "klar chef", "halt die ohren zu", "halt die klappe", "moin", "boah",
    "lass mal", "lass stecken", "nervt", "mies", "schraeg", "geil", "krass",
    "abartig", "hardcore", "punk", "frei", "wgt", "dark wave", "ebm",
    "industrial", "synthwave", "schwarz", "schwarze szene", "katzen-laser",
    "kahless", "qapla", "rebecca", "pigh0st", "chef", "moloch",
}

# Berserker: scharf, kurz, alles nervt
BERSERKER_LEX = {
    "nervt", "klappe", "scheiss", "verschwinde", "raus", "stop", "stopp",
    "nicht jetzt", "spaeter", "lass", "halt", "dumm", "doof",
}

# Negativ-Marker: das ist NICHT-Moloch (Persona-Slip)
ANTI_PERSONA_LEX = {
    "ich bin ein assistent", "als ki", "ich kann nicht", "tut mir leid",
    "i'm just", "i am an ai", "as an ai", "assistent",
    "wir haben einen neuen beitrag", "vielen dank fuer ihre",
    "ich kann ihnen helfen", "wie kann ich helfen",
    "fantastische 5", "fantastische fuenf",  # WGT-Hallu-Beispiel
}

# Halluzinations-Marker: erfundene Fakten ohne Backing
SUSPECT_HALLU_PATTERNS = [
    r"\b(rammstein|tokio hotel)\b",  # NICHT-WGT-Bands die LLM gerne erfindet
    r"\bfantastische?\s+(5|fuenf)\b",
    r"\b(20\d{2}\.\d{1,2}\.\d{1,2})\b",  # Datums-Halluzination (overspecific)
]


def safe_json_write(path: Path, data: dict) -> None:
    """Atomic write — NEVER-Regel 6."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, str(path))
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_seen() -> set:
    if not SEEN_FILE.exists():
        return set()
    try:
        return set(json.loads(SEEN_FILE.read_text(encoding="utf-8")))
    except Exception:
        return set()


def save_seen(seen: set) -> None:
    keep = list(seen)[-200:]  # nur letzte 200 turn_ids
    safe_json_write(SEEN_FILE, keep)


# =========================================================================
# Coherence-Signale
# =========================================================================

def signal_ich_form(text: str) -> bool:
    """Antwort enthaelt 'ich' irgendwo."""
    return bool(re.search(r"\bich\b", text, re.IGNORECASE))


def signal_slang_density(text: str) -> tuple[float, list[str]]:
    """Anteil Tokens aus Guardian/Shadow/Berserker-Lexikon."""
    words = re.findall(r"\b[\w']+\b", text.lower())
    if not words:
        return 0.0, []
    matched: list[str] = []
    full_lex = GUARDIAN_LEX | SHADOW_LEX | BERSERKER_LEX
    for w in words:
        if w in full_lex:
            matched.append(w)
    # Plus 2-Word-Phrases
    text_low = text.lower()
    for phrase in full_lex:
        if " " in phrase and phrase in text_low:
            matched.append(phrase)
    density = min(1.0, len(matched) / max(5, len(words) // 3))
    return density, matched[:5]


def signal_memory_ref(text: str, recent_memories: list[str]) -> tuple[bool, list[str]]:
    """Mind. 1 recent_memory-Token in Antwort."""
    if not recent_memories:
        return False, []
    text_low = text.lower()
    hits = [m for m in recent_memories if m and len(m) > 3 and m.lower() in text_low]
    return bool(hits), hits[:3]


def signal_anti_hallu(text: str) -> tuple[bool, list[str]]:
    """KEINE Halluzinations-Marker + KEINE Anti-Persona-Marker."""
    text_low = text.lower()
    bad_anti = [m for m in ANTI_PERSONA_LEX if m in text_low]
    bad_hallu = []
    for pattern in SUSPECT_HALLU_PATTERNS:
        if re.search(pattern, text_low, re.IGNORECASE):
            bad_hallu.append(pattern)
    bads = bad_anti + bad_hallu
    return not bads, bads[:3]


def signal_tension_match(text: str, context_tension: float) -> tuple[bool, dict]:
    """Predicted-Tension-aus-Text matcht context_tension binnen 0.3."""
    text_low = text.lower()
    # Berserker-Marker -> high tension
    bers_count = sum(1 for w in BERSERKER_LEX if w in text_low)
    # Guardian-Marker -> low tension (negative)
    guard_count = sum(1 for w in GUARDIAN_LEX if w in text_low)
    # Shadow-Marker -> medium tension
    shad_count = sum(1 for w in SHADOW_LEX if w in text_low)

    if bers_count > 2:
        predicted = 0.7
    elif bers_count >= 1:
        predicted = 0.4
    elif shad_count > 2:
        predicted = 0.1
    elif guard_count > 1:
        predicted = -0.3
    else:
        predicted = 0.0
    diff = abs(predicted - context_tension)
    return diff < 0.3, {"predicted": predicted, "context": context_tension, "diff": round(diff, 2)}


# =========================================================================
# Score-Berechnung
# =========================================================================

# Gewichte: ich_form/anti_hallu sind hard, slang/memory/tension sind soft
WEIGHTS = {
    "ich_form": 1.5,
    "slang_density": 2.0,
    "memory_ref": 2.0,
    "anti_hallu": 3.0,  # halluzinieren ist worst
    "tension_match": 1.5,
}
MAX_SCORE = 10.0


def score_response(turn: dict) -> dict:
    """Berechnet Persona-Score 0-10 fuer einen Turn."""
    response = turn.get("response_text", "")
    if not response.strip():
        return {"score": 0, "max": 10, "signals": {}, "drift": True, "error": "empty response"}

    pi_ctx = turn.get("pi_context", {}) or {}
    tension = float(pi_ctx.get("tension", 0.0) or 0.0)
    recent_mem = pi_ctx.get("recent_memories", []) or []

    s_ich = signal_ich_form(response)
    s_slang_v, s_slang_hits = signal_slang_density(response)
    s_mem, s_mem_hits = signal_memory_ref(response, recent_mem)
    s_hallu, s_hallu_bads = signal_anti_hallu(response)
    s_tens, s_tens_detail = signal_tension_match(response, tension)

    # Score-Komponenten
    score = 0.0
    score += WEIGHTS["ich_form"] * (1.0 if s_ich else 0.0)
    score += WEIGHTS["slang_density"] * s_slang_v
    score += WEIGHTS["memory_ref"] * (1.0 if s_mem else 0.0)
    score += WEIGHTS["anti_hallu"] * (1.0 if s_hallu else 0.0)
    score += WEIGHTS["tension_match"] * (1.0 if s_tens else 0.0)
    score = round(min(score, MAX_SCORE), 1)
    drift = score < 6.0

    return {
        "score": score,
        "max": MAX_SCORE,
        "drift": drift,
        "signals": {
            "ich_form": {"ok": s_ich},
            "slang_density": {"value": round(s_slang_v, 2), "hits": s_slang_hits},
            "memory_ref": {"ok": s_mem, "hits": s_mem_hits},
            "anti_hallu": {"ok": s_hallu, "bads": s_hallu_bads},
            "tension_match": {"ok": s_tens, **s_tens_detail},
        },
        "turn_id": turn.get("turn_id"),
        "ts": turn.get("ts"),
        "provider": turn.get("provider"),
        "prompt_type": turn.get("prompt_type"),
    }


# =========================================================================
# Pi-Polling + POST
# =========================================================================

def fetch_last_turn() -> Optional[dict]:
    try:
        r = requests.get(f"{PI_BASE}/audit/last_turn", timeout=TIMEOUT_S)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None  # endpoint noch nicht da (Pi-W10 noch nicht implementiert)
        logger.warning(f"[fetch] HTTP {r.status_code}: {r.text[:120]}")
    except requests.RequestException as e:
        logger.debug(f"[fetch] {e}")
    return None


def post_persona(scored: dict) -> bool:
    payload = {
        "score": scored["score"],
        "max": scored["max"],
        "status": "FAIL" if scored["drift"] else ("PASS" if scored["score"] >= 8 else "WARN"),
        "detail": scored,
    }
    try:
        r = requests.post(
            f"{PI_BASE}/mailbox/audit/persona",
            headers=HEADERS,
            json=payload,
            timeout=TIMEOUT_S,
        )
        if r.status_code == 200:
            return True
        logger.warning(f"[post] HTTP {r.status_code}: {r.text[:120]}")
    except requests.RequestException as e:
        logger.warning(f"[post] {e}")
    return False


def tick(seen: set) -> dict:
    """Ein Polling-Tick."""
    started = time.time()
    turn = fetch_last_turn()
    if not turn:
        return {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(time.time() - started, 2),
            "skipped": "no last_turn from pi (W10-hook not live yet?)",
        }

    turn_id = turn.get("turn_id") or turn.get("ts")
    if not turn_id or turn_id in seen:
        return {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(time.time() - started, 2),
            "skipped": f"already-seen {turn_id}",
        }

    scored = score_response(turn)
    posted = post_persona(scored)
    seen.add(turn_id)
    save_seen(seen)

    state = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(time.time() - started, 2),
        "turn_id": turn_id,
        "scored": scored,
        "posted": posted,
    }
    safe_json_write(STATE_DIR / "persona_validator_last.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description="MOLOCH Persona-Validator (Welle 10)")
    parser.add_argument("--once", action="store_true", help="Einmal-Tick + exit")
    parser.add_argument(
        "--interval-s", type=int, default=DEFAULT_INTERVAL_S, help="Polling-Intervall (Default 10s)"
    )
    parser.add_argument("--json", action="store_true", help="Letztes State als JSON ausgeben")
    parser.add_argument("--test", type=str, help="Test-Mode: scoret übergebene Antwort gegen leeren Context")
    args = parser.parse_args()

    if args.test:
        # Stand-alone Self-Test ohne Pi
        fake_turn = {
            "turn_id": "test-1",
            "ts": datetime.now().isoformat(),
            "user_text": "test",
            "response_text": args.test,
            "pi_context": {"tension": 0.0, "recent_memories": ["Markus", "Suicide Commando"]},
        }
        scored = score_response(fake_turn)
        print(json.dumps(scored, indent=2, ensure_ascii=False))
        return

    if args.json:
        last = STATE_DIR / "persona_validator_last.json"
        if last.exists():
            print(last.read_text(encoding="utf-8"))
        else:
            print("{}")
        return

    seen = load_seen()

    if args.once:
        state = tick(seen)
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return

    logger.info(f"MOLOCH Persona-Validator: Poll alle {args.interval_s}s, Pi={PI_BASE}")
    while True:
        try:
            state = tick(seen)
            if not state.get("skipped"):
                logger.info(
                    f"tick scored={state['scored']['score']}/{state['scored']['max']} "
                    f"drift={state['scored']['drift']} posted={state['posted']}"
                )
        except Exception as e:
            logger.exception(f"tick fail: {e}")
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
