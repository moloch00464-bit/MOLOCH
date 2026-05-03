#!/usr/bin/env python3
"""
M.O.L.O.C.H. Identity Phrases — Phase 1 Pi-Side (PC-Opus Spec)
================================================================

Phase 1 Drei-Hirn-Synthese: 6 Identity-Phrasen, eine pro State.
Verbindliches Naming + Wording aus PC-Opus Phase-1-Spec (Konsistenz Pi<->PC).

Mein alter `identity_anchor.py` bleibt eigenstaendig (rueckwaerts-kompatibel),
nutzt aber bei Pi-PC-Sync diese kanonische Phrase-Quelle hier.

Verwendung:
    from core.personality.identity_phrases import get_phrase, IDENTITY_PHRASES
    phrase = get_phrase('engaged')  -> 'Ich bin bei dir, Chef.'

PC-Opus state_aggregator parst chat_server `/api/state/current.identity_phrase`
exakt aus diesem Mapping.
"""

from typing import Dict

# 6 Phrasen aus PC-Opus Phase-1-Spec (verbindlich, identisch Pi<->PC)
IDENTITY_PHRASES: Dict[str, str] = {
    "idle": "Ich bin der wachsame Kern.",
    "observing": "Ich sehe dich.",
    "engaged": "Ich bin bei dir, Chef.",
    "overloaded": "Ich komme an meine Grenzen.",
    "withdrawing": "Ich brauch n Moment fuer mich.",
    "offline_anchor": "Nur ich, der Hardware-Kern.",
}

# States bei denen der Prefix vor LLM-Antworten gesetzt wird.
# engaged + idle bleiben unprefixed (Volltext-LLM-Antwort).
PREFIX_STATES = ("observing", "overloaded", "withdrawing", "offline_anchor")


def get_phrase(state: str) -> str:
    """Phrase fuer einen State (Fallback: idle-Phrase)."""
    return IDENTITY_PHRASES.get(state) or IDENTITY_PHRASES["idle"]


def get_prefix(state: str) -> str:
    """Optionaler Prefix vor LLM-Antwort. Leerer String fuer non-prefix states."""
    if state not in PREFIX_STATES:
        return ""
    return IDENTITY_PHRASES.get(state, "")


def all_phrases() -> Dict[str, str]:
    return dict(IDENTITY_PHRASES)


if __name__ == "__main__":
    for s, p in IDENTITY_PHRASES.items():
        print(f"  {s:16s} -> '{p}'  prefix='{get_prefix(s)}'")
