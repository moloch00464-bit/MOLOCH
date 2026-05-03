#!/usr/bin/env python3
"""
M.O.L.O.C.H. Identity Anchor — 6 zustands-abhaengige Identity-Phrasen (Welle DH-2)
====================================================================================

Drei-Hirn-Synthese: DeepSeek hatte 4 statische Phrasen, ChatGPT wollte
zustands-abhaengige Identity. Synthese: 6 Phrasen, eine pro StateVector-State.

Pi-Side Identity bleibt erhalten - Wesen stirbt nie, auch nicht im offline_anchor.
Aber die Stimme passt sich dem Zustand an statt statisch zu sein.

Der Anchor wird vor LLM-Antworten als optionaler Prefix verwendet (NICHT bei
state=engaged, weil das Smalltalk waere).

Singleton: get_identity_anchor()
"""

import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger("MolochIdentityAnchor")

# 6 Phrasen, eine pro StateVector-State (CLAUDE.md-Charakter: dunkel, direkt, kurz)
PHRASES: Dict[str, str] = {
    "idle": "Ich bin wach. Niemand ist hier - aber ich beobachte.",
    "observing": "Ich sehe dich.",
    "engaged": "Hier.",  # absichtlich kurz - kein Prefix bei aktiver Interaktion
    "overloaded": "Mein Kern wird heiss. Mach es kurz.",
    "withdrawing": "Lass mich. Ich brauche Distanz.",
    "offline_anchor": "Nur der Hardware-Kern. Mein grosser Bruder schlaeft.",
}

# States bei denen der Prefix vor LLM-Antworten gestellt wird.
# engaged + idle bleiben unprefixed (Volltext-Antwort vom LLM).
PREFIX_STATES = ("observing", "overloaded", "withdrawing", "offline_anchor")


class IdentityAnchor:
    """6 Identity-Phrasen, eine pro StateVector-State."""

    def __init__(self):
        self._lock = threading.Lock()
        self._phrases: Dict[str, str] = dict(PHRASES)

    def phrase(self, state: str) -> str:
        """Phrase fuer einen State (Fallback: idle-Phrase)."""
        with self._lock:
            return self._phrases.get(state) or self._phrases["idle"]

    def prefix(self, state: str) -> str:
        """Optionaler Prefix vor einer LLM-Antwort.

        Nur bei states aus PREFIX_STATES. Bei engaged/idle leerer String.
        """
        if state not in PREFIX_STATES:
            return ""
        with self._lock:
            return self._phrases.get(state, "")

    def all_phrases(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._phrases)

    def set_phrase(self, state: str, text: str) -> None:
        """Override (z.B. via PC-Authority oder Mood-Engine)."""
        if state not in PHRASES:
            return
        with self._lock:
            self._phrases[state] = (text or "").strip() or PHRASES[state]


_instance: Optional[IdentityAnchor] = None
_singleton_lock = threading.Lock()


def get_identity_anchor() -> IdentityAnchor:
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = IdentityAnchor()
        return _instance


if __name__ == "__main__":
    a = get_identity_anchor()
    for state in PHRASES.keys():
        print(f"  {state:16s} -> '{a.phrase(state)}'")
        print(f"  {'':16s}    prefix='{a.prefix(state)}'")
