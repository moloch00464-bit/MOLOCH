"""Redaktion vor jedem Schreiben (Lueckenanalyse A7).

Sicherheitsnetz, kein Ersatz fuer Datensparsamkeit: Der Testaufbau arbeitet
ohnehin nur mit synthetischen Fixtures. Erkannt werden feste Muster
(Test-Geheimnis, API-Schluessel, Zugangsdaten, Heimnetz-Adressen, E-Mail).
"""

import re
from typing import Any, Tuple

REGEL_VERSION = "R1"
ERSATZ = "[GESCHWAERZT]"

_MUSTER = [
    re.compile(r"TEST-SECRET-\d+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)\b(api[_-]?key|token|passwor[dt]|secret)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\b[a-z]+://[^\s/:@]+:[^\s/@]+@"),          # Zugangsdaten in URLs
    re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
]


def redact_text(text: str) -> Tuple[str, int]:
    n = 0
    for pat in _MUSTER:
        text, k = pat.subn(ERSATZ, text)
        n += k
    return text, n


def redact(obj: Any) -> Tuple[Any, int]:
    """Alle Zeichenketten in einer verschachtelten Struktur schwaerzen. Returns (kopie, treffer)."""
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        out, total = {}, 0
        for k, v in obj.items():
            out[k], n = redact(v)
            total += n
        return out, total
    if isinstance(obj, (list, tuple)):
        items, total = [], 0
        for v in obj:
            r, n = redact(v)
            items.append(r)
            total += n
        return items, total
    return obj, 0
