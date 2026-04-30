"""Capability-Inventory — Welle 17 (L5).

Leitet aus `/dev/shm/audit_state.json` ab, was Moloch JETZT kann und was nicht.

Mapping Layer-Status -> Faehigkeit:
- PASS  -> can_do
- WARN  -> degraded (geht, aber eingeschraenkt)
- FAIL  -> cannot_do (mit Grund)
- PENDING/None -> ignoriert (Layer noch nicht gemessen)

API:
    collect_capabilities() -> dict mit can_do, cannot_do, degraded, summary_de.

CLI:
    python3 -m core.audit.self_awareness.capability_inventory
        -> JSON-Print auf stdout

Best-effort: Modul crasht nie, gibt bei fehlenden Daten leere Listen +
status='PENDING' zurueck.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("self_awareness.capability_inventory")

AUDIT_STATE_PATH = Path("/dev/shm/audit_state.json")

# Layer-Name -> (Faehigkeit-de, Grund-bei-FAIL-de)
# Reihenfolge bestimmt summary-Wichtigkeit (vorne = wichtiger).
_LAYER_TO_CAPABILITY: List[Tuple[str, str, str]] = [
    ("vision", "sehen", "Pipeline tot"),
    ("npu", "KI-Inferenz", "NPU nicht verfuegbar"),
    ("tracking", "schwenken/folgen", "PTZ-Tracker tot"),
    ("voice", "sprechen und hoeren", "Audio-Pipeline tot"),
    ("personality", "fuehlen", "Personality-Engine tot"),
    ("memory", "erinnern", "Gedaechtnis nicht erreichbar"),
    ("awareness", "wahrnehmen", "Awareness-Engine tot"),
    ("unconscious", "unbewusst denken", "Unterbewusstsein tot"),
    ("bridge", "mit PC reden", "Bridge zur PC offline"),
    ("tentacle", "WiFi-Mic empfangen", "Tentakel offline"),
    ("spotify", "Musik spielen", "Spotify nicht erreichbar"),
    ("hardware", "Kamera/Mic/Disk OK", "Hardware-Defekt"),
]


def _read_audit_state() -> Optional[Dict[str, Any]]:
    """Liest /dev/shm/audit_state.json best-effort. None bei Fehler."""
    try:
        if not AUDIT_STATE_PATH.exists():
            return None
        with open(AUDIT_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[self_awareness] audit_state lesen fehlgeschlagen: %s", e)
        return None


def _layer_status(state: Dict[str, Any], layer_name: str) -> Optional[str]:
    """Gibt Status-String fuer Layer zurueck oder None wenn nicht da."""
    layers = state.get("layers", {})
    if not isinstance(layers, dict):
        return None
    layer = layers.get(layer_name)
    if not isinstance(layer, dict):
        return None
    status = layer.get("status")
    if not isinstance(status, str):
        return None
    return status.upper()


def _layer_detail_reason(state: Dict[str, Any], layer_name: str) -> Optional[str]:
    """Versucht aus layer.detail einen kompakten Grund zu extrahieren."""
    layers = state.get("layers", {})
    if not isinstance(layers, dict):
        return None
    layer = layers.get(layer_name)
    if not isinstance(layer, dict):
        return None
    detail = layer.get("detail")
    if not isinstance(detail, dict):
        return None
    # Bevorzugte Felder fuer kurze Erklaerung
    for key in ("error", "reason", "message", "hint"):
        val = detail.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()[:120]
    return None


def _build_summary_de(can_do: List[str],
                      cannot_do: List[Dict[str, str]],
                      degraded: List[Dict[str, str]]) -> str:
    """Ein Satz fuer LLM-System-Prompt. Max ~200 Zeichen, deutsch, kurz."""
    n_can = len(can_do)
    if n_can == 0 and not cannot_do and not degraded:
        return "Ich weiss noch nicht, was ich kann — Audit laeuft."
    head_caps = can_do[:3]
    head_str = ", ".join(head_caps) if head_caps else "noch nichts gemessen"
    parts: List[str] = []
    if n_can > 0:
        if n_can <= 3:
            parts.append(f"Ich kann gerade: {head_str}.")
        else:
            parts.append(f"Ich kann gerade {n_can} Dinge: {head_str} und mehr.")
    if cannot_do:
        broken = ", ".join(c.get("capability", "?") for c in cannot_do[:3])
        parts.append(f"Was nicht klappt: {broken}.")
    elif degraded:
        weak = ", ".join(d.get("capability", "?") for d in degraded[:2])
        parts.append(f"Eingeschraenkt: {weak}.")
    summary = " ".join(parts)
    if len(summary) > 200:
        summary = summary[:197] + "..."
    return summary


def _compute_status(can_count: int,
                    cannot_count: int,
                    degraded_count: int) -> Tuple[int, int, str]:
    """Aggregierter Self-Awareness-Status fuer Audit-Ankoppelung."""
    total = can_count + cannot_count + degraded_count
    if total == 0:
        return 0, 0, "PENDING"
    score = can_count
    max_score = total
    if cannot_count >= 2:
        status = "FAIL"
    elif cannot_count >= 1 or degraded_count >= 3:
        status = "WARN"
    else:
        status = "PASS"
    return score, max_score, status


def collect_capabilities() -> Dict[str, Any]:
    """Sammelt aktuelle Faehigkeiten aus audit_state.json.

    Returns:
        dict mit:
        - can_do: List[str]
        - cannot_do: List[{"capability", "reason"}]
        - degraded: List[{"capability", "reason"}]
        - summary_de: str (ein Satz fuer System-Prompt)
        - score, max, status: aggregierter Self-Awareness-Score
        - timestamp: ISO
        - detail: rohe Mapping-Info
    """
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state = _read_audit_state()

    can_do: List[str] = []
    cannot_do: List[Dict[str, str]] = []
    degraded: List[Dict[str, str]] = []
    detail: Dict[str, Any] = {"layers_seen": [], "layers_missing": []}

    if state is None:
        return {
            "can_do": [],
            "cannot_do": [],
            "degraded": [],
            "summary_de": "Audit-State nicht erreichbar — ich kenne meinen Zustand nicht.",
            "score": 0,
            "max": 0,
            "status": "PENDING",
            "timestamp": now_iso,
            "detail": {"error": "audit_state.json nicht lesbar"},
        }

    for layer_name, capability, fail_reason in _LAYER_TO_CAPABILITY:
        status = _layer_status(state, layer_name)
        if status is None:
            detail["layers_missing"].append(layer_name)
            continue
        detail["layers_seen"].append(layer_name)
        if status == "PASS":
            can_do.append(capability)
        elif status == "WARN":
            extra = _layer_detail_reason(state, layer_name)
            reason = extra if extra else "eingeschraenkt"
            degraded.append({"capability": capability, "reason": reason})
        elif status == "FAIL":
            extra = _layer_detail_reason(state, layer_name)
            reason = extra if extra else fail_reason
            cannot_do.append({"capability": capability, "reason": reason})
        else:
            # PENDING o.ae. — ignorieren
            detail["layers_missing"].append(layer_name)

    score, max_score, agg_status = _compute_status(
        len(can_do), len(cannot_do), len(degraded)
    )

    summary_de = _build_summary_de(can_do, cannot_do, degraded)

    return {
        "can_do": can_do,
        "cannot_do": cannot_do,
        "degraded": degraded,
        "summary_de": summary_de,
        "score": score,
        "max": max_score,
        "status": agg_status,
        "timestamp": now_iso,
        "detail": detail,
    }


def _main() -> int:
    try:
        out = collect_capabilities()
    except Exception as e:  # last-resort
        out = {
            "can_do": [],
            "cannot_do": [],
            "degraded": [],
            "summary_de": "Selbst-Check fehlgeschlagen.",
            "score": 0,
            "max": 0,
            "status": "FAIL",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "detail": {"error": str(e)},
        }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
