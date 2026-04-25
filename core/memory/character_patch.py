#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Character Patch
==============================

Welle 1 / W1.1 von ThreeBrain FineTune Loop.

WAS DAS HIER IST: Eine Sammlung von Verhaltens-Regeln, die Moloch zugewiesen
bekommen hat. character_drift.json sagt WAS PASSIERTE — character_patch.json
sagt WAS TUN. Distiller schlaegt vor (pending), Markus approved (active),
ledger protokolliert.

Storage:
  /mnt/moloch-data/memory/character_patch.json

Schema:
  {
    "updated_at": "...",
    "active_rules": [{"id", "trigger", "behavior", "source_event_ids",
                      "approved_at", "approved_by", "active"}, ...],
    "pending_rules": [{...}],
    "rejected_rules": [{... "rejected_at", "rejected_by", "reason"}],
    "next_rule_id": 1
  }

Singleton: get_patch()

API:
  patch.add_pending_rule(trigger, behavior, source_event_ids, **meta) -> rule_id
  patch.approve(rule_id, by="markus")
  patch.reject(rule_id, reason, by="markus")
  patch.get_active_rules() -> List[dict]
  patch.get_pending_rules() -> List[dict]
  patch.prompt_snippet(max_chars=400) -> str   # fuer Cloud System-Prompt
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("CharacterPatch")

PATCH_PATH = "/mnt/moloch-data/memory/character_patch.json"
MAX_PROMPT_RULES = 8        # max in prompt_snippet
MAX_TRIGGER_LEN = 200
MAX_BEHAVIOR_LEN = 280


def _utc_iso_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_write_json(path: str, data: Any) -> None:
    """Atomar schreiben (Pattern aus core/memory/character_journal.py)."""
    tmp_path = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp_path, path)
        except OSError:
            with open(tmp_path, "r", encoding="utf-8") as f_src:
                content = f_src.read()
            with open(path, "w", encoding="utf-8") as f_dst:
                f_dst.write(content)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        logger.error(f"[PATCH] Schreiben fehlgeschlagen ({path}): {e}")
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _safe_read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[PATCH] Lesen fehlgeschlagen ({path}): {e}")
        return default


def _empty_doc() -> Dict[str, Any]:
    return {
        "updated_at": _utc_iso_ms(),
        "active_rules": [],
        "pending_rules": [],
        "rejected_rules": [],
        "next_rule_id": 1,
    }


def _try_log_ledger(event: str, **meta) -> None:
    """Best-effort log to behavior_mutation_ledger (lazy import, never crashes patch)."""
    try:
        from core.memory.behavior_mutation_ledger import get_ledger
        get_ledger().log(event, **meta)
    except Exception as e:
        logger.debug(f"[PATCH] Ledger-Log Fehler: {e}")


# =============================================================================
# CharacterPatch
# =============================================================================

class CharacterPatch:
    """Verhaltens-Regeln mit Approval-Workflow."""

    def __init__(self):
        self._lock = threading.Lock()
        self._doc = _safe_read_json(PATCH_PATH, default=_empty_doc())
        # Schema-Migration falls Felder fehlen
        for k, v in _empty_doc().items():
            self._doc.setdefault(k, v)
        logger.info(
            f"[PATCH] Initialisiert: active={len(self._doc['active_rules'])} "
            f"pending={len(self._doc['pending_rules'])} "
            f"rejected={len(self._doc['rejected_rules'])}"
        )

    # ---------------------------------------------------------------- WRITE

    def add_pending_rule(
        self,
        trigger: str,
        behavior: str,
        source_event_ids: Optional[List[str]] = None,
        proposed_by: str = "distiller",
        **meta,
    ) -> Optional[str]:
        """Neue Regel als pending eintragen. Returns rule_id."""
        if not trigger or not trigger.strip():
            logger.warning("[PATCH] Leerer trigger — verworfen")
            return None
        if not behavior or not behavior.strip():
            logger.warning("[PATCH] Leeres behavior — verworfen")
            return None

        trigger = trigger.strip()[:MAX_TRIGGER_LEN]
        behavior = behavior.strip()[:MAX_BEHAVIOR_LEN]
        source_event_ids = list(source_event_ids or [])

        with self._lock:
            rid = self._doc["next_rule_id"]
            rule_id = f"rule_{rid:08d}"
            self._doc["next_rule_id"] = rid + 1
            rule = {
                "id": rule_id,
                "trigger": trigger,
                "behavior": behavior,
                "source_event_ids": source_event_ids,
                "proposed_at": _utc_iso_ms(),
                "proposed_by": proposed_by,
                "active": False,
            }
            for k, v in meta.items():
                if k not in rule:
                    rule[k] = v
            self._doc["pending_rules"].append(rule)
            self._doc["updated_at"] = _utc_iso_ms()
            _safe_write_json(PATCH_PATH, self._doc)

        _try_log_ledger("rule_proposed", rule_id=rule_id, by=proposed_by,
                        trigger=trigger[:80], behavior=behavior[:80])
        logger.info(f"[PATCH] Pending: {rule_id} — {trigger[:50]}")
        return rule_id

    def approve(self, rule_id: str, by: str = "markus") -> bool:
        """Pending-Rule -> active."""
        with self._lock:
            rule = self._pop_from(self._doc["pending_rules"], rule_id)
            if not rule:
                logger.warning(f"[PATCH] approve: {rule_id} nicht in pending")
                return False
            rule["approved_at"] = _utc_iso_ms()
            rule["approved_by"] = by
            rule["active"] = True
            self._doc["active_rules"].append(rule)
            self._doc["updated_at"] = _utc_iso_ms()
            _safe_write_json(PATCH_PATH, self._doc)

        _try_log_ledger("rule_approved", rule_id=rule_id, by=by)
        logger.info(f"[PATCH] APPROVED: {rule_id}")
        return True

    def reject(self, rule_id: str, reason: str = "", by: str = "markus") -> bool:
        """Pending-Rule -> rejected (mit Grund)."""
        with self._lock:
            rule = self._pop_from(self._doc["pending_rules"], rule_id)
            if not rule:
                logger.warning(f"[PATCH] reject: {rule_id} nicht in pending")
                return False
            rule["rejected_at"] = _utc_iso_ms()
            rule["rejected_by"] = by
            rule["reason"] = (reason or "").strip()[:200]
            rule["active"] = False
            self._doc["rejected_rules"].append(rule)
            self._doc["updated_at"] = _utc_iso_ms()
            _safe_write_json(PATCH_PATH, self._doc)

        _try_log_ledger("rule_rejected", rule_id=rule_id, by=by, reason=reason[:80])
        logger.info(f"[PATCH] REJECTED: {rule_id} ({reason[:50]})")
        return True

    def deactivate(self, rule_id: str, by: str = "markus") -> bool:
        """Aktive Regel ausschalten (bleibt in active_rules, aber active=False)."""
        with self._lock:
            for r in self._doc["active_rules"]:
                if r.get("id") == rule_id:
                    r["active"] = False
                    r["deactivated_at"] = _utc_iso_ms()
                    r["deactivated_by"] = by
                    self._doc["updated_at"] = _utc_iso_ms()
                    _safe_write_json(PATCH_PATH, self._doc)
                    _try_log_ledger("rule_deactivated", rule_id=rule_id, by=by)
                    logger.info(f"[PATCH] deactivated: {rule_id}")
                    return True
        return False

    @staticmethod
    def _pop_from(lst: List[Dict], rule_id: str) -> Optional[Dict]:
        for i, r in enumerate(lst):
            if r.get("id") == rule_id:
                return lst.pop(i)
        return None

    # ----------------------------------------------------------------- READ

    def get_active_rules(self) -> List[Dict]:
        """Nur tatsaechlich aktive Regeln (active=True)."""
        with self._lock:
            return [dict(r) for r in self._doc["active_rules"] if r.get("active")]

    def get_pending_rules(self) -> List[Dict]:
        with self._lock:
            return [dict(r) for r in self._doc["pending_rules"]]

    def get_rejected_rules(self) -> List[Dict]:
        with self._lock:
            return [dict(r) for r in self._doc["rejected_rules"]]

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "updated_at": self._doc.get("updated_at"),
                "active_count": sum(1 for r in self._doc["active_rules"] if r.get("active")),
                "pending_count": len(self._doc["pending_rules"]),
                "rejected_count": len(self._doc["rejected_rules"]),
                "next_rule_id": self._doc.get("next_rule_id", 1),
            }

    # ----------------------------------------------- PROMPT-SNIPPET (Cloud)

    def prompt_snippet(self, max_chars: int = 400) -> str:
        """Kompakte Darstellung der aktiven Regeln fuer Cloud-System-Prompt.

        Beispiel-Output:
          === AKTIVE VERHALTENSREGELN (gelernt aus Erfahrung) ===
          - Wenn tension>0.8 + Markus: schaerfere Antworten, kuerzer
          - Wenn Beleidigung detektiert: 1 trockener Satz, kein Kommentar
        """
        active = self.get_active_rules()
        if not active:
            return ""
        lines = ["=== AKTIVE VERHALTENSREGELN (gelernt aus Erfahrung) ==="]
        used = len(lines[0])
        for r in active[:MAX_PROMPT_RULES]:
            line = f"- Wenn {r.get('trigger', '?')}: {r.get('behavior', '?')}"
            if used + len(line) + 1 > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)


# =============================================================================
# Singleton
# =============================================================================

_instance: Optional[CharacterPatch] = None
_instance_lock = threading.Lock()


def get_patch() -> CharacterPatch:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = CharacterPatch()
    return _instance


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    p = get_patch()
    state0 = p.get_state()
    print(f"\n[Initial] {state0}")

    rid1 = p.add_pending_rule(
        trigger="tension > 0.8 AND person == 'markus'",
        behavior="schaerfere Antworten, kuerzer, 30% mehr Sarkasmus",
        source_event_ids=["evt_00000042"],
    )
    rid2 = p.add_pending_rule(
        trigger="Beleidigung detektiert",
        behavior="ein trockener Satz, kein Kommentar danach",
        source_event_ids=["evt_00000087"],
    )
    rid3 = p.add_pending_rule(
        trigger="Markus laeuft 3x ueber Bild ohne Stop",
        behavior="ignorieren, kein Hello-Spam",
        source_event_ids=["evt_00000123"],
    )
    assert rid1 and rid2 and rid3, "add_pending_rule failed"

    print(f"\n[After 3 pending] {p.get_state()}")
    assert len(p.get_pending_rules()) >= 3

    # Approve 2, reject 1
    assert p.approve(rid1) is True
    assert p.approve(rid2) is True
    assert p.reject(rid3, reason="zu generisch") is True

    state2 = p.get_state()
    print(f"\n[After approve/reject] {state2}")
    assert state2["active_count"] >= 2
    assert state2["rejected_count"] >= 1

    # Validation fail
    assert p.add_pending_rule("", "irgendwas") is None
    assert p.add_pending_rule("trigger", "") is None

    # Re-approve nicht-existierender
    assert p.approve("rule_99999999") is False

    # Snippet
    snip = p.prompt_snippet(max_chars=300)
    print(f"\n[Snippet]\n{snip}")
    assert "AKTIVE VERHALTENSREGELN" in snip

    # Deactivate
    assert p.deactivate(rid1) is True
    state3 = p.get_state()
    assert state3["active_count"] == state2["active_count"] - 1

    print(f"\nSelf-Test PASS — File: {PATCH_PATH}")
