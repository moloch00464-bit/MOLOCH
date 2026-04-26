#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Feedback Store — Trainings-Sample-Pool

Welle 3 / W3.2 von ThreeBrain FineTune Loop.

Sammelt zwei Quellen von Trainings-Samples in EINEN Pool:
  1) **Critic-Samples** (vom finetune_orchestrator, automatisch nachts)
     - source="critic", enthaelt score + critique + better_response
  2) **Markus-Feedback** (👍/👎 im Cockpit)
     - source="thumbs_up" / "thumbs_down", schnelles Markus-Urteil

Pool: /mnt/moloch-data/memory/finetune_samples.jsonl  (append-only)
Counter: /mnt/moloch-data/memory/finetune_samples_state.json

Sample-Schema:
{
  "ts": "...",
  "sample_id": "smp_00000042",
  "source": "critic" | "thumbs_up" | "thumbs_down",
  "situation": "...",        # Frage/Kontext
  "pi_response": "...",      # was Moloch gesagt hat
  "score": 0-10 | null,      # Critic-Score (null bei thumbs)
  "critique": "..." | null,
  "better_response": "..." | null,  # vom Critic, fuer LoRA-Target
  "approved": null | true | false,  # Markus-Review-Status
  "reviewed_at": "..." | null,
  "tags": []
}

Status:
  pending  -> approved=null            (warten auf Markus' Review)
  approved -> approved=true            (geht an LoRA-Trainer)
  rejected -> approved=false           (ignoriert)

Singleton: get_feedback_store()

API:
  add_critic_sample(situation, pi_response, score, critique, better_response, **meta)
  add_thumbs(situation, pi_response, label, **meta)        # label: "up" | "down"
  read_recent(n=50)
  read_pending(limit=200)
  read_approved(limit=200)
  approve(sample_id, by="markus")
  reject(sample_id, by="markus")
  get_state()
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("FeedbackStore")

POOL_PATH = "/mnt/moloch-data/memory/finetune_samples.jsonl"
STATE_PATH = "/mnt/moloch-data/memory/finetune_samples_state.json"

ALLOWED_SOURCES = frozenset({"critic", "thumbs_up", "thumbs_down"})
MAX_SITUATION_LEN = 500
MAX_RESPONSE_LEN = 800


def _utc_iso_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_write_json(path: str, data: Any) -> None:
    """Atomar + NTFS-Fallback."""
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
        logger.error(f"[FEEDBACK] State-Write fehlgeschlagen: {e}")
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _try_log_ledger(event: str, **meta) -> None:
    """Best-effort Ledger-Log (lazy import)."""
    try:
        from core.memory.behavior_mutation_ledger import get_ledger
        get_ledger().log(event, **meta)
    except Exception as e:
        logger.debug(f"[FEEDBACK] Ledger-Log Fehler: {e}")


def _read_all_samples() -> List[Dict]:
    """Alle Samples laden (Pool ist fuer Phase 1 klein, < 10k Eintraege)."""
    if not os.path.exists(POOL_PATH):
        return []
    out: List[Dict] = []
    try:
        with open(POOL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"[FEEDBACK] Read fehlgeschlagen: {e}")
    return out


def _rewrite_pool(samples: List[Dict]) -> None:
    """Komplett neu schreiben (atomar). Fuer approve/reject Updates."""
    tmp = POOL_PATH + ".rewrite.tmp"
    try:
        os.makedirs(os.path.dirname(POOL_PATH), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, POOL_PATH)
    except Exception as e:
        logger.error(f"[FEEDBACK] Rewrite fehlgeschlagen: {e}")
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


# =============================================================================
# FeedbackStore
# =============================================================================

class FeedbackStore:
    """Pool aus Critic-Samples + Markus-Thumbs-Feedback."""

    def __init__(self):
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(POOL_PATH), exist_ok=True)
        self._last_id = self._load_last_id()
        logger.info(f"[FEEDBACK] Init: pool={POOL_PATH} last_id={self._last_id}")

    def _load_last_id(self) -> int:
        if not os.path.exists(STATE_PATH):
            return 0
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return int(json.load(f).get("last_id", 0))
        except Exception:
            return 0

    def _save_last_id(self, last_id: int) -> None:
        _safe_write_json(STATE_PATH, {"last_id": last_id, "updated": _utc_iso_ms()})

    # ---------------------------------------------------------------- WRITE

    def _append(self, entry: Dict) -> Optional[str]:
        """Generischer Append + Counter-Update."""
        with self._lock:
            new_id = self._last_id + 1
            sample_id = f"smp_{new_id:08d}"
            entry["sample_id"] = sample_id
            entry["ts"] = _utc_iso_ms()
            try:
                with open(POOL_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                logger.error(f"[FEEDBACK] Append fehlgeschlagen: {e}")
                return None
            self._last_id = new_id
            self._save_last_id(new_id)
            return sample_id

    def add_critic_sample(
        self,
        situation: str,
        pi_response: str,
        score: int,
        critique: str = "",
        better_response: str = "",
        **meta,
    ) -> Optional[str]:
        """Sample vom finetune_orchestrator (Critic-bewertet)."""
        if not situation or not pi_response:
            logger.warning("[FEEDBACK] add_critic_sample: leer — verworfen")
            return None
        entry = {
            "source": "critic",
            "situation": situation.strip()[:MAX_SITUATION_LEN],
            "pi_response": pi_response.strip()[:MAX_RESPONSE_LEN],
            "score": int(score) if score is not None else None,
            "critique": (critique or "").strip()[:300],
            "better_response": (better_response or "").strip()[:MAX_RESPONSE_LEN],
            "approved": None,        # pending
            "reviewed_at": None,
            "reviewed_by": None,
            "tags": list(meta.pop("tags", []) or []),
        }
        for k, v in meta.items():
            if k not in entry:
                entry[k] = v
        sid = self._append(entry)
        if sid:
            _try_log_ledger("sample_proposed", sample_id=sid, source="critic", score=score)
            logger.info(f"[FEEDBACK] critic-sample {sid} score={score}")
        return sid

    def add_thumbs(
        self,
        situation: str,
        pi_response: str,
        label: str,
        **meta,
    ) -> Optional[str]:
        """Markus-Feedback: 'up' = approved sofort, 'down' = rejected sofort.

        Thumbs sind sofort-entschieden (Markus drueckt = entscheidet).
        Kein zusaetzlicher Review-Step.
        """
        if label not in ("up", "down"):
            logger.warning(f"[FEEDBACK] add_thumbs: invalid label '{label}'")
            return None
        if not pi_response:
            logger.warning("[FEEDBACK] add_thumbs: leere response")
            return None
        source = "thumbs_up" if label == "up" else "thumbs_down"
        approved = (label == "up")
        entry = {
            "source": source,
            "situation": (situation or "").strip()[:MAX_SITUATION_LEN],
            "pi_response": pi_response.strip()[:MAX_RESPONSE_LEN],
            "score": 10 if label == "up" else 0,
            "critique": "Markus 👍" if label == "up" else "Markus 👎",
            "better_response": "",  # bei thumbs gibt's keinen Vorschlag
            "approved": approved,
            "reviewed_at": _utc_iso_ms(),
            "reviewed_by": "markus",
            "tags": list(meta.pop("tags", []) or []),
        }
        for k, v in meta.items():
            if k not in entry:
                entry[k] = v
        sid = self._append(entry)
        if sid:
            _try_log_ledger("sample_thumbs", sample_id=sid, label=label,
                            text=pi_response[:60])
            logger.info(f"[FEEDBACK] thumbs-{label} {sid}")
        return sid

    # ---------------------------------------------------------------- REVIEW

    def approve(self, sample_id: str, by: str = "markus") -> bool:
        """Critic-Sample auf approved=True setzen."""
        return self._set_review(sample_id, approved=True, by=by)

    def reject(self, sample_id: str, by: str = "markus") -> bool:
        return self._set_review(sample_id, approved=False, by=by)

    def _set_review(self, sample_id: str, approved: bool, by: str) -> bool:
        with self._lock:
            samples = _read_all_samples()
            hit = False
            for s in samples:
                if s.get("sample_id") == sample_id:
                    s["approved"] = approved
                    s["reviewed_at"] = _utc_iso_ms()
                    s["reviewed_by"] = by
                    hit = True
                    break
            if hit:
                _rewrite_pool(samples)
        if hit:
            _try_log_ledger(
                "sample_approved" if approved else "sample_rejected",
                sample_id=sample_id, by=by,
            )
            logger.info(f"[FEEDBACK] {'APPROVED' if approved else 'REJECTED'}: {sample_id}")
        else:
            logger.warning(f"[FEEDBACK] {sample_id} nicht gefunden")
        return hit

    # ----------------------------------------------------------------- READ

    def read_recent(self, n: int = 50) -> List[Dict]:
        return _read_all_samples()[-n:]

    def read_pending(self, limit: int = 200) -> List[Dict]:
        """Nur Critic-Samples mit approved=None (thumbs sind schon entschieden)."""
        return [s for s in _read_all_samples()
                if s.get("source") == "critic" and s.get("approved") is None][:limit]

    def read_approved(self, limit: int = 500) -> List[Dict]:
        """Alle approved Samples (egal welche Quelle) — fuer LoRA-Trainer."""
        return [s for s in _read_all_samples() if s.get("approved") is True][:limit]

    def get_state(self) -> Dict[str, Any]:
        all_s = _read_all_samples()
        crit = [s for s in all_s if s.get("source") == "critic"]
        return {
            "total": len(all_s),
            "critic": len(crit),
            "thumbs_up": sum(1 for s in all_s if s.get("source") == "thumbs_up"),
            "thumbs_down": sum(1 for s in all_s if s.get("source") == "thumbs_down"),
            "pending_review": sum(1 for s in crit if s.get("approved") is None),
            "approved": sum(1 for s in all_s if s.get("approved") is True),
            "rejected": sum(1 for s in all_s if s.get("approved") is False),
            "last_id": self._last_id,
            "pool_path": POOL_PATH,
        }


# =============================================================================
# Singleton
# =============================================================================

_instance: Optional[FeedbackStore] = None
_instance_lock = threading.Lock()


def get_feedback_store() -> FeedbackStore:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = FeedbackStore()
    return _instance


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    fs = get_feedback_store()
    print(f"\n[Initial] {fs.get_state()}")

    # Critic-Sample
    sid1 = fs.add_critic_sample(
        situation="Markus fragte: wie geht's dir?",
        pi_response="Hallo Markus, schoen dich zu sehen!",
        score=2, critique="zu generisch, kein Charakter",
        better_response="Laeuft. Bisschen hungrig auf Strom. Du?",
    )
    sid2 = fs.add_critic_sample(
        situation="Markus fragte: was hast du heute gemacht?",
        pi_response="Ich habe verschiedene Aufgaben erledigt.",
        score=1, critique="Assistent-Sprech, kein Moloch",
        better_response="Geguckt. Geguckt. Mal wieder geguckt. Du?",
    )

    # Thumbs
    sid3 = fs.add_thumbs(
        situation="Markus fragte: siehst du mich?",
        pi_response="Klar seh ich dich. Du stehst da wie 'n kleiner Pinguin im Morgenmantel.",
        label="up",
    )
    sid4 = fs.add_thumbs(
        situation="Markus fragte: erzaehl was witziges",
        pi_response="Ich kann Ihnen einen Witz erzaehlen.",
        label="down",
    )
    assert all([sid1, sid2, sid3, sid4])

    state = fs.get_state()
    print(f"\n[After 4 samples] {state}")
    assert state["critic"] == 2
    assert state["thumbs_up"] == 1
    assert state["thumbs_down"] == 1
    assert state["pending_review"] == 2
    assert state["approved"] == 1   # nur thumbs_up
    assert state["rejected"] == 1   # thumbs_down setzt approved=False -> zaehlt als rejected
    print(f"  thumbs_down → approved=False → rejected count = {state['rejected']} ✓")

    # Approve einen pending
    assert fs.approve(sid1, by="markus_test") is True
    state2 = fs.get_state()
    print(f"\n[After approve {sid1}] {state2}")
    assert state2["approved"] == state["approved"] + 1
    assert state2["pending_review"] == state["pending_review"] - 1

    # Reject einen pending
    assert fs.reject(sid2, by="markus_test") is True
    state3 = fs.get_state()
    print(f"\n[After reject {sid2}] {state3}")
    assert state3["pending_review"] == 0

    # Read approved (fuer LoRA)
    approved = fs.read_approved()
    print(f"\n[Approved fuer LoRA] {len(approved)} Samples")
    for s in approved:
        print(f"  {s['sample_id']} ({s['source']}): {s['pi_response'][:50]}...")

    # Validation
    assert fs.add_critic_sample("", "x", 5) is None
    assert fs.add_thumbs("x", "y", "invalid_label") is None

    # Re-approve nicht-existent
    assert fs.approve("smp_99999999") is False

    print(f"\nSelf-Test PASS — Pool: {POOL_PATH}")
