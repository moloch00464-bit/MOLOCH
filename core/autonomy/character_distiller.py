#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Character Distiller
==================================

Naechtlicher Verarbeiter des Character Journals (Phase 4 von Gate 1.5).

Liest Journal/{date}.jsonl, ruft LLM (Tentakel oder Qwen lokal) zur Bewertung
jedes Events auf, berechnet Recency-Decay (Half-Life 7d), aggregiert eine
30-Tage Drift, schreibt ein kumulatives Drift-Profil und feuert ein
EventBus-Event 'character_drift_updated' an die laufende PersonalityEngine.

Storage:
  /mnt/moloch-data/memory/distill/{date}.json     - pro Tag
  /mnt/moloch-data/memory/character_drift.json    - kumulativ rolling 30 Tage

Singleton: get_distiller()

API:
  distiller.run(date)              -> Dict (distillates)
  distiller.force_distill_today()  -> Dict (manueller Trigger fuer MCP/Test)
  distiller.get_drift()            -> Dict (kumulatives Drift-Profil)
"""

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("CharacterDistiller")

JOURNAL_DIR = "/mnt/moloch-data/memory/journal"
DISTILL_DIR = "/mnt/moloch-data/memory/distill"
DRIFT_PATH = "/mnt/moloch-data/memory/character_drift.json"

HALF_LIFE_DAYS = 7.0
DRIFT_WINDOW_DAYS = 30
MAX_EVENTS_PER_PROMPT = 120  # haerte Grenze fuer LLM-Kontext
LLM_MAX_TOKENS = 2048
LLM_TIMEOUT_S = 120.0  # Naechtlicher Lauf — generos


def _utc_iso_ms() -> str:
    """UTC ISO-Zeitstempel mit Millisekunden + Z."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_write_json(path: str, data: Any) -> None:
    """Atomares JSON-Write mit NTFS-Fallback (Pattern aus core/longterm_memory.py)."""
    tmp_path = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
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
        logger.error(f"[DISTILL] Schreiben fehlgeschlagen ({path}): {e}")
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _safe_read_json(path: str, default: Any = None) -> Any:
    """JSON lesen mit Default-Fallback."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[DISTILL] Lesen fehlgeschlagen ({path}): {e}")
        return default


def _read_journal_day(date: str) -> List[Dict]:
    """Alle Eintraege eines Journal-Tages als Liste laden."""
    path = os.path.join(JOURNAL_DIR, f"{date}.jsonl")
    if not os.path.exists(path):
        return []
    entries: List[Dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"[DISTILL] Journal-Read fehlgeschlagen ({path}): {e}")
    return entries


def _recency_for_date(date_str: str, today: Optional[datetime] = None) -> float:
    """Half-Life 7 Tage Decay: recency = 0.5 ** (days_old / 7)."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return 0.0
    today = today or datetime.now()
    days_old = max(0.0, (today - d).total_seconds() / 86400.0)
    return 0.5 ** (days_old / HALF_LIFE_DAYS)


def _heuristic_fallback(events: List[Dict]) -> Dict[str, Any]:
    """Wenn LLM-Output unparsbar: deterministische Heuristik."""
    type_weights = {
        "tension": 1.0, "protective": 0.9, "mode_switch": 0.7,
        "chat": 0.5, "camera": 0.4, "audio": 0.3, "spotify": 0.2,
    }
    drift = {"mood_shift": 0.0, "energy_shift": 0.0, "dominance_shift": 0.0}
    enriched: Dict[str, Dict] = {}
    n = max(1, len(events))
    for e in events:
        td = float(e.get("tension_delta", 0.0) or 0.0)
        importance = min(1.0, abs(td) + type_weights.get(e.get("type"), 0.3) * 0.3)
        relevance = 0.5  # neutral default
        citation = f"{e.get('type', '?')}: {(e.get('interpretation') or '')[:60]}"
        enriched[e.get("event_id", "?")] = {
            "importance": round(importance, 3),
            "relevance": relevance,
            "citation": citation,
        }
        drift["mood_shift"] += td * -0.05  # negative tension = positive mood
    # Normieren auf [-1, 1]
    for k in drift:
        drift[k] = round(max(-1.0, min(1.0, drift[k] / max(1, n / 10))), 3)
    return {
        "summary": f"(heuristic fallback, {len(events)} Events)",
        "drift": drift,
        "events": enriched,
    }


def _sample_events_for_prompt(events: List[Dict], limit: int) -> List[Dict]:
    """Bei zu vielen Events: prioritaere nach |tension_delta|, dann stride-sample Rest."""
    if len(events) <= limit:
        return events
    # Prio: alle mit |tension_delta| > 0
    prio = [e for e in events if abs(float(e.get("tension_delta", 0.0) or 0.0)) > 0.01]
    rest = [e for e in events if e not in prio]
    if len(prio) >= limit:
        return prio[:limit]
    # Rest mit stride-sample auffuellen
    needed = limit - len(prio)
    if not rest or needed <= 0:
        return prio
    stride = max(1, len(rest) // needed)
    sampled_rest = rest[::stride][:needed]
    # Chronologie wiederherstellen
    combined = prio + sampled_rest
    combined.sort(key=lambda e: e.get("ts", ""))
    return combined


def _build_llm_prompt(date: str, events: List[Dict]) -> tuple:
    """LLM System + User Prompt fuer Distill."""
    system = (
        "Du bist M.O.L.O.C.H. Distiller. Du wertest Charakter-formende Events eines Tages aus.\n"
        "AUFGABE: Bewerte JEDEN Event nach importance (0.0-1.0) + relevance (0.0-1.0).\n"
        "Schreibe pro Event eine kurze citation (max 120 Zeichen, Deutsch).\n"
        "Berechne einen Tages-Drift-Vector: mood_shift, energy_shift, dominance_shift (jeweils -1.0..+1.0).\n"
        "Schreibe eine knappe Tages-Summary (1-2 Saetze, Deutsch).\n"
        "ANTWORTE AUSSCHLIESSLICH ALS GUELTIGES JSON, KEIN PROSA-PREAMBLE."
    )
    # Kompakte Event-Darstellung (Token-sparsam)
    event_lines = []
    for e in events:
        eid = e.get("event_id", "?")
        t = e.get("type", "?")
        td = e.get("tension_delta", 0.0)
        interp = (e.get("interpretation") or "")[:80]
        tags = ",".join(e.get("tags") or [])
        event_lines.append(f"{eid}|{t}|td={td:+.2f}|{interp}|tags={tags}")
    events_block = "\n".join(event_lines)
    user = (
        f"Tag: {date}\n"
        f"Events ({len(events)}):\n{events_block}\n\n"
        "JSON-Schema:\n"
        "{\n"
        '  "summary": "1-2 Saetze",\n'
        '  "drift": {"mood_shift": 0.0, "energy_shift": 0.0, "dominance_shift": 0.0},\n'
        '  "events": {\n'
        '    "evt_NNNNNNNN": {"importance": 0.0, "relevance": 0.0, "citation": "..."},\n'
        "    ...\n"
        "  }\n"
        "}\n"
    )
    return system, user


def _extract_json(text: str) -> Optional[Dict]:
    """Robuster JSON-Parse: extrahiert {...} aus LLM-Output, auch wenn Prosa drumherum."""
    if not text:
        return None
    # Suche groesstes {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Aufraeumen: trailing commas etc.
        raw_clean = re.sub(r",\s*([\}\]])", r"\1", raw)
        try:
            return json.loads(raw_clean)
        except json.JSONDecodeError:
            return None


def _validate_distill_dict(data: Dict) -> bool:
    """Pruefen ob LLM-Output Schema entspricht (toleranz)."""
    if not isinstance(data, dict):
        return False
    if "drift" not in data or not isinstance(data["drift"], dict):
        return False
    if "events" not in data or not isinstance(data["events"], dict):
        return False
    return True


# =============================================================================
# CharacterDistiller
# =============================================================================

class CharacterDistiller:
    """Liest Journal -> LLM bewertet -> Drift-Profil + EventBus-Update."""

    def __init__(self):
        self._lock = threading.Lock()
        os.makedirs(DISTILL_DIR, exist_ok=True)
        logger.info(f"[DISTILL] Initialisiert: distill={DISTILL_DIR}")

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def run(self, date: str) -> Dict[str, Any]:
        """Distillation fuer ein Datum (YYYY-MM-DD).

        Returns dict mit allen Schritt-Ergebnissen + drift.
        """
        t0 = time.monotonic()
        with self._lock:
            return self._run_inner(date, t0)

    def force_distill_today(self) -> Dict[str, Any]:
        """Manueller Trigger heute (fuer Test/MCP)."""
        return self.run(datetime.now().strftime("%Y-%m-%d"))

    def get_drift(self) -> Dict[str, Any]:
        """Aktuelles kumulatives Drift-Profil zurueckgeben (oder leer wenn noch nichts)."""
        return _safe_read_json(DRIFT_PATH, default={
            "updated_at": None,
            "window_days": DRIFT_WINDOW_DAYS,
            "rolling_drift": {"mood_baseline": 0.0, "energy_baseline": 0.0, "dominance_baseline": 0.0},
            "recency_weighted_top": [],
            "daily_distillates": [],
        })

    # =========================================================================
    # CORE WORKFLOW
    # =========================================================================

    def _run_inner(self, date: str, t0: float) -> Dict[str, Any]:
        events = _read_journal_day(date)
        if not events:
            logger.info(f"[DISTILL] Kein Journal fuer {date}, ueberspringe")
            return {"date": date, "skipped": "kein Journal", "event_count": 0}

        sampled = _sample_events_for_prompt(events, MAX_EVENTS_PER_PROMPT)
        logger.info(f"[DISTILL] {date}: {len(events)} Events, sample={len(sampled)}")

        # LLM-Aufruf
        llm_result, llm_provider = self._call_llm(date, sampled)

        # Anreichern: alle Events bekommen entries (auch nicht im Sample)
        full_events_dict = {}
        sampled_ids = {e.get("event_id") for e in sampled}
        sampled_enriched = llm_result.get("events", {}) if llm_result else {}
        # Fuer Events im Sample: LLM-Werte nehmen (oder heuristic falls fehlt)
        # Fuer Events nicht im Sample: heuristic
        heuristic_full = _heuristic_fallback(events)["events"]
        for e in events:
            eid = e.get("event_id", "?")
            if eid in sampled_ids and eid in sampled_enriched:
                full_events_dict[eid] = sampled_enriched[eid]
            else:
                full_events_dict[eid] = heuristic_full.get(eid, {
                    "importance": 0.3, "relevance": 0.5, "citation": ""
                })

        distill_dict = {
            "date": date,
            "generated_at": _utc_iso_ms(),
            "duration_s": round(time.monotonic() - t0, 1),
            "llm_provider": llm_provider,
            "event_count": len(events),
            "sampled_to_llm": len(sampled),
            "summary": (llm_result or {}).get("summary", "(keine Summary)"),
            "drift": (llm_result or {}).get("drift", {
                "mood_shift": 0.0, "energy_shift": 0.0, "dominance_shift": 0.0
            }),
            "events": full_events_dict,
        }

        # Tagesdatei schreiben
        day_path = os.path.join(DISTILL_DIR, f"{date}.json")
        _safe_write_json(day_path, distill_dict)
        logger.info(f"[DISTILL] {date} geschrieben: drift={distill_dict['drift']}")

        # character_drift.json updaten
        self._update_rolling_drift()

        # EventBus
        self._publish_drift_event(date, distill_dict["drift"])

        return distill_dict

    def _call_llm(self, date: str, events: List[Dict]) -> tuple:
        """LLM-Aufruf mit Robustheit. Returns (parsed_dict_or_None, provider_name)."""
        try:
            from core.autonomy.local_llm_bridge import get_llm_bridge
            bridge = get_llm_bridge()
        except Exception as e:
            logger.warning(f"[DISTILL] LLM-Bridge nicht verfuegbar: {e}")
            return _heuristic_fallback(events), "heuristic"

        system, user = _build_llm_prompt(date, events)

        # Versuch 1: Tentakel (Mistral 7B, besser fuer JSON)
        text = None
        provider = "unknown"
        try:
            text = bridge.generate(
                prompt=user, system=system,
                max_tokens=LLM_MAX_TOKENS, use_local=False,
            )
            provider = getattr(bridge, "_last_provider", "tentacle")
        except Exception as e:
            logger.warning(f"[DISTILL] Tentakel-Aufruf fehlgeschlagen: {e}")

        # Versuch 2: Qwen lokal (Fallback)
        if not text:
            try:
                text = bridge.generate(
                    prompt=user, system=system,
                    max_tokens=LLM_MAX_TOKENS, use_local=True,
                )
                provider = getattr(bridge, "_last_provider", "qwen_local")
            except Exception as e:
                logger.warning(f"[DISTILL] Qwen-Fallback fehlgeschlagen: {e}")

        if not text:
            logger.warning("[DISTILL] Beide LLM-Pfade leer, Heuristik-Fallback")
            return _heuristic_fallback(events), "heuristic"

        parsed = _extract_json(text)
        if parsed and _validate_distill_dict(parsed):
            return parsed, provider

        logger.warning(f"[DISTILL] LLM-JSON unparsbar ({len(text or '')} Zeichen), Heuristik-Fallback")
        return _heuristic_fallback(events), f"{provider}_then_heuristic"

    def _update_rolling_drift(self) -> None:
        """character_drift.json aus letzten 30 distill/{date}.json zusammenbauen."""
        today = datetime.now()
        window: List[Dict] = []
        for days_back in range(DRIFT_WINDOW_DAYS):
            d = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
            path = os.path.join(DISTILL_DIR, f"{d}.json")
            data = _safe_read_json(path)
            if data and "drift" in data:
                window.append(data)

        if not window:
            logger.warning("[DISTILL] Keine Distillate im 30d-Fenster")
            return

        # Rolling Drift: gewichtetes Mittel aller drift-Werte mit recency-Gewicht
        sum_w = 0.0
        agg = {"mood_baseline": 0.0, "energy_baseline": 0.0, "dominance_baseline": 0.0}
        for d in window:
            w = _recency_for_date(d.get("date", ""), today)
            drift = d.get("drift", {})
            agg["mood_baseline"] += drift.get("mood_shift", 0.0) * w
            agg["energy_baseline"] += drift.get("energy_shift", 0.0) * w
            agg["dominance_baseline"] += drift.get("dominance_shift", 0.0) * w
            sum_w += w
        if sum_w > 0:
            for k in agg:
                agg[k] = round(agg[k] / sum_w, 4)

        # Top-Events nach recency * importance
        top_candidates: List[Dict] = []
        for d in window:
            w = _recency_for_date(d.get("date", ""), today)
            for eid, ev_data in (d.get("events") or {}).items():
                imp = float(ev_data.get("importance", 0.0) or 0.0)
                if imp <= 0:
                    continue
                top_candidates.append({
                    "event_id": eid,
                    "weight": round(w * imp, 4),
                    "citation": ev_data.get("citation", ""),
                    "date": d.get("date", ""),
                })
        top_candidates.sort(key=lambda x: x["weight"], reverse=True)
        top10 = top_candidates[:10]

        # Daily-Distillate-Liste (kompakt)
        daily = [{
            "date": d.get("date", ""),
            "drift": d.get("drift", {}),
            "summary": (d.get("summary") or "")[:160],
            "event_count": d.get("event_count", 0),
        } for d in sorted(window, key=lambda x: x.get("date", ""))]

        drift_doc = {
            "updated_at": _utc_iso_ms(),
            "window_days": DRIFT_WINDOW_DAYS,
            "rolling_drift": agg,
            "recency_weighted_top": top10,
            "daily_distillates": daily,
        }
        _safe_write_json(DRIFT_PATH, drift_doc)
        logger.info(f"[DISTILL] character_drift.json aktualisiert: {agg}")

    def _publish_drift_event(self, date: str, drift: Dict) -> None:
        """EventBus-Notification an PersonalityEngine."""
        try:
            from core.moloch_event_bus import get_event_bus, PRIO_INFO
            get_event_bus().publish(
                event_type="character_drift_updated",
                source="character_distiller",
                priority=PRIO_INFO,
                payload={"date": date, "drift": drift},
            )
            logger.info(f"[DISTILL] character_drift_updated event published")
        except Exception as e:
            logger.debug(f"[DISTILL] Event-Publish: {e}")


# =============================================================================
# Singleton
# =============================================================================

_instance: Optional[CharacterDistiller] = None
_instance_lock = threading.Lock()


def get_distiller() -> CharacterDistiller:
    """Globale CharacterDistiller-Instanz (Singleton)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = CharacterDistiller()
    return _instance


# =============================================================================
# Self-Test — `python3 -m core.autonomy.character_distiller`
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    d = get_distiller()
    today = datetime.now().strftime("%Y-%m-%d")

    # Recency-Decay sanity
    print(f"\n[Recency-Decay Check]")
    for days in [0, 7, 14, 30]:
        d_str = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        r = _recency_for_date(d_str)
        print(f"  {days:3d} Tage alt: recency={r:.4f}")

    # Force run today
    print(f"\n[Distillation {today}]")
    result = d.force_distill_today()
    print(f"  event_count = {result.get('event_count')}")
    print(f"  llm_provider = {result.get('llm_provider')}")
    print(f"  duration_s = {result.get('duration_s')}")
    print(f"  drift = {result.get('drift')}")
    print(f"  summary = {(result.get('summary') or '')[:160]}")

    # Drift-Datei pruefen
    print(f"\n[Drift-Profil]")
    drift = d.get_drift()
    print(f"  rolling = {drift.get('rolling_drift')}")
    print(f"  daily count = {len(drift.get('daily_distillates', []))}")
    print(f"  top events = {len(drift.get('recency_weighted_top', []))}")

    # Schema-Check
    assert os.path.exists(os.path.join(DISTILL_DIR, f"{today}.json")), "distill/{today}.json fehlt"
    assert os.path.exists(DRIFT_PATH), "character_drift.json fehlt"
    assert "rolling_drift" in drift, "Schema falsch"
    print("\nSelf-Test PASS")
