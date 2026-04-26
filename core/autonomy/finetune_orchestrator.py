#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Finetune Orchestrator

Welle 3 / W3.1 von ThreeBrain FineTune Loop.

Critic-Actor-Loop:
  1) Seed-Events aus character_drift.recency_weighted_top holen (Top-Erlebnisse)
  2) Pro Seed: Critic generiert plausible neue Markus-Situation (PC-Ollama)
  3) Pi-Ghost (qwen2.5:1.5b lokal via local_llm_bridge force_local) antwortet
  4) Critic bewertet Pi-Antwort gegen Charakter-State, schlaegt bessere vor
  5) Sample wird im feedback_store als 'critic'-Sample gespeichert (pending)
  6) Markus reviewt via scripts/review_pending_rules.py --samples

Voraussetzungen (alle als best-effort gecheckt, kein crash bei Fehler):
  - PC-Ollama erreichbar (critic_client.health())
  - hailo-ollama lokal lebt (qwen fuer Pi-Antwort)
  - character_drift.json existiert (sonst kein Seed)

Trigger:
  - Manuell via run_session(max_samples=N) — z.B. von MCP / scripts
  - Spaeter (Welle 4): cron / session_modes wenn Markus weg + Ryzen on

Singleton: get_orchestrator()

CLI: python3 -m core.autonomy.finetune_orchestrator [--max N] [--dry]
"""

import argparse
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("FinetuneOrchestrator")

DEFAULT_MAX_SAMPLES = 5
LOCAL_LLM_TIMEOUT_S = 60.0


def _try_log_ledger(event: str, **meta) -> None:
    try:
        from core.memory.behavior_mutation_ledger import get_ledger
        get_ledger().log(event, **meta)
    except Exception as e:
        logger.debug(f"[ORCH] Ledger-Log Fehler: {e}")


def _gather_character_state() -> Dict[str, Any]:
    """Snapshot des aktuellen Charakter-State fuer Critic-Bewertung."""
    state: Dict[str, Any] = {"rolling_drift": {}, "active_rules": [], "zone": "guardian"}
    try:
        from core.autonomy.character_distiller import get_distiller
        d = get_distiller().get_drift() or {}
        state["rolling_drift"] = d.get("rolling_drift") or {}
    except Exception as e:
        logger.debug(f"[ORCH] drift fail: {e}")
    try:
        from core.memory.character_patch import get_patch
        state["active_rules"] = get_patch().get_active_rules() or []
    except Exception as e:
        logger.debug(f"[ORCH] patch fail: {e}")
    # Zone aus CoreIntegrator-Effects (Welle 3 Feature A1):
    # Critic + Pi-Ghost koennen nur dann zone-gerecht antworten, wenn die
    # aktuelle Zone bekannt ist. Default 'guardian' wenn Effects nicht ladbar.
    try:
        from core.core_integrator import get_core_integrator
        eff = get_core_integrator().get_effects() or {}
        zone = eff.get("zone")
        if isinstance(zone, str) and zone:
            state["zone"] = zone
        # Plus die effects-Zahlen — Critic kann sie auswerten
        state["effects"] = {
            "language_sharpness": round(float(eff.get("language_sharpness", 0.0)), 2),
            "voice_intensity": round(float(eff.get("voice_intensity", 0.0)), 2),
            "guardian_influence": round(float(eff.get("guardian_influence", 0.0)), 2),
            "shadow_influence": round(float(eff.get("shadow_influence", 0.0)), 2),
        }
    except Exception as e:
        logger.debug(f"[ORCH] zone/effects fail: {e}")
    return state


def _gather_seed_events(n: int = 8) -> List[Dict]:
    """Top-Erlebnisse aus character_drift als Seeds.

    Falls noch nichts destilliert: nimmt einfach die letzten Journal-Events.
    """
    seeds: List[Dict] = []
    try:
        from core.autonomy.character_distiller import get_distiller
        d = get_distiller().get_drift() or {}
        top = d.get("recency_weighted_top") or []
        for t in top[:n]:
            seeds.append({
                "event_id": t.get("event_id"),
                "type": "top",
                "interpretation": t.get("citation") or "",
                "weight": t.get("weight", 0),
                "tension_delta": 0.0,
            })
    except Exception as e:
        logger.debug(f"[ORCH] drift seeds fail: {e}")

    if not seeds:
        # Fallback: letzte Journal-Events
        try:
            from core.memory.character_journal import get_journal
            recent = get_journal().read_recent(n) or []
            for e in recent:
                if e.get("type") in ("tension", "protective", "mode_switch", "chat"):
                    seeds.append({
                        "event_id": e.get("event_id"),
                        "type": e.get("type"),
                        "interpretation": e.get("interpretation"),
                        "tension_delta": e.get("tension_delta", 0.0),
                    })
        except Exception as e:
            logger.debug(f"[ORCH] journal seeds fail: {e}")
    return seeds[:n]


_PI_GHOST_SYSTEM = (
    "Du bist Moloch, der Geist auf dem Pi. Antworte kurz, kumpelhaft, "
    "trocken-humorig, max 2 Saetze. Kein Hoeflichkeitsquatsch."
)


def _ask_pi_ghost(situation: str) -> Optional[str]:
    """Pi-Ghost (qwen2.5:1.5b lokal) auf Situation antworten lassen.

    Wichtig: ruft direkt _generate_ollama() — die public ask_external()
    respektiert llm_mode='cloud_only' und wuerde NPU skippen. Wir muessen
    aber gerade qwen ansprechen (Trainings-Sample-Generation).
    """
    try:
        from core.autonomy.local_llm_bridge import get_llm_bridge
        bridge = get_llm_bridge()
        if not getattr(bridge, "_ollama_available", False):
            logger.warning("[ORCH] hailo-ollama nicht verfuegbar")
            return None
        out = bridge._generate_ollama(
            prompt=situation,
            system=_PI_GHOST_SYSTEM,
            max_tokens=200,
            model="qwen2.5:1.5b",
            timeout=30,
            force_local=True,  # umgeht backoff
        )
        return out
    except Exception as e:
        logger.warning(f"[ORCH] Pi-Ghost-Antwort fehlgeschlagen: {e}")
        return None


# =============================================================================
# FinetuneOrchestrator
# =============================================================================

class FinetuneOrchestrator:
    """Critic-Actor-Loop fuer LoRA-Sample-Generation."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_run_at: Optional[float] = None
        self._last_run_stats: Dict[str, Any] = {}
        logger.info("[ORCH] Init")

    def preflight(self) -> Dict[str, bool]:
        """Pruefe ob alle benoetigten Komponenten verfuegbar sind."""
        checks: Dict[str, bool] = {}
        # Critic-Client (PC-Ollama)
        try:
            from core.bridge.critic_client import get_critic_client
            checks["critic_pc_ollama"] = get_critic_client().health(force=True)
        except Exception as e:
            logger.warning(f"[ORCH] critic check fail: {e}")
            checks["critic_pc_ollama"] = False
        # Lokale LLM-Bridge
        try:
            from core.autonomy.local_llm_bridge import get_llm_bridge
            b = get_llm_bridge()
            checks["pi_local_llm"] = bool(getattr(b, "_ollama_available", False))
        except Exception as e:
            logger.warning(f"[ORCH] local llm check fail: {e}")
            checks["pi_local_llm"] = False
        # FeedbackStore
        try:
            from core.memory.feedback_store import get_feedback_store
            get_feedback_store()  # only init test
            checks["feedback_store"] = True
        except Exception as e:
            logger.warning(f"[ORCH] feedback store fail: {e}")
            checks["feedback_store"] = False
        # Drift (kann leer sein, aber muss ladbar)
        try:
            from core.autonomy.character_distiller import get_distiller
            get_distiller().get_drift()
            checks["distiller"] = True
        except Exception as e:
            logger.warning(f"[ORCH] distiller fail: {e}")
            checks["distiller"] = False
        return checks

    def run_session(self, max_samples: int = DEFAULT_MAX_SAMPLES,
                    dry: bool = False) -> Dict[str, Any]:
        """Eine Trainings-Session: bis zu max_samples Samples erzeugen.

        Args:
            max_samples: Hoechstzahl pro Lauf (Pi-Schoner)
            dry: nur erzeugen + zeigen, nicht in feedback_store speichern

        Returns:
            stats dict mit produced/skipped/errors/duration_s
        """
        t0 = time.monotonic()
        with self._lock:
            return self._run_inner(max_samples, dry, t0)

    def _run_inner(self, max_samples: int, dry: bool, t0: float) -> Dict[str, Any]:
        stats = {
            "produced": 0, "skipped": 0, "errors": 0, "samples": [],
            "started_at": time.time(), "duration_s": 0.0,
        }

        pf = self.preflight()
        logger.info(f"[ORCH] preflight: {pf}")
        if not pf.get("critic_pc_ollama"):
            stats["aborted"] = "critic_pc_ollama_offline"
            stats["preflight"] = pf
            return stats
        if not pf.get("pi_local_llm"):
            stats["aborted"] = "pi_local_llm_offline"
            stats["preflight"] = pf
            return stats

        seeds = _gather_seed_events(n=max_samples * 2)  # 2x Buffer fuer Skip
        logger.info(f"[ORCH] {len(seeds)} Seeds gesammelt")
        if not seeds:
            stats["aborted"] = "no_seeds"
            return stats

        char_state = _gather_character_state()

        from core.bridge.critic_client import get_critic_client
        critic = get_critic_client()

        from core.memory.feedback_store import get_feedback_store
        store = get_feedback_store() if not dry else None

        _try_log_ledger("training_session_started", max_samples=max_samples, seeds=len(seeds))

        for seed in seeds:
            if stats["produced"] >= max_samples:
                break
            try:
                # Step 1: Situation aus Seed
                logger.info(f"[ORCH] Seed: {seed.get('event_id')} '{(seed.get('interpretation') or '')[:60]}'")
                sit = critic.generate_situation(seed_event=seed, character_state=char_state)
                situation = (sit.get("situation_text") or "").strip()
                if not situation:
                    logger.debug("[ORCH] leere Situation — skip")
                    stats["skipped"] += 1
                    continue

                # Step 2: Pi-Ghost antwortet (qwen lokal)
                pi_resp = _ask_pi_ghost(situation)
                if not pi_resp:
                    logger.debug("[ORCH] Pi-Ghost stille — skip")
                    stats["skipped"] += 1
                    continue

                # Step 3: Critic bewertet
                ev = critic.evaluate(
                    situation=situation, pi_response=pi_resp,
                    character_state=char_state,
                )
                score = ev.get("score", -1)
                if score < 0:
                    logger.debug("[ORCH] Critic-Eval fail — skip")
                    stats["skipped"] += 1
                    continue

                # Step 4: speichern
                sample = {
                    "situation": situation,
                    "pi_response": pi_resp,
                    "score": score,
                    "critique": ev.get("critique", ""),
                    "better_response": ev.get("better_response", ""),
                    "seed_event_id": seed.get("event_id"),
                }
                stats["samples"].append(sample)

                if not dry and store:
                    sid = store.add_critic_sample(
                        situation=situation,
                        pi_response=pi_resp,
                        score=score,
                        critique=ev.get("critique", ""),
                        better_response=ev.get("better_response", ""),
                        seed_event_id=seed.get("event_id"),
                    )
                    sample["sample_id"] = sid

                stats["produced"] += 1
                logger.info(
                    f"[ORCH] Sample {stats['produced']}/{max_samples} "
                    f"score={score} critique='{(ev.get('critique') or '')[:60]}'"
                )

            except Exception as e:
                logger.error(f"[ORCH] Sample-Gen Fehler: {e}")
                stats["errors"] += 1

        stats["duration_s"] = round(time.monotonic() - t0, 1)
        self._last_run_at = time.time()
        self._last_run_stats = stats

        _try_log_ledger("training_session_done", **{
            k: v for k, v in stats.items() if k != "samples"
        })
        logger.info(
            f"[ORCH] Session done — produced={stats['produced']} "
            f"skipped={stats['skipped']} errors={stats['errors']} "
            f"({stats['duration_s']}s)"
        )
        return stats

    def get_state(self) -> Dict[str, Any]:
        return {
            "last_run_at": self._last_run_at,
            "last_run_stats": self._last_run_stats,
        }


# =============================================================================
# Singleton
# =============================================================================

_instance: Optional[FinetuneOrchestrator] = None
_instance_lock = threading.Lock()


def get_orchestrator() -> FinetuneOrchestrator:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = FinetuneOrchestrator()
    return _instance


# =============================================================================
# CLI
# =============================================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser(description="FinetuneOrchestrator manuell ausfuehren")
    parser.add_argument("--max", type=int, default=DEFAULT_MAX_SAMPLES,
                        help=f"Max Samples pro Lauf (default {DEFAULT_MAX_SAMPLES})")
    parser.add_argument("--dry", action="store_true",
                        help="Nur erzeugen + zeigen, nicht in feedback_store schreiben")
    args = parser.parse_args()

    o = get_orchestrator()
    print(f"\n[Preflight] {o.preflight()}")
    print(f"\n[Run Session max={args.max} dry={args.dry}]")
    stats = o.run_session(max_samples=args.max, dry=args.dry)
    print(f"\n[Stats]")
    for k, v in stats.items():
        if k != "samples":
            print(f"  {k}: {v}")
    print(f"\n[Samples ({len(stats.get('samples', []))})]")
    for s in stats.get("samples", []):
        print(f"  --- {s.get('sample_id', '(dry)')} score={s['score']} ---")
        print(f"  Situation: {s['situation'][:120]}")
        print(f"  Pi-Antwort: {s['pi_response'][:120]}")
        print(f"  Kritik: {s['critique'][:120]}")
        print(f"  Besser: {s['better_response'][:120]}")
        print()
    return 0 if stats.get("produced", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
