#!/usr/bin/env python3
"""
M.O.L.O.C.H. Night Cycle — Naechtliche Tagesverarbeitung
==========================================================

Nach 23:00 Uhr startet die Tagesverarbeitung:
  1. Episodic Memory lesen → Tages-Zusammenfassung
  2. Music Memory Scores updaten (Recency-Decay)
  3. Tages-Statistiken berechnen und speichern

Laeuft als Background-Thread, schlaeft tagsueber.
Fuehrt die Verarbeitung EINMAL pro Tag aus (nach 23:00).

Publiziert night_cycle_complete Event (Priority 9) wenn fertig.

Singleton: get_night_cycle()
Gate 5: Autonomous Environmental Agent
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

logger = logging.getLogger("MolochNightCycle")

# Nachtzyklus-Fenster
NIGHT_START_HOUR = 23
NIGHT_END_HOUR = 6

# Verarbeitungs-Ergebnisse
RESULTS_PATH = "/mnt/moloch-data/memory/night_cycle"


class NightCycle:
    """Naechtliche Tagesverarbeitung als Background-Thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_run_date: Optional[str] = None  # "YYYY-MM-DD"
        self._last_result: Optional[Dict[str, Any]] = None
        self._load_state()

    def _load_state(self):
        """Letzten Lauf-Status laden (persistent)."""
        state_path = os.path.join(RESULTS_PATH, "night_cycle_state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path) as f:
                    state = json.load(f)
                self._last_run_date = state.get("last_run_date")
                logger.info(f"[NIGHT] Letzter Lauf: {self._last_run_date}")
            except Exception as e:
                logger.debug(f"[NIGHT] State laden: {e}")

    def _save_state(self):
        """Lauf-Status persistent speichern."""
        os.makedirs(RESULTS_PATH, exist_ok=True)
        state_path = os.path.join(RESULTS_PATH, "night_cycle_state.json")
        try:
            with open(state_path, "w") as f:
                json.dump({
                    "last_run_date": self._last_run_date,
                    "timestamp": time.time(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"[NIGHT] State speichern: {e}")

    def start(self):
        """Night Cycle Thread starten."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._cycle_loop, daemon=True, name="NightCycle"
        )
        self._thread.start()
        logger.info("[NIGHT] Night Cycle Thread gestartet")

    def stop(self):
        """Night Cycle Thread stoppen."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _cycle_loop(self):
        """Hauptschleife: Prueft jede Minute ob Nachtzyklus faellig."""
        while self._running:
            try:
                now = datetime.now()
                today = now.strftime("%Y-%m-%d")

                # Nur zwischen 23:00 und 06:00
                if NIGHT_START_HOUR <= now.hour or now.hour < NIGHT_END_HOUR:
                    # Schon heute gelaufen?
                    if self._last_run_date != today:
                        logger.info(f"[NIGHT] Starte Tagesverarbeitung fuer {today}")
                        self._run_cycle(today)
            except Exception as e:
                logger.error(f"[NIGHT] Cycle-Loop Fehler: {e}")

            # 60s warten — nicht oefter pruefen noetig
            time.sleep(60)

    def _run_cycle(self, date: str):
        """Kompletten Nachtzyklus ausfuehren."""
        result = {
            "date": date,
            "start_time": time.time(),
            "steps": {},
        }

        # Schritt 1: Episodic Memory — Tages-Zusammenfassung
        try:
            summary = self._summarize_episodes(date)
            result["steps"]["episodes"] = summary
            logger.info(f"[NIGHT] Episoden: {summary.get('count', 0)} Eintraege")
        except Exception as e:
            result["steps"]["episodes"] = {"error": str(e)}
            logger.error(f"[NIGHT] Episoden-Zusammenfassung: {e}")

        # Schritt 2: Music Memory — Score-Decay
        try:
            music = self._decay_music_scores()
            result["steps"]["music"] = music
            logger.info(f"[NIGHT] Musik: {music.get('processed', 0)} Assoziationen")
        except Exception as e:
            result["steps"]["music"] = {"error": str(e)}
            logger.error(f"[NIGHT] Music Score Decay: {e}")

        # Schritt 3: Tages-Statistiken
        try:
            stats = self._compute_daily_stats(date)
            result["steps"]["stats"] = stats
            logger.info(f"[NIGHT] Stats berechnet")
        except Exception as e:
            result["steps"]["stats"] = {"error": str(e)}
            logger.error(f"[NIGHT] Tages-Statistiken: {e}")

        # Schritt 4: LLM-Reflexion via Qwen2.5-1.5B auf NPU (nur wenn verfuegbar)
        try:
            from core.autonomy.local_llm_bridge import get_llm_bridge
            bridge = get_llm_bridge()
            if bridge and bridge._ollama_available:
                reflection = self._generate_llm_reflection(date, result)
                result["steps"]["llm_reflection"] = reflection
                logger.info(f"[NIGHT] LLM-Reflexion: {len(reflection.get('text', ''))} Zeichen")
            else:
                result["steps"]["llm_reflection"] = {"skipped": "hailo-ollama nicht verfuegbar"}
        except Exception as e:
            result["steps"]["llm_reflection"] = {"error": str(e)}
            logger.error(f"[NIGHT] LLM-Reflexion: {e}")

        # Ergebnis speichern
        result["end_time"] = time.time()
        result["duration_s"] = round(result["end_time"] - result["start_time"], 1)

        with self._lock:
            self._last_run_date = date
            self._last_result = result
        self._save_state()
        self._save_result(date, result)

        # Event publizieren
        try:
            from core.moloch_event_bus import get_event_bus
            get_event_bus().publish(
                event_type="night_cycle_complete",
                source="night_cycle",
                priority=9,
                payload={"date": date, "duration_s": result["duration_s"]},
            )
        except Exception:
            pass

        logger.info(f"[NIGHT] Tagesverarbeitung abgeschlossen ({result['duration_s']}s)")

    def _generate_llm_reflection(self, date: str, cycle_result: Dict) -> Dict[str, Any]:
        """Tages-Reflexion via Qwen2.5-1.5B auf NPU generieren und speichern."""
        from core.autonomy.local_llm_bridge import get_llm_bridge
        bridge = get_llm_bridge()

        # Kontext aus Tages-Ergebnissen zusammenbauen
        episodes = cycle_result.get("steps", {}).get("episodes", {})
        stats = cycle_result.get("steps", {}).get("stats", {})
        events = stats.get("event_counts", {})

        prompt_lines = [f"Datum: {date}"]
        if episodes.get("count"):
            prompt_lines.append(f"Episoden heute: {episodes['count']}")
        if events:
            top = sorted(events.items(), key=lambda x: x[1], reverse=True)[:5]
            prompt_lines.append("Haeufigste Ereignisse: " + ", ".join(f"{k}={v}" for k, v in top))

        # Tages-Introspections als Kontext einbauen
        try:
            import json as _json
            _refl_path = f"/mnt/moloch-data/memory/reflections/{date}.jsonl"
            if os.path.exists(_refl_path):
                _thoughts = []
                with open(_refl_path, "r") as f:
                    for line in f:
                        try:
                            _entry = _json.loads(line.strip())
                            if _entry.get("thought"):
                                _thoughts.append(_entry["thought"][:80])
                        except Exception:
                            pass
                if _thoughts:
                    prompt_lines.append(f"Meine Gedanken heute ({len(_thoughts)}x):")
                    for t in _thoughts[-5:]:  # Letzte 5 Gedanken
                        prompt_lines.append(f"  - {t}")
        except Exception:
            pass

        prompt = "\n".join(prompt_lines) + "\n\nFasse den Tag in 2-3 Saetzen zusammen."
        system = (
            "Du bist M.O.L.O.C.H., ein autonomes KI-System. "
            "Reflektiere knapp ueber den vergangenen Tag. Deutsch, praegnant."
        )

        text = bridge.generate(prompt=prompt, system=system, max_tokens=256, use_local=True)

        reflection = {
            "date": date,
            "text": text or "",
            "provider": bridge._last_provider,
        }

        # Auf SSD2 speichern
        try:
            os.makedirs(RESULTS_PATH, exist_ok=True)
            path = os.path.join(RESULTS_PATH, f"reflection_{date}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text or "(keine Antwort)")
        except Exception as e:
            logger.warning(f"[NIGHT] Reflexion speichern: {e}")

        return reflection

    def _summarize_episodes(self, date: str) -> Dict[str, Any]:
        """Episodic Memory des Tages zusammenfassen."""
        try:
            from core.memory.episodic_memory import get_episodic_memory
            em = get_episodic_memory()
            stats = em.get_stats()

            if not stats.get("ready"):
                return {"count": 0, "note": "Qdrant nicht verfuegbar"}

            # Qdrant hat keinen Date-Range Filter fuer embedded mode
            # → Stats als Naherung verwenden
            return {
                "count": stats.get("points", 0),
                "collection": stats.get("collection", "?"),
                "note": "Gesamt-Episoden (kein Date-Filter im embedded Modus)",
            }
        except ImportError:
            return {"count": 0, "note": "episodic_memory nicht importierbar"}

    def _decay_music_scores(self) -> Dict[str, Any]:
        """Music Memory Assoziationen: Alte Eintraege abwerten."""
        try:
            from core.music.music_memory import get_music_memory, MEMORY_PATH
            mm = get_music_memory()
            stats = mm.get_stats()

            # Keine direkte Manipulation — MusicMemory Scores sind implizit
            # durch Recency im suggest_track() gewichtet.
            # Hier nur alte Eintraege (>30 Tage) entfernen um Speicher zu sparen.
            if not os.path.exists(MEMORY_PATH):
                return {"processed": 0, "removed": 0}

            with open(MEMORY_PATH) as f:
                associations = json.load(f)

            cutoff = time.time() - (30 * 86400)  # 30 Tage
            original_count = len(associations)
            associations = [a for a in associations if a.get("timestamp", 0) > cutoff]
            removed = original_count - len(associations)

            if removed > 0:
                os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
                with open(MEMORY_PATH, "w") as f:
                    json.dump(associations, f, indent=2, ensure_ascii=False)
                logger.info(f"[NIGHT] Music Memory: {removed} alte Eintraege entfernt")

            return {
                "processed": original_count,
                "removed": removed,
                "remaining": len(associations),
            }
        except Exception as e:
            return {"error": str(e)}

    def _compute_daily_stats(self, date: str) -> Dict[str, Any]:
        """Tages-Statistiken aus Event-Logs berechnen."""
        event_log_dir = os.path.expanduser("~/moloch/logs/events")
        log_file = os.path.join(event_log_dir, f"events_{date}.jsonl")

        if not os.path.exists(log_file):
            return {"events": 0, "note": "Kein Event-Log fuer heute"}

        event_counts: Dict[str, int] = {}
        total = 0
        try:
            with open(log_file) as f:
                for line in f:
                    try:
                        evt = json.loads(line.strip())
                        et = evt.get("event_type", "unknown")
                        event_counts[et] = event_counts.get(et, 0) + 1
                        total += 1
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            return {"error": str(e)}

        return {
            "events": total,
            "by_type": event_counts,
        }

    def _save_result(self, date: str, result: Dict[str, Any]):
        """Ergebnis persistent auf SSD2 speichern."""
        os.makedirs(RESULTS_PATH, exist_ok=True)
        filepath = os.path.join(RESULTS_PATH, f"night_{date}.json")
        try:
            with open(filepath, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"[NIGHT] Ergebnis speichern: {e}")

    # =====================================================================
    # Public API
    # =====================================================================

    def get_state(self) -> Dict[str, Any]:
        """Aktueller State fuer IPC/Panel."""
        with self._lock:
            return {
                "last_run_date": self._last_run_date,
                "running": self._running,
                "last_result_summary": {
                    "duration_s": self._last_result.get("duration_s") if self._last_result else None,
                    "steps": list(self._last_result.get("steps", {}).keys()) if self._last_result else [],
                } if self._last_result else None,
            }


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[NightCycle] = None
_instance_lock = threading.Lock()


def get_night_cycle() -> NightCycle:
    """Singleton-Zugriff auf Night Cycle."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = NightCycle()
    return _instance
