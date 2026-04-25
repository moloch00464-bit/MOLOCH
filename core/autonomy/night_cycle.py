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

                # processing_date = der Tag der verarbeitet werden soll
                # 23:XX → heute (Tag der gerade endet)
                # 00:XX-05:XX → gestern (Tag der gerade endete)
                # 06:XX-22:XX → None (tagsueber nichts tun)
                if now.hour >= NIGHT_START_HOUR:
                    processing_date = now.strftime("%Y-%m-%d")
                elif now.hour < NIGHT_END_HOUR:
                    processing_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    processing_date = None

                if processing_date and self._last_run_date != processing_date:
                    logger.info(f"[NIGHT] Starte Tagesverarbeitung fuer {processing_date}")
                    self._run_cycle(processing_date)
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

        # Schritt 5: Character Distiller (Phase 4 Gate 1.5)
        # Liest character_journal/{date}.jsonl, bewertet Events mit LLM,
        # aktualisiert character_drift.json, feuert 'character_drift_updated'.
        try:
            from core.autonomy.character_distiller import get_distiller
            distill = get_distiller().run(date)
            result["steps"]["character_distill"] = {
                "event_count": distill.get("event_count", 0),
                "summary": (distill.get("summary") or "")[:200],
                "drift": distill.get("drift", {}),
                "duration_s": distill.get("duration_s", 0),
                "llm_provider": distill.get("llm_provider"),
            }
            logger.info(
                f"[NIGHT] Character Distill: {distill.get('event_count', 0)} Events, "
                f"drift={distill.get('drift', {})}"
            )
        except Exception as e:
            result["steps"]["character_distill"] = {"error": str(e)}
            logger.error(f"[NIGHT] Character Distill: {e}")

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
        stats = cycle_result.get("steps", {}).get("stats", {})
        events = stats.get("by_type", {})

        prompt_lines = [f"=== Tagesbericht {date} ==="]

        # Personen-Statistiken aus Sampling
        person_min = stats.get("person_minutes", 0)
        markus_min = stats.get("markus_minutes", 0)
        first_seen = stats.get("first_seen")
        last_seen = stats.get("last_seen")

        if person_min > 0:
            prompt_lines.append(f"Personen erkannt: {person_min} Minuten (Markus: {markus_min} Min)")
            if first_seen and last_seen:
                prompt_lines.append(f"Erste Erkennung: {first_seen}, Letzte: {last_seen}")
        else:
            prompt_lines.append("Keine Personen erkannt heute.")

        # Aktivitaeten
        activities = stats.get("activities", {})
        if activities:
            act_str = ", ".join(f"{k} ({v}x)" for k, v in
                                sorted(activities.items(), key=lambda x: x[1], reverse=True))
            prompt_lines.append(f"Aktivitaeten: {act_str}")

        # Interessante Events (ohne context_update — redundant)
        interesting = {k: v for k, v in events.items() if k != "context_update"}
        if interesting:
            top = sorted(interesting.items(), key=lambda x: x[1], reverse=True)[:5]
            prompt_lines.append("Ereignisse: " + ", ".join(f"{k}={v}" for k, v in top))

        # Music Stats
        music = cycle_result.get("steps", {}).get("music", {})
        if music.get("processed", 0) > 0:
            remaining = music.get("remaining", 0)
            removed = music.get("removed", 0)
            prompt_lines.append(f"Musik-Assoziationen: {remaining}" +
                                (f" ({removed} entfernt)" if removed else ""))

        # Tages-Introspektionen als Kontext einbauen
        try:
            _refl_path = f"/mnt/moloch-data/memory/reflections/{date}.jsonl"
            if os.path.exists(_refl_path):
                _thoughts = []
                with open(_refl_path, "r") as f:
                    for line in f:
                        try:
                            _entry = json.loads(line.strip())
                            if _entry.get("thought"):
                                _thoughts.append(_entry["thought"][:100])
                        except Exception:
                            pass
                if _thoughts:
                    # Max 8 Gedanken, gleichmaessig verteilt
                    if len(_thoughts) > 8:
                        step = len(_thoughts) // 8
                        _thoughts = _thoughts[::step][:8]
                    prompt_lines.append(f"Meine Gedanken ({len(_thoughts)}x):")
                    for t in _thoughts:
                        prompt_lines.append(f"  - {t}")
        except Exception:
            pass

        prompt = "\n".join(prompt_lines)
        prompt += "\n\nReflektiere NUR basierend auf diesen Daten. Erfinde nichts."
        system = (
            "Du bist M.O.L.O.C.H., ein KI-System auf einem Raspberry Pi. "
            "Du beobachtest einen Raum mit einer PTZ-Kamera. "
            "Reflektiere knapp ueber den vergangenen Tag. "
            "Beziehe dich AUSSCHLIESSLICH auf die gegebenen Daten. "
            "Erfinde KEINE Fakten. Deutsch, 2-3 Saetze."
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
        """Tages-Statistiken aus Event-Logs berechnen (mit Sampling)."""
        event_log_dir = os.path.expanduser("~/moloch/logs/events")
        log_file = os.path.join(event_log_dir, f"events_{date}.jsonl")

        if not os.path.exists(log_file):
            return {"events": 0, "note": "Kein Event-Log fuer heute"}

        SAMPLE_STRIDE = 500  # Jede 500. Zeile parsen (Pi-schonend)
        by_type: Dict[str, int] = {}
        person_samples = 0
        markus_samples = 0
        activity_counts: Dict[str, int] = {}
        first_person_ts: Optional[float] = None
        last_person_ts: Optional[float] = None
        sampled = 0
        line_count = 0

        try:
            with open(log_file) as f:
                for line_num, line in enumerate(f):
                    line_count = line_num + 1
                    if line_num % SAMPLE_STRIDE != 0:
                        continue
                    try:
                        evt = json.loads(line.strip())
                    except (json.JSONDecodeError, ValueError):
                        continue

                    et = evt.get("event_type", "unknown")
                    by_type[et] = by_type.get(et, 0) + 1

                    if et == "context_update":
                        sampled += 1
                        p = evt.get("payload", evt)
                        ts = p.get("timestamp", 0)
                        pc = p.get("person_count", 0)
                        fid = p.get("face_id")
                        act = p.get("activity", "unknown")

                        activity_counts[act] = activity_counts.get(act, 0) + 1

                        if pc > 0:
                            person_samples += 1
                            if first_person_ts is None:
                                first_person_ts = ts
                            last_person_ts = ts

                        if fid == "markus":
                            markus_samples += 1
        except Exception as e:
            return {"error": str(e)}

        # Hochrechnung: ~20 Events/Sek → 500 Events ≈ 25 Sekunden
        minutes_per_sample = SAMPLE_STRIDE / 20.0 / 60.0

        result: Dict[str, Any] = {
            "events": line_count,
            "by_type": by_type,
            "sampled": sampled,
        }

        if sampled > 0:
            result["person_minutes"] = round(person_samples * minutes_per_sample)
            result["markus_minutes"] = round(markus_samples * minutes_per_sample)
            result["activities"] = activity_counts
            if first_person_ts:
                result["first_seen"] = datetime.fromtimestamp(first_person_ts).strftime("%H:%M")
            if last_person_ts:
                result["last_seen"] = datetime.fromtimestamp(last_person_ts).strftime("%H:%M")

        return result

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
