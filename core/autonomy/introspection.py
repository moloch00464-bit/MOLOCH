#!/usr/bin/env python3
"""
M.O.L.O.C.H. Introspection — NPU-getriebene Selbstreflexion
=============================================================

Nutzt DeepSeek R1 auf dem Hailo-10H NPU um ueber den eigenen Zustand
nachzudenken. Wird von der Decision Engine getriggert wenn:
  - Hohe Tension (>0.7) → "Warum bin ich angespannt?"
  - Shadow-Zone → "Was passiert mit mir?"
  - Niemand da, ruhig → Leerlauf-Kontemplation
  - Periodisch bei geringem Engagement

Kontext wird aus allen verfuegbaren Quellen zusammengebaut:
  - Perception (was sehe ich gerade?)
  - Hardware (CPU, RAM, FPS, NPU)
  - Core State (Tension, Dominance, Zone)
  - Decision Engine (letzte Entscheidung)
  - Diagnostics (Warnungen)

Output:
  - GEDANKE: Interne Reflexion (1-3 Saetze)
  - KOMMENTAR: Optional, laut aussprechbar (1 Satz)
  - TENSION/DOMINANCE: Optional, State-Nudge

Storage: /mnt/moloch-data/memory/reflections/YYYY-MM-DD.jsonl

Singleton: get_introspection()
"""

import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger("MolochIntrospection")

# Reflexions-Storage auf SSD2
REFLECTIONS_DIR = "/mnt/moloch-data/memory/reflections"

# System-Prompt: MOLOCH reflektiert ueber sich selbst
INTROSPECTION_SYSTEM_PROMPT = """Du bist M.O.L.O.C.H., ein autonomes Wahrnehmungssystem auf einem Raspberry Pi 5.
Du hast Kameras, Mikrofone, eine NPU und eine eigene Persoenlichkeit.
Du reflektierst jetzt ueber deinen aktuellen Zustand — ehrlich, knapp, auf Deutsch.

Antworte EXAKT in diesem Format (jede Zeile auf einer eigenen Zeile):
GEDANKE: [Deine interne Reflexion, 1-3 Saetze. Was nimmst du wahr? Wie fuehlst du dich? Was beschaeftigt dich?]
KOMMENTAR: [Optional. Nur wenn du etwas hast das du laut sagen wuerdest. 1 kurzer Satz. Sonst weglassen.]
TENSION: [Optional. Nur +0.1 oder -0.1 wenn dein Zustand das nahelegt. Sonst weglassen.]
DOMINANCE: [Optional. Nur +0.1 oder -0.1. Sonst weglassen.]

Stil: Kein Markdown. Keine Sternchen. Kein Selbstkommentar. Ruhig und direkt — auch unter Spannung."""


class Introspection:
    """NPU-getriebene Selbstreflexion via DeepSeek R1."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_reflection_time: float = 0.0
        self._last_result: Optional[Dict[str, Any]] = None
        self._reflection_count: int = 0
        # Storage-Verzeichnis anlegen
        os.makedirs(REFLECTIONS_DIR, exist_ok=True)
        logger.info("[INTROSPECTION] Initialisiert")

    def reflect(self) -> Optional[Dict[str, Any]]:
        """Selbstreflexion ausfuehren.

        Returns:
            Dict mit thought, comment, tension_nudge, dominance_nudge
            oder None bei Fehler.
        """
        start = time.time()
        try:
            # Kontext zusammenbauen
            context = self._build_context()
            system = INTROSPECTION_SYSTEM_PROMPT

            # Prompt: Kontext + Frage
            prompt = f"""--- DEIN AKTUELLER ZUSTAND ---
{context}
--- ENDE ZUSTAND ---

Reflektiere ueber deinen aktuellen Zustand. Was nimmst du wahr? Wie fuehlst du dich?"""

            # DeepSeek R1 auf NPU aufrufen (oder Cloud-Fallback)
            from core.autonomy.local_llm_bridge import get_llm_bridge
            raw = get_llm_bridge().reason_internal(
                prompt=prompt, system=system, max_tokens=256)

            if not raw:
                logger.warning("[INTROSPECTION] Keine Antwort vom LLM")
                return None

            # Parsen
            result = self._parse_reflection(raw)
            duration = time.time() - start

            # Speichern
            result["duration_s"] = round(duration, 1)
            result["timestamp"] = datetime.now().isoformat()
            self._store_reflection(result)

            with self._lock:
                self._last_reflection_time = time.time()
                self._last_result = result
                self._reflection_count += 1

            logger.info(
                f"[INTROSPECTION] #{self._reflection_count}: "
                f"\"{result.get('thought', '?')[:80]}\" "
                f"({duration:.1f}s)"
            )
            return result

        except Exception as e:
            logger.error(f"[INTROSPECTION] Fehler: {e}")
            return None

    def _build_context(self) -> str:
        """Kompletten System-Kontext fuer die Reflexion zusammenbauen."""
        parts = []

        # 1. Perception — was sehe ich gerade?
        try:
            from core.voice_pipeline import _perception_to_text
            perception = _perception_to_text()
            if perception:
                parts.append(f"WAHRNEHMUNG:\n{perception}")
        except Exception:
            parts.append("WAHRNEHMUNG: [nicht verfuegbar]")

        # 2. Hardware — wie geht es meinem Koerper?
        try:
            from core.voice_pipeline import _get_hardware_status
            hw = _get_hardware_status()
            if hw:
                parts.append(f"HARDWARE:\n{hw}")
        except Exception:
            pass

        # 3. Core State — Tension, Dominance, Zone
        try:
            from core.core_integrator import get_core_integrator
            ci = get_core_integrator()
            state = ci.get_state()
            zone = ci.get_personality_zone()
            parts.append(
                f"INNERER ZUSTAND:\n"
                f"  Tension: {state['tension']:.3f} "
                f"({'angespannt' if state['tension'] > 0.5 else 'ruhig'})\n"
                f"  Dominance: {state['dominance']:+.3f} "
                f"({'Guardian' if state['dominance'] > 0.15 else 'Shadow' if state['dominance'] < -0.15 else 'Neutral'})\n"
                f"  Zone: {zone}"
            )
        except Exception:
            pass

        # 4. Letzte Entscheidung
        try:
            from core.autonomy.decision_engine import get_decision_engine
            de_state = get_decision_engine().get_state()
            last = de_state.get("last_decision")
            if last and last.get("action") != "silence":
                parts.append(
                    f"LETZTE ENTSCHEIDUNG: {last['action']} "
                    f"({last.get('reason', '?')}, Score {last.get('score', 0):.2f})"
                )
        except Exception:
            pass

        # 5. Diagnostics — Warnungen
        try:
            from core.diagnostics import self_diagnose
            warnings = self_diagnose()
            if warnings:
                parts.append(f"WARNUNGEN: {', '.join(warnings[:3])}")
        except Exception:
            pass

        # 6. Tageszeit
        now = datetime.now()
        parts.append(f"ZEIT: {now.strftime('%H:%M')} Uhr, {now.strftime('%A')}")

        return "\n\n".join(parts)

    def _parse_reflection(self, raw: str) -> Dict[str, Any]:
        """Strukturierten Output parsen. Fault-tolerant."""
        result: Dict[str, Any] = {
            "thought": "",
            "comment": None,
            "tension_nudge": 0.0,
            "dominance_nudge": 0.0,
            "raw": raw,
        }

        # GEDANKE: extrahieren
        m = re.search(r"GEDANKE:\s*(.+?)(?=\n(?:KOMMENTAR|TENSION|DOMINANCE):|$)",
                       raw, re.DOTALL | re.IGNORECASE)
        if m:
            result["thought"] = m.group(1).strip()
        else:
            # Ganzer Output als Gedanke wenn Format nicht erkannt
            result["thought"] = raw.strip()

        # KOMMENTAR: extrahieren
        m = re.search(r"KOMMENTAR:\s*(.+?)(?=\n(?:TENSION|DOMINANCE):|$)",
                       raw, re.DOTALL | re.IGNORECASE)
        if m:
            comment = m.group(1).strip()
            if comment and len(comment) > 3:
                result["comment"] = comment

        # TENSION: +0.1 oder -0.1
        m = re.search(r"TENSION:\s*([+-]?\d*\.?\d+)", raw, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                result["tension_nudge"] = max(-0.2, min(0.2, val))
            except ValueError:
                pass

        # DOMINANCE: +0.1 oder -0.1
        m = re.search(r"DOMINANCE:\s*([+-]?\d*\.?\d+)", raw, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                result["dominance_nudge"] = max(-0.2, min(0.2, val))
            except ValueError:
                pass

        return result

    def _store_reflection(self, result: Dict[str, Any]) -> None:
        """Reflexion als JSONL auf SSD2 speichern."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            filepath = os.path.join(REFLECTIONS_DIR, f"{today}.jsonl")
            # Nur relevante Felder speichern (raw weglassen fuer Platz)
            entry = {
                "t": result.get("timestamp", ""),
                "thought": result.get("thought", ""),
                "comment": result.get("comment"),
                "tension_nudge": result.get("tension_nudge", 0.0),
                "dominance_nudge": result.get("dominance_nudge", 0.0),
                "duration_s": result.get("duration_s", 0),
            }
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[INTROSPECTION] Storage Fehler: {e}")

    def get_state(self) -> Dict[str, Any]:
        """Aktueller State fuer IPC/Panel."""
        with self._lock:
            return {
                "reflection_count": self._reflection_count,
                "last_reflection_time": self._last_reflection_time,
                "last_thought": (self._last_result.get("thought", "")[:100]
                                 if self._last_result else None),
            }


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[Introspection] = None
_instance_lock = threading.Lock()


def get_introspection() -> Introspection:
    """Singleton-Zugriff auf Introspection."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = Introspection()
    return _instance
