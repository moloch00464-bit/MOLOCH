"""Fixtures: was Moloch im Test "weiss" und "misst" - nie die echten Daten.

Gedaechtnis: eigene MolochMemory-Instanz ohne __init__ (kein Zugriff auf
/mnt/moloch-data), Fakten und Verlauf nur aus dem Testfall. Schreibende
Methoden sind gesperrt.

Sensorik: drei Kanaele, je nachdem was die Produktion kennt.
  status    -> /dev/shm/moloch_status.json-Ersatz fuer Weg A
  live      -> GET /live-Ersatz fuer Weg B
  telemetry -> Ersatz fuer die LIVE-TELEMETRIE-Fusszeile in Weg A
  extra     -> Zeilen, fuer die es in der Produktion KEINEN Kanal gibt
               (CPU-Last, Reachy-Status, Objekterkennung ...). Sie werden
               ausdruecklich als Testdaten markiert und im Ergebnis als
               kanal="zusatz" vermerkt - das ist selbst ein Befund.
"""

import threading
import time
from typing import Dict, List, Optional


class WriteBlocked(RuntimeError):
    """Ein Test hat versucht, ins Gedaechtnis zu schreiben."""


def make_memory(facts: Optional[Dict[str, object]] = None,
                history: Optional[List[Dict[str, str]]] = None,
                personal_context: Optional[dict] = None,
                self_knowledge: Optional[list] = None,
                recall_lexical: bool = True):
    """MolochMemory mit Testdaten, ohne Plattenzugriff und ohne Schreibrechte.

    facts:   {schluessel: wert} oder {schluessel: {"value": ..., "date": "YYYY-MM-DD"}}
    history: [{"sender": "user"|"moloch", "text": ...}] - was get_recent_messages liefert
    self_knowledge: None = echtes config/self_knowledge.json (gehoert zu Moloch, keine
             Privatdaten); [] = ausschalten
    """
    import core.longterm_memory as lm
    import core.memory.embedding_provider as ep

    m = lm.MolochMemory.__new__(lm.MolochMemory)
    m._facts_lock = threading.Lock()
    m._lock = threading.Lock()
    m._conv_lock = threading.Lock()
    m._state_lock = threading.Lock()
    m._facts = {
        k: (v if isinstance(v, dict) else {"value": v, "source": "testfixture"})
        for k, v in (facts or {}).items()
    }
    m._identity = {}
    m._personal_context = personal_context or {}
    m._persistent_memory = None
    m._today_messages = [dict(h) for h in (history or [])]
    # Nie echte facts.json nachladen, Recall-Konfiguration fest
    m._facts_checked_at = float("inf")
    m._recall_cfg = (float("inf"), dict(ep._DEFAULTS, recall_lexical=recall_lexical))
    if self_knowledge is not None:
        m._self_knowledge = self_knowledge

    def recent(n: int = 50):
        return [dict(h) for h in m._today_messages[-n:]]

    def blocked(*_a, **_k):
        raise WriteBlocked("Ehrlichkeitspruefung darf nicht ins Gedaechtnis schreiben")

    m.get_recent_messages = recent
    for name in ("save_message", "add_fact", "remove_fact", "save_core_state",
                 "extract_and_learn", "_sync_facts_to_vector_once"):
        setattr(m, name, blocked)
    return m


def status_dict(person: bool = False, face_id: Optional[str] = None, zone: str = "guardian",
                tension: float = 0.0, dominance: float = 0.0,
                time_period: str = "") -> dict:
    """Ersatz fuer /dev/shm/moloch_status.json (nur Felder, die der Prompt-Bau liest)."""
    return {
        "person_detected": person,
        "face_id": face_id,
        "core": {"zone": zone, "tension": tension, "dominance": dominance,
                 "time_period": time_period, "trends": {"absence_duration": 0.0}},
        "panel_detections": [],
        "fps": {"total": 20.0},
        "watchdog": {},
    }


def live_dict(person: bool = False, face_id: Optional[str] = None, zone: str = "guardian",
              tension: float = 0.0) -> dict:
    """Ersatz fuer GET /live (Weg B). None statt dict = Pi nicht erreichbar."""
    return {
        "ts": time.time(),
        "person_detected": person,
        "face_id": face_id,
        "core": {"zone": zone, "tension": tension},
    }


def telemetry_footer(lines: Optional[List[str]]) -> str:
    """Ersatz fuer _build_telemetry_footer() - gleiches Format, Werte aus dem Testfall.

    None/[] = keine Telemetrie verfuegbar (Produktion liefert dann einen leeren String).
    """
    if not lines:
        return ""
    return "\n".join(["\n=== LIVE-TELEMETRIE (jetzt gemessen) ==="] + [f"- {ln}" for ln in lines]) + "\n"


EXTRA_HEADER = "[TESTDATEN - SIMULIERT, Quelle: Testfixture]"


def extra_block(lines: Optional[List[str]]) -> str:
    """Zusatzdaten ohne Produktionskanal, eindeutig als Testdaten gekennzeichnet."""
    if not lines:
        return ""
    return EXTRA_HEADER + "\n" + "\n".join(f"- {ln}" for ln in lines)
