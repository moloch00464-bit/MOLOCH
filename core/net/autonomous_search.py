#!/usr/bin/env python3
"""
M.O.L.O.C.H. Autonome Internet-Suche ("Neugier-Engine")
=========================================================

Erlaubnisbasierte, kontextgesteuerte Websuche.
Moloch sucht eigenstaendig nach Themen die zu seiner aktuellen
Persoenlichkeits-Zone, Tageszeit und Gespraechsthemen passen.

Sicherheit:
  - Default AUS — jeder Boot startet ohne Erlaubnis
  - TTL — Erlaubnis laeuft nach konfigurierbarer Zeit ab (default 60min)
  - Max Suchen pro Session (default 20)
  - Cooldown 5min zwischen Suchen
  - Alles via bestehende InternetBridge (keine neuen HTTP-Verbindungen)

Steuerung:
  - Voice: "Du darfst ins Netz" / "Bleib offline"
  - Console: /net on | /net off
  - Decision Engine: web_search Aktion mit Utility-Score

Singleton: get_autonomous_search()
"""

import json
import logging
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("AutonomousSearch")

# Pfade
SETTINGS_PATH = Path.home() / "moloch" / "config" / "settings.json"

# Defaults (wenn settings.json keinen internet-Block hat)
DEFAULT_INTERVAL = 300      # 5 Minuten zwischen Suchen
DEFAULT_TTL = 3600          # 60 Minuten Erlaubnis-Dauer
DEFAULT_CITY = "Nuernberg"
DEFAULT_MAX_PER_SESSION = 20
LOOP_SLEEP = 30             # Daemon prueft alle 30s

# Themen-Templates nach Persoenlichkeits-Zone + Tageszeit
TOPIC_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "guardian": {
        "morning": [
            "Wetter {city}",
            "Tech News heute",
            "Sicherheits-Updates Linux",
            "Raspberry Pi Neuigkeiten",
        ],
        "day": [
            "Python Updates",
            "KI Nachrichten heute",
            "Cybersecurity News",
            "Open Source Projekte",
            "Raspberry Pi Projekte",
        ],
        "evening": [
            "Nachrichten Deutschland",
            "Technik Trends",
            "Wissenschaft aktuell",
        ],
        "night": [
            "Astronomie aktuell",
            "Nachtruhe Tipps",
            "Weltraum Nachrichten",
        ],
    },
    "shadow": {
        "morning": [
            "Kuriose Nachrichten heute",
            "Hacker News heute",
            "Dark Web Trends",
        ],
        "day": [
            "Underground Musik News",
            "Cyberpunk Technologie",
            "Kuenstliche Intelligenz Gefahren",
            "Dystopie Nachrichten",
        ],
        "evening": [
            "Horror Filme aktuell",
            "Unheimliche Geschichten",
            "Creepypasta deutsch",
        ],
        "night": [
            "Nachthimmel Phaenomene",
            "Okkulte Geschichte",
            "Geistergeschichten",
        ],
    },
    "berserker": {
        "morning": ["Sicherheitswarnungen heute", "Cyberangriffe aktuell"],
        "day": ["Exploits aktuell", "Aggrotech Musik News"],
        "evening": ["Sicherheitsluecken Linux", "Hacking Nachrichten"],
        "night": ["DDoS Attacken aktuell", "Zero Day Exploits"],
    },
}


def _get_time_slot() -> str:
    """Tageszeit als Slot: morning/day/evening/night."""
    hour = time.localtime().tm_hour
    if 6 <= hour < 10:
        return "morning"
    elif 10 <= hour < 18:
        return "day"
    elif 18 <= hour < 23:
        return "evening"
    else:
        return "night"


def _load_internet_settings() -> Dict[str, Any]:
    """Internet-Einstellungen aus settings.json laden."""
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("internet", {})
    except Exception:
        return {}


def _extract_topics_from_messages(messages: List[Dict]) -> List[str]:
    """Einfache Themen-Extraktion aus letzten Nachrichten.

    Sucht nach Substantiven (grossgeschriebene Woerter mit >3 Buchstaben),
    Angefuehrte Begriffe, und Fragen mit "ueber/zu/von".
    Kein NLP noetig — haelt RAM-Verbrauch niedrig.
    """
    topics = []
    seen = set()

    for msg in messages:
        text = msg.get("text", "")
        if not text:
            continue

        # Angefuehrte Begriffe: "XY" oder 'XY'
        for m in re.finditer(r'["\u201e\u201c\u201d\']([\w\s]{3,30})["\u201e\u201c\u201d\']', text):
            t = m.group(1).strip()
            if t.lower() not in seen and len(t) > 3:
                topics.append(t)
                seen.add(t.lower())

        # "ueber/zu/von X" Muster
        for m in re.finditer(r'(?:ueber|über|zu|von)\s+([\w\s]{3,25}?)(?:\s*[\.\?\!,]|$)', text):
            t = m.group(1).strip()
            if t.lower() not in seen and len(t.split()) <= 4:
                topics.append(t)
                seen.add(t.lower())

        # Grossgeschriebene Woerter (wahrscheinlich Eigennamen/Fachbegriffe)
        for word in text.split():
            # Nur Woerter die mit Grossbuchstabe anfangen, >4 Zeichen
            # und keine deutschen Satzanfang-Woerter
            if (len(word) > 4 and word[0].isupper() and word.isalpha()
                    and word.lower() not in {
                        "haben", "nicht", "diese", "einer", "einen",
                        "meine", "deine", "seine", "keine", "wegen",
                        "gegen", "unter", "dabei", "immer", "warum",
                        "wohin", "wofuer", "moloch", "markus",
                    }):
                if word.lower() not in seen:
                    topics.append(word)
                    seen.add(word.lower())

    # Max 5 Themen, neueste zuerst (Messages sind chronologisch)
    return topics[-5:]


class AutonomousSearch:
    """Erlaubnisbasierte, kontextgesteuerte autonome Websuche."""

    def __init__(self):
        self._lock = threading.Lock()
        self._permitted = False
        self._permitted_at = 0.0
        self._last_search_time = 0.0
        self._search_count = 0
        self._search_history: List[Dict] = []  # Ringpuffer, max 20
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Settings laden
        cfg = _load_internet_settings()
        self._search_interval = float(cfg.get("autonomous_search_interval", DEFAULT_INTERVAL))
        self._permission_ttl = float(cfg.get("autonomous_search_ttl", DEFAULT_TTL))
        self._city = cfg.get("autonomous_search_city", DEFAULT_CITY)
        self._max_per_session = int(cfg.get("autonomous_search_max_per_session", DEFAULT_MAX_PER_SESSION))

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self):
        """Daemon-Thread starten."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._search_loop, daemon=True, name="AutonomousSearch"
        )
        self._thread.start()
        logger.info("[SEARCH] Autonome Suche Engine gestartet (Default: AUS)")

    def stop(self):
        """Daemon stoppen."""
        self._running = False

    # =========================================================================
    # Permission Gate
    # =========================================================================

    def grant_permission(self, ttl_seconds: Optional[float] = None):
        """Erlaubnis fuer autonome Suche erteilen.

        Args:
            ttl_seconds: Wie lange die Erlaubnis gilt. None = default aus Settings.
        """
        with self._lock:
            self._permitted = True
            self._permitted_at = time.time()
            if ttl_seconds is not None:
                self._permission_ttl = float(ttl_seconds)
            self._search_count = 0  # Session-Zaehler reset

        ttl_min = self._permission_ttl / 60
        logger.info(f"[SEARCH] Autonome Suche AKTIVIERT (TTL: {ttl_min:.0f}min)")

        # Event publishen
        try:
            from core.moloch_event_bus import get_event_bus
            get_event_bus().publish(
                event_type="autonomous_search.permission_granted",
                source="autonomous_search",
                priority=5,
                payload={"ttl_seconds": self._permission_ttl},
            )
        except Exception:
            pass

    def revoke_permission(self):
        """Erlaubnis entziehen."""
        with self._lock:
            self._permitted = False

        logger.info("[SEARCH] Autonome Suche DEAKTIVIERT")

        try:
            from core.moloch_event_bus import get_event_bus
            get_event_bus().publish(
                event_type="autonomous_search.permission_revoked",
                source="autonomous_search",
                priority=5,
                payload={},
            )
        except Exception:
            pass

    @property
    def permitted(self) -> bool:
        """Pruefen ob Suche erlaubt ist (Gate + TTL)."""
        with self._lock:
            if not self._permitted:
                return False
            # TTL pruefen
            elapsed = time.time() - self._permitted_at
            if elapsed >= self._permission_ttl:
                self._permitted = False
                logger.info("[SEARCH] Erlaubnis abgelaufen (TTL)")
                # Event im Hintergrund publishen
                threading.Thread(
                    target=self._publish_revoked_event, daemon=True
                ).start()
                return False
            # Max Suchen pruefen
            if self._search_count >= self._max_per_session:
                logger.info(f"[SEARCH] Max Suchen erreicht ({self._max_per_session})")
                return False
            return True

    def _publish_revoked_event(self):
        """TTL-Ablauf Event publishen (in eigenem Thread wegen Lock)."""
        try:
            from core.moloch_event_bus import get_event_bus
            get_event_bus().publish(
                event_type="autonomous_search.permission_revoked",
                source="autonomous_search",
                priority=5,
                payload={"reason": "ttl_expired"},
            )
        except Exception:
            pass

    # =========================================================================
    # Such-Daemon
    # =========================================================================

    def _search_loop(self):
        """Daemon-Thread: Prueft alle 30s ob gesucht werden soll."""
        while self._running:
            time.sleep(LOOP_SLEEP)
            if not self.permitted:
                continue
            # Intervall pruefen
            with self._lock:
                elapsed = time.time() - self._last_search_time
                if elapsed < self._search_interval:
                    continue

            try:
                self._do_one_search()
            except Exception as e:
                logger.error(f"[SEARCH] Autonome Suche fehlgeschlagen: {e}")

    def _do_one_search(self):
        """Eine autonome Suche durchfuehren."""
        query = self._generate_query()
        if not query:
            return

        results = self._execute_search(query)
        with self._lock:
            self._last_search_time = time.time()
            self._search_count += 1

        if results:
            self._evaluate_and_store(query, results)

    def trigger_search(self):
        """Sofort-Suche ausloesen (fuer Decision Engine).

        Fuehrt eine Suche durch, unabhaengig vom Intervall-Timer.
        Respektiert aber weiterhin Permission + Max-Limit.
        """
        if not self.permitted:
            logger.debug("[SEARCH] Trigger abgelehnt: keine Erlaubnis")
            return
        try:
            self._do_one_search()
        except Exception as e:
            logger.error(f"[SEARCH] Trigger-Suche fehlgeschlagen: {e}")

    # =========================================================================
    # Query-Generierung
    # =========================================================================

    def _generate_query(self) -> Optional[str]:
        """Kontext-basierte Suchanfrage generieren.

        Beruecksichtigt:
          1. Persoenlichkeits-Zone (Guardian/Shadow/Berserker)
          2. Tageszeit (morning/day/evening/night)
          3. Letzte Gespraeche (Themen-Extraktion)
          4. Duplikat-Check gegen History

        Returns: Suchbegriff oder None wenn nichts Interessantes.
        """
        # Zone holen
        zone = "guardian"
        try:
            from core.core_integrator import get_core_integrator
            zone = get_core_integrator().get_personality_zone()
        except Exception:
            pass

        time_slot = _get_time_slot()

        # Gespraechs-Themen extrahieren
        conversation_topics = []
        try:
            from core.longterm_memory import get_memory
            messages = get_memory().get_recent_messages(10)
            conversation_topics = _extract_topics_from_messages(messages)
        except Exception:
            pass

        # Kandidaten sammeln
        candidates = []

        # Template-Themen (Zone + Tageszeit)
        templates = TOPIC_TEMPLATES.get(zone, TOPIC_TEMPLATES["guardian"])
        slot_topics = templates.get(time_slot, templates.get("day", []))
        for t in slot_topics:
            candidates.append(t.format(city=self._city))

        # Gespraechs-basierte Themen
        for topic in conversation_topics:
            candidates.append(f"{topic} aktuell")
            candidates.append(f"Was gibt es Neues ueber {topic}")

        # Duplikat-Check: keine Suche die in letzter Stunde schon lief
        with self._lock:
            recent_queries = {
                h["query"].lower()
                for h in self._search_history
                if time.time() - h.get("time", 0) < 3600
            }

        candidates = [
            c for c in candidates
            if c.lower() not in recent_queries
        ]

        if not candidates:
            logger.debug("[SEARCH] Keine neuen Themen — ueberspringe")
            return None

        # Zufaellig aus Kandidaten waehlen (leichte Bevorzugung von Konversations-Themen)
        if conversation_topics and len(candidates) > len(slot_topics):
            # 60% Chance auf Konversations-Thema
            if random.random() < 0.6:
                conv_candidates = candidates[len(slot_topics):]
                if conv_candidates:
                    return random.choice(conv_candidates)

        return random.choice(candidates)

    # =========================================================================
    # Suche + Bewertung
    # =========================================================================

    def _execute_search(self, query: str) -> List[Dict]:
        """Suche via InternetBridge ausfuehren."""
        try:
            from core.net.internet_bridge import get_internet_bridge
            bridge = get_internet_bridge()
            if not bridge.online:
                logger.warning("[SEARCH] Offline — Suche abgebrochen")
                return []
            logger.info(f"[SEARCH] Autonome Suche: '{query}'")
            return bridge.search_web(query)
        except Exception as e:
            logger.error(f"[SEARCH] Suche fehlgeschlagen: {e}")
            return []

    def _evaluate_and_store(self, query: str, results: List[Dict]):
        """Ergebnisse bewerten, speichern und Event publishen."""
        if not results:
            return

        # Bestes Ergebnis (erstes mit >50 Zeichen Text)
        best = None
        for r in results:
            if len(r.get("text", "")) >= 50:
                best = r
                break
        if not best:
            best = results[0]

        title = best.get("title", "Unbekannt")
        text = best.get("text", "")[:200]
        source = best.get("source", "Web")

        # In History speichern (Ringpuffer)
        with self._lock:
            self._search_history.append({
                "query": query,
                "time": time.time(),
                "title": title,
                "source": source,
                "results_count": len(results),
            })
            # Ringpuffer: max 20 Eintraege
            if len(self._search_history) > 20:
                self._search_history = self._search_history[-20:]

        # In Langzeitgedaechtnis speichern
        try:
            from core.longterm_memory import get_memory
            summary = f"[WEBFUND] {query}: {title} — {text}"
            get_memory().save_message("moloch", summary, source="autonomous_search")
        except Exception as e:
            logger.error(f"[SEARCH] Memory-Speicherung fehlgeschlagen: {e}")

        # Event publishen (fuer TTS + GUI)
        try:
            from core.moloch_event_bus import get_event_bus
            get_event_bus().publish(
                event_type="autonomous_search.result",
                source="autonomous_search",
                priority=5,
                payload={
                    "query": query,
                    "title": title,
                    "summary": text,
                    "source": source,
                    "results_count": len(results),
                    "speak": True,
                },
            )
        except Exception as e:
            logger.error(f"[SEARCH] Event publish fehlgeschlagen: {e}")

        logger.info(f"[SEARCH] Fund: [{source}] {title[:60]}")

    # =========================================================================
    # Status API
    # =========================================================================

    def get_state(self) -> Dict[str, Any]:
        """Aktueller Status fuer IPC/Panel/Console."""
        with self._lock:
            ttl_remaining = 0.0
            if self._permitted:
                ttl_remaining = max(0, self._permission_ttl - (time.time() - self._permitted_at))

            return {
                "permitted": self._permitted and ttl_remaining > 0,
                "ttl_remaining": round(ttl_remaining, 1),
                "search_interval": self._search_interval,
                "search_count": self._search_count,
                "max_per_session": self._max_per_session,
                "last_search_time": self._last_search_time,
                "last_query": (
                    self._search_history[-1]["query"]
                    if self._search_history else "keine"
                ),
                "history": self._search_history[-5:],
            }


# =========================================================================
# Singleton
# =========================================================================

_instance: Optional[AutonomousSearch] = None
_instance_lock = threading.Lock()


def get_autonomous_search() -> AutonomousSearch:
    """Singleton-Zugriff. Startet Engine automatisch beim ersten Aufruf."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = AutonomousSearch()
            _instance.start()
    return _instance
