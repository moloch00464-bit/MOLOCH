#!/usr/bin/env python3
"""
M.O.L.O.C.H. Internet Bridge
==============================

Ping-basierter Online/Offline-Status + DuckDuckGo Websuche.
Kein API-Key noetig. Internet ist Tentakel, nicht Abhaengigkeit.
Wenn offline: normale Antwort ohne Suche.
"""

import json
import logging
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("InternetBridge")

# Ping-Ziel: Google DNS (zuverlaessig, schnell)
PING_HOST = "8.8.8.8"
PING_INTERVAL = 30  # Sekunden

# DuckDuckGo Instant Answer API — kein Key, kostenlos
DDG_API = "https://api.duckduckgo.com/"


class InternetBridge:
    """Online/Offline-Monitor + DuckDuckGo Suche."""

    def __init__(self):
        self._online = False
        self._latency_ms = -1
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # Erster Ping sofort beim Start
        self._first_ping_done = False

    def start(self):
        """Ping-Monitor starten."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._ping_loop, daemon=True, name="InternetBridge"
        )
        self._thread.start()
        logger.info("[NET] Internet Bridge gestartet")

    def stop(self):
        self._running = False

    @property
    def online(self) -> bool:
        with self._lock:
            return self._online

    @property
    def latency_ms(self) -> int:
        with self._lock:
            return self._latency_ms

    def _ping_loop(self):
        """Alle 30s pingen, Status + Event publishen."""
        while self._running:
            online, latency = self._do_ping()
            changed = False
            with self._lock:
                if online != self._online or not self._first_ping_done:
                    changed = True
                self._online = online
                self._latency_ms = latency
                self._first_ping_done = True

            if changed:
                self._publish_status(online, latency)

            time.sleep(PING_INTERVAL)

    def _do_ping(self) -> Tuple[bool, int]:
        """Ping via subprocess. Returns (online, latency_ms)."""
        try:
            t0 = time.monotonic()
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "2", PING_HOST],
                capture_output=True,
                timeout=5,
            )
            latency = int((time.monotonic() - t0) * 1000)
            if result.returncode == 0:
                return True, latency
        except Exception as e:
            logger.debug(f"[NET] Ping fehlgeschlagen: {e}")
        return False, -1

    def _publish_status(self, online: bool, latency_ms: int):
        """network.status_changed Event auf Event Bus publishen."""
        try:
            from core.moloch_event_bus import get_event_bus
            get_event_bus().publish(
                event_type="network.status_changed",
                payload={"online": online, "latency_ms": latency_ms},
                source="internet_bridge",
                priority=5,
            )
            logger.info(f"[NET] Status geaendert: {'ONLINE' if online else 'OFFLINE'} {latency_ms}ms")
        except Exception as e:
            logger.debug(f"[NET] Event Bus nicht erreichbar: {e}")

    def search_web(self, query: str) -> List[Dict]:
        """Websuche: DuckDuckGo Instant Answer API + wttr.in fuer Wetter.

        Kein API-Key noetig. Kostenlos.
        Returns: Liste von max 3 Ergebnissen mit 'title', 'text', 'url'
        Bei Fehler oder Offline: leere Liste.
        """
        if not self.online:
            logger.warning("[NET] Offline — Suche nicht moeglich")
            return []

        # Wetter-Spezialfall via wttr.in (kein Key, JSON-API)
        if self._is_weather_query(query):
            weather = self._search_weather(query)
            if weather:
                return weather

        # DuckDuckGo Instant Answer API
        return self._search_ddg(query)

    def _is_weather_query(self, query: str) -> bool:
        """Pruefen ob Wetter-Anfrage."""
        q = query.lower()
        return any(w in q for w in ["wetter", "temperatur", "regen", "schnee", "grad", "weather", "forecast"])

    def _search_weather(self, query: str) -> List[Dict]:
        """Wetter via wttr.in JSON API — kein Key, kostenlos."""
        import re as _re
        # Stadt aus Query extrahieren (letztes Wort oder nach "in")
        q = query.lower()
        m = _re.search(r"(?:in|fuer|für)\s+(\w[\w\s]+?)(?:\s+heute|\s+morgen|$)", q)
        if m:
            city = m.group(1).strip().replace(" ", "+")
        else:
            # Letztes Wort nehmen
            words = [w for w in query.split() if w.lower() not in
                     ["wetter", "temperatur", "wie", "ist", "mal", "in", "für", "heute", "morgen"]]
            city = words[-1] if words else "Nuernberg"

        try:
            url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "MOLOCH/1.0"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            current = data.get("current_condition", [{}])[0]
            temp_c = current.get("temp_C", "?")
            feels = current.get("FeelsLikeC", "?")
            desc = current.get("weatherDesc", [{"value": ""}])[0].get("value", "")
            humidity = current.get("humidity", "?")
            wind = current.get("windspeedKmph", "?")

            text = (
                f"{city.replace('+', ' ').title()}: {temp_c}°C (gefühlt {feels}°C), "
                f"{desc}, Luftfeuchtigkeit {humidity}%, Wind {wind} km/h"
            )

            logger.info(f"[NET] Wetter '{city}': OK")
            return [{"title": f"Wetter {city.replace('+', ' ').title()}", "text": text, "url": f"https://wttr.in/{city}"}]

        except Exception as e:
            logger.error(f"[NET] Wetter-Abfrage fehlgeschlagen: {e}")
            return []

    def _search_ddg(self, query: str) -> List[Dict]:
        """DuckDuckGo Instant Answer API."""
        try:
            params = urllib.parse.urlencode({
                "q": query,
                "format": "json",
                "no_redirect": "1",
                "no_html": "1",
                "skip_disambig": "1",
            })
            url = f"{DDG_API}?{params}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "MOLOCH/1.0 (Pi5 HomeAssistant)"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = []

            # Direktantwort (z.B. Umrechnung, Definition)
            answer = data.get("Answer", "").strip()
            if answer:
                results.append({
                    "title": "Direkte Antwort",
                    "text": str(answer)[:300],
                    "url": "",
                })

            # Hauptartikel (Wikipedia-Zusammenfassung etc.)
            abstract = data.get("AbstractText", "").strip()
            if abstract and len(results) < 3:
                results.append({
                    "title": data.get("Heading", query),
                    "text": abstract[:300],
                    "url": data.get("AbstractURL", ""),
                })

            # Verwandte Themen
            for topic in data.get("RelatedTopics", []):
                if len(results) >= 3:
                    break
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "")[:80],
                        "text": topic.get("Text", "")[:200],
                        "url": topic.get("FirstURL", ""),
                    })

            logger.info(f"[NET] DDG '{query}': {len(results)} Ergebnis(se)")
            return results[:3]

        except Exception as e:
            logger.error(f"[NET] DDG-Suche fehlgeschlagen: {e}")
            return []


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_bridge: Optional[InternetBridge] = None
_bridge_lock = threading.Lock()


def get_internet_bridge() -> InternetBridge:
    """Singleton-Zugriff. Startet Bridge automatisch beim ersten Aufruf."""
    global _bridge
    with _bridge_lock:
        if _bridge is None:
            _bridge = InternetBridge()
            _bridge.start()
    return _bridge
