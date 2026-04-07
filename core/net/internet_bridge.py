#!/usr/bin/env python3
"""
M.O.L.O.C.H. Internet Bridge
==============================

Ping-basierter Online/Offline-Status + universelle Websuche.
Drei Backends: DuckDuckGo API, DDG HTML Scraper, Wikipedia API.
Kein API-Key noetig. Internet ist Tentakel, nicht Abhaengigkeit.
Wenn offline: normale Antwort ohne Suche.
"""

import json
import logging
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("InternetBridge")

# Ping-Ziel: Google DNS
PING_HOST = "8.8.8.8"
PING_INTERVAL = 30  # Sekunden

# Backends
DDG_API = "https://api.duckduckgo.com/"
DDG_LITE = "https://lite.duckduckgo.com/lite/"
WIKI_API = "https://de.wikipedia.org/api/rest_v1/page/summary/"

# User-Agent (kein Bot-Block)
UA = "MOLOCH/2.0 (Pi5 HomeAssistant; Raspberry Pi; nicht kommerziell)"

# Timeout pro Backend
TIMEOUT = 8


class InternetBridge:
    """Online/Offline-Monitor + universelle Websuche mit 3 Backends."""

    def __init__(self):
        self._online = False
        self._latency_ms = -1
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._first_ping_done = False

    # -------------------------------------------------------------------------
    # Ping-Monitor
    # -------------------------------------------------------------------------

    def start(self):
        """Ping-Monitor starten."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._ping_loop, daemon=True, name="InternetBridge"
        )
        self._thread.start()
        logger.info("[NET] Internet Bridge v2 gestartet (3 Backends)")

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

    def get_network_status(self) -> Dict:
        """Netzwerk-Status als Dict fuer Chat-Kontext."""
        with self._lock:
            return {
                "online": self._online,
                "latency_ms": self._latency_ms,
                "status_text": (
                    f"ONLINE ({self._latency_ms}ms)"
                    if self._online
                    else "OFFLINE"
                ),
            }

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
            logger.info(
                f"[NET] Status: {'ONLINE' if online else 'OFFLINE'} {latency_ms}ms"
            )
        except Exception as e:
            logger.debug(f"[NET] Event Bus nicht erreichbar: {e}")

    # -------------------------------------------------------------------------
    # Haupt-Suchfunktion
    # -------------------------------------------------------------------------

    def search_web(self, query: str) -> List[Dict]:
        """Universelle Websuche — bis zu 5 Ergebnisse aus 3 Backends.

        Reihenfolge:
          1. Wetter-Spezialfall → wttr.in (live, immer aktuell)
          2. Wikipedia API (DE) → tiefe Informationen
          3. DDG Instant Answer API → schnelle Fakten, Umrechnungen
          4. DDG HTML Lite Scraper → echte Websuche, Fallback

        Returns: Liste von max 5 Ergebnissen mit 'title', 'text', 'url', 'source'
        Bei Fehler oder Offline: leere Liste.
        """
        if not self.online:
            logger.warning("[NET] Offline — Suche nicht moeglich")
            return []

        results: List[Dict] = []

        # 1. Wetter via wttr.in
        if self._is_weather_query(query):
            weather = self._search_weather(query)
            results.extend(weather)

        # 2. News-Anfragen → Google News RSS (aktuell), sonst Wikipedia
        if self._is_news_query(query):
            news = self._search_news_rss(query)
            for r in news:
                if len(results) < 5 and not self._is_duplicate(r, results):
                    results.append(r)
        else:
            wiki = self._search_wikipedia(query)
            for r in wiki:
                if len(results) < 5 and not self._is_duplicate(r, results):
                    results.append(r)

        # 3. DDG Instant Answer (Fakten, Definitionen, Umrechnungen)
        ddg = self._search_ddg_api(query)
        for r in ddg:
            if len(results) < 5 and not self._is_duplicate(r, results):
                results.append(r)

        # 4. DDG HTML Scraper als Fallback wenn noch < 3 Ergebnisse
        if len(results) < 3:
            html = self._search_ddg_html(query)
            for r in html:
                if len(results) < 5 and not self._is_duplicate(r, results):
                    results.append(r)

        logger.info(f"[NET] Suche '{query[:50]}': {len(results)} Ergebnis(se) total")
        return results[:5]

    def fetch_page(self, url: str, max_chars: int = 2000) -> Optional[str]:
        """Seiteninhalt holen und Text extrahieren (kein JS, kein Selenium).

        Returns: Reiner Text-Inhalt, max max_chars Zeichen. None bei Fehler.
        """
        if not self.online:
            return None
        try:
            import requests
            from bs4 import BeautifulSoup
            resp = requests.get(
                url, timeout=TIMEOUT, headers={"User-Agent": UA},
                allow_redirects=True,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            # Navigations-Bloat entfernen
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            # Mehrfache Leerzeichen bereinigen
            text = re.sub(r"\s+", " ", text).strip()
            return text[:max_chars]
        except Exception as e:
            logger.error(f"[NET] fetch_page fehlgeschlagen: {e}")
            return None

    # -------------------------------------------------------------------------
    # Backend 1: Wetter (wttr.in)
    # -------------------------------------------------------------------------

    def _is_weather_query(self, query: str) -> bool:
        q = query.lower()
        return any(
            w in q
            for w in ["wetter", "temperatur", "regen", "schnee", "grad",
                       "wind", "weather", "forecast", "niederschlag",
                       "sonnenschein", "bewölkt", "bewoelkt"]
        )

    def _is_news_query(self, query: str) -> bool:
        """Pruefen ob Anfrage aktuelle Nachrichten betrifft."""
        q = query.lower()
        return any(w in q for w in [
            "nachrichten", "news", "meldung", "aktuell", "heute",
            "passiert", "ereignis", "schlagzeilen", "top-news",
        ])

    def _search_news_rss(self, query: str) -> List[Dict]:
        """Google News RSS — aktuelle Nachrichten ohne API-Key."""
        try:
            import xml.etree.ElementTree as ET
            params = urllib.parse.urlencode({
                "q": query, "hl": "de", "gl": "DE", "ceid": "DE:de"
            })
            url = f"https://news.google.com/rss/search?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                tree = ET.fromstring(resp.read())
            results = []
            for item in tree.findall(".//item")[:4]:
                title = (item.findtext("title") or "").strip()
                desc  = re.sub(r"<[^>]+>", "", item.findtext("description") or "").strip()
                link  = (item.findtext("link") or "").strip()
                pub   = (item.findtext("pubDate") or "").strip()
                if title and desc:
                    results.append({
                        "title": title[:100],
                        "text":  (f"[{pub[:16]}] " if pub else "") + desc[:250],
                        "url":   link,
                        "source": "Google News",
                    })
            logger.info(f"[NET] News RSS '{query[:40]}': {len(results)} Ergebnis(se)")
            return results
        except Exception as e:
            logger.debug(f"[NET] News RSS fehlgeschlagen: {e}")
            return []

    def _search_weather(self, query: str) -> List[Dict]:
        """Wetter via wttr.in JSON API — kein Key, kostenlos."""
        # Stadt extrahieren
        q = query.lower()
        m = re.search(r"(?:in|fuer|für|bei)\s+(\w[\w\s]{1,20}?)(?:\s+(?:heute|morgen|wetter)|[\?\.\!]|$)", q)
        if m:
            city = m.group(1).strip().replace(" ", "+")
        else:
            words = [
                w for w in query.split()
                if w.lower() not in
                ["wetter", "temperatur", "wie", "ist", "mal", "in", "für",
                 "fuer", "heute", "morgen", "beim", "bei", "das", "der", "die"]
            ]
            city = words[-1] if words else "Nuernberg"

        try:
            url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            current = data.get("current_condition", [{}])[0]
            temp_c = current.get("temp_C", "?")
            feels = current.get("FeelsLikeC", "?")
            desc_list = current.get("weatherDesc", [{"value": ""}])
            desc = desc_list[0].get("value", "") if desc_list else ""
            humidity = current.get("humidity", "?")
            wind = current.get("windspeedKmph", "?")

            # Deutsch-Uebersetzung der Beschreibung
            desc_de = _translate_weather_desc(desc)

            text = (
                f"{city.replace('+', ' ').title()}: {temp_c}°C "
                f"(gefühlt {feels}°C), {desc_de}, "
                f"Luftfeuchtigkeit {humidity}%, Wind {wind} km/h"
            )

            logger.info(f"[NET] Wetter '{city}': OK")
            return [{
                "title": f"Wetter {city.replace('+', ' ').title()}",
                "text": text,
                "url": f"https://wttr.in/{city}",
                "source": "wttr.in",
            }]
        except Exception as e:
            logger.error(f"[NET] Wetter-Abfrage fehlgeschlagen: {e}")
            return []

    # -------------------------------------------------------------------------
    # Backend 2: Wikipedia REST API (Deutsch)
    # -------------------------------------------------------------------------

    def _search_wikipedia(self, query: str) -> List[Dict]:
        """Wikipedia Zusammenfassung holen via REST API.

        Strategie: Suchbegriff normalisieren → direkt abrufen.
        Bei 404 → englische Wikipedia versuchen.
        """
        results = []

        # Suchbegriff fuer Wikipedia normalisieren
        # Deutsch zuerst, dann Englisch falls nix
        for lang, base_url in [
            ("de", "https://de.wikipedia.org/api/rest_v1/page/summary/"),
            ("en", "https://en.wikipedia.org/api/rest_v1/page/summary/"),
        ]:
            if results:
                break
            try:
                title = urllib.parse.quote(query.strip().replace(" ", "_"))
                url = base_url + title
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                extract = data.get("extract", "").strip()
                if not extract or len(extract) < 50:
                    continue

                # Zusammenfassung kuerzen (max 400 Zeichen)
                if len(extract) > 400:
                    extract = extract[:397] + "..."

                results.append({
                    "title": data.get("title", query),
                    "text": extract,
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "source": f"Wikipedia ({lang.upper()})",
                })
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    logger.debug(f"[NET] Wikipedia ({lang}) HTTP {e.code}: {query}")
            except Exception as e:
                logger.debug(f"[NET] Wikipedia ({lang}) fehlgeschlagen: {e}")

        # Falls kein Direkt-Hit: Wikipedia-Suche nutzen
        if not results:
            results = self._search_wikipedia_full_text(query)

        return results

    def _search_wikipedia_full_text(self, query: str) -> List[Dict]:
        """Wikipedia Volltextsuche als Fallback."""
        try:
            params = urllib.parse.urlencode({
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 2,
                "format": "json",
            })
            url = f"https://de.wikipedia.org/w/api.php?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = []
            for item in data.get("query", {}).get("search", []):
                snippet = re.sub(r"<[^>]+>", "", item.get("snippet", "")).strip()
                if snippet:
                    results.append({
                        "title": item.get("title", query),
                        "text": snippet[:300],
                        "url": f"https://de.wikipedia.org/wiki/{urllib.parse.quote(item.get('title', ''))}",
                        "source": "Wikipedia DE",
                    })
            return results[:2]
        except Exception as e:
            logger.debug(f"[NET] Wikipedia Volltextsuche fehlgeschlagen: {e}")
            return []

    # -------------------------------------------------------------------------
    # Backend 3: DDG Instant Answer API
    # -------------------------------------------------------------------------

    def _search_ddg_api(self, query: str) -> List[Dict]:
        """DuckDuckGo Instant Answer API — schnell, kein Key."""
        try:
            params = urllib.parse.urlencode({
                "q": query,
                "format": "json",
                "no_redirect": "1",
                "no_html": "1",
                "skip_disambig": "1",
            })
            url = f"{DDG_API}?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = []

            # Direktantwort (Umrechnungen, Definitionen, Fakten)
            answer = data.get("Answer", "").strip()
            if answer:
                results.append({
                    "title": "Direkte Antwort",
                    "text": str(answer)[:400],
                    "url": "",
                    "source": "DuckDuckGo",
                })

            # Wikipedia-Zusammenfassung via DDG
            abstract = data.get("AbstractText", "").strip()
            if abstract and len(results) < 3:
                results.append({
                    "title": data.get("Heading", query),
                    "text": abstract[:400],
                    "url": data.get("AbstractURL", ""),
                    "source": f"DDG/{data.get('AbstractSource', 'Web')}",
                })

            # Verwandte Themen
            for topic in data.get("RelatedTopics", []):
                if len(results) >= 3:
                    break
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "")[:80],
                        "text": topic.get("Text", "")[:250],
                        "url": topic.get("FirstURL", ""),
                        "source": "DuckDuckGo",
                    })

            logger.info(f"[NET] DDG API '{query[:40]}': {len(results)} Ergebnis(se)")
            return results[:3]

        except Exception as e:
            logger.debug(f"[NET] DDG API fehlgeschlagen: {e}")
            return []

    # -------------------------------------------------------------------------
    # Backend 4: DDG HTML Lite Scraper (Fallback, echte Websuche)
    # -------------------------------------------------------------------------

    def _search_ddg_html(self, query: str) -> List[Dict]:
        """DDG Lite HTML Scraper — keine JS noetig, leichtgewichtig.

        Nutzt requests + BeautifulSoup. Gibt echte Web-Ergebnisse.
        """
        try:
            import requests
            from bs4 import BeautifulSoup

            params = {"q": query, "kl": "de-de"}
            resp = requests.post(
                DDG_LITE,
                data=params,
                headers={
                    "User-Agent": UA,
                    "Accept-Language": "de-DE,de;q=0.9",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                timeout=TIMEOUT,
                allow_redirects=True,
            )
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            results = []

            # DDG Lite Struktur: <table> mit Ergebnissen
            for row in soup.select("tr"):
                if len(results) >= 4:
                    break

                # Titel + URL
                link = row.select_one("a.result-link")
                snippet_td = row.select_one("td.result-snippet")

                if not link or not snippet_td:
                    continue

                title = link.get_text(strip=True)
                href = link.get("href", "")
                snippet = snippet_td.get_text(strip=True)

                if not title or not snippet or len(snippet) < 20:
                    continue

                results.append({
                    "title": title[:100],
                    "text": snippet[:300],
                    "url": href,
                    "source": "Web (DDG)",
                })

            logger.info(f"[NET] DDG HTML '{query[:40]}': {len(results)} Ergebnis(se)")
            return results

        except Exception as e:
            logger.debug(f"[NET] DDG HTML fehlgeschlagen: {e}")
            return []

    # -------------------------------------------------------------------------
    # Hilfsfunktionen
    # -------------------------------------------------------------------------

    def _is_duplicate(self, candidate: Dict, existing: List[Dict]) -> bool:
        """Duplikat-Check: Text-Overlap > 60% = Duplikat."""
        c_text = candidate.get("text", "").lower()[:100]
        for e in existing:
            e_text = e.get("text", "").lower()[:100]
            if c_text and e_text and c_text[:80] == e_text[:80]:
                return True
        return False


# -------------------------------------------------------------------------
# Wetter-Beschreibung Deutsch
# -------------------------------------------------------------------------

_WEATHER_DE = {
    "Sunny": "sonnig",
    "Clear": "klar",
    "Partly cloudy": "teils bewölkt",
    "Cloudy": "bewölkt",
    "Overcast": "bedeckt",
    "Mist": "Nebel",
    "Fog": "Nebel",
    "Light rain": "leichter Regen",
    "Moderate rain": "mäßiger Regen",
    "Heavy rain": "starker Regen",
    "Light snow": "leichter Schnee",
    "Moderate snow": "mäßiger Schnee",
    "Heavy snow": "starker Schnee",
    "Light rain shower": "Regenschauer",
    "rain shower": "Regenschauer",
    "Thunderstorm": "Gewitter",
    "Blizzard": "Schneesturm",
    "Drizzle": "Nieselregen",
    "Freezing drizzle": "gefrierender Nieselregen",
    "Light sleet": "leichter Schneeregen",
    "Patchy rain possible": "vereinzelt Regen möglich",
}


def _translate_weather_desc(desc: str) -> str:
    for en, de in _WEATHER_DE.items():
        if en.lower() in desc.lower():
            return de
    return desc  # Original wenn kein Treffer


# -------------------------------------------------------------------------
# Singleton
# -------------------------------------------------------------------------

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
