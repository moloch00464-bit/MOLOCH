#!/usr/bin/env python3
"""
M.O.L.O.C.H. Spotify Controller
=================================

ARCHITEKTUR: Lokaler Track-Index ist die EINZIGE Quelle fuer Musikauswahl.
  - 4941 Artists, 24454 Tracks, 151790 Plays aus Markus' Streaming History
  - KEINE Spotify-Suche (sp.search), KEINE Recommendations API
  - Spotify Web API wird NUR fuer Playback-Steuerung genutzt (play/pause/skip)
  - Track-Index: /mnt/moloch-data/memory/spotify/track_index.json

Features:
  - Auto-DJ: Wechselt automatisch Musik bei Zone-Wechsel (Guardian/Shadow/Berserker)
  - Zeitbasiert: Morgens ruhiger, abends intensiver, nachts dunkel
  - spotifyd Health Check: Automatischer Restart wenn down
  - API Retry: Token-Refresh, Device-Recovery, Netzwerk-Retry

Singleton: get_spotify() -> globale Instanz

Befehle:
  play() / pause() / toggle() / next_track() / previous_track()
  set_volume(0-100) / get_current_track() / shuffle(on/off)
  search_and_play(query) / play_by_mood(zone)
  play_artist(name) / play_similar()
  play_top_tracks() / play_new_music()
  auto_dj_start() / auto_dj_stop() / auto_dj_toggle()

Credentials: /home/molochzuhause/moloch/.env.spotify
"""

import os
import json
import logging
import subprocess
import threading
import time
import random
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger("MolochSpotify")

# Pfade
_ENV_PATH = os.path.expanduser("~/moloch/.env.spotify")
_TOKEN_CACHE = os.path.expanduser("~/.cache/spotipy/.cache")
_PROFILE_PATH = "/mnt/moloch-data/memory/spotify/spotify_profile.json"
_TRACK_INDEX_PATH = "/mnt/moloch-data/memory/spotify/track_index.json"

# =========================================================================
# Zone-Kuenstler Mapping (aus Markus' Spotify-Profil, 6833 Stunden)
# =========================================================================

ZONE_ARTISTS = {
    "guardian": [
        # Ruhigere, atmosphaerische Tracks — Futurepop, Synthwave, Dark Ambient
        "VNV Nation", "Depeche Mode", "[:SITD:]", "KANGA",
        "Apoptygma Berzerk", "Diva Destruction", "Daniel Deluxe",
        "Perturbator", "S U R V I V E", "Carpenter Brut",
        "Assemblage 23", "Mesh", "De/Vision", "Covenant",
        "Geistform", "Haujobb", "Front Line Assembly", "SIERRA",
    ],
    "shadow": [
        # Dunkle, treibende Musik — Dark Electro, EBM, Industrial Techno
        "Phase Fatale", "Ancient Methods", "I Hate Models",
        "Schwefelgelb", ":Wumpscut:", "Vomito Negro", "ESA",
        "Alien Vampires", "Leather Strip",
        "Hocico", "Velvet Acid Christ", "Nitzer Ebb",
        "DAF", "Front 242", "Skinny Puppy",
    ],
    "berserker": [
        # Hart, aggressiv, laut — Aggrotech, Power Noise, Industrial Metal
        "Suicide Commando", "Combichrist", "Ministry", "Chainreactor",
        "16Volt", "Prong", "Terrorfakt", "Xotox",
        "FabrikC", "Orange Sector", "Ambassador21",
        "Feindflug", "Agonoize", "Funker Vogt",
        "KMFDM", "Rammstein", "Eisbrecher",
    ],
}

# =========================================================================
# Zeitbasierte Modifikatoren
# =========================================================================

def _get_time_zone() -> str:
    """Tageszeit -> Stimmungs-Modifikator.

    Returns:
        "morgen" (06-10), "tag" (10-17), "abend" (17-22), "nacht" (22-06)
    """
    hour = datetime.now().hour
    if 6 <= hour < 10:
        return "morgen"
    elif 10 <= hour < 17:
        return "tag"
    elif 17 <= hour < 22:
        return "abend"
    else:
        return "nacht"


# Zeitbasierte Artist-Gewichtung: Welche Artists pro Tageszeit bevorzugt
TIME_ZONE_WEIGHTS = {
    "morgen": {
        # Morgens ruhiger Einstieg — atmosphaerische Kuenstler bevorzugen
        "guardian": 1.5,   # Futurepop/Synthwave bevorzugt
        "shadow": 0.5,     # Weniger dunkel morgens
        "berserker": 0.2,  # Fast nie morgens
    },
    "tag": {
        "guardian": 1.0,
        "shadow": 1.0,
        "berserker": 0.8,
    },
    "abend": {
        # Abends intensiver
        "guardian": 0.7,
        "shadow": 1.5,   # Dark Electro abends
        "berserker": 1.2,
    },
    "nacht": {
        # Nachtschicht: Dunkler Mix
        "guardian": 0.5,
        "shadow": 1.8,    # Maximum dunkel
        "berserker": 1.0,
    },
}


def _load_env():
    """Lade Spotify Credentials aus .env.spotify"""
    if not os.path.exists(_ENV_PATH):
        logger.warning(f"[SPOTIFY] .env.spotify nicht gefunden: {_ENV_PATH}")
        return False

    with open(_ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

    return all(os.environ.get(k) for k in [
        "SPOTIPY_CLIENT_ID", "SPOTIPY_CLIENT_SECRET", "SPOTIPY_REDIRECT_URI"
    ])


class SpotifyController:
    """
    Spotify Web API Controller fuer M.O.L.O.C.H.

    Nutzt spotipy fuer Playback-Kontrolle.
    Thread-safe. Lazy Auth (erst bei erstem Befehl).

    Features:
      - Auto-DJ: Ueberwacht Core State, wechselt Musik bei Zone-Aenderung
      - Genre-Lock: Filtert Suchergebnisse gegen erlaubte Genres
      - Zeitbasiert: Morgens ruhiger, abends intensiver
      - Smart Requests: Jahresfilter, Aehnliches, Top Tracks, Neue Musik
    """

    # Retry-Konfiguration
    _AUTH_RETRY_COOLDOWN = 30.0   # Sekunden bis naechster Auth-Versuch
    _MAX_API_RETRIES = 2          # Max Retries pro API-Call
    _SPOTIFYD_CHECK_INTERVAL = 60 # Sekunden zwischen spotifyd-Checks

    def __init__(self):
        self._sp = None
        self._auth_manager = None
        self._lock = threading.Lock()
        self._device_id = None
        self._profile = None
        self._initialized = False
        self._last_auth_attempt = 0.0  # monotonic, kein permanentes Aufgeben

        # Auto-DJ State
        self._auto_dj_active = False
        self._auto_dj_thread = None
        self._auto_dj_zone = None       # Letzte Zone fuer die Musik gewaehlt wurde
        self._auto_dj_lock = threading.Lock()

        # Profil-Cache (geladene Top-Artists fuer schnelle Suche)
        self._profile_artists = {}      # {artist_name_lower: {rank, plays, genre}}
        self._profile_top_tracks = []   # [{name, artist, uri, plays}]

        # Track-Index: {artist_lower -> [{name, artist, uri, plays}]}
        # Gebaut aus 151K Streaming History — EINZIGE Quelle fuer Musikauswahl
        self._track_index = {}

        # spotifyd Health Check
        self._last_spotifyd_check = 0.0

    def _ensure_auth(self) -> bool:
        """Lazy Authentication mit Retry-Cooldown (kein permanentes Aufgeben)."""
        if self._sp is not None:
            return True

        # Cooldown: Nicht staendig retrien
        now = time.monotonic()
        if now - self._last_auth_attempt < self._AUTH_RETRY_COOLDOWN:
            return False

        with self._lock:
            if self._sp is not None:
                return True

            self._last_auth_attempt = now

            if not _load_env():
                logger.error("[SPOTIFY] Credentials nicht geladen")
                return False

            try:
                import spotipy
                from spotipy.oauth2 import SpotifyOAuth

                scope = (
                    "user-read-playback-state "
                    "user-modify-playback-state "
                    "user-read-currently-playing "
                    "playlist-read-private "
                    "user-library-read "
                    "user-top-read"
                )

                os.makedirs(os.path.dirname(_TOKEN_CACHE), exist_ok=True)

                self._auth_manager = SpotifyOAuth(
                    scope=scope,
                    cache_path=_TOKEN_CACHE,
                    open_browser=False,
                )

                # Token pruefen — spotipy refresht automatisch wenn abgelaufen
                token_info = self._auth_manager.get_cached_token()
                if not token_info:
                    logger.error("[SPOTIFY] Kein Token gecached — spotify_auth.py nochmal ausfuehren!")
                    return False

                self._sp = spotipy.Spotify(auth_manager=self._auth_manager)

                # Device ID von MOLOCH finden
                self._find_device()

                # Profil laden
                self._load_profile()

                self._initialized = True
                logger.info("[SPOTIFY] Authentifiziert und bereit")
                return True

            except Exception as e:
                logger.error(f"[SPOTIFY] Auth fehlgeschlagen: {e}")
                self._sp = None
                self._auth_manager = None
                return False

    def _api_call(self, func, *args, **kwargs):
        """Wrapper fuer Spotify API Calls mit Retry und Device-Recovery.

        Fängt Token-Ablauf, Device-Verlust und Netzwerkfehler ab.
        Returnt das Ergebnis oder raised Exception nach allen Retries.
        """
        last_err = None
        for attempt in range(self._MAX_API_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_err = e
                err_str = str(e).lower()

                # Token abgelaufen — spotipy sollte autorefreshen,
                # aber manchmal braucht es einen Kick
                if "token" in err_str or "401" in err_str or "expired" in err_str:
                    logger.warning(f"[SPOTIFY] Token-Problem (Versuch {attempt + 1}): {e}")
                    self._force_token_refresh()
                    continue

                # 403 Restriction (z.B. Shuffle ohne aktiven Context) — nicht retrien
                if "403" in err_str or "restriction" in err_str:
                    raise

                # Device nicht gefunden — neu suchen
                if "no active device" in err_str or "404" in err_str or "not found" in err_str:
                    logger.warning(f"[SPOTIFY] Device verloren (Versuch {attempt + 1}): {e}")
                    self._ensure_spotifyd()
                    time.sleep(2)
                    self._find_device()
                    continue

                # Netzwerk/Server-Fehler — kurz warten und retrien
                if "connection" in err_str or "timeout" in err_str or "50" in err_str:
                    logger.warning(f"[SPOTIFY] Netzwerk-Fehler (Versuch {attempt + 1}): {e}")
                    time.sleep(2)
                    continue

                # Unbekannter Fehler — nicht retrien
                raise

        raise last_err

    def _force_token_refresh(self):
        """Erzwingt Token-Refresh ueber auth_manager."""
        try:
            if self._auth_manager:
                token_info = self._auth_manager.get_cached_token()
                if token_info:
                    # Refresh erzwingen indem wir den Token als abgelaufen markieren
                    if self._auth_manager.is_token_expired(token_info):
                        new_token = self._auth_manager.refresh_access_token(
                            token_info["refresh_token"]
                        )
                        logger.info("[SPOTIFY] Token erfolgreich refreshed")
        except Exception as e:
            logger.error(f"[SPOTIFY] Token-Refresh fehlgeschlagen: {e}")
            # Auth komplett neu aufbauen
            self._sp = None
            self._auth_manager = None
            self._initialized = False

    def _ensure_spotifyd(self):
        """Prueft ob spotifyd laeuft und startet ihn neu wenn noetig."""
        now = time.monotonic()
        if now - self._last_spotifyd_check < self._SPOTIFYD_CHECK_INTERVAL:
            return
        self._last_spotifyd_check = now

        try:
            result = subprocess.run(
                ["systemctl", "is-active", "spotifyd"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip() == "active":
                return

            logger.warning("[SPOTIFY] spotifyd ist down — starte neu...")
            subprocess.run(
                ["sudo", "systemctl", "restart", "spotifyd"],
                capture_output=True, timeout=15,
            )
            # Warten bis spotifyd sich bei Spotify registriert hat
            time.sleep(5)
            logger.info("[SPOTIFY] spotifyd neugestartet")
        except Exception as e:
            logger.error(f"[SPOTIFY] spotifyd Health Check fehlgeschlagen: {e}")

    def _find_device(self):
        """Finde das MOLOCH Geraet (spotifyd)."""
        try:
            devices = self._sp.devices()
            for d in devices.get("devices", []):
                if "MOLOCH" in d.get("name", "").upper():
                    self._device_id = d["id"]
                    logger.info(f"[SPOTIFY] MOLOCH Device gefunden: {d['name']} ({d['id'][:8]}...)")
                    return
            # Kein MOLOCH-Device? Erstes aktives nehmen
            active = [d for d in devices.get("devices", []) if d.get("is_active")]
            if active:
                self._device_id = active[0]["id"]
                logger.info(f"[SPOTIFY] Aktives Device: {active[0]['name']}")
            elif devices.get("devices"):
                self._device_id = devices["devices"][0]["id"]
                logger.info(f"[SPOTIFY] Erstes Device: {devices['devices'][0]['name']}")
            else:
                logger.warning("[SPOTIFY] Kein Geraet gefunden! spotifyd laeuft?")
        except Exception as e:
            logger.error(f"[SPOTIFY] Device-Suche fehlgeschlagen: {e}")

    def _load_profile(self):
        """Lade Spotify-Profil + Track-Index (151K Tracks mit URIs)."""
        try:
            if os.path.exists(_PROFILE_PATH):
                with open(_PROFILE_PATH) as f:
                    self._profile = json.load(f)

                # Artist-Lookup Cache aufbauen
                for a in self._profile.get("top_artists", []):
                    self._profile_artists[a["name"].lower()] = {
                        "rank": a.get("rank", 999),
                        "plays": a.get("plays", 0),
                        "genre": a.get("genre", ""),
                        "name": a["name"],
                    }

                # Top Tracks Cache
                self._profile_top_tracks = self._profile.get("top_tracks", [])

                logger.info(f"[SPOTIFY] Profil geladen: {len(self._profile_artists)} Artists")
        except Exception as e:
            logger.debug(f"[SPOTIFY] Profil laden fehlgeschlagen: {e}")

        # Track-Index laden (EINZIGE Quelle fuer Musikauswahl)
        try:
            if os.path.exists(_TRACK_INDEX_PATH):
                with open(_TRACK_INDEX_PATH) as f:
                    self._track_index = json.load(f)
                total_tracks = sum(len(v) for v in self._track_index.values())
                logger.info(f"[SPOTIFY] Track-Index geladen: {len(self._track_index)} Artists, "
                            f"{total_tracks} Tracks")
            else:
                logger.warning(f"[SPOTIFY] Track-Index nicht gefunden: {_TRACK_INDEX_PATH}")
        except Exception as e:
            logger.error(f"[SPOTIFY] Track-Index laden fehlgeschlagen: {e}")

    def _get_artist_tracks(self, artist_name: str, limit: int = 20) -> List[str]:
        """Hole Track-URIs fuer einen Artist aus dem lokalen Index.

        KEINE Spotify-Suche — alles aus Markus' 151K Streaming History.
        Sortiert nach Plays (meistgehoert zuerst).
        """
        key = artist_name.lower()
        # Exakter Match
        tracks = self._track_index.get(key, [])
        if tracks:
            return [t["uri"] for t in tracks[:limit]]

        # Fuzzy: Teilstring-Match (z.B. "ESA" -> "esa (electronic substance abuse)")
        for idx_key, idx_tracks in self._track_index.items():
            if key in idx_key or idx_key in key:
                return [t["uri"] for t in idx_tracks[:limit]]

        return []

    def _get_zone_uris(self, zone: str, count: int = 20) -> List[str]:
        """Sammle Track-URIs fuer eine Zone aus dem lokalen Index.

        Nimmt Artists aus ZONE_ARTISTS, holt deren Tracks aus dem Index.
        Mischt die Tracks und gibt 'count' URIs zurueck.
        """
        artists = list(ZONE_ARTISTS.get(zone, ZONE_ARTISTS["shadow"]))
        random.shuffle(artists)

        uris = []
        for artist_name in artists:
            artist_uris = self._get_artist_tracks(artist_name, limit=10)
            uris.extend(artist_uris)

        random.shuffle(uris)
        return uris[:count]

    def _refresh_device(self):
        """Device-ID auffrischen: sucht neu wenn verloren, checkt spotifyd."""
        if not self._device_id:
            self._ensure_spotifyd()
            self._find_device()

    # =========================================================================
    # PLAYBACK CONTROLS (alle mit _api_call Retry-Wrapper)
    # =========================================================================

    def play(self, uri: str = None, context_uri: str = None) -> bool:
        """Playback starten. Optional: URI eines Tracks/Albums/Playlists."""
        if not self._ensure_auth():
            return False
        try:
            self._refresh_device()
            kwargs = {}
            if self._device_id:
                kwargs["device_id"] = self._device_id
            if uri:
                kwargs["uris"] = [uri] if isinstance(uri, str) else uri
            if context_uri:
                kwargs["context_uri"] = context_uri

            self._api_call(self._sp.start_playback, **kwargs)
            logger.info("[SPOTIFY] Play")
            return True
        except Exception as e:
            logger.error(f"[SPOTIFY] Play fehlgeschlagen: {e}")
            # Device verloren? Beim naechsten Versuch neu suchen
            self._device_id = None
            return False

    def pause(self) -> bool:
        """Playback pausieren."""
        if not self._ensure_auth():
            return False
        try:
            self._api_call(self._sp.pause_playback, device_id=self._device_id)
            logger.info("[SPOTIFY] Pause")
            return True
        except Exception as e:
            logger.error(f"[SPOTIFY] Pause fehlgeschlagen: {e}")
            return False

    def toggle(self) -> bool:
        """Play/Pause umschalten."""
        if not self._ensure_auth():
            return False
        try:
            current = self._api_call(self._sp.current_playback)
            if current and current.get("is_playing"):
                return self.pause()
            else:
                return self.play()
        except Exception as e:
            logger.error(f"[SPOTIFY] Toggle fehlgeschlagen: {e}")
            return False

    def next_track(self) -> bool:
        """Naechster Track."""
        if not self._ensure_auth():
            return False
        try:
            self._api_call(self._sp.next_track, device_id=self._device_id)
            logger.info("[SPOTIFY] Skip >>")
            return True
        except Exception as e:
            logger.error(f"[SPOTIFY] Skip fehlgeschlagen: {e}")
            return False

    def previous_track(self) -> bool:
        """Vorheriger Track."""
        if not self._ensure_auth():
            return False
        try:
            self._api_call(self._sp.previous_track, device_id=self._device_id)
            logger.info("[SPOTIFY] Skip <<")
            return True
        except Exception as e:
            logger.error(f"[SPOTIFY] Previous fehlgeschlagen: {e}")
            return False

    def set_volume(self, volume_pct: int) -> bool:
        """Volume setzen (0-100)."""
        if not self._ensure_auth():
            return False
        try:
            vol = max(0, min(100, int(volume_pct)))
            self._api_call(self._sp.volume, vol, device_id=self._device_id)
            logger.info(f"[SPOTIFY] Volume: {vol}%")
            return True
        except Exception as e:
            logger.error(f"[SPOTIFY] Volume fehlgeschlagen: {e}")
            return False

    def shuffle(self, state: bool) -> bool:
        """Shuffle ein/ausschalten."""
        if not self._ensure_auth():
            return False
        try:
            self._api_call(self._sp.shuffle, state, device_id=self._device_id)
            logger.info(f"[SPOTIFY] Shuffle: {'AN' if state else 'AUS'}")
            return True
        except Exception as e:
            logger.error(f"[SPOTIFY] Shuffle fehlgeschlagen: {e}")
            return False

    # =========================================================================
    # TRACK INFO
    # =========================================================================

    def get_current_track(self) -> Optional[Dict[str, Any]]:
        """Aktuellen Track holen."""
        if not self._ensure_auth():
            return None
        try:
            current = self._api_call(self._sp.current_playback)
            if not current or not current.get("item"):
                return None

            item = current["item"]
            artists = ", ".join(a["name"] for a in item.get("artists", []))
            return {
                "artist": artists,
                "track": item.get("name", "?"),
                "album": item.get("album", {}).get("name", "?"),
                "is_playing": current.get("is_playing", False),
                "progress_ms": current.get("progress_ms", 0),
                "duration_ms": item.get("duration_ms", 0),
                "uri": item.get("uri", ""),
                "device": current.get("device", {}).get("name", "?"),
                "shuffle": current.get("shuffle_state", False),
                "volume": current.get("device", {}).get("volume_percent", 0),
            }
        except Exception as e:
            logger.error(f"[SPOTIFY] Current Track fehlgeschlagen: {e}")
            return None

    def get_current_track_str(self) -> str:
        """Aktuellen Track als lesbaren String."""
        t = self.get_current_track()
        if not t:
            return "Nichts laeuft gerade"
        status = "spielt" if t["is_playing"] else "pausiert"
        progress = t["progress_ms"] // 1000
        duration = t["duration_ms"] // 1000
        return f"{t['artist']} - {t['track']} [{progress // 60}:{progress % 60:02d}/{duration // 60}:{duration % 60:02d}] ({status})"

    # =========================================================================
    # MUSIK-AUSWAHL (NUR aus lokalem Track-Index, KEINE Spotify-Suche)
    # =========================================================================

    def search_and_play(self, query: str) -> bool:
        """Suche im lokalen Index und spiele.

        Sucht in Artist-Namen und Track-Namen. KEINE Spotify-API-Suche.
        """
        if not self._ensure_auth():
            return False
        if not self._track_index:
            logger.warning("[SPOTIFY] Kein Track-Index geladen")
            return False

        query_lower = query.lower()
        uris = []

        # 1. Artist-Match: Suche im Index nach Artist-Namen
        for artist_key, tracks in self._track_index.items():
            if query_lower in artist_key or artist_key in query_lower:
                uris.extend(t["uri"] for t in tracks[:15])
                if len(uris) >= 20:
                    break

        # 2. Track-Match: Suche in Track-Namen
        if not uris:
            for tracks in self._track_index.values():
                for t in tracks:
                    if query_lower in t["name"].lower():
                        uris.append(t["uri"])
                        if len(uris) >= 20:
                            break
                if len(uris) >= 20:
                    break

        if not uris:
            logger.warning(f"[SPOTIFY] Nichts im Index fuer: {query}")
            return False

        random.shuffle(uris)
        logger.info(f"[SPOTIFY] {len(uris)} Tracks aus Index fuer '{query}'")
        return self.play(uri=uris[:20])

    def play_artist(self, artist_name: str) -> bool:
        """Spiele Tracks eines Artists aus dem lokalen Index.

        KEINE Spotify-Suche — nur Tracks die Markus tatsaechlich gehoert hat.
        """
        if not self._ensure_auth():
            return False

        uris = self._get_artist_tracks(artist_name, limit=20)
        if not uris:
            logger.debug(f"[SPOTIFY] '{artist_name}' nicht im Index")
            return False

        random.shuffle(uris)  # Python-Shuffle statt Spotify-API (vermeidet 403)
        logger.info(f"[SPOTIFY] {len(uris)} Tracks von {artist_name} aus Index")
        return self.play(uri=uris)

    # =========================================================================
    # MOOD / ZONE INTEGRATION (mit Zeitmodifikator, NUR lokaler Index)
    # =========================================================================

    def play_by_mood(self, zone: str) -> bool:
        """Spiele Musik passend zur Zone + Tageszeit.

        Alle Tracks kommen aus Markus' Streaming History (lokaler Index).
        KEINE Spotify-Suche, KEINE Recommendations API.
        """
        if not self._ensure_auth():
            return False
        if not self._track_index:
            logger.warning("[SPOTIFY] Kein Track-Index geladen")
            return False

        time_zone = _get_time_zone()
        weights = TIME_ZONE_WEIGHTS.get(time_zone, TIME_ZONE_WEIGHTS["tag"])
        zone_weight = weights.get(zone, 1.0)

        # Zeitbasierte Zone-Verschiebung
        effective_zone = zone
        if zone_weight < 0.3:
            effective_zone = "guardian"
            logger.info(f"[SPOTIFY] Zone '{zone}' zu intensiv fuer {time_zone}, "
                        f"verwende 'guardian'")

        # URIs aus dem lokalen Index holen
        uris = self._get_zone_uris(effective_zone, count=30)

        # Zeitbasierte Beimischung
        if time_zone == "morgen" and effective_zone != "guardian":
            guardian_uris = self._get_zone_uris("guardian", count=10)
            uris = guardian_uris[:5] + uris
        elif time_zone == "nacht" and effective_zone == "guardian":
            shadow_uris = self._get_zone_uris("shadow", count=10)
            uris = uris + shadow_uris[:5]

        if not uris:
            logger.warning(f"[SPOTIFY] Keine Tracks im Index fuer Zone '{zone}'")
            return False

        random.shuffle(uris)
        uris = uris[:20]
        logger.info(f"[SPOTIFY] Mood '{effective_zone}' ({time_zone}): {len(uris)} Tracks aus Index")
        return self.play(uri=uris)

    def get_recommendations_for_zone(self, zone: str, limit: int = 10) -> List[str]:
        """Empfehlungen aus dem lokalen Index (KEINE Spotify Recommendations API)."""
        artists = ZONE_ARTISTS.get(zone, ZONE_ARTISTS["shadow"])
        result = []
        for artist_name in artists:
            tracks = self._track_index.get(artist_name.lower(), [])
            for t in tracks[:3]:
                result.append(f"{t['artist']} - {t['name']}")
                if len(result) >= limit:
                    return result
        return result

    # =========================================================================
    # SMART REQUESTS (alle aus lokalem Index)
    # =========================================================================

    def play_similar(self) -> bool:
        """Spiele aehnliche Musik zum aktuell laufenden Track.

        Findet den Artist im lokalen Index und spielt andere Tracks.
        KEINE Spotify Recommendations API.
        """
        if not self._ensure_auth():
            return False
        if not self._track_index:
            return False

        try:
            current = self._api_call(self._sp.current_playback)
            if not current or not current.get("item"):
                logger.warning("[SPOTIFY] Kein Track laeuft")
                return False

            item = current["item"]
            current_artist = item["artists"][0]["name"].lower()

            uris = []
            # 1. Andere Tracks vom selben Artist
            artist_uris = self._get_artist_tracks(item["artists"][0]["name"], limit=30)
            # Aktuellen Track rausfiltern
            current_uri = item.get("uri", "")
            uris.extend(u for u in artist_uris if u != current_uri)

            # 2. Aus der gleichen Zone andere Artists beimischen
            for zone_name, zone_artists in ZONE_ARTISTS.items():
                if current_artist in [a.lower() for a in zone_artists]:
                    zone_uris = self._get_zone_uris(zone_name, count=20)
                    uris.extend(u for u in zone_uris if u != current_uri)
                    break

            if not uris:
                logger.warning("[SPOTIFY] Keine aehnlichen Tracks im Index")
                return False

            random.shuffle(uris)
            uris = list(dict.fromkeys(uris))[:20]  # Duplikate entfernen
            logger.info(f"[SPOTIFY] {len(uris)} aehnliche Tracks zu "
                        f"'{item['artists'][0]['name']} - {item['name']}'")
            return self.play(uri=uris)

        except Exception as e:
            logger.error(f"[SPOTIFY] Similar fehlgeschlagen: {e}")
            return False

    def play_top_tracks(self) -> bool:
        """Spiele Markus' meistgehoerte Tracks aus dem lokalen Index."""
        if not self._ensure_auth():
            return False
        if not self._track_index:
            return False

        # Top Tracks aus dem Index: die mit den meisten Plays
        all_tracks = []
        for tracks in self._track_index.values():
            all_tracks.extend(tracks)

        all_tracks.sort(key=lambda t: t["plays"], reverse=True)
        uris = [t["uri"] for t in all_tracks[:50]]

        if not uris:
            logger.warning("[SPOTIFY] Keine Top Tracks im Index")
            return False

        random.shuffle(uris)
        uris = uris[:20]
        logger.info(f"[SPOTIFY] {len(uris)} Top Tracks aus Index")
        return self.play(uri=uris)

    def play_new_music(self) -> bool:
        """Spiele weniger gehoerte Tracks aus dem Index.

        Statt Spotify Recommendations: Tracks die Markus selten gehoert hat
        (1-5 Plays) von Artists die er sonst viel hoert.
        """
        if not self._ensure_auth():
            return False
        if not self._track_index:
            return False

        # Top-30 Artists nach Gesamtplays, aber deren SELTENSTE Tracks
        artist_plays = []
        for artist_key, tracks in self._track_index.items():
            total = sum(t["plays"] for t in tracks)
            artist_plays.append((artist_key, total, tracks))

        artist_plays.sort(key=lambda x: x[1], reverse=True)

        uris = []
        for _, _, tracks in artist_plays[:50]:
            # Tracks mit 1-5 Plays (selten gehoert, aber vorhanden)
            rare = [t for t in tracks if 1 <= t["plays"] <= 5]
            if rare:
                random.shuffle(rare)
                uris.extend(t["uri"] for t in rare[:3])

        if not uris:
            logger.warning("[SPOTIFY] Keine seltenen Tracks im Index")
            return False

        random.shuffle(uris)
        uris = uris[:20]
        logger.info(f"[SPOTIFY] {len(uris)} selten gehoerte Tracks entdeckt")
        return self.play(uri=uris)

    def play_from_year(self, year: int) -> bool:
        """Kann ohne Jahr-Info im Index nicht filtern — spielt stattdessen Top Tracks."""
        logger.info(f"[SPOTIFY] Jahr-Filter nicht verfuegbar im Index, spiele Top Tracks")
        return self.play_top_tracks()

    # =========================================================================
    # AUTO-DJ: Automatischer Zone-Wechsel
    # =========================================================================

    def auto_dj_start(self) -> bool:
        """Auto-DJ starten — ueberwacht Core State, wechselt Musik bei Zone-Aenderung."""
        with self._auto_dj_lock:
            if self._auto_dj_active:
                logger.info("[AUTO-DJ] Laeuft bereits")
                return True

            self._auto_dj_active = True
            self._auto_dj_thread = threading.Thread(
                target=self._auto_dj_loop,
                daemon=True,
                name="SpotifyAutoDJ",
            )
            self._auto_dj_thread.start()
            logger.info("[AUTO-DJ] Gestartet")

            # Memory-Fakt speichern
            try:
                from core.longterm_memory import get_memory
                get_memory().add_fact(
                    "spotify_auto_dj",
                    "Markus nutzt Auto-DJ: Musik wechselt automatisch mit Personality-Zone",
                    source="system",
                )
            except Exception:
                pass

            return True

    def auto_dj_stop(self):
        """Auto-DJ stoppen."""
        with self._auto_dj_lock:
            if not self._auto_dj_active:
                return
            self._auto_dj_active = False
            logger.info("[AUTO-DJ] Gestoppt")

    def auto_dj_toggle(self) -> bool:
        """Auto-DJ ein/ausschalten. Gibt neuen State zurueck."""
        if self._auto_dj_active:
            self.auto_dj_stop()
            return False
        else:
            self.auto_dj_start()
            return True

    def _auto_dj_loop(self):
        """Auto-DJ Hauptschleife — laeuft als Daemon-Thread.

        Liest Core State alle 5 Sekunden. Bei Zone-Wechsel -> neue Musik.
        Prueft spotifyd alle 60 Sekunden.
        """
        logger.info("[AUTO-DJ] Loop gestartet")
        health_counter = 0

        while self._auto_dj_active:
            try:
                # spotifyd Health Check alle 12 Zyklen (~60 Sekunden)
                health_counter += 1
                if health_counter >= 12:
                    health_counter = 0
                    self._ensure_spotifyd()

                # Core Integrator Zone lesen
                zone = self._get_current_zone()
                if zone and zone != self._auto_dj_zone:
                    old_zone = self._auto_dj_zone
                    self._auto_dj_zone = zone
                    if old_zone is not None:
                        logger.info(f"[AUTO-DJ] Zone-Wechsel: {old_zone} -> {zone}")
                        if not self.play_by_mood(zone):
                            # Retry nach Device-Recovery
                            self._device_id = None
                            self._refresh_device()
                            self.play_by_mood(zone)
                    else:
                        logger.info(f"[AUTO-DJ] Initiale Zone: {zone}")
                        self.play_by_mood(zone)
            except Exception as e:
                logger.error(f"[AUTO-DJ] Fehler: {e}")

            # 5 Sekunden warten (in 0.5s Schritten fuer schnelles Shutdown)
            for _ in range(10):
                if not self._auto_dj_active:
                    break
                time.sleep(0.5)

        logger.info("[AUTO-DJ] Loop beendet")

    def _get_current_zone(self) -> Optional[str]:
        """Aktuelle Personality-Zone aus Core Integrator lesen."""
        try:
            from core.core_integrator import get_core_integrator
            ci = get_core_integrator()
            return ci.get_personality_zone()
        except Exception:
            return None

    # =========================================================================
    # STATUS (fuer Panel / IPC)
    # =========================================================================

    def is_available(self) -> bool:
        """Pruefe ob Spotify verfuegbar ist (Credentials + Device)."""
        if not self._ensure_auth():
            return False
        return self._device_id is not None

    def get_devices(self) -> List[Dict]:
        """Liste aller verfuegbaren Geraete."""
        if not self._ensure_auth():
            return []
        try:
            return self._api_call(self._sp.devices).get("devices", [])
        except Exception:
            return []

    def get_status(self) -> Dict[str, Any]:
        """Gesamtstatus fuer Panel/IPC."""
        track = self.get_current_track() if self._initialized else None

        # spotifyd Status pruefen
        spotifyd_ok = False
        try:
            result = subprocess.run(
                ["systemctl", "is-active", "spotifyd"],
                capture_output=True, text=True, timeout=3,
            )
            spotifyd_ok = result.stdout.strip() == "active"
        except Exception:
            pass

        return {
            "initialized": self._initialized,
            "auth_ok": self._sp is not None,
            "spotifyd_ok": spotifyd_ok,
            "device_id": self._device_id[:8] + "..." if self._device_id else None,
            "auto_dj": self._auto_dj_active,
            "auto_dj_zone": self._auto_dj_zone,
            "current_track": track,
            "current_track_str": self.get_current_track_str() if self._initialized else "nicht initialisiert",
        }


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[SpotifyController] = None
_instance_lock = threading.Lock()


def get_spotify() -> SpotifyController:
    """Globale SpotifyController Instanz."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = SpotifyController()
    return _instance
