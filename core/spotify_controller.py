#!/usr/bin/env python3
"""
M.O.L.O.C.H. Spotify Controller
=================================

Steuert Spotify ueber die Web API (spotipy).
Integriert sich in Molochs Zone-System fuer stimmungsbasierte Musik.

Singleton: get_spotify() -> globale Instanz

Befehle:
  play() / pause() / toggle() / next_track() / previous_track()
  set_volume(0-100) / get_current_track()
  search_and_play(query) / play_by_mood(zone)
  play_artist(name) / play_playlist(name)

Credentials: ~/.env.spotify oder /home/molochzuhause/moloch/.env.spotify
"""

import os
import json
import logging
import threading
import time
from typing import Optional, Dict, Any, List

logger = logging.getLogger("MolochSpotify")

# Pfade
_ENV_PATH = os.path.expanduser("~/moloch/.env.spotify")
_TOKEN_CACHE = os.path.expanduser("~/.cache/spotipy/.cache")
_PROFILE_PATH = "/mnt/moloch-data/memory/spotify/spotify_profile.json"

# Mood-Playlists basierend auf Markus' Spotify-Profil
# Jede Zone hat Artists/Genres die dazu passen
ZONE_ARTISTS = {
    "guardian": [
        # Ruhigere, atmosphaerische Tracks
        "VNV Nation", "Depeche Mode", "[:SITD:]", "KANGA",
        "Apoptygma Berzerk", "Diva Destruction", "Daniel Deluxe",
        "Perturbator", "S U R V I V E", "Carpenter Brut",
    ],
    "shadow": [
        # Dunkle, treibende Musik
        "Suicide Commando", "Vomito Negro", "ESA",
        ":Wumpscut:", "Alien Vampires", "Combichrist",
        "Phase Fatale", "Ancient Methods", "I Hate Models",
        "Schwefelgelb", "Chainreactor",
    ],
    "berserker": [
        # Hart, aggressiv, laut
        "AC/DC", "Ministry", "16Volt", "Prong",
        "Terrorfakt", "Xotox", "FabrikC", "Orange Sector",
        "Combichrist", "Ambassador21", "Airbourne",
    ],
}

# Search-Queries fuer Moods (Fallback wenn kein Artist Match)
ZONE_SEARCH = {
    "guardian": "futurepop synthwave dark ambient",
    "shadow": "dark electro EBM industrial",
    "berserker": "aggrotech power noise industrial metal",
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
    """

    def __init__(self):
        self._sp = None
        self._lock = threading.Lock()
        self._device_id = None
        self._profile = None
        self._initialized = False
        self._auth_failed = False

    def _ensure_auth(self) -> bool:
        """Lazy Authentication — erst wenn gebraucht."""
        if self._auth_failed:
            return False
        if self._sp is not None:
            return True

        with self._lock:
            if self._sp is not None:
                return True

            if not _load_env():
                logger.error("[SPOTIFY] Credentials nicht geladen")
                self._auth_failed = True
                return False

            try:
                import spotipy
                from spotipy.oauth2 import SpotifyOAuth

                # OAuth mit allen relevanten Scopes
                scope = (
                    "user-read-playback-state "
                    "user-modify-playback-state "
                    "user-read-currently-playing "
                    "playlist-read-private "
                    "user-library-read "
                    "user-top-read"
                )

                os.makedirs(os.path.dirname(_TOKEN_CACHE), exist_ok=True)

                auth_manager = SpotifyOAuth(
                    scope=scope,
                    cache_path=_TOKEN_CACHE,
                    open_browser=False,
                )

                self._sp = spotipy.Spotify(auth_manager=auth_manager)

                # Device ID von MOLOCH finden
                self._find_device()

                # Profil laden
                self._load_profile()

                self._initialized = True
                logger.info("[SPOTIFY] Authentifiziert und bereit")
                return True

            except Exception as e:
                logger.error(f"[SPOTIFY] Auth fehlgeschlagen: {e}")
                self._auth_failed = True
                return False

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
        """Lade Spotify-Profil fuer Empfehlungen."""
        try:
            if os.path.exists(_PROFILE_PATH):
                with open(_PROFILE_PATH) as f:
                    self._profile = json.load(f)
                logger.info("[SPOTIFY] Profil geladen")
        except Exception as e:
            logger.debug(f"[SPOTIFY] Profil laden fehlgeschlagen: {e}")

    def _refresh_device(self):
        """Device-ID auffrischen falls verloren."""
        if not self._device_id:
            self._find_device()

    # === PLAYBACK CONTROLS ===

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

            self._sp.start_playback(**kwargs)
            logger.info("[SPOTIFY] Play")
            return True
        except Exception as e:
            logger.error(f"[SPOTIFY] Play fehlgeschlagen: {e}")
            return False

    def pause(self) -> bool:
        """Playback pausieren."""
        if not self._ensure_auth():
            return False
        try:
            self._sp.pause_playback(device_id=self._device_id)
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
            current = self._sp.current_playback()
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
            self._sp.next_track(device_id=self._device_id)
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
            self._sp.previous_track(device_id=self._device_id)
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
            self._sp.volume(vol, device_id=self._device_id)
            logger.info(f"[SPOTIFY] Volume: {vol}%")
            return True
        except Exception as e:
            logger.error(f"[SPOTIFY] Volume fehlgeschlagen: {e}")
            return False

    # === TRACK INFO ===

    def get_current_track(self) -> Optional[Dict[str, Any]]:
        """Aktuellen Track holen. Gibt Dict mit artist, track, album, progress, duration."""
        if not self._ensure_auth():
            return None
        try:
            current = self._sp.current_playback()
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

    # === SEARCH & PLAY ===

    def search_and_play(self, query: str) -> bool:
        """Suche und spiele den ersten Treffer."""
        if not self._ensure_auth():
            return False
        try:
            results = self._sp.search(q=query, limit=1, type="track")
            tracks = results.get("tracks", {}).get("items", [])
            if not tracks:
                logger.warning(f"[SPOTIFY] Keine Treffer fuer: {query}")
                return False

            track = tracks[0]
            artist = track["artists"][0]["name"]
            name = track["name"]
            logger.info(f"[SPOTIFY] Spiele: {artist} - {name}")
            return self.play(uri=track["uri"])
        except Exception as e:
            logger.error(f"[SPOTIFY] Search fehlgeschlagen: {e}")
            return False

    def play_artist(self, artist_name: str) -> bool:
        """Spiele Musik von einem Artist (Top Tracks)."""
        if not self._ensure_auth():
            return False
        try:
            results = self._sp.search(q=f"artist:{artist_name}", limit=1, type="artist")
            artists = results.get("artists", {}).get("items", [])
            if not artists:
                logger.warning(f"[SPOTIFY] Artist nicht gefunden: {artist_name}")
                return False

            artist_uri = artists[0]["uri"]
            # Top Tracks des Artists holen und abspielen
            top = self._sp.artist_top_tracks(artists[0]["id"])
            uris = [t["uri"] for t in top.get("tracks", [])[:10]]
            if uris:
                return self.play(uri=uris)
            return False
        except Exception as e:
            logger.error(f"[SPOTIFY] Artist Play fehlgeschlagen: {e}")
            return False

    # === MOOD / ZONE INTEGRATION ===

    def play_by_mood(self, zone: str) -> bool:
        """Spiele Musik passend zur aktuellen Personality Zone.

        guardian -> Futurepop, Synthwave, atmosphaerisch
        shadow   -> Dark Electro, EBM, Industrial Techno
        berserker -> Aggrotech, Power Noise, Industrial Metal
        """
        if not self._ensure_auth():
            return False

        artists = ZONE_ARTISTS.get(zone, ZONE_ARTISTS["shadow"])
        # Zufaellig einen Artist waehlen und dessen Top Tracks spielen
        import random
        random.shuffle(artists)

        for artist_name in artists[:3]:  # Maximal 3 Versuche
            if self.play_artist(artist_name):
                logger.info(f"[SPOTIFY] Mood '{zone}': Spiele {artist_name}")
                return True

        # Fallback: Genre-Suche
        search = ZONE_SEARCH.get(zone, "dark electro EBM")
        logger.info(f"[SPOTIFY] Mood '{zone}' Fallback: Suche '{search}'")
        return self.search_and_play(search)

    def get_recommendations_for_zone(self, zone: str, limit: int = 5) -> List[str]:
        """Empfehlungen basierend auf Zone und Profil."""
        if not self._ensure_auth():
            return []
        try:
            artists = ZONE_ARTISTS.get(zone, ZONE_ARTISTS["shadow"])[:2]
            # Artist-IDs holen
            seed_ids = []
            for name in artists:
                results = self._sp.search(q=f"artist:{name}", limit=1, type="artist")
                items = results.get("artists", {}).get("items", [])
                if items:
                    seed_ids.append(items[0]["id"])

            if not seed_ids:
                return []

            recs = self._sp.recommendations(seed_artists=seed_ids[:5], limit=limit)
            return [
                f"{t['artists'][0]['name']} - {t['name']}"
                for t in recs.get("tracks", [])
            ]
        except Exception as e:
            logger.error(f"[SPOTIFY] Recommendations fehlgeschlagen: {e}")
            return []

    # === STATUS ===

    def is_available(self) -> bool:
        """Pruefe ob Spotify verfuegbar ist (Credentials + Device)."""
        if self._auth_failed:
            return False
        if not self._ensure_auth():
            return False
        return self._device_id is not None

    def get_devices(self) -> List[Dict]:
        """Liste aller verfuegbaren Geraete."""
        if not self._ensure_auth():
            return []
        try:
            return self._sp.devices().get("devices", [])
        except Exception:
            return []

    def get_status(self) -> Dict[str, Any]:
        """Gesamtstatus fuer Panel/Diagnose."""
        return {
            "initialized": self._initialized,
            "auth_ok": self._sp is not None,
            "auth_failed": self._auth_failed,
            "device_id": self._device_id[:8] + "..." if self._device_id else None,
            "current_track": self.get_current_track_str() if self._initialized else "nicht initialisiert",
        }


# === SINGLETON ===

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
