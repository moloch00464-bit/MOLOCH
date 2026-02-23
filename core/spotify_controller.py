#!/usr/bin/env python3
"""
M.O.L.O.C.H. Spotify Controller
=================================

Steuert Spotify ueber die Web API (spotipy).
Integriert sich in Molochs Zone-System fuer stimmungsbasierte Musik.

Features:
  - Auto-DJ: Wechselt automatisch Musik bei Zone-Wechsel (Guardian/Shadow/Berserker)
  - Genre-Lock: NUR Schwarze Szene, kein Mainstream
  - Zeitbasiert: Morgens ruhiger, abends intensiver, nachts dunkel
  - Smart Requests: Jahres-Filter, Aehnliches, Top Tracks, Neue Musik

Singleton: get_spotify() -> globale Instanz

Befehle:
  play() / pause() / toggle() / next_track() / previous_track()
  set_volume(0-100) / get_current_track() / shuffle(on/off)
  search_and_play(query) / play_by_mood(zone)
  play_artist(name) / play_from_year(year) / play_similar()
  play_top_tracks() / play_new_music()
  auto_dj_start() / auto_dj_stop() / auto_dj_toggle()

Credentials: ~/.env.spotify oder /home/molochzuhause/moloch/.env.spotify
"""

import os
import json
import logging
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
        "Geistform", "Haujobb", "Front Line Assembly",
    ],
    "shadow": [
        # Dunkle, treibende Musik — Dark Electro, EBM, Industrial Techno
        "Suicide Commando", "Vomito Negro", "ESA",
        ":Wumpscut:", "Alien Vampires", "Combichrist",
        "Phase Fatale", "Ancient Methods", "I Hate Models",
        "Schwefelgelb", "Chainreactor", "Leather Strip",
        "Hocico", "Velvet Acid Christ", "Nitzer Ebb",
        "DAF", "Front 242", "Skinny Puppy",
    ],
    "berserker": [
        # Hart, aggressiv, laut — Aggrotech, Power Noise, Industrial Metal
        "Ministry", "16Volt", "Prong",
        "Terrorfakt", "Xotox", "FabrikC", "Orange Sector",
        "Combichrist", "Ambassador21",
        "Feindflug", "Agonoize", "Funker Vogt",
        "KMFDM", "Rammstein", "Eisbrecher",
    ],
}

# Search-Queries fuer Moods (Fallback wenn kein Artist Match)
ZONE_SEARCH = {
    "guardian": "futurepop synthwave dark ambient",
    "shadow": "dark electro EBM industrial",
    "berserker": "aggrotech power noise industrial metal",
}

# =========================================================================
# Genre-Lock: Erlaubte und verbotene Genres
# =========================================================================

# Alles was erlaubt ist (lowercase, Teilstring-Match)
ALLOWED_GENRES = [
    "dark", "ebm", "industrial", "goth", "synth", "wave",
    "electro", "noise", "aggrotech", "futurepop", "coldwave",
    "minimal", "techno", "post-punk", "deathrock", "neofolk",
    "martial", "ambient", "metal", "punk", "alternative",
    "electronic", "experimental", "new wave", "darksynth",
]

# Explizit verbotene Genres (lowercase, Teilstring-Match)
BANNED_GENRES = [
    "hip hop", "rap", "pop", "schlager", "reggaeton",
    "country", "latin", "k-pop", "j-pop", "r&b",
    "soul", "jazz", "blues", "folk", "singer-songwriter",
    "children", "disney", "christian", "gospel",
    "tropical", "dancehall", "afrobeat",
]

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

    def __init__(self):
        self._sp = None
        self._lock = threading.Lock()
        self._device_id = None
        self._profile = None
        self._initialized = False
        self._auth_failed = False

        # Auto-DJ State
        self._auto_dj_active = False
        self._auto_dj_thread = None
        self._auto_dj_zone = None       # Letzte Zone fuer die Musik gewaehlt wurde
        self._auto_dj_lock = threading.Lock()

        # Profil-Cache (geladene Top-Artists fuer schnelle Suche)
        self._profile_artists = {}      # {artist_name_lower: {rank, plays, genre}}
        self._profile_top_tracks = []   # [{name, artist, uri}]

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
        """Lade Spotify-Profil und baue Lookup-Caches auf."""
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

    def _refresh_device(self):
        """Device-ID auffrischen falls verloren."""
        if not self._device_id:
            self._find_device()

    # =========================================================================
    # PLAYBACK CONTROLS
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

    def shuffle(self, state: bool) -> bool:
        """Shuffle ein/ausschalten."""
        if not self._ensure_auth():
            return False
        try:
            self._sp.shuffle(state, device_id=self._device_id)
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
    # GENRE-LOCK: Suchergebnisse filtern
    # =========================================================================

    def _is_genre_allowed(self, artist_id: str) -> bool:
        """Prueft ob ein Artist in erlaubte Genres faellt.

        Checkt zuerst ob Artist in Markus' Profil ist (= immer erlaubt).
        Dann Spotify-Genre-Tags gegen Whitelist/Blacklist.
        """
        try:
            # Profil-Artists sind IMMER erlaubt (Markus hoert sie = gut)
            artist_info = self._sp.artist(artist_id)
            name = artist_info.get("name", "").lower()
            if name in self._profile_artists:
                return True

            # Genre-Tags vom Artist pruefen
            genres = [g.lower() for g in artist_info.get("genres", [])]
            if not genres:
                # Kein Genre-Tag? Erlauben (besser als blockieren)
                return True

            # Banned-Check: Ein verbotenes Genre -> raus
            for genre in genres:
                for banned in BANNED_GENRES:
                    if banned in genre:
                        logger.debug(f"[GENRE-LOCK] Blockiert: {artist_info['name']} ({genre})")
                        return False

            # Allowed-Check: Mindestens ein erlaubtes Genre muss matchen
            for genre in genres:
                for allowed in ALLOWED_GENRES:
                    if allowed in genre:
                        return True

            # Kein Match in beiden Listen? Blockieren (sicherheitshalber)
            logger.debug(f"[GENRE-LOCK] Kein Genre-Match: {artist_info['name']} ({genres})")
            return False

        except Exception:
            # API-Fehler? Erlauben um Playback nicht zu blockieren
            return True

    def _filter_search_results(self, tracks: List[Dict], max_results: int = 5) -> List[Dict]:
        """Filtert Suchergebnisse durch Genre-Lock.

        Args:
            tracks: Liste von Spotify Track-Dicts
            max_results: Maximale Anzahl Ergebnisse

        Returns:
            Gefilterte Track-Liste
        """
        filtered = []
        for track in tracks:
            if len(filtered) >= max_results:
                break
            artist_id = track.get("artists", [{}])[0].get("id")
            if artist_id and self._is_genre_allowed(artist_id):
                filtered.append(track)
        return filtered

    # =========================================================================
    # SEARCH & PLAY (mit Genre-Lock)
    # =========================================================================

    def search_and_play(self, query: str) -> bool:
        """Suche und spiele — mit Genre-Lock Filter."""
        if not self._ensure_auth():
            return False
        try:
            # Mehr Ergebnisse holen fuer Genre-Filter
            results = self._sp.search(q=query, limit=10, type="track")
            tracks = results.get("tracks", {}).get("items", [])
            if not tracks:
                logger.warning(f"[SPOTIFY] Keine Treffer fuer: {query}")
                return False

            # Genre-Lock anwenden
            filtered = self._filter_search_results(tracks, max_results=1)
            if not filtered:
                logger.warning(f"[SPOTIFY] Alle Treffer geblockt (Genre-Lock): {query}")
                # Fallback: Ersten Track nehmen wenn nichts durchkommt
                filtered = tracks[:1]

            track = filtered[0]
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
                self.shuffle(True)
                return self.play(uri=uris)
            return False
        except Exception as e:
            logger.error(f"[SPOTIFY] Artist Play fehlgeschlagen: {e}")
            return False

    # =========================================================================
    # MOOD / ZONE INTEGRATION (mit Zeitmodifikator)
    # =========================================================================

    def play_by_mood(self, zone: str) -> bool:
        """Spiele Musik passend zur Zone + Tageszeit.

        guardian -> Futurepop, Synthwave, atmosphaerisch
        shadow   -> Dark Electro, EBM, Industrial Techno
        berserker -> Aggrotech, Power Noise, Industrial Metal

        Zeitmodifikator beeinflusst Artist-Auswahl:
        - Morgens: Ruhigere Artists bevorzugt
        - Abends/Nachts: Dunklere Artists bevorzugt
        """
        if not self._ensure_auth():
            return False

        time_zone = _get_time_zone()
        weights = TIME_ZONE_WEIGHTS.get(time_zone, TIME_ZONE_WEIGHTS["tag"])
        zone_weight = weights.get(zone, 1.0)

        # Zeitbasierte Zone-Verschiebung
        effective_zone = zone
        if zone_weight < 0.3:
            # Zone passt nicht zur Tageszeit -> Guardian als Fallback
            effective_zone = "guardian"
            logger.info(f"[SPOTIFY] Zone '{zone}' zu intensiv fuer {time_zone}, "
                        f"verwende 'guardian'")

        artists = list(ZONE_ARTISTS.get(effective_zone, ZONE_ARTISTS["shadow"]))
        random.shuffle(artists)

        # Zeitbasierte Priorisierung: Profil-Artists die zur Tageszeit passen
        if time_zone == "morgen" and effective_zone != "guardian":
            # Morgens Guardian-Artists beimischen
            guardian_artists = list(ZONE_ARTISTS["guardian"])
            random.shuffle(guardian_artists)
            artists = guardian_artists[:3] + artists
        elif time_zone == "nacht" and effective_zone == "guardian":
            # Nachts Shadow-Artists beimischen
            shadow_artists = list(ZONE_ARTISTS["shadow"])
            random.shuffle(shadow_artists)
            artists = artists + shadow_artists[:3]

        for artist_name in artists[:5]:
            if self.play_artist(artist_name):
                logger.info(f"[SPOTIFY] Mood '{zone}' ({time_zone}): Spiele {artist_name}")
                return True

        # Fallback: Genre-Suche
        search = ZONE_SEARCH.get(effective_zone, "dark electro EBM")
        logger.info(f"[SPOTIFY] Mood '{zone}' Fallback: Suche '{search}'")
        return self.search_and_play(search)

    def get_recommendations_for_zone(self, zone: str, limit: int = 10) -> List[str]:
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

    # =========================================================================
    # SMART REQUESTS
    # =========================================================================

    def play_from_year(self, year: int) -> bool:
        """Spiele Tracks von Artists die Markus hoert, aus einem bestimmten Jahr.

        Filtert NUR aus Markus' gehoerten Artists die im Jahr aktiv waren.
        """
        if not self._ensure_auth():
            return False

        if not self._profile_artists:
            logger.warning("[SPOTIFY] Kein Profil geladen, kann nicht nach Jahr filtern")
            return self.search_and_play(f"year:{year} dark electro EBM")

        try:
            # Top-Artists aus Profil nehmen und Tracks aus dem Jahr suchen
            top_artists = sorted(
                self._profile_artists.values(),
                key=lambda a: a["plays"],
                reverse=True,
            )[:20]

            uris = []
            for artist_info in top_artists:
                if len(uris) >= 15:
                    break
                query = f"artist:{artist_info['name']} year:{year}"
                results = self._sp.search(q=query, limit=3, type="track")
                tracks = results.get("tracks", {}).get("items", [])
                for t in tracks:
                    # Nur Tracks die wirklich aus dem Jahr sind
                    release = t.get("album", {}).get("release_date", "")
                    if release.startswith(str(year)):
                        uris.append(t["uri"])

            if uris:
                random.shuffle(uris)
                logger.info(f"[SPOTIFY] {len(uris)} Tracks aus {year} gefunden")
                self.shuffle(True)
                return self.play(uri=uris[:20])

            logger.warning(f"[SPOTIFY] Keine Tracks aus {year} bei bekannten Artists")
            return self.search_and_play(f"year:{year} dark electro EBM industrial")

        except Exception as e:
            logger.error(f"[SPOTIFY] Year-Filter fehlgeschlagen: {e}")
            return False

    def play_similar(self) -> bool:
        """Spiele aehnliche Musik zum aktuell laufenden Track.

        Nutzt Spotify Recommendations API mit Seed aus aktuellem Track.
        """
        if not self._ensure_auth():
            return False

        try:
            current = self._sp.current_playback()
            if not current or not current.get("item"):
                logger.warning("[SPOTIFY] Kein Track laeuft -> kann keine Empfehlung geben")
                return False

            item = current["item"]
            track_id = item["id"]
            artist_id = item["artists"][0]["id"]

            # Recommendations basierend auf aktuellem Track + Artist
            recs = self._sp.recommendations(
                seed_tracks=[track_id],
                seed_artists=[artist_id],
                limit=20,
            )

            tracks = recs.get("tracks", [])
            # Genre-Lock anwenden
            filtered = self._filter_search_results(tracks, max_results=15)

            if filtered:
                uris = [t["uri"] for t in filtered]
                logger.info(f"[SPOTIFY] {len(uris)} aehnliche Tracks zu "
                            f"'{item['artists'][0]['name']} - {item['name']}'")
                self.shuffle(True)
                return self.play(uri=uris)

            logger.warning("[SPOTIFY] Keine passenden Empfehlungen")
            return False

        except Exception as e:
            logger.error(f"[SPOTIFY] Similar fehlgeschlagen: {e}")
            return False

    def play_top_tracks(self) -> bool:
        """Spiele Markus' Top Tracks aus dem Spotify-Profil."""
        if not self._ensure_auth():
            return False

        try:
            # Zuerst Spotify API Top Tracks versuchen
            top = self._sp.current_user_top_tracks(limit=30, time_range="medium_term")
            tracks = top.get("items", [])

            if tracks:
                # Genre-Lock
                filtered = self._filter_search_results(tracks, max_results=20)
                if filtered:
                    uris = [t["uri"] for t in filtered]
                    random.shuffle(uris)
                    logger.info(f"[SPOTIFY] {len(uris)} Top Tracks")
                    self.shuffle(True)
                    return self.play(uri=uris)

            # Fallback: Profil-Top-Tracks
            if self._profile_top_tracks:
                uris = []
                for tt in self._profile_top_tracks[:20]:
                    query = f"artist:{tt.get('artist', '')} track:{tt.get('name', '')}"
                    results = self._sp.search(q=query, limit=1, type="track")
                    items = results.get("tracks", {}).get("items", [])
                    if items:
                        uris.append(items[0]["uri"])
                if uris:
                    random.shuffle(uris)
                    self.shuffle(True)
                    return self.play(uri=uris)

            logger.warning("[SPOTIFY] Keine Top Tracks gefunden")
            return False

        except Exception as e:
            logger.error(f"[SPOTIFY] Top Tracks fehlgeschlagen: {e}")
            return False

    def play_new_music(self) -> bool:
        """Entdecke neue Musik basierend auf Markus' Top-Artists.

        Nutzt Recommendations API mit Seeds aus den meistgehoerten Artists.
        """
        if not self._ensure_auth():
            return False

        try:
            # Seeds aus Top-Artists im Profil
            seed_ids = []
            top_names = sorted(
                self._profile_artists.values(),
                key=lambda a: a["plays"],
                reverse=True,
            )[:10]

            for artist_info in top_names:
                if len(seed_ids) >= 5:
                    break
                results = self._sp.search(
                    q=f"artist:{artist_info['name']}", limit=1, type="artist",
                )
                items = results.get("artists", {}).get("items", [])
                if items:
                    seed_ids.append(items[0]["id"])

            if not seed_ids:
                return self.search_and_play("new dark electro EBM 2026")

            recs = self._sp.recommendations(
                seed_artists=seed_ids[:5],
                limit=30,
            )

            tracks = recs.get("tracks", [])
            # Genre-Lock + bekannte Artists rausfiltern (wir wollen NEUES)
            filtered = []
            for t in tracks:
                if len(filtered) >= 20:
                    break
                artist_name = t["artists"][0]["name"].lower()
                # Nur Artists die NICHT in den Top-50 sind
                if artist_name not in self._profile_artists or \
                   self._profile_artists[artist_name]["rank"] > 50:
                    artist_id = t["artists"][0]["id"]
                    if self._is_genre_allowed(artist_id):
                        filtered.append(t)

            if filtered:
                uris = [t["uri"] for t in filtered]
                logger.info(f"[SPOTIFY] {len(uris)} neue Tracks entdeckt")
                self.shuffle(True)
                return self.play(uri=uris)

            logger.warning("[SPOTIFY] Keine neuen Empfehlungen nach Genre-Filter")
            return False

        except Exception as e:
            logger.error(f"[SPOTIFY] New Music fehlgeschlagen: {e}")
            return False

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
        """
        logger.info("[AUTO-DJ] Loop gestartet")

        while self._auto_dj_active:
            try:
                # Core Integrator Zone lesen
                zone = self._get_current_zone()
                if zone and zone != self._auto_dj_zone:
                    old_zone = self._auto_dj_zone
                    self._auto_dj_zone = zone
                    if old_zone is not None:
                        # Zone hat gewechselt -> Musik wechseln
                        logger.info(f"[AUTO-DJ] Zone-Wechsel: {old_zone} -> {zone}")
                        self.play_by_mood(zone)
                    else:
                        # Erster Start -> aktuelle Zone spielen
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
        """Gesamtstatus fuer Panel/IPC.

        Wird in _write_status_json eingebaut und vom Panel gelesen.
        """
        track = self.get_current_track() if self._initialized else None
        return {
            "initialized": self._initialized,
            "auth_ok": self._sp is not None,
            "auth_failed": self._auth_failed,
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
