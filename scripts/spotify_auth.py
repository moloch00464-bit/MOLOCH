#!/usr/bin/env python3
"""
Spotify OAuth Token erstmalig generieren.

ANLEITUNG:
1. Gehe auf developer.spotify.com/dashboard
2. Oeffne deine App → Settings
3. Fuege als Redirect URI hinzu: http://127.0.0.1:8888/callback
4. Starte dieses Script: python3 scripts/spotify_auth.py
5. Oeffne die angezeigte URL im Browser (am PC/Handy)
6. Logge dich bei Spotify ein und erlaube Zugriff
7. Du wirst zu 127.0.0.1:8888/callback?code=... weitergeleitet
   (Die Seite wird NICHT laden — das ist OK!)
8. Kopiere die GESAMTE URL aus der Browser-Adressleiste
9. Paste sie hier ein

Danach ist der Token gespeichert und Moloch kann Spotify steuern.
"""

import os
import sys

# .env laden
env_path = os.path.expanduser("~/moloch/.env.spotify")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

import spotipy
from spotipy.oauth2 import SpotifyOAuth

cache_path = os.path.expanduser("~/.cache/spotipy/.cache")
os.makedirs(os.path.dirname(cache_path), exist_ok=True)

scope = (
    "user-read-playback-state "
    "user-modify-playback-state "
    "user-read-currently-playing "
    "playlist-read-private "
    "user-library-read "
    "user-top-read"
)

auth = SpotifyOAuth(
    scope=scope,
    cache_path=cache_path,
    open_browser=False,
)

# Token schon vorhanden?
token = auth.get_cached_token()
if token:
    print("Token bereits vorhanden und gueltig!")
    sp = spotipy.Spotify(auth_manager=auth)
    user = sp.me()
    print(f"Eingeloggt als: {user['display_name']} ({user['id']})")
    devices = sp.devices()
    print(f"\nVerfuegbare Geraete:")
    for d in devices.get("devices", []):
        active = " [AKTIV]" if d.get("is_active") else ""
        print(f"  - {d['name']} ({d['type']}){active}")
    sys.exit(0)

# Neuen Token holen
url = auth.get_authorize_url()
print("=" * 60)
print("SPOTIFY AUTHORIZATION")
print("=" * 60)
print()
print("1. Oeffne diese URL im Browser:")
print()
print(f"   {url}")
print()
print("2. Logge dich ein und erlaube Zugriff")
print("3. Kopiere die Redirect-URL aus der Adressleiste")
print("   (Die Seite laedt nicht — das ist normal!)")
print()

redirect_url = input("Paste die Redirect-URL hier: ").strip()
if not redirect_url:
    print("Abgebrochen.")
    sys.exit(1)

code = auth.parse_response_code(redirect_url)
token = auth.get_access_token(code)

if token:
    print("\nToken erfolgreich generiert!")
    sp = spotipy.Spotify(auth_manager=auth)
    user = sp.me()
    print(f"Eingeloggt als: {user['display_name']} ({user['id']})")
    devices = sp.devices()
    print(f"\nVerfuegbare Geraete:")
    for d in devices.get("devices", []):
        active = " [AKTIV]" if d.get("is_active") else ""
        print(f"  - {d['name']} ({d['type']}){active}")
    print(f"\nToken gespeichert: {cache_path}")
    print("Moloch kann jetzt Spotify steuern!")
else:
    print("\nToken-Generierung fehlgeschlagen!")
    sys.exit(1)
