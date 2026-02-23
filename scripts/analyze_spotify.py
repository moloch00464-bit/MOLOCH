#!/usr/bin/env python3
"""
Spotify Extended Streaming History Analyse fuer M.O.L.O.C.H.
Liest ALLE JSON-Dateien, berechnet echte Stunden, Tageszeiten, Plattformen.
Ergebnis: /mnt/moloch-data/memory/spotify/spotify_profile.json
"""

import json
import os
import glob
from collections import defaultdict
from datetime import datetime

RAW_DIR = "/mnt/moloch-data/memory/spotify/raw/Spotify Extended Streaming History"
OUTPUT_PATH = "/mnt/moloch-data/memory/spotify/spotify_profile.json"

# Alle Audio-JSON Dateien laden
all_streams = []
for filepath in sorted(glob.glob(os.path.join(RAW_DIR, "Streaming_History_Audio_*.json"))):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        all_streams.extend(data)
        print(f"  {os.path.basename(filepath)}: {len(data)} Eintraege")

print(f"\nTotal Streams geladen: {len(all_streams)}")

# Filter: Nur echte Plays (mindestens 30 Sekunden gehoert)
MIN_MS = 30000
plays = [s for s in all_streams if s.get("ms_played", 0) >= MIN_MS and s.get("master_metadata_track_name")]
print(f"Plays (>30s, mit Track): {len(plays)}")

# === ARTIST STATISTIKEN ===
artist_ms = defaultdict(int)
artist_plays = defaultdict(int)
for s in plays:
    artist = s.get("master_metadata_album_artist_name", "Unknown")
    if artist:
        artist_ms[artist] += s["ms_played"]
        artist_plays[artist] += 1

# Top 50 Artists nach Hoerzeit
top_artists_by_hours = sorted(artist_ms.items(), key=lambda x: x[1], reverse=True)[:50]

# === TRACK STATISTIKEN ===
track_plays = defaultdict(int)
track_ms = defaultdict(int)
for s in plays:
    artist = s.get("master_metadata_album_artist_name", "Unknown")
    track = s.get("master_metadata_track_name", "Unknown")
    key = f"{artist} - {track}"
    track_plays[key] += 1
    track_ms[key] += s["ms_played"]

top_tracks = sorted(track_plays.items(), key=lambda x: x[1], reverse=True)[:50]

# === ALBUM STATISTIKEN ===
album_plays = defaultdict(int)
for s in plays:
    artist = s.get("master_metadata_album_artist_name", "Unknown")
    album = s.get("master_metadata_album_album_name", "Unknown")
    if album:
        key = f"{artist} - {album}"
        album_plays[key] += 1

top_albums = sorted(album_plays.items(), key=lambda x: x[1], reverse=True)[:20]

# === TAGESZEIT-ANALYSE ===
hour_distribution = defaultdict(int)
hour_ms = defaultdict(int)
for s in plays:
    ts = s.get("ts", "")
    if "T" in ts:
        try:
            hour = int(ts.split("T")[1][:2])
            hour_distribution[hour] += 1
            hour_ms[hour] += s["ms_played"]
        except (ValueError, IndexError):
            pass

# Tageszeit-Kategorien
time_slots = {
    "Nacht (0-6)": sum(hour_distribution.get(h, 0) for h in range(0, 6)),
    "Morgen (6-12)": sum(hour_distribution.get(h, 0) for h in range(6, 12)),
    "Nachmittag (12-18)": sum(hour_distribution.get(h, 0) for h in range(12, 18)),
    "Abend (18-24)": sum(hour_distribution.get(h, 0) for h in range(18, 24)),
}

# === JAHRES-ANALYSE ===
yearly_stats = defaultdict(lambda: {"streams": 0, "ms": 0})
for s in plays:
    ts = s.get("ts", "")
    if ts:
        try:
            year = ts[:4]
            yearly_stats[year]["streams"] += 1
            yearly_stats[year]["ms"] += s["ms_played"]
        except (ValueError, IndexError):
            pass

# === PLATTFORM-ANALYSE ===
platform_stats = defaultdict(int)
for s in plays:
    platform = s.get("platform", "unknown")
    platform_stats[platform] += 1

# === SKIP-ANALYSE ===
total_skipped = sum(1 for s in plays if s.get("skipped"))
total_completed = sum(1 for s in plays if s.get("reason_end") == "trackdone")

# === WOCHENTAG-ANALYSE ===
weekday_stats = defaultdict(int)
weekday_names = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
for s in plays:
    ts = s.get("ts", "")
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            weekday_stats[weekday_names[dt.weekday()]] += 1
        except (ValueError, IndexError):
            pass

# === GENRE-MAPPING (manuell fuer Top Artists) ===
genre_map = {
    "Suicide Commando": "Dark Electro / EBM",
    "SIERRA": "Dark Electro / Industrial",
    "Vomito Negro": "EBM / Dark Electro",
    "Chainreactor": "Harsh EBM / Industrial",
    "ESA (Electronic Substance Abuse)": "Dark Electro / Power Noise",
    "Autodafeh": "EBM / Synthpop",
    "Portion Control": "EBM / Industrial",
    "VNV Nation": "Futurepop / EBM",
    "AC/DC": "Hard Rock",
    "Geistform": "EBM / Minimal Electro",
    "Orange Sector": "EBM / Aggrotech",
    "16Volt": "Industrial Rock",
    "Combichrist": "Aggrotech / Industrial",
    "HEALTH": "Industrial / Noise Rock",
    ":Wumpscut:": "Dark Electro / EBM",
    "Phase Fatale": "Industrial Techno",
    "Ancient Methods": "Industrial Techno",
    "E-Craft": "EBM / Dark Electro",
    "SAM": "Dark Electro",
    "Daniel Deluxe": "Darksynth / Synthwave",
    "Life Cried": "Dark Electro / EBM",
    "Depeche Mode": "Synthpop / New Wave",
    "Ministry": "Industrial Metal",
    "Schwefelgelb": "EBM / Minimal Wave",
    "Xotox": "Rhythmic Noise / Industrial",
    "I Hate Models": "Industrial Techno",
    "Alien Vampires": "Aggrotech / Dark Electro",
    "Terrorfakt": "Power Noise / Rhythmic Industrial",
    "Spark!": "EBM",
    "ALVABEAT": "Dark Electro / EBM",
    "Perturbator": "Darksynth / Synthwave",
    "Front 242": "EBM",
    "The Prodigy": "Big Beat / Electronic",
    "Airbourne": "Hard Rock",
    "Rammstein": "Neue Deutsche Haerte",
    "Carpenter Brut": "Darksynth / Synthwave",
    "Perc": "Industrial Techno",
    "New Frames": "Industrial Techno",
    "Codex Empire": "Industrial Techno",
    "Feindflug": "EBM / Martial Industrial",
    "FabrikC": "Rhythmic Noise",
    "iVardensphere": "Tribal Industrial",
    "Ambassador21": "Digital Hardcore / Industrial",
    "Wynardtage": "Dark Electro / EBM",
    "Komor Kommando": "EBM / Dark Electro",
    "Prong": "Industrial Metal",
    "Velvet Acid Christ": "Dark Electro / EBM",
    "Apoptygma Berzerk": "EBM / Futurepop",
    "[:SITD:]": "EBM / Futurepop",
    "KANGA": "Dark Electro / Synthpop",
    "Diva Destruction": "Dark Wave / Goth Rock",
}

# Genre-Verteilung berechnen basierend auf echten Hoerminuten
genre_hours = defaultdict(float)
for artist, ms in artist_ms.items():
    genre = genre_map.get(artist, "Sonstige")
    genre_hours[genre] += ms / 3600000.0

# Genre-Gruppen (aehnliche zusammenfassen)
genre_groups = {
    "Dark Electro / EBM": ["Dark Electro / EBM", "EBM / Dark Electro", "Dark Electro / Power Noise",
                           "EBM / Synthpop", "EBM / Industrial", "EBM / Minimal Electro",
                           "Dark Electro / EBM", "Dark Electro", "EBM", "Dark Electro / Industrial",
                           "EBM / Minimal Wave", "EBM / Futurepop", "EBM / Martial Industrial",
                           "Dark Electro / Synthpop", "Dark Wave / Goth Rock"],
    "Industrial Techno": ["Industrial Techno"],
    "Harsh EBM / Aggrotech": ["Harsh EBM / Industrial", "Aggrotech / Industrial", "EBM / Aggrotech",
                               "Aggrotech / Dark Electro"],
    "Industrial Rock/Metal": ["Industrial Rock", "Industrial / Noise Rock", "Industrial Metal",
                               "Neue Deutsche Haerte"],
    "Darksynth / Synthwave": ["Darksynth / Synthwave"],
    "Power Noise / Rhythmic Industrial": ["Power Noise / Rhythmic Industrial", "Rhythmic Noise / Industrial",
                                           "Rhythmic Noise", "Tribal Industrial"],
    "Futurepop / Synthpop": ["Futurepop / EBM", "Synthpop / New Wave"],
    "Hard Rock": ["Hard Rock"],
    "Big Beat / Electronic": ["Big Beat / Electronic", "Digital Hardcore / Industrial"],
}

grouped_hours = {}
for group_name, genres in genre_groups.items():
    total = sum(genre_hours.get(g, 0) for g in genres)
    if total > 0:
        grouped_hours[group_name] = total

# "Sonstige" = alles was nicht gemappt ist
sonstige = genre_hours.get("Sonstige", 0)
if sonstige > 0:
    grouped_hours["Sonstige (ungemappt)"] = sonstige

total_genre_hours = sum(grouped_hours.values())
genre_pcts = []
for name, hours in sorted(grouped_hours.items(), key=lambda x: x[1], reverse=True):
    pct = (hours / total_genre_hours * 100) if total_genre_hours > 0 else 0
    genre_pcts.append({"genre": name, "hours": round(hours, 1), "share_pct": round(pct, 1)})

# === GESAMTSTATISTIKEN ===
total_ms = sum(s["ms_played"] for s in plays)
total_hours = total_ms / 3600000.0
total_streams = len(plays)
unique_artists = len(artist_plays)
unique_tracks = len(track_plays)

# === PROFIL ZUSAMMENBAUEN ===
# Top 5 fuer Summary
t5 = top_artists_by_hours[:5]
t5_str = ", ".join(f"{a} ({ms/3600000:.0f}h)" for a, ms in t5)

# Aktivstes Jahr
peak_year = max(yearly_stats.items(), key=lambda x: x[1]["streams"])
peak_year_name = peak_year[0]

# Tageszeit-Peak
peak_time = max(time_slots.items(), key=lambda x: x[1])

profile = {
    "meta": {
        "source": "Spotify Extended Streaming History Export",
        "period": f"{min(yearly_stats.keys())}-{max(yearly_stats.keys())}",
        "total_streams": total_streams,
        "unique_artists": unique_artists,
        "unique_tracks": unique_tracks,
        "total_listening_hours": round(total_hours, 1),
        "avg_minutes_per_stream": round(total_ms / total_streams / 60000, 1) if total_streams > 0 else 0,
        "analyzed_at": datetime.now().isoformat()[:19],
        "raw_files": len(glob.glob(os.path.join(RAW_DIR, "*.json")))
    },

    "summary": (
        f"Markus hat zwischen {min(yearly_stats.keys())} und {max(yearly_stats.keys())} "
        f"insgesamt {total_hours:.0f} Stunden Musik gehoert ({total_streams:,} Streams). "
        f"Top 5: {t5_str}. "
        f"Genres: hauptsaechlich Dark Electro/EBM, Industrial Techno und Aggrotech. "
        f"Aktivstes Jahr: {peak_year_name} ({peak_year[1]['streams']:,} Streams). "
        f"Hoert am meisten {peak_time[0].lower()} ({peak_time[1]:,} Plays). "
        f"Schwarze Szene durch und durch."
    ),

    "top_artists": [
        {
            "rank": i + 1,
            "name": name,
            "plays": artist_plays[name],
            "hours": round(ms / 3600000, 1),
            "genre": genre_map.get(name, "")
        }
        for i, (name, ms) in enumerate(top_artists_by_hours)
    ],

    "top_tracks": [
        {
            "rank": i + 1,
            "name": name,
            "plays": count,
            "hours": round(track_ms[name] / 3600000, 1)
        }
        for i, (name, count) in enumerate(top_tracks)
    ],

    "top_albums": [
        {"rank": i + 1, "name": name, "plays": count}
        for i, (name, count) in enumerate(top_albums)
    ],

    "yearly_activity": {
        year: {
            "streams": stats["streams"],
            "hours": round(stats["ms"] / 3600000, 1)
        }
        for year, stats in sorted(yearly_stats.items())
    },

    "hourly_distribution": {
        f"{h:02d}:00": {
            "plays": hour_distribution.get(h, 0),
            "hours": round(hour_ms.get(h, 0) / 3600000, 1)
        }
        for h in range(24)
    },

    "time_of_day": time_slots,

    "weekday_distribution": weekday_stats,

    "platform_distribution": dict(sorted(platform_stats.items(), key=lambda x: x[1], reverse=True)),

    "listening_behavior": {
        "total_skipped": total_skipped,
        "total_completed": total_completed,
        "skip_rate_pct": round(total_skipped / total_streams * 100, 1) if total_streams > 0 else 0,
        "completion_rate_pct": round(total_completed / total_streams * 100, 1) if total_streams > 0 else 0,
    },

    "genre_profile": {
        "primary_genres": genre_pcts,
        "scene": "Schwarze Szene / Gothic-Industrial / WGT-Gaenger",
        "description": (
            "Markus ist tief verwurzelt in der Dark Electro und EBM Szene. "
            "Sein Geschmack reicht von klassischem EBM (Front 242, VNV Nation) "
            "ueber harten Dark Electro (Suicide Commando, Vomito Negro) bis zu "
            "modernem Industrial Techno (Phase Fatale, Ancient Methods). "
            "Die Hard Rock Seite (AC/DC, Airbourne) zeigt eine 'Musik muss laut und hart' Mentalitaet."
        )
    },

    "personality_insights": {
        "music_identity": "Schwarze Szene durch und durch. EBM/Industrial ist Lebenseinstellung. WGT ist Pflichttermin.",
        "mood_preferences": "Dunkle, treibende, aggressive Musik. Wenig Balladen. Musik muss Energie haben.",
        "discovery_style": "Hoert Alben komplett durch. Bleibt Kuenstlern treu. Entdeckt im Genre-Umfeld.",
        "peak_listening": f"{peak_time[0]} ist Haupthoerzeit.",
    }
}

# Schreiben
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(profile, f, ensure_ascii=False, indent=2)

print(f"\n=== ERGEBNIS ===")
print(f"Total: {total_streams:,} Streams, {total_hours:.0f}h, {unique_artists} Artists, {unique_tracks} Tracks")
print(f"Avg Track: {total_ms / total_streams / 60000:.1f} min")
print(f"\nTop 10 Artists (echte Stunden):")
for name, ms in top_artists_by_hours[:10]:
    print(f"  {name}: {ms/3600000:.1f}h ({artist_plays[name]} Plays)")
print(f"\nTop 10 Tracks:")
for name, count in top_tracks[:10]:
    print(f"  {name}: {count}x ({track_ms[name]/3600000:.1f}h)")
print(f"\nTageszeit:")
for slot, count in sorted(time_slots.items(), key=lambda x: x[1], reverse=True):
    print(f"  {slot}: {count:,} Plays")
print(f"\nPlattformen:")
for plat, count in sorted(platform_stats.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {plat}: {count:,}")
print(f"\nSkip-Rate: {total_skipped}/{total_streams} ({total_skipped/total_streams*100:.1f}%)")
print(f"\nGespeichert: {OUTPUT_PATH}")
