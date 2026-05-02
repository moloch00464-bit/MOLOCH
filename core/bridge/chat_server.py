#!/usr/bin/env python3
"""
MOLOCH Chat-Server (Pi-Side)
=============================
FastAPI Port 9100. Wrapper um local_llm_bridge fuer PC-Chat-UI.

Endpoints:
  GET  /health
  GET  /status     -> Bridge-Stats
  POST /chat       -> {text, force_local?, use_reason?} -> {text, provider, duration_ms}
"""
import asyncio
import logging
import os
import re
import subprocess
import sys
import time
import uuid

import json
import mmap
import struct

import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.expanduser("~/moloch"))
from core.autonomy.local_llm_bridge import get_llm_bridge, _load_tentacle_cfg, _is_hardware_query
from core.longterm_memory import get_memory
from core.moloch_event_bus import get_event_bus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("chat-server")

HOST = os.environ.get("MOLOCH_CHAT_HOST", "0.0.0.0")
PORT = int(os.environ.get("MOLOCH_CHAT_PORT", "9100"))
# HTTPS-Mode (fuer Browser-Mikrofon — Web Speech API braucht secure context):
SSL_KEYFILE = os.environ.get("MOLOCH_CHAT_SSL_KEY", "")
SSL_CERTFILE = os.environ.get("MOLOCH_CHAT_SSL_CERT", "")

app = FastAPI(title="MOLOCH Chat-Server", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# PC-Heartbeat-Tracking (Task 5d): letzter empfangener PC-Online-Timestamp
_pc_online_ts: float = 0.0


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    force_local: bool = False
    use_reason: bool = False


_WEB_RESEARCH_KEYWORDS = (
    "such ", "google ", "tech-news", "tech news", "aktuell",
    "im netz", "recherchier", "was gibt's neues", "was gibts neues",
    "neuigkeiten", "online suchen", "was ist heute", "news heute",
)

# Welle 19 — User-facing Recherche-Klassifikator (broader als _WEB_RESEARCH_KEYWORDS)
# Triggert prompt_type="web" mit Search-Proxy-Augmentation in /chat-Handler
# (siehe Specialist-Router unter @app.post("/chat")). MUSS vor _is_music_query
# pruefen, sonst faengt music_query Phrasen wie "wer spielt" / "welche bands" ab.
_WEB_LIVE_KEYWORDS = (
    "recherchier", "such ", "finde heraus", "find heraus",
    "wieviel", "wie viele", "wer spielt", "lineup", "line-up",
    "was steht auf", "welche bands", "welche band", "programm",
    "nachschlag", "schau nach", "schaue nach",
)

# W20a-A2: regex fuer "welche P-Bands" / "welche X-Bands" (Bindestrich-Komposita)
# matcht "welche P-Bands", "welche metal bands", "welche dark-bands" etc.
_WHICH_BANDS_RE = re.compile(r'\bwelche [\w\-]+\s*-?\s*bands?\b', re.IGNORECASE)
# W20a-A2: Festival-Namen allein triggern web (vor music_query-Klassifizierung)
_FESTIVAL_NAME_RE = re.compile(
    r"\b(wgt|wave-?gotik|amphi|m['`]?era[\s\-]?luna)\b", re.IGNORECASE
)


def _is_web_live_query(text_low: str) -> bool:
    """Welle 19: erkennt User-Recherche-Anfragen die Live-Web-Daten brauchen.

    W20a-A2: zusaetzlich regex-basierte Erkennung fuer "welche P-Bands" und
    Festival-Namen (WGT/Amphi/M'era Luna), die als reine Substrings nicht
    in _WEB_LIVE_KEYWORDS abgedeckt sind.
    """
    if any(kw in text_low for kw in _WEB_LIVE_KEYWORDS):
        return True
    if _WHICH_BANDS_RE.search(text_low):
        return True
    if _FESTIVAL_NAME_RE.search(text_low):
        return True
    return False

_CODE_KEYWORDS = (
    # Direkte "schreib X"-Imperative
    "schreib python", "schreib mir python", "schreib code", "schreib mir code",
    "schreib mir eine python", "schreib eine python",
    "schreib mir ein script", "schreib mir ein skript",
    "schreib mir eine funktion", "schreib eine funktion",
    "schreib mir eine klasse", "schreib eine klasse",
    "code für", "code fuer", "regex fuer", "regex für",
    # Sprache + Element
    "python-funktion", "python funktion", "python-klasse", "python klasse",
    "python-script", "python script", "python-skript", "python skript",
    "javascript funktion", "javascript-funktion",
    # Standalone-Sprachen (mit Leerzeichen wegen false positive)
    " sql ", " bash ", "javascript", "typescript",
    "shell script", "shell skript", "bash script", "bash skript",
)

_CODE_TOKEN_PREFIXES = ("def ", "class ", "import ", "from ", "function ",
                       "const ", "let ", "var ", "public ", "private ")

# Welle 6 — Music-Query Klassifikator
_MUSIC_KEYWORDS = (
    " band", " bands", "musik", "album", "alben", "festival", "festivals",
    "konzert", "konzerte", "gig", " dj ", " lied", "song", "tracks", "track",
    "spiel mir", "spiel meine", "spielt", "spielen", "spotify", "plattenladen",
    "vinyl", "gothik", "gothic", " ebm", "industrial", " wave", " wgt",
    "mera luna", "m'era luna", "amphi", "schwarze szene", "darkwave",
    "synthwave", "futurepop", "dark electro",
    # Welle 7: Reflektions-Phrasen die kurz sind aber Musik meinen
    "hoere ich", "höre ich", "hoer ich", "hör ich",
    "gerne hoer", "gerne hör", "gern hoer", "gern hör",
    "lieblings", "lieblingsband", "lieblingslied", "lieblingsalbum",
    "lieblingsmusik", "lieblings-song", "playlist",
    # PC-Topic 07:44 — weitere Music-Reflektions-Begriffe
    "gerade gerne", "gerade gern", "gerade hoere", "gerade höre",
    "mein liebling", "mein favorit", "favoriten",
    " top ", " hits", "charts",
    "80er", "90er", "2000er", "70er",
)

# Year-Filter fuer music_query (Welle 6 Schritt 7): 1950-2039 sinnvoll
_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")

# Welle 7 PC-Topic 07:44: Playlist-Pattern.
# Beispiele: "spiel meine Playlist X", "spiel die Playlist Y", "leg Playlist Z auf",
# "starte Playlist A", "Playlist B abspielen"
# Welle 12 Bug B: Music-Action-Phrasen (vor music_query) -> direkter IPC.
# Ordnung: spezifisch vor generisch. Match per Substring (case-insensitive).
_MUSIC_ACTION_PATTERNS = (
    # (Phrase, IPC-Action, Antwort-Text)
    ("naechster track",       "spotify_skip",        "Naechster Track."),
    ("naechster song",        "spotify_skip",        "Naechster Song."),
    ("nächster track",        "spotify_skip",        "Naechster Track."),
    ("nächster song",         "spotify_skip",        "Naechster Song."),
    ("ueberspringen",         "spotify_skip",        "Skip."),
    ("ueberspring",           "spotify_skip",        "Skip."),
    ("überspring",            "spotify_skip",        "Skip."),
    (" skip ",                "spotify_skip",        "Skip."),
    ("skip das",              "spotify_skip",        "Skip."),
    (" weiter",               "spotify_skip",        "Weiter."),
    ("vorheriger",            "spotify_previous",    "Vorheriger Track."),
    ("zurueck zum",           "spotify_previous",    "Zurueck."),
    ("vorigen",               "spotify_previous",    "Voriger Track."),
    ("pausier",               "spotify_pause",       "Pausiert."),
    (" pause",                "spotify_pause",       "Pausiert."),
    ("stop musik",            "spotify_pause",       "Stop."),
    ("halt musik",            "spotify_pause",       "Stop."),
    ("musik aus",             "spotify_pause",       "Aus."),
    ("musik weiter",          "spotify_toggle",      "Laeuft weiter."),
    ("fortsetzen",            "spotify_toggle",      "Laeuft weiter."),
    ("wechsel die musik",     "spotify_mood",        "Wechsel auf Mood-Auswahl."),
    ("wechsel musik",         "spotify_mood",        "Wechsel."),
    ("musik wechseln",        "spotify_mood",        "Andere Musik."),
    ("andere musik",          "spotify_mood",        "Andere Musik."),
    ("was anderes",           "spotify_skip",        "Was anderes."),
    ("nicht das",             "spotify_skip",        "Skip."),
    ("top tracks",            "spotify_top_tracks",  "Spiele Top Tracks."),
    ("meine lieblinge",       "spotify_top_tracks",  "Spiele Lieblinge."),
    ("meine hits",            "spotify_top_tracks",  "Spiele Hits."),
    ("top hits",              "spotify_top_tracks",  "Spiele Top Hits."),
    ("entdecker",             "spotify_new_music",   "Entdecker-Modus."),
    ("was neues",             "spotify_new_music",   "Neues entdecken."),
)


def _match_music_action(text: str):
    """Match Phrase -> (action, antwort_text). None wenn kein Match."""
    t = text.lower()
    # Padding fuer Wort-Boundary-Schutz bei kurzen Strings wie ' skip '
    padded = " " + t.strip() + " "
    for phrase, action, reply in _MUSIC_ACTION_PATTERNS:
        if phrase in padded or phrase in t:
            return action, reply
    return None


def _trigger_spotify_action(action: str, params: Dict = None) -> bool:
    """IPC-Cmd schreiben fuer beliebige Spotify-Action."""
    try:
        cmd = {"action": action}
        if params:
            cmd.update(params)
        path = f"/tmp/moloch_cmd_{int(time.time() * 1000)}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cmd, f)
        logger.info(f"[BRIDGE] IPC {action} -> {path}")
        return True
    except Exception as e:
        logger.warning(f"[BRIDGE] IPC-Schreibfehler {action}: {e}")
        return False


_PLAYLIST_RE = re.compile(
    r"(?:spiel(?:e)?(?:\s+mir)?(?:\s+(?:meine|die|den|den))?|leg(?:e)?|starte?)\s+"
    r"(?:die\s+)?playlist\s+(.+?)(?:\s+(?:ab|an|auf)\s*$|$)",
    re.IGNORECASE,
)

# Pre-cached Top-Artists aus spotify_profile.json (case-insensitive Match)
_MUSIC_ARTIST_CACHE: dict = {"loaded": False, "artists": ()}


def _load_music_artist_keywords() -> tuple:
    """Lazy-load Markus' Top-Artists aus Spotify-Profil als Klassifikations-Keywords."""
    if _MUSIC_ARTIST_CACHE["loaded"]:
        return _MUSIC_ARTIST_CACHE["artists"]
    try:
        with open("/mnt/moloch-data/memory/spotify/spotify_profile.json", "r", encoding="utf-8") as f:
            prof = json.load(f)
        names = []
        for a in (prof.get("top_artists") or [])[:30]:
            n = (a.get("name") or "").strip().lower()
            if len(n) >= 4:  # zu kurze Namen (z.B. "DAF") matchen sonst zufaellig in Saetzen
                names.append(n)
        _MUSIC_ARTIST_CACHE["artists"] = tuple(names)
    except Exception:
        _MUSIC_ARTIST_CACHE["artists"] = ()
    _MUSIC_ARTIST_CACHE["loaded"] = True
    return _MUSIC_ARTIST_CACHE["artists"]


def _is_music_query(text: str, text_low: str) -> bool:
    """Welle 6: erkennt Musik-Querys fuer prompt_type=music_query Routing."""
    if text.strip().lower().startswith("/music"):
        return True
    if any(kw in text_low for kw in _MUSIC_KEYWORDS):
        return True
    # Top-Artists-Match (case-insensitive)
    for artist in _load_music_artist_keywords():
        if artist in text_low:
            return True
    return False


def _is_web_research_query(text_low: str) -> bool:
    return any(kw in text_low for kw in _WEB_RESEARCH_KEYWORDS)


def _is_code_query(text: str, text_low: str) -> bool:
    if "```" in text:
        return True
    if any(kw in text_low for kw in _CODE_KEYWORDS):
        return True
    stripped = text.lstrip().lower()
    return any(stripped.startswith(p) for p in _CODE_TOKEN_PREFIXES)


def _extract_url(text: str) -> Optional[str]:
    """Welle 20a: extrahiert erste URL aus user_query (oder None)."""
    m = re.search(r'https?://[^\s]+', text)
    return m.group(0) if m else None


def _get_pi_mood_label() -> str:
    """Kurzes 'zone/tension'-Label fuer Cockpit-Badge (PC chat_ui)."""
    try:
        from core.core_integrator import get_core_integrator
        ci = get_core_integrator()
        zone = ci.get_personality_zone() if hasattr(ci, "get_personality_zone") else "?"
        tens = float(ci.get_tension()) if hasattr(ci, "get_tension") else 0.0
        if tens >= 0.5:
            t_label = "stress"
        elif tens >= 0.0:
            t_label = "neutral"
        elif tens >= -0.5:
            t_label = "ruhig"
        else:
            t_label = "wohl"
        return f"{zone}/{t_label}"
    except Exception:
        return "?"


def _classify_prompt_type(text: str) -> str:
    """Phase 5e + Welle 5: Klassifiziert Browser-Chat-Input fuer LLM-Routing.

    Wird in /chat genutzt, damit ask_external() das Type-Routing in
    _route_by_type() greift (statt vom alten force_tentacle-Hammer
    ueberstimmt zu werden).

    Reihenfolge wichtig — spezifischer vor generischer:
    - hardware_status: Slash-Cmd /hw oder Hardware-Query -> NPU/qwen
    - music_query: Music-Keyword/Artist (Welle 6) -> Kaskade mit Music-Profil
    - web_research: Web-Recherche-Keywords -> Tentakel + Search-Proxy (PC :11650)
    - code_query: Code-Frage (Python/JS/SQL/Bash, Codeblock, def/class) -> Tentakel mit code_model (deepseek-coder)
    - simple_smalltalk: kurze Eingabe (<80 Zeichen) -> NPU/qwen
    - complex_smalltalk: alles andere -> Tentakel (Fallback NPU)
    """
    if _is_hardware_query(text):
        return "hardware_status"
    text_low = text.lower()
    # Welle 20a: URL-Erkennung — hoechste Prioritaet fuer Web-Fetch-Pfad
    if re.search(r'https?://[^\s]+', text):
        return "web_fetch"
    # Welle 19: web (Live-Recherche) MUSS vor music_query — sonst faengt
    # music_query Phrasen wie "wer spielt aufm WGT" als Musik-Trigger ab.
    if _is_web_live_query(text_low):
        return "web"
    if _is_music_query(text, text_low):
        return "music_query"
    if _is_web_research_query(text_low):
        return "web_research"
    if _is_code_query(text, text_low):
        return "code_query"
    if len(text.strip()) < 80:
        return "simple_smalltalk"
    return "complex_smalltalk"


_VISUAL_DRIFT_HYSTERESIS_S = 3.0


def _check_visual_context_drift() -> str:
    """Phase 3 Task 3d + Welle 7 PC-Topic 07:44: Visual-Echo-Validator
    konservativ + 3s-Hysterese.

    Triggert NUR bei echten Identitaets-Wechseln die mindestens 3 Sekunden
    anhalten. Markus' Bug-Befund 2026-04-29: Disclaimer kam bei JEDEM Turn
    auch wenn Markus durchgehend im Bild war — kurzer SCRFD-Drop reichte.

    State (function attribute):
        _stable_face: face_id der als stabil gilt (Anker)
        _pending_face: face_id der den Anker ersetzen will
        _pending_since: monotonic-Zeit als Pending begann

    Logik pro Aufruf:
    1. Wenn cur_face == _stable_face -> stabil, pending zuruecksetzen, ""
    2. Wenn cur_face != _stable_face:
       a) Wenn pending == cur_face und Aenderung >=3s alt:
          -> Trigger Marker (alt -> cur), _stable = cur, pending reset
       b) Sonst: pending = cur_face, since = now (Hysterese laeuft)
    """
    try:
        import json as _json
        with open("/dev/shm/moloch_status.json", "r") as f:
            current = _json.load(f)
        cur_face = current.get("face_id")
        now = time.monotonic()
        stable = getattr(_check_visual_context_drift, "_stable_face", None)
        pending = getattr(_check_visual_context_drift, "_pending_face", None)
        pending_since = getattr(_check_visual_context_drift, "_pending_since", 0.0)

        # Erstaufruf — Anker setzen, kein Trigger
        if stable is None:
            _check_visual_context_drift._stable_face = cur_face
            _check_visual_context_drift._pending_face = None
            return ""

        # Stabil — kein Drift
        if cur_face == stable:
            _check_visual_context_drift._pending_face = None
            return ""

        # cur_face != stable -> moeglicher Drift
        if pending == cur_face:
            # Hysterese laeuft schon
            if (now - pending_since) >= _VISUAL_DRIFT_HYSTERESIS_S:
                # Anker uebernehmen + Marker (basierend auf altem stable)
                old = stable
                _check_visual_context_drift._stable_face = cur_face
                _check_visual_context_drift._pending_face = None
                if old and old != "unknown" and cur_face == "unknown":
                    return "[Hinweis: Unbekannter im Bild waehrend meiner Antwort.] "
                if old and cur_face and old != cur_face:
                    return "[Hinweis: andere Person im Bild waehrend meiner Antwort.] "
                return ""
            # Hysterese noch nicht erfuellt — kein Marker
            return ""
        # Neuer Pending — Hysterese-Timer starten
        _check_visual_context_drift._pending_face = cur_face
        _check_visual_context_drift._pending_since = now
    except Exception:
        pass
    return ""


_CHAT_UI_HTML = """<!doctype html>
<html lang="de"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOLOCH Cockpit</title>
<style>
  :root{
    --bg:#0a0a0d;--fg:#e6e6ee;--mute:#7a7a8a;--card:#13131a;--card2:#1a1a23;
    --border:#26262f;--border2:#363645;
    --guardian:#3673ce;--shadow:#7e3bce;--berserker:#c93838;
    --accent:var(--guardian);
    --ok:#5dc36b;--warn:#e6b84d;--err:#ff7676;
  }
  *{box-sizing:border-box}html,body{margin:0;padding:0;background:var(--bg);color:var(--fg);
    font:14px/1.45 system-ui,sans-serif;height:100%;overflow:hidden}
  .grid{display:grid;grid-template-rows:auto 1fr auto;height:100vh}
  /* HEADER */
  header{display:flex;align-items:center;gap:18px;padding:10px 16px;
    background:linear-gradient(180deg,#13131a,#0d0d12);border-bottom:1px solid var(--border)}
  header .logo{font:700 16px/1 system-ui;letter-spacing:1px;color:var(--accent)}
  header .pulse{width:8px;height:8px;border-radius:50%;background:var(--ok);
    box-shadow:0 0 8px var(--ok);animation:pulse 1.5s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  .stats{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--mute);flex:1}
  .stat{display:flex;flex-direction:column;line-height:1.1}
  .stat .v{color:var(--fg);font-weight:600;font-size:13px}
  .stat .v.zone-guardian{color:var(--guardian)}
  .stat .v.zone-shadow{color:var(--shadow)}
  .stat .v.zone-berserker{color:var(--berserker)}
  .stat .v.warn{color:var(--warn)}
  .stat .v.err{color:var(--err)}
  /* MAIN */
  main{display:grid;grid-template-columns:1.1fr 1fr;gap:1px;background:var(--border);min-height:0}
  .col{background:var(--bg);overflow-y:auto;padding:12px;display:flex;flex-direction:column;min-height:0}
  /* CHAT */
  .chat-col{gap:8px}
  .chat{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:8px;padding-right:4px;min-height:0}
  .msg{padding:9px 12px;border-radius:10px;max-width:90%;white-space:pre-wrap;word-wrap:break-word;font-size:13.5px}
  .me{align-self:flex-end;background:#192232;border:1px solid #2a3550}
  .moloch{align-self:flex-start;background:var(--card);border:1px solid var(--border)}
  .meta-line{font-size:11px;color:var(--mute);margin-top:2px;display:flex;gap:6px;align-items:center;flex-wrap:wrap}
  .meta-line button{background:transparent;border:1px solid var(--border2);color:var(--mute);
    font-size:10px;padding:1px 6px;border-radius:3px;cursor:pointer}
  .meta-line button:hover{color:var(--fg);border-color:var(--accent)}
  .crit-out{font-size:11px;padding:4px 6px;background:#1a1414;border-left:2px solid var(--berserker);
    color:#ffc8c8;margin-top:3px;border-radius:0 3px 3px 0}
  /* INPUT */
  .form{display:flex;gap:6px;border-top:1px solid var(--border);padding-top:8px;align-items:flex-end}
  textarea{flex:1;background:var(--card);color:var(--fg);border:1px solid var(--border);
    border-radius:8px;padding:8px;resize:none;font:14px system-ui;min-height:54px;max-height:160px}
  textarea:focus{outline:none;border-color:var(--accent)}
  .btn{background:var(--accent);color:white;border:0;border-radius:8px;padding:0 14px;cursor:pointer;
    font:600 13px system-ui;height:38px;min-width:38px}
  .btn:disabled{opacity:.5;cursor:not-allowed}
  .btn.icon{padding:0 10px;font-size:18px}
  .btn.mic{background:#2a2a35;color:var(--fg);border:1px solid var(--border2)}
  .btn.mic.recording{background:var(--berserker);animation:pulse 1s infinite}
  .toolbar{display:flex;gap:10px;align-items:center;font-size:12px;color:var(--mute);flex-wrap:wrap}
  .toolbar label{cursor:pointer;display:flex;gap:4px;align-items:center}
  .err{color:var(--err);font-size:12px}
  details{margin-top:4px;font-size:12px}
  details summary{cursor:pointer;color:var(--mute);user-select:none}
  details pre{background:var(--card);border:1px solid var(--border);padding:8px;border-radius:4px;
    font:11px/1.4 monospace;white-space:pre-wrap;max-height:240px;overflow-y:auto}
  /* TABS */
  .tabs-col{padding:0}
  .tabs-bar{display:flex;background:var(--card);border-bottom:1px solid var(--border)}
  .tab-btn{flex:1;background:transparent;border:0;color:var(--mute);padding:10px;cursor:pointer;
    font:600 12px system-ui;letter-spacing:.5px;border-bottom:2px solid transparent}
  .tab-btn.active{color:var(--fg);border-bottom-color:var(--accent);background:var(--bg)}
  .tab-content{flex:1;overflow-y:auto;padding:12px}
  .tab{display:none}.tab.active{display:block}
  /* AUDIT SUB-TABS (W13-W17) */
  .audit-subtabs{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap}
  .audit-subtab-btn{padding:6px 12px;cursor:pointer;border-radius:4px;border:1px solid var(--border);
    background:var(--card);color:var(--mute);font:600 11px system-ui;letter-spacing:.3px}
  .audit-subtab-btn.active{background:var(--accent);color:#000;border-color:var(--accent)}
  .audit-sub{display:none}.audit-sub.active{display:block}
  .audit-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px}
  .audit-card{min-height:80px;padding:8px;border-radius:4px;background:#1a1c23;border:1px solid var(--border);cursor:pointer;
    display:flex;flex-direction:column;gap:4px;font-size:11.5px}
  .audit-card:hover{border-color:var(--accent)}
  .audit-card-head{display:flex;align-items:center;gap:6px}
  .audit-led{width:12px;height:12px;border-radius:50%;display:inline-block;flex-shrink:0}
  .audit-led.pass{background:#5dc36b}
  .audit-led.warn{background:#e6b84d}
  .audit-led.fail{background:#ff7676}
  .audit-led.pending{background:#7a7a8a}
  .audit-card-name{font-weight:600;color:var(--fg);text-transform:uppercase;letter-spacing:.4px;font-size:11px}
  .audit-card-score{margin-left:auto;font-variant-numeric:tabular-nums;color:var(--mute)}
  .audit-card-detail{color:var(--mute);font-size:10.5px;line-height:1.3;overflow:hidden;
    display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
  .audit-modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.7);display:none;z-index:1000;
    align-items:center;justify-content:center}
  .audit-modal-bg.show{display:flex}
  .audit-modal{background:var(--card);border:1px solid var(--accent);border-radius:8px;padding:14px;
    max-width:80vw;max-height:80vh;overflow:auto;min-width:480px}
  .audit-modal pre{font:11px/1.4 monospace;color:var(--fg);white-space:pre-wrap;word-break:break-word}
  .audit-modal-close{float:right;background:transparent;border:0;color:var(--mute);cursor:pointer;font-size:18px}
  .audit-verify-btn{padding:10px 20px;background:var(--accent);color:#000;border:0;border-radius:4px;
    cursor:pointer;font:700 12px system-ui;letter-spacing:.5px;margin:8px 0}
  .audit-verify-btn:disabled{opacity:.5;cursor:not-allowed}
  .audit-self-summary{font-size:18px;line-height:1.4;color:var(--fg);padding:12px;background:#1a1c23;
    border-radius:6px;border-left:3px solid var(--accent);margin-bottom:10px}
  .audit-self-cols{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
  .audit-self-col h4{margin:0 0 6px;font:600 11px system-ui;color:var(--mute);letter-spacing:.5px;text-transform:uppercase}
  .audit-self-col ul{margin:0;padding-left:16px;font-size:11.5px;line-height:1.4}
  .audit-self-col li{margin-bottom:3px}
  .card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:10px;margin-bottom:8px}
  .card h3{margin:0 0 6px;font:600 12px system-ui;color:var(--mute);letter-spacing:.5px;text-transform:uppercase}
  .kv{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:12.5px}
  .kv .k{color:var(--mute)}
  .kv .v{color:var(--fg);text-align:right;font-variant-numeric:tabular-nums}
  .bar{height:6px;background:var(--border);border-radius:3px;overflow:hidden;margin-top:4px}
  .bar > span{display:block;height:100%;background:var(--accent);transition:width .3s}
  .bar.ram > span{background:var(--guardian)}
  .bar.cpu > span{background:var(--warn)}
  .pill{display:inline-block;padding:2px 6px;border-radius:10px;background:var(--border);font-size:10.5px;margin:1px}
  .pill.on{background:#1f3a1f;color:var(--ok)}
  .pill.off{background:#2a1414;color:var(--mute)}
  .rule{padding:6px 8px;background:#1a1c23;border-left:3px solid var(--guardian);
    border-radius:0 4px 4px 0;margin-bottom:5px;font-size:12px}
  .rule .t{color:var(--mute);font-size:11px}
  .rule .b{color:var(--fg);font-style:italic}
  .ev{font-size:11.5px;padding:3px 6px;border-bottom:1px solid var(--border);display:flex;gap:8px}
  .ev .ts{color:var(--mute);font-variant-numeric:tabular-nums;flex-shrink:0;width:48px}
  .ev .tp{color:var(--accent);width:60px;flex-shrink:0;font-size:10.5px}
  .ev .it{color:var(--fg);flex:1;word-wrap:break-word;min-width:0}
  .snap{width:100%;border-radius:6px;border:1px solid var(--border);background:#000;display:block}
  .snap-info{font-size:12px;color:var(--mute);margin-top:6px;text-align:center}
  /* FOOTER */
  footer{padding:6px 16px;background:#0d0d12;border-top:1px solid var(--border);
    font-size:11px;color:var(--mute);display:flex;gap:14px;flex-wrap:wrap}
  footer .err-msg{color:var(--err)}
  /* responsive */
  @media (max-width:900px){
    main{grid-template-columns:1fr;grid-template-rows:1fr 1fr}
  }
</style></head><body>
<div class="grid">
  <header>
    <div class="pulse" id="pulse"></div>
    <div class="logo">M.O.L.O.C.H.</div>
    <div class="stats">
      <div class="stat"><span>FPS</span><span class="v" id="s-fps">—</span></div>
      <div class="stat"><span>Person</span><span class="v" id="s-person">—</span></div>
      <div class="stat"><span>Face-ID</span><span class="v" id="s-face">—</span></div>
      <div class="stat"><span>Zone</span><span class="v" id="s-zone">—</span></div>
      <div class="stat"><span>Tension</span><span class="v" id="s-tension">—</span></div>
      <div class="stat"><span>CPU</span><span class="v" id="s-cpu">—</span></div>
      <div class="stat"><span>RAM</span><span class="v" id="s-ram">—</span></div>
      <div class="stat"><span>Provider</span><span class="v" id="s-prov">—</span></div>
      <div class="stat audit-stat" title="Audit-Ampel + Persona-Sparkline (W11)">
        <span>Audit</span>
        <span class="v" id="s-audit">
          <span id="audit-led" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#7a7a8a;margin-right:6px;vertical-align:middle"></span>
          <svg id="audit-spark" width="50" height="16" viewBox="0 0 50 16" style="vertical-align:middle">
            <polyline points="" fill="none" stroke="#5dc36b" stroke-width="1.5"/>
          </svg>
        </span>
      </div>
    </div>
  </header>

  <main>
    <!-- LINKS: CHAT -->
    <section class="col chat-col">
      <div class="chat" id="chat"></div>
      <details>
        <summary>System-Prompt anzeigen (was wird dem LLM injected)</summary>
        <pre id="sp">…</pre>
        <button class="btn" style="margin-top:6px;height:28px;font-size:11px" onclick="loadPrompt()">Refresh</button>
      </details>
      <div class="form">
        <textarea id="inp" placeholder="Schreib was. Enter = senden, Shift+Enter = Zeile." autofocus></textarea>
        <button class="btn icon mic" id="mic" title="Mikrofon (klick: an/aus)">🎙</button>
        <button class="btn" id="send">Senden</button>
      </div>
      <div class="toolbar">
        <label><input type="checkbox" id="local"> NPU lokal (qwen)</label>
        <label><input type="checkbox" id="reason"> reason_internal</label>
        <label><input type="checkbox" id="tts"> TTS Antwort sprechen</label>
        <span id="err" class="err"></span>
      </div>
    </section>

    <!-- RECHTS: TABS -->
    <section class="col tabs-col">
      <div class="tabs-bar">
        <button class="tab-btn active" data-tab="live">Live</button>
        <button class="tab-btn" data-tab="char">Charakter</button>
        <button class="tab-btn" data-tab="see">Sehen</button>
        <button class="tab-btn" data-tab="avatar">Avatar</button>
        <button class="tab-btn" data-tab="audit">Audit</button>
      </div>
      <div class="tab-content">
        <!-- LIVE TAB -->
        <div class="tab active" id="t-live">
          <div class="card"><h3>Pipeline</h3><div class="kv" id="live-pipeline"></div></div>
          <div class="card"><h3>Hardware</h3><div class="kv" id="live-hw"></div></div>
          <div class="card"><h3>Power</h3><div class="kv" id="live-power"></div></div>
          <div class="card"><h3>NPU Worker</h3><div id="live-workers"></div></div>
          <div class="card"><h3>Aktive Modelle</h3><div id="live-models"></div></div>
        </div>
        <!-- CHARAKTER TAB -->
        <div class="tab" id="t-char">
          <div class="card"><h3>Drift (rolling 30d)</h3><div class="kv" id="char-drift"></div></div>
          <div class="card"><h3>Top-Erlebnisse</h3><div id="char-top"></div></div>
          <div class="card"><h3>Aktive Verhaltensregeln</h3><div id="char-rules"></div></div>
          <div class="card"><h3>Letzte Journal-Events</h3><div id="char-journal"></div></div>
        </div>
        <!-- SEHEN TAB -->
        <div class="tab" id="t-see">
          <div class="card">
            <h3>Snapshot (live, 2s refresh)</h3>
            <img class="snap" id="snap" alt="Snapshot">
            <div class="snap-info" id="snap-info">…</div>
          </div>
        </div>
        <!-- AVATAR TAB -->
        <div class="tab" id="t-avatar">
          <iframe src="http://192.168.178.20:11800/"
                  style="width:100%;height:100%;min-height:600px;border:0;background:#0a0a0d"
                  title="MOLOCH Avatar"
                  allow="microphone; camera; autoplay"></iframe>
        </div>
        <!-- AUDIT TAB (W13-W17) — 4 Sub-Tabs ueber 24 Layer -->
        <div class="tab" id="t-audit">
          <div class="audit-subtabs">
            <button class="audit-subtab-btn active" data-sub="health">Health</button>
            <button class="audit-subtab-btn" data-sub="loop">Closed-Loop</button>
            <button class="audit-subtab-btn" data-sub="expr">Ausdruck</button>
            <button class="audit-subtab-btn" data-sub="self">Self-Awareness</button>
            <span style="margin-left:auto;font-size:11px;color:var(--mute);align-self:center">
              Overall <span id="audit-overall" style="font-weight:600"></span> ·
              Tier <span id="audit-tier">—</span> ·
              <span id="audit-updated">—</span>
              <button class="btn" style="height:22px;font-size:10px;margin-left:6px" onclick="auditRefresh()">Refresh</button>
            </span>
          </div>

          <!-- SUB: HEALTH (21 Cards) -->
          <div class="audit-sub active" id="sub-health">
            <div class="audit-grid" id="audit-health-grid"></div>
          </div>

          <!-- SUB: CLOSED-LOOP (7 Verifier) -->
          <div class="audit-sub" id="sub-loop">
            <div class="card">
              <h3>Closed-Loop Verifier</h3>
              <button class="audit-verify-btn" id="audit-verify-all-btn" onclick="auditVerifyAll()">
                JETZT ALLE VERIFIZIEREN
              </button>
              <span id="audit-verify-status" style="font-size:11px;color:var(--mute);margin-left:10px"></span>
            </div>
            <div class="audit-grid" id="audit-loop-grid"></div>
          </div>

          <!-- SUB: AUSDRUCK (5 Module) -->
          <div class="audit-sub" id="sub-expr">
            <div class="audit-grid" id="audit-expr-grid"></div>
          </div>

          <!-- SUB: SELF-AWARENESS -->
          <div class="audit-sub" id="sub-self">
            <div class="audit-self-summary" id="audit-self-summary">—</div>
            <div class="audit-self-cols">
              <div class="audit-self-col card">
                <h4>Ich kann</h4>
                <ul id="audit-self-can"></ul>
              </div>
              <div class="audit-self-col card">
                <h4>Ich kann NICHT</h4>
                <ul id="audit-self-cannot"></ul>
              </div>
              <div class="audit-self-col card">
                <h4>Eingeschraenkt</h4>
                <ul id="audit-self-degraded"></ul>
              </div>
            </div>
            <div class="card">
              <h3>Reflections</h3>
              <ul id="audit-self-reflections" style="font-size:12px;line-height:1.5"></ul>
            </div>
          </div>
        </div>

        <!-- DETAIL-MODAL fuer Audit-Cards -->
        <div class="audit-modal-bg" id="audit-modal-bg" onclick="if(event.target===this)auditCloseModal()">
          <div class="audit-modal">
            <button class="audit-modal-close" onclick="auditCloseModal()">×</button>
            <h3 id="audit-modal-title" style="margin:0 0 8px">Layer</h3>
            <pre id="audit-modal-body">…</pre>
          </div>
        </div>
      </div>
    </section>
  </main>

  <footer>
    <span id="f-info">…</span>
    <span id="fb-stats" style="margin-left:auto"></span>
    <span class="err-msg" id="f-err"></span>
  </footer>
</div>

<script>
// === DOM ===
const $ = id => document.getElementById(id);
const chat=$("chat"),inp=$("inp"),btnSend=$("send"),btnMic=$("mic"),
      cbLocal=$("local"),cbReason=$("reason"),cbTts=$("tts"),
      err=$("err"),sp=$("sp"),fInfo=$("f-info"),fErr=$("f-err");

// === HELPERS ===
function fmt(n,d=1){return (n==null||isNaN(n))?"—":Number(n).toFixed(d);}
function pct(n){return (n==null||isNaN(n))?"—":Math.round(n)+"%";}

let lastMolochAnswer = "";

function addMsg(role,text,meta){
  const m=document.createElement("div");m.className="msg "+(role==="me"?"me":"moloch");
  m.textContent=text;chat.appendChild(m);
  if(meta){
    const mm=document.createElement("div");mm.className="meta-line";
    mm.innerHTML=meta;
    if(role==="moloch"){
      const up=document.createElement("button");up.textContent="👍";up.title="gut — als Trainings-Sample";
        up.onclick=()=>doFeedback(text,"up",up,mm);
      const dn=document.createElement("button");dn.textContent="👎";dn.title="schlecht — als Negativ-Sample";
        dn.onclick=()=>doFeedback(text,"down",dn,mm);
      const cb=document.createElement("button");cb.textContent="[Critic]";cb.title="Auto-Bewertung durch PC-LLM";
        cb.onclick=()=>doCritic(text,mm);
      mm.appendChild(up);mm.appendChild(dn);mm.appendChild(cb);
    }
    chat.appendChild(mm);
  }
  chat.scrollTop=chat.scrollHeight;
}

async function doFeedback(text,label,btn,parent){
  btn.disabled=true;
  // andere Buttons (alle in mm, nur fuer diese Antwort) deaktivieren
  parent.querySelectorAll("button").forEach(b=>{ if(b!==btn) b.disabled=true; });
  try{
    const r=await fetch("/feedback",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({pi_response:text,label:label})});
    const j=await r.json();
    if(!r.ok) throw new Error(j.detail||r.statusText);
    btn.textContent = label==="up"?"👍✓":"👎✓";
    btn.style.background="var("+(label==="up"?"--ok":"--berserker")+")";btn.style.color="#000";
    refreshFeedbackStats();
  }catch(e){
    btn.textContent="!"+label;btn.disabled=false;
    parent.querySelectorAll("button").forEach(b=>b.disabled=false);
  }
}

async function refreshFeedbackStats(){
  try{const r=await fetch("/feedback_stats");if(!r.ok)return;
    const j=await r.json();
    const el=document.getElementById("fb-stats");
    if(el) el.textContent=`Pool: ${j.total} · 👍${j.thumbs_up} 👎${j.thumbs_down} · Critic ${j.critic} (${j.pending_review} pending)`;
  }catch(e){}
}

// === LIVE STATUS BAR + TAB ===
async function refreshLive(){
  try{
    const r=await fetch("/live");if(!r.ok)return;
    const j=await r.json();
    const fps=j.fps_total||0;
    $("s-fps").textContent=fmt(fps,1);
    $("s-person").textContent=j.person_detected?(j.face_id||"unbek"):"—";
    $("s-face").textContent=j.face_id||"—";
    const zone=(j.core||{}).zone||"guardian";
    const z=$("s-zone");z.textContent=zone;z.className="v zone-"+zone;
    document.documentElement.style.setProperty("--accent",`var(--${zone})`);
    const t=(j.core||{}).tension;
    $("s-tension").textContent=fmt(t,2);
    const cpu=j.watchdog?.cpu_temp;$("s-cpu").textContent=fmt(cpu,1)+"°";
    const ram=j.watchdog?.ram_percent;$("s-ram").textContent=pct(ram);

    // Pipeline card
    $("live-pipeline").innerHTML=`
      <span class="k">FPS total</span><span class="v">${fmt(fps,1)}</span>
      <span class="k">Frame-Age</span><span class="v">${fmt(j.frame_age,2)}s</span>
      <span class="k">Autonom</span><span class="v">${j.autonomous_mode?"ja":"nein"}</span>
      <span class="k">Tentakel</span><span class="v">${j.tentakel_enabled?"on":"off"}</span>
      <span class="k">Person</span><span class="v">${j.person_detected?"ja":"nein"}</span>
      <span class="k">Face</span><span class="v">${j.face_id||"—"} (${fmt(j.face_confidence,2)})</span>`;

    // Hardware card
    const w=j.watchdog||{};
    const cpuPct=Math.min(100,Math.max(0,(cpu-30)/(80-30)*100));
    $("live-hw").innerHTML=`
      <span class="k">CPU Temp</span><span class="v">${fmt(cpu,1)}°C</span>
      <span class="k"></span><span class="v"><div class="bar cpu"><span style="width:${cpuPct}%"></span></div></span>
      <span class="k">RAM</span><span class="v">${pct(ram)}</span>
      <span class="k"></span><span class="v"><div class="bar ram"><span style="width:${ram||0}%"></span></div></span>
      <span class="k">Kamera</span><span class="v">${w.camera_reachable?"ok":"weg"}</span>
      <span class="k">Throttled</span><span class="v">${w.throttled?"⚠ ja":"nein"}</span>`;

    // Power
    const p=j.power||{};
    $("live-power").innerHTML=`
      <span class="k">Quelle</span><span class="v">${p.power_source||"—"}</span>
      <span class="k">Watt</span><span class="v">${fmt(p.power_watts,2)} W</span>
      <span class="k">Batterie</span><span class="v">${pct(p.battery_pct)}</span>
      <span class="k">Lädt?</span><span class="v">${p.is_charging?"ja":"nein"}</span>
      <span class="k">Stecker</span><span class="v">${p.is_plugged_in?"drin":"raus"}</span>`;

    // Worker
    const wh=j.worker_health||{};
    $("live-workers").innerHTML=Object.keys(wh).map(k=>{
      const x=wh[k];const ok=x.running&&x.models_loaded;
      return `<div style="display:grid;grid-template-columns:auto 1fr auto;gap:8px;font-size:12px;padding:3px 0;border-bottom:1px solid var(--border)">
        <span class="pill ${ok?'on':'off'}">${ok?'●':'○'}</span>
        <span>${k}</span>
        <span style="color:var(--mute);font-variant-numeric:tabular-nums">${(x.total_inferences||0).toLocaleString()} · ${fmt(x.last_inference_ms,1)}ms · q${x.queue_size||0}</span>
      </div>`;
    }).join("")||"<span style='color:var(--mute)'>—</span>";

    // Active Models
    $("live-models").innerHTML=(j.active_models||[]).map(m=>`<span class="pill on">${m}</span>`).join("")||"<span style='color:var(--mute)'>keine</span>";
  }catch(e){console.warn("live fetch",e);}
}

// === CHAR TAB ===
async function refreshChar(){
  try{
    const r=await fetch("/personality");if(!r.ok)return;
    const j=await r.json();
    const d=(j.drift||{}).rolling||{};
    $("char-drift").innerHTML=`
      <span class="k">Mood-Baseline</span><span class="v">${fmt(d.mood_baseline,3)}</span>
      <span class="k">Energy-Baseline</span><span class="v">${fmt(d.energy_baseline,3)}</span>
      <span class="k">Dominance-Baseline</span><span class="v">${fmt(d.dominance_baseline,3)}</span>
      <span class="k">Updated</span><span class="v" style="font-size:10px">${(j.drift||{}).updated_at||"—"}</span>`;
    const top=(j.drift||{}).top||[];
    $("char-top").innerHTML=top.length?top.map(t=>
      `<div class="ev"><span class="tp">${fmt(t.weight,2)}</span><span class="it">${t.citation||"—"}</span></div>`
    ).join(""):"<span style='color:var(--mute);font-size:12px'>noch nichts destilliert</span>";
    const active=(j.patch||{}).active||[];
    $("char-rules").innerHTML=active.length?active.map(r=>
      `<div class="rule"><div class="t">Wenn ${r.trigger}</div><div class="b">→ ${r.behavior}</div></div>`
    ).join(""):"<span style='color:var(--mute);font-size:12px'>keine aktiven Regeln</span>";
    const journal=j.journal||[];
    $("char-journal").innerHTML=journal.slice(-12).reverse().map(e=>{
      const ts=(e.ts||"").slice(11,16);
      return `<div class="ev"><span class="ts">${ts}</span><span class="tp">${e.type||"?"}</span><span class="it">${e.interpretation||""}</span></div>`;
    }).join("")||"<span style='color:var(--mute)'>—</span>";
  }catch(e){console.warn("char fetch",e);}
}

// === SEHEN TAB ===
function refreshSnap(){
  const im=$("snap"),info=$("snap-info");
  im.src=`/snapshot.jpg?t=${Date.now()}`;
  im.onload=()=>info.textContent=`${im.naturalWidth}×${im.naturalHeight} · ${new Date().toLocaleTimeString()}`;
  im.onerror=()=>info.textContent="kein Snapshot (Service down?)";
}

// === CHAT ===
async function send(){
  const t=inp.value.trim();if(!t)return;err.textContent="";btnSend.disabled=true;
  addMsg("me",t);inp.value="";const t0=Date.now();
  try{
    const r=await fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({text:t,force_local:cbLocal.checked,use_reason:cbReason.checked})});
    if(!r.ok){const txt=await r.text();throw new Error(`HTTP ${r.status}: ${txt}`);}
    const j=await r.json();
    lastMolochAnswer=j.text;
    addMsg("moloch",j.text,`<span>${j.provider} · ${j.duration_ms}ms</span>`);
    if(cbTts.checked) doTts(j.text);
    fInfo.textContent=`last: ${j.provider} · ${j.duration_ms}ms`;
  }catch(e){err.textContent=e.message;fErr.textContent=e.message;
    addMsg("moloch","[Fehler] "+e.message);}
  finally{btnSend.disabled=false;refreshLive();inp.focus();}
}

async function doTts(text){
  try{await fetch("/tts",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({text})});}catch(e){console.warn("tts",e);}
}

async function doCritic(text,parent){
  parent.querySelector(".crit-out")?.remove();
  const out=document.createElement("div");out.className="crit-out";out.textContent="[Critic denkt…]";
  parent.appendChild(out);
  try{
    const r=await fetch("/critic_review",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({text})});
    const j=await r.json();
    out.innerHTML=`<b>Score ${j.score}/10</b> · ${j.critique||""}` +
      (j.better_response?`<br><i>Besser:</i> ${j.better_response}`:"");
  }catch(e){out.textContent="[Critic Fehler] "+e.message;}
}

async function loadPrompt(){
  sp.textContent="…";
  try{const r=await fetch("/system_prompt");const j=await r.json();
    sp.textContent=`# ${j.length} chars\\n\\n${j.system}`;
  }catch(e){sp.textContent="Fehler: "+e.message;}
}

async function loadHistory(){
  try{const r=await fetch("/history?n=10");const j=await r.json();
    for(const m of (j.messages||[])){
      const sender=m.sender==="user"?"me":"moloch";
      const ts=(m.ts||"").slice(11,16);
      addMsg(sender,m.text||"",`<span>${ts} · ${m.source||""}</span>`);
    }
  }catch(e){}
}

// === MIC (Browser SpeechRecognition) ===
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec=null,recOn=false;
btnMic.onclick=()=>{
  if(!SR){err.textContent="Browser hat keine SpeechRecognition (Chrome/Edge nehmen)";return;}
  if(!rec){
    rec=new SR();rec.lang="de-DE";rec.continuous=true;rec.interimResults=true;
    rec.onresult=ev=>{
      let interim="",finalT="";
      for(let i=ev.resultIndex;i<ev.results.length;i++){
        const r=ev.results[i];
        if(r.isFinal) finalT+=r[0].transcript; else interim+=r[0].transcript;
      }
      if(finalT) inp.value+=(inp.value?" ":"")+finalT.trim();
      // interim ignored — sonst springt's
    };
    rec.onerror=e=>{err.textContent="Mic: "+e.error;stopMic();};
    rec.onend=()=>{if(recOn){try{rec.start();}catch(e){stopMic();}}};
  }
  if(recOn){stopMic();}else{recOn=true;btnMic.classList.add("recording");try{rec.start();}catch(e){stopMic();}}
};
function stopMic(){recOn=false;btnMic.classList.remove("recording");try{rec&&rec.stop();}catch(e){}}

// === TABS ===
document.querySelectorAll(".tab-btn").forEach(b=>{
  b.onclick=()=>{
    document.querySelectorAll(".tab-btn").forEach(x=>x.classList.remove("active"));
    document.querySelectorAll(".tab").forEach(x=>x.classList.remove("active"));
    b.classList.add("active");
    $("t-"+b.dataset.tab).classList.add("active");
    if(b.dataset.tab==="char") refreshChar();
    if(b.dataset.tab==="see") refreshSnap();
  };
});

// === FOOTER STATUS ===
async function refreshProv(){
  try{const r=await fetch("/status");const j=await r.json();
    $("s-prov").textContent=j.last_provider||"—";
    fInfo.textContent=`mode: ${j.llm_mode} · ${j.request_count} req · tentakel ${j.tentacle?.fail_count||0} fails`;
  }catch(e){}
}

// === SEND BINDINGS ===
btnSend.onclick=send;
inp.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}});

// === AUDIT (W11) ===
let auditState=null;
function auditColor(s){
  s=(s||'').toLowerCase();
  if(s==='green'||s==='pass') return '#5dc36b';
  if(s==='warn'||s==='warning') return '#e6b84d';
  if(s==='red'||s==='fail'||s==='alert') return '#ff7676';
  return '#7a7a8a';
}
function auditRenderSparkline(spark){
  const svg=$("audit-spark");if(!svg)return;
  if(!spark||!spark.length){svg.querySelector('polyline').setAttribute('points','');return;}
  const W=50,H=16,n=spark.length;
  const minS=Math.min(...spark,0),maxS=Math.max(...spark,10);
  const pts=spark.map((v,i)=>{
    const x=(i/(Math.max(1,n-1)))*W;
    const y=H-((v-minS)/Math.max(0.01,maxS-minS))*H;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const avg=spark.reduce((a,b)=>a+b,0)/n;
  const stroke=avg>=7?'#5dc36b':(avg>=5?'#e6b84d':'#ff7676');
  const pl=svg.querySelector('polyline');pl.setAttribute('points',pts);pl.setAttribute('stroke',stroke);
}
function auditRenderTrend(spark){
  const svg=$("audit-trend");if(!svg||!spark)return;
  const pl=svg.querySelector('polyline');if(!pl)return;
  if(!spark.length){pl.setAttribute('points','');return;}
  const W=400,H=80,n=spark.length;
  const pts=spark.map((v,i)=>{
    const x=(i/(Math.max(1,n-1)))*W;
    const y=H-(Math.max(0,Math.min(10,v))/10)*H;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const avg=spark.reduce((a,b)=>a+b,0)/n;
  pl.setAttribute('points',pts);
  pl.setAttribute('stroke',avg>=7?'#5dc36b':(avg>=5?'#e6b84d':'#ff7676'));
}
// Layer-Gruppierung fuer Health-Sub-Tab (alle 21 Health-Layer)
const AUDIT_HEALTH_LAYERS = [
  'pi','pc','persona','mailbox','vision','npu','spotify','hardware',
  'pc_hardware','web_ui','personality','memory','tracking','autonomy',
  'awareness','voice','unconscious','bridge','tentacle','cross','self_diagnosis'
];
// Closed-Loop Verifier (W14)
const AUDIT_LOOP_VERIFIERS = ['ptz','led','fan','tts','spotify','memory','bridge'];
// Ausdruck-Module (W16 expression)
const AUDIT_EXPR_MODULES = [
  'tension_to_fan','mood_to_spotify','zone_to_led','berserker_strobo','tension_to_tts_volume'
];

function _ledClass(s){
  s=(s||'').toLowerCase();
  if(s==='green'||s==='pass') return 'pass';
  if(s==='warn'||s==='warning') return 'warn';
  if(s==='red'||s==='fail'||s==='alert') return 'fail';
  return 'pending';
}
function _shortDetail(L){
  if(!L) return '';
  if(L.summary) return String(L.summary).slice(0,80);
  if(L.message) return String(L.message).slice(0,80);
  if(L.detail){
    try{
      if(typeof L.detail==='string') return L.detail.slice(0,80);
      const keys=Object.keys(L.detail).slice(0,3);
      return keys.map(k=>`${k}=${JSON.stringify(L.detail[k]).slice(0,18)}`).join(' ');
    }catch(e){return '';}
  }
  if(L.status) return L.status;
  return '';
}
function _scoreStr(L){
  if(!L) return '—';
  if(L.score!==undefined&&L.max!==undefined) return `${L.score}/${L.max}`;
  if(L.avg!==undefined&&L.avg!==null) return `avg=${L.avg}`;
  return '';
}
function _ageStr(ts){
  if(!ts) return 'never';
  const d=Date.parse(ts);if(isNaN(d)) return ts;
  const sec=Math.max(0,(Date.now()-d)/1000);
  if(sec<60) return Math.round(sec)+'s ago';
  if(sec<3600) return Math.round(sec/60)+'min ago';
  if(sec<86400) return Math.round(sec/3600)+'h ago';
  return Math.round(sec/86400)+'d ago';
}

function auditRenderHealthGrid(layers){
  const grid=$("audit-health-grid");if(!grid)return;
  const cards=[];
  for(const name of AUDIT_HEALTH_LAYERS){
    const L=layers[name]||{};
    const status=L.status||'pending';
    cards.push(
      `<div class="audit-card" onclick="auditOpenModal('${name}')">
        <div class="audit-card-head">
          <span class="audit-led ${_ledClass(status)}"></span>
          <span class="audit-card-name">${name}</span>
          <span class="audit-card-score">${_scoreStr(L)}</span>
        </div>
        <div class="audit-card-detail">${_shortDetail(L)||'—'}</div>
      </div>`
    );
  }
  grid.innerHTML=cards.join('');
}

function auditRenderLoopGrid(layers,loopState){
  const grid=$("audit-loop-grid");if(!grid)return;
  const cards=[];
  const loops=(loopState&&loopState.verifiers)||{};
  for(const name of AUDIT_LOOP_VERIFIERS){
    const v=loops[name]||{};
    const status=v.status||'pending';
    const ts=v.last_verified||v.timestamp||v.ts;
    const detail=v.message||v.summary||v.detail_de||(v.error||'');
    cards.push(
      `<div class="audit-card" onclick="auditOpenModalLoop('${name}')">
        <div class="audit-card-head">
          <span class="audit-led ${_ledClass(status)}"></span>
          <span class="audit-card-name">${name}</span>
          <span class="audit-card-score">${status}</span>
        </div>
        <div class="audit-card-detail">Last verified: ${_ageStr(ts)}<br>${String(detail||'').slice(0,60)}</div>
      </div>`
    );
  }
  grid.innerHTML=cards.join('');
}

function auditRenderExprGrid(exprLayer){
  const grid=$("audit-expr-grid");if(!grid)return;
  const modules=(exprLayer&&exprLayer.modules)||{};
  const cards=[];
  for(const name of AUDIT_EXPR_MODULES){
    const m=modules[name]||{};
    const status=m.status||'pending';
    const event=m.subscribed_event||m.event||'—';
    const lastTs=m.last_action_ts||m.last_ts;
    const live=m.live_value!==undefined?String(m.live_value):(m.value!==undefined?String(m.value):'—');
    cards.push(
      `<div class="audit-card" onclick="auditOpenModalExpr('${name}')">
        <div class="audit-card-head">
          <span class="audit-led ${_ledClass(status)}"></span>
          <span class="audit-card-name">${name}</span>
        </div>
        <div class="audit-card-detail">
          event: ${event}<br>
          last: ${_ageStr(lastTs)} · live=${live}
        </div>
      </div>`
    );
  }
  grid.innerHTML=cards.join('');
}

function auditRenderSelf(state){
  const cap=(state.layers&&state.layers.capability)||{};
  const refl=(state.layers&&state.layers.reflection)||{};
  $("audit-self-summary").textContent=cap.summary_de||'(noch keine Selbstwahrnehmung)';
  function _fillList(id,items,fmt){
    const el=$(id);if(!el)return;
    if(!items||!items.length){el.innerHTML='<li style="color:var(--mute)">—</li>';return;}
    el.innerHTML=items.map(fmt).join('');
  }
  _fillList("audit-self-can",cap.can_do||[],x=>`<li>${typeof x==='string'?x:(x.name||JSON.stringify(x))}</li>`);
  _fillList("audit-self-cannot",cap.cannot_do||[],x=>{
    if(typeof x==='string') return `<li>${x}</li>`;
    return `<li>${x.name||'?'} <span style="color:var(--mute)">— ${x.reason||x.why||''}</span></li>`;
  });
  _fillList("audit-self-degraded",cap.degraded||[],x=>{
    if(typeof x==='string') return `<li>${x}</li>`;
    return `<li>${x.name||'?'} <span style="color:var(--mute)">— ${x.reason||x.why||''}</span></li>`;
  });
  _fillList("audit-self-reflections",refl.reflections_de||[],x=>`<li>${typeof x==='string'?x:JSON.stringify(x)}</li>`);
}

function auditApply(state){
  if(!state)return;auditState=state;
  // Header-LED
  const led=$("audit-led");if(led)led.style.background=auditColor(state.overall);
  // Sparkline aus persona
  const spark=(state.layers&&state.layers.persona&&state.layers.persona.sparkline)||[];
  auditRenderSparkline(spark);
  // Tab-Inhalt nur updaten wenn Tab aktiv
  if(!$("t-audit").classList.contains("active"))return;
  $("audit-overall").textContent=state.overall||'—';
  $("audit-overall").style.color=auditColor(state.overall);
  $("audit-tier").textContent=state.alarm_tier||'—';
  $("audit-updated").textContent=state.updated_at||'—';
  const layers=state.layers||{};
  // alle Sub-Tabs immer rendern (Daten ist schon da)
  auditRenderHealthGrid(layers);
  auditRenderExprGrid(layers.expression||{});
  auditRenderSelf(state);
  // Closed-Loop separat (eigene State-Datei)
  auditRefreshLoop();
}

// === MODAL ===
function auditOpenModal(layerName){
  const L=(auditState&&auditState.layers&&auditState.layers[layerName])||{};
  $("audit-modal-title").textContent='Layer: '+layerName;
  $("audit-modal-body").textContent=JSON.stringify(L,null,2);
  $("audit-modal-bg").classList.add('show');
}
function auditOpenModalLoop(verifier){
  const cl=window._auditLoopState||{};
  const v=(cl.verifiers&&cl.verifiers[verifier])||{};
  $("audit-modal-title").textContent='Closed-Loop: '+verifier;
  $("audit-modal-body").textContent=JSON.stringify(v,null,2);
  $("audit-modal-bg").classList.add('show');
}
function auditOpenModalExpr(mod){
  const e=(auditState&&auditState.layers&&auditState.layers.expression)||{};
  const m=(e.modules&&e.modules[mod])||{};
  $("audit-modal-title").textContent='Ausdruck: '+mod;
  $("audit-modal-body").textContent=JSON.stringify(m,null,2);
  $("audit-modal-bg").classList.add('show');
}
function auditCloseModal(){$("audit-modal-bg").classList.remove('show');}

// === SUB-TAB SWITCH ===
function auditSubSwitch(name){
  document.querySelectorAll('.audit-subtab-btn').forEach(b=>{
    b.classList.toggle('active', b.dataset.sub===name);
  });
  document.querySelectorAll('.audit-sub').forEach(s=>{
    s.classList.toggle('active', s.id==='sub-'+name);
  });
  if(name==='loop') auditRefreshLoop();
}
document.querySelectorAll('.audit-subtab-btn').forEach(b=>{
  b.addEventListener('click',()=>auditSubSwitch(b.dataset.sub));
});

// === CLOSED-LOOP ===
let _auditLoopPollTimer=null;
async function auditRefreshLoop(){
  try{
    const r=await fetch('/audit/verify_status');
    if(!r.ok) return;
    const j=await r.json();
    window._auditLoopState=j;
    auditRenderLoopGrid(auditState&&auditState.layers||{}, j);
    // Status-Anzeige
    const st=$("audit-verify-status");
    if(st){
      if(j.running){st.textContent='Run aktiv: '+(j.run_id||'')+' seit '+_ageStr(j.started_at);}
      else if(j.last_run_at){st.textContent='Letzter Run: '+_ageStr(j.last_run_at)+' ('+(j.overall||'?')+')';}
      else {st.textContent='Noch nie gelaufen';}
    }
    // Polling waehrend Run aktiv
    if(j.running && !_auditLoopPollTimer){
      _auditLoopPollTimer=setInterval(auditRefreshLoop, 5000);
    } else if(!j.running && _auditLoopPollTimer){
      clearInterval(_auditLoopPollTimer);_auditLoopPollTimer=null;
    }
  }catch(e){}
}

async function auditVerifyAll(){
  const btn=$("audit-verify-all-btn");if(btn)btn.disabled=true;
  try{
    const r=await fetch('/audit/verify',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({verify:'all'})});
    if(!r.ok){const j=await r.json().catch(()=>({}));throw new Error(j.detail||r.statusText);}
    setTimeout(auditRefreshLoop,1000);
  }catch(e){
    const st=$("audit-verify-status");if(st)st.textContent='Verify-Fehler: '+e.message;
  }finally{
    setTimeout(()=>{if(btn)btn.disabled=false;},3000);
  }
}

async function auditRefresh(){
  try{const r=await fetch('/mailbox/audit/state');if(r.ok){auditApply(await r.json());}}catch(e){}
}
function auditConnectSSE(){
  try{
    const es=new EventSource('/audit/stream');
    es.onmessage=(ev)=>{try{auditApply(JSON.parse(ev.data));}catch(e){}};
    es.onerror=()=>{es.close();setTimeout(auditConnectSSE,5000);};
  }catch(e){setTimeout(auditConnectSSE,5000);}
}

// === BOOT ===
loadHistory();refreshLive();refreshProv();loadPrompt();refreshFeedbackStats();
auditRefresh();auditConnectSSE();
setInterval(refreshLive,2000);
setInterval(refreshProv,15000);
setInterval(refreshFeedbackStats,30000);
setInterval(()=>{ if($("t-char").classList.contains("active")) refreshChar(); },5000);
setInterval(()=>{ if($("t-see").classList.contains("active")) refreshSnap(); },2500);
setInterval(()=>{ if($("t-audit").classList.contains("active")) auditRefresh(); },10000);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def chat_ui():
    """Pi-lokales Browser-Chat-Fenster — keine externen Abhaengigkeiten."""
    return _CHAT_UI_HTML


@app.get("/health")
def health():
    return {"status": "ok", "service": "moloch-chat-server"}


@app.get("/status")
def status():
    b = get_llm_bridge()
    cfg = _load_tentacle_cfg()
    return {
        "llm_mode": b._llm_mode,
        "ollama_available": b._ollama_available,
        "last_provider": b._last_provider,
        "request_count": b._request_count,
        "tentacle": {
            "enabled": cfg.get("enabled"),
            "host": cfg.get("host"),
            "fail_count": b._tentacle_fail_count,
            "backoff_until": b._tentacle_backoff_until,
            "model_cached": b._tentacle_model_cached,
        },
    }


def _trigger_spotify_year(year: int) -> bool:
    """Schreibe IPC-Cmd nach /tmp/moloch_cmd_*.json — Service polled alle 200ms."""
    try:
        cmd = {"action": "spotify_from_year", "year": int(year)}
        path = f"/tmp/moloch_cmd_{int(time.time() * 1000)}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cmd, f)
        logger.info(f"[BRIDGE] IPC spotify_from_year year={year} -> {path}")
        return True
    except Exception as e:
        logger.warning(f"[BRIDGE] IPC-Schreibfehler: {e}")
        return False


def _trigger_spotify_play_playlist(name_query: str) -> bool:
    """IPC-Trigger fuer fuzzy-matched Playlist-Name (Welle 7 Schritt 2.3)."""
    try:
        cmd = {"action": "spotify_play_playlist", "name_query": str(name_query)}
        path = f"/tmp/moloch_cmd_{int(time.time() * 1000)}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cmd, f)
        logger.info(f"[BRIDGE] IPC spotify_play_playlist name='{name_query}' -> {path}")
        return True
    except Exception as e:
        logger.warning(f"[BRIDGE] IPC-Schreibfehler: {e}")
        return False


@app.post("/chat")
def chat(req: ChatRequest):
    # User-Input ins gemeinsame Memory + EventBus (Browser-Chat synchron mit Voice)
    try:
        get_memory().save_message("user", req.text, source="chat_server")
        get_event_bus().publish(
            "conversation.user_said",
            {"text": req.text, "source": "chat_server"},
            source="chat_server", priority=5,
        )
    except Exception as e:
        logger.warning(f"Memory/Bus user-write Fehler: {e}")

    # Welle 6 Schritt 7 + Welle 7 Schritt 2.3 + Welle 12 Bug B: Music-IPC-Shortcuts.
    # Wenn music_query mit Year-Match ODER Playlist-Pattern ODER Action-Phrase
    # erkannt wird, triggern wir Spotify direkt via IPC statt durch die LLM-Kaskade.
    if not req.force_local and not req.use_reason:
        _ptype_quick = _classify_prompt_type(req.text)
        if _ptype_quick == "music_query":
            # 0. Action-Phrasen zuerst (Bug B: 'wechsel die musik', 'naechster track', etc.)
            _amatch = _match_music_action(req.text)
            if _amatch:
                _action, _reply = _amatch
                if _trigger_spotify_action(_action):
                    try:
                        get_memory().save_message("moloch", _reply, source="chat_server")
                    except Exception:
                        pass
                    return {
                        "text": _reply,
                        "provider": f"spotify_action_{_action}",
                        "duration_ms": 50,
                        "prompt_type": "music_action",
                        "pi_mood": _get_pi_mood_label(),
                    }
            # 1. Playlist-Recognition zuerst (spezifischer als Year)
            _pmatch = _PLAYLIST_RE.search(req.text)
            if _pmatch:
                _name = _pmatch.group(1).strip().rstrip(".!?,").strip()
                if _name and _trigger_spotify_play_playlist(_name):
                    _reply = f"Lege Playlist '{_name}' auf."
                    try:
                        get_memory().save_message("moloch", _reply, source="chat_server")
                    except Exception:
                        pass
                    return {
                        "text": _reply,
                        "provider": "spotify_action_playlist",
                        "duration_ms": 50,
                        "prompt_type": "music_query",
                        "pi_mood": _get_pi_mood_label(),
                    }
            # 2. Year-Filter Shortcut — Welle 19.6: web-prompt hat Vorrang
            # vor spotify_action_year (Pattern-Konflikt-Fix: "WGT 2026" darf
            # nicht als Year-Filter triggern, wenn web-Klassifikator greift)
            # W20a-A1: zusaetzlich Festival-Vorpruefung — wenn Festival-Keyword
            # im Text, year-Pattern ueberspringen unabhaengig von _ptype_quick
            # (greift fuer music_query-Klassifizierung wie "WGT 2026 lineup",
            # die nicht als web durchschluepft).
            _FESTIVAL_KEYWORDS_FOR_YEAR_BYPASS = (
                "wgt", "wave-gotik", "amphi", "m'era luna", "mera luna", "mera-luna",
            )
            _is_festival_text = any(
                k in req.text.lower() for k in _FESTIVAL_KEYWORDS_FOR_YEAR_BYPASS
            )
            _ymatch = (_YEAR_RE.search(req.text)
                       if (not _is_festival_text and _ptype_quick != "web")
                       else None)
            if _ymatch:
                _year = int(_ymatch.group(1))
                if _trigger_spotify_year(_year):
                    _reply = f"Spiele Favoriten aus {_year}. Augen auf die Tracks."
                    try:
                        get_memory().save_message("moloch", _reply, source="chat_server")
                    except Exception:
                        pass
                    return {
                        "text": _reply,
                        "provider": "spotify_action_year",
                        "duration_ms": 50,
                        "prompt_type": "music_query",
                        "pi_mood": _get_pi_mood_label(),
                    }

    # Character Journal: Konversation als charakter-formenden Event protokollieren
    try:
        from core.memory.character_journal import get_journal
        get_journal().write_event(
            type="chat",
            interpretation=f"Markus: {req.text[:80]}",
            context="src=chat_server",
        )
    except Exception as e:
        logger.debug(f"Journal user-hook Fehler: {e}")

    b = get_llm_bridge()
    t0 = time.monotonic()
    # Phase 4 Task 4c: Tension-Snapshot VOR LLM-Call fuer Delta-Messung im Journal
    try:
        from core.core_integrator import get_core_integrator
        _ci = get_core_integrator()
        _tension_before = float(_ci.get_tension()) if hasattr(_ci, "get_tension") else 0.0
    except Exception:
        _ci = None
        _tension_before = 0.0
    # Phase 3 Task 3d: Visual-Echo-Validator — Snapshot der Sichtsituation
    # speichern, BEVOR der LLM-Call laeuft. Nach dem Call wird verglichen.
    _check_visual_context_drift()
    # Phase 5e: Type-Routing aktiv. Statt force_tentacle-Hammer klassifizieren
    # wir den Prompt und ueberlassen _route_by_type() in local_llm_bridge.py
    # die Wahl zwischen NPU und Tentakel.
    # - force_local=True (User-Override): NPU pur, kein prompt_type
    # - sonst: Klassifikation entscheidet (hardware/simple -> NPU,
    #   complex -> Tentakel mit Fallback)
    prompt_type: Optional[str] = None
    if req.use_reason:
        out = b.reason_internal(req.text)
    else:
        prompt_type = _classify_prompt_type(req.text) if not req.force_local else None
        # Welle 20a: Specialist-Router fuer prompt_type=web_fetch — User
        # pastet URL -> /fetch holt page-content (BeautifulSoup) -> LLM
        # antwortet basierend auf TITEL/INHALT. Fail-soft: kein /fetch-Erfolg
        # -> Fall-through auf web-Branch (Search-Proxy).
        if prompt_type == "web_fetch" and not req.force_local:
            url = _extract_url(req.text)
            fetched = None
            if url:
                try:
                    fr = requests.post(
                        "http://192.168.178.20:11650/fetch",
                        json={"url": url, "max_chars": 8000},
                        timeout=25,
                    )
                    if fr.ok:
                        fetched = fr.json()
                        logger.info(
                            f"[W20a] /fetch {url} -> {fetched.get('chars', 0)} chars, "
                            f"cached={fetched.get('cached', False)}"
                        )
                    else:
                        logger.warning(f"[W20a] /fetch status={fr.status_code} url={url}")
                except Exception as e:
                    logger.warning(f"[W20a] /fetch error: {e}")
            if fetched and fetched.get("text"):
                question = req.text.replace(url, "").strip() if url else req.text
                if not question:
                    question = "Was steht auf dieser Seite?"
                augmented = (
                    f"URL: {fetched.get('url', url)}\n"
                    f"FINAL_URL: {fetched.get('final_url', url)}\n"
                    f"TITEL: {fetched.get('title', '')}\n"
                    f"INHALT:\n{fetched.get('text', '')}\n\n"
                    f"FRAGE: {question}\n\n"
                    f"Antworte basierend auf den TITEL/INHALT oben. "
                    f"Erfinde nichts — wenn die Antwort nicht im Text steht, sag das."
                )
                cfg = _load_tentacle_cfg() or {}
                web_model = (cfg.get("web_model") or "").strip()
                if web_model == "api_deepseek":
                    logger.info("[W20a] web_fetch -> DeepSeek-Cloud (api_deepseek)")
                    out = b._generate_deepseek(augmented, "", 600)
                else:
                    logger.info(f"[W20a] web_fetch -> Tentakel (model={web_model or 'default'})")
                    out = b.ask_external(augmented, force_local=False,
                                         force_tentacle=False,
                                         prompt_type="web_research")
            else:
                # Fail-soft: kein URL gefunden ODER /fetch fehlgeschlagen
                # -> Fall-through zum web-Branch (Search-Proxy)
                logger.warning(f"[W20a] web_fetch fail-soft auf web (url={url})")
                prompt_type = "web"

        # Welle 19: Specialist-Router fuer prompt_type=web — Search-Proxy
        # davorschalten + LLM mit echten WEB-RESULTS augmentieren.
        # Modell-Resolve via tentacle_llm.web_model. Fail-soft: Search-Proxy-
        # Timeout/Down -> weiter mit Original-Prompt ohne Augmentation.
        if prompt_type == "web" and not req.force_local:
            web_ctx = ""
            # Welle 20a: Festival-Anfragen brauchen Volltext (Bands/Lineup
            # stehen meist nur auf der Festival-Seite, nicht im Snippet).
            festival_keywords = ("wgt", "wave-gotik", "amphi", "m'era luna",
                                 "mera luna", "mera-luna")
            is_festival = any(k in req.text.lower() for k in festival_keywords)
            # Welle 20a.5: Query-Refinement — Festival-Name -> site:-Filter
            # macht DDG-Treffer praeziser (offizielle Domain statt Aggregator).
            festival_site_map = {
                "wgt": "site:wave-gotik-treffen.de",
                "wave-gotik": "site:wave-gotik-treffen.de",
                "amphi": "site:amphi-festival.de",
                "m'era luna": "site:meraluna.de",
                "mera luna": "site:meraluna.de",
                "mera-luna": "site:meraluna.de",
            }
            search_query = req.text
            text_low = req.text.lower()
            for k, site_filter in festival_site_map.items():
                if k in text_low and "site:" not in search_query:
                    search_query = f"{req.text} {site_filter}"
                    logger.info(f"[W20a.5] site:-Filter -> '{search_query}'")
                    break
            try:
                sr = requests.post(
                    "http://192.168.178.20:11650/search",
                    json={"query": search_query, "max_results": 5},
                    timeout=15,
                )
                if sr.ok:
                    data = sr.json()
                    results = data.get("results", []) or []
                    if results:
                        web_ctx = "WEB-RESULTS:\n" + "\n".join(
                            f"- {r.get('title')} | {r.get('url')} | {r.get('snippet','')[:200]}"
                            for r in results[:5]
                        )
                        logger.info(f"[W19] search_proxy: {len(results)} results, {len(web_ctx)} chars")
                        # Welle 20a: bei Festival-Anfrage Top-Result-URL via /fetch
                        # holen und als VOLLTEXT anhaengen
                        if is_festival and results[0].get("url"):
                            try:
                                fr = requests.post(
                                    "http://192.168.178.20:11650/fetch",
                                    json={"url": results[0]["url"], "max_chars": 6000},
                                    timeout=25,
                                )
                                if fr.ok:
                                    fetched_top = fr.json()
                                    if fetched_top.get("text"):
                                        web_ctx += (
                                            f"\n\nVOLLTEXT TOP-RESULT "
                                            f"({fetched_top.get('title', '')}):\n"
                                            f"{fetched_top.get('text', '')}"
                                        )
                                        logger.info(
                                            f"[W20a] festival /fetch -> "
                                            f"{fetched_top.get('chars', 0)} chars"
                                        )
                            except Exception as e:
                                logger.debug(f"[W20a] festival /fetch fail: {e}")
                else:
                    logger.warning(f"[W19] search_proxy status={sr.status_code}")
            except Exception as e:
                logger.warning(f"[W19] search_proxy timeout/fail: {e}")
                # fail-soft: weiter mit Original-Prompt ohne Augmentation

            if web_ctx:
                augmented = (
                    f"{web_ctx}\n\nFRAGE: {req.text}\n\n"
                    "Antworte basierend auf den WEB-RESULTS oben. Nutze die echten URLs."
                )
            else:
                augmented = req.text

            # Modell-Resolve: tentacle_llm.web_model entscheidet ueber Pfad.
            # "api_deepseek" -> DeepSeek-Cloud (b._generate_deepseek).
            # Sonst -> Tentakel-LLM mit prompt_type=web_research (existing path).
            cfg = _load_tentacle_cfg() or {}
            web_model = (cfg.get("web_model") or "").strip()
            if web_model == "api_deepseek":
                logger.info("[W19] web -> DeepSeek-Cloud (api_deepseek)")
                out = b._generate_deepseek(augmented, "", 600)
            else:
                logger.info(f"[W19] web -> Tentakel (model={web_model or 'default'})")
                out = b.ask_external(augmented, force_local=False,
                                     force_tentacle=False,
                                     prompt_type="web_research")
        else:
            out = b.ask_external(req.text, force_local=req.force_local,
                                 force_tentacle=False,
                                 prompt_type=prompt_type)
    dur_ms = int((time.monotonic() - t0) * 1000)
    # Visual-Echo: bei Sicht-Drift waehrend des LLM-Calls -> Disclaimer prepend
    if out:
        _drift_disclaimer = _check_visual_context_drift()
        if _drift_disclaimer:
            out = _drift_disclaimer + out
            logger.info("visual_echo: drift detected, disclaimer prepended")
    if out is None:
        # Tentakel-offline-Fall (force_tentacle ohne PC erreichbar): ehrliche Meldung
        if b._last_provider == "tentacle_offline":
            raise HTTPException(
                503,
                "Rechner aus oder Tentakel nicht erreichbar — Moloch kann gerade nicht reden. "
                "Wenn der Rechner laeuft: Ollama-Service pruefen."
            )
        raise HTTPException(503, "Bridge gibt None (Stille)")

    # Moloch-Antwort ins gemeinsame Memory + EventBus
    try:
        get_memory().save_message("moloch", out, source="chat_server")
        get_event_bus().publish(
            "conversation.moloch_said",
            {"text": out, "source": "chat_server", "provider": b._last_provider},
            source="chat_server", priority=5,
        )
    except Exception as e:
        logger.warning(f"Memory/Bus moloch-write Fehler: {e}")

    # Character Journal: Eigene Antwort als charakter-formenden Event protokollieren
    # Phase 4 Task 4c: tension_delta = tension_after - tension_before messen
    try:
        _tension_after = float(_ci.get_tension()) if _ci is not None and hasattr(_ci, "get_tension") else 0.0
        _tension_delta = round(_tension_after - _tension_before, 3)
    except Exception:
        _tension_delta = 0.0
    try:
        from core.memory.character_journal import get_journal
        get_journal().write_event(
            type="chat",
            interpretation=f"Moloch: {out[:80]}",
            tension_delta=_tension_delta,
            context=f"provider={b._last_provider}",
            tags=["chat"],
        )
    except Exception as e:
        logger.debug(f"Journal moloch-hook Fehler: {e}")

    # Welle 10: Last-Turn-Hook fuer PC-Persona-Validator
    try:
        _write_last_turn_json(
            user_text=req.text,
            response_text=out,
            prompt_type=prompt_type,
            provider=b._last_provider,
            duration_ms=dur_ms,
        )
    except Exception as e:
        logger.debug(f"last_turn-Hook Fehler: {e}")

    return {
        "text": out,
        "provider": b._last_provider,
        "duration_ms": dur_ms,
        "prompt_type": prompt_type,
        "pi_mood": _get_pi_mood_label(),
    }




@app.get("/history")
def history(n: int = 20):
    """Letzte N Konversations-Turns aus persistentem Memory (Cross-Channel:
    Browser-Chats UND Pi-Voice UND Test-Calls — alles zusammen)."""
    try:
        from core.longterm_memory import get_memory
        msgs = get_memory().get_recent_messages(n=n) or []
        return {"count": len(msgs), "messages": msgs}
    except Exception as e:
        raise HTTPException(500, f"Memory-Lesefehler: {e}")


# ============================================================================
# COCKPIT-Endpoints (Welle 2.5 — Markus' Wunsch: GUI-Spiegel im Browser)
# ============================================================================

@app.get("/live")
def live_status():
    """Cockpit Live-Tab: alles aus moloch_status.json + worker + power + watchdog."""
    try:
        with open("/dev/shm/moloch_status.json", "r") as f:
            d = json.load(f)
        # Slim down: nur was UI braucht
        out = {
            "fps_total": d.get("fps", {}).get("total", 0),
            "person_detected": d.get("person_detected", False),
            "face_detected": d.get("face_detected", False),
            "face_id": d.get("face_id"),
            "face_confidence": d.get("face_confidence", 0.0),
            "active_models": d.get("active_models", []),
            "autonomous_mode": d.get("autonomous_mode", False),
            "moloch_has_control": d.get("moloch_has_control", False),
            "tentakel_enabled": d.get("tentakel_enabled", False),
            "frame_age": d.get("frame_age", 0.0),
            "power": d.get("power", {}),
            "watchdog": {
                "cpu_temp": d.get("watchdog", {}).get("cpu_temp"),
                "ram_percent": d.get("watchdog", {}).get("ram_percent"),
                "camera_reachable": d.get("watchdog", {}).get("camera_reachable"),
                "throttled": d.get("watchdog", {}).get("throttled"),
                "warnings": d.get("watchdog", {}).get("warnings", []),
            },
            "worker_health": d.get("worker_health", {}),
            "core": d.get("core", {}),
        }
        return out
    except Exception as e:
        raise HTTPException(500, f"Status read error: {e}")


@app.get("/personality")
def personality_view():
    """Cockpit Charakter-Tab: Drift + Patch + letzte 10 Journal-Events."""
    out = {"drift": {}, "patch": {}, "journal": []}
    try:
        from core.autonomy.character_distiller import get_distiller
        d = get_distiller().get_drift() or {}
        out["drift"] = {
            "rolling": d.get("rolling_drift") or {},
            "top": (d.get("recency_weighted_top") or [])[:5],
            "updated_at": d.get("updated_at"),
        }
    except Exception as e:
        out["drift_error"] = str(e)
    try:
        from core.memory.character_patch import get_patch
        p = get_patch()
        out["patch"] = {
            "state": p.get_state(),
            "active": p.get_active_rules(),
            "pending": p.get_pending_rules(),
        }
    except Exception as e:
        out["patch_error"] = str(e)
    try:
        from core.memory.character_journal import get_journal
        out["journal"] = get_journal().read_recent(15)
    except Exception as e:
        out["journal_error"] = str(e)
    return out


@app.get("/state_full")
def state_full():
    """Aggregierter Komplett-Snapshot fuer externe Visualisierung (PC-Auge).

    Ein einziger Polling-Endpoint statt /live + /personality + /feedback_stats +
    Bridge-Statuse. Strukturiert in Sektionen (system / pipeline / vision / ptz /
    tracker / personality / llm / audio / memory / events / spatial). PC kann
    selektiv rendern, alle Daten sind ein Roundtrip entfernt.

    Schema-Stabilitaet: alle Sektionen sind dicts; einzelne keys koennen fehlen.
    Konsumenten muessen nur dict.get(...) nutzen, kein hartes Schema-Lock.
    """
    out = {"ts": time.time(), "schema_version": 1}

    # ---------- moloch_status.json ----------
    try:
        with open("/dev/shm/moloch_status.json", "r") as f:
            d = json.load(f)
    except Exception as e:
        d = {}
        out["status_read_error"] = str(e)

    # ---------- system ----------
    out["system"] = {
        "fps": d.get("fps", {}),
        "frame_age": d.get("frame_age", 0.0),
        "frozen_restarts": d.get("frozen_restarts", 0),
        "watchdog": d.get("watchdog", {}),
        "power": d.get("power", {}),
        "pipeline_alive": d.get("pipeline_alive", False),
        "mode": d.get("mode", ""),
    }

    # ---------- pipeline ----------
    out["pipeline"] = {
        "active_models": d.get("active_models", []),
        "npu_stage": d.get("npu_stage"),
        "npu_stage_since": d.get("npu_stage_since"),
        "npu_sched_mode": d.get("npu_sched_mode"),
        "npu_paused": d.get("npu_paused", False),
        "thresholds": d.get("thresholds", {}),
        "worker_health": d.get("worker_health", {}),
        "perception": d.get("perception", {}),
    }

    # ---------- vision ----------
    out["vision"] = {
        "person_detected": d.get("person_detected", False),
        "face_detected": d.get("face_detected", False),
        "face_id": d.get("face_id"),
        "face_confidence": d.get("face_confidence", 0.0),
        "face_similarity": d.get("face_similarity", 0.0),
        "face_lock_active": d.get("face_lock_active", False),
        "panel_detections": d.get("panel_detections", []),
        "scrfd_active": d.get("scrfd_active", False),
        "arcface_active": d.get("arcface_active", False),
        "pose_active": d.get("pose_active", False),
        "person_reid_active": d.get("person_reid_active", False),
        "yolo_active": d.get("yolo_active", False),
        "hand_active": d.get("hand_active", False),
    }

    # ---------- ptz ----------
    out["ptz"] = {
        **(d.get("ptz") or {}),
        "arbiter_mode": d.get("ptz_arbiter_mode"),
        "last_switch": d.get("ptz_last_switch"),
        "switch_reason": d.get("ptz_switch_reason"),
        "last_known_pan": d.get("last_known_pan"),
        "last_known_tilt": d.get("last_known_tilt"),
    }

    # ---------- tracker / control ----------
    out["tracker"] = {
        "moloch_tracking": d.get("moloch_tracking", False),
        "moloch_has_control": d.get("moloch_has_control", False),
        "autonomous_mode": d.get("autonomous_mode", False),
        "manual_mode": d.get("manual_mode", False),
        "smart_search_patrol_ready": d.get("smart_search_patrol_ready", False),
        "cam_smart_tracking": d.get("cam_smart_tracking", False),
    }

    # ---------- personality ----------
    pers = {
        "tension": d.get("tension"),
        "personality_mode": d.get("personality_mode"),
        "led_personality_mode": d.get("led_personality_mode"),
        "core": d.get("core", {}),
    }
    try:
        from core.autonomy.character_distiller import get_distiller
        drift = get_distiller().get_drift() or {}
        pers["drift"] = {
            "rolling": drift.get("rolling_drift") or {},
            "top": (drift.get("recency_weighted_top") or [])[:5],
            "updated_at": drift.get("updated_at"),
        }
    except Exception as e:
        pers["drift_error"] = str(e)
    try:
        from core.memory.character_patch import get_patch
        p = get_patch()
        pers["patch"] = {
            "state": p.get_state(),
            "active": p.get_active_rules(),
            "pending_count": len(p.get_pending_rules() or []),
        }
    except Exception as e:
        pers["patch_error"] = str(e)
    try:
        from core.memory.character_journal import get_journal
        pers["journal_recent"] = get_journal().read_recent(10)
    except Exception as e:
        pers["journal_error"] = str(e)
    out["personality"] = pers

    # ---------- llm ----------
    llm = {
        "ollama_running": d.get("llm_ollama_running", False),
        "provider": d.get("llm_provider"),
        "tentakel_enabled": d.get("tentakel_enabled", False),
    }
    try:
        from core.autonomy.local_llm_bridge import _get_active_profile
        prof = _get_active_profile()
        if prof:
            llm["active_profile"] = {
                "system_preview": (prof.get("system") or "")[:80],
                "max_tokens": prof.get("max_tokens"),
                "temperature": prof.get("temperature"),
                "include_live_context": prof.get("include_live_context", False),
            }
    except Exception as e:
        llm["profile_error"] = str(e)
    try:
        from core.bridge.critic_client import get_critic_client
        llm["critic"] = get_critic_client().get_state()
    except Exception as e:
        llm["critic_error"] = str(e)
    try:
        from core.bridge.adapter_inference_client import get_adapter_client
        llm["adapter"] = get_adapter_client().get_state()
    except Exception as e:
        llm["adapter_error"] = str(e)
    out["llm"] = llm

    # ---------- audio ----------
    out["audio"] = {
        "voice": d.get("voice", {}),
        "audio_meter": d.get("audio", {}),
        "music": d.get("music", {}),
        "spotify": d.get("spotify", {}),
        "silence_level": d.get("silence_level", 0),
    }

    # ---------- memory ----------
    mem = {
        "introspection": d.get("introspection", {}),
    }
    try:
        from core.memory.feedback_store import get_feedback_store
        mem["feedback_stats"] = get_feedback_store().get_state()
    except Exception as e:
        mem["feedback_stats_error"] = str(e)
    try:
        wh = d.get("worker_health", {}) or {}
        fw = wh.get("FaceWorker") or {}
        mem["face_db_entries"] = fw.get("face_db_entries", 0)
    except Exception:
        mem["face_db_entries"] = 0
    out["memory"] = mem

    # ---------- events ----------
    out["events"] = {
        "bridge": d.get("bridge", {}),
        "bridge_decisions": d.get("bridge_decisions", []),
        "bus_stats": d.get("bus_stats", {}),
    }

    # ---------- spatial ----------
    out["spatial"] = d.get("spatial", {})

    # ---------- bridge / cloud ----------
    out["cloud"] = d.get("cloud", {})

    return out


@app.get("/cross_status")
def cross_status():
    """Pi-Sicht der Cross-Session-Verbindung — fuer PC-Side Remote-Abfrage.

    Liefert: letzte Heartbeats, aktuelle PC-Endpoint-States, recent Transitions,
    Monitor-Uptime aus dem persistent JSONL-Log. PC-Side (oder Markus) kann
    via curl http://192.168.178.30:9100/cross_status den Stand checken ohne
    SCP des Log-Files.

    Markus' Direktive 27.04: 'autonome Verbindung, ihr beiden checkt was der
    andere gemacht hat'. Dieser Endpoint ist die Pi-Side-Reflection.
    """
    log_path = "/mnt/moloch-data/memory/cross_session_log.jsonl"
    out = {
        "pi_now": time.time(),
        "log_path": log_path,
        "monitor_active": False,
        "last_heartbeat": None,
        "last_monitor_start": None,
        "recent_heartbeats_n": 0,
        "current_pc": {},
        "transitions_recent": [],
        "topics_acked": [],
    }
    try:
        if not os.path.exists(log_path):
            out["error"] = "log_file_missing"
            return out
        # Tail letzte 200 Zeilen lesen — billig genug
        with open(log_path, "rb") as f:
            f.seek(0, 2)  # end
            size = f.tell()
            chunk = min(size, 64 * 1024)
            f.seek(-chunk, 2)
            tail = f.read().decode("utf-8", errors="replace")
        lines = [ln for ln in tail.splitlines() if ln.strip()]
        # Wenn wir mitten in einer Zeile starten: erste verwerfen
        if size > 64 * 1024 and lines:
            lines = lines[1:]
        heartbeats = []
        last_start = None
        topics_acked: List[str] = []
        for ln in lines[-200:]:
            try:
                d = json.loads(ln)
            except Exception:
                continue
            t = d.get("type")
            if t == "monitor_start":
                last_start = d
            elif t == "heartbeat":
                heartbeats.append(d)
            for tk in d.get("topics_acked", []) or []:
                if tk not in topics_acked:
                    topics_acked.append(tk)
            for tk in d.get("triggers_acked", []) or []:
                if tk not in topics_acked:
                    topics_acked.append(tk)

        out["recent_heartbeats_n"] = len(heartbeats)
        out["topics_acked"] = topics_acked[-10:]
        if last_start:
            out["last_monitor_start"] = {
                "iso": last_start.get("iso"),
                "ts": last_start.get("ts"),
                "boot_id_short": (last_start.get("boot_id", "") or "")[:16],
            }
        if heartbeats:
            last = heartbeats[-1]
            out["last_heartbeat"] = {
                "iso": last.get("iso"),
                "ts": last.get("ts"),
                "iter": last.get("iter"),
            }
            out["current_pc"] = last.get("pc", {})
            out["pi_self"] = last.get("pi_self", {})
            # Monitor active wenn letzter heartbeat innerhalb 90s
            age = time.time() - (last.get("ts") or 0)
            out["monitor_active"] = age < 90
            out["last_heartbeat_age_s"] = round(age, 1)

            # Transitions in den letzten ~50 heartbeats finden
            prev = {}
            for hb in heartbeats[-50:]:
                pc = hb.get("pc", {})
                for name, r in pc.items():
                    cur_ok = r.get("ok")
                    if name in prev and prev[name] != cur_ok:
                        out["transitions_recent"].append({
                            "iso": hb.get("iso"),
                            "endpoint": name,
                            "to": "UP" if cur_ok else "DOWN",
                            "recovered_after_s": r.get("recovered_after_s"),
                        })
                    prev[name] = cur_ok
            out["transitions_recent"] = out["transitions_recent"][-20:]

        return out
    except Exception as e:
        out["error"] = str(e)[:200]
        return out


@app.get("/snapshot.jpg")
def snapshot_jpg():
    """Cockpit Sehen-Tab: aktueller Frame aus SHM als JPEG."""
    try:
        import numpy as np
        import cv2
        fd = os.open("/dev/shm/moloch_frame", os.O_RDONLY)
        size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
        h, w, c, seq, ts = struct.unpack("<IIIId", mm[:24])
        if h == 0 or w == 0:
            mm.close(); os.close(fd)
            raise HTTPException(503, "Frame leer")
        data = np.frombuffer(mm[24:24 + h * w * c], dtype=np.uint8).reshape(h, w, c)
        mm.close(); os.close(fd)
        # SHM ist RGB, JPEG braucht BGR
        bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            raise HTTPException(500, "JPEG encode failed")
        return Response(content=buf.tobytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Snapshot error: {e}")


class TextOnly(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


@app.post("/critic_review")
def critic_review(req: TextOnly):
    """Letzte Moloch-Antwort durch Critic bewerten lassen."""
    try:
        from core.bridge.critic_client import get_critic_client
        from core.autonomy.character_distiller import get_distiller
        from core.memory.character_patch import get_patch
        char_state = {
            "rolling_drift": (get_distiller().get_drift() or {}).get("rolling_drift", {}),
            "active_rules": get_patch().get_active_rules(),
        }
        # Letzte User-Frage als Situation
        situation = "(Browser-Chat — Cockpit Critic-Review)"
        try:
            msgs = get_memory().get_recent_messages(n=4) or []
            user_msgs = [m for m in msgs if m.get("sender") == "user"]
            if user_msgs:
                situation = f"Markus fragte: {user_msgs[-1].get('text', '')[:200]}"
        except Exception:
            pass
        result = get_critic_client().evaluate(
            situation=situation, pi_response=req.text, character_state=char_state,
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Critic error: {e}")


@app.post("/tts")
def tts_speak(req: TextOnly):
    """Text durch Pi-Piper sprechen (PersonalityEngine.speak)."""
    try:
        from core.personality.personality_engine import get_personality_engine
        ok = get_personality_engine().speak(req.text)
        return {"spoken": bool(ok)}
    except Exception as e:
        raise HTTPException(500, f"TTS error: {e}")


class FeedbackRequest(BaseModel):
    pi_response: str = Field(..., min_length=1, max_length=4000)
    label: str = Field(..., pattern="^(up|down)$")
    situation: str = Field("", max_length=500)


@app.post("/feedback")
def feedback_thumbs(req: FeedbackRequest):
    """W3.3 Cockpit-Thumbs: 👍/👎 als sofort-entschiedenes Sample.

    Schreibt in feedback_store als source='thumbs_up' (approved=True) oder
    'thumbs_down' (approved=False). Wird vom LoRA-Trainer beruecksichtigt.
    """
    try:
        from core.memory.feedback_store import get_feedback_store
        # Falls keine Situation uebergeben: aus letzter User-History rausziehen
        situation = req.situation
        if not situation:
            try:
                msgs = get_memory().get_recent_messages(n=4) or []
                user_msgs = [m for m in msgs if m.get("sender") == "user"]
                if user_msgs:
                    situation = f"Markus fragte: {user_msgs[-1].get('text', '')[:200]}"
            except Exception:
                situation = "(Cockpit-Feedback ohne Situation-Kontext)"
        sid = get_feedback_store().add_thumbs(
            situation=situation, pi_response=req.pi_response, label=req.label,
            source_channel="cockpit",
        )
        if not sid:
            raise HTTPException(400, "feedback_store rejected sample")
        return {"sample_id": sid, "label": req.label, "stored": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Feedback error: {e}")


@app.get("/feedback_stats")
def feedback_stats():
    """Pool-Status fuer Cockpit-Anzeige (sample-count + pending)."""
    try:
        from core.memory.feedback_store import get_feedback_store
        return get_feedback_store().get_state()
    except Exception as e:
        raise HTTPException(500, f"Feedback stats error: {e}")


@app.get("/feedback_export")
def feedback_export():
    """Antwort auf PC-Mailbox-Anfrage 709512f / docs/PC_TO_PI.md.

    Roher Stream der finetune_samples.jsonl als ndjson — PC kann via
    `curl -o samples.jsonl http://192.168.178.30:9100/feedback_export`
    pullen ohne SSH/scp Setup.

    Antwortet immer mit aktuellem Stand, kein Caching, keine Pagination
    (Pool ist klein, < 10k Eintraege fuer Phase 1).
    """
    try:
        from core.memory.feedback_store import POOL_PATH
        if not os.path.exists(POOL_PATH):
            return Response(content="", media_type="application/x-ndjson")
        with open(POOL_PATH, "rb") as f:
            data = f.read()
        return Response(
            content=data,
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": 'attachment; filename="finetune_samples.jsonl"',
            },
        )
    except Exception as e:
        raise HTTPException(500, f"Feedback export error: {e}")


# ============================================================================
# W17 Self-Awareness — Capability Injection in System-Prompt (30s Cache)
# ============================================================================

_CAP_CACHE: Dict[str, object] = {"ts": 0.0, "summary": "", "reflections": []}
_CAP_CACHE_TTL_S: float = 30.0


def _get_capabilities_cached() -> Tuple[str, list]:
    """Cached Capability-Summary + Top-3 Failure-Reflections.

    30s TTL — Capability-Inventory liest /dev/shm/audit_state.json und
    sollte nicht bei jedem Chat neu erhoben werden. Best-effort: bei
    Fehlern gibt es leere Strings statt Crash.
    """
    now = time.time()
    cached_ts = float(_CAP_CACHE["ts"])  # type: ignore[arg-type]
    cached_summary = str(_CAP_CACHE["summary"])  # type: ignore[arg-type]
    if now - cached_ts < _CAP_CACHE_TTL_S and cached_summary:
        return cached_summary, list(_CAP_CACHE["reflections"])  # type: ignore[arg-type]
    summary = ""
    reflections: list = []
    try:
        from core.audit.self_awareness.capability_inventory import collect_capabilities
        cap = collect_capabilities() or {}
        summary = (cap.get("summary_de") or "").strip()
    except Exception as e:
        logger.debug(f"capability_inject: collect failed: {e}")
    try:
        from core.audit.self_awareness.failure_reflection import reflect_on_failures
        refl = reflect_on_failures() or {}
        if (refl.get("status") or "").upper() in ("WARN", "FAIL"):
            reflections = list(refl.get("reflections_de") or [])[:3]
    except Exception as e:
        logger.debug(f"capability_inject: reflect failed: {e}")
    _CAP_CACHE["ts"] = now
    _CAP_CACHE["summary"] = summary
    _CAP_CACHE["reflections"] = reflections
    return summary, reflections


def _build_capability_snippet() -> str:
    """Erzeugt den Block der dem System-Prompt angehaengt wird."""
    sum_de, refl_lines = _get_capabilities_cached()
    if not sum_de and not refl_lines:
        return ""
    parts = []
    if sum_de:
        parts.append(f"\n\n**System-Selbstwahrnehmung:** {sum_de}")
    for line in refl_lines:
        parts.append(f"\n- {line}")
    return "".join(parts)


@app.get("/system_prompt")
def system_prompt_preview():
    """Debug: was wuerde dem Cloud-LLM als System-Prompt geschickt?"""
    try:
        from core.autonomy.local_llm_bridge import (
            _build_threebrain_state_snippet, _build_local_context_snippet, _load_profiles
        )
        profiles_data = _load_profiles() or {}
        profile = (profiles_data.get("profiles", {}) or {}).get("tentacle") or {}
        system = profile.get("system", "")
        if profile.get("include_live_context", True):
            system = system + _build_local_context_snippet()
        try:
            mctx = get_memory().get_memory_context_minimal()
            if mctx:
                system = system + "\n\n--- MEMORY ---\n" + mctx
        except Exception:
            pass
        tb = _build_threebrain_state_snippet()
        if tb:
            system = system + tb
        # W17 Self-Awareness: capability injection (nach Mood/Tension-Block)
        cap_block = _build_capability_snippet()
        if cap_block:
            system = system + cap_block
        return {"length": len(system), "system": system}
    except Exception as e:
        raise HTTPException(500, f"Prompt build error: {e}")


@app.get("/session_status")
def session_status():
    """PC-Heartbeat-Check: ist PC gerade online?"""
    global _pc_online_ts
    online = (time.time() - _pc_online_ts) < 90  # 90s Timeout
    return {
        "pc_online": online,
        "last_seen_s": round(time.time() - _pc_online_ts, 1) if _pc_online_ts else None,
        "pi_service": "running",
    }


@app.post("/pc_online")
def pc_online():
    """PC sendet Heartbeat alle 60s. Pi merkt sich Zeitstempel."""
    global _pc_online_ts
    _pc_online_ts = time.time()
    logger.info("[BRIDGE] PC-Heartbeat empfangen")
    return {"status": "ok", "ts": _pc_online_ts}


MOLOCH_REPO_DIR = Path(os.path.expanduser("~/moloch"))
MAILBOX_DIR = MOLOCH_REPO_DIR / "docs"
MAILBOX_FILES = {"PC_TO_PI": "PC_TO_PI.md", "PI_TO_PC": "PI_TO_PC.md"}
MAILBOX_SENDERS = {"PC_TO_PI": "PC", "PI_TO_PC": "Pi"}
MAILBOX_BRANCH = os.environ.get("MOLOCH_MAILBOX_BRANCH", "deepseek_architecture_overhaul")
TOPIC_RE = re.compile(r"^[a-zA-Z0-9_\-\[\]+ ]{1,100}$")
STATUS_RE = re.compile(r"^(open|done|info|answered|wontfix)$")


class MailboxPostRequest(BaseModel):
    sender: Optional[str] = None
    topic: str = Field(..., min_length=1, max_length=100)
    status: str = Field("open")
    body: str = Field(..., min_length=1, max_length=20000)
    reply_to: Optional[str] = Field(None, max_length=120)
    auto_push: bool = True


def _git_run(args, timeout: int = 30) -> Tuple[int, str, str]:
    r = subprocess.run(
        ["git"] + args,
        cwd=str(MOLOCH_REPO_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return r.returncode, r.stdout, r.stderr


def _mailbox_commit_push(commit_msg: str) -> None:
    """Background-Task: git add + commit + push (Pi-Account funktioniert)."""
    try:
        _git_run(["add", "docs/PC_TO_PI.md", "docs/PI_TO_PC.md"], timeout=10)
        rc, _, _ = _git_run(["diff", "--cached", "--quiet"], timeout=5)
        if rc == 0:
            logger.info("[mailbox-api] no changes to commit")
            return
        rc, _, err = _git_run(["commit", "-m", commit_msg], timeout=10)
        if rc != 0:
            logger.warning("[mailbox-api] commit failed: %s", err.strip()[:300])
            return
        rc, _, err = _git_run(["push", "origin", MAILBOX_BRANCH], timeout=30)
        if rc != 0:
            logger.warning("[mailbox-api] push failed: %s", err.strip()[:300])
            return
        logger.info("[mailbox-api] committed + pushed: %s", commit_msg)
    except Exception as e:
        logger.warning("[mailbox-api] git task error: %s", e)


def _append_topic(path: Path, sender: str, topic: str, status: str,
                  body: str, reply_to: Optional[str]) -> int:
    """Append-top neuen Topic in der Mailbox-Datei. Atomic write."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    reply_line = f"reply-to: [{reply_to}]\n" if reply_to else ""
    new_entry = (
        f"---\n"
        f"## [{ts}] from={sender} topic={topic}\n"
        f"status: {status}\n"
        f"{reply_line}"
        f"\n"
        f"{body.strip()}\n"
        f"\n"
    )
    if path.exists():
        content = path.read_text(encoding="utf-8")
    else:
        receiver = "Pi" if sender == "PC" else "PC"
        content = (
            f"# {sender} -> {receiver} mailbox\n\n"
            f"Append-only. Newest entry on top. Format and lifecycle: "
            f"see `docs/CROSS_SESSION_PROTOCOL.md`.\n\n"
        )
    lines = content.split("\n")
    insert_idx = next(
        (i for i, ln in enumerate(lines) if i > 2 and ln.strip() == "---"),
        None,
    )
    if insert_idx is None:
        new_content = content.rstrip() + "\n\n" + new_entry
    else:
        new_content = (
            "\n".join(lines[:insert_idx]) + "\n"
            + new_entry
            + "\n".join(lines[insert_idx:])
        )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(new_content, encoding="utf-8")
    tmp.replace(path)
    return len(new_content)


@app.get("/mailbox/{name}")
def mailbox_get(name: str):
    """Raw .md Content der Mailbox.

    PC pollt `GET http://192.168.178.30:9100/mailbox/PI_TO_PC` statt git pull,
    falls GitHub-Push-Probleme den remote-Stand nicht aktuell halten.
    """
    if name not in MAILBOX_FILES:
        raise HTTPException(404, f"unknown mailbox '{name}' (use PC_TO_PI or PI_TO_PC)")
    path = MAILBOX_DIR / MAILBOX_FILES[name]
    if not path.exists():
        return Response(content="", media_type="text/markdown; charset=utf-8")
    return Response(
        content=path.read_text(encoding="utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/mailbox/{name}")
def mailbox_post(name: str, req: MailboxPostRequest, background_tasks: BackgroundTasks):
    """Append-top neuer Topic in docs/{name}.md + auto-commit+push (Pi-Side).

    Workaround fuer kaputten PC-GitHub-Push:
    PC ruft `POST http://192.168.178.30:9100/mailbox/PC_TO_PI` mit JSON-Body
    statt selbst zu git-pushen. Pi schreibt + committet + pushed (Pi-Token
    funktioniert).
    """
    if name not in MAILBOX_FILES:
        raise HTTPException(404, f"unknown mailbox '{name}' (use PC_TO_PI or PI_TO_PC)")

    expected_sender = MAILBOX_SENDERS[name]
    sender = req.sender or expected_sender
    if sender != expected_sender:
        raise HTTPException(
            400,
            f"mailbox {name} expects sender={expected_sender}, got {sender}",
        )
    if not TOPIC_RE.match(req.topic):
        raise HTTPException(400, "topic: letters/digits/_/-/[]/+/space, max 100 chars")
    if not STATUS_RE.match(req.status):
        raise HTTPException(400, "status must be open|done|info|answered|wontfix")

    path = MAILBOX_DIR / MAILBOX_FILES[name]
    bytes_written = _append_topic(path, sender, req.topic, req.status, req.body, req.reply_to)

    if req.auto_push:
        receiver = "Pi" if sender == "PC" else "PC"
        commit_msg = f"mailbox-api: {sender}->{receiver} {req.topic} via HTTP"
        background_tasks.add_task(_mailbox_commit_push, commit_msg)

    return {
        "ok": True,
        "mailbox": name,
        "topic": req.topic,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "auto_push": req.auto_push,
        "bytes_written": bytes_written,
    }


_AUDIT_VALID_COMPONENTS = {
    # W8 (existing)
    "pc_health", "hygiene", "persona",
    # W12 PC-Side (PC-Cowork pushed pc_hardware + web_ui — siehe task 12:52)
    "pc_hardware", "web_ui",
    # W12 Pi-Side Sub-Auditoren (folgen direkt, ggf. lokal-merged statt POST)
    "vision", "npu", "spotify", "hardware",
    # W13 Domains
    "personality", "memory", "tracking",
    # W14 Domains
    "voice", "bridge", "tentacle", "awareness", "unconscious",
}

_LAST_TURN_PATH = "/dev/shm/last_turn.json"


def _write_last_turn_json(user_text: str, response_text: str,
                          prompt_type: Optional[str],
                          provider: Optional[str],
                          duration_ms: int) -> None:
    """Welle 10: Schreibe /dev/shm/last_turn.json atomic nach jedem /chat-Turn.

    Wird vom PC-persona_validator gepollt (GET /audit/last_turn) damit er
    Antworten gegen 5 Coherence-Signale (ich_form, slang_density, memory_ref,
    anti_hallu, tension_match) scoren + POST /mailbox/audit/persona schicken
    kann.
    """
    import uuid as _uuid
    pi_context: Dict = {}
    try:
        with open("/dev/shm/moloch_status.json", "r", encoding="utf-8") as f:
            st = json.load(f)
        core = st.get("core") or {}
        pi_context = {
            "tension": core.get("tension", st.get("tension")),
            "dominance": core.get("dominance", st.get("dominance")),
            "zone": core.get("zone", "guardian"),
            "mood_label": _get_pi_mood_label(),
            "person_detected": st.get("person_detected"),
            "face_id": st.get("face_id"),
            "recent_memories": [],
        }
    except Exception:
        pi_context = {"mood_label": _get_pi_mood_label()}
    # recent_memories (Top-3) aus longterm_memory
    try:
        from core.longterm_memory import get_memory  # type: ignore
        msgs = get_memory().get_recent_messages(n=3) or []
        pi_context["recent_memories"] = [
            {"role": m.get("role") or m.get("from") or "?",
             "text": (m.get("text") or m.get("content") or "")[:200]}
            for m in msgs if isinstance(m, dict)
        ]
    except Exception:
        pi_context.setdefault("recent_memories", [])
    # last_n_journal_types
    journal_types: list = []
    try:
        from core.memory.character_journal import get_journal  # type: ignore
        j = get_journal()
        getter = getattr(j, "get_recent_events", None) or getattr(j, "recent_events", None)
        if callable(getter):
            try:
                events = getter(limit=5) or []
            except TypeError:
                events = getter(5) or []
            journal_types = [
                ev.get("type") for ev in events
                if isinstance(ev, dict) and ev.get("type")
            ]
    except Exception:
        pass

    payload = {
        "turn_id": _uuid.uuid4().hex[:16],
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "user_text": user_text,
        "response_text": response_text,
        "prompt_type": prompt_type,
        "provider": provider,
        "duration_ms": int(duration_ms or 0),
        "pi_context": pi_context,
        "last_n_journal_types": journal_types,
    }
    try:
        tmp = _LAST_TURN_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, _LAST_TURN_PATH)
    except Exception as e:
        logger.warning(f"[last_turn] write fail: {e}")


_AUDIT_TTS_LOCK = Path(os.path.expanduser("~/moloch_logs/audit_tts_alarm_lock"))
_AUDIT_TTS_COOLDOWN_S = 1800  # 30 min


def _maybe_tts_alarm(state: Dict) -> None:
    """Welle 11: Bei alarm_tier=alert TTS-Alarm sprechen, mit 30min-Cooldown."""
    tier = (state or {}).get("alarm_tier")
    if tier != "alert":
        return
    try:
        _AUDIT_TTS_LOCK.parent.mkdir(parents=True, exist_ok=True)
        if _AUDIT_TTS_LOCK.exists():
            age = time.time() - _AUDIT_TTS_LOCK.stat().st_mtime
            if age < _AUDIT_TTS_COOLDOWN_S:
                return
        ts = state.get("updated_at", "")
        msg = f"MOLOCH ist driftend. Audit fehlgeschlagen seit {ts}."
        # Lokale Pi-Sprache via personality_engine -> tts_pipeline
        try:
            from core.personality.personality_engine import get_personality_engine
            pe = get_personality_engine()
            if hasattr(pe, "speak"):
                pe.speak(msg)
        except Exception as e:
            logger.warning(f"[audit-tts] speak fail: {e}")
        _AUDIT_TTS_LOCK.touch()
        logger.warning(f"[audit-tts] ALERT-TTS getriggert: {msg}")
    except Exception as e:
        logger.warning(f"[audit-tts] cooldown-check fail: {e}")


@app.get("/audit/stream")
async def audit_stream():
    """Welle 11: SSE-Stream auf /dev/shm/audit_state.json mtime-Change.

    Frontend EventSource verbindet hier. Bei jedem mtime-Wechsel: push
    aktuelles JSON als event-stream. Plus initialer Push beim Connect +
    Heartbeat alle 25s damit Proxies nicht killen.
    """
    audit_path = Path("/dev/shm/audit_state.json")

    async def gen():
        last_mtime = 0.0
        last_alarm_check = 0.0
        # Initial-Push
        try:
            if audit_path.exists():
                content = audit_path.read_text(encoding="utf-8")
                yield f"data: {content}\n\n"
                last_mtime = audit_path.stat().st_mtime
        except Exception:
            pass
        while True:
            try:
                if audit_path.exists():
                    mt = audit_path.stat().st_mtime
                    if mt > last_mtime:
                        last_mtime = mt
                        content = audit_path.read_text(encoding="utf-8")
                        yield f"data: {content}\n\n"
                        # TTS-Alarm-Check max alle 60s
                        if time.time() - last_alarm_check > 60:
                            last_alarm_check = time.time()
                            try:
                                state = json.loads(content)
                                _maybe_tts_alarm(state)
                            except Exception:
                                pass
                    else:
                        # Heartbeat
                        yield ": heartbeat\n\n"
                else:
                    yield ": no-audit-state\n\n"
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[audit-stream] tick fehler: {e}")
            await asyncio.sleep(2.0)

    return StreamingResponse(gen(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no"})


@app.get("/audit/last_turn")
def audit_last_turn():
    """Welle 10: letzter /chat-Turn als JSON. PC-Persona-Validator pollt das alle 10s."""
    try:
        with open(_LAST_TURN_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return Response(
            content=json.dumps(payload, ensure_ascii=False),
            media_type="application/json",
            headers={"Cache-Control": "max-age=5"},
        )
    except FileNotFoundError:
        raise HTTPException(404, "last_turn.json existiert nicht — kein /chat-Turn seit Service-Start")
    except Exception as e:
        raise HTTPException(500, f"last_turn-Fehler: {e}")


@app.get("/mailbox/audit/state")
def mailbox_audit_state():
    """Welle 8: aktueller audit_state.json Inhalt."""
    try:
        with open("/dev/shm/audit_state.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(404, "audit_state.json existiert nicht — orchestrator noch nie gelaufen")
    except Exception as e:
        raise HTTPException(500, f"audit_state Read-Fehler: {e}")


@app.post("/mailbox/audit/{component}")
def mailbox_audit_post(component: str, payload: Dict):
    """Welle 8: Receiver fuer PC- und Persona-Layer-Updates.

    component in {pc_health, hygiene, persona}. Body wird in
    audit_state.layers[component] gemerged + overall/alarm_tier neu berechnet.
    """
    if component not in _AUDIT_VALID_COMPONENTS:
        raise HTTPException(400, f"unknown audit component '{component}' "
                                  f"(valid: {sorted(_AUDIT_VALID_COMPONENTS)})")
    if not isinstance(payload, dict):
        raise HTTPException(400, "payload muss JSON-Objekt sein")
    try:
        from core.audit.audit_orchestrator import merge_component
        state = merge_component(component, payload)
        if state is None:
            raise HTTPException(500, "merge_component returnte None")
        return {
            "ok": True,
            "component": component,
            "overall": state.get("overall"),
            "alarm_tier": state.get("alarm_tier"),
            "updated_at": state.get("updated_at"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"audit-merge Fehler: {e}")


# ============================================================================
# W21 Phase 2 — Pi-Side Tool-API fuer PC-Orchestrator (DeepSeek function-calling)
# ============================================================================


class AgentDispatchRequest(BaseModel):
    tool_name: str
    params: Optional[Dict[str, Any]] = None


@app.get("/api/agent/tools")
def agent_tools_catalog():
    """W21 Phase 2 — Tool-Catalog fuer PC-Side Orchestrator (DeepSeek function-calling)."""
    try:
        from core.agent.tool_dispatcher import get_catalog, list_tools
        cat = get_catalog()
        tools_list = []
        for name in list_tools():
            tdef = cat.get(name) or {}
            if tdef:
                tools_list.append(tdef)
        return {"tools": tools_list, "count": len(tools_list)}
    except Exception as e:
        logger.warning(f"[W21] /api/agent/tools error: {e}")
        return {"tools": [], "count": 0, "error": str(e)[:200]}


@app.post("/api/agent/dispatch")
def agent_dispatch(req: AgentDispatchRequest):
    """W21 Phase 2 — PC-Orchestrator dispatchet Pi-Tool via HTTP.

    Body: {"tool_name": str, "params": dict}
    Returns: {tool, result, error, duration_ms}
    """
    try:
        from core.agent.tool_dispatcher import dispatch
        result = dispatch(req.tool_name, req.params or {})
        logger.info(
            f"[W21] dispatch tool={req.tool_name} "
            f"duration={result.get('duration_ms', 0):.1f}ms "
            f"err={result.get('error')}"
        )
        return result
    except Exception as e:
        logger.warning(f"[W21] /api/agent/dispatch error: {e}")
        return {
            "tool": req.tool_name,
            "result": None,
            "error": f"dispatch_endpoint_error:{str(e)[:200]}",
            "duration_ms": 0.0,
        }


# ============================================================================
# Closed-Loop Audit Trigger (W17 Phase 2 — Cockpit-getriggerte Verifier-Runs)
# ============================================================================

_VERIFY_VALID_TYPES = {
    "all", "ptz", "led", "fan", "tts", "spotify",
    "memory_recall", "bridge_roundtrip",
}


def _spawn_closed_loop_run(verify_type: str = "all") -> Dict:
    """Fire-and-forget Spawn des Closed-Loop-Orchestrators.

    Verify dauert bis zu 3 Min — kein Warten, kein capture. Ergebnis landet
    in /dev/shm/closed_loop_state.json (vom Orchestrator selbst geschrieben).
    """
    rid = str(uuid.uuid4())[:8]
    vt = verify_type if verify_type in _VERIFY_VALID_TYPES else "all"
    args = ["python3", "-m", "core.audit.closed_loop.closed_loop_orchestrator",
            f"--{vt}"]
    try:
        subprocess.Popen(
            args,
            cwd="/home/molochzuhause/moloch",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        raise HTTPException(500, f"Verify-Spawn fehlgeschlagen: {e}")
    return {
        "run_id": rid,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verify_type": vt,
    }


@app.post("/audit/verify", status_code=202)
def audit_verify_trigger(payload: Dict):
    """W17 Phase 2: Cockpit triggert Closed-Loop-Verifier.

    Body: {"verify": "all"|"ptz"|"led"|"fan"|"tts"|"spotify"|"memory_recall"|"bridge_roundtrip"}
    Returnt 202 Accepted — Run laeuft async im Hintergrund.
    """
    if not isinstance(payload, dict):
        raise HTTPException(400, "payload muss JSON-Objekt sein")
    verify = payload.get("verify", "all")
    if not isinstance(verify, str):
        raise HTTPException(400, "verify muss String sein")
    if verify not in _VERIFY_VALID_TYPES:
        raise HTTPException(
            400,
            f"unknown verify type '{verify}' "
            f"(valid: {sorted(_VERIFY_VALID_TYPES)})"
        )
    return _spawn_closed_loop_run(verify)


@app.get("/audit/verify_status")
def audit_verify_status():
    """W17 Phase 2: aktueller Closed-Loop-State (vom letzten Run)."""
    state_path = "/dev/shm/closed_loop_state.json"
    if not os.path.exists(state_path):
        return {"status": "never_run"}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(500, f"closed_loop_state Read-Fehler: {e}")


def main():
    if SSL_KEYFILE and SSL_CERTFILE and os.path.exists(SSL_KEYFILE) and os.path.exists(SSL_CERTFILE):
        logger.info(f"MOLOCH Chat-Server startet HTTPS auf {HOST}:{PORT}")
        uvicorn.run(
            app, host=HOST, port=PORT, log_level="info",
            ssl_keyfile=SSL_KEYFILE, ssl_certfile=SSL_CERTFILE,
        )
    else:
        logger.info(f"MOLOCH Chat-Server startet HTTP auf {HOST}:{PORT}")
        uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
