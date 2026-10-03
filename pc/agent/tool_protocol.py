"""Tool-Protokoll fuer lokale Modelle ohne verlaessliches Function-Calling.

Drei Stufen (PC_BRAIN_ARCHITEKTUR.md Abschnitt 5):
  1. Regel-Router: aus dem Text einen Intent + hoechstens 6 passende Tools waehlen.
     Smalltalk bekommt gar keine Tools (und laeuft frei, ohne JSON-Zwang).
  2. JSON-Protokoll mit erzwungenem Schema (Ollama-Parameter `format`):
       {"say": str, "tool": null | {"name": <eines der Tools>, "args": {...}}, "done": bool}
  3. Strenger Validator gegen den Pi-Katalog; bei Fehler EIN Reparaturversuch.

Der ganze Katalog (28 Tools) waere ~2000 Prompt-Token = ~50 s auf dieser CPU.
Mit der Vorauswahl bleiben es wenige hundert.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

MAX_TOOLS = 6

# Reihenfolge = Prioritaet. Erster Treffer gewinnt.
INTENTS: List[Tuple[str, str, List[str]]] = [
    ("web",
     r"\b(such\w*|google\w*|recherchier\w*|im netz|im internet|web|wetter|nachrichten|news|https?://\S+|www\.\S+)\b",
     ["web_search", "web_fetch"]),
    ("musikwissen",
     r"\b(top[- ]?\d*\s*(artists?|k(ue|ü)nstler|bands?|tracks?|songs?|lieder)|lieblings(band|bands|k(ue|ü)nstler|song|songs|lied|lieder)|"
     r"was l(ae|ä)uft|welcher (song|track)|welches lied|meistgeh(oe|ö)rt\w*)\b",
     ["spotify_top_artists", "spotify_top_tracks", "spotify_now_playing", "spotify_search"]),
    ("musik",
     r"\b(spiel\w*|musik|song|lied|track|playlist|spotify|lauter|leiser|lautst(ae|ä)rke|pause|pausier\w*|"
     r"n(ae|ä)chste[rs]?|skip|weiter|stopp die musik|musik aus)\b",
     ["spotify_play", "spotify_play_genre", "spotify_pause", "spotify_next", "spotify_volume",
      "spotify_now_playing"]),
    ("blick",
     r"\b(dreh\w*|schwenk\w*|schau\w* (nach|hoch|runter|rauf|links|rechts)|guck\w* (nach|hoch|runter|links|rechts)|"
     r"kamera|nach (links|rechts|oben|unten)|led|licht|leuchte\w*|farbe)\b",
     ["ptz_pan", "ptz_tilt", "led_set"]),
    ("sehen",
     r"\b(siehst du|wen siehst|wer ist (da|das)|erkennst du|wer steht|ist jemand|was siehst|im bild|vor dir)\b",
     ["get_face_id", "moloch_status_summary", "camera_snapshot"]),
    ("gedaechtnis",
     r"\b(erinner\w*|wei(ss|ß)t du noch|ged(ae|ä)chtnis|hab\w* ich .*gesagt|letztes mal|gestern|neulich|fr(ue|ü)her)\b",
     ["read_memory"]),
    ("status",
     r"\b(status|wie geht|stimmung|laune|zone|tension|temperatur|hei(ss|ß)|l(ue|ü)fter|npu|fps|audit|"
     r"system|ram|cpu|l(ae|ä)uft alles|alles ok)\b",
     ["get_mood", "moloch_status_summary", "get_npu_status", "get_audit_state",
      "thermal_set_tension_pwm"]),
]
_COMPILED = [(name, re.compile(rx, re.IGNORECASE), tools) for name, rx, tools in INTENTS]

# Wertelisten, die der Pi-Katalog nur im Beschreibungstext nennt
EXTRA_ENUMS: Dict[str, Dict[str, List[str]]] = {
    "led_set": {"color": ["red", "green", "blue", "yellow", "magenta", "cyan", "white", "off"]},
}

PROTOCOL_RULES = (
    "Antworte NUR mit einem JSON-Objekt in dieser Form:\n"
    '{"say": "<dein Satz an Markus>", "tool": null, "done": true}\n'
    'oder, wenn du erst ein Werkzeug brauchst:\n'
    '{"say": "", "tool": {"name": "<werkzeug>", "args": {<argumente>}}, "done": false}\n'
    "Regeln:\n"
    "- Nutze ein Werkzeug nur, wenn die Frage es wirklich braucht. Sonst tool = null.\n"
    "- Erfinde keine Messwerte, Namen oder Ergebnisse. Was du nicht weisst, holst du per Werkzeug.\n"
    "- Nach einem TOOL-ERGEBNIS: antworte mit say in deinem Ton, tool = null, done = true.\n"
    "- say ist kurz: ein bis zwei Saetze, deutsch."
)


@dataclass
class Parsed:
    say: str = ""
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    done: bool = True
    error: Optional[str] = None      # gesetzt = ungueltig, Reparatur noetig


def route(text: str, available: Optional[List[str]] = None) -> Tuple[str, List[str]]:
    """Text -> (intent, Tool-Namen). Unbekannt = smalltalk ohne Tools."""
    for name, rx, tools in _COMPILED:
        if rx.search(text or ""):
            chosen = [t for t in tools if available is None or t in available][:MAX_TOOLS]
            if chosen:
                return name, chosen
    return "smalltalk", []


# --- Regel-Aufloesung: Tool + Argumente direkt aus dem Text -------------------
# Modelltest 2026-10-03 (dolphin-llama3:8b, 40 Faelle): nur 62 % richtige Tool-Wahl,
# 15x erfundene Antwort statt Werkzeug, Drehrichtung vertauscht. Deshalb entscheidet
# bei erkanntem Intent der Code, was geholt/getan wird; das Modell formuliert nur.

PAN_LINKS_SIGN = 1.0      # NEVER 2: positiver Pan-Wert = physisch LINKS. Nicht aendern.
TILT_HOCH_SIGN = 1.0      # Annahme positiv = hoch — vor Etappe 5 live pruefen.
DEFAULT_STEP_DEG = 15

LED_FARBEN = {
    "rot": "red", "gruen": "green", "grün": "green", "blau": "blue", "gelb": "yellow",
    "magenta": "magenta", "pink": "magenta", "lila": "magenta", "cyan": "cyan",
    "tuerkis": "cyan", "türkis": "cyan", "weiss": "white", "weiß": "white",
}
_WEB_PREFIX = re.compile(
    r"^\s*(such\w*|recherchier\w*|google\w*)\s+(bitte\s+)?(mal\s+)?"
    r"((im|in dem)\s+(netz|internet|web)\s+)?(nach\s+)?", re.IGNORECASE)
_PLAY_PREFIX = re.compile(r"^(bitte\s+|mal\s+|mir\s+|was\s+von\s+|etwas\s+von\s+|den\s+song\s+|das\s+lied\s+)+",
                          re.IGNORECASE)
_GENRE_PREFIX = re.compile(r"^(was|etwas|irgendwas)\s+(aus|vom|von)\s+(dem\s+genre\s+)?", re.IGNORECASE)


def resolve(text: str, intent: str, available: List[str]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Intent + Text -> (tool, args) per Regel. None = Regel reicht nicht, Modell fragen."""
    t = (text or "").lower()
    zahl = re.search(r"\b(\d{1,3})\b", t)
    n = int(zahl.group(1)) if zahl else None

    def pick(tool: str, args: Optional[Dict[str, Any]] = None) -> Optional[Tuple[str, Dict[str, Any]]]:
        return (tool, args or {}) if tool in available else None

    if intent == "status":
        if re.search(r"stimmung|laune|zone|tension", t):
            return pick("get_mood")
        if "npu" in t:
            return pick("get_npu_status")
        if "audit" in t:
            return pick("get_audit_state")
        if re.search(r"l(ue|ü)fter", t) and n is not None:
            return pick("thermal_set_tension_pwm", {"percent": n})
        return pick("moloch_status_summary")

    if intent == "sehen":
        return pick("get_face_id")

    if intent == "gedaechtnis":
        return pick("read_memory", {"query": text.strip(" ?!."), "limit": 3})

    if intent == "musikwissen":
        if re.search(r"was l(ae|ä)uft|welcher (song|track)|welches lied", t):
            return pick("spotify_now_playing")
        if re.search(r"track|song|lied", t):
            return pick("spotify_top_tracks", {"n": n or 5})
        return pick("spotify_top_artists", {"n": n or 5})

    if intent == "musik":
        if re.search(r"\bpaus\w*|musik aus|stopp die musik", t):
            return pick("spotify_pause")
        if re.search(r"n(ae|ä)chste|skip|\bweiter\b", t):
            return pick("spotify_next")
        if re.search(r"lautst(ae|ä)rke|lauter|leiser", t) and n is not None:
            return pick("spotify_volume", {"percent": n})
        m = re.search(r"\bspiel\w*\s+(.+)", text or "", re.IGNORECASE)
        if m:
            rest = m.group(1).strip(" ?!.")
            if _GENRE_PREFIX.match(rest):
                return pick("spotify_play_genre", {"genre": _GENRE_PREFIX.sub("", rest).strip()})
            rest = _PLAY_PREFIX.sub("", rest).strip()
            if rest:
                return pick("spotify_play", {"query_or_uri": rest})
        return None            # z.B. "leiser" ohne Zahl: aktuelle Lautstaerke unbekannt

    if intent == "blick":
        if re.search(r"\bled\b|licht|leucht|farbe", t):
            if re.search(r"\baus\b|ausschalt|ausmach", t):
                return pick("led_set", {"color": "off"})
            for wort, farbe in LED_FARBEN.items():
                if wort in t:
                    return pick("led_set", {"color": farbe})
            return None
        grad = float(n) if n is not None else float(DEFAULT_STEP_DEG)
        if "links" in t:
            return pick("ptz_pan", {"angle": PAN_LINKS_SIGN * grad})
        if "rechts" in t:
            return pick("ptz_pan", {"angle": -PAN_LINKS_SIGN * grad})
        if re.search(r"oben|hoch|rauf", t):
            return pick("ptz_tilt", {"angle": TILT_HOCH_SIGN * grad})
        if re.search(r"unten|runter", t):
            return pick("ptz_tilt", {"angle": -TILT_HOCH_SIGN * grad})
        return None

    if intent == "web":
        url = re.search(r"(https?://\S+|www\.\S+)", text or "")
        if url:
            return pick("web_fetch", {"url": url.group(1).rstrip("?.,!)"), "max_chars": 1500})
        query = _WEB_PREFIX.sub("", text or "").strip(" ?!.")
        return pick("web_search", {"query": query or text, "max_results": 3})

    return None


def _fn(tool_def: Dict[str, Any]) -> Dict[str, Any]:
    return tool_def.get("function", tool_def)


def catalog_index(catalog: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {_fn(t).get("name"): _fn(t) for t in catalog if _fn(t).get("name")}


def schema(tool_names: List[str]) -> Dict[str, Any]:
    """JSON-Schema fuer Ollama `format` — der Tool-Name ist auf die Vorauswahl beschraenkt."""
    return {
        "type": "object",
        "properties": {
            "say": {"type": "string"},
            "tool": {"anyOf": [
                {"type": "null"},
                {"type": "object",
                 "properties": {"name": {"type": "string", "enum": list(tool_names)},
                                "args": {"type": "object"}},
                 "required": ["name", "args"]},
            ]},
            "done": {"type": "boolean"},
        },
        "required": ["say", "tool", "done"],
    }


def prompt_block(tool_names: List[str], index: Dict[str, Dict[str, Any]]) -> str:
    """Kompakter Werkzeug-Block fuer den Prompt (technisch, ohne Charakteraussagen)."""
    lines = ["WERKZEUGE fuer diese Anfrage:"]
    for name in tool_names:
        fn = index.get(name, {})
        props = (fn.get("parameters") or {}).get("properties") or {}
        required = set((fn.get("parameters") or {}).get("required") or [])
        parts = []
        for arg, spec in props.items():
            typ = spec.get("type", "string")
            enum = spec.get("enum") or EXTRA_ENUMS.get(name, {}).get(arg)
            if enum:
                typ = "|".join(map(str, enum))
            parts.append(f"{arg}{'' if arg in required else '?'}: {typ}")
        desc = (fn.get("description") or "").split(". ")[0][:110]
        lines.append(f"- {name}({', '.join(parts)}) — {desc}")
    return "\n".join(lines) + "\n\n" + PROTOCOL_RULES


def _coerce(value: Any, typ: str) -> Tuple[Any, bool]:
    """Wert auf den Katalog-Typ bringen. Returns (wert, ok)."""
    try:
        if typ == "integer":
            if isinstance(value, bool):
                return value, False
            f = float(value)
            return int(f), f == int(f)
        if typ == "number":
            if isinstance(value, bool):
                return value, False
            return float(value), True
        if typ == "boolean":
            if isinstance(value, bool):
                return value, True
            if str(value).lower() in ("true", "false"):
                return str(value).lower() == "true", True
            return value, False
        if typ == "string":
            return (value, True) if isinstance(value, str) else (str(value), not isinstance(value, (dict, list)))
    except (TypeError, ValueError):
        return value, False
    return value, True


def parse(raw: str, allowed: List[str], index: Dict[str, Dict[str, Any]]) -> Parsed:
    """Modell-Ausgabe pruefen. Bei Fehler steht in .error, was das Modell korrigieren muss."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        data = json.loads(text)
    except Exception:
        return Parsed(error="Kein gueltiges JSON. Antworte nur mit dem JSON-Objekt.")
    if not isinstance(data, dict):
        return Parsed(error="JSON muss ein Objekt sein.")

    say = data.get("say")
    say = say.strip() if isinstance(say, str) else ""
    tool = data.get("tool")
    if tool in (None, "", {}, "null"):
        if not say:
            return Parsed(error="say ist leer und tool ist null. Gib eine Antwort oder waehle ein Werkzeug.")
        return Parsed(say=say, done=True)

    if not isinstance(tool, dict):
        return Parsed(error='tool muss null sein oder {"name": ..., "args": {...}}.')
    name = tool.get("name")
    if name not in allowed:
        return Parsed(error=f"Werkzeug '{name}' gibt es hier nicht. Erlaubt: {', '.join(allowed)}.")
    args = tool.get("args")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return Parsed(error="args muss ein Objekt sein.")

    params = index.get(name, {}).get("parameters") or {}
    props = params.get("properties") or {}
    clean: Dict[str, Any] = {}
    for arg, value in args.items():
        if arg not in props:
            continue                     # erfundene Argumente verwerfen
        spec = props[arg]
        v, ok = _coerce(value, spec.get("type", "string"))
        if not ok:
            return Parsed(error=f"{name}.{arg} muss vom Typ {spec.get('type')} sein.")
        enum = spec.get("enum") or EXTRA_ENUMS.get(name, {}).get(arg)
        if enum and v not in enum:
            return Parsed(error=f"{name}.{arg} muss einer von {', '.join(map(str, enum))} sein.")
        clean[arg] = v
    missing = [r for r in (params.get("required") or []) if r not in clean or clean[r] in ("", None)]
    if missing:
        return Parsed(error=f"{name}: Pflicht-Argument fehlt: {', '.join(missing)}.")
    return Parsed(say=say, tool=name, args=clean, done=False)
