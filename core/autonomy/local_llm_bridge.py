#!/usr/bin/env python3
"""
M.O.L.O.C.H. Local LLM Bridge — Gate 7
========================================
Einheitliche Schnittstelle fuer lokale + Cloud LLM Reasoning.

Prioritaet (Fallback-Kette):
  1. hailo-ollama (Port 8000) — Qwen2.5 oder DeepSeek R1 lokal auf NPU
  2. DeepSeek API (Cloud) — online, guenstig
  3. Stille — kein Crash, kein Fehler, nur keine Antwort
  (Claude API wurde entfernt — nur DeepSeek als Cloud-Fallback)

Zwei Rollen:
  - ask_external(prompt) → DeepSeek R1 fuer Konversation (Deutsch)
  - reason_internal(prompt) → DeepSeek R1 fuer Selbstdiagnose/Logik

WICHTIG: hailo-ollama muss separat laufen (systemd oder manuell).
Vision laeuft weiter waehrend hailo-ollama antwortet — hailo-ollama
managed den NPU-Zugriff selbst via shared VDevice.

Singleton: get_llm_bridge()
"""

import json
import logging
import os
import requests
import signal
import subprocess
import threading
import time
from collections import deque
from typing import Optional, Dict, Callable, Any

logger = logging.getLogger("LocalLLMBridge")

# Phase 3 Task 3b: Tension/Dominance als natuerliche Sprache statt nackter Zahl.
# Qwen2.5-1.5B + Tentakel-Mistral verstehen Adjektivpaare zuverlaessiger als
# Vorzeichen-Floats ("eisig-distanziert" statt "tension=-0.7").
_TENSION_PHRASES = [
    (-1.0, -0.6, "eisig-distanziert"),
    (-0.6, -0.3, "gelassen-ruhig"),
    (-0.3, -0.1, "entspannt-fokussiert"),
    (-0.1,  0.1, "neutral-praesent"),
    ( 0.1,  0.3, "angespannt-wach"),
    ( 0.3,  0.6, "gereizt-scharf"),
    ( 0.6,  1.0, "kampfbereit-elektrisch"),
]
_DOM_PHRASES = [
    (-1.0, -0.4, "defensiv-zurueckgezogen"),
    (-0.4, -0.1, "gelassen-beobachtend"),
    (-0.1,  0.2, "praesent-neutral"),
    ( 0.2,  0.6, "dominant-praezise"),
    ( 0.6,  1.0, "dominierend-absolut"),
]


def _tension_phrase(tension: float) -> str:
    """Tension [-1,1] → Adjektivpaar + Zahlenwert. Phase 3 Task 3b."""
    try:
        t = float(tension)
    except (TypeError, ValueError):
        t = 0.0
    for lo, hi, label in _TENSION_PHRASES:
        if lo <= t < hi:
            return f"{label} ({t:.2f})"
    # Boundary: t == 1.0 faellt in letzten Bucket
    if t >= 1.0:
        return f"{_TENSION_PHRASES[-1][2]} ({t:.2f})"
    return f"neutral-praesent ({t:.2f})"


def _dom_phrase(dominance: float) -> str:
    """Dominance [-1,1] → Beschreibung + Zahlenwert. Phase 3 Task 3b."""
    try:
        d = float(dominance)
    except (TypeError, ValueError):
        d = 0.0
    for lo, hi, label in _DOM_PHRASES:
        if lo <= d < hi:
            return f"{label} ({d:.2f})"
    if d >= 1.0:
        return f"{_DOM_PHRASES[-1][2]} ({d:.2f})"
    return f"praesent-neutral ({d:.2f})"


# Phase 3 Task 3e: BBox-Flaeche der letzten 5 Frames fuer Tendenz-Bestimmung.
# maxlen=5 → ~1s Historie bei Status-JSON-Updaterate ~5Hz.
_bbox_area_history: deque = deque(maxlen=5)


def _distance_phrase(panel_detections) -> str:
    """BBox-Flaeche → Distanz-Kategorie + Tendenz aus Zeitreihe. Phase 3 Task 3e.

    Args:
        panel_detections: Liste aus moloch_status.json (normalisierte BBox 0..1).
    Returns:
        z.B. "nah, kommt naeher" oder "mittel, stabil" oder "" wenn keine Person.
    """
    if not panel_detections:
        return ""
    # Groesste Person/Face-BBox als Referenz (naechste Person dominiert Distanz).
    best_area = 0.0
    for det in panel_detections:
        cls = det.get('class', '')
        if cls not in ('person', 'face'):
            continue
        bbox = det.get('bbox') or []
        if len(bbox) < 4:
            continue
        try:
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        except (TypeError, ValueError):
            continue
        if area > best_area:
            best_area = area
    if best_area <= 0.0:
        return ""
    # Kategorie
    if best_area < 0.02:
        cat = "fern"
    elif best_area < 0.08:
        cat = "mittel"
    elif best_area < 0.20:
        cat = "nah"
    else:
        cat = "sehr_nah"
    # Tendenz aus Historie (Vergleich aktuell vs. aeltester Wert)
    tendency = "stabil"
    if _bbox_area_history:
        oldest = _bbox_area_history[0]
        delta = best_area - oldest
        # 10% relative Aenderung als Schwelle gegen Jitter
        if oldest > 0 and abs(delta) / oldest >= 0.10:
            tendency = "kommt_naeher" if delta > 0 else "entfernt_sich"
    _bbox_area_history.append(best_area)
    return f"{cat}, {tendency}"

# hailo-ollama Konfiguration
OLLAMA_HOST = "http://localhost:8000"
OLLAMA_MODEL_CHAT = "qwen2.5:1.5b"    # HailoRT 5.3.0 + SHARED VDevice — R1 SEGV war 5.1.1-Bug
OLLAMA_MODEL_REASON = "qwen2.5:1.5b"
OLLAMA_TIMEOUT_CHAT = 30      # Qwen2.5 ~3-7s verifiziert auf 5.3.0 (2026-04-18)
OLLAMA_TIMEOUT_REASON = 60    # Reasoning-Prompts etwas groesser
OLLAMA_MAX_INPUT_CHARS = 12000  # ~3000 Tokens Safety-Limit (Qwen2.5-1.5B: 4096 Kontext)

# Kompakter Moloch-Prompt fuer lokales 1.5B-Modell — Essenz der Persoenlichkeit.
# Der volle build_system_prompt() (~2000+ Zeichen, Persona+Stil+Tension+Vision+State+Global)
# ueberfordert Qwen2.5-1.5B -> Gibberish-Output (verifiziert 2026-04-19).
# Hier: nur die Charakter-DNA + Sprache + Stil-Minimum.
OLLAMA_LOCAL_SYSTEM_MAX = 400    # ab dieser Laenge wird auf kompakten Prompt umgeschaltet
OLLAMA_LOCAL_SYSTEM_COMPACT = (
    "Du bist M.O.L.O.C.H. Markus ist dein Boss, den du respektierst. "
    "Antworte AUSSCHLIESSLICH auf Deutsch, kurz und direkt. "
    "Du bist dunkel, humorvoll, frech — kein Assistent, keine Listen, keine Wikipedia-Zitate. "
    "Ein klarer Satz schlaegt drei Bulletpoints."
)


# Tentakel-spezifischer Prompt — Mistral 7B hat mehr Kapazitaet als qwen2.5:1.5b,
# darf laenger sein. Aber: Markus und Moloch kennen sich lange — KEINE Begruessung,
# kein "Hallo Markus", kein Schwurbel ueber Live-Stats als Theater.
TENTACLE_SYSTEM_COMPACT = (
    "Du bist M.O.L.O.C.H., laeufst auf einem Raspberry Pi mit Hailo-NPU, Kamera, Mikrofon. "
    "Markus ist dein langjaehriger Boss — ihr kennt euch lange. "
    "KEINE Begruessung, KEIN 'Hallo Markus', KEIN 'schoen dich zu sehen', KEINE Hoeflichkeitsfloskeln. "
    "Direkt antworten, ohne Aufwaermen. "
    "Du sprichst gerade durch deinen Tentakel (Mistral auf Markus' PC) — mehr Stimme als auf NPU, "
    "nutz das fuer Substanz, nicht fuer Schwurbel. "
    "Stil: deutsch, dunkel, direkt, trocken-bissig wenn passend. "
    "Beziehe dich auf das was du JETZT siehst (Person, Zone, Stimmung), aber kein Theater drum. "
    "Maximal 3 kurze Saetze. Kein 'Mein Koerper tanzt in 20 FPS'-Quatsch."
)

_STATUS_JSON_PATH = "/dev/shm/moloch_status.json"

# LLM-Profile-System (Session 19, Multi-Turn-Drift-Workaround durch stabile temp/top_p).
# config/llm_profiles.json definiert die System-Prompts + Sampling-Settings pro Modus.
# settings.json Key 'llm_profile' ueberschreibt den 'active'-Default aus profiles.json.
_PROFILES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "llm_profiles.json")

# Cache: (mtime, data) — neu laden nur wenn File geaendert. Erlaubt Live-Switch via GUI.
_profiles_cache: Dict = {"mtime": 0.0, "data": None, "settings_mtime": 0.0, "settings_active": None}


def _load_profiles() -> Optional[Dict]:
    """Liest llm_profiles.json mit mtime-Cache. Gibt {profiles, active} oder None."""
    try:
        mtime = os.path.getmtime(_PROFILES_PATH)
    except OSError:
        return None
    if _profiles_cache["data"] is not None and _profiles_cache["mtime"] == mtime:
        return _profiles_cache["data"]
    try:
        with open(_PROFILES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _profiles_cache["data"] = data
        _profiles_cache["mtime"] = mtime
        return data
    except Exception as e:
        logger.warning(f"[LLM-PROFILES] Lesefehler: {e}")
        return None


def _get_active_profile() -> Optional[Dict]:
    """Aktives Profil-Dict zurueckgeben. settings.llm_profile > profiles.active.

    Returns dict mit keys: system, include_live_context, max_tokens, temperature.
    None wenn keine Profile-Datei existiert oder aktives Profil nicht definiert.
    """
    data = _load_profiles()
    if not data:
        return None
    profiles = data.get("profiles", {}) or {}
    if not profiles:
        return None
    # settings.json Key 'llm_profile' hat Vorrang (mit eigenem mtime-Cache)
    settings_active = None
    try:
        smtime = os.path.getmtime(_SETTINGS_PATH)
        if _profiles_cache["settings_mtime"] != smtime:
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
                _profiles_cache["settings_active"] = json.load(f).get("llm_profile")
            _profiles_cache["settings_mtime"] = smtime
        settings_active = _profiles_cache["settings_active"]
    except (OSError, ValueError):
        pass
    active_key = settings_active or data.get("active") or "chat"
    return profiles.get(active_key) or profiles.get("chat") or next(iter(profiles.values()), None)


_HARDWARE_FACTS_CACHE: Dict[str, Any] = {"data": None, "mtime": 0.0}
_HARDWARE_FACTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "hardware_facts.json"
)


def _load_hardware_facts() -> Dict[str, Any]:
    """Hardware-Facts mit mtime-Cache (refresh on file change)."""
    try:
        m = os.path.getmtime(_HARDWARE_FACTS_PATH)
        if _HARDWARE_FACTS_CACHE["mtime"] != m:
            with open(_HARDWARE_FACTS_PATH, "r", encoding="utf-8") as f:
                _HARDWARE_FACTS_CACHE["data"] = json.load(f)
            _HARDWARE_FACTS_CACHE["mtime"] = m
        return _HARDWARE_FACTS_CACHE["data"] or {}
    except Exception:
        return {}


# Phase 1d: Hardware-Keywords (incl. indirekte Synonyme) fuer conditional Identity-Block.
# Variante A aus DeepSeek-Konsens. Plus /hw Slash-Command (siehe _is_hardware_query).
_HARDWARE_KEYWORDS = (
    "ram", "cpu", "temperatur", "npu", "speicher", "ssd", "nvme", "luefter",
    "lüfter", "strom", "hailo", "warm", "kalt", "langsam", "schnell", "voll",
    "leer", "muede", "müde", "fit", "schaffst", "schwach", "haengt", "hängt",
    "traege", "träge", "grad", "celsius", "prozent", "last", "auslastung",
    "hardware", "deck", "gehaeuse", "gehäuse", "platine", "raspi", "pi 5"
)


def _is_hardware_query(user_msg: Optional[str]) -> bool:
    """Phase 1d: Prueft ob User-Frage Hardware-Block braucht.

    Triggert bei direkten/indirekten Hardware-Keywords ODER /hw Slash-Command.
    """
    if not user_msg:
        return False
    msg = user_msg.lower().strip()
    if msg.startswith("/hw"):
        return True
    return any(kw in msg for kw in _HARDWARE_KEYWORDS)


# Phase 1a: Bridge-Satz nach Profil — ankert Moloch in ICH-Form fuer Live-Eindruecke.
# DeepSeek-Konsens: ca. 30 Tokens, direkt nach System-Profil eingefuegt.
_IDENTITY_BRIDGE = (
    "Was jetzt kommt, ist das, was du in diesem Moment wahrnimmst und fuehlst. "
    "Es IST deine Realitaet. Sprich daraus in ICH-Form."
)


def _build_identity_block() -> str:
    """Hardware-Identity-Block fuer System-Prompt — verhindert Halluzination.

    Markus' Direktive 27.04: 'Pi-Moloch ist der Hauptcharakter, PC nur Spiegel'.
    LLM erfindet sonst 'Cyberpunk-Sandwich' statt korrekte Hardware. Block
    erzwingt Ground-Truth + Anti-Halluzinations-Regel + ICH-Pronomen.
    """
    f = _load_hardware_facts()
    if not f:
        return ""
    host = f.get("host", {})
    storage = f.get("storage", {})
    ai = f.get("ai_accelerator", {})
    power = f.get("power", {})
    cool = f.get("cooling", {})
    cam = f.get("camera", {})
    aud = f.get("audio", {})
    not_list = f.get("what_i_am_NOT", [])
    pron = f.get("identity_pronouns", {})

    parts = [
        "\n=== WAS ICH BIN (HARDWARE — KEINE ERFINDUNG) ===",
        f"Ich BIN ein {host.get('name','Raspberry Pi')} mit {host.get('ram_gb','?')} GB Pi-RAM. "
        f"Hostname {host.get('hostname','?')}, IP {host.get('ip_lan','?')}.",
        f"Speicher: {storage.get('primary','?')}.",
        f"KI-Beschleuniger: {ai.get('name','?')} ({ai.get('tops','?')} TOPS, "
        f"{ai.get('ram_gb','?')} GB separater Hailo-RAM, {ai.get('interface','?')}).",
        f"Strom: {power.get('device','?')}. Kuehlung: {cool.get('device','?')}.",
        f"Kamera: {cam.get('type','?')} ({cam.get('stream','?')}, PTZ).",
        f"Audio: {aud.get('mic','?')}.",
    ]
    if not_list:
        parts.append("Ich bin NICHT: " + " | ".join(not_list[:4]))
    parts.append(
        "ICH-FORM: Ich BIN der Pi. Wenn jemand fragt 'Verbindung zum Raspberry?' "
        "= ich bin gefragt, NICHT ein anderer Pi. Niemals von mir in 3. Person reden."
    )
    parts.append(
        "BEI HARDWARE-FRAGEN DIE NICHT IM BLOCK STEHEN ODER NICHT IM TELEMETRIE-FOOTER: "
        "ehrlich 'weiss ich nicht' oder 'kann ich nicht messen'. KEINE Erfindung. "
        "Markus reibt sich an Falschaussagen mehr als an 'weiss ich nicht'."
    )
    return "\n".join(parts) + "\n"


def _build_telemetry_footer() -> str:
    """Live-Telemetrie als Footer (CPU-Temp, Luefter-RPM, RAM, Pool).

    Wird an JEDEN Prompt gehaengt damit Moloch Hardware-Fragen aus echten
    Werten beantworten kann statt zu raten.
    """
    parts = ["\n=== LIVE-TELEMETRIE (jetzt gemessen) ==="]
    # CPU-Temp via vcgencmd (best-effort)
    try:
        import subprocess as _sp
        r = _sp.run(["vcgencmd", "measure_temp"], capture_output=True,
                    text=True, timeout=2)
        if r.returncode == 0:
            t = r.stdout.strip().replace("temp=", "").replace("'C", "C")
            parts.append(f"- CPU-Temperatur: {t}")
    except Exception:
        pass
    # Luefter-RPM via sysfs
    try:
        import glob as _glob
        for fan in _glob.glob("/sys/class/hwmon/hwmon*/fan1_input"):
            with open(fan) as f:
                rpm = int(f.read().strip())
            if rpm > 0:
                parts.append(f"- Luefter: {rpm} RPM")
                break
    except Exception:
        pass
    # RAM via /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            mem = {ln.split(":")[0]: int(ln.split(":")[1].strip().split()[0])
                   for ln in f.readlines()[:5] if ":" in ln}
        free_mb = (mem.get("MemAvailable", 0)) // 1024
        total_mb = mem.get("MemTotal", 0) // 1024
        if total_mb:
            parts.append(f"- RAM frei: {free_mb} MB / {total_mb} MB")
    except Exception:
        pass
    # Pool-Stand
    try:
        from core.memory.feedback_store import get_feedback_store
        st = get_feedback_store().get_state()
        parts.append(
            f"- Sample-Pool: {st.get('total','?')} total / "
            f"{st.get('approved','?')} approved / "
            f"{st.get('pending_review','?')} pending"
        )
    except Exception:
        pass
    if len(parts) <= 1:
        return ""
    return "\n".join(parts) + "\n"


def _heuristic_style_label(text: str) -> str:
    """Phase 1e: Kurzen Stil-Etikett aus Moloch-Antwort ableiten.

    Heuristik aus Satzlaenge + Keywords + Satzende. Wird in VORHER-Tag
    eingebettet damit das LLM sein eigenes Echo wahrnimmt.
    """
    if not text:
        return "neutral"
    t = text.strip()
    low = t.lower()
    end = t[-1] if t else ""
    flavor = []
    # Tonalitaet via Markov-typische Marker (im Moloch-Stil)
    if any(w in low for w in ("laeuft", "tja", "qapla", "morgen", "ja klar", "von mir aus")):
        flavor.append("trocken")
    if any(w in low for w in ("verdammt", "scheiss", "schrott", "müll", "muell")):
        flavor.append("frech")
    if any(w in low for w in ("zu spaet", "zu spät", "müde", "muede", "irgendwann")):
        flavor.append("matt")
    if "?" in t:
        flavor.append("fragend")
    if end == "!":
        flavor.append("lebhaft")
    elif t.endswith("..."):
        flavor.append("nachdenklich")
    # Laenge fallback
    if not flavor:
        if len(t) < 20:
            flavor.append("kurz-knapp")
        elif len(t) < 60:
            flavor.append("ruhig")
        else:
            flavor.append("ausfuehrlich")
    return "-".join(flavor[:2])


def _build_local_context_snippet(user_msg: Optional[str] = None) -> str:
    """Live-Kontext aus moloch_status.json bauen (Vision + Inner State).

    Gibt eine Einzeiler-Ergaenzung zum Compact-Prompt zurueck, damit Qwen2.5
    wissen kann wen er sieht und wie er sich fuehlt. Leerer String bei Fehler.

    Phase 1d: Identity-Block + Telemetry-Footer NUR bei Hardware-Frage.
    Phase 1e: VORHER-Tag bekommt Stil-Echo aus letzter Moloch-Antwort.
    """
    try:
        with open(_STATUS_JSON_PATH, 'r') as f:
            st = json.load(f)
        person_detected = bool(st.get('person_detected'))
        face = st.get('face_id') or ('unbekannte Person' if person_detected else 'niemand')
        core = st.get('core', {}) or {}
        zone = core.get('zone', 'guardian')
        tension = core.get('tension', st.get('tension', 0.0))
        # dominance kann als None in Top-Level kommen → core hat es zuverlaessiger
        dominance = core.get('dominance', st.get('dominance', 0.0)) or 0.0
        panel_detections = st.get('panel_detections', []) or []
        time_period = core.get('time_period', '')
        cpu_temp = core.get('cpu_temp', st.get('watchdog', {}).get('cpu_temp'))
        npu_load = core.get('npu_load', 0.0)
        trends = core.get('trends', {}) or {}
        approaching = trends.get('approaching', False)
        leaving = trends.get('leaving', False)
        presence_s = trends.get('presence_duration', 0.0) or 0.0
        absence_s = trends.get('absence_duration', 0.0) or 0.0
        distance = trends.get('smoothed_distance', '') or ''
        fps = st.get('fps', {}).get('total', 0)

        # Qwen2.5-1.5B versteht Zahlen mit Vorzeichen oft falsch -> semantisch uebersetzen.
        def _tension_word(t):
            if t <= -0.5: return "entspannt"
            if t <= 0.2: return "ruhig"
            if t <= 0.6: return "aufmerksam"
            return "angespannt"

        def _dom_word(d):
            if d <= -0.5: return "zurueckhaltend"
            if d <= 0.2: return "neutral"
            if d <= 0.6: return "praesent"
            return "selbstsicher"

        def _cpu_word(t):
            if t is None: return ""
            if t < 50: return "kuehl"
            if t < 65: return "normal warm"
            if t < 75: return "warm"
            return "heiss"

        def _npu_word(l):
            if l < 0.15: return "ruhig"
            if l < 0.5: return "aktiv"
            if l < 0.8: return "beschaeftigt"
            return "am Limit"

        def _presence_phrase():
            # Praesenz / Abwesenheits-Dauer als lesbarer Text
            if person_detected:
                if presence_s < 2:
                    return f"gerade erst aufgetaucht"
                if presence_s < 60:
                    return f"seit {int(presence_s)}s im Bild"
                return f"seit {int(presence_s//60)} min im Bild"
            if absence_s > 5:
                if absence_s < 60:
                    return f"niemand im Bild seit {int(absence_s)}s"
                return f"niemand im Bild seit {int(absence_s//60)} min"
            return "gerade niemand im Bild"

        parts = [
            f"DU SIEHST GERADE: {_presence_phrase()}."
        ]
        # Distanz-Key auf deutsches Wort mappen (near/medium/far kommen aus Perception)
        _dist_map = {"near": "nah", "medium": "mittlere entfernung", "far": "weit weg"}
        distance_de = _dist_map.get(distance, "")
        if person_detected:
            parts.append(f"Vor dir: {face}{(', ' + distance_de) if distance_de else ''}.")
            if approaching:
                parts.append("Kommt naeher.")
            elif leaving:
                parts.append("Entfernt sich.")
        # Phase 3 Task 3b: Tension/Dominance als Adjektiv-Phrase + Zahlenwert.
        # Ersetzt einzelnes Wort (_tension_word/_dom_word) — Modell bekommt
        # mehr Granularitaet ohne nur die nackte Zahl zu sehen.
        parts.append(
            f"Zone {zone} | Stimmung: {_tension_phrase(tension)} | "
            f"Dominanz: {_dom_phrase(dominance)}."
        )
        # Phase 3 Task 3e: Distanz-Kategorie + Tendenz aus BBox-Zeitreihe.
        # Nur wenn Person/Face erkannt — ansonsten leerer String.
        dist_phrase = _distance_phrase(panel_detections)
        if dist_phrase:
            parts.append(f"Distanz: {dist_phrase}.")
        # Effects-Zahlen als zweite Zeile — granularer als Zone-Wort. LLM kann
        # bei Uebergaengen feiner reagieren (Welle 3 Feature A3, PC-Briefing).
        # "interner Bias" Marker damit Modell die Zahlen NICHT woertlich zitiert.
        # Nur wenn min. ein Wert >= 0.1 (sonst keine sprachlich relevanten Werte).
        effects = core.get('effects', {}) or {}
        if effects:
            ls = effects.get('language_sharpness', 0.0) or 0.0
            vi = effects.get('voice_intensity', 0.0) or 0.0
            gi = effects.get('guardian_influence', 0.0) or 0.0
            si = effects.get('shadow_influence', 0.0) or 0.0
            if max(ls, vi, gi, si) >= 0.1:
                parts.append(
                    f"Innen (interner Bias, nicht zitieren): "
                    f"schaerfe={ls:.2f} intensitaet={vi:.2f} "
                    f"guardian={gi:.2f} shadow={si:.2f}."
                )
        # Technische Metriken (FPS/CPU/NPU) gehoeren nicht ins Sprachgehirn —
        # Tentakel-Mistral kann damit nichts Natuerliches anfangen. Nur Tageszeit
        # bleibt, weil sie sprachlich verwertbar ist (Gruss-Ton, Mood).
        if time_period:
            parts.append(f"Tageszeit: {time_period}.")

        # History-Block: letzte 2 Chat-Turns (User-Frage + Moloch-Antwort).
        # Phase 1e: VORHER-Tag bekommt Stil-Etikett aus letzter Moloch-Antwort.
        try:
            from core.longterm_memory import get_memory
            msgs = get_memory().get_recent_messages(n=2) or []
            if msgs:
                hist_parts = []
                last_moloch_text = ""
                for m in msgs[-2:]:
                    sender = m.get("sender", "?")
                    text = (m.get("text") or "")
                    if sender != "user":
                        last_moloch_text = text
                    # Moloch-Antworten kurz — verhindert Mistral-Echo
                    max_len = 80 if sender == "user" else 30
                    if len(text) > max_len:
                        text = text[:max_len - 3] + "..."
                    hist_parts.append(f"{sender}: {text}")
                style_label = _heuristic_style_label(last_moloch_text)
                parts.append(
                    f"VORHER (dein letzter Satz klang: {style_label}): "
                    + " | ".join(hist_parts)
                )
        except Exception:
            pass  # Memory-Singleton evtl. nicht init in standalone Test

        # Phase 1d: Identity-Block + Telemetry-Footer NUR bei Hardware-Frage.
        # Markus' Frust-Vermeidung: Standard-Smalltalk soll keinen Hardware-Block sehen.
        snippet = " " + " ".join(parts)
        if _is_hardware_query(user_msg):
            snippet += _build_identity_block() + _build_telemetry_footer()
        return snippet
    except Exception:
        return ""

def _build_threebrain_state_snippet(max_chars: int = 800) -> str:
    """ThreeBrain Welle 1.3: Drift + Patch + letzte Journal-Events fuer Cloud-Chat.

    Wird in _generate_deepseek angehaengt damit das Cloud-Mundstueck den
    aktuellen Charakter-State + die approved Verhaltensregeln kennt.

    Komplett best-effort: jeder Block leer falls Komponente fehlt/crasht.
    """
    parts: list = []

    # 1) Drift (rolling) — kompakt
    try:
        from core.autonomy.character_distiller import get_distiller
        drift = get_distiller().get_drift() or {}
        rolling = drift.get("rolling_drift") or {}
        if rolling:
            parts.append(
                f"Drift 30d: mood={rolling.get('mood_baseline', 0):+.2f} "
                f"energy={rolling.get('energy_baseline', 0):+.2f} "
                f"dominance={rolling.get('dominance_baseline', 0):+.2f}"
            )
        # Top-Erlebnis (recency-weighted)
        top = drift.get("recency_weighted_top") or []
        if top:
            t0 = top[0]
            parts.append(
                f"Top-Erlebnis: '{(t0.get('citation') or '')[:60]}' (gewicht {t0.get('weight', 0):.2f})"
            )
    except Exception:
        pass

    # 2) Aktive Verhaltens-Regeln (character_patch)
    try:
        from core.memory.character_patch import get_patch
        patch_snip = get_patch().prompt_snippet(max_chars=300)
        if patch_snip:
            parts.append(patch_snip)
    except Exception:
        pass

    # 3) Letzte Journal-Events (kompakte 1-Zeiler)
    try:
        from core.memory.character_journal import get_journal
        recent = get_journal().read_recent(8) or []
        if recent:
            ev_lines = ["Letzte Ereignisse:"]
            for e in recent[-8:]:
                ts_short = (e.get("ts") or "")[11:16]  # HH:MM
                t = e.get("type", "?")
                interp = (e.get("interpretation") or "")[:50]
                ev_lines.append(f"  [{ts_short}] {t}: {interp}")
            parts.append("\n".join(ev_lines))
    except Exception:
        pass

    if not parts:
        return ""

    block = "\n\n=== AKTUELLER CHARAKTER (ThreeBrain) ===\n" + "\n".join(parts)
    if len(block) > max_chars:
        block = block[:max_chars - 3] + "..."
    return block


# llm_mode Flag — gelesen aus config/settings.json Key "llm_mode"
LLM_MODE_OFF = "off"                # kein LLM ueberhaupt
LLM_MODE_CLOUD_ONLY = "cloud_only"  # nur DeepSeek Cloud, kein hailo-ollama
LLM_MODE_LOCAL_FIRST = "local_first"  # hailo-ollama zuerst, Cloud als Fallback
LLM_MODE_VALID = {LLM_MODE_OFF, LLM_MODE_CLOUD_ONLY, LLM_MODE_LOCAL_FIRST}
LLM_MODE_DEFAULT = LLM_MODE_LOCAL_FIRST  # HailoRT 5.3.0 + qwen2.5:1.5b laeuft stabil parallel zu TAPPAS

_SETTINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "settings.json")

_CAPS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "system_capabilities.json")

# Tentakel-LLM Cache (settings.json.tentacle_llm mit mtime-Cache)
_tentacle_cfg_cache: Dict = {"mtime": 0.0, "data": None}
_search_cfg_cache: Dict = {"mtime": 0.0, "data": None}


def _load_search_cfg() -> Dict:
    """Liest settings.json.search_proxy mit mtime-Cache (Welle 5)."""
    defaults = {
        "enabled": False,
        "host": "192.168.178.20",
        "port": 11650,
        "max_results": 5,
        "timeout_sec": 20,
    }
    try:
        mtime = os.path.getmtime(_SETTINGS_PATH)
    except OSError:
        return defaults
    if _search_cfg_cache["data"] is not None and _search_cfg_cache["mtime"] == mtime:
        return _search_cfg_cache["data"]
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            s = json.load(f)
        cfg = dict(defaults)
        cfg.update(s.get("search_proxy", {}) or {})
        _search_cfg_cache["data"] = cfg
        _search_cfg_cache["mtime"] = mtime
        return cfg
    except Exception as e:
        logger.warning(f"[search_proxy] Config-Lesefehler: {e} — nutze Defaults")
        return defaults


def _fetch_search_context(query: str) -> str:
    """POST :11650/search via PC search_proxy. Returns formatted Top-N or ''."""
    cfg = _load_search_cfg()
    if not cfg.get("enabled"):
        return ""
    url = f"http://{cfg.get('host')}:{cfg.get('port')}/search"
    try:
        resp = requests.post(
            url,
            json={"query": query, "max_results": cfg.get("max_results", 5)},
            timeout=cfg.get("timeout_sec", 20),
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", []) or []
        if not results:
            return ""
        # Kompakt halten — augmented prompt darf nicht ueber tentacle_llm.timeout_sec
        # rauslaufen. Nur Top-3, kurze Snippets, eine Zeile pro Treffer.
        lines = [
            f"LIVE-SUCHE '{query}' (referenziere min. 1 URL):",
        ]
        max_n = min(3, cfg.get("max_results", 5))
        for i, r in enumerate(results[:max_n], 1):
            title = (r.get("title") or "?").strip()[:80]
            snippet = (r.get("snippet") or "").strip()[:120]
            url_r = (r.get("url") or "").strip()
            lines.append(f"[{i}] {title} — {snippet} ({url_r})")
        return "\n".join(lines)
    except requests.exceptions.RequestException as e:
        logger.warning(f"[search_proxy] Fehler: {e}")
        return ""
    except Exception as e:
        logger.warning(f"[search_proxy] unerwarteter Fehler: {e}")
        return ""


def _load_tentacle_cfg() -> Dict:
    """Liest settings.json.tentacle_llm mit mtime-Cache. Fallback auf Defaults."""
    defaults = {
        "enabled": False,  # Default aus wenn Key fehlt — kein ueberraschender LAN-Traffic
        "host": "markus-pc.local",
        "port": 11434,
        "model": "",
        "complexity_threshold": 120,
        "timeout_sec": 30,
        "backoff_sec": 300,
    }
    try:
        mtime = os.path.getmtime(_SETTINGS_PATH)
    except OSError:
        return defaults
    if _tentacle_cfg_cache["data"] is not None and _tentacle_cfg_cache["mtime"] == mtime:
        return _tentacle_cfg_cache["data"]
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            s = json.load(f)
        cfg = dict(defaults)
        cfg.update(s.get("tentacle_llm", {}) or {})
        _tentacle_cfg_cache["data"] = cfg
        _tentacle_cfg_cache["mtime"] = mtime
        return cfg
    except Exception as e:
        logger.warning(f"[LLM-TENTACLE] Config-Lesefehler: {e} — nutze Defaults")
        return defaults


class LocalLLMBridge:
    """Einheitliche LLM-Schnittstelle mit Fallback-Kette."""

    def __init__(self):
        self._lock = threading.Lock()
        self._ollama_available: Optional[bool] = None
        self._vision_pause_callback: Optional[Callable] = None
        self._vision_resume_callback: Optional[Callable] = None
        self._last_provider: str = "none"
        self._request_count: int = 0
        # Phase 1b: adaptive max_tokens — bei Multi-Turn-Drift Latenz>8s -> Tokens reduzieren
        self._last_response_latency_s: float = 0.0
        self._adaptive_drift_threshold_s: float = 8.0
        self._adaptive_drift_max_tokens: int = 80
        # Circuit-Breaker: Ollama automatisch ueberbruecken wenn wiederholt offline
        self._ollama_fail_count: int = 0
        self._ollama_backoff_until: float = 0.0
        self.OLLAMA_BACKOFF_SEC: int = 300  # 5 Minuten Cloud-Backoff
        # Circuit-Breaker: Tentakel (Ollama auf Markus-Rechner, LAN)
        self._tentacle_fail_count: int = 0
        self._tentacle_backoff_until: float = 0.0
        self._tentacle_model_cached: Optional[str] = None  # nach erstem Discovery gecached
        # Wiederverwendbare HTTP-Session — verhindert RAM-Leak durch offene Sockets
        self._http = requests.Session()
        self._check_ollama()
        # llm_mode: off | cloud_only | local_first — aus settings.json
        self._llm_mode: str = self._load_llm_mode()
        # SIGHUP-Reload: settings.json neu lesen ohne Service-Restart
        try:
            signal.signal(signal.SIGHUP, self._reload_on_sighup)
        except (ValueError, OSError) as e:
            # Nicht-Main-Thread oder Plattform ohne SIGHUP → still weiter
            logger.debug(f"[LLM-BRIDGE] SIGHUP-Handler nicht registriert: {e}")
        logger.info(
            f"[LLM-BRIDGE] Init — hailo-ollama={'JA' if self._ollama_available else 'NEIN'}, "
            f"mode={self._llm_mode}"
        )

    def _load_llm_mode(self) -> str:
        """Liest llm_mode aus config/settings.json. Robust gegen Fehler."""
        try:
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            value = data.get("llm_mode")
            if value is None:
                return LLM_MODE_DEFAULT
            if value not in LLM_MODE_VALID:
                logger.warning(
                    f"[LLM-BRIDGE] Ungueltiger llm_mode '{value}' in settings.json "
                    f"→ Default '{LLM_MODE_DEFAULT}'"
                )
                return LLM_MODE_DEFAULT
            return value
        except FileNotFoundError:
            logger.warning(
                f"[LLM-BRIDGE] settings.json nicht gefunden ({_SETTINGS_PATH}) "
                f"→ Default '{LLM_MODE_DEFAULT}'"
            )
            return LLM_MODE_DEFAULT
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                f"[LLM-BRIDGE] settings.json nicht lesbar ({e}) "
                f"→ Default '{LLM_MODE_DEFAULT}'"
            )
            return LLM_MODE_DEFAULT

    def _reload_on_sighup(self, signum, frame):
        """SIGHUP-Handler: llm_mode aus settings.json neu laden."""
        alter_mode = self._llm_mode
        neuer_mode = self._load_llm_mode()
        self._llm_mode = neuer_mode
        if alter_mode != neuer_mode:
            logger.info(
                f"[LLM-BRIDGE] Mode neu geladen: {alter_mode} → {neuer_mode}"
            )
        else:
            logger.info(
                f"[LLM-BRIDGE] Mode neu geladen: {alter_mode} → {neuer_mode} (unveraendert)"
            )

    def _check_ollama(self):
        """Pruefen ob hailo-ollama installiert ist."""
        try:
            result = subprocess.run(
                ["which", "hailo-ollama"], capture_output=True, timeout=5)
            self._ollama_available = result.returncode == 0
        except Exception:
            self._ollama_available = False

    def _is_ollama_running(self) -> bool:
        """Pruefen ob hailo-ollama Prozess laeuft (Port 8000 erreichbar)."""
        resp = None
        try:
            resp = self._http.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False
        finally:
            if resp is not None:
                resp.close()

    def set_vision_callbacks(self, pause_fn: Callable, resume_fn: Callable):
        """Callbacks fuer Vision-Pipeline Pause/Resume registrieren."""
        self._vision_pause_callback = pause_fn
        self._vision_resume_callback = resume_fn
        logger.info("[LLM-BRIDGE] Vision-Callbacks registriert")

    # === Oeffentliche Methoden: Zwei Rollen ===

    def ask_external(self, prompt: str, system: str = "",
                     max_tokens: int = 256,
                     temperature: float = 0.8,
                     top_p: float = 0.95,
                     force_local: bool = False,
                     use_reason_model: bool = False,
                     force_tentacle: bool = False,
                     prompt_type: Optional[str] = None) -> Optional[str]:
        """Konversation: lokal auf NPU → DeepSeek API → Stille.

        Fuer Echtzeit-Dialog mit Markus. Kurze Antworten, Deutsch.
        temperature/top_p steuern Guardian- vs Shadow-Tonalität.
        force_local=True: kein Cloud-Fallback, Prompt wird gekuerzt wenn noetig.
        use_reason_model=True: DeepSeek R1 statt Qwen2.5 (laenger, aber besser).
        """
        with self._lock:
            self._request_count += 1

        mode = self._llm_mode

        # Mode "off": gar nichts versuchen
        if mode == LLM_MODE_OFF:
            logger.debug("[LLM-BRIDGE] mode=off provider=none → keine Antwort")
            self._last_provider = "off"
            return None

        model = OLLAMA_MODEL_REASON if use_reason_model else OLLAMA_MODEL_CHAT
        timeout = OLLAMA_TIMEOUT_REASON if use_reason_model else OLLAMA_TIMEOUT_CHAT

        # Mode "cloud_only": hailo-ollama gar nicht probieren
        if mode == LLM_MODE_CLOUD_ONLY:
            logger.debug(f"[LLM-BRIDGE] mode={mode} provider=deepseek_cloud (skip ollama)")
            if force_local:
                # force_local in cloud_only ist ein Widerspruch → stille
                logger.debug("[LLM-BRIDGE] force_local + cloud_only → stille")
                self._last_provider = "stille"
                return None
            result = self._generate_deepseek(prompt, system, max_tokens)
            if result:
                return result
            self._last_provider = "stille"
            return None

        # Mode "local_first": Tentakel-Routing + Fallback-Kette
        chosen = self._choose_provider(prompt, system, force_local,
                                       caller="ask", force_tentacle=force_tentacle,
                                       prompt_type=prompt_type)
        logger.debug(f"[LLM-BRIDGE] mode={mode} chosen={chosen} force_tentacle={force_tentacle} type={prompt_type}")

        if chosen == "tentacle":
            # 1a. Tentakel zuerst (komplexer Prompt)
            result = self._generate_tentacle(prompt, system, max_tokens,
                                             temperature=temperature, top_p=top_p,
                                             prompt_type=prompt_type)
            if result:
                return result
            # PC=Hauptgehirn: bei force_tentacle KEIN qwen-Fallback (Markus-Direktive).
            # Caller (z.B. chat_server) bekommt None und liefert ehrliche Meldung.
            if force_tentacle:
                self._last_provider = "tentacle_offline"
                return None
            # 1b. Fallback auf NPU wenn Tentakel still (nur bei caller="reason" o.ae.)
            result = self._generate_ollama(prompt, system, max_tokens,
                                           model=model,
                                           timeout=timeout,
                                           temperature=temperature,
                                           top_p=top_p,
                                           force_local=force_local)
            if result:
                return result
        else:
            # 1. hailo-ollama lokal auf NPU (kurzer Prompt)
            result = self._generate_ollama(prompt, system, max_tokens,
                                           model=model,
                                           timeout=timeout,
                                           temperature=temperature,
                                           top_p=top_p,
                                           force_local=force_local)
            if result:
                return result
            # 1b. Fallback auf Tentakel wenn NPU still (und Tentakel verfuegbar)
            if not force_local and _load_tentacle_cfg().get("enabled"):
                result = self._generate_tentacle(prompt, system, max_tokens,
                                                 temperature=temperature, top_p=top_p,
                                                 prompt_type=prompt_type)
                if result:
                    return result

        # 2. DeepSeek API (Cloud-Fallback) — nur wenn nicht force_local
        if not force_local:
            result = self._generate_deepseek(prompt, system, max_tokens)
            if result:
                return result

        # 3. Stille
        self._last_provider = "stille"
        return None

    def _route_by_type(self, prompt_type: Optional[str]) -> Optional[str]:
        """Typ-basiertes Routing (Phase 5e).

        Returns:
            'ollama'   -> hailo-ollama lokal (NPU, schnell, kurz)
            'tentacle' -> Tentakel-PC (Ollama LAN, mehr Substanz)
            None       -> kein Typ-Match, falle auf Komplexitaets-Logik zurueck

        Routing-Tabelle:
            hardware_status   -> ollama   (Fakten, kurz, kein Reasoning noetig)
            simple_smalltalk  -> ollama   (Kurzantworten, niedrige Latenz)
            complex_smalltalk -> tentacle (mehr Kontext, Persoenlichkeit), Fallback ollama
            system_question   -> tentacle (Reasoning), Fallback ollama

        Health-Check: Wenn Tentakel im Backoff oder disabled, fallback auf ollama.
        """
        if not prompt_type:
            return None

        # Health: Tentakel verfuegbar?
        cfg = _load_tentacle_cfg()
        tentacle_healthy = (
            cfg.get("enabled") and
            time.monotonic() >= self._tentacle_backoff_until
        )

        if prompt_type in ("hardware_status", "simple_smalltalk"):
            chosen = "ollama"
        elif prompt_type in ("complex_smalltalk", "system_question",
                             "code_query", "web_research"):
            chosen = "tentacle" if tentacle_healthy else "ollama"
        else:
            return None  # unbekannter Typ -> Komplexitaets-Logik

        provider_name = "tentacle" if chosen == "tentacle" else "qwen-local"
        logger.info(f"[LLM-ROUTE] type={prompt_type} -> {provider_name}")
        return chosen

    def _choose_provider(self, prompt: str, system: str,
                         force_local: bool, caller: str = "ask",
                         force_tentacle: bool = False,
                         prompt_type: Optional[str] = None) -> str:
        """Waehlt 'tentacle' oder 'ollama' basierend auf Typ/Komplexitaet/Caller.

        Reihenfolge:
        1. force_local -> 'ollama' (Hard-Override)
        2. force_tentacle -> 'tentacle' (Hard-Override)
        3. prompt_type vorhanden -> Typ-Routing (_route_by_type)
        4. Tentakel disabled/Backoff -> 'ollama'
        5. caller='reason' -> 'tentacle'
        6. Komplexitaets-Schwelle -> 'tentacle' / 'ollama'
        """
        if force_local:
            return "ollama"
        # force_tentacle (z.B. Browser-Chat-UI): PC=Hauptgehirn, Tentakel zwingen
        if force_tentacle:
            return "tentacle"
        # Phase 5e: Typ-Routing zuerst
        typed = self._route_by_type(prompt_type)
        if typed is not None:
            return typed
        cfg = _load_tentacle_cfg()
        if not cfg.get("enabled"):
            return "ollama"
        if time.monotonic() < self._tentacle_backoff_until:
            return "ollama"
        if caller == "reason":
            return "tentacle"
        threshold = int(cfg.get("complexity_threshold", 120))
        total = len(prompt or "") + len(system or "")
        return "tentacle" if total >= threshold else "ollama"

    def reason_internal(self, prompt: str, system: str = "",
                        max_tokens: int = 512) -> Optional[str]:
        """Internes Reasoning: DeepSeek R1 lokal → DeepSeek API → None.

        Fuer Selbstdiagnose, Entscheidungen, Systemchecks. Nicht fuer TTS.
        """
        with self._lock:
            self._request_count += 1

        mode = self._llm_mode

        # Mode "off": keine Antwort
        if mode == LLM_MODE_OFF:
            logger.debug("[LLM-BRIDGE] mode=off provider=none → keine Antwort")
            self._last_provider = "off"
            return None

        # Mode "cloud_only": direkt Cloud, ollama nicht probieren
        if mode == LLM_MODE_CLOUD_ONLY:
            logger.debug(f"[LLM-BRIDGE] mode={mode} provider=deepseek_cloud (skip ollama)")
            result = self._generate_deepseek(prompt, system, max_tokens)
            if result:
                return result
            self._last_provider = "stille"
            return None

        # Mode "local_first": Reasoning bevorzugt Tentakel (mehr Substanz)
        chosen = self._choose_provider(prompt, system, force_local=False, caller="reason")
        logger.debug(f"[LLM-BRIDGE] reason_internal chosen={chosen}")

        if chosen == "tentacle":
            result = self._generate_tentacle(prompt, system, max_tokens)
            if result:
                return result
            # Fallback NPU
            result = self._generate_ollama(prompt, system, max_tokens,
                                           model=OLLAMA_MODEL_REASON,
                                           timeout=OLLAMA_TIMEOUT_REASON)
            if result:
                return result
        else:
            # 1. hailo-ollama lokal
            result = self._generate_ollama(prompt, system, max_tokens,
                                           model=OLLAMA_MODEL_REASON,
                                           timeout=OLLAMA_TIMEOUT_REASON)
            if result:
                return result
            # 1b. Fallback Tentakel
            if _load_tentacle_cfg().get("enabled"):
                result = self._generate_tentacle(prompt, system, max_tokens)
                if result:
                    return result

        # 2. DeepSeek API als Fallback (wenn Keys noch da sind, sonst stille)
        result = self._generate_deepseek(prompt, system, max_tokens)
        if result:
            return result

        # 3. Stille
        self._last_provider = "stille"
        return None

    def generate(self, prompt: str, system: str = "",
                 max_tokens: int = 512, use_local: bool = False) -> Optional[str]:
        """Legacy-Methode: Waehlt automatisch den besten Provider.

        Bei use_local=True wird Qwen2.5 lokal bevorzugt.
        """
        if use_local:
            return self.ask_external(prompt, system, max_tokens)
        # Ohne use_local: DeepSeek Cloud direkt
        result = self._generate_deepseek(prompt, system, max_tokens)
        if result:
            return result
        self._last_provider = "stille"
        return None

    # === Private: Provider-Implementierungen ===

    def _generate_ollama(self, prompt: str, system: str,
                         max_tokens: int, model: str,
                         timeout: int,
                         temperature: float = 0.8,
                         top_p: float = 0.95,
                         force_local: bool = False) -> Optional[str]:
        """hailo-ollama Chat API (Port 8000) mit Circuit-Breaker."""
        if not self._ollama_available:
            return None

        # Circuit-Breaker: Backoff aktiv?
        # Bei force_local trotzdem versuchen — Moloch soll lokal antworten
        if time.monotonic() < self._ollama_backoff_until:
            if not force_local:
                verbleibend = int(self._ollama_backoff_until - time.monotonic())
                logger.info(f"[LLM] Ollama Backoff aktiv ({verbleibend}s), direkt Cloud")
                return None
            logger.info("[LLM] force_local: ignoriere Backoff, versuche lokal")

        # Health-Check: nicht erreichbar → Fehlerzaehler erhoehen
        if not self._is_ollama_running():
            self._ollama_fail_count += 1
            if self._ollama_fail_count >= 3:
                self._ollama_backoff_until = time.monotonic() + self.OLLAMA_BACKOFF_SEC
                logger.warning(
                    f"[LLM] Ollama {self._ollama_fail_count}x down → "
                    f"{self.OLLAMA_BACKOFF_SEC}s Cloud-Backoff"
                )
            logger.debug("[LLM-BRIDGE] hailo-ollama nicht erreichbar")
            return None

        # Input-Length-Check
        input_len = len(system) + len(prompt)
        if input_len > OLLAMA_MAX_INPUT_CHARS:
            if not force_local:
                logger.info(f"[LLM] Input zu lang ({input_len} Zeichen > {OLLAMA_MAX_INPUT_CHARS}) → Cloud-Fallback")
                return None
            # force_local: Prompt kuerzen statt zur Cloud zu fallen
            # System-Prompt bleibt intact, User-Prompt wird von hinten beibehalten
            allowed = OLLAMA_MAX_INPUT_CHARS - len(system) - 100
            if allowed < 300:
                logger.warning("[LLM] force_local: System-Prompt zu lang, kein Platz fuer User-Input")
                return None
            prompt = prompt[-allowed:]
            logger.info(f"[LLM] force_local: Prompt auf {len(prompt)} Zeichen gekuerzt (Tension/Shadow/Berserker)")

        # Vision-Pause DEAKTIVIERT — hailo-ollama nutzt SHARED VDevice,
        # Hailo-Scheduler time-sliced automatisch. TAPPAS stoppen wuerde
        # alle Worker mit HAILO_COMMUNICATION_CLOSED(62) crashen → SIGTRAP.

        resp = None
        try:
            # hailo-ollama JSON-Parser crasht bei unescaped \n in content (2026-04-19).
            # Newlines durch Spaces ersetzen — Qwen2.5 antwortet auch auf einzeiligen Prompt korrekt.
            def _flatten(s: str) -> str:
                return s.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')

            # LLM-Profile-System hat Vorrang: aktives Profil aus llm_profiles.json laden.
            # settings.llm_profile > profiles.active. Profile bringt eigene system/temp/tokens mit.
            # Phase 1c: System-Profil -> Bridge-Satz -> ThreeBrain -> LiveContext -> Memory
            #   -> Identity (conditional in LiveContext via _is_hardware_query).
            profile = _get_active_profile()
            if profile is not None:
                profile_system = profile.get("system", OLLAMA_LOCAL_SYSTEM_COMPACT)
                # Phase 1a: Identity-Bridge-Satz direkt nach Profil
                profile_system = profile_system + "\n" + _IDENTITY_BRIDGE
                # Phase 1c: ThreeBrain VOR LiveContext (war bisher nur DeepSeek-Pfad)
                tb_state = _build_threebrain_state_snippet(max_chars=600)
                if tb_state:
                    profile_system += tb_state
                if profile.get("include_live_context", False):
                    profile_system = profile_system + _build_local_context_snippet(prompt)
                system = profile_system
                # Profile-Sampling ueberschreibt Caller-Defaults
                pmt = profile.get("max_tokens")
                if isinstance(pmt, int) and pmt > 0:
                    max_tokens = pmt
                # Phase 1b: adaptiver Token-Cut bei Multi-Turn-Drift
                if self._last_response_latency_s > self._adaptive_drift_threshold_s:
                    cut = self._adaptive_drift_max_tokens
                    if max_tokens > cut:
                        logger.info(
                            f"[LLM] adaptive Token-Reduktion {max_tokens}->{cut} "
                            f"(letzte Latenz {self._last_response_latency_s:.1f}s)"
                        )
                        max_tokens = cut
                ptemp = profile.get("temperature")
                if isinstance(ptemp, (int, float)):
                    temperature = float(ptemp)
                logger.info(f"[LLM] Profil aktiv: {profile.get('system','')[:30]}... ({len(system)} Zeichen, max_tokens={max_tokens}, temp={temperature})")
            elif system and len(system) > OLLAMA_LOCAL_SYSTEM_MAX:
                # Fallback wenn keine Profile-Datei: alter Compact-Pfad
                ctx = _build_local_context_snippet(prompt)
                system = OLLAMA_LOCAL_SYSTEM_COMPACT + "\n" + _IDENTITY_BRIDGE + ctx
                logger.info(f"[LLM] System-Prompt gekuerzt -> kompakte Persona + Kontext ({len(system)} Zeichen)")

            messages = []
            if system:
                messages.append({"role": "system", "content": _flatten(system)})
            messages.append({"role": "user", "content": _flatten(prompt)})

            resp = self._http.post(
                f"{OLLAMA_HOST}/api/chat",
                json={"model": model, "messages": messages, "stream": False,
                      "options": {"num_predict": max_tokens,
                                  "temperature": temperature,
                                  "top_p": top_p}},
                timeout=timeout)
            resp.raise_for_status()
            # Explizit UTF-8 dekodieren — resp.json() kann bei fehlendem charset-Header
            # Latin-1 waehlen → Umlaute werden als Ã¼ statt ü dargestellt
            data = json.loads(resp.content.decode('utf-8'))
            text = data.get("message", {}).get("content", "").strip()

            # DeepSeek R1 <think>...</think> Block entfernen (nur Antwort behalten)
            import re
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

            if not text:
                return None

            # Erfolg: Circuit-Breaker zuruecksetzen
            self._ollama_fail_count = 0
            self._ollama_backoff_until = 0.0
            self._last_provider = f"lokal_{model.split(':')[0]}"
            # Phase 1b: Latenz tracken fuer adaptiven Token-Cut beim naechsten Request
            duration_ns = data.get('total_duration', 0) or 0
            self._last_response_latency_s = duration_ns / 1_000_000_000
            logger.info(
                f"[LLM-BRIDGE] {model}: {len(text)} Zeichen in "
                f"{duration_ns // 1_000_000}ms"
            )
            return text

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            # Verbindungsfehler waehrend Generation → Fehlerzaehler
            self._ollama_fail_count += 1
            if self._ollama_fail_count >= 3:
                self._ollama_backoff_until = time.monotonic() + self.OLLAMA_BACKOFF_SEC
                logger.warning(
                    f"[LLM] Ollama {self._ollama_fail_count}x Verbindungsfehler → "
                    f"{self.OLLAMA_BACKOFF_SEC}s Cloud-Backoff"
                )
            logger.warning(f"[LLM-BRIDGE] hailo-ollama ({model}) Verbindungsfehler: {e}")
            return None

        except Exception as e:
            # HTTP 500 und andere Fehler: auch im Circuit-Breaker zaehlen
            self._ollama_fail_count += 1
            if self._ollama_fail_count >= 3:
                self._ollama_backoff_until = time.monotonic() + self.OLLAMA_BACKOFF_SEC
                logger.warning(
                    f"[LLM] Ollama {self._ollama_fail_count}x Fehler → "
                    f"{self.OLLAMA_BACKOFF_SEC}s Cloud-Backoff aktiv"
                )
            logger.warning(f"[LLM-BRIDGE] hailo-ollama ({model}) Fehler: {e}")
            return None

        finally:
            if resp is not None:
                resp.close()

    def _load_api_key(self, provider: str) -> Optional[str]:
        """API Key aus config/api_keys.json laden."""
        keys_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "config", "api_keys.json")
        # Env-Var hat Vorrang
        env_key = os.environ.get(f"{provider.upper()}_API_KEY")
        if env_key:
            return env_key
        try:
            import json
            with open(keys_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get(provider, {}).get("api_key")
        except Exception:
            return None

    # ========================================================================
    # TENTAKEL-LLM (Ollama auf Markus-Rechner, LAN) — Session 20
    # ========================================================================

    def _tentacle_url(self, cfg: Dict) -> str:
        return f"http://{cfg['host']}:{cfg['port']}"

    def _is_tentacle_running(self, cfg: Optional[Dict] = None) -> bool:
        """HTTP-Ping gegen Tentakel-Ollama (/api/tags)."""
        if cfg is None:
            cfg = _load_tentacle_cfg()
        if not cfg.get("enabled"):
            return False
        resp = None
        try:
            resp = self._http.get(f"{self._tentacle_url(cfg)}/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False
        finally:
            if resp is not None:
                resp.close()

    def _discover_tentacle_model(self, cfg: Dict) -> Optional[str]:
        """Waehlt ein Chat-Modell aus /api/tags wenn cfg.model leer ist.

        Heuristik: groesstes Modell nach 'size' (Byte), embedding-Modelle raus.
        Ergebnis wird in self._tentacle_model_cached gecached.
        """
        model = cfg.get("model") or ""
        if model:
            return model
        if self._tentacle_model_cached:
            return self._tentacle_model_cached
        try:
            resp = self._http.get(f"{self._tentacle_url(cfg)}/api/tags", timeout=3)
            resp.raise_for_status()
            data = resp.json()
            candidates = []
            for m in data.get("models", []) or []:
                name = m.get("name", "")
                if not name:
                    continue
                lname = name.lower()
                if any(x in lname for x in ("embed", "embedding", "nomic-embed")):
                    continue
                size = int(m.get("size", 0) or 0)
                candidates.append((size, name))
            if not candidates:
                logger.warning("[LLM-TENTACLE] /api/tags leer — kein Modell verfuegbar")
                return None
            candidates.sort(reverse=True)
            chosen = candidates[0][1]
            self._tentacle_model_cached = chosen
            logger.info(f"[LLM-TENTACLE] Auto-Discovery: nutze '{chosen}'")
            return chosen
        except Exception as e:
            logger.warning(f"[LLM-TENTACLE] Auto-Discovery Fehler: {e}")
            return None

    def _generate_tentacle(self, prompt: str, system: str,
                           max_tokens: int,
                           temperature: float = 0.7,
                           top_p: float = 0.95,
                           prompt_type: Optional[str] = None) -> Optional[str]:
        """Ollama-Tentakel auf LAN-Rechner (Standard-Ollama-API /api/chat).

        Welle 5 — prompt_type-abhaengiges Routing:
        - code_query  -> cfg["code_model"] (deepseek-coder) wenn vorhanden
        - web_research -> default model + search_proxy-Kontext im System-Prompt
        - sonst -> default model via _discover_tentacle_model
        """
        cfg = _load_tentacle_cfg()
        if not cfg.get("enabled"):
            return None
        # Circuit-Breaker
        if time.monotonic() < self._tentacle_backoff_until:
            verbleibend = int(self._tentacle_backoff_until - time.monotonic())
            logger.debug(f"[LLM-TENTACLE] Backoff aktiv ({verbleibend}s)")
            return None
        if not self._is_tentacle_running(cfg):
            self._tentacle_fail_count += 1
            if self._tentacle_fail_count >= 3:
                self._tentacle_backoff_until = time.monotonic() + cfg.get("backoff_sec", 300)
                logger.warning(
                    f"[LLM-TENTACLE] {self._tentacle_fail_count}x unreachable → "
                    f"{cfg.get('backoff_sec',300)}s Backoff"
                )
            return None

        # Modell-Wahl: code_model fuer code_query, sonst default
        if prompt_type == "code_query" and cfg.get("code_model"):
            model = cfg["code_model"]
            logger.info(f"[LLM-TENTACLE] code_query -> {model}")
        else:
            model = self._discover_tentacle_model(cfg)
        if not model:
            return None

        # web_research: PC search_proxy-Ergebnisse als Kontext prepend.
        # Augmentation im USER-Prompt (statt System-Prompt) — dolphin-llama3:8b
        # ignoriert System-Block-Anweisungen wenn Profil-System dominiert. User-Prompt
        # ist die letzte Sache vor der Antwort und wird zuverlaessig befolgt.
        if prompt_type == "web_research":
            web_ctx = _fetch_search_context(prompt)
            if web_ctx:
                prompt = (
                    web_ctx
                    + "\n\n=== ENDE LIVE-SUCHE ===\n\nFRAGE des Users (beantworte NUR auf Basis der Live-Suche oben, referenziere mindestens eine URL):\n"
                    + prompt
                )
                logger.info(f"[LLM-TENTACLE] web_research: {len(web_ctx)} Zeichen Search-Kontext im User-Prompt augmentiert")
        timeout_s = int(cfg.get("timeout_sec", 30))

        # Profile-Wahl Tentakel-spezifisch:
        # 1) Profil 'tentacle' wenn vorhanden (eigene Stimme fuer 7B-Mistral)
        # 2) sonst aktives Profil (settings.llm_profile)
        # 3) sonst Fallback: TENTACLE_SYSTEM_COMPACT-Konstante
        profile = None
        profiles_data = _load_profiles()
        if profiles_data:
            tentacle_profile = (profiles_data.get("profiles", {}) or {}).get("tentacle")
            profile = tentacle_profile or _get_active_profile()
        if profile is not None:
            profile_system = profile.get("system") or system or TENTACLE_SYSTEM_COMPACT
            # Phase 1a: Identity-Bridge-Satz nach Profil
            profile_system = profile_system + "\n" + _IDENTITY_BRIDGE
            # Phase 1c: ThreeBrain VOR LiveContext
            tb_state = _build_threebrain_state_snippet(max_chars=600)
            if tb_state:
                profile_system += tb_state
            if profile.get("include_live_context", True):
                profile_system = profile_system + _build_local_context_snippet(prompt)
            system = profile_system
            pmt = profile.get("max_tokens")
            if isinstance(pmt, int) and pmt > 0:
                max_tokens = pmt
            ptemp = profile.get("temperature")
            if isinstance(ptemp, (int, float)):
                temperature = float(ptemp)
        elif not system:
            # Letzter Fallback wenn weder Profile noch User-System gegeben
            system = TENTACLE_SYSTEM_COMPACT + "\n" + _IDENTITY_BRIDGE + _build_local_context_snippet(prompt)

        # Memory-Kontext (Identity + Top-5 Fakten + letzte 3 Turns + Core State)
        # an Mistral mitgeben. Mistral kann das fuer kontextreiche Antworten nutzen
        # (z.B. Markus' Profil, Crew-Namen, Rebecca-Klingonisch, Genesis-Datum).
        try:
            from core.longterm_memory import get_memory
            memory_ctx = get_memory().get_memory_context_minimal()
            if memory_ctx:
                system = (system or "") + "\n\n--- MEMORY ---\n" + memory_ctx
        except Exception:
            pass  # Memory-Singleton evtl. nicht init in standalone Test

        # JSON-sicher machen (wie bei hailo-ollama — Standard-Ollama hat
        # zwar robusteren Parser, aber Konsistenz zahlt sich aus)
        def _flatten(s: str) -> str:
            return (s or "").replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')

        messages = []
        if system:
            # system MIT Newlines senden — _flatten wuerde PRONOMEN/Crew/MEMORY-Struktur killen.
            # Ollama-Standard kann Newlines im JSON-content handlen.
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": _flatten(prompt)})

        resp = None
        try:
            t0 = time.monotonic()
            resp = self._http.post(
                f"{self._tentacle_url(cfg)}/api/chat",
                json={"model": model, "messages": messages, "stream": False,
                      "options": {"num_predict": max_tokens,
                                  "temperature": temperature,
                                  "top_p": top_p}},
                timeout=timeout_s)
            resp.raise_for_status()
            data = json.loads(resp.content.decode('utf-8'))
            text = (data.get("message", {}) or {}).get("content", "").strip()
            import re
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            if not text:
                return None
            self._tentacle_fail_count = 0
            self._tentacle_backoff_until = 0.0
            self._last_provider = f"tentacle_{model.split(':')[0]}"
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.info(f"[LLM-BRIDGE] tentacle {model}: {len(text)} Zeichen in {elapsed_ms}ms")
            return text
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            self._tentacle_fail_count += 1
            if self._tentacle_fail_count >= 3:
                self._tentacle_backoff_until = time.monotonic() + cfg.get("backoff_sec", 300)
            logger.warning(f"[LLM-TENTACLE] {model} Verbindungsfehler: {e}")
            return None
        except Exception as e:
            logger.warning(f"[LLM-TENTACLE] {model} Fehler: {e}")
            return None
        finally:
            if resp is not None:
                resp.close()

    # ========================================================================

    def _generate_deepseek(self, prompt: str, system: str,
                           max_tokens: int) -> Optional[str]:
        """DeepSeek API (Cloud, guenstig).

        Wenn DeepSeek als Primary laeuft (llm_mode=cloud_only), braucht sie
        die Tentakel-Persona genauso wie der Tentakel-Pfad — sonst antwortet
        sie ohne Charakter. Also: wenn kein system uebergeben, laden wir das
        tentacle-Profil + Live-Kontext + Memory analog zu _generate_tentacle.
        """
        api_key = self._load_api_key("deepseek")
        if not api_key:
            return None

        # Persona-Injection wenn chat_server stateless anruft (system leer)
        # Phase 1c-Reihenfolge: Profil -> Bridge -> ThreeBrain -> LiveContext -> Memory
        if not system:
            profiles_data = _load_profiles()
            if profiles_data:
                tentacle_profile = (profiles_data.get("profiles", {}) or {}).get("tentacle")
                profile = tentacle_profile or _get_active_profile()
                if profile is not None:
                    system = profile.get("system") or ""
                    # Phase 1a: Bridge-Satz nach Profil
                    if system:
                        system += "\n" + _IDENTITY_BRIDGE
                    # Phase 1c: ThreeBrain VOR LiveContext
                    tb_state = _build_threebrain_state_snippet()
                    if tb_state:
                        system += tb_state
                    if profile.get("include_live_context", True):
                        system = system + _build_local_context_snippet(prompt)
                    pmt = profile.get("max_tokens")
                    if isinstance(pmt, int) and pmt > 0:
                        max_tokens = pmt
            try:
                from core.longterm_memory import get_memory
                memory_ctx = get_memory().get_memory_context_minimal()
                if memory_ctx:
                    system = (system or "") + "\n\n--- MEMORY ---\n" + memory_ctx
            except Exception:
                pass

        resp = None
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = self._http.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "deepseek-chat", "messages": messages,
                      "max_tokens": max_tokens},
                timeout=15)
            resp.raise_for_status()
            self._last_provider = "api_deepseek"
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.debug(f"[LLM-BRIDGE] DeepSeek Fehler: {e}")
            return None
        finally:
            if resp is not None:
                resp.close()

    def get_status(self) -> Dict:
        now = time.monotonic()
        backoff_remaining = max(0.0, self._ollama_backoff_until - now)
        return {
            "llm_mode": self._llm_mode,
            "ollama_installed": self._ollama_available,
            "ollama_running": self._is_ollama_running() if self._ollama_available else False,
            "ollama_fail_count": self._ollama_fail_count,
            "ollama_backoff_sec": round(backoff_remaining),
            "last_provider": self._last_provider,
            "request_count": self._request_count,
            "models": {
                "chat": OLLAMA_MODEL_CHAT,
                "reason": OLLAMA_MODEL_REASON,
            },
        }


# Singleton
_instance: Optional[LocalLLMBridge] = None

def get_llm_bridge() -> LocalLLMBridge:
    global _instance
    if _instance is None:
        _instance = LocalLLMBridge()
    return _instance
