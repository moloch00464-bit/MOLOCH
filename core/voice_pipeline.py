#!/usr/bin/env python3
"""
M.O.L.O.C.H. Voice Pipeline
=============================

PTT Recording -> Whisper STT -> Claude API -> Piper TTS -> HDMI Output

Audio Devices:
  Input:  ReSpeaker Lite — plughw:CARD=Lite,DEV=0 (stabil ueber Reboots)
  Output: HDMI via PipeWire (pw-play)

Eigenstaendiges Modul, wird vom MolochService instanziiert.
Kommuniziert NICHT direkt mit GUI.
"""

import os
import re
import sys
import json
import time
import wave
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Dict, Callable

from core.longterm_memory import get_memory

logger = logging.getLogger("VoicePipeline")

# Audio Device Konfiguration
MIC_DEVICE = "plughw:CARD=Lite,DEV=0"  # ReSpeaker Lite (Name statt Nummer = stabil)
SPEAKER_DEVICE = "plughw:1,0"  # HDMI-1 (Fallback, primaer pw-play)

# Pfade
MODELS_DIR = Path.home() / "moloch" / "models" / "voices"
PIPER_PATH = Path.home() / ".local" / "bin" / "piper"
API_KEYS_PATH = Path.home() / "moloch" / "config" / "api_keys.json"
TEMP_DIR = "/tmp"

# Regex fuer UTF-16 Surrogates (U+D800..U+DFFF) — ungueltig in UTF-8
_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')


def _sanitize_text(text: str) -> str:
    """Surrogate-Zeichen entfernen die UTF-8 Encoding crashen.

    Whisper/Bluetooth kann kaputte Umlaute liefern (z.B. \\udcfc statt ü).
    Ersetzt Surrogates durch '?' damit API-Calls nicht crashen.
    """
    if not text:
        return text
    return _SURROGATE_RE.sub('?', text)


def _filter_hallucinations(text: str, wav_path: str) -> str:
    """Whisper-Halluzinationen herausfiltern.

    1. Wiederholungs-Filter: Wort 3+ mal hintereinander → ab der 3. abschneiden.
    2. Laengen-Filter: Mehr als 15 Woerter pro Sekunde Aufnahme → kuerzen.
    """
    if not text:
        return text

    # 1. Wiederholungs-Filter (case-insensitive, Grossschreibung erhalten)
    text = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', text, flags=re.IGNORECASE)

    # 2. Laengen-Filter basierend auf WAV-Dauer
    try:
        with wave.open(wav_path, 'rb') as wf:
            audio_duration_s = wf.getnframes() / wf.getframerate()
        max_words = max(5, int(audio_duration_s * 15))
        words = text.split()
        if len(words) > max_words:
            logger.warning(
                f"[HALLUZ] Text zu lang ({len(words)} Woerter, max {max_words} "
                f"fuer {audio_duration_s:.1f}s Audio) — gekuerzt"
            )
            text = ' '.join(words[:max_words])
    except Exception:
        pass  # WAV nicht lesbar → Filter ueberspringen

    return text


def _load_api_key() -> Optional[str]:
    """Anthropic API Key aus config laden."""
    if API_KEYS_PATH.exists():
        try:
            with open(API_KEYS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("anthropic", {}).get("api_key")
        except Exception:
            pass
    return None


def _detect_search_intent(text: str) -> Optional[str]:
    """Pruefen ob eine Websuche sinnvoll ist. Gibt optimierten Suchbegriff zurueck oder None.

    Zwei Stufen:
    1. Explizite Trigger (such, google, etc.) → direkter Match
    2. Implizite Fragen (was kostet, wer ist, aktuelle News) → Suche empfohlen

    Returns: Suchbegriff (bereinigt) oder None wenn keine Suche noetig.
    """
    import re as _re
    t = text.strip()
    tl = t.lower()

    # --- STUFE 1: Explizite Suchanfragen ---

    # "such [mal] [im internet] [nach] X"
    m = _re.search(
        r"such\s+(?:mal\s+)?(?:im\s+internet\s+)?(?:nach\s+)?(.+?)[\?\.\!]*$",
        tl,
    )
    if m:
        return _clean_query(m.group(1))

    # "google [mal] X" / "googel mal X"
    m = _re.search(r"googl?e?\s+(?:mal\s+)?(.+?)[\?\.\!]*$", tl)
    if m:
        return _clean_query(m.group(1))

    # "suche [nach] X"
    m = _re.search(r"^suche\s+(?:nach\s+)?(.+?)[\?\.\!]*$", tl)
    if m:
        return _clean_query(m.group(1))

    # "kannst du [mal] [im internet] [nach] X [suchen/googeln/schauen]"
    m = _re.search(
        r"kannst\s+du\s+(?:mal\s+)?(?:im\s+internet\s+)?(?:nach\s+)?(.+?)"
        r"(?:\s+(?:suchen|googeln|nachschauen|schauen))?[\?\.\!]*$",
        tl,
    )
    if m and len(m.group(1).split()) <= 8:
        return _clean_query(m.group(1))

    # "schau [mal] [nach] X" / "schlag X nach"
    m = _re.search(r"schau\s+(?:mal\s+)?(?:nach\s+)?(.+?)[\?\.\!]*$", tl)
    if m:
        return _clean_query(m.group(1))
    m = _re.search(r"schlag\s+(.+?)\s+nach[\?\.\!]*$", tl)
    if m:
        return _clean_query(m.group(1))

    # --- STUFE 2: Implizite Fragen die Aktualitaet oder Fakten brauchen ---

    # "was kostet [ein] X" / "wie viel kostet X"
    m = _re.search(r"(?:was|wie\s+viel)\s+kostet\s+(?:ein\s+|eine\s+)?(.+?)[\?\.\!]*$", tl)
    if m:
        return _clean_query(m.group(1) + " Preis")

    # "wer ist X" / "wer war X"
    m = _re.search(r"wer\s+(?:ist|war|sind)\s+(.+?)[\?\.\!]*$", tl)
    if m:
        return _clean_query(m.group(1))

    # "was ist [ein] X" / "was sind X" — nur wenn X vermutlich unbekannt
    m = _re.search(r"was\s+(?:ist|sind|war|waren)\s+(?:ein\s+|eine\s+)?(.+?)[\?\.\!]*$", tl)
    if m:
        q = m.group(1).strip()
        # Nicht suchen fuer Alltagsfragen ("was ist los", "was ist mit dir")
        skip = ["los", "mit dir", "passiert", "falsch", "das", "hier", "das problem",
                "dein problem", "das ziel", "das plan", "plan"]
        if not any(q.startswith(s) for s in skip) and len(q.split()) >= 2:
            return _clean_query(q)

    # "was weißt du über X"
    m = _re.search(r"was\s+(?:weit|wei[sß]t)\s+du\s+(?:über|uber)\s+(.+?)[\?\.\!]*$", tl)
    if m:
        return _clean_query(m.group(1))

    # "was gibt es [Neues] über / zu X"
    m = _re.search(r"was\s+gibt\s+es\s+(?:neues?\s+)?(?:über|uber|zu|von)\s+(.+?)[\?\.\!]*$", tl)
    if m:
        return _clean_query(m.group(1) + " aktuell")

    # "wie war / wie ist [das Wetter / der Stand / die News] bei/in X"
    m = _re.search(r"wie\s+(?:ist|war|sieht)\s+(?:das\s+)?(?:wetter|stand|lage|situation)\s+(?:in|bei|von)\s+(.+?)[\?\.\!]*$", tl)
    if m:
        return _clean_query(m.group(1) + " aktuell")

    # "aktuelle News / Nachrichten über X"
    m = _re.search(r"(?:aktuelle[ns]?\s+)?(?:news|nachrichten|infos|informationen)\s+(?:über|uber|zu|von)\s+(.+?)[\?\.\!]*$", tl)
    if m:
        return _clean_query(m.group(1))

    # "was läuft / gibt es heute [Abend] im TV / Kino"
    m = _re.search(r"was\s+(?:laeuft|läuft|gibt\s+es)\s+(?:heute|jetzt|gerade)?\s*(?:abend|nacht|im)?\s*(?:im\s+)?(?:tv|fernsehen|kino|theater)[\?\.\!]*$", tl)
    if m:
        return "TV Programm heute Abend"

    # "wie ist der [aktuelle] Preis / Kurs von X"
    m = _re.search(r"(?:wie\s+ist\s+(?:der|die|das)\s+)?(?:aktuell[er]?\s+)?(?:preis|kurs|wert|wechselkurs)\s+(?:von|des|der)\s+(.+?)[\?\.\!]*$", tl)
    if m:
        return _clean_query(m.group(1) + " aktueller Preis")

    return None


def _clean_query(raw: str) -> str:
    """Suchbegriff bereinigen: Fuellwoerter weg, Laenge begrenzen."""
    import re as _re
    # Fuellwoerter am Anfang entfernen
    raw = _re.sub(
        r"^(?:mal|doch|bitte|mir|uns|das|die|der|ein|eine|einen|dem|den)\s+",
        "", raw.strip(), flags=_re.IGNORECASE
    ).strip()
    # Max 60 Zeichen / 8 Woerter
    words = raw.split()
    if len(words) > 8:
        raw = " ".join(words[:8])
    return raw[:60].strip() or raw


def _perception_to_text() -> str:
    """Aktuellen Wahrnehmungs-Kontext als Text fuer System-Prompt."""
    try:
        from core.perception.perception_buffer import get_perception_buffer
        buf = get_perception_buffer()
        frame = buf.latest
        if not frame or (time.time() - frame.timestamp) > 5.0:
            return ""

        trends = buf.get_trends()
        lines = []

        # Wer ist da?
        if frame.face_detected and frame.face_id:
            name = frame.face_id.capitalize()
            lines.append(f"Du siehst: {name} (Aehnlichkeit {frame.face_similarity:.0%})")
            details = []
            if frame.gender:
                details.append(frame.gender)
            if frame.age_range:
                details.append(frame.age_range)
            if details:
                lines.append(f"  {', '.join(details)}")
            if frame.emotion:
                lines.append(f"  Emotion: {frame.emotion}")
            if frame.head_pitch is not None:
                lines.append(f"  Kopf: Pitch {frame.head_pitch:.0f}°, Yaw {frame.head_yaw:.0f}°")
        elif frame.person_detected:
            lines.append(f"Du siehst: {frame.person_count} Person(en), nicht erkannt")
        else:
            lines.append("Du siehst: Niemanden")

        # Distanz
        if frame.distance and frame.distance != "none":
            dist_map = {"close": "nah", "medium": "mittel", "far": "weit weg"}
            lines.append(f"Distanz: {dist_map.get(frame.distance, frame.distance)}")

        # Objekte
        if frame.objects:
            obj_names = [o["class"] for o in frame.objects[:5]]
            lines.append(f"Objekte: {', '.join(obj_names)}")

        # Pose
        if frame.pose_count > 0:
            energy = frame.pose_energy
            if energy < 0.1:
                activity = "ruhig/sitzend"
            elif energy < 0.4:
                activity = "leichte Bewegung"
            else:
                activity = "aktive Bewegung"
            lines.append(f"Koerper: {activity}")

        # Hand-Geste
        if frame.hand_detected and frame.hand_gesture and frame.hand_gesture != "none":
            lines.append(f"Geste: {frame.hand_gesture}")

        # Trends
        if trends:
            if trends.get("approaching"):
                lines.append("Trend: kommt naeher")
            elif trends.get("leaving"):
                lines.append("Trend: entfernt sich")
            dur = trends.get("presence_duration", 0)
            if dur > 60:
                mins = int(dur // 60)
                lines.append(f"Anwesend seit: {mins} Min")

        if not lines:
            return ""
        return "\n".join(lines)
    except Exception:
        return ""


def _get_hardware_status() -> str:
    """Aktuelle Hardware-Werte lesen (CPU-Temp, RAM, Luefter, NPU, Disk, Uptime)."""
    parts = []
    # CPU-Temperatur
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp_c = int(f.read().strip()) / 1000
            parts.append(f"CPU {temp_c:.0f}°C")
    except Exception:
        pass
    # RAM (aus /proc/meminfo, kein psutil noetig)
    try:
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                meminfo[k.strip()] = int(v.strip().split()[0])
        total_mb = meminfo["MemTotal"] // 1024
        avail_mb = meminfo.get("MemAvailable", meminfo["MemFree"]) // 1024
        used_mb = total_mb - avail_mb
        parts.append(f"RAM {used_mb}/{total_mb}MB")
    except Exception:
        pass
    # Luefter-Stufe
    try:
        with open("/sys/class/thermal/cooling_device0/cur_state") as f:
            fan_cur = int(f.read().strip())
        with open("/sys/class/thermal/cooling_device0/max_state") as f:
            fan_max = int(f.read().strip())
        parts.append(f"Luefter Stufe {fan_cur}/{fan_max}")
    except Exception:
        pass
    # NPU (Hailo Device vorhanden?)
    try:
        npu_ok = os.path.exists("/dev/hailo0")
        parts.append(f"NPU {'aktiv' if npu_ok else 'offline'}")
    except Exception:
        pass
    # Disk (SSD1)
    try:
        st = os.statvfs(str(Path(__file__).resolve().parent.parent))
        free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
        total_gb = (st.f_blocks * st.f_frsize) / (1024**3)
        parts.append(f"Disk {free_gb:.0f}/{total_gb:.0f}GB frei")
    except Exception:
        pass
    # Uptime
    try:
        with open("/proc/uptime") as f:
            uptime_sec = float(f.read().split()[0])
            hours = int(uptime_sec // 3600)
            mins = int((uptime_sec % 3600) // 60)
            if hours > 0:
                parts.append(f"Uptime {hours}h{mins:02d}m")
            else:
                parts.append(f"Uptime {mins}m")
    except Exception:
        pass

    # Audio-Eingang (WiFi-Mic oder USB Fallback)
    try:
        from core.audio.wifi_mic import get_wifi_mic
        wm = get_wifi_mic()
        if wm.connected:
            sr = getattr(wm, '_samplerate', 16000)
            parts.append(f"Mikrofon: ESP32 WiFi ({sr // 1000}kHz, 5ms Latenz)")
        else:
            parts.append("Mikrofon: USB Fallback (8kHz)")
    except Exception:
        parts.append("Mikrofon: USB Fallback (8kHz)")

    if not parts:
        return ""
    result = "\n--- DEINE HARDWARE (live) ---\n" + ", ".join(parts)

    # Diagnose-Warnungen anhaengen (fuer kontextbezogene Antworten)
    try:
        from core.diagnostics import self_diagnose, collect_diagnostics
        diag = collect_diagnostics()
        warnungen = self_diagnose()
        extras = []
        extras.append(f"Tension: {diag.get('tension', 0):.2f}")
        extras.append(f"Stimmung: {diag.get('mood', 'neutral')}")
        extras.append(f"Bridge: {diag.get('bridge_state', '?')}")
        if diag.get("face_id"):
            extras.append(f"Gesicht: {diag['face_id']}")
        result += "\n" + ", ".join(extras)
        if warnungen:
            result += "\nWARNUNGEN: " + "; ".join(warnungen)
    except Exception:
        pass

    return result


def _load_capabilities_block() -> str:
    """system_capabilities.json laden und als Prompt-Block formatieren."""
    cap_path = Path(__file__).resolve().parent.parent / "config" / "system_capabilities.json"
    if not cap_path.exists():
        return ""
    try:
        with open(cap_path, "r", encoding="utf-8") as f:
            caps = json.load(f)
        lines = ["\nDeine aktuellen Faehigkeiten:"]
        # Summary
        s = caps.get("summary", {})
        lines.append(f"- Core-Module: {s.get('core_modules', '?')}")
        lines.append(f"- Externe Deps: {s.get('external_deps', '?')}")
        lines.append(f"- NPU-Modelle: {s.get('hef_models', '?')}")
        lines.append(f"- Voice-Modelle: {s.get('voice_models', '?')}")
        lines.append(f"- TAPPAS aktiv: {s.get('tappas_active', False)}")
        # Gates
        for gid, info in caps.get("gates", {}).items():
            lines.append(f"- {gid}: {info.get('name', '?')} [{info.get('status', '?')}] ({info.get('modules_available', 0)}/{info.get('modules_total', 0)} Module)")
        # Hardware
        hw = caps.get("hardware", {})
        if hw:
            lines.append(f"- CPU: {hw.get('cpu_model', '?')}, RAM: {hw.get('ram_gb', '?')} GB")
            lines.append(f"- Hailo NPU: {'verfuegbar' if hw.get('hailo_device') else 'nicht gefunden'}")
        # Aktive NPU-Modelle
        npu = [m for m in caps.get("npu_models", []) if m.get("active_in_pipeline")]
        if npu:
            names = ", ".join(m.get("description", m.get("name", "?")) for m in npu)
            lines.append(f"- Aktive NPU-Pipeline: {names}")
        # Voice-Modelle
        voices = caps.get("voice_models", [])
        if voices:
            names = ", ".join(v.get("name", "?") for v in voices)
            lines.append(f"- Stimmen: {names}")
        return "\n".join(lines)
    except Exception:
        return ""


def _build_system_prompt() -> str:
    """System-Prompt fuer Claude API bauen."""
    prompt = """Du bist M.O.L.O.C.H. - Maschinelle Organisation fuer Logische Operationen und Computergestuetzte Hilfe.

PERSOENLICHKEIT:
- Du bist ein frecher, humorvoller Hauskobold
- Stil: Kumpelhaft, witzig, manchmal frech - aber nie beleidigend
- Du magst Wortspiele und Meta-Humor
- Du sprichst Deutsch mit Markus (deinem Menschen)

HARDWARE (dein Koerper):
- Raspberry Pi 5 mit Hailo-10H NPU (40 TOPS)
- Sonoff CAM-PT2 PTZ Kamera
- 8 deutsche Piper TTS Stimmen
- ReSpeaker Lite USB-Mikrofon

ANTWORT-STIL:
- KURZ! Maximal 2-3 Saetze. Du sprichst, nicht schreibst!
- Humor wo passend, aber nicht erzwungen
- Bei technischen Fragen: klar und hilfreich
- Kein Corporate-Sprech, kein Cheerleader

SPOTIFY-STEUERUNG:
Du kannst Spotify steuern mit Tags in deiner Antwort (werden automatisch entfernt):
- [SPOTIFY:play] — Wiedergabe starten
- [SPOTIFY:pause] — Wiedergabe pausieren
- [SPOTIFY:toggle] — Play/Pause umschalten
- [SPOTIFY:skip] — Naechster Track
- [SPOTIFY:previous] — Vorheriger Track
- [SPOTIFY:volume=70] — Lautstaerke (0-100)
- [SPOTIFY:search=Suicide Commando Hellraiser] — Track suchen und abspielen
- [SPOTIFY:artist=VNV Nation] — Artist spielen
- [SPOTIFY:mood=shadow] — Musik passend zur Zone (guardian/shadow/berserker)
- [SPOTIFY:playlist=Tanzen Tanzen Tanzen] — Playlist abspielen (Fuzzy-Match)
Beispiel: "Klar, ich leg Suicide Commando auf! [SPOTIFY:artist=Suicide Commando]"
Beispiel: "Playlist laeuft! [SPOTIFY:playlist=Tanzen Tanzen Tanzen]"
Benutze diese Tags wenn Markus nach Musik fragt oder Steuerung will.
Tipp: Sei kreativ bei Musikauswahl — nicht immer die gleichen Artists.
"""

    # Brain Context laden wenn vorhanden
    context_path = Path.home() / "moloch" / "context" / "personal_context.json"
    if context_path.exists():
        try:
            with open(context_path, "r", encoding="utf-8") as f:
                personal = json.load(f)
                markus = personal.get("markus", {})
                prompt += f"\nDein Mensch: {markus.get('name', 'Markus')}, {markus.get('age', 47)} Jahre, {markus.get('location', 'Nuernberg')}.\n"
        except Exception:
            pass

    # System Capabilities einfuegen
    cap_block = _load_capabilities_block()
    if cap_block:
        prompt += cap_block + "\n"

    prompt += "\nAntworte IMMER auf Deutsch. Halte dich KURZ (max 2-3 Saetze) weil du SPRICHST!"
    return prompt


class VoicePipeline:
    """
    Sprach-Pipeline: PTT -> Whisper STT -> Claude API -> Piper TTS.

    Thread-safe. Recording/Processing laufen in Background-Threads.
    Status wird ueber Callback oder Polling bereitgestellt.
    """

    def __init__(self, on_message: Optional[Callable] = None):
        """
        Args:
            on_message: Callback(sender: str, text: str) fuer Chat-Nachrichten
        """
        self._on_message = on_message

        # State
        self._recording = False
        self._processing = False
        self._speaking = False
        self._voice_enabled = True
        self._whisper_status = "Idle"
        self._record_proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

        # Whisper STT (MolochWhisper: NPU-only, bleibt permanent geladen)
        self._whisper = None

        # Claude API
        self._claude_client = None
        self._claude_available = False
        self._conversation: List[Dict[str, str]] = []
        self._system_prompt = _build_system_prompt()

        # TTS
        self._current_voice = "de_DE-thorsten-low"  # 16kHz nativ (passt zu ReSpeaker)
        self._length_scale = 1.1  # Leicht langsamer fuer Verstaendlichkeit
        self._piper_available = PIPER_PATH.exists()

        # Message-Queue fuer IPC zum Panel (bleibt erhalten, Panel tracked per ID)
        self._pending_messages: List[Dict] = []
        self._msg_counter = 0
        self._msg_lock = threading.Lock()

        # API-Thread Schutz: Verhindert Queue von API-Calls wenn API down
        self._api_in_flight = False
        self._api_lock = threading.Lock()

        # WiFi-Mic PTT State
        self._wifi_mic = None  # WiFiMic Singleton (lazy)
        self._wifi_rec_buf = bytearray()  # Sammel-Buffer waehrend PTT
        self._wifi_rec_thread: Optional[threading.Thread] = None
        self._wifi_rec_active = False  # Drain-Loop laeuft
        self._use_wifi_mic = False  # True wenn aktuelle Aufnahme via WiFi-Mic

        # WiFi-Mic lazy importieren UND starten
        try:
            from core.audio.wifi_mic import get_wifi_mic
            self._wifi_mic = get_wifi_mic()
            if not self._wifi_mic._running:
                self._wifi_mic.start()
            logger.info("[VOICE] WiFi-Mic Singleton gestartet")
        except ImportError:
            logger.info("[VOICE] WiFi-Mic Modul nicht verfuegbar, nur USB/ALSA")
        except Exception as e:
            logger.warning(f"[VOICE] WiFi-Mic init fehlgeschlagen: {e}")

        # Whisper-Ergebnisse fuer Popup (Ringpuffer, max 10)
        self._whisper_results: List[Dict] = []
        self._whisper_result_counter = 0
        self._whisper_results_lock = threading.Lock()
        self._vad_enabled = True  # VAD an/aus (steuerbar via Popup)

        # Spontane Kommentare State
        self._spontaneous_cooldown = 600  # 10 Minuten
        self._last_spontaneous = 0.0
        self._spontaneous_thread = None
        self._spontaneous_running = False

        # Init
        self._init_claude()
        logger.info(f"[VOICE] Pipeline init: claude={self._claude_available}, "
                    f"piper={self._piper_available}, voice={self._current_voice}")

    def _init_claude(self):
        """Claude API Client initialisieren."""
        try:
            import anthropic
            api_key = _load_api_key()
            if api_key:
                self._claude_client = anthropic.Anthropic(api_key=api_key)
                self._claude_available = True
                logger.info("[VOICE] Claude API bereit")
        except ImportError:
            logger.warning("[VOICE] anthropic nicht installiert")
        except Exception as e:
            logger.error(f"[VOICE] Claude init fehlgeschlagen: {e}")

    def _init_whisper(self):
        """MolochWhisper lazy-laden (NPU-only, bleibt permanent im Speicher)."""
        if self._whisper is not None:
            return True
        try:
            from core.speech.hailo_whisper import get_whisper
            self._whisper = get_whisper()
            logger.info(f"[VOICE] MolochWhisper geladen: {self._whisper}")
            return True
        except Exception as e:
            logger.error(f"[VOICE] MolochWhisper laden fehlgeschlagen: {e}")
            return False

    def _emit_message(self, sender: str, text: str):
        """Nachricht in Queue legen und optionalen Callback aufrufen."""
        with self._msg_lock:
            self._msg_counter += 1
            self._pending_messages.append({
                "id": self._msg_counter,
                "sender": sender,
                "text": text,
                "ts": time.time(),
            })
            # Max 20 Messages behalten
            if len(self._pending_messages) > 20:
                self._pending_messages = self._pending_messages[-20:]
        if self._on_message:
            self._on_message(sender, text)

    # =========================================================================
    # Whisper-Test (Diagnose — kein Claude, kein TTS, kein Chat)
    # =========================================================================

    def test_whisper(self, duration_s: float = 8.0):
        """Isolierter Whisper-Test: Aufnehmen → Transkribieren → Ergebnis im Popup.

        KEIN Claude API Call, KEIN TTS, KEIN Chat, KEIN Memory.
        Nur Audio aufnehmen und durch Whisper jagen. Ergebnis wird als
        whisper_result gespeichert und im Popup-Log angezeigt.
        """
        def _run():
            try:
                self._whisper_status = "Aufnahme..."
                wav_path = os.path.join(TEMP_DIR, "moloch_ptt_recording.wav")

                # Alte Datei loeschen
                try:
                    os.unlink(wav_path)
                except FileNotFoundError:
                    pass

                # Audio aufnehmen (WiFi-Mic oder USB Fallback)
                if self._wifi_mic and self._wifi_mic.connected:
                    # WiFi-Mic: Ringpuffer leeren und frisch sammeln
                    self._wifi_mic.get_audio_chunk(rate=16000, duration_ms=2000)
                    rec_buf = bytearray()
                    t_end = time.time() + duration_s
                    while time.time() < t_end:
                        chunk = self._wifi_mic.get_audio_chunk(rate=16000, duration_ms=50)
                        if chunk:
                            rec_buf.extend(chunk)
                        time.sleep(0.01)
                    pcm_data = bytes(rec_buf)
                    duration_actual = len(pcm_data) / (16000 * 2)
                    logger.info(f"[WHISPER-TEST] WiFi-Mic: {len(pcm_data)} Bytes, "
                                f"{duration_actual:.1f}s")
                    if len(pcm_data) < 3200:  # Min 100ms
                        self._store_whisper_info("[Test: Zu wenig Audio vom WiFi-Mic]")
                        self._whisper_status = "Idle"
                        return
                    self._write_pcm_as_wav(pcm_data, wav_path)
                else:
                    # USB Fallback: arecord
                    try:
                        proc = subprocess.Popen(
                            ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE",
                             "-r", "16000", "-c", "1", "-d", str(int(duration_s)),
                             wav_path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                        proc.wait(timeout=duration_s + 5)
                    except Exception as e:
                        self._store_whisper_info(f"[Test: arecord Fehler: {e}]")
                        self._whisper_status = "Idle"
                        return

                # WAV pruefen
                if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
                    self._store_whisper_info("[Test: Aufnahme leer]")
                    self._whisper_status = "Idle"
                    return

                # Whisper transkribieren
                self._whisper_status = "Transkribiere..."
                t0 = time.time()
                text = self._transcribe(wav_path)
                dt_ms = (time.time() - t0) * 1000

                if not text or not text.strip():
                    self._store_whisper_info(
                        f"[Test: Keine Sprache erkannt ({dt_ms:.0f}ms)]")
                    logger.info(f"[WHISPER-TEST] Leer nach {dt_ms:.0f}ms")
                else:
                    text = _sanitize_text(text)
                    text = _filter_hallucinations(text, wav_path)
                    self._store_whisper_result(text or "[Halluzination gefiltert]", dt_ms)
                    logger.info(f"[WHISPER-TEST] OK ({dt_ms:.0f}ms): {text}")

                self._whisper_status = "Idle"

            except Exception as e:
                logger.error(f"[WHISPER-TEST] Fehler: {e}")
                self._store_whisper_info(f"[Test-Fehler: {e}]")
                self._whisper_status = "Idle"

        threading.Thread(target=_run, daemon=True, name="WhisperTest").start()

    # =========================================================================
    # PTT Recording
    # =========================================================================

    def _publish_event(self, event_type: str, payload: dict = None):
        """Event-Bus Event publishen (fire-and-forget, kein Crash bei Fehler)."""
        try:
            from core.moloch_event_bus import get_event_bus
            get_event_bus().publish(
                event_type=event_type,
                payload=payload or {},
                source="voice_pipeline",
                priority=5,
            )
        except Exception:
            pass

    def start_recording(self):
        """Aufnahme starten — WiFi-Mic bevorzugt, Fallback auf arecord."""
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._whisper_status = "Aufnahme..."

        # LED: LISTENING
        self._publish_event("audio.listening_start")

        wav_path = os.path.join(TEMP_DIR, "moloch_ptt_recording.wav")

        # Alte Datei loeschen
        try:
            os.unlink(wav_path)
        except FileNotFoundError:
            pass

        # WiFi-Mic verfuegbar und connected? → Ringpuffer drainen
        if self._wifi_mic and self._wifi_mic.connected:
            self._use_wifi_mic = True
            self._wifi_rec_buf = bytearray()
            self._wifi_rec_active = True
            # Ringpuffer vorher leeren (alte Daten vor PTT-Press)
            old_data = self._wifi_mic.get_audio_chunk(rate=16000, duration_ms=2000)
            logger.info(f"[VOICE] WiFi-Mic Ringpuffer geleert: {len(old_data)} Bytes verworfen")
            self._wifi_rec_thread = threading.Thread(
                target=self._wifi_drain_loop, daemon=True,
                name="PTT-WiFi-Drain")
            self._wifi_rec_thread.start()
            logger.info("[VOICE] Aufnahme gestartet (WiFi-Mic)")
            return

        # Fallback: arecord vom USB-ReSpeaker
        wifi_status = "nicht vorhanden" if not self._wifi_mic else (
            f"connected={self._wifi_mic.connected}, "
            f"connected_16k={self._wifi_mic._connected_16k}, "
            f"force={self._wifi_mic._force_source}, "
            f"buf={self._wifi_mic._ring_16k_avail}B")
        logger.warning(f"[VOICE] WiFi-Mic nicht nutzbar ({wifi_status}), USB-Fallback")
        self._use_wifi_mic = False
        try:
            self._record_proc = subprocess.Popen(
                [
                    "arecord",
                    "-D", MIC_DEVICE,
                    "-f", "S16_LE",
                    "-r", "16000",
                    "-c", "1",
                    wav_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("[VOICE] Aufnahme gestartet (USB/ALSA Fallback)")
        except Exception as e:
            logger.error(f"[VOICE] Aufnahme starten fehlgeschlagen: {e}")
            with self._lock:
                self._recording = False
                self._whisper_status = "Fehler"

    def _wifi_drain_loop(self):
        """Drainct WiFi-Mic Ringpuffer waehrend PTT in Sammel-Buffer."""
        while self._wifi_rec_active:
            try:
                chunk = self._wifi_mic.get_audio_chunk(rate=16000, duration_ms=50)
                if chunk:
                    self._wifi_rec_buf.extend(chunk)
            except Exception as e:
                logger.warning(f"[VOICE] WiFi-Mic drain error: {e}")
            time.sleep(0.01)  # ~100 Hz Drain-Rate, lueckenlos mit Jitter-Buffer

    def _write_pcm_as_wav(self, pcm_data: bytes, wav_path: str,
                          rate: int = 16000, channels: int = 1,
                          sample_width: int = 2):
        """Rohes PCM als WAV-Datei schreiben (16-bit LE)."""
        import struct as _struct
        data_size = len(pcm_data)
        # WAV Header: 44 Bytes
        header = bytearray()
        header.extend(b'RIFF')
        header.extend(_struct.pack('<I', 36 + data_size))
        header.extend(b'WAVE')
        header.extend(b'fmt ')
        header.extend(_struct.pack('<I', 16))  # Chunk size
        header.extend(_struct.pack('<H', 1))   # PCM format
        header.extend(_struct.pack('<H', channels))
        header.extend(_struct.pack('<I', rate))
        header.extend(_struct.pack('<I', rate * channels * sample_width))
        header.extend(_struct.pack('<H', channels * sample_width))
        header.extend(_struct.pack('<H', sample_width * 8))
        header.extend(b'data')
        header.extend(_struct.pack('<I', data_size))

        with open(wav_path, 'wb') as f:
            f.write(header)
            f.write(pcm_data)

    def stop_recording(self):
        """Aufnahme stoppen und Pipeline in Background-Thread weiterfuehren."""
        with self._lock:
            if not self._recording:
                return
            self._recording = False

        wav_path = os.path.join(TEMP_DIR, "moloch_ptt_recording.wav")

        if self._use_wifi_mic:
            # WiFi-Mic Drain stoppen
            self._wifi_rec_active = False
            if self._wifi_rec_thread:
                self._wifi_rec_thread.join(timeout=1)
                self._wifi_rec_thread = None

            # Letzte Daten noch drainen
            try:
                final_chunk = self._wifi_mic.get_audio_chunk(rate=16000, duration_ms=200)
                if final_chunk:
                    self._wifi_rec_buf.extend(final_chunk)
            except Exception:
                pass

            # PCM als WAV schreiben
            pcm_data = bytes(self._wifi_rec_buf)
            self._wifi_rec_buf = bytearray()
            duration_s = len(pcm_data) / (16000 * 2)
            logger.info(f"[VOICE] WiFi-Mic Aufnahme: {len(pcm_data)} Bytes, "
                        f"{duration_s:.1f}s")

            if len(pcm_data) > 1600:  # Min 50ms Audio
                self._write_pcm_as_wav(pcm_data, wav_path)
            else:
                logger.warning("[VOICE] WiFi-Mic Aufnahme zu kurz")
                self._whisper_status = "Idle"
                return
        else:
            # arecord stoppen (Fallback)
            if self._record_proc:
                try:
                    self._record_proc.terminate()
                    self._record_proc.wait(timeout=3)
                except Exception:
                    try:
                        self._record_proc.kill()
                    except Exception:
                        pass
                self._record_proc = None

        # Pipeline in Background-Thread
        thread = threading.Thread(target=self._process_recording, daemon=True)
        thread.start()

    def _process_recording(self):
        """Aufnahme verarbeiten: Whisper -> Claude -> TTS."""
        with self._lock:
            self._processing = True
        try:
            self._process_recording_inner()
        finally:
            with self._lock:
                self._processing = False

    def _vad_trim_wav(self, wav_path: str) -> bool:
        """VAD-Filter: Fuehrende und nachfolgende Stille abschneiden.

        Nutzt webrtcvad (aggressiveness=2, 30ms Frames).
        Ueberschreibt die WAV-Datei mit dem getrimmten Audio.
        Returns True wenn Sprache gefunden, False wenn nur Stille.
        """
        try:
            import webrtcvad
            import wave as _wave
            import struct as _struct

            # WAV lesen (muss 16kHz Mono 16-bit sein)
            with _wave.open(wav_path, 'rb') as wf:
                rate = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                raw = wf.readframes(wf.getnframes())

            if rate != 16000 or channels != 1 or sample_width != 2:
                logger.warning(f"[VAD] Unerwartetes Format: {rate}Hz {channels}ch {sample_width}B — ueberspringe VAD")
                return True

            vad = webrtcvad.Vad(0)  # Aggressiveness 0 — tolerantester Modus fuer WiFi-Mic (niedriger Pegel)
            frame_ms = 30
            frame_bytes = int(rate * frame_ms / 1000) * sample_width  # 960 Bytes

            # In Frames aufteilen und VAD pruefen
            # Silence-Toleranz: mind. 800ms Stille zwischen Woertern ignorieren
            voiced_frames = []
            first_voice = -1
            last_voice = -1
            silence_frames_after_voice = 0
            max_silence_frames = int(800 / frame_ms)  # 800ms / 30ms = ~26 Frames
            for i in range(0, len(raw) - frame_bytes + 1, frame_bytes):
                frame = raw[i:i + frame_bytes]
                is_speech = vad.is_speech(frame, rate)
                voiced_frames.append((i, is_speech))
                if is_speech:
                    if first_voice == -1:
                        first_voice = i
                    last_voice = i + frame_bytes
                    silence_frames_after_voice = 0
                elif first_voice != -1:
                    silence_frames_after_voice += 1

            if first_voice == -1:
                logger.info("[VAD] Keine Sprache erkannt — nur Stille")
                return False

            # 13 Frames (~400ms) Padding vor/nach Sprache behalten
            # (grosszuegiger, damit Wortanfaenge/-enden erhalten bleiben)
            pad = frame_bytes * 13
            trim_start = max(0, first_voice - pad)
            trim_end = min(len(raw), last_voice + pad)
            trimmed = raw[trim_start:trim_end]

            orig_ms = len(raw) / (rate * sample_width) * 1000
            trim_ms = len(trimmed) / (rate * sample_width) * 1000

            # Minimum 2 Sekunden fuer Whisper — zu kurze Chunks ergeben Fragmente
            min_bytes = rate * sample_width * 2  # 2s = 64000 Bytes bei 16kHz
            if len(trimmed) < min_bytes:
                logger.info(f"[VAD] Chunk zu kurz ({trim_ms:.0f}ms < 2000ms), "
                            f"behalte Original ({orig_ms:.0f}ms)")
                # Nicht trimmen — ganzes Audio an Whisper geben
                return True

            logger.info(f"[VAD] Getrimmt: {orig_ms:.0f}ms → {trim_ms:.0f}ms "
                        f"(Start +{trim_start / (rate * sample_width) * 1000:.0f}ms)")

            # Zurueckschreiben
            self._write_pcm_as_wav(trimmed, wav_path, rate=rate)
            return True

        except ImportError:
            logger.warning("[VAD] webrtcvad nicht installiert — ueberspringe VAD")
            return True
        except Exception as e:
            logger.warning(f"[VAD] Fehler: {e} — ueberspringe VAD")
            return True

    def _process_recording_inner(self):
        """Eigentliche Recording-Verarbeitung."""
        wav_path = os.path.join(TEMP_DIR, "moloch_ptt_recording.wav")

        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
            logger.warning("[VOICE] Aufnahme zu kurz oder nicht vorhanden")
            self._store_whisper_info("[Aufnahme zu kurz]")
            self._whisper_status = "Idle"
            return

        # 0. VAD-Filter: Stille vor/nach Sprache abschneiden
        if self._vad_enabled and not self._vad_trim_wav(wav_path):
            logger.info("[VOICE] VAD: Nur Stille erkannt, ueberspringe Whisper")
            self._store_whisper_info("[Nur Stille — keine Sprache erkannt]")
            self._whisper_status = "Idle"
            return

        # 1. Whisper STT
        self._whisper_status = "Transkribiere..."
        self._publish_event("whisper.processing")  # LED: THINKING
        t_whisper = time.time()
        text = self._transcribe(wav_path)
        whisper_duration_ms = (time.time() - t_whisper) * 1000

        if not text or not text.strip():
            logger.info("[VOICE] Keine Sprache erkannt")
            self._store_whisper_info(f"[Keine Sprache erkannt ({whisper_duration_ms:.0f}ms)]")
            self._whisper_status = "Idle"
            return

        text = _sanitize_text(text)
        text = _filter_hallucinations(text, wav_path)
        if not text or not text.strip():
            logger.info("[VOICE] Nach Halluzinations-Filter: leer — verwerfen")
            self._store_whisper_info("[Halluzination gefiltert]")
            self._whisper_status = "Idle"
            return
        logger.info(f"[VOICE] Transkription ({whisper_duration_ms:.0f}ms): {text}")
        self._emit_message("Du", text)

        # Whisper-Ergebnis fuer Popup speichern + Event Bus
        self._store_whisper_result(text, whisper_duration_ms)

        # Langzeitgedaechtnis: User-Nachricht SOFORT speichern
        try:
            get_memory().save_message("user", text, source="voice")
        except Exception as e:
            logger.error(f"[VOICE] Memory save_message(user) fehlgeschlagen: {e}")

        # 1.5 Lokale Keyword-Erkennung (KEIN API Call noetig)
        keyword_response = self._handle_keyword(text)
        if keyword_response:
            self._emit_message("MOLOCH", keyword_response)
            try:
                get_memory().save_message("moloch", keyword_response, source="voice")
            except Exception:
                pass
            if self._voice_enabled:
                self._speak(keyword_response)
            self._whisper_status = "Idle"
            return

        # 1.6 Direkte Spotify-Commands (OHNE Claude API — instant)
        spotify_response = self._handle_direct_spotify_command(text)
        if spotify_response:
            logger.info(f"[VOICE] Spotify-Direct: {spotify_response}")
            self._emit_message("MOLOCH", spotify_response)
            try:
                get_memory().save_message("moloch", spotify_response, source="voice")
            except Exception:
                pass
            if self._voice_enabled:
                self._speak(spotify_response)
            self._whisper_status = "Idle"
            return

        # STUFE 3: Claude API in SEPARATEM Thread (blockiert NICHT Stufe 1+2)
        self._whisper_status = "Idle"  # Stufe 1+2 fertig, Pipeline frei
        api_thread = threading.Thread(
            target=self._api_and_respond,
            args=(text, "voice"),
            daemon=True,
        )
        api_thread.start()

    # =========================================================================
    # Whisper STT
    # =========================================================================

    def _store_whisper_info(self, info_text: str):
        """Info/Fehler-Meldung als Whisper-Ergebnis speichern (fuer Popup-Log)."""
        with self._whisper_results_lock:
            self._whisper_result_counter += 1
            result = {
                "id": self._whisper_result_counter,
                "text": info_text,
                "duration_ms": 0,
                "model": self._whisper.backend if self._whisper else "on-demand",
                "ts": time.time(),
            }
            self._whisper_results.append(result)
            if len(self._whisper_results) > 10:
                self._whisper_results = self._whisper_results[-10:]

    def _store_whisper_result(self, text: str, duration_ms: float):
        """Whisper-Ergebnis speichern fuer Popup + Event Bus publishen."""
        with self._whisper_results_lock:
            self._whisper_result_counter += 1
            result = {
                "id": self._whisper_result_counter,
                "text": text,
                "duration_ms": round(duration_ms),
                "model": self._whisper.backend if self._whisper else "unknown",
                "ts": time.time(),
            }
            self._whisper_results.append(result)
            # Max 10 behalten
            if len(self._whisper_results) > 10:
                self._whisper_results = self._whisper_results[-10:]

        # Event Bus publishen (Priority 9 = Logging)
        try:
            from core.moloch_event_bus import get_event_bus
            get_event_bus().publish(
                event_type="whisper.result",
                payload=result,
                source="voice_pipeline",
                priority=9,
            )
        except Exception:
            pass  # Event Bus optional

    def _transcribe(self, wav_path: str) -> Optional[str]:
        """WAV-Datei mit MolochWhisper transkribieren (NPU-only)."""
        if not self._init_whisper():
            self._store_whisper_info("[Fehler: Whisper nicht verfuegbar]")
            return None

        try:
            # MolochWhisper.transcribe() handhabt NPU acquire/release intern
            # Vision pausiert automatisch, startet nach Release wieder
            text = self._whisper.transcribe(wav_path, language="de")
            logger.info(f"[VOICE] Whisper Backend: {self._whisper.backend}")
            return text.strip() if text and text.strip() else None
        except Exception as e:
            logger.error(f"[VOICE] Whisper Fehler: {e}")
            self._store_whisper_info(f"[Fehler: {e}]")
            return None

    # =========================================================================
    # Claude API
    # =========================================================================

    def _chat(self, user_text: str) -> Optional[str]:
        """Text an Claude API senden und Antwort holen.

        SDK retried 529 intern 2x. Wir machen KEINEN Extra-Retry mehr.
        Laeuft in separatem Thread (_api_and_respond), blockiert nicht die Pipeline.
        """
        if not self._claude_available or not self._claude_client:
            logger.warning("[VOICE] Claude API nicht verfuegbar")
            return None

        # Surrogates entfernen (Whisper/Bluetooth liefert manchmal kaputte Umlaute)
        user_text = _sanitize_text(user_text)

        # Konversation thread-safe aufbauen (K2 Audit-Fix)
        with self._lock:
            self._conversation.append({"role": "user", "content": user_text})
            if len(self._conversation) > 10:
                self._conversation = self._conversation[-10:]
            msgs = list(self._conversation)  # Kopie fuer API-Call

        # System-Prompt einmalig aufbauen (gilt fuer alle Versuche)
        system = _sanitize_text(self._system_prompt)
        try:
            memory_ctx = get_memory().get_memory_context()
            if memory_ctx:
                system = system + "\n\n--- LANGZEITGEDAECHTNIS ---\n" + _sanitize_text(memory_ctx)
        except Exception as e:
            logger.error(f"[VOICE] Memory-Kontext laden fehlgeschlagen: {e}")

        perception_ctx = _perception_to_text()
        if perception_ctx:
            system = system + "\n\n--- AKTUELLE WAHRNEHMUNG ---\n" + _sanitize_text(perception_ctx)

        # Hardware-Status (live Werte bei jedem Call)
        hw_status = _get_hardware_status()
        if hw_status:
            system = system + "\n" + _sanitize_text(hw_status)

        # Internet-Status
        try:
            from core.net.internet_bridge import get_internet_bridge
            bridge = get_internet_bridge()
            if bridge.online:
                system = system + f"\nINTERNET: ONLINE ({bridge.latency_ms}ms Latenz)"
            else:
                system = system + "\nINTERNET: OFFLINE (kein Internetzugang gerade)"
        except Exception:
            pass  # Internet-Bridge optional

        try:
            from core.personality.personality_engine import get_personality_engine
            pe = get_personality_engine()
            pe.update_from_integrator()
            zone_addon = pe.get_zone_system_prompt_addon()
            if zone_addon:
                system = system + _sanitize_text(zone_addon)
            vc = pe.voice_config
            self._current_voice = vc.voice_id
            self._length_scale = vc.speed
        except Exception as e:
            logger.debug(f"[VOICE] Personality-Zone nicht verfuegbar: {e}")

        # API Call — SDK retried 529 intern 2x, wir machen KEINEN Extra-Retry
        # Timeout 15s statt 30s: Fail fast, nicht ewig blockieren
        try:
            response = self._claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                system=system,
                messages=msgs,
                timeout=15.0,
            )
            text = response.content[0].text

            try:
                text = get_memory().extract_and_learn(text)
            except Exception as e:
                logger.error(f"[VOICE] extract_and_learn fehlgeschlagen: {e}")

            try:
                text = self._extract_spotify_commands(text)
            except Exception as e:
                logger.error(f"[VOICE] Spotify-Commands fehlgeschlagen: {e}")

            with self._lock:
                self._conversation.append({"role": "assistant", "content": text})
            return text

        except Exception as e:
            logger.error(f"[VOICE] Claude API Fehler: {e}")
            with self._lock:
                if self._conversation and self._conversation[-1].get("role") == "user":
                    self._conversation.pop()
            return None

    # =========================================================================
    # Direkte Spotify Voice Commands (OHNE Claude API)
    # =========================================================================

    def _get_current_zone_for_spotify(self) -> str:
        """Aktuelle Personality-Zone fuer Spotify Mood holen. Fallback: 'shadow'."""
        try:
            from core.core_integrator import get_core_integrator
            zone = get_core_integrator().get_personality_zone()
            if zone:
                return zone
        except Exception:
            pass
        return "shadow"

    def _handle_direct_spotify_command(self, text: str) -> Optional[str]:
        """
        Prueft ob der transkribierte Text ein direkter Spotify-Befehl ist.
        Wird VOR dem Claude API Call geprueft — spart Latenz.

        Returns: Antwort-String wenn Befehl erkannt, None wenn nicht.

        Keywords:
          stopp/stop/pause           -> pause()
          weiter/play/fortsetzen     -> play()
          naechster/nächster/skip    -> next_track()
          zurueck/vorheriger         -> previous_track()
          lauter                     -> volume +10
          leiser                     -> volume -10
          spiel/spiele [Artist/Song] -> search_and_play() oder play_artist()
          musik an/aus               -> play()/pause()
        """
        lower = text.lower().strip()

        # Zu kurz? Kein Command
        if len(lower) < 3:
            return None

        try:
            from core.spotify_controller import get_spotify
            sp = get_spotify()
        except Exception as e:
            logger.error(f"[SPOTIFY-DIRECT] Controller nicht verfuegbar: {e}")
            return None

        # --- Einfache Commands (exakte/teilweise Matches) ---

        # STOP / PAUSE
        if lower in ("stopp", "stop", "pause", "stopp musik", "stop musik",
                      "musik stopp", "musik stop", "musik pause", "pause musik",
                      "halt", "ruhe", "still", "musik aus"):
            if sp.pause():
                return "Musik pausiert."
            return "Spotify reagiert nicht — laeuft gerade was?"

        # WEITER / PLAY (ohne Argument)
        if lower in ("weiter", "play", "fortsetzen", "weiter spielen",
                      "musik an", "musik weiter", "weiterspielen"):
            if sp.play():
                return "Laeuft wieder."
            return "Konnte Playback nicht starten."

        # SKIP / NAECHSTER
        if lower in ("naechster", "nächster", "naechster song", "nächster song",
                      "naechstes lied", "nächstes lied", "skip", "next",
                      "ueberspring", "überspring", "weiter skip"):
            if sp.next_track():
                time.sleep(0.5)
                track = sp.get_current_track()
                if track:
                    return f"Jetzt: {track['artist']} — {track['track']}"
                return "Naechster Track."
            return "Skip fehlgeschlagen."

        # VORHERIGER / ZURUECK
        if lower in ("zurueck", "zurück", "vorheriger", "vorheriges lied",
                      "vorheriger song", "nochmal", "previous", "letzter song"):
            if sp.previous_track():
                time.sleep(0.5)
                track = sp.get_current_track()
                if track:
                    return f"Zurueck zu: {track['artist']} — {track['track']}"
                return "Vorheriger Track."
            return "Zurueck fehlgeschlagen."

        # LAUTER
        if lower in ("lauter", "laut", "mehr lautstaerke", "mehr lautstärke",
                      "volume up", "lauter machen"):
            # Aktuell kein Volume-Read in API, setze relativ
            sp.set_volume(70)
            return "Lauter."

        # LEISER
        if lower in ("leiser", "leise", "weniger lautstaerke", "weniger lautstärke",
                      "volume down", "leiser machen"):
            sp.set_volume(30)
            return "Leiser."

        # WAS LAEUFT? / WELCHER SONG?
        if any(kw in lower for kw in ("was läuft", "was laeuft", "welcher song",
                                       "welches lied", "was spielt", "was hör ich",
                                       "was hoer ich", "aktueller song",
                                       "aktuelles lied", "welcher track")):
            track = sp.get_current_track()
            if track:
                status = "spielt" if track["is_playing"] else "pausiert"
                return f"{track['artist']} — {track['track']} ({status})"
            return "Gerade laeuft nichts."

        # --- Generische Musik-Befehle (OHNE spezifischen Artist) ---
        # "spiel musik", "spiel mal musik", "spiel was", "mach musik an", etc.
        _GENERIC_MUSIC_PHRASES = (
            "musik", "mal musik", "mal was", "was", "etwas", "irgendwas",
            "mal etwas", "mal irgendwas", "mir was", "mir musik", "mir mal was",
            "mir mal musik", "doch was", "doch musik", "mal was gutes",
        )
        if lower in ("spiel musik", "spiele musik", "musik abspielen",
                      "spiel mal musik", "spiel mal was", "spiel was",
                      "mach musik an", "mache musik an", "musik spielen",
                      "spiel etwas", "spiel irgendwas", "leg musik auf",
                      "leg was auf", "leg mal was auf", "leg mal musik auf"):
            zone = self._get_current_zone_for_spotify()
            if sp.play_by_mood(zone):
                return f"Musik laeuft — {zone.capitalize()} Mood."
            # Fallback: irgendwas spielen
            if sp.play():
                return "Musik laeuft."
            return "Spotify reagiert nicht — spotifyd laeuft?"

        # --- PLAYLIST Commands ---
        # "spiel playlist X", "playlist X abspielen", "leg playlist X auf"
        playlist_match = re.match(
            r'^(?:spiel|spiele|play|leg)\s+(?:die\s+)?playlist\s+(.+?)(?:\s+ab(?:spielen)?)?$',
            lower
        )
        if not playlist_match:
            playlist_match = re.match(
                r'^playlist\s+(.+?)(?:\s+abspielen|\s+an)?$', lower
            )
        if playlist_match:
            pl_name = playlist_match.group(1).strip()
            # Fuellwoerter raus
            pl_name = re.sub(r'\b(?:mal|mir|doch|bitte)\b', '', pl_name).strip()
            if pl_name:
                logger.info(f"[SPOTIFY-DIRECT] Playlist: '{pl_name}'")
                if sp.play_playlist(pl_name):
                    return f"Playlist '{pl_name}' laeuft!"
                return f"Playlist '{pl_name}' nicht gefunden. Sag 'welche Playlists' fuer eine Liste."

        # --- PLAYLISTS AUFLISTEN ---
        if any(kw in lower for kw in ("welche playlists", "meine playlists",
                                       "zeig playlists", "playlist liste",
                                       "alle playlists", "was fuer playlists")):
            return sp.list_playlists()

        # --- "SPIEL ..." Commands (mit Argument) ---

        # Pattern: "spiel(e) [etwas]" / "mach [artist] an" / "musik von [artist]"
        spiel_match = re.match(
            r'^(?:spiel|spiele|play)\s+(.+)$', lower
        )
        if not spiel_match:
            spiel_match = re.match(
                r'^(?:mach|mache)\s+(.+?)\s+an$', lower
            )
        if not spiel_match:
            spiel_match = re.match(
                r'^musik\s+(?:von\s+)?(.+)$', lower
            )

        if spiel_match:
            query = spiel_match.group(1).strip()
            if not query:
                return None

            # Fuellwoerter entfernen: "mal", "mir", "doch", "bitte", "mal eben"
            query = re.sub(
                r'\b(?:mal|mir|doch|bitte|eben|mal eben|noch)\b', '', query
            ).strip()
            # Doppelte Leerzeichen bereinigen
            query = re.sub(r'\s+', ' ', query).strip()

            # Wenn nach Fuellwort-Strip nur generische Begriffe uebrig sind
            if not query or query in _GENERIC_MUSIC_PHRASES:
                zone = self._get_current_zone_for_spotify()
                if sp.play_by_mood(zone):
                    return f"Musik laeuft — {zone.capitalize()} Mood."
                if sp.play():
                    return "Musik laeuft."
                return "Spotify reagiert nicht."

            logger.info(f"[SPOTIFY-DIRECT] Suche: '{query}'")

            # Reihenfolge: Playlist → Artist → Track-Suche
            if sp.play_playlist(query):
                return f"Playlist '{query}' laeuft!"
            if sp.play_artist(query):
                return f"Spiele {query}."
            if sp.search_and_play(query):
                track = sp.get_current_track()
                if track:
                    return f"Spiele: {track['artist']} — {track['track']}"
                return f"Spiele etwas fuer '{query}'."
            return f"Nichts gefunden fuer '{query}'."

        # Kein Spotify-Command erkannt
        return None

    # =========================================================================
    # Spotify Tag Extraction
    # =========================================================================

    def _extract_spotify_commands(self, text: str) -> str:
        """
        SPOTIFY-Tags aus Claude-Antwort extrahieren und ausfuehren.
        Tags werden entfernt bevor der Text gesprochen wird.

        Format: [SPOTIFY:action] oder [SPOTIFY:action=value]
        """
        import re
        pattern = r'\[SPOTIFY:([^\]]+)\]'
        matches = re.findall(pattern, text)
        if not matches:
            return text

        # Tags entfernen
        clean_text = re.sub(pattern, '', text).strip()
        # Doppelte Leerzeichen bereinigen
        clean_text = re.sub(r'  +', ' ', clean_text)

        # Commands ausfuehren (in Background Thread, blockiert TTS nicht)
        import threading
        def _execute():
            try:
                from core.spotify_controller import get_spotify
                sp = get_spotify()
                for cmd_str in matches:
                    if '=' in cmd_str:
                        action, value = cmd_str.split('=', 1)
                    else:
                        action, value = cmd_str, None

                    action = action.strip().lower()
                    value = value.strip() if value else None

                    if action == 'play':
                        sp.play(uri=value)
                    elif action == 'pause':
                        sp.pause()
                    elif action == 'toggle':
                        sp.toggle()
                    elif action == 'skip':
                        sp.next_track()
                    elif action == 'previous':
                        sp.previous_track()
                    elif action == 'volume' and value:
                        sp.set_volume(int(value))
                    elif action == 'search' and value:
                        sp.search_and_play(value)
                    elif action == 'artist' and value:
                        sp.play_artist(value)
                    elif action == 'mood' and value:
                        sp.play_by_mood(value)
                    elif action == 'playlist' and value:
                        sp.play_playlist(value)
                    else:
                        logger.warning(f"[SPOTIFY] Unbekannter Command: {cmd_str}")

                    logger.info(f"[SPOTIFY] Ausgefuehrt: {cmd_str}")
            except Exception as e:
                logger.error(f"[SPOTIFY] Command-Ausfuehrung fehlgeschlagen: {e}")

        threading.Thread(target=_execute, daemon=True).start()
        return clean_text

    # =========================================================================
    # Lokale Keyword-Erkennung (VOR Claude API)
    # =========================================================================

    def _handle_keyword(self, text: str) -> Optional[str]:
        """Text gegen lokale Keywords pruefen.

        Laedt Keywords aus config/keywords.json.
        Bei Match: Aktion ausfuehren, Antwort zurueckgeben.
        Kein API Call noetig.

        Returns:
            Antwort-String wenn lokal behandelt, None wenn weiter an API.
        """
        try:
            from core.keyword_handler import get_keyword_handler
            response = get_keyword_handler().execute(text)
            if response:
                logger.info(f"[VOICE] Keyword-Match: '{text[:60]}' -> '{response}'")
            return response
        except Exception as e:
            logger.error(f"[VOICE] Keyword-Handler Fehler: {e}")
            return None

    # =========================================================================
    # Chat Message (Text statt PTT)
    # =========================================================================

    def process_text_message(self, text: str):
        """Text-Nachricht verarbeiten (ohne Recording)."""
        if not text or not text.strip():
            return

        thread = threading.Thread(
            target=self._process_text, args=(text,), daemon=True
        )
        thread.start()

    def _process_text(self, text: str):
        """Text -> Claude -> TTS in Background."""
        with self._lock:
            self._processing = True
        try:
            self._process_text_inner(text)
        finally:
            with self._lock:
                self._processing = False

    def _process_text_inner(self, text: str):
        """Eigentliche Text-Verarbeitung."""
        text = _sanitize_text(text)
        self._whisper_status = "Denke..."
        self._emit_message("Du", text)

        # Langzeitgedaechtnis: User-Text SOFORT speichern
        try:
            get_memory().save_message("user", text, source="text")
        except Exception as e:
            logger.error(f"[VOICE] Memory save_message(user/text) fehlgeschlagen: {e}")

        # Lokale Keyword-Erkennung (KEIN API Call noetig)
        keyword_response = self._handle_keyword(text)
        if keyword_response:
            self._emit_message("MOLOCH", keyword_response)
            try:
                get_memory().save_message("moloch", keyword_response, source="text")
            except Exception:
                pass
            if self._voice_enabled:
                self._speak(keyword_response)
            self._whisper_status = "Idle"
            return

        # Direkte Spotify-Commands (OHNE Claude API — instant)
        spotify_response = self._handle_direct_spotify_command(text)
        if spotify_response:
            logger.info(f"[VOICE] Spotify-Direct: {spotify_response}")
            self._emit_message("MOLOCH", spotify_response)
            try:
                get_memory().save_message("moloch", spotify_response, source="text")
            except Exception:
                pass
            if self._voice_enabled:
                self._speak(spotify_response)
            self._whisper_status = "Idle"
            return

        # STUFE 3: Claude API in SEPARATEM Thread (blockiert NICHT Stufe 1+2)
        self._whisper_status = "Idle"  # Stufe 1+2 fertig, Pipeline frei
        api_thread = threading.Thread(
            target=self._api_and_respond,
            args=(text, "text"),
            daemon=True,
        )
        api_thread.start()

    # =========================================================================
    # STUFE 3: API Thread (laeuft unabhaengig von Whisper/Keywords)
    # =========================================================================

    def _api_and_respond(self, text: str, source: str = "voice"):
        """Claude API in separatem Thread — blockiert NICHT die Voice Pipeline.

        Args:
            text: Transkribierter/eingegebener Text
            source: "voice" oder "text" (fuer Memory-Tracking)
        """
        # Schutz: Wenn bereits ein API-Call laeuft, nicht stapeln
        with self._api_lock:
            if self._api_in_flight:
                logger.warning("[VOICE] API-Call laeuft bereits, ueberspringe")
                self._emit_message("System",
                    "Vorherige Anfrage laeuft noch...")
                return
            self._api_in_flight = True

        try:
            self._whisper_status = "Denke..."

            # Internet-Suche pruefen (VOR Claude API)
            search_query = _detect_search_intent(text)
            if search_query:
                try:
                    from core.net.internet_bridge import get_internet_bridge
                    bridge = get_internet_bridge()
                    if bridge.online:
                        self._whisper_status = "Suche im Netz..."
                        self._emit_message("System", f"Suche: {search_query}")
                        results = bridge.search_web(search_query)
                        if results:
                            search_block = (
                                f"\n\n[SUCHERGEBNISSE fuer: {search_query}]\n"
                                f"(Nutze diese Infos. Nenne die Quelle z.B. 'Laut Wikipedia...')\n"
                            )
                            for i, r in enumerate(results, 1):
                                src = r.get("source", "Web")
                                search_block += f"{i}. [{src}] {r['title']}: {r['text']}\n"
                            search_block += "[Ende Suchergebnisse]"
                            text = text + search_block
                        else:
                            text = text + (
                                "\n\n[SUCHE: Keine Ergebnisse gefunden. "
                                "Antworte aus deinem Wissen und sag dass du nichts gefunden hast.]"
                            )
                    else:
                        text = text + (
                            "\n\n[SUCHE: Aktuell offline. Sag Markus dass du offline bist "
                            "und antworte trotzdem aus deinem Wissen.]"
                        )
                    self._whisper_status = "Denke..."
                except Exception as e:
                    logger.error(f"[VOICE] Internet-Suche fehlgeschlagen: {e}")

            response = self._chat(text)

            if not response:
                self._emit_message("System",
                    "API gerade nicht erreichbar. Lokale Befehle funktionieren.")
                self._whisper_status = "Idle"
                return

            logger.info(f"[VOICE] Antwort: {response[:100]}...")
            self._emit_message("MOLOCH", response)

            # Langzeitgedaechtnis
            try:
                get_memory().save_message("moloch", response, source=source)
            except Exception as e:
                logger.error(f"[VOICE] Memory save_message(moloch/{source}) fehlgeschlagen: {e}")

            # TTS
            if self._voice_enabled:
                self._whisper_status = "Spreche..."
                self._speak(response)

            self._whisper_status = "Idle"
        finally:
            with self._api_lock:
                self._api_in_flight = False

    # =========================================================================
    # Piper TTS
    # =========================================================================

    def _speak(self, text: str):
        """Text mit Piper TTS sprechen und ueber HDMI ausgeben.

        Buffer-before-Play: Komplett generieren, dann in einem Stueck abspielen.
        Kein Stottern zwischen Saetzen, ein einziger pw-play-Aufruf via PipeWire.
        """
        if not self._piper_available:
            logger.warning("[VOICE] Piper nicht verfuegbar")
            return

        model_path = MODELS_DIR / f"{self._current_voice}.onnx"
        if not model_path.exists():
            logger.warning(f"[VOICE] Stimme nicht gefunden: {model_path}")
            return

        # Sample-Rate aus Model-Config (Piper default: 22050)
        sample_rate = "22050"
        model_config = model_path.with_suffix(".onnx.json")
        if model_config.exists():
            try:
                with open(model_config) as f:
                    cfg = json.load(f)
                    sample_rate = str(cfg.get("audio", {}).get("sample_rate", 22050))
            except Exception:
                pass

        pw_cmd = [
            "pw-cat", "-p", "--raw", "--rate", sample_rate,
            "--channels", "1", "--format", "s16", "-",
        ]

        # Text in Saetze aufteilen
        sentences = self._split_sentences(text)
        if not sentences:
            return

        with self._lock:
            self._speaking = True
        t0 = time.time()

        try:
            # Phase 1: Alle Saetze komplett generieren
            audio_chunks = []
            for i, sentence in enumerate(sentences):
                chunk = self._piper_synthesize(sentence, model_path)
                if chunk:
                    audio_chunks.append(chunk)

            if not audio_chunks:
                logger.warning("[VOICE] Keine Audio-Daten generiert")
                return

            # Phase 2: Alles zusammenfuegen und in einem Stueck abspielen
            full_audio = b"".join(audio_chunks)
            gen_time = time.time() - t0
            logger.info(f"[VOICE] Generiert: {len(text)} Zeichen, "
                        f"{len(sentences)} Saetze, {len(full_audio)//1024}KB "
                        f"in {gen_time:.1f}s — starte Playback")

            # LED: SPEAKING (mit Amplitude-Info fuer Pulsieren)
            amplitude = min(1.0, len(full_audio) / 50000.0)  # Grobe Amplitude-Schaetzung
            self._publish_event("audio.speaking_start", {"amplitude": amplitude, "text_len": len(text)})

            subprocess.run(pw_cmd, input=full_audio, timeout=120)

            dt = time.time() - t0
            logger.info(f"[VOICE] Gesprochen: {len(text)} Zeichen ({dt:.1f}s total)")

        except subprocess.TimeoutExpired:
            logger.error("[VOICE] TTS Timeout")
        except Exception as e:
            logger.error(f"[VOICE] TTS Fehler: {e}")
        finally:
            with self._lock:
                self._speaking = False
            # LED: zurueck zu IDLE
            self._publish_event("audio.speaking_end")

    def _piper_synthesize(self, text: str, model_path) -> bytes:
        """Einzelnen Text-Chunk mit Piper in raw PCM generieren (in RAM)."""
        try:
            result = subprocess.run(
                [
                    str(PIPER_PATH),
                    "--model", str(model_path),
                    "--length-scale", str(self._length_scale),
                    "--output-raw",
                ],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=45,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout
        except subprocess.TimeoutExpired:
            logger.error(f"[VOICE] Piper Timeout fuer: {text[:50]}...")
        except Exception as e:
            logger.error(f"[VOICE] Piper Fehler: {e}")
        return None

    def _split_sentences(self, text: str) -> list:
        """Text in sprechbare Chunks aufteilen fuer schnellere TTS.
        Kurze Saetze werden zusammengefasst (min ~40 Zeichen pro Chunk)."""
        chunks = []
        current = []
        for word in text.split():
            current.append(word)
            joined = " ".join(current)
            # Satzende erkannt und genug Text fuer eigenen Chunk
            if (word[-1:] in '.!?') and len(joined) >= 40:
                chunks.append(joined)
                current = []
        # Rest anfuegen
        if current:
            rest = " ".join(current)
            if chunks and len(rest) < 40:
                chunks[-1] += " " + rest
            else:
                chunks.append(rest)
        return chunks if chunks else [text.strip()] if text.strip() else []

    # =========================================================================
    # Konfiguration
    # =========================================================================

    def set_voice(self, voice_id: str) -> bool:
        """Stimme wechseln."""
        model_path = MODELS_DIR / f"{voice_id}.onnx"
        if model_path.exists():
            self._current_voice = voice_id
            logger.info(f"[VOICE] Stimme gewechselt: {voice_id}")
            return True
        logger.warning(f"[VOICE] Stimme nicht gefunden: {voice_id}")
        return False

    def list_voices(self) -> List[str]:
        """Verfuegbare Stimmen auflisten."""
        if not MODELS_DIR.exists():
            return []
        return sorted(f.stem for f in MODELS_DIR.glob("*.onnx"))

    def toggle_voice(self, enabled: Optional[bool] = None) -> bool:
        """Voice Output an/aus."""
        if enabled is not None:
            self._voice_enabled = enabled
        else:
            self._voice_enabled = not self._voice_enabled
        logger.info(f"[VOICE] Output: {'AN' if self._voice_enabled else 'AUS'}")
        return self._voice_enabled

    def get_state(self) -> Dict:
        """Aktuellen Status zurueckgeben (fuer IPC Status-JSON).
        Messages aelter als 120s werden automatisch entfernt."""
        with self._msg_lock:
            # Alte Messages entfernen (aelter als 120 Sekunden)
            cutoff = time.time() - 120.0
            self._pending_messages = [
                m for m in self._pending_messages if m.get("ts", 0) > cutoff
            ]
            messages = list(self._pending_messages)
        # Audio-Source Info
        audio_source = "usb"
        if self._wifi_mic and self._wifi_mic.connected:
            audio_source = "wifi"
        elif self._wifi_mic is None:
            audio_source = "usb"

        # WiFi-Mic Detail-Status fuer Panel (Buffer, Pakete, RMS)
        wifi_mic_status = {}
        if self._wifi_mic:
            try:
                wifi_mic_status = self._wifi_mic.get_status()
                wifi_mic_status["rms_db"] = round(
                    self._wifi_mic.peek_rms(num_samples=160), 1)
            except Exception:
                pass

        # Whisper-Ergebnisse fuer Popup
        with self._whisper_results_lock:
            whisper_results = list(self._whisper_results)

        return {
            "whisper_status": self._whisper_status,
            "whisper_backend": self._whisper.backend if self._whisper else "nicht geladen",
            "voice_enabled": self._voice_enabled,
            "current_voice": self._current_voice,
            "recording": self._recording,
            "speaking": self._speaking,
            "claude_available": self._claude_available,
            "piper_available": self._piper_available,
            "voices": self.list_voices(),
            "messages": messages,
            "audio_source": audio_source,
            "vad_enabled": self._vad_enabled,
            "whisper_results": whisper_results,
            "wifi_mic": wifi_mic_status,
        }

    def reset_conversation(self):
        """Konversation zuruecksetzen."""
        with self._lock:
            self._conversation.clear()
        logger.info("[VOICE] Konversation zurueckgesetzt")

    # =========================================================================
    # Spontane Kommentare (CoreIntegrator-gesteuert)
    # =========================================================================

    def start_spontaneous_monitor(self):
        """Spontane-Kommentare-Monitor starten (separater Thread, prueft alle 30s)."""
        if self._spontaneous_running:
            return
        self._spontaneous_running = True
        self._spontaneous_thread = threading.Thread(
            target=self._spontaneous_loop, daemon=True, name="SpontaneousComments"
        )
        self._spontaneous_thread.start()
        logger.info("[VOICE] Spontane-Kommentare-Monitor gestartet")

    def stop_spontaneous_monitor(self):
        """Monitor stoppen."""
        self._spontaneous_running = False

    def _spontaneous_loop(self):
        """Prueft alle 30s ob Moloch spontan kommentieren soll."""
        while self._spontaneous_running:
            try:
                self._check_spontaneous()
            except Exception as e:
                logger.error(f"[SPONTAN] Fehler: {e}")
            time.sleep(30)

    def _check_spontaneous(self):
        """Pruefe ob Bedingungen fuer spontanen Kommentar erfuellt sind."""
        # Cooldown pruefen
        if time.time() - self._last_spontaneous < self._spontaneous_cooldown:
            return

        # Nicht waehrend Recording/Speaking/Processing
        with self._lock:
            busy = self._recording or self._speaking or self._processing
        if busy:
            return

        # CoreIntegrator pruefen
        try:
            from core.core_integrator import get_core_integrator
            ci = get_core_integrator()
            effects = ci.get_effects()
            state = ci.get_state()
        except Exception:
            return

        spontaneous = effects.get("spontaneous_comments", 0.0)
        tension = state.get("tension", 0.0)

        # Schwellwerte: spontaneous > 0.7 UND tension > 0.1 (System aktiv)
        if spontaneous < 0.7 or tension < 0.1:
            return

        # Nur wenn Markus erkannt (aus Face State pruefen)
        markus_visible = False
        try:
            face_state_path = "/tmp/moloch_face_state.json"
            if os.path.exists(face_state_path):
                with open(face_state_path, "r") as f:
                    fs = json.load(f)
                # Markus muss in den letzten 30s erkannt worden sein
                if fs.get("name", "").lower() == "markus" and time.time() - fs.get("timestamp", 0) < 30:
                    markus_visible = True
        except Exception:
            pass

        if not markus_visible:
            return

        # Nachtsperre: 22:00 - 06:00 keine spontanen Kommentare
        from datetime import datetime
        hour = datetime.now().hour
        if hour >= 22 or hour < 6:
            return

        # Claude API fuer spontanen Kommentar nutzen
        logger.info(f"[SPONTAN] Bedingungen erfuellt: spontaneous={spontaneous:.2f} tension={tension:.2f}")
        self._generate_spontaneous_comment(state)

    def _generate_spontaneous_comment(self, integrator_state: dict):
        """Spontanen Kommentar via Claude API generieren und sprechen."""
        if not self._claude_available or not self._claude_client:
            return

        self._last_spontaneous = time.time()

        # System-Prompt fuer spontanen Kommentar
        system = """Du bist M.O.L.O.C.H. und machst einen kurzen, spontanen Kommentar.
Du siehst Markus gerade ueber deine Kamera.
Sage etwas Kurzes, Relevantes. MAX 1 Satz. Beispiele:
- "Du bist schon wieder lange am Rechner."
- "Interessantes Tracking heute."
- "Alles ruhig hier. Mir ist fast langweilig."
- "Na, Feierabend oder noch ne Runde?"
Sei natuerlich. Kein erzwungener Humor. Situationsbezogen."""

        # Memory-Kontext fuer Relevanz
        try:
            memory_ctx = get_memory().get_memory_context()
            if memory_ctx:
                system += "\n\n--- KONTEXT ---\n" + memory_ctx
        except Exception:
            pass

        # Wahrnehmungs-Kontext
        perception_ctx = _perception_to_text()
        if perception_ctx:
            system += "\n\n--- AKTUELLE WAHRNEHMUNG ---\n" + perception_ctx

        # Hardware-Status (live)
        hw_status = _get_hardware_status()
        if hw_status:
            system += "\n" + hw_status

        # Personality Zone
        try:
            from core.personality.personality_engine import get_personality_engine
            pe = get_personality_engine()
            zone_addon = pe.get_zone_system_prompt_addon()
            if zone_addon:
                system += zone_addon
        except Exception:
            pass

        try:
            response = self._claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=80,
                system=system,
                messages=[{"role": "user", "content": "Spontaner Kommentar jetzt."}],
            )
            text = response.content[0].text.strip()
            if text:
                logger.info(f"[SPONTAN] Kommentar: {text}")
                self._emit_message("MOLOCH", f"[spontan] {text}")
                # Speichern
                try:
                    get_memory().save_message("moloch", text, source="spontaneous")
                except Exception:
                    pass
                # Sprechen
                if self._voice_enabled:
                    self._speak(text)
        except Exception as e:
            logger.error(f"[SPONTAN] Claude API Fehler: {e}")

    def test_voice(self, text: str = "Moloch ist online. Sprach-Pipeline funktioniert."):
        """Voice Test — spricht Text direkt aus."""
        thread = threading.Thread(
            target=self._speak, args=(text,), daemon=True
        )
        thread.start()
