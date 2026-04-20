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
from typing import Optional, Dict, Callable

logger = logging.getLogger("LocalLLMBridge")

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


def _build_local_context_snippet() -> str:
    """Live-Kontext aus moloch_status.json bauen (Vision + Inner State).

    Gibt eine Einzeiler-Ergaenzung zum Compact-Prompt zurueck, damit Qwen2.5
    wissen kann wen er sieht und wie er sich fuehlt. Leerer String bei Fehler.
    """
    try:
        with open(_STATUS_JSON_PATH, 'r') as f:
            st = json.load(f)
        person_detected = bool(st.get('person_detected'))
        face = st.get('face_id') or ('unbekannte Person' if person_detected else 'niemand')
        core = st.get('core', {}) or {}
        zone = core.get('zone', 'guardian')
        tension = core.get('tension', st.get('tension', 0.0))
        dominance = core.get('dominance', 0.0)
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
            f"JETZT: {_presence_phrase()}."
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
        parts.append(
            f"Zone {zone}, Stimmung {_tension_word(tension)}, Haltung {_dom_word(dominance)}."
        )
        body_parts = []
        if fps:
            body_parts.append(f"Bild {int(fps)} FPS")
        if cpu_temp is not None:
            body_parts.append(f"CPU {_cpu_word(cpu_temp)}")
        body_parts.append(f"NPU {_npu_word(npu_load)}")
        if body_parts:
            parts.append("Koerper: " + ", ".join(body_parts) + ".")
        if time_period:
            parts.append(f"Tageszeit: {time_period}.")

        # History-Block: letzte 5 Chat-Turns aus persistentem Memory.
        # Loest Multi-Turn-Drift + Cross-Channel-Sync Browser/Voice.
        try:
            from core.longterm_memory import get_memory
            msgs = get_memory().get_recent_messages(n=5) or []
            if msgs:
                hist_parts = []
                for m in msgs[-5:]:
                    sender = m.get("sender", "?")
                    text = (m.get("text") or "")[:100]
                    hist_parts.append(f"{sender}: {text}")
                parts.append("VORHER: " + " | ".join(hist_parts))
        except Exception:
            pass  # Memory-Singleton evtl. nicht init in standalone Test

        return " " + " ".join(parts)
    except Exception:
        return ""

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
                     use_reason_model: bool = False) -> Optional[str]:
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
        chosen = self._choose_provider(prompt, system, force_local, caller="ask")
        logger.debug(f"[LLM-BRIDGE] mode={mode} chosen={chosen}")

        if chosen == "tentacle":
            # 1a. Tentakel zuerst (komplexer Prompt)
            result = self._generate_tentacle(prompt, system, max_tokens,
                                             temperature=temperature, top_p=top_p)
            if result:
                return result
            # 1b. Fallback auf NPU wenn Tentakel still
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
                                                 temperature=temperature, top_p=top_p)
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

    def _choose_provider(self, prompt: str, system: str,
                         force_local: bool, caller: str = "ask") -> str:
        """Waehlt 'tentacle' oder 'ollama' basierend auf Komplexitaet + Caller.

        force_local -> immer 'ollama' (NPU). Tentakel ist LAN, nicht streng 'lokal'.
        caller='reason' -> Tentakel bevorzugt (internes Denken braucht Substanz).
        sonst: prompt+system >= complexity_threshold -> 'tentacle', sonst 'ollama'.
        """
        if force_local:
            return "ollama"
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
            # Fallback (kein Profil-File): bisheriger Compact-Override + Live-Kontext.
            profile = _get_active_profile()
            if profile is not None:
                profile_system = profile.get("system", OLLAMA_LOCAL_SYSTEM_COMPACT)
                if profile.get("include_live_context", False):
                    profile_system = profile_system + _build_local_context_snippet()
                system = profile_system
                # Profile-Sampling ueberschreibt Caller-Defaults
                pmt = profile.get("max_tokens")
                if isinstance(pmt, int) and pmt > 0:
                    max_tokens = pmt
                ptemp = profile.get("temperature")
                if isinstance(ptemp, (int, float)):
                    temperature = float(ptemp)
                logger.info(f"[LLM] Profil aktiv: {profile.get('system','')[:30]}... ({len(system)} Zeichen, max_tokens={max_tokens}, temp={temperature})")
            elif system and len(system) > OLLAMA_LOCAL_SYSTEM_MAX:
                # Fallback wenn keine Profile-Datei: alter Compact-Pfad
                ctx = _build_local_context_snippet()
                system = OLLAMA_LOCAL_SYSTEM_COMPACT + ctx
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
            logger.info(
                f"[LLM-BRIDGE] {model}: {len(text)} Zeichen in "
                f"{data.get('total_duration', 0) // 1_000_000}ms"
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
                           top_p: float = 0.95) -> Optional[str]:
        """Ollama-Tentakel auf LAN-Rechner (Standard-Ollama-API /api/chat)."""
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
        model = self._discover_tentacle_model(cfg)
        if not model:
            return None
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
            if profile.get("include_live_context", True):
                profile_system = profile_system + _build_local_context_snippet()
            system = profile_system
            pmt = profile.get("max_tokens")
            if isinstance(pmt, int) and pmt > 0:
                max_tokens = pmt
            ptemp = profile.get("temperature")
            if isinstance(ptemp, (int, float)):
                temperature = float(ptemp)
        elif not system:
            # Letzter Fallback wenn weder Profile noch User-System gegeben
            system = TENTACLE_SYSTEM_COMPACT + _build_local_context_snippet()

        # JSON-sicher machen (wie bei hailo-ollama — Standard-Ollama hat
        # zwar robusteren Parser, aber Konsistenz zahlt sich aus)
        def _flatten(s: str) -> str:
            return (s or "").replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')

        messages = []
        if system:
            messages.append({"role": "system", "content": _flatten(system)})
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
        """DeepSeek API (Cloud, guenstig)."""
        api_key = self._load_api_key("deepseek")
        if not api_key:
            return None
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
