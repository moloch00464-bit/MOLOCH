#!/usr/bin/env python3
"""
M.O.L.O.C.H. Voice Pipeline
=============================

PTT Recording -> Whisper STT -> Claude API -> Piper TTS -> HDMI Output

Audio Devices:
  Input:  ReSpeaker Lite (card 2, device 0) — plughw:2,0
  Output: HDMI via PipeWire (pw-play)

Eigenstaendiges Modul, wird vom MolochService instanziiert.
Kommuniziert NICHT direkt mit GUI.
"""

import os
import re
import sys
import json
import time
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Dict, Callable

from core.longterm_memory import get_memory

logger = logging.getLogger("VoicePipeline")

# Audio Device Konfiguration
MIC_DEVICE = "plughw:2,0"      # ReSpeaker Lite
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
Beispiel: "Klar, ich leg Suicide Commando auf! [SPOTIFY:artist=Suicide Commando]"
Benutze diese Tags wenn Markus nach Musik fragt oder Steuerung will.
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
        self._current_voice = "de_DE-thorsten-high"
        self._length_scale = 1.1  # Leicht langsamer fuer Verstaendlichkeit
        self._piper_available = PIPER_PATH.exists()

        # Message-Queue fuer IPC zum Panel (bleibt erhalten, Panel tracked per ID)
        self._pending_messages: List[Dict] = []
        self._msg_counter = 0
        self._msg_lock = threading.Lock()

        # API-Thread Schutz: Verhindert Queue von API-Calls wenn API down
        self._api_in_flight = False
        self._api_lock = threading.Lock()

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
    # PTT Recording
    # =========================================================================

    def start_recording(self):
        """Aufnahme vom ReSpeaker Lite starten."""
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._whisper_status = "Aufnahme..."

        wav_path = os.path.join(TEMP_DIR, "moloch_ptt_recording.wav")

        # Alte Datei loeschen
        try:
            os.unlink(wav_path)
        except FileNotFoundError:
            pass

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
            logger.info("[VOICE] Aufnahme gestartet")
        except Exception as e:
            logger.error(f"[VOICE] Aufnahme starten fehlgeschlagen: {e}")
            with self._lock:
                self._recording = False
                self._whisper_status = "Fehler"

    def stop_recording(self):
        """Aufnahme stoppen und Pipeline in Background-Thread weiterfuehren."""
        with self._lock:
            if not self._recording:
                return
            self._recording = False

        # arecord stoppen
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

    def _process_recording_inner(self):
        """Eigentliche Recording-Verarbeitung."""
        wav_path = os.path.join(TEMP_DIR, "moloch_ptt_recording.wav")

        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
            logger.warning("[VOICE] Aufnahme zu kurz oder nicht vorhanden")
            self._whisper_status = "Idle"
            return

        # 1. Whisper STT
        self._whisper_status = "Transkribiere..."
        text = self._transcribe(wav_path)

        if not text or not text.strip():
            logger.info("[VOICE] Keine Sprache erkannt")
            self._whisper_status = "Idle"
            return

        text = _sanitize_text(text)
        logger.info(f"[VOICE] Transkription: {text}")
        self._emit_message("Du", text)

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

    def _transcribe(self, wav_path: str) -> Optional[str]:
        """WAV-Datei mit MolochWhisper transkribieren (NPU-only)."""
        if not self._init_whisper():
            return None

        try:
            # MolochWhisper.transcribe() handhabt NPU acquire/release intern
            # Vision pausiert automatisch, startet nach Release wieder
            text = self._whisper.transcribe(wav_path, language="de")
            logger.info(f"[VOICE] Whisper Backend: {self._whisper.backend}")
            return text.strip() if text and text.strip() else None
        except Exception as e:
            logger.error(f"[VOICE] Whisper Fehler: {e}")
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

            # Erst als Artist versuchen, dann als allgemeine Suche
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
