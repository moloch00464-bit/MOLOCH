#!/usr/bin/env python3
"""
M.O.L.O.C.H. Voice Pipeline
=============================

PTT Recording -> Whisper STT -> Claude API -> Piper TTS -> HDMI Output

Audio Devices:
  Input:  ReSpeaker Lite (card 2, device 0) — plughw:2,0
  Output: HDMI (card 0, device 0) — plughw:0,0

Eigenstaendiges Modul, wird vom MolochService instanziiert.
Kommuniziert NICHT direkt mit GUI.
"""

import os
import sys
import json
import time
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Dict, Callable

logger = logging.getLogger("VoicePipeline")

# Audio Device Konfiguration
MIC_DEVICE = "plughw:2,0"      # ReSpeaker Lite
SPEAKER_DEVICE = "plughw:0,0"  # HDMI-0

# Pfade
MODELS_DIR = Path.home() / "moloch" / "models" / "voices"
PIPER_PATH = Path.home() / ".local" / "bin" / "piper"
API_KEYS_PATH = Path.home() / "moloch" / "config" / "api_keys.json"
TEMP_DIR = "/tmp"


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

        # Whisper STT (MolochWhisper: NPU primary, CPU fallback)
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

        # Message-Queue fuer IPC zum Panel (wird via get_state() abgeholt)
        self._pending_messages: List[Dict[str, str]] = []
        self._msg_lock = threading.Lock()

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
        """MolochWhisper lazy-laden (NPU primary, CPU fallback)."""
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
            self._pending_messages.append({"sender": sender, "text": text})
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

        logger.info(f"[VOICE] Transkription: {text}")
        self._emit_message("Du", text)

        # 2. Claude API
        self._whisper_status = "Denke..."
        response = self._chat(text)

        if not response:
            self._whisper_status = "Idle"
            return

        logger.info(f"[VOICE] Antwort: {response[:100]}...")
        self._emit_message("MOLOCH", response)

        # 3. TTS
        if self._voice_enabled:
            self._whisper_status = "Spreche..."
            self._speak(response)

        self._whisper_status = "Idle"

    # =========================================================================
    # Whisper STT
    # =========================================================================

    def _transcribe(self, wav_path: str) -> Optional[str]:
        """WAV-Datei mit MolochWhisper transkribieren (NPU + HailoManager)."""
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
        """Text an Claude API senden und Antwort holen."""
        if not self._claude_available or not self._claude_client:
            logger.warning("[VOICE] Claude API nicht verfuegbar")
            return None

        # Konversation aufbauen
        self._conversation.append({"role": "user", "content": user_text})

        # Maximal 10 Nachrichten behalten
        if len(self._conversation) > 10:
            self._conversation = self._conversation[-10:]

        try:
            response = self._claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                system=self._system_prompt,
                messages=self._conversation,
            )
            text = response.content[0].text
            self._conversation.append({"role": "assistant", "content": text})
            return text
        except Exception as e:
            logger.error(f"[VOICE] Claude API Fehler: {e}")
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
        self._whisper_status = "Denke..."

        response = self._chat(text)
        if not response:
            self._whisper_status = "Idle"
            return

        logger.info(f"[VOICE] Antwort: {response[:100]}...")
        self._emit_message("MOLOCH", response)

        if self._voice_enabled:
            self._whisper_status = "Spreche..."
            self._speak(response)

        self._whisper_status = "Idle"

    # =========================================================================
    # Piper TTS
    # =========================================================================

    def _speak(self, text: str):
        """Text mit Piper TTS sprechen und ueber HDMI ausgeben."""
        if not self._piper_available:
            logger.warning("[VOICE] Piper nicht verfuegbar")
            return

        model_path = MODELS_DIR / f"{self._current_voice}.onnx"
        if not model_path.exists():
            logger.warning(f"[VOICE] Stimme nicht gefunden: {model_path}")
            return

        self._speaking = True
        wav_path = os.path.join(TEMP_DIR, f"moloch_tts_{os.getpid()}.wav")

        try:
            # Piper TTS: Text -> WAV
            piper_result = subprocess.run(
                [
                    str(PIPER_PATH),
                    "--model", str(model_path),
                    "--length-scale", str(self._length_scale),
                    "--output_file", wav_path,
                ],
                input=text,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if piper_result.returncode != 0:
                logger.error(f"[VOICE] Piper Fehler: {piper_result.stderr}")
                return

            if not os.path.exists(wav_path):
                logger.error("[VOICE] Piper hat keine WAV erzeugt")
                return

            # Playback ueber HDMI
            subprocess.run(
                ["aplay", "-D", SPEAKER_DEVICE, wav_path],
                capture_output=True,
                timeout=30,
            )
            logger.info(f"[VOICE] Gesprochen: {len(text)} Zeichen mit {self._current_voice}")

        except subprocess.TimeoutExpired:
            logger.error("[VOICE] TTS Timeout")
        except Exception as e:
            logger.error(f"[VOICE] TTS Fehler: {e}")
        finally:
            self._speaking = False
            # Temp-Datei aufraeumen
            try:
                os.unlink(wav_path)
            except FileNotFoundError:
                pass

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
        Pending Messages werden mitgesendet und danach geleert."""
        # Messages atomar abholen und leeren
        with self._msg_lock:
            messages = list(self._pending_messages)
            self._pending_messages.clear()
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
        self._conversation.clear()
        logger.info("[VOICE] Konversation zurueckgesetzt")

    def test_voice(self, text: str = "Moloch ist online. Sprach-Pipeline funktioniert."):
        """Voice Test — spricht Text direkt aus."""
        thread = threading.Thread(
            target=self._speak, args=(text,), daemon=True
        )
        thread.start()
