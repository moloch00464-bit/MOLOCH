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

from core.longterm_memory import get_memory

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

        # Langzeitgedaechtnis: User-Nachricht SOFORT speichern
        try:
            get_memory().save_message("user", text, source="voice")
        except Exception as e:
            logger.error(f"[VOICE] Memory save_message(user) fehlgeschlagen: {e}")

        # 2. Claude API
        self._whisper_status = "Denke..."
        response = self._chat(text)

        if not response:
            self._whisper_status = "Idle"
            return

        logger.info(f"[VOICE] Antwort: {response[:100]}...")
        self._emit_message("MOLOCH", response)

        # Langzeitgedaechtnis: Moloch-Antwort SOFORT speichern
        try:
            get_memory().save_message("moloch", response, source="voice")
        except Exception as e:
            logger.error(f"[VOICE] Memory save_message(moloch) fehlgeschlagen: {e}")

        # 3. TTS
        if self._voice_enabled:
            self._whisper_status = "Spreche..."
            self._speak(response)

        self._whisper_status = "Idle"

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
            # System-Prompt mit Memory-Kontext anreichern
            system = self._system_prompt
            try:
                memory_ctx = get_memory().get_memory_context()
                if memory_ctx:
                    system = system + "\n\n--- LANGZEITGEDAECHTNIS ---\n" + memory_ctx
            except Exception as e:
                logger.error(f"[VOICE] Memory-Kontext laden fehlgeschlagen: {e}")

            response = self._claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                system=system,
                messages=self._conversation,
            )
            text = response.content[0].text

            # REMEMBER-Tags extrahieren und lernen (persistiert in beide Systeme)
            try:
                text = get_memory().extract_and_learn(text)
            except Exception as e:
                logger.error(f"[VOICE] extract_and_learn fehlgeschlagen: {e}")

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
        self._emit_message("Du", text)

        # Langzeitgedaechtnis: User-Text SOFORT speichern
        try:
            get_memory().save_message("user", text, source="text")
        except Exception as e:
            logger.error(f"[VOICE] Memory save_message(user/text) fehlgeschlagen: {e}")

        response = self._chat(text)
        if not response:
            self._whisper_status = "Idle"
            return

        logger.info(f"[VOICE] Antwort: {response[:100]}...")
        self._emit_message("MOLOCH", response)

        # Langzeitgedaechtnis: Moloch-Antwort SOFORT speichern
        try:
            get_memory().save_message("moloch", response, source="text")
        except Exception as e:
            logger.error(f"[VOICE] Memory save_message(moloch/text) fehlgeschlagen: {e}")

        if self._voice_enabled:
            self._whisper_status = "Spreche..."
            self._speak(response)

        self._whisper_status = "Idle"

    # =========================================================================
    # Piper TTS
    # =========================================================================

    def _speak(self, text: str):
        """Text mit Piper TTS sprechen und ueber HDMI ausgeben.

        Satzweise Synthese mit Vorgeneration:
        1. Text in Saetze splitten
        2. Ersten Satz generieren → sofort abspielen
        3. Naechsten Satz im Hintergrund vorgenerieren waehrend aktueller spielt
        Ergebnis: Erster Ton nach ~8-10s statt ~26s.
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

        aplay_cmd = [
            "aplay", "-D", SPEAKER_DEVICE,
            "-r", sample_rate, "-f", "S16_LE", "-c", "1", "-q",
        ]

        # Text in Saetze aufteilen
        sentences = self._split_sentences(text)
        if not sentences:
            return

        self._speaking = True
        t0 = time.time()

        try:
            pre_generated = None

            for i, sentence in enumerate(sentences):
                # Audio holen: vorgeneriert oder jetzt generieren
                if pre_generated is not None:
                    audio = pre_generated
                    pre_generated = None
                else:
                    audio = self._piper_synthesize(sentence, model_path)

                if not audio:
                    continue

                if i == 0:
                    logger.info(f"[VOICE] Erster Satz nach {time.time()-t0:.1f}s "
                                f"({len(sentence)} Zeichen)")

                # Naechsten Satz im Hintergrund vorgenerieren
                gen_result = [None]
                gen_thread = None
                if i + 1 < len(sentences):
                    next_sentence = sentences[i + 1]
                    def _generate(s, result, mp=model_path):
                        result[0] = self._piper_synthesize(s, mp)
                    gen_thread = threading.Thread(
                        target=_generate, args=(next_sentence, gen_result),
                        daemon=True,
                    )
                    gen_thread.start()

                # Aktuellen Satz fluessig abspielen
                subprocess.run(aplay_cmd, input=audio, timeout=30)

                # Auf Vorgeneration warten
                if gen_thread:
                    gen_thread.join(timeout=45)
                    pre_generated = gen_result[0]

            dt = time.time() - t0
            logger.info(f"[VOICE] Gesprochen: {len(text)} Zeichen, "
                        f"{len(sentences)} Saetze ({dt:.1f}s)")

        except subprocess.TimeoutExpired:
            logger.error("[VOICE] TTS Timeout")
        except Exception as e:
            logger.error(f"[VOICE] TTS Fehler: {e}")
        finally:
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
        self._conversation.clear()
        logger.info("[VOICE] Konversation zurueckgesetzt")

    def test_voice(self, text: str = "Moloch ist online. Sprach-Pipeline funktioniert."):
        """Voice Test — spricht Text direkt aus."""
        thread = threading.Thread(
            target=self._speak, args=(text,), daemon=True
        )
        thread.start()
