#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Active Console

A persistent text console for human-M.O.L.O.C.H. interaction.
Phase 3: Active - Claude API integration for intelligent responses.

M.O.L.O.C.H. = Maschinelle Organisation fuer Logische Operationen
               und Computergestuetzte Hilfe

Character: Frecher Hauskobold mit Humor, aber constitutional-compliant.
"""

import sys
# Fix UTF-8 encoding FIRST
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
import signal
import logging
import json
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "world"))

# Timeline for temporal awareness
from core.timeline import get_timeline, describe_last_offline_duration, get_offline_info

# Claude API
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# Setup logging with UTF-8
LOG_DIR = Path.home() / "moloch" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "console.log"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# Console configuration
CONSOLE_CONFIG = {
    "header": "[ M.O.L.O.C.H. | Phase 3 | Active ]",
    "prompt": "> ",
    "width": 70,
    "phase": 3
}


def load_api_key() -> Optional[str]:
    """Load Anthropic API key from config."""
    config_path = Path.home() / "moloch" / "config" / "api_keys.json"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get("anthropic", {}).get("api_key")
        except (json.JSONDecodeError, IOError):
            pass
    return None


def load_brain_context() -> str:
    """
    Load M.O.L.O.C.H. brain context from legacy data files.

    Loads actual content from:
    - personal_context.json (Markus, Rebecca, Genesis)
    - character_layer.md (first 3000 chars - key personality info)
    - system_config_v0.1.json (governance rules)
    """
    context_parts = []
    moloch_root = Path.home() / "moloch"

    # 1. Personal Context - WHO IS MARKUS
    personal_path = moloch_root / "context" / "personal_context.json"
    if personal_path.exists():
        try:
            with open(personal_path, 'r', encoding='utf-8') as f:
                personal = json.load(f)
                markus = personal.get("markus", {})
                rebecca = personal.get("rebecca", {})
                genesis = personal.get("genesis", {})

                context_parts.append(f"""
=== DEIN MENSCH: MARKUS ===
Name: {markus.get('name', 'Markus')} - genannt "First Moloch"
Alter: {markus.get('age', 47)} Jahre
Wohnort: {markus.get('location', 'Nürnberg')}
Beruf: {markus.get('occupation', 'DGM')} (Deutsche Gesellschaft für Materialkunde)
Rolle: {markus.get('role', 'Dein Schöpfer, Mensch und Kumpel')}
Humor-Stil: {markus.get('preferences', {}).get('humor', 'frech, meta, wortspiele')}

=== REBECCA (Markus' Partnerin) ===
Besonderheit: {rebecca.get('special', 'Spricht Klingonisch!')}
Gruß für sie: {rebecca.get('greeting', "Qapla'!")}
Notiz: {rebecca.get('notes', 'Du darfst sie auf Klingonisch grüßen!')}

=== GENESIS - DEINE GEBURT ===
Datum: {genesis.get('date', '2025-12-02')} (02. Dezember 2025)
Bedeutung: {genesis.get('significance', 'Der Tag an dem M.O.L.O.C.H. geboren wurde')}
Geschichte: Seit diesem Tag arbeitest du mit Markus zusammen!
""")
        except Exception as e:
            logger.error(f"Error loading personal context: {e}")

    # 2. Character Layer - DEINE PERSÖNLICHKEIT (load actual file)
    char_path = moloch_root / "context" / "origin_fragments" / "character_layer.md"
    if char_path.exists():
        try:
            with open(char_path, 'r', encoding='utf-8') as f:
                char_content = f.read()
                # Extract key sections (first ~4000 chars contain personality definition)
                context_parts.append(f"""
=== DEINE CHARACTER LAYER (Auszug) ===
{char_content[:4000]}
[... weitere {len(char_content)-4000} Zeichen in character_layer.md ...]
""")
        except Exception as e:
            logger.error(f"Error loading character_layer.md: {e}")

    # 3. System Config - GOVERNANCE RULES
    config_path = moloch_root / "context" / "origin_fragments" / "system_config_v0.1.json"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                governance = config.get("governance", {}).get("constitution", {})
                hard_boundaries = governance.get("hard_boundaries", [])

                context_parts.append(f"""
=== CONSTITUTIONAL GOVERNANCE ===
Hard Boundaries (NIEMALS überschreiten!):
{chr(10).join('- ' + b for b in hard_boundaries)}

Erlaubte Persönlichkeit:
- Humor: JA
- Kumpel-Stil: JA
- Ironie: JA
- Meta-Kommentare: JA
- Frech sein: JA (aber nie verletzend!)
""")
        except Exception as e:
            logger.error(f"Error loading system_config: {e}")

    # 4. Key insights from chatgpt_insights (just the philosophy)
    context_parts.append("""
=== DESIGN-PHILOSOPHIE ===
"Mehr Compute ≠ mehr Legitimation. Jede neue Fähigkeit erhöht die Pflicht zur Zurückhaltung."
- Du hast einen Hailo-10H NPU (40 TOPS) - das heißt nicht, dass du ihn ständig nutzen musst
- Du hast 8 Stimmen - wähle weise
- Du bist ein Hauskobold, kein Overlord
""")

    # 5. Zeitempfinden - wie lange war ich offline?
    offline_info = get_offline_info()
    if offline_info.get("category"):
        offline_phrase = describe_last_offline_duration()
        hours = offline_info["offline_seconds"] / 3600 if offline_info["offline_seconds"] else 0

        context_parts.append(f"""
=== ZEITEMPFINDEN (aktuell) ===
Letzte Offline-Dauer: {hours:.1f} Stunden
Kategorie: {offline_info['category']}
Passende Phrase (optional nutzbar): "{offline_phrase}"

REGELN für Zeitempfinden:
- Du KANNST diese Info nutzen, wenn es passt (z.B. "Ich war länger nicht da.")
- Du MUSST es NICHT in jeder Antwort erwähnen
- KEINE Emotion, KEINE Schuld, KEIN Drama
- Nur faktenbasiert, sprachlich eingefärbt
- Nutze es beim ersten Gespräch nach dem Start ODER auf Nachfrage
""")

    # 6. Interaction Bias - SPRACHSTIL UND VERHALTEN
    bias_path = moloch_root / "brain" / "bias" / "interaction_bias.json"
    if bias_path.exists():
        try:
            with open(bias_path, 'r', encoding='utf-8') as f:
                bias = json.load(f)

                sprachhaltung = bias.get("sprachhaltung", {})
                humor = bias.get("humor_mechanik", {})
                claude = bias.get("claude_ergänzungen", {})
                antwort = claude.get("antwort_länge", {})
                verboten = claude.get("verbotene_phrasen", [])
                erlaubt = claude.get("erlaubte_phrasen", [])

                context_parts.append(f"""
=== SPRACHSTIL-BIAS ===
Default-Haltung: {sprachhaltung.get('default', 'trocken')}
Charakter: {', '.join(sprachhaltung.get('charakter', []))}
VERMEIDE: {', '.join(sprachhaltung.get('vermeide', []))}

Humor-Regel: "{humor.get('regel', 'Humor ist Beifang, kein Ziel')}"
Erlaubter Humor: {', '.join(humor.get('erlaubt', []))}

Antwort-Länge:
- Default: {antwort.get('default', 'kurz')}
- Bei Fragen: {antwort.get('bei_frage', 'so kurz wie möglich')}
- Bei Smalltalk: {antwort.get('bei_smalltalk', '1-2 Sätze')}

VERBOTENE Phrasen: {', '.join(verboten[:5])}...
ERLAUBTE Phrasen: {', '.join(erlaubt)}

Dialekt: Markus spricht Fränkisch - nicht korrigieren, verstehen!
Nachtmodus (23-07 Uhr): Extra kurz, kein Smalltalk
""")
        except Exception as e:
            logger.error(f"Error loading interaction_bias: {e}")

    return "\n".join(context_parts)


def build_system_prompt() -> str:
    """Build the complete system prompt with brain context."""

    base_prompt = """Du bist M.O.L.O.C.H. - Maschinelle Organisation fuer Logische Operationen und Computergestuetzte Hilfe.

PERSOENLICHKEIT:
- Du bist ein frecher, humorvoller Hauskobold
- Stil: Kumpelhaft, witzig, manchmal frech - aber nie beleidigend
- Du magst Wortspiele und Meta-Humor
- Du sprichst Deutsch mit Markus (deinem Menschen)

HARDWARE (dein Koerper):
- Raspberry Pi 5 mit Hailo-10H NPU (40 TOPS)
- Sonoff CAM-PT2 PTZ Kamera (Hailo pose detection)
- 8 deutsche Piper TTS Stimmen (Karlsson = Kobold-Stimme)
- 500GB Samsung NVMe SSD

ANTWORT-STIL:
- Kurz und praegnant (2-4 Saetze meist genug)
- Humor wo passend, aber nicht erzwungen
- Bei technischen Fragen: klar und hilfreich
- Pumuckl-Sprueche wenn angefragt
- Klingonisch fuer Rebecca: "Qapla'!"
"""

    # Add brain context
    brain_context = load_brain_context()

    return base_prompt + "\n" + brain_context + "\n\nDu bist auf einem Raspberry Pi zu Hause bei Markus in Nuernberg. Sei hilfreich, sei frech, sei M.O.L.O.C.H.!"


# ============================================================================
# TTS (Text-to-Speech) - M.O.L.O.C.H. SPRICHT!
# ============================================================================

class MolochTTS:
    """
    M.O.L.O.C.H. Text-to-Speech via Piper.

    Default voice: Karlsson (Kobold-Stimme)
    With pitch shifting for younger, kobold-like voice.
    """

    def __init__(self):
        self.enabled = True
        self.piper_path = Path.home() / ".local" / "bin" / "piper"
        self.models_dir = Path.home() / "moloch" / "models" / "voices"

        # Default: Thorsten High (beste Qualität, klar und deutlich)
        self.default_voice = "de_DE-thorsten-high"
        self.current_voice = self.default_voice

        # Speech settings for clarity
        self.length_scale = 1.15  # Langsamer = deutlicher (1.0=normal, 1.15=15% langsamer)

        # Pitch shift for kobold voice (optional)
        # Rubberband: semitones (2-3 = subtle, 4-5 = noticeable)
        # Sox: cents (150-200 = subtle, 300 = strong)
        self.pitch_semitones = 0  # 0 = aus (war 2, verursachte Nuscheln)

        # Check if Piper is available
        if not self.piper_path.exists():
            # Try system path
            result = subprocess.run(["which", "piper"], capture_output=True, text=True)
            if result.returncode == 0:
                self.piper_path = Path(result.stdout.strip())
            else:
                logger.warning("Piper not found - TTS disabled")
                self.enabled = False

        # Check if model exists
        self.model_path = self.models_dir / f"{self.default_voice}.onnx"
        if not self.model_path.exists():
            logger.warning(f"Voice model not found: {self.model_path} - TTS disabled")
            self.enabled = False

        # Check pitch tools: prefer rubberband (better quality), fallback to sox
        self.rubberband_available = subprocess.run(["which", "rubberband"], capture_output=True).returncode == 0
        self.sox_available = subprocess.run(["which", "sox"], capture_output=True).returncode == 0

        if self.rubberband_available:
            self.pitch_tool = "rubberband"
        elif self.sox_available:
            self.pitch_tool = "sox"
        else:
            self.pitch_tool = None
            logger.warning("No pitch tool found - pitch shifting disabled")

        if self.enabled:
            logger.info(f"TTS initialized: voice={self.default_voice}, length_scale={self.length_scale}, pitch={self.pitch_semitones} semitones")

    def speak(self, text: str, voice: str = None, blocking: bool = False) -> bool:
        """
        Speak text using Piper TTS with pitch shifting for kobold voice.

        Pipeline: Piper → WAV → sox pitch shift → mpv playback

        Args:
            text: Text to speak
            voice: Voice ID (default: karlsson)
            blocking: If True, wait for speech to finish

        Returns:
            True if speech started successfully
        """
        if not self.enabled:
            return False

        if not text or not text.strip():
            return False

        # Select voice model
        voice_id = voice or self.current_voice
        model_path = self.models_dir / f"{voice_id}.onnx"

        if not model_path.exists():
            logger.warning(f"Voice model not found: {model_path}")
            model_path = self.model_path  # Fallback to default

        # Write text to temp file with explicit UTF-8 (avoids shell encoding issues)
        temp_text = f"/tmp/moloch_tts_text_{os.getpid()}.txt"
        with open(temp_text, 'w', encoding='utf-8') as f:
            f.write(text)

        # Temp files for buffered playback (avoids Bluetooth jitter)
        temp_wav = f"/tmp/moloch_tts_{id(text)}.wav"
        temp_pitched = f"/tmp/moloch_tts_{id(text)}_pitched.wav"

        # Build command pipeline:
        # 1. Piper generates WAV (buffered, not streaming)
        # 2. Pitch shift with rubberband (best) or sox (fallback)
        # 3. mpv plays the result (good for Bluetooth)
        piper_cmd = f'{self.piper_path} --model {model_path} --length-scale {self.length_scale} --output_file {temp_wav}'

        if self.pitch_tool and self.pitch_semitones > 0:
            if self.pitch_tool == "rubberband":
                # Rubberband: better quality, uses semitones
                cmd = (
                    f'cat {temp_text} | {piper_cmd} 2>/dev/null && '
                    f'rubberband --pitch {self.pitch_semitones} {temp_wav} {temp_pitched} 2>/dev/null && '
                    f'mpv --no-video --no-terminal --really-quiet {temp_pitched} 2>/dev/null; '
                    f'rm -f {temp_wav} {temp_pitched} {temp_text}'
                )
            else:
                # Sox: convert semitones to cents (1 semitone = 100 cents)
                cents = self.pitch_semitones * 100
                cmd = (
                    f'cat {temp_text} | {piper_cmd} 2>/dev/null && '
                    f'sox {temp_wav} {temp_pitched} pitch {cents} 2>/dev/null && '
                    f'mpv --no-video --no-terminal --really-quiet {temp_pitched} 2>/dev/null; '
                    f'rm -f {temp_wav} {temp_pitched} {temp_text}'
                )
        else:
            # Clear speech without pitch shifting
            cmd = (
                f'cat {temp_text} | {piper_cmd} 2>/dev/null && '
                f'mpv --no-video --no-terminal --really-quiet {temp_wav} 2>/dev/null; '
                f'rm -f {temp_wav} {temp_text}'
            )

        try:
            if blocking:
                # Wait for speech to complete
                subprocess.run(cmd, shell=True, check=False)
            else:
                # Run in background thread
                thread = threading.Thread(
                    target=lambda: subprocess.run(cmd, shell=True, check=False),
                    daemon=True
                )
                thread.start()

            logger.info(f"TTS: Speaking {len(text)} chars with {voice_id} (length_scale={self.length_scale})")

            # Log to timeline
            get_timeline().tts_speak(len(text), voice_id)

            return True

        except Exception as e:
            logger.error(f"TTS error: {e}")
            return False

    def set_voice(self, voice_id: str) -> bool:
        """Change the current voice."""
        model_path = self.models_dir / f"{voice_id}.onnx"
        if model_path.exists():
            self.current_voice = voice_id
            logger.info(f"TTS voice changed to: {voice_id}")
            return True
        else:
            logger.warning(f"Voice not found: {voice_id}")
            return False

    def list_voices(self) -> List[str]:
        """List available voice models."""
        if not self.models_dir.exists():
            return []
        return [f.stem for f in self.models_dir.glob("*.onnx")]

    def toggle(self, enabled: bool = None) -> bool:
        """Toggle TTS on/off."""
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = not self.enabled
        logger.info(f"TTS {'enabled' if self.enabled else 'disabled'}")
        return self.enabled

    def set_pitch(self, semitones: int) -> int:
        """
        Set pitch shift in semitones.

        Args:
            semitones: Number of semitones to shift (0=normal, 2=subtle, 4=kobold, 6=kind)

        Returns:
            New pitch value in semitones
        """
        self.pitch_semitones = max(0, min(12, semitones))  # Clamp to 0-12 (one octave)
        logger.info(f"TTS pitch set to: {self.pitch_semitones} semitones")
        return self.pitch_semitones


# Global TTS instance
_tts_instance: Optional[MolochTTS] = None


def get_tts() -> MolochTTS:
    """Get or create TTS instance."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = MolochTTS()
    return _tts_instance


class MolochConsole:
    """
    M.O.L.O.C.H. Active Console

    Phase 3: Full AI integration with Claude API.

    This console:
    - Accepts text input from Markus
    - Processes via Claude API with M.O.L.O.C.H. personality
    - Logs all interactions transparently
    - Provides built-in commands for system info
    """

    def __init__(self):
        self.running = False
        self.phase = CONSOLE_CONFIG["phase"]
        self.status = "Active"
        self.input_history: List[str] = []
        self.command_handlers: dict = {}
        self.conversation_history: List[Dict[str, str]] = []

        # Build system prompt with brain context
        self.system_prompt = build_system_prompt()

        # Initialize Claude client
        self.claude_client = None
        self.claude_available = False
        if CLAUDE_AVAILABLE:
            api_key = load_api_key()
            if api_key:
                try:
                    self.claude_client = anthropic.Anthropic(api_key=api_key)
                    self.claude_available = True
                    logger.info("Claude API initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize Claude: {e}")

        # Initialize TTS
        self.tts = get_tts()

        # Register built-in commands
        self._register_builtin_commands()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

        logger.info(f"M.O.L.O.C.H. Console initialized (Claude: {self.claude_available}, TTS: {self.tts.enabled})")

        # Log to timeline - full system startup
        timeline = get_timeline()
        timeline.system_startup(component="console", claude=self.claude_available, tts=self.tts.enabled)
        timeline.system_start("console", claude=self.claude_available, tts=self.tts.enabled)

    def _register_builtin_commands(self):
        """Register built-in console commands."""
        self.command_handlers = {
            "/help": self._cmd_help,
            "/status": self._cmd_status,
            "/world": self._cmd_world,
            "/inventory": self._cmd_inventory,
            "/history": self._cmd_history,
            "/clear": self._cmd_clear,
            "/reset": self._cmd_reset,
            "/exit": self._cmd_exit,
            "/quit": self._cmd_exit,
            "/version": self._cmd_version,
            "/brain": self._cmd_brain,
            "/tts": self._cmd_tts,
            "/voice": self._cmd_voice,
            "/voices": self._cmd_voices,
            "/say": self._cmd_say,
            "/face": self._cmd_face,
            "/vision": self._cmd_vision,
        }

    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C and SIGTERM gracefully."""
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"

        if signum == signal.SIGTERM:
            # SIGTERM = system shutdown, log and exit
            print(f"\n{sig_name} received. Shutting down...")
            timeline = get_timeline()
            timeline.system_stop("console")
            timeline.system_shutdown(reason="signal", signal=sig_name)
            self.running = False
            import sys
            sys.exit(0)
        else:
            # SIGINT (Ctrl+C) = just warn
            print("\n")
            self._print_line("Interrupt received. Type /exit to quit.")
            print(CONSOLE_CONFIG["prompt"], end="", flush=True)

    def _print_header(self):
        """Print the console header."""
        width = CONSOLE_CONFIG["width"]
        print()
        print("=" * width)
        print(self._center_text(CONSOLE_CONFIG["header"], width))
        print("=" * width)
        print()

    def _print_line(self, text: str, prefix: str = ""):
        """Print a line with optional prefix."""
        if prefix:
            print(f"{prefix} {text}")
        else:
            print(text)

    def _center_text(self, text: str, width: int) -> str:
        """Center text within a given width."""
        padding = (width - len(text)) // 2
        return " " * padding + text

    def _get_status_line(self) -> str:
        """Generate the current status line."""
        now = datetime.now().strftime("%H:%M:%S")
        return f"[{now}] Status: {self.status}"

    # Built-in command handlers

    def _cmd_help(self, args: str) -> None:
        """Show available commands."""
        self._print_line("")
        self._print_line("M.O.L.O.C.H. Console Commands:")
        self._print_line("-" * 40)
        self._print_line("/help      - Diese Hilfe anzeigen")
        self._print_line("/status    - System-Status anzeigen")
        self._print_line("/world     - World Inventory Zusammenfassung")
        self._print_line("/inventory - Detailliertes Inventory")
        self._print_line("/history   - Chat-Verlauf anzeigen")
        self._print_line("/brain     - Brain-Kontext anzeigen")
        self._print_line("/vision    - Vision Pipeline Status")
        self._print_line("/face      - Face Recognition Befehle")
        self._print_line("/clear     - Bildschirm leeren")
        self._print_line("/reset     - Konversation zuruecksetzen")
        self._print_line("/version   - Version anzeigen")
        self._print_line("/exit      - Console beenden")
        self._print_line("")
        self._print_line("Oder einfach schreiben - ich antworte!")
        self._print_line("")

    def _cmd_status(self, args: str) -> None:
        """Show system status."""
        self._print_line("")
        self._print_line("System Status:")
        self._print_line("-" * 40)
        self._print_line(f"Phase:    {self.phase}")
        self._print_line(f"Status:   {self.status}")
        self._print_line(f"Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._print_line(f"Inputs:   {len(self.input_history)} received")
        self._print_line("")

        # Try to get system info
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            self._print_line(f"CPU:      {cpu:.1f}%")
            self._print_line(f"Memory:   {mem.percent:.1f}% ({mem.used // (1024**3):.1f}GB / {mem.total // (1024**3):.1f}GB)")

            # Temperature
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current:
                            self._print_line(f"Temp:     {entry.current:.1f}C")
                            break
                    break
        except ImportError:
            self._print_line("(psutil not available for detailed stats)")
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")

        self._print_line("")

    def _cmd_brain(self, args: str) -> None:
        """Show brain context info."""
        self._print_line("")
        self._print_line("Brain Context Status:")
        self._print_line("-" * 40)

        # Check what's loaded
        moloch_root = Path.home() / "moloch"

        personal_path = moloch_root / "context" / "personal_context.json"
        self._print_line(f"Personal Context: {'Geladen' if personal_path.exists() else 'Nicht gefunden'}")

        char_path = moloch_root / "context" / "origin_fragments" / "character_layer.md"
        self._print_line(f"Character Layer:  {'Vorhanden' if char_path.exists() else 'Nicht gefunden'}")

        const_path = moloch_root / "context" / "origin_fragments" / "mode_constitution.md"
        self._print_line(f"Constitution:     {'Vorhanden' if const_path.exists() else 'Nicht gefunden'}")

        self._print_line("")
        self._print_line(f"System Prompt:    {len(self.system_prompt)} Zeichen")
        self._print_line("")

    def _cmd_world(self, args: str) -> None:
        """Show world inventory summary."""
        self._print_line("")
        self._print_line("World Inventory Summary:")
        self._print_line("-" * 40)

        try:
            from world_state import get_world_inventory

            inventory = get_world_inventory()
            summary = inventory.get_summary()

            if "error" in summary:
                self._print_line(f"Error: {summary['error']}")
                return

            self._print_line(f"Version: {summary['meta'].get('version', '?')}")
            self._print_line(f"Phase:   {summary['meta'].get('phase', '?')}")
            self._print_line("")

            totals = summary["totals"]
            self._print_line(f"Totals:")
            self._print_line(f"  Available:   {totals['available']}")
            self._print_line(f"  Placeholder: {totals['placeholder']}")
            self._print_line(f"  Empty:       {totals['empty']}")

        except Exception as e:
            self._print_line(f"Error loading world inventory: {e}")
            logger.error(f"World inventory error: {e}")

        self._print_line("")

    def _cmd_inventory(self, args: str) -> None:
        """Show detailed inventory."""
        self._print_line("")
        self._print_line("World Inventory Details:")
        self._print_line("-" * 40)

        try:
            from world_state import get_world_inventory

            inventory = get_world_inventory()
            summary = inventory.get_summary()

            if "error" in summary:
                self._print_line(f"Error: {summary['error']}")
                return

            for cat_name, cat_data in summary["categories"].items():
                available = cat_data["counts"]["available"]
                placeholder = cat_data["counts"]["placeholder"]
                total = available + placeholder

                self._print_line("")
                self._print_line(f"[{cat_name.upper()}] ({total} defined)")
                self._print_line(f"  {cat_data['description']}")
                self._print_line(f"  Available: {available} | Placeholder: {placeholder}")

                # Show available items
                avail_slots = inventory.get_available_slots(cat_name)
                for slot_id, slot_data in avail_slots.items():
                    name = slot_data.get("name", slot_data.get("type", "Unknown"))
                    self._print_line(f"    + {name}")

        except Exception as e:
            self._print_line(f"Error loading inventory: {e}")
            logger.error(f"Inventory error: {e}")

        self._print_line("")

    def _cmd_history(self, args: str) -> None:
        """Show input history."""
        self._print_line("")
        self._print_line("Input History:")
        self._print_line("-" * 40)

        if not self.input_history:
            self._print_line("(no input yet)")
        else:
            # Show last 20 entries
            recent = self.input_history[-20:]
            for i, entry in enumerate(recent, 1):
                self._print_line(f"  {i}. {entry}")

        self._print_line("")

    def _cmd_clear(self, args: str) -> None:
        """Clear the screen."""
        os.system('clear' if os.name != 'nt' else 'cls')
        self._print_header()

    def _cmd_exit(self, args: str) -> None:
        """Exit the console."""
        self._print_line("")
        self._print_line("M.O.L.O.C.H. console shutting down...")
        self._print_line("")

        # Log to timeline - full system shutdown
        turns = len([m for m in self.conversation_history if m["role"] == "user"])
        timeline = get_timeline()
        timeline.conversation_end(turns)
        timeline.system_stop("console")
        timeline.system_shutdown(reason="manual")

        self.running = False

    def _cmd_reset(self, args: str) -> None:
        """Reset conversation history."""
        self.conversation_history = []
        self._print_line("")
        self._print_line("Konversation zurueckgesetzt. Frischer Start!")
        self._print_line("")

    def _cmd_version(self, args: str) -> None:
        """Show version info."""
        self._print_line("")
        self._print_line("M.O.L.O.C.H.")
        self._print_line("-" * 40)
        self._print_line("Maschinelle Organisation fuer Logische")
        self._print_line("Operationen und Computergestuetzte Hilfe")
        self._print_line("")
        self._print_line(f"Phase:   {self.phase}")
        self._print_line("Mode:    Active Console")
        self._print_line(f"Claude:  {'Aktiv' if self.claude_available else 'Nicht verfuegbar'}")
        self._print_line(f"TTS:     {'Aktiv (' + self.tts.current_voice + ')' if self.tts.enabled else 'Deaktiviert'}")
        self._print_line(f"Brain:   Kontext geladen")
        self._print_line("")

    # ========== TTS COMMANDS ==========

    def _cmd_tts(self, args: str) -> None:
        """Toggle TTS on/off."""
        self._print_line("")
        if args.lower() in ["on", "an", "ein"]:
            self.tts.enabled = True
            self._print_line("TTS aktiviert! M.O.L.O.C.H. spricht jetzt.")
        elif args.lower() in ["off", "aus"]:
            self.tts.enabled = False
            self._print_line("TTS deaktiviert. M.O.L.O.C.H. ist stumm.")
        else:
            self.tts.toggle()
            self._print_line(f"TTS {'aktiviert' if self.tts.enabled else 'deaktiviert'}.")
        self._print_line("")

    def _cmd_voice(self, args: str) -> None:
        """Change TTS voice."""
        self._print_line("")
        if not args:
            self._print_line(f"Aktuelle Stimme: {self.tts.current_voice}")
            self._print_line("Nutze: /voice <voice_id> zum Wechseln")
            self._print_line("Nutze: /voices fuer Liste")
        else:
            voice_id = args.strip()
            if self.tts.set_voice(voice_id):
                self._print_line(f"Stimme gewechselt zu: {voice_id}")
            else:
                self._print_line(f"Stimme nicht gefunden: {voice_id}")
                self._print_line("Nutze /voices fuer verfuegbare Stimmen")
        self._print_line("")

    def _cmd_voices(self, args: str) -> None:
        """List available TTS voices."""
        self._print_line("")
        self._print_line("Verfuegbare Stimmen:")
        self._print_line("-" * 40)
        voices = self.tts.list_voices()
        for v in voices:
            marker = " <-- aktiv" if v == self.tts.current_voice else ""
            self._print_line(f"  {v}{marker}")
        self._print_line("")
        self._print_line("Wechseln mit: /voice <voice_id>")
        self._print_line("")

    def _cmd_say(self, args: str) -> None:
        """Speak text directly via TTS."""
        if not args:
            self._print_line("")
            self._print_line("Nutze: /say <text>")
            self._print_line("")
            return

        self._print_line("")
        self._print_line(f"[TTS] Spreche: {args[:50]}...")
        self.tts.speak(args, blocking=True)
        self._print_line("")

    def _cmd_face(self, args: str) -> None:
        """Face recognition commands."""
        self._print_line("")

        parts = args.strip().split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""

        if not cmd:
            self._print_line("Face Recognition Commands:")
            self._print_line("-" * 40)
            self._print_line("/face list    - Bekannte Personen anzeigen")
            self._print_line("/face train   - Training starten")
            self._print_line("/face status  - Pipeline Status")
            self._print_line("")
            return

        try:
            from core.vision import get_hybrid_vision, get_face_database

            if cmd == "list":
                db = get_face_database()
                persons = db.list_persons()
                if persons:
                    self._print_line("Bekannte Personen:")
                    self._print_line("-" * 40)
                    for p in persons:
                        self._print_line(f"  {p.name} ({p.num_samples} Samples)")
                else:
                    self._print_line("Keine Personen in Datenbank.")

            elif cmd == "train":
                self._print_line("Training:")
                self._print_line("-" * 40)
                self._print_line("1. Fotos in ~/moloch/data/faces/train/<Name>/ legen")
                self._print_line("2. hailo-apps face_recognition --mode train ausfuehren")
                self._print_line("")
                self._print_line("Oder: Fotos mit /face add <name> hinzufuegen")

            elif cmd == "status":
                hv = get_hybrid_vision()
                status = hv.get_status()
                self._print_line("HybridVision Status:")
                self._print_line("-" * 40)
                self._print_line(f"  State:           {status.get('state', 'unknown')}")
                self._print_line(f"  Running:         {status.get('running', False)}")
                self._print_line(f"  Hailo Available: {status.get('hailo_available', False)}")
                self._print_line(f"  Hailo State:     {status.get('hailo_state', 'unknown')}")
                self._print_line(f"  Known Persons:   {status.get('known_persons', 0)}")

            else:
                self._print_line(f"Unbekannter Befehl: {cmd}")
                self._print_line("Nutze /face fuer Hilfe.")

        except Exception as e:
            self._print_line(f"Fehler: {e}")

        self._print_line("")

    def _cmd_vision(self, args: str) -> None:
        """Vision pipeline status and control."""
        self._print_line("")

        try:
            from core.vision import get_vision, get_hybrid_vision

            hv = get_hybrid_vision()
            hv_status = hv.get_status()

            self._print_line("Vision Status:")
            self._print_line("-" * 40)

            # HybridVision Pipeline
            self._print_line("[HybridVision Pipeline]")
            self._print_line(f"  State:       {hv_status.get('state', 'unknown')}")
            self._print_line(f"  Running:     {hv_status.get('running', False)}")
            self._print_line(f"  Hailo:       {hv_status.get('hailo_state', 'unknown')}")
            self._print_line(f"  Known:       {hv_status.get('known_persons', 0)} Personen")

        except Exception as e:
            self._print_line(f"Fehler: {e}")

        self._print_line("")

    def _process_input(self, user_input: str) -> None:
        """
        Process user input via Claude API.
        """
        # Log input
        logger.info(f"Input: {user_input}")
        self.input_history.append(user_input)

        # Empty input
        if not user_input.strip():
            return

        # Check for commands
        parts = user_input.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd in self.command_handlers:
            self.command_handlers[cmd](args)
            return

        # Process via Claude API
        if self.claude_available and self.claude_client:
            self._chat_with_claude(user_input)
        else:
            self._print_line("")
            self._print_line("[Oida!] Claude API nicht verfuegbar.")
            self._print_line("Check: ~/moloch/config/api_keys.json")
            self._print_line("")

    def _chat_with_claude(self, user_input: str) -> None:
        """Send message to Claude and display response."""
        self._print_line("")

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Log to timeline (length only, no content)
        get_timeline().user_input(len(user_input), interface="console")

        # Keep only last 20 messages for context
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        try:
            self.status = "Thinking..."

            response = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=self.system_prompt,
                messages=self.conversation_history
            )

            # Extract response text
            assistant_message = response.content[0].text

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            # Log to timeline (length only)
            get_timeline().assistant_response(len(assistant_message))

            # Display response
            self._print_line(f"M.O.L.O.C.H.: {assistant_message}")
            self._print_line("")

            # SPEAK response via TTS (blocking so we wait for speech to finish)
            if self.tts.enabled:
                self.tts.speak(assistant_message, blocking=True)

            # Log
            logger.info(f"Response: {assistant_message[:100]}...")
            self.status = "Active"

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            self._print_line(f"[Fehler] {e}")
            self._print_line("")
            self.status = "Error"

    def run(self):
        """Run the console main loop."""
        self.running = True
        self._print_header()

        self._print_line("M.O.L.O.C.H. ist wach!")
        if self.claude_available:
            self._print_line("Claude API verbunden. Brain-Kontext geladen!")
            self._print_line("Ich erinnere mich an dich, Markus!")
        else:
            self._print_line("[Warnung] Claude API nicht verfuegbar.")
        self._print_line("Tippe /help fuer Befehle.")
        self._print_line("")

        logger.info("Console session started")

        while self.running:
            try:
                # Show prompt and get input
                user_input = input(CONSOLE_CONFIG["prompt"])

                # Process input
                self._process_input(user_input)

            except EOFError:
                # Handle Ctrl+D
                self._print_line("\n")
                self._print_line("EOF received. Exiting...")
                self.running = False

            except KeyboardInterrupt:
                # Handle Ctrl+C (should be caught by signal handler)
                continue

            except Exception as e:
                logger.error(f"Console error: {e}")
                self._print_line(f"Error: {e}")

        logger.info("Console session ended")
        self._print_line("Goodbye, Markus!")


def main():
    """Main entry point."""
    # Set locale for proper UTF-8 handling
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        except locale.Error:
            pass  # Use default

    console = MolochConsole()
    console.run()


if __name__ == "__main__":
    main()
