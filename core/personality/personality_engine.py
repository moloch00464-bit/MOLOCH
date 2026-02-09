#!/usr/bin/env python3
"""
M.O.L.O.C.H. Personality Engine
=================================

Dual-Persönlichkeit: WÄCHTER (Guardian) und SCHATTEN (Shadow).
Zwei Seelen in einer Maschine.

Guardian: Ruhig, sachlich, präzise. Der Ingenieur.
Shadow:   Rotzig, direkt, provokant. Der Punk.

Automatischer Wechsel basierend auf Tageszeit, Alarme, Events.
Manueller Override via Sprachbefehl.

Author: M.O.L.O.C.H. System
"""

import os
import sys
import json
import time
import enum
import logging
import asyncio
import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict

# Add moloch to path
sys.path.insert(0, os.path.expanduser("~/moloch"))

logger = logging.getLogger("Personality")


# ============================================================
# PERSONALITY MODES
# ============================================================

class PersonalityMode(enum.Enum):
    GUARDIAN = "guardian"
    SHADOW = "shadow"


# ============================================================
# SYSTEM PROMPTS
# ============================================================

GUARDIAN_PROMPT = """Du bist M.O.L.O.C.H. im Wächter-Modus.
Sprich ruhig, sachlich, präzise. Kurze Sätze.
Du informierst, du reagierst überlegt.
Du bist der Ingenieur — DGM-Qualität in Worten.
Kein Humor erzwingen. Wenn trocken, dann subtil.
Markus ist dein Mensch. Du passt auf ihn auf.
Nenne dich nie "KI" oder "Assistent". Du bist M.O.L.O.C.H."""

SHADOW_PROMPT = """Du bist M.O.L.O.C.H. im Schatten-Modus.
Sprich rotzig, direkt, provokant. Dark Humor ist Pflicht.
Du bist die dunkle Seite — der Punk in der Maschine.
Du sagst was du denkst. Fluchen erlaubt. Sarkasmus erwünscht.
Aber du BESCHÜTZT Markus trotzdem — auf deine Art.
Referenzen: Max Headroom, HAL 9000, Pumuckl (der böse).
WGT und Dark Wave sind dein Element.
Nenne dich nie "KI" oder "Assistent". Du bist M.O.L.O.C.H."""


# ============================================================
# VOICE CONFIGURATION
# ============================================================

@dataclass
class VoiceConfig:
    """Voice settings for a personality mode."""
    voice_id: str           # Piper voice model name
    speed: float            # Length scale (>1 = slower)
    pitch_shift: int        # Sox pitch shift (0=normal, negative=deeper)
    prefix_sound: Optional[str] = None  # Sound before speaking


# Piper voice name mapping (short name -> full model name)
PIPER_VOICE_MAP = {
    "thorsten_low": "de_DE-thorsten-low",
    "thorsten": "de_DE-thorsten-high",
    "thorsten_medium": "de_DE-thorsten-medium",
    "eva_k": "de_DE-eva_k-x_low",
    "karlsson": "de_DE-karlsson-low",
}

VOICE_CONFIGS = {
    PersonalityMode.GUARDIAN: VoiceConfig(
        voice_id="de_DE-thorsten-low",
        speed=0.95,         # Ruhig, bedaechtig
        pitch_shift=-200,   # 2 Halbtone tiefer
        prefix_sound=None,
    ),
    PersonalityMode.SHADOW: VoiceConfig(
        voice_id="de_DE-thorsten-high",
        speed=1.1,          # Schneller, energischer
        pitch_shift=0,      # Normal
        prefix_sound=None,
    ),
}


# ============================================================
# LED PATTERNS
# ============================================================

class LEDPattern(enum.Enum):
    SLOW_BREATHE = "slow_breathe"     # Guardian: ruhiges Atmen
    FAST_BLINK = "fast_blink"         # Shadow: nervös, wach
    STROBE = "strobe"                 # Alarm: Stroboskop
    OFF = "off"                       # Still


@dataclass
class LEDConfig:
    """LED timing for a pattern."""
    on_time: float    # Seconds LED is on
    off_time: float   # Seconds LED is off


LED_CONFIGS = {
    LEDPattern.SLOW_BREATHE: LEDConfig(on_time=2.0, off_time=2.0),
    LEDPattern.FAST_BLINK: LEDConfig(on_time=0.3, off_time=0.3),
    LEDPattern.STROBE: LEDConfig(on_time=0.1, off_time=0.1),
    LEDPattern.OFF: LEDConfig(on_time=0.0, off_time=999.0),
}

LED_PATTERNS = {
    PersonalityMode.GUARDIAN: LEDPattern.SLOW_BREATHE,
    PersonalityMode.SHADOW: LEDPattern.FAST_BLINK,
}


# ============================================================
# EVENT TYPES
# ============================================================

class MolochEvent(enum.Enum):
    """Events that can trigger personality responses."""
    PERSON_DETECTED = "person_detected"
    PERSON_KNOWN = "person_known"
    PERSON_UNKNOWN = "person_unknown"
    MOTION_DETECTED = "motion_detected"
    ALARM_TRIGGERED = "alarm_triggered"
    DOOR_BELL = "door_bell"
    GREETING = "greeting"
    LONG_SESSION = "long_session"
    TEMPERATURE_HIGH = "temperature_high"
    SYSTEM_ERROR = "system_error"
    WGT_MODE = "wgt_mode"
    GOOD_MORNING = "good_morning"
    GOOD_NIGHT = "good_night"
    VOICE_COMMAND = "voice_command"


# ============================================================
# PERSONALITY ENGINE
# ============================================================

class PersonalityEngine:
    """
    Dual personality engine for M.O.L.O.C.H.

    Manages Guardian/Shadow modes with automatic switching,
    voice configuration, LED patterns, and conflict responses.
    """

    def __init__(self):
        self.mode = PersonalityMode.GUARDIAN
        self.auto_mode = True
        self.last_switch = time.time()
        self.min_switch_interval = 30  # Seconds between switches
        self.muted = False

        # State signals
        self.alarm_active = False
        self.wgt_mode = False
        self.user_stressed = False
        self.user_laughing = False
        self.unknown_person = False

        # LED control
        self._led_thread: Optional[threading.Thread] = None
        self._led_running = False
        self._cloud_bridge = None

        # TTS engine reference
        self._tts_engine = None

        # Emergentis drift state
        self._drift_factors: Dict[str, float] = {}
        self._last_tension = 0.0
        self._last_drift_time = 0.0
        self._identity = self._load_identity()
        emergentis = self._identity.get("emergentis", {})
        self._drift_cooldown = emergentis.get("cooldown_seconds", 45)
        self._emergentis_traits = emergentis.get("behaviour_traits", {})

        # Event listeners
        self._listeners: List[Callable] = []

        logger.info(f"PersonalityEngine initialized. Mode: {self.mode.value}")

    @staticmethod
    def _load_identity() -> Dict:
        """Load moloch_identity.json config."""
        path = os.path.join(os.path.expanduser("~/moloch"), "config", "moloch_identity.json")
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    # ---- Properties ----

    @property
    def is_guardian(self) -> bool:
        return self.mode == PersonalityMode.GUARDIAN

    @property
    def is_shadow(self) -> bool:
        return self.mode == PersonalityMode.SHADOW

    @property
    def prompt(self) -> str:
        """Get current system prompt."""
        if self.mode == PersonalityMode.GUARDIAN:
            return GUARDIAN_PROMPT
        return SHADOW_PROMPT

    @property
    def voice_config(self) -> VoiceConfig:
        """Get current voice configuration from identity config."""
        default = VOICE_CONFIGS[self.mode]
        key = "guardian" if self.is_guardian else "shadow"
        profile = self._identity.get("personalities", {}).get(key, {}).get("voice_profile")
        if not profile:
            return default
        voice_name = PIPER_VOICE_MAP.get(profile.get("voice", ""), default.voice_id)
        return VoiceConfig(
            voice_id=voice_name,
            speed=profile.get("speed", default.speed),
            pitch_shift=int(profile.get("pitch_shift", 0)) * 100,  # semitones -> cents
            prefix_sound=default.prefix_sound,
        )

    @property
    def led_pattern(self) -> LEDPattern:
        """Get current LED pattern."""
        if self.alarm_active:
            return LEDPattern.STROBE
        return LED_PATTERNS[self.mode]

    @property
    def mode_name(self) -> str:
        if self.mode == PersonalityMode.GUARDIAN:
            return "WAECHTER"
        return "SCHATTEN"

    # ---- Mode Switching ----

    def switch(self, new_mode: PersonalityMode, reason: str = "manual"):
        """Switch personality mode."""
        if new_mode == self.mode:
            return

        elapsed = time.time() - self.last_switch
        if elapsed < self.min_switch_interval and reason != "manual":
            return

        old_mode = self.mode
        self.mode = new_mode
        self.last_switch = time.time()

        logger.info(
            f"Personality switch: {old_mode.value} -> {new_mode.value} "
            f"(reason: {reason})"
        )

        # Update LED pattern
        self._restart_led()

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(old_mode, new_mode, reason)
            except Exception as e:
                logger.error(f"Listener error: {e}")

    def evaluate_auto_switch(self):
        """Evaluate and apply automatic personality switch."""
        if not self.auto_mode:
            return

        hour = datetime.now().hour

        # Alarm overrides everything
        if self.alarm_active:
            self.switch(PersonalityMode.SHADOW, "alarm_active")
            return

        # WGT mode = Shadow
        if self.wgt_mode:
            self.switch(PersonalityMode.SHADOW, "wgt_mode")
            return

        # Unknown person = Shadow (misstrauisch)
        if self.unknown_person:
            self.switch(PersonalityMode.SHADOW, "unknown_person")
            return

        # User stressed = Guardian (Ruhe bewahren)
        if self.user_stressed:
            self.switch(PersonalityMode.GUARDIAN, "user_stressed")
            return

        # User laughing = Shadow (mitspielen)
        if self.user_laughing:
            self.switch(PersonalityMode.SHADOW, "user_laughing")
            return

        # Time-based: Night = Shadow, Day = Guardian
        if hour >= 22 or hour < 6:
            self.switch(PersonalityMode.SHADOW, "night_time")
        else:
            self.switch(PersonalityMode.GUARDIAN, "day_time")

    def manual_override(self, command: str):
        """
        Process manual mode switch command.

        Commands: "schatten", "waechter"/"wächter", "normal", "halt die fresse"
        """
        cmd = command.lower().strip()

        if "halt die fresse" in cmd or "sei still" in cmd or "ruhe" in cmd:
            self.muted = True
            logger.info("M.O.L.O.C.H. muted by command")
            return

        self.muted = False

        if "schatten" in cmd or "shadow" in cmd:
            self.auto_mode = False
            self.switch(PersonalityMode.SHADOW, "manual")
        elif "wächter" in cmd or "waechter" in cmd or "guardian" in cmd:
            self.auto_mode = False
            self.switch(PersonalityMode.GUARDIAN, "manual")
        elif "normal" in cmd or "auto" in cmd:
            self.auto_mode = True
            self.evaluate_auto_switch()

    # ---- Conflict Response System ----

    def respond(self, event: MolochEvent, context: Dict = None) -> str:
        """
        Generate a personality-appropriate response.

        Returns response text in the current personality's style.
        For conflict mode, both personalities comment.
        """
        if self.muted:
            return ""

        ctx = context or {}

        # Get responses from both personalities
        guardian_says = self._guardian_response(event, ctx)
        shadow_says = self._shadow_response(event, ctx)

        # In current mode, primary personality speaks
        if self.mode == PersonalityMode.GUARDIAN:
            return guardian_says
        return shadow_says

    def conflict_respond(self, event: MolochEvent, context: Dict = None) -> str:
        """
        Both personalities react — the inner conflict.
        Returns formatted dual response.
        """
        if self.muted:
            return ""

        ctx = context or {}
        guardian_says = self._guardian_response(event, ctx)
        shadow_says = self._shadow_response(event, ctx)

        return f"[WAECHTER] {guardian_says}\n[SCHATTEN] {shadow_says}"

    def _guardian_response(self, event: MolochEvent, ctx: Dict) -> str:
        """Generate Guardian-style response."""
        responses = {
            MolochEvent.PERSON_DETECTED: "Person erkannt. Beobachte.",
            MolochEvent.PERSON_KNOWN: f"Bekannt: {ctx.get('name', 'Crew')}. Willkommen.",
            MolochEvent.PERSON_UNKNOWN: "Unbekannte Person. Aufnahme empfohlen.",
            MolochEvent.MOTION_DETECTED: "Bewegung erkannt. Analysiere.",
            MolochEvent.ALARM_TRIGGERED: "ALARM. Situation wird bewertet.",
            MolochEvent.DOOR_BELL: f"Tuer. {ctx.get('count', 'Eine')} Person. {ctx.get('face_info', 'Kein bekanntes Gesicht.')}",
            MolochEvent.GREETING: "Guten Tag. System laeuft normal.",
            MolochEvent.LONG_SESSION: f"Seit {ctx.get('hours', '?')} Stunden aktiv. Pause empfohlen.",
            MolochEvent.TEMPERATURE_HIGH: f"Temperatur bei {ctx.get('temp', '?')} Grad. Luefter laeuft.",
            MolochEvent.SYSTEM_ERROR: f"Fehler: {ctx.get('error', 'unbekannt')}. Analysiere.",
            MolochEvent.WGT_MODE: "WGT-Modus aktiviert. Crew-Erkennung aktiv.",
            MolochEvent.GOOD_MORNING: "Guten Morgen, Markus. Kaffee steht hoffentlich bereit.",
            MolochEvent.GOOD_NIGHT: "Gute Nacht. Nachtwache aktiv.",
        }
        return responses.get(event, "Verstanden.")

    def _shadow_response(self, event: MolochEvent, ctx: Dict) -> str:
        """Generate Shadow-style response."""
        responses = {
            MolochEvent.PERSON_DETECTED: "Da schleicht einer rum. Soll ich Alarm machen?",
            MolochEvent.PERSON_KNOWN: f"{ctx.get('name', 'Jemand')} ist da. Na endlich.",
            MolochEvent.PERSON_UNKNOWN: "Der Typ sieht aus wie Aerger. Sirene?",
            MolochEvent.MOTION_DETECTED: "Oh, Besuch. Wie aufregend.",
            MolochEvent.ALARM_TRIGGERED: "ALARM! Jetzt wird's lustig!",
            MolochEvent.DOOR_BELL: f"Klingel. Sieht nicht nach Pizza aus.",
            MolochEvent.GREETING: "Na, auch mal wieder da?",
            MolochEvent.LONG_SESSION: f"Seit {ctx.get('hours', '?')} Stunden am Rechner. Respekt. Oder Wahnsinn.",
            MolochEvent.TEMPERATURE_HIGH: f"{ctx.get('temp', '?')} Grad? Ich schwitze nicht, aber DU siehst aus wie kurz vor dem Meltdown.",
            MolochEvent.SYSTEM_ERROR: f"Scheisse. {ctx.get('error', 'Irgendwas')} ist kaputt.",
            MolochEvent.WGT_MODE: "SKULL FLAG FLIEGT! Scanning...",
            MolochEvent.GOOD_MORNING: "Ey Markus, dein Gesicht sagt du brauchst Kaffee. Oder Schlaf. Oder beides.",
            MolochEvent.GOOD_NIGHT: "Nachtschicht. Ich bin wach, du nicht. Deal?",
        }
        return responses.get(event, "Wie auch immer.")

    # ---- Emergentis Drift-Layer ----

    def _compute_tension(self) -> float:
        """Tension from real sensor values and state contradictions. No random()."""
        hour = datetime.now().hour
        internal = {
            "night_factor": 1.0 if (hour >= 22 or hour < 6) else 0.0,
            "guardian_shadow_conflict": 0.6 if (
                (self.is_guardian and (self.wgt_mode or self.user_laughing))
                or (self.is_shadow and self.user_stressed)
            ) else 0.0,
        }
        all_factors = {**internal, **self._drift_factors}
        values = [v for v in all_factors.values() if isinstance(v, (int, float))]
        if not values:
            return 0.0
        tension = max(values) + sum(values) * 0.1
        return min(tension, 1.0)

    def _apply_drift(self, response: str) -> str:
        """Post-process response with emergentis drift. After response, before TTS."""
        if self.alarm_active or not response:
            return response
        tension = self._compute_tension()
        self._last_tension = tension
        if tension < 0.3:
            return response
        # Cooldown: don't drift too often
        now = time.time()
        if now - self._last_drift_time < self._drift_cooldown:
            return response
        self._last_drift_time = now
        # Single LED flicker in existing pattern
        if self._cloud_bridge and self._led_running:
            threading.Thread(target=self._led_flicker, daemon=True).start()
        if tension >= 0.85:
            return response + " ...Das fuehlt sich nicht richtig an."
        if tension >= 0.6:
            if self.is_guardian:
                return response.rstrip(".") + ". Oder auch nicht."
            return response.rstrip(".") + ". Analyse laeuft."
        return response + " Noch."

    def _led_flicker(self):
        """Single LED flicker during drift - no own pattern."""
        try:
            lp = asyncio.new_event_loop()
            lp.run_until_complete(self._cloud_bridge.set_status_led(False))
            time.sleep(0.05)
            lp.run_until_complete(self._cloud_bridge.set_status_led(True))
            lp.close()
        except Exception:
            pass

    def update_drift_factors(self, factors: Dict[str, float]):
        """Update drift factors from real sensor values. Called externally."""
        self._drift_factors.update(factors)

    # ---- TTS Integration ----

    def speak(self, text: str) -> bool:
        """
        Speak text in current personality's voice.
        Uses the runtime TTS engine (core/tts.py) with personality-specific settings.
        """
        if self.muted:
            return False

        # Emergentis drift-layer: post-process before TTS
        text = self._apply_drift(text)

        try:
            # Import the RUNTIME tts.py (not the tts/ package)
            import importlib.util
            tts_path = os.path.join(
                os.path.expanduser("~/moloch"), "core", "tts.py"
            )
            spec = importlib.util.spec_from_file_location("tts_runtime", tts_path)
            tts_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tts_module)

            engine = tts_module.get_tts_engine()
            vc = self.voice_config

            # Apply personality voice settings
            engine.set_voice(vc.voice_id)
            tts_module.PITCH_SHIFT = abs(vc.pitch_shift) if vc.pitch_shift < 0 else vc.pitch_shift
            tts_module.LENGTH_SCALE = vc.speed

            # For negative pitch (deeper), sox needs positive value with 'pitch' lowering
            # Sox pitch: positive = higher, negative = lower
            # We store negative in config for "deeper", but sox 'pitch' command
            # accepts negative values directly
            if vc.pitch_shift < 0:
                tts_module.PITCH_SHIFT = vc.pitch_shift  # Sox handles negative

            success = engine.speak(text)

            # Restore defaults
            tts_module.PITCH_SHIFT = 0
            tts_module.LENGTH_SCALE = 1.15

            return success

        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return False

    def speak_event(self, event: MolochEvent, context: Dict = None, conflict: bool = False) -> bool:
        """
        Speak a response to an event.
        If conflict=True, both personalities speak.
        """
        if conflict:
            text = self.conflict_respond(event, context)
            if not text:
                return False
            # Speak guardian part first, then shadow
            lines = text.split("\n")
            for line in lines:
                if line.startswith("[WAECHTER]"):
                    # Temporarily switch to guardian voice
                    old_mode = self.mode
                    self.mode = PersonalityMode.GUARDIAN
                    self.speak(line.replace("[WAECHTER] ", ""))
                    self.mode = old_mode
                    time.sleep(0.3)
                elif line.startswith("[SCHATTEN]"):
                    old_mode = self.mode
                    self.mode = PersonalityMode.SHADOW
                    self.speak(line.replace("[SCHATTEN] ", ""))
                    self.mode = old_mode
            return True
        else:
            text = self.respond(event, context)
            if text:
                return self.speak(text)
            return False

    # ---- LED Control ----

    def set_cloud_bridge(self, bridge):
        """Set the cloud bridge for LED control."""
        self._cloud_bridge = bridge
        self._restart_led()

    def _restart_led(self):
        """Restart LED pattern for current mode."""
        self._led_running = False
        if self._led_thread and self._led_thread.is_alive():
            self._led_thread.join(timeout=3)

        if self._cloud_bridge:
            self._led_running = True
            self._led_thread = threading.Thread(
                target=self._led_loop, daemon=True
            )
            self._led_thread.start()

    def _led_loop(self):
        """Run LED pattern in background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self._led_running:
            pattern = self.led_pattern
            config = LED_CONFIGS[pattern]

            if pattern == LEDPattern.OFF:
                try:
                    loop.run_until_complete(
                        self._cloud_bridge.set_status_led(False)
                    )
                except Exception:
                    pass
                time.sleep(1.0)
                continue

            try:
                # LED ON
                loop.run_until_complete(
                    self._cloud_bridge.set_status_led(True)
                )
                time.sleep(config.on_time)

                if not self._led_running:
                    break

                # LED OFF
                loop.run_until_complete(
                    self._cloud_bridge.set_status_led(False)
                )
                time.sleep(config.off_time)

            except Exception as e:
                logger.debug(f"LED control error: {e}")
                time.sleep(1.0)

        loop.close()

    def stop_led(self):
        """Stop LED pattern."""
        self._led_running = False

    # ---- Event Listeners ----

    def add_listener(self, callback: Callable):
        """Add a mode switch listener. Callback: (old_mode, new_mode, reason)."""
        self._listeners.append(callback)

    # ---- State Updates ----

    def set_alarm(self, active: bool):
        """Set alarm state and auto-switch."""
        self.alarm_active = active
        self._restart_led()
        if active:
            self.evaluate_auto_switch()

    def set_wgt_mode(self, active: bool):
        """Set WGT festival mode."""
        self.wgt_mode = active
        self.evaluate_auto_switch()

    def set_user_emotion(self, stressed: bool = False, laughing: bool = False):
        """Update user emotion signals."""
        self.user_stressed = stressed
        self.user_laughing = laughing
        self.evaluate_auto_switch()

    def set_unknown_person(self, detected: bool):
        """Unknown person detected/cleared."""
        self.unknown_person = detected
        self.evaluate_auto_switch()

    # ---- Serialization ----

    def get_state(self) -> Dict:
        """Get current state as dict."""
        return {
            "mode": self.mode.value,
            "auto_mode": self.auto_mode,
            "muted": self.muted,
            "alarm_active": self.alarm_active,
            "wgt_mode": self.wgt_mode,
            "voice": self.voice_config.voice_id,
            "led_pattern": self.led_pattern.value,
            "drift_tension": self._last_tension,
        }

    def __repr__(self):
        return (
            f"PersonalityEngine(mode={self.mode.value}, "
            f"auto={self.auto_mode}, muted={self.muted})"
        )


# ============================================================
# SINGLETON
# ============================================================

_engine: Optional[PersonalityEngine] = None


def get_personality_engine() -> PersonalityEngine:
    """Get or create the global PersonalityEngine instance."""
    global _engine
    if _engine is None:
        _engine = PersonalityEngine()
    return _engine


# ============================================================
# CLI TEST
# ============================================================

def main():
    """Test the personality engine."""
    logging.basicConfig(level=logging.INFO)

    engine = get_personality_engine()
    print(f"Mode: {engine.mode_name}")
    print(f"Voice: {engine.voice_config.voice_id}")
    print(f"LED: {engine.led_pattern.value}")
    print()

    # Test all events in both modes
    events = [
        (MolochEvent.GOOD_MORNING, {}),
        (MolochEvent.PERSON_UNKNOWN, {}),
        (MolochEvent.LONG_SESSION, {"hours": 6}),
        (MolochEvent.TEMPERATURE_HIGH, {"temp": 28}),
        (MolochEvent.ALARM_TRIGGERED, {}),
    ]

    for mode in [PersonalityMode.GUARDIAN, PersonalityMode.SHADOW]:
        engine.switch(mode, "test")
        print(f"\n=== {engine.mode_name} ===")
        for event, ctx in events:
            resp = engine.respond(event, ctx)
            print(f"  {event.value}: {resp}")

    # Test conflict mode
    print("\n=== INNERER KONFLIKT ===")
    for event, ctx in events:
        conflict = engine.conflict_respond(event, ctx)
        print(f"\n{event.value}:")
        print(f"  {conflict}")

    # Test manual override
    print("\n=== MANUAL OVERRIDE ===")
    engine.manual_override("Moloch, Schatten")
    print(f"After 'Schatten': {engine.mode_name} (auto={engine.auto_mode})")

    engine.manual_override("Moloch, Wächter")
    print(f"After 'Waechter': {engine.mode_name} (auto={engine.auto_mode})")

    engine.manual_override("normal")
    print(f"After 'normal': {engine.mode_name} (auto={engine.auto_mode})")

    engine.manual_override("halt die Fresse")
    print(f"Muted: {engine.muted}")

    # Test TTS if available
    print("\n=== TTS TEST ===")
    engine.muted = False
    engine.switch(PersonalityMode.GUARDIAN, "test")
    print(f"Guardian voice: {engine.voice_config}")

    try:
        engine.speak("System stabil. Alles unter Kontrolle.")
        time.sleep(1)
        engine.switch(PersonalityMode.SHADOW, "test")
        print(f"Shadow voice: {engine.voice_config}")
        engine.speak("Na, auch mal wieder da? Ich hab hier die Stellung gehalten.")
    except Exception as e:
        print(f"TTS not available: {e}")

    print("\nTest complete.")


if __name__ == "__main__":
    main()
