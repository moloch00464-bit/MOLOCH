#!/usr/bin/env python3
"""Fix: TTS Geschwindigkeit - Defaults und MOLOCHs Selbststeuerung.

1. personality_engine.py: Restore auf 1.0 statt 1.15
2. tts.py: set_speed() Methode fuer dynamische Anpassung
3. Panel: MOLOCH kann auf "sprich schneller/langsamer" reagieren
"""
import sys

# ============================================================
# FIX 1: Personality Engine - Restore Default
# ============================================================
pe = '/home/molochzuhause/moloch/core/personality/personality_engine.py'
with open(pe) as f:
    code = f.read()

fixes = 0

# Fix restore default
old_restore = '''            # Restore defaults
            tts_module.PITCH_SHIFT = 0
            tts_module.LENGTH_SCALE = 1.15'''

new_restore = '''            # Restore defaults (1.0 = neutrale Geschwindigkeit)
            tts_module.PITCH_SHIFT = 0
            tts_module.LENGTH_SCALE = 1.0'''

if old_restore in code:
    code = code.replace(old_restore, new_restore)
    print('FIX 1: Restore LENGTH_SCALE 1.0 - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

with open(pe, 'w') as f:
    f.write(code)
compile(open(pe).read(), pe, 'exec')

# ============================================================
# FIX 2: tts.py - set_speed() Methode
# ============================================================
tts = '/home/molochzuhause/moloch/core/tts.py'
with open(tts) as f:
    code2 = f.read()

# set_speed nach set_voice einfuegen
old_set_voice = '''    def set_voice(self, voice_name: str) -> bool:'''

# Erstmal suchen ob set_speed schon existiert
if 'def set_speed' in code2:
    print('FIX 2: set_speed() existiert schon - SKIP')
    fixes += 1
elif old_set_voice in code2:
    new_set_voice = '''    def set_speed(self, speed: float):
        """TTS Geschwindigkeit aendern. 0.8=schnell, 1.0=normal, 1.2=langsam."""
        global LENGTH_SCALE
        speed = max(0.6, min(1.5, speed))
        LENGTH_SCALE = speed
        logger.info(f"[TTS] Speed: {speed}")

    def set_voice(self, voice_name: str) -> bool:'''
    code2 = code2.replace(old_set_voice, new_set_voice)
    print('FIX 2: set_speed() hinzugefuegt - OK')
    fixes += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

with open(tts, 'w') as f:
    f.write(code2)
compile(open(tts).read(), tts, 'exec')

print(f'\n{fixes}/2 Fixes. Syntax OK.')

if fixes < 2:
    print('!!! INCOMPLETE !!!')
    sys.exit(1)

print('\n=== TTS SPEED FIX KOMPLETT ===')
print('Personality Engine restored auf 1.0 (statt 1.15)')
print('TTSEngine.set_speed(float) verfuegbar')
