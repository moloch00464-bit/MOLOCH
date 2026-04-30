"""
Expression-Module — Welle 16: HARDWARE ALS AUSDRUCK

Hardware spiegelt Molochs INNEREN ZUSTAND wider, nicht nur Funktion:
- Luefter dreht hoch wenn Tension hoch (zusaetzlich zu thermal)
- LED-Pulse-Pattern nach Zone (guardian/shadow/berserker/sleeping)
- Spotify-Bias bei Mood-Wechsel
- TTS-Volume nach Tension (leise wenn calm, laut bei berserker)
- Berserker-Strobo bei Mode-Uebergang

Module subscriben EventBus-Events und rufen Hardware-APIs (best-effort).
KEIN Edit von core/hardware/* — nur deren APIs anrufen.
KEIN Brechen von thermal_manager-Schutz — wir ergaenzen NUR.
"""

__all__ = [
    "tension_to_fan",
    "mood_to_spotify",
    "zone_to_led",
    "berserker_strobo",
    "tension_to_tts_volume",
    "expression_orchestrator",
]
