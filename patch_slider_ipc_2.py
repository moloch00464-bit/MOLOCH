#!/usr/bin/env python3
"""Nachpatch: 2 fehlende Fixes (Anchor-Korrektur)."""
import sys

fixes_ok = 0

# === Fix 1b: _syncing_thresholds Flag in Panel ===
panel_path = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel_path, 'r') as f:
    panel = f.read()

old_1b = """        self.service = None
        self._syncing = False"""

new_1b = """        self.service = None
        self._syncing = False
        self._syncing_thresholds = False"""

if new_1b in panel:
    print('FIX 1b: Bereits vorhanden - SKIP')
    fixes_ok += 1
elif old_1b in panel:
    panel = panel.replace(old_1b, new_1b, 1)  # Nur 1. Occurrence
    print('FIX 1b: _syncing_thresholds Flag - OK')
    fixes_ok += 1
else:
    print('FIX 1b: ANCHOR NOT FOUND!')

with open(panel_path, 'w') as f:
    f.write(panel)

# === Fix 6: Hand params in perception get_state() ===
pe_path = '/home/molochzuhause/moloch/core/perception_engine.py'
with open(pe_path, 'r') as f:
    pe = f.read()

old_state = """            "hand_occlusion": self._hand_occlusion,
            "face_streak": self._face_streak,
        }"""

new_state = """            "hand_occlusion": self._hand_occlusion,
            "face_streak": self._face_streak,
            "hand_timeout": self._HAND_TIMEOUT,
            "hand_streak_min": self._MIN_FACE_STREAK,
            "hand_recency": self._FACE_RECENCY,
        }"""

if old_state in pe:
    pe = pe.replace(old_state, new_state)
    print('FIX 6: Hand params in get_state() - OK')
    fixes_ok += 1
else:
    print('FIX 6: ANCHOR NOT FOUND!')

with open(pe_path, 'w') as f:
    f.write(pe)

print(f'\n{fixes_ok}/2 Fixes erfolgreich.')
if fixes_ok < 2:
    sys.exit(1)
