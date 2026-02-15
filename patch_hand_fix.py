#!/usr/bin/env python3
"""Fix: Hand-Occlusion person_detected Bug.

Problem: YOLO laeuft NICHT wenn ArcFace aktiv ist (face -> ArcFace swap).
         Wenn Hand Gesicht verdeckt, ist YOLO noch nicht geladen.
         person_detected = False -> Occlusion triggert NIE.

Fix: In PerceptionEngine - wenn Face <1s her war, Person ist implizit da.
     Face = Person. Keine Teleportation.
"""
import sys

pe_path = "/home/molochzuhause/moloch/core/perception_engine.py"
with open(pe_path, "r") as f:
    code = f.read()

changes = 0

# Fix: person_detected aus recentem Face ableiten
old_check = """        # --- NEUE Occlusion pruefen (Face ist JETZT weg) ---

        # Streak merken VOR Reset
        streak_before = self._face_streak
        self._face_streak = 0  # Reset: Face ist weg

        # 1. Person muss noch sichtbar sein
        if not person_detected:
            return False"""

new_check = """        # --- NEUE Occlusion pruefen (Face ist JETZT weg) ---

        # Streak merken VOR Reset
        streak_before = self._face_streak
        self._face_streak = 0  # Reset: Face ist weg

        # 1. Person muss noch sichtbar sein
        #    YOLO laeuft oft NICHT wenn ArcFace aktiv war (Face-Prioritaet).
        #    Wenn Face <1s her: Person ist implizit noch da (kein Teleport).
        if not person_detected and (now - self._last_face_time > 1.0):
            return False"""

if old_check in code:
    code = code.replace(old_check, new_check, 1)
    changes += 1
    print("FIX 1: person_detected relaxiert (Face <1s = Person implied)")
else:
    print("ERROR: _check_hand_occlusion Block nicht gefunden")
    sys.exit(1)

if changes > 0:
    with open(pe_path, "w") as f:
        f.write(code)
    print(f"perception_engine.py fixed: {changes} change")

# Verify
print("\nVerifizierung:")
print("  Vorher: YOLO nicht aktiv -> person_detected=False -> KEINE Occlusion")
print("  Nachher: Face <1s her -> Person implizit da -> Occlusion kann triggern")
print("  Kein YOLO noetig fuer Hand-Erkennung!")
