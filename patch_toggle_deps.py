#!/usr/bin/env python3
"""Fix: toggle_model max-2 Loop behaelt Dependencies.

Bug: keep={hand_landmark} -> Loop entfernt pose -> Post-check wirft hand_landmark raus.
Fix: keep muss IMMER Dependencies enthalten.
"""
import sys

svc_path = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc_path, 'r') as f:
    code = f.read()

fixes_ok = 0

# === Fix 1: keep-Set mit allen Dependencies ===
old_loop = """            # Max 2 Modelle: neues Modell hat Prioritaet, aelteste raus
            while len(wanted) > 2:
                # model_key und seine Abhaengigkeiten behalten, Rest weg
                keep = {model_key}
                if model_key == "arcface":
                    keep.add("scrfd")
                removable = wanted - keep
                if removable:
                    wanted.discard(removable.pop())
                else:
                    break
            # Post-loop: Dependencies validieren
            if "arcface" in wanted and "scrfd" not in wanted:
                wanted.discard("arcface")
            if "hand_landmark" in wanted and "pose" not in wanted:
                wanted.discard("hand_landmark")
                logger.info(f"[TOGGLE] arcface ohne scrfd entfernt -> wanted={wanted}")
            logger.info(f"[TOGGLE] wanted={wanted} (max 2 enforced)")"""

new_loop = """            # Max 2 Modelle: neues Modell + Dependencies behalten, Rest weg
            keep = {model_key}
            # Dependencies in keep aufnehmen
            DEPS = {"arcface": "scrfd", "hand_landmark": "pose"}
            if model_key in DEPS:
                keep.add(DEPS[model_key])
            while len(wanted) > 2:
                removable = wanted - keep
                if removable:
                    wanted.discard(removable.pop())
                else:
                    break
            # Post-loop: Dependencies nochmal validieren (Sicherheit)
            if "arcface" in wanted and "scrfd" not in wanted:
                wanted.discard("arcface")
                logger.info(f"[TOGGLE] arcface ohne scrfd entfernt -> wanted={wanted}")
            if "hand_landmark" in wanted and "pose" not in wanted:
                wanted.discard("hand_landmark")
                logger.info(f"[TOGGLE] hand_landmark ohne pose entfernt -> wanted={wanted}")
            logger.info(f"[TOGGLE] wanted={wanted} (max 2 enforced)")"""

if old_loop in code:
    code = code.replace(old_loop, new_loop)
    print('FIX 1: keep-Set mit Dependencies - OK')
    fixes_ok += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

with open(svc_path, 'w') as f:
    f.write(code)

print(f'\n{fixes_ok}/1 Fixes erfolgreich.')
if fixes_ok < 1:
    sys.exit(1)
