#!/usr/bin/env python3
"""Fix: Hysterese blockiert Hand-Occlusion Swap.

Root Cause:
  entering_worst(pose=1.05) - leaving_best(scrfd=1.1) = -0.05 < 0.15
  → Swap wird geblockt obwohl hand_occlusion = True

Fix: Hysterese bei hand_occlusion ueberspringen.
     Hand-Occlusion ist event-basiert, nicht graduell.
"""
import sys

pe_path = '/home/molochzuhause/moloch/core/perception_engine.py'
with open(pe_path, 'r') as f:
    code = f.read()

fixes_ok = 0

# === Fix 1: Hysterese bei Occlusion ueberspringen ===
old_hysteresis = """        # Hysterese: Eintretende muessen deutlich besser sein als Gehende
        leaving = set(self.slots) - set(new_slots)
        entering = set(new_slots) - set(self.slots)

        if leaving and entering:
            leaving_best = max(scores.get(m, 0) for m in leaving)
            entering_worst = min(scores.get(m, 0) for m in entering)
            if entering_worst - leaving_best < self._hysteresis:
                return None"""

new_hysteresis = """        # Hysterese: Eintretende muessen deutlich besser sein als Gehende
        # Bei Hand-Occlusion: Skip (Event-basierter Swap, nicht graduell)
        if not self._hand_occlusion:
            leaving = set(self.slots) - set(new_slots)
            entering = set(new_slots) - set(self.slots)

            if leaving and entering:
                leaving_best = max(scores.get(m, 0) for m in leaving)
                entering_worst = min(scores.get(m, 0) for m in entering)
                if entering_worst - leaving_best < self._hysteresis:
                    return None"""

if old_hysteresis in code:
    code = code.replace(old_hysteresis, new_hysteresis)
    print('FIX 1: Hysterese-Skip bei Occlusion - OK')
    fixes_ok += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# === Fix 2: Debug-Logging fuer Swap-Entscheidungen ===
old_swap = """        # Swap!
        for m in leaving:
            self._last_active[m] = now
        self._last_rotation = now
        self.slots = new_slots
        return list(new_slots)"""

new_swap = """        # Swap!
        leaving = set(self.slots) - set(new_slots)
        for m in leaving:
            self._last_active[m] = now
        self._last_rotation = now
        self.slots = new_slots
        import logging
        logging.getLogger("PerceptionEngine").info(
            f"[SWAP] {list(leaving)} -> {list(set(new_slots) - set(self.slots) if hasattr(self, '_prev_slots') else new_slots)} "
            f"occlusion={self._hand_occlusion} scores={{{', '.join(f'{k}:{v:.2f}' for k,v in scores.items())}}}")
        return list(new_slots)"""

if old_swap in code:
    code = code.replace(old_swap, new_swap)
    print('FIX 2: Debug-Logging fuer Swap - OK')
    fixes_ok += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

# === Fix 3: _last_scores speichern (fuer Status-JSON) ===
old_scores_save = """        # Scores berechnen
        scores = self._compute_scores(context)
        self._scores = scores"""

new_scores_save = """        # Scores berechnen
        scores = self._compute_scores(context)
        self._scores = scores
        self._last_scores = scores"""

if old_scores_save in code:
    code = code.replace(old_scores_save, new_scores_save)
    print('FIX 3: _last_scores speichern - OK')
    fixes_ok += 1
else:
    # Vielleicht schon vorhanden
    if "self._last_scores = scores" in code:
        print('FIX 3: Bereits vorhanden - SKIP')
        fixes_ok += 1
    else:
        print('FIX 3: ANCHOR NOT FOUND!')

with open(pe_path, 'w') as f:
    f.write(code)

print(f'\n{fixes_ok}/3 Fixes erfolgreich.')
if fixes_ok < 3:
    sys.exit(1)
