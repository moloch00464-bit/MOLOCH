#!/usr/bin/env python3
"""Temporaeres Debug-Logging fuer Hand-Landmark Inference Pipeline."""
import sys

svc_path = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc_path, 'r') as f:
    code = f.read()

# Debug-Logging in hand_landmark Inference Block
old_block = """                        # === Hand Landmark: Crop um Wrists, 21 Finger-Landmarks ===
                        if self.hand_active and "hand_landmark" in self._active_ctx:
                            for _pose in poses[:1]:"""

new_block = """                        # === Hand Landmark: Crop um Wrists, 21 Finger-Landmarks ===
                        if self.hand_active and "hand_landmark" in self._active_ctx:
                            logger.info(f"[HAND_LM] poses={len(poses)}, hand_active={self.hand_active}")
                            for _pose in poses[:1]:"""

if old_block in code:
    code = code.replace(old_block, new_block)
    print('DEBUG 1: Hand-LM entry log - OK')
else:
    print('DEBUG 1: ANCHOR NOT FOUND!')
    sys.exit(1)

# Debug: Wrist confidence
old_wrist = """                                    if _wvis < 0.3:
                                        continue"""

new_wrist = """                                    logger.info(f"[HAND_LM] Wrist {_wi}: vis={_wvis:.2f}, pos=({_wx:.0f},{_wy:.0f})")
                                    if _wvis < 0.3:
                                        continue"""

if old_wrist in code:
    code = code.replace(old_wrist, new_wrist, 1)  # Nur erste Occurrence
    print('DEBUG 2: Wrist vis log - OK')
else:
    print('DEBUG 2: ANCHOR NOT FOUND!')

# Debug: decode result
old_decode = """                                    _hand_res = decode_hand_landmark(_hand_out)
                                    if _hand_res and "hand" in _allowed_draws:"""

new_decode = """                                    _hand_res = decode_hand_landmark(_hand_out)
                                    logger.info(f"[HAND_LM] decode result: {'presence=' + str(round(_hand_res['presence'],2)) + ' hand=' + _hand_res['handedness'] if _hand_res else 'None'}, allowed_draws={_allowed_draws}")
                                    if _hand_res and "hand" in _allowed_draws:"""

if old_decode in code:
    code = code.replace(old_decode, new_decode)
    print('DEBUG 3: Decode result log - OK')
else:
    print('DEBUG 3: ANCHOR NOT FOUND!')

with open(svc_path, 'w') as f:
    f.write(code)
print('Debug-Logging eingebaut.')
