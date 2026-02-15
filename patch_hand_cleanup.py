#!/usr/bin/env python3
"""Entferne Debug-Logging, behalte Fix."""
import sys

svc_path = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc_path, 'r') as f:
    code = f.read()

fixes = 0

# Debug 1 entfernen
old1 = """                        if self.hand_active and "hand_landmark" in self._active_ctx:
                            logger.info(f"[HAND_LM] poses={len(poses)}, hand_active={self.hand_active}")
                            for _pose in poses[:1]:"""
new1 = """                        if self.hand_active and "hand_landmark" in self._active_ctx:
                            for _pose in poses[:1]:"""
if old1 in code:
    code = code.replace(old1, new1)
    fixes += 1

# Debug 2 entfernen
old2 = """                                    logger.info(f"[HAND_LM] Wrist {_wi}: vis={_wvis:.2f}, pos=({_wx:.0f},{_wy:.0f})")
                                    if _wvis < 0.3:
                                        continue"""
new2 = """                                    if _wvis < 0.3:
                                        continue"""
if old2 in code:
    code = code.replace(old2, new2)
    fixes += 1

# Debug 3 entfernen
old3 = """                                    logger.info(f"[HAND_LM] decode result: {'presence=' + str(round(_hand_res['presence'],2)) + ' hand=' + _hand_res['handedness'] if _hand_res else 'None'}, allowed_draws={_allowed_draws}")
                                    if _hand_res and "hand" in _allowed_draws:"""
new3 = """                                    if _hand_res and "hand" in _allowed_draws:"""
if old3 in code:
    code = code.replace(old3, new3)
    fixes += 1

with open(svc_path, 'w') as f:
    f.write(code)
print(f'{fixes}/3 Debug-Logs entfernt.')
