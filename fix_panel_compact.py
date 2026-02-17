#!/usr/bin/env python3
"""Fix: Panel kompakter - Luecken/Padding reduzieren."""
import sys

panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    code = f.read()

fixes = 0

# Systematisch grosse Paddings reduzieren
replacements = [
    # Hauptlayout
    ('pady=(5, 0))', 'pady=(2, 0))'),
    ('pady=(0, 6))', 'pady=(0, 2))'),
    ('pady=(8, 4))', 'pady=(3, 1))'),
    ('pady=(6, 2))', 'pady=(2, 1))'),
    ('pady=(6, 0))', 'pady=(2, 0))'),
    # Tab-Inhalte: padx=5, pady=3 -> padx=3, pady=1
    ('padx=5, pady=3)', 'padx=3, pady=1)'),
    # padx=(0, 10) fuer Phase-Frames -> (0, 5)
    ('padx=(0, 10))', 'padx=(0, 5))'),
]

for old, new in replacements:
    count = code.count(old)
    if count > 0:
        code = code.replace(old, new)
        print(f'  {old} -> {new}  ({count}x)')
        fixes += count

# Perception Tab interne Paddings
code = code.replace('padx=5, pady=2)', 'padx=3, pady=1)')
code = code.replace('padx=2, pady=2)', 'padx=1, pady=1)')

# Preview canvas Abstand
code = code.replace("pady=(0, 3))", "pady=(0, 1))")

with open(panel, 'w') as f:
    f.write(code)

compile(open(panel).read(), panel, 'exec')
print(f'\n{fixes} Padding-Aenderungen. Syntax OK.')
print('Panel ist jetzt kompakter.')
