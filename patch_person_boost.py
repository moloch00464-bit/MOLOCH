#!/usr/bin/env python3
"""Fix: Person-Boost Prioritaet - YOLO hoeher, Pose niedriger."""
import sys

pe_path = "/home/molochzuhause/moloch/core/perception_engine.py"
with open(pe_path, "r") as f:
    code = f.read()

old = """        if person:
            scores["pose"] += 0.3
            scores["yolov8m"] += 0.1"""

new = """        if person:
            scores["yolov8m"] += 0.2
            scores["pose"] += 0.15"""

if old in code:
    code = code.replace(old, new, 1)
    with open(pe_path, "w") as f:
        f.write(code)
    print("OK: yolov8m +0.2 (war +0.1), pose +0.15 (war +0.3)")
else:
    print("ERROR: Block nicht gefunden")
    sys.exit(1)
