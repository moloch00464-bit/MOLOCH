#!/usr/bin/env python3
"""Fix: Inference-Flags mit korrekten String-Quotes."""
path = "/home/molochzuhause/moloch/core/moloch_service.py"
with open(path, "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "# Inference-Flags synchronisieren" in line:
        print(f"Gefunden auf Zeile {i+1}")
        lines[i+1] = '                        self.scrfd_active = "scrfd" in self._active_ctx\n'
        lines[i+2] = '                        self.arcface_active = "arcface" in self._active_ctx\n'
        lines[i+3] = '                        self.yolo_active = "yolov8m" in self._active_ctx\n'
        lines[i+4] = '                        self.pose_active = "pose" in self._active_ctx\n'
        lines[i+5] = '                        self._notify("model_toggle", {\n'
        lines[i+6] = '                            "scrfd": self.scrfd_active, "arcface": self.arcface_active,\n'
        lines[i+7] = '                            "yolov8m": self.yolo_active, "pose": self.pose_active})\n'
        print("Zeilen ersetzt")
        break

with open(path, "w") as f:
    f.writelines(lines)

import py_compile
py_compile.compile(path, doraise=True)
print("Syntax OK")
