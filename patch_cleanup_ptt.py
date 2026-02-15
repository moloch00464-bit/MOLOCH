#!/usr/bin/env python3
"""Cleanup: push_to_talk Referenzen aus aktivem Code entfernen."""

# hailo_manager.py: Process protection list bereinigen
hm_path = "/home/molochzuhause/moloch/core/hardware/hailo_manager.py"
with open(hm_path, "r") as f:
    code = f.read()

old = """        (push_to_talk, hailo_control_panel, moloch_service, etc.)
        so we don't accidentally kill our own PTT or GUI."""

new = """        (moloch_service, moloch_unified_panel, etc.)
        so we don't accidentally kill our own service or GUI."""

if old in code:
    code = code.replace(old, new, 1)
    print("hailo_manager: Kommentar aktualisiert")
else:
    print("WARN: Kommentar-Block nicht gefunden")

old_markers = '''            return any(marker in cmdline for marker in [
                "moloch", "push_to_talk", "hailo_control_panel",
            ])'''

new_markers = '''            return any(marker in cmdline for marker in [
                "moloch", "unified_panel",
            ])'''

if old_markers in code:
    code = code.replace(old_markers, new_markers, 1)
    with open(hm_path, "w") as f:
        f.write(code)
    print("hailo_manager: Marker-Liste bereinigt")
else:
    print("WARN: Marker-Liste nicht gefunden")

print("\nDone: push_to_talk Referenzen bereinigt")
print("  push_to_talk.py -> backup/push_to_talk.py.bak")
print("  run_ptt.sh -> backup/run_ptt.sh.bak")
print("  hailo_manager.py: process markers aktualisiert")
print("  hailo_control_panel.py: AUCH BACKUP, nicht angefasst")
