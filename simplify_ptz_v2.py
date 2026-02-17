#!/usr/bin/env python3
"""PTZ Simplification - Careful Version"""

import subprocess
import time

panel_path = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'

print("="*60)
print("PTZ SIMPLIFICATION v2")
print("="*60)

# STEP 1: Remove ST button line + pack line
print("\n1. ST Button entfernen...")
subprocess.run(['sed', '-i', '/self.st_btn = tk.Button.*text="ST"/,+2d', panel_path])
print("  ST button definition removed")

# STEP 2: Remove st_btn.pack line if it still exists
subprocess.run(['sed', '-i', '/self.st_btn.pack/d', panel_path])

# STEP 3: Change auto_btn text to AUTONOM, bg to green
print("\n2. auto_btn -> AUTONOM (green)...")
subprocess.run(['sed', '-i', 's/text="MANUELL", bg="#1a1a3e"/text="AUTONOM", bg="#00aa00"/', panel_path])
subprocess.run(['sed', '-i', 's/width=9/width=12/', panel_path], check=False)
print("  auto_btn updated")

# STEP 4: Remove all st_btn.config lines
print("\n3. st_btn.config Zeilen entfernen...")
subprocess.run(['sed', '-i', '/self.st_btn.config/d', panel_path])
print("  st_btn updates removed")

# STEP 5: Add _set_ptz_state method (insert before _toggle_autonomous if exists)
print("\n4. _set_ptz_state() hinzufügen...")

ptz_method = '''    def _set_ptz_state(self, state):
        """Enable/Disable PTZ buttons based on mode."""
        try:
            # Find PTZ frame and disable/enable all buttons in it
            for widget in self.root.winfo_children():
                self._set_ptz_recursive(widget, state)
        except Exception as e:
            logger.error(f"PTZ state change error: {e}")

    def _set_ptz_recursive(self, widget, state):
        """Recursively set PTZ button states."""
        import tkinter as tk
        if isinstance(widget, tk.Button):
            text = widget.cget('text')
            if text in ['^', 'v', '<', '>', 'H']:
                widget.config(state=state)
        try:
            for child in widget.winfo_children():
                self._set_ptz_recursive(child, state)
        except:
            pass

'''

# Insert before _toggle_autonomous
with open(panel_path) as f:
    lines = f.readlines()

# Find _toggle_autonomous
for i, line in enumerate(lines):
    if '    def _toggle_autonomous' in line:
        # Check if _set_ptz_state already there
        if i > 0 and 'def _set_ptz_state' in ''.join(lines[max(0,i-30):i]):
            print("  _set_ptz_state already exists")
        else:
            lines.insert(i, ptz_method)
            print(f"  _set_ptz_state inserted at line {i+1}")
        break

with open(panel_path, 'w') as f:
    f.writelines(lines)

# STEP 6: Simplify _toggle_autonomous
print("\n5. _toggle_autonomous() vereinfachen...")

with open(panel_path) as f:
    code = f.read()

# Find and replace _toggle_autonomous method
import re

old_method = r'    def _toggle_autonomous\(self\):.*?(?=\n    def [a-z_]|\nclass )'

new_method = '''    def _toggle_autonomous(self):
        """Toggle AUTONOM/MANUELL."""
        import tkinter as tk
        if not self.service:
            return

        # Current state
        current_auto = getattr(self.service, '_autonomous_mode', True)
        new_auto = not current_auto

        # Notify service
        if isinstance(self.service, ServiceProxy):
            self.service._send_cmd({"action": "toggle_autonomous"})
            self.service._autonomous_mode = new_auto
        else:
            if hasattr(self.service, 'toggle_autonomous_manual'):
                self.service.toggle_autonomous_manual()

        # Update button + PTZ state
        if new_auto:  # AUTONOM
            self.auto_btn.config(text="AUTONOM", bg="#00aa00")
            self._set_ptz_state(tk.DISABLED)
            logger.info("[MODE] AUTONOM: PTZ gesperrt")
        else:  # MANUELL
            self.auto_btn.config(text="MANUELL", bg="#dd2222")
            self._set_ptz_state(tk.NORMAL)
            logger.info("[MODE] MANUELL: PTZ aktiv")

'''

code = re.sub(old_method, new_method, code, flags=re.DOTALL, count=1)

with open(panel_path, 'w') as f:
    f.write(code)

print("  _toggle_autonomous simplified")

# STEP 7: Remove _toggle_smart_tracking method
print("\n6. _toggle_smart_tracking() entfernen...")
with open(panel_path) as f:
    lines = f.readlines()

new_lines = []
skip = 0
for i, line in enumerate(lines):
    if skip > 0:
        skip -= 1
        continue

    if '    def _toggle_smart_tracking' in line:
        # Find next def
        j = i + 1
        while j < len(lines):
            if lines[j].startswith('    def ') and j != i:
                break
            j += 1
        skip = j - i - 1
        print(f"  Removed _toggle_smart_tracking at line {i+1}")
        continue

    new_lines.append(line)

with open(panel_path, 'w') as f:
    f.writelines(new_lines)

# TEST
print("\n" + "="*60)
print("SYNTAX CHECK")
print("="*60)

try:
    subprocess.run(['python3', '-m', 'py_compile', panel_path],
                  check=True, capture_output=True)
    print("✅ Syntax OK")
except subprocess.CalledProcessError as e:
    print("❌ SYNTAX ERROR!")
    err = e.stderr.decode() if e.stderr else ""
    print(err)
    subprocess.run(['git', 'checkout', panel_path])
    exit(1)

# Service restart
print("\nService Neustart...")
subprocess.run(['sudo', 'systemctl', 'restart', 'moloch.service'])
time.sleep(3)

result = subprocess.run(['systemctl', 'is-active', 'moloch.service'],
                      capture_output=True, text=True)
if result.stdout.strip() == 'active':
    print("✅ Service läuft")
else:
    print("❌ Service ERROR")
    exit(1)

print("\n" + "="*60)
print("✅ FERTIG!")
print("="*60)
print("\n>>> PANEL NEUSTARTEN <<<")
print("\nTEST:")
print("- Button zeigt 'AUTONOM' (grün)?")
print("- Drücken -> wechselt zu 'MANUELL' (rot)?")
print("- Im MANUELL: PTZ Buttons funktionieren?")
print("- Im AUTONOM: PTZ Buttons grau?")
