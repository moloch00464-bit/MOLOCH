#!/usr/bin/env python3
"""Simplify PTZ Control: 3 Buttons -> 1 Button

ÄNDERUNGEN:
1. LÖSCHE: ST Button, separate MANUELL/AUTONOM Logik
2. EIN Button: "AUTONOM" (grün) / "MANUELL" (rot)
3. AUTONOM: PTZ disabled (grau), MOLOCH steuert
4. MANUELL: PTZ enabled, Markus steuert
"""

import subprocess
import time
import re

def remove_st_button():
    """Remove ST button completely"""
    print("\n" + "="*60)
    print("STEP 1: ST Button entfernen")
    print("="*60)

    with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py') as f:
        lines = f.readlines()

    new_lines = []
    skip_next = 0
    removed = 0

    for i, line in enumerate(lines):
        if skip_next > 0:
            skip_next -= 1
            continue

        # Remove ST button creation
        if 'self.st_btn = tk.Button' in line and 'text="ST' in line:
            # Skip this line and next (pack line)
            skip_next = 1
            removed += 1
            print(f"  Zeile {i+1}: ST Button Definition entfernt")
            continue

        # Remove ST button updates
        if 'self.st_btn.config' in line:
            removed += 1
            print(f"  Zeile {i+1}: ST Button Update entfernt")
            continue

        # Remove _toggle_smart_tracking method
        if 'def _toggle_smart_tracking' in line:
            # Skip entire method (find next def)
            j = i + 1
            while j < len(lines) and not (lines[j].startswith('    def ') and lines[j] != line):
                j += 1
            skip_next = j - i - 1
            removed += 1
            print(f"  Zeile {i+1}-{j}: _toggle_smart_tracking() entfernt")
            continue

        new_lines.append(line)

    with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py', 'w') as f:
        f.writelines(new_lines)

    print(f"✓ ST Button komplett entfernt ({removed} Änderungen)")
    return True

def simplify_auto_button():
    """Simplify auto button to simple AUTONOM/MANUELL toggle"""
    print("\n" + "="*60)
    print("STEP 2: AUTONOM Button vereinfachen")
    print("="*60)

    with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py') as f:
        lines = f.readlines()

    # Find auto_btn definition and change it
    for i, line in enumerate(lines):
        if 'self.auto_btn = tk.Button' in line:
            # Change to default AUTONOM state (green)
            lines[i] = '        self.auto_btn = tk.Button(brow, text="AUTONOM", bg="#00aa00", fg="white", width=12,\n'
            lines[i+1] = '                                  font=("Helvetica", 11, "bold"),\n'
            lines[i+2] = '                                  command=self._toggle_autonomous)\n'
            print(f"  Zeile {i+1}: auto_btn zu AUTONOM/MANUELL Toggle geändert")
            break

    with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py', 'w') as f:
        f.writelines(lines)

    print("✓ auto_btn vereinfacht")
    return True

def update_toggle_logic():
    """Update _toggle_autonomous to simple toggle + PTZ enable/disable"""
    print("\n" + "="*60)
    print("STEP 3: Toggle-Logik vereinfachen")
    print("="*60)

    with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py') as f:
        code = f.read()

    # Find _toggle_autonomous and replace entire method
    pattern = r'    def _toggle_autonomous\(self\):.*?(?=\n    def |\nclass |\Z)'

    new_method = '''    def _toggle_autonomous(self):
        """Toggle AUTONOM/MANUELL - vereinfachte Version."""
        if not self.service:
            return

        # Aktuellen Zustand lesen
        current_auto = getattr(self.service, '_autonomous_mode', True)
        new_auto = not current_auto

        # Service informieren
        if isinstance(self.service, ServiceProxy):
            # Via IPC
            self.service._send_cmd({"action": "toggle_autonomous"})
            self.service._autonomous_mode = new_auto
        else:
            # Direct mode
            if hasattr(self.service, 'toggle_autonomous_manual'):
                self.service.toggle_autonomous_manual()
                new_auto = not self.service._manual_mode

        # Button Update
        if new_auto:  # AUTONOM
            self.auto_btn.config(text="AUTONOM", bg="#00aa00")
            # PTZ Buttons DEAKTIVIEREN
            self._set_ptz_state(tk.DISABLED)
            logger.info("[MODE] AUTONOM: MOLOCH steuert, PTZ gesperrt")
        else:  # MANUELL
            self.auto_btn.config(text="MANUELL", bg="#dd2222")
            # PTZ Buttons AKTIVIEREN
            self._set_ptz_state(tk.NORMAL)
            logger.info("[MODE] MANUELL: Markus steuert, PTZ aktiv")
'''

    if re.search(pattern, code, re.DOTALL):
        code = re.sub(pattern, new_method, code, flags=re.DOTALL)
        print("  _toggle_autonomous() ersetzt")
    else:
        print("  WARNUNG: _toggle_autonomous nicht gefunden!")

    with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py', 'w') as f:
        f.write(code)

    print("✓ Toggle-Logik vereinfacht")
    return True

def add_ptz_state_method():
    """Add method to enable/disable PTZ buttons"""
    print("\n" + "="*60)
    print("STEP 4: PTZ State Control hinzufügen")
    print("="*60)

    with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py') as f:
        lines = f.readlines()

    # Check if _set_ptz_state already exists
    if any('def _set_ptz_state' in line for line in lines):
        print("  _set_ptz_state() bereits vorhanden")
        return True

    # Find _toggle_autonomous and add _set_ptz_state before it
    for i, line in enumerate(lines):
        if 'def _toggle_autonomous' in line:
            method = '''    def _set_ptz_state(self, state):
        """Enable/Disable PTZ control buttons."""
        # Find all PTZ buttons (up, down, left, right, home)
        for widget in self.root.winfo_children():
            self._set_ptz_recursive(widget, state)

    def _set_ptz_recursive(self, widget, state):
        """Recursively find and set PTZ button states."""
        if isinstance(widget, tk.Button):
            text = widget.cget('text')
            if text in ['^', 'v', '<', '>', 'H']:  # PTZ buttons
                widget.config(state=state)
        for child in widget.winfo_children():
            self._set_ptz_recursive(child, state)

'''
            lines.insert(i, method)
            print(f"  Zeile {i+1}: _set_ptz_state() hinzugefügt")
            break

    with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py', 'w') as f:
        f.writelines(lines)

    print("✓ PTZ State Control hinzugefügt")
    return True

def update_button_updates():
    """Remove old button update logic in _update_fps"""
    print("\n" + "="*60)
    print("STEP 5: Button-Update-Logik bereinigen")
    print("="*60)

    with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py') as f:
        lines = f.readlines()

    new_lines = []
    skip = False
    removed = 0

    for i, line in enumerate(lines):
        # Remove old auto_btn update blocks
        if 'Update AUTONOM Button' in line or 'Update Mode Label' in line:
            # Skip this comment and following if block
            skip = True
            removed += 1
            print(f"  Zeile {i+1}: Altes Button-Update entfernt")
            continue

        if skip:
            if line.strip().startswith('if ') or line.strip().startswith('try:'):
                # Count braces/indentation to know when block ends
                pass
            if 'self.root.after(' in line or 'def ' in line:
                skip = False

        if not skip:
            new_lines.append(line)

    with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py', 'w') as f:
        f.writelines(new_lines)

    print(f"✓ Update-Logik bereinigt ({removed} Blöcke)")
    return True

def run_tests():
    """Test compilation and service"""
    print("\n" + "="*60)
    print("TESTS")
    print("="*60)

    # Syntax check
    try:
        subprocess.run(['python3', '-m', 'py_compile',
                       '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'],
                      check=True, capture_output=True)
        print("✅ Panel Syntax OK")
    except subprocess.CalledProcessError as e:
        print("❌ PANEL SYNTAX ERROR!")
        print(e.stderr.decode() if e.stderr else "")
        subprocess.run(['git', 'checkout', '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'])
        return False

    # Service restart
    print("\nService Neustart...")
    subprocess.run(['sudo', 'systemctl', 'restart', 'moloch.service'], check=True)
    time.sleep(3)

    result = subprocess.run(['systemctl', 'is-active', 'moloch.service'],
                          capture_output=True, text=True)
    if result.stdout.strip() == 'active':
        print("✅ Service läuft")
    else:
        print("❌ Service ERROR!")
        return False

    return True

def main():
    print("\n" + "="*70)
    print("PTZ CONTROL SIMPLIFICATION")
    print("3 Buttons -> 1 Button")
    print("="*70)

    if not remove_st_button():
        return False
    if not simplify_auto_button():
        return False
    if not add_ptz_state_method():
        return False
    if not update_toggle_logic():
        return False
    if not update_button_updates():
        return False

    if not run_tests():
        print("\n❌ TESTS FAILED!")
        return False

    print("\n" + "="*70)
    print("✅ VEREINFACHUNG KOMPLETT!")
    print("="*70)
    print("\nJETZT:")
    print("1. Panel neustarten")
    print("2. Button testen:")
    print("   - AUTONOM (grün): PTZ Buttons grau/gesperrt")
    print("   - MANUELL (rot): PTZ Buttons aktiv")
    print("3. Logs checken: autonomous_mode wechselt?")
    return True

if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
