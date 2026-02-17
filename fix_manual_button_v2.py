#!/usr/bin/env python3
"""Manual-Button Fix - Saubere Python-basierte Implementation"""

import re
import subprocess
import time
import json

def fix_service():
    """Add manual_mode to status JSON in service"""
    print("\n" + "="*60)
    print("FIX 1: Service - manual_mode in Status-JSON")
    print("="*60)

    with open('/home/molochzuhause/moloch/core/moloch_service.py') as f:
        lines = f.readlines()

    # Find line with "autonomous_mode": self._autonomous_mode,
    for i, line in enumerate(lines):
        if '"autonomous_mode": self._autonomous_mode,' in line:
            # Check if manual_mode already added
            if i+1 < len(lines) and '"manual_mode"' in lines[i+1]:
                print("✓ manual_mode bereits vorhanden")
                return True
            # Insert after autonomous_mode line
            indent = ' ' * 16
            lines.insert(i+1, f'{indent}"manual_mode": self._manual_mode,\n')
            print(f"✓ manual_mode bei Zeile {i+1} eingefuegt")
            break
    else:
        print("✗ Anchor nicht gefunden!")
        return False

    with open('/home/molochzuhause/moloch/core/moloch_service.py', 'w') as f:
        f.writelines(lines)
    return True

def fix_panel_proxy_init():
    """Add _manual_mode to ServiceProxy.__init__"""
    print("\n" + "="*60)
    print("FIX 2: Panel - _manual_mode in ServiceProxy")
    print("="*60)

    with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py') as f:
        lines = f.readlines()

    # Find _tentakel_enabled = False in ServiceProxy
    for i, line in enumerate(lines):
        if 'self._tentakel_enabled = False' in line and 'ServiceProxy' in ''.join(lines[max(0,i-50):i]):
            # Check if already added
            if i+1 < len(lines) and '_manual_mode' in lines[i+1]:
                print("✓ _manual_mode bereits in __init__")
                return True
            # Add after _tentakel_enabled
            indent = '        '
            lines.insert(i+1, f'{indent}self._manual_mode = False  # Remote state\n')
            print(f"✓ _manual_mode bei Zeile {i+1} eingefuegt")
            break
    else:
        print("✗ Anchor nicht gefunden!")
        return False

    with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py', 'w') as f:
        f.writelines(lines)
    return True

def fix_panel_apply_status():
    """Read manual_mode from status in _apply_status"""
    print("\n" + "="*60)
    print("FIX 3: Panel - manual_mode aus Status lesen")
    print("="*60)

    with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py') as f:
        lines = f.readlines()

    # Find: self._tentakel_enabled = s.get('tentakel_enabled', False)
    for i, line in enumerate(lines):
        if "self._tentakel_enabled = s.get('tentakel_enabled', False)" in line:
            # Check if already added
            if i+1 < len(lines) and '_manual_mode' in lines[i+1]:
                print("✓ manual_mode bereits gelesen")
                return True
            # Add after tentakel_enabled
            indent = '        '
            lines.insert(i+1, f'{indent}self._manual_mode = s.get("manual_mode", False)\n')
            print(f"✓ manual_mode read bei Zeile {i+1} eingefuegt")
            break
    else:
        print("✗ Anchor nicht gefunden!")
        return False

    with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py', 'w') as f:
        f.writelines(lines)
    return True

def fix_panel_button_update():
    """Add AUTONOM button color update in _update_fps"""
    print("\n" + "="*60)
    print("FIX 4: Panel - AUTONOM Button Update in _update_fps")
    print("="*60)

    with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py') as f:
        lines = f.readlines()

    # Find _update_fps and the line before self.root.after
    for i, line in enumerate(lines):
        if 'self.root.after(500, self._update_fps)' in line:
            # Check if already added
            for j in range(max(0, i-20), i):
                if 'Update AUTONOM Button' in lines[j]:
                    print("✓ AUTONOM Button Update bereits vorhanden")
                    return True

            # Insert before self.root.after
            button_code = '''
        # Update AUTONOM Button from service state
        try:
            if hasattr(self.service, "_manual_mode"):
                manual = self.service._manual_mode
                self.auto_btn.config(
                    text="MANUELL" if manual else "AUTONOM",
                    bg="#00aa00" if manual else "#dd2222"
                )
        except Exception:
            pass

'''
            lines.insert(i, button_code)
            print(f"✓ AUTONOM Button Update bei Zeile {i} eingefuegt")
            break
    else:
        print("✗ Anchor nicht gefunden!")
        return False

    with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py', 'w') as f:
        f.writelines(lines)
    return True

def run_tests():
    """Run automated tests"""
    print("\n" + "="*60)
    print("TESTS")
    print("="*60)

    # Test: Service running?
    result = subprocess.run(['systemctl', 'is-active', 'moloch.service'],
                          capture_output=True, text=True)
    if result.stdout.strip() == 'active':
        print("✅ TEST A: Service laeuft")
    else:
        print("❌ TEST A: Service NICHT aktiv!")
        return False

    # Test: manual_mode in status?
    time.sleep(2)
    try:
        with open('/dev/shm/moloch_status.json') as f:
            status = json.load(f)
        if 'manual_mode' in status:
            print(f"✅ TEST B: manual_mode im Status = {status['manual_mode']}")
        else:
            print("❌ TEST B: manual_mode FEHLT im Status!")
            return False
    except Exception as e:
        print(f"❌ TEST B: Fehler: {e}")
        return False

    # Test: toggle method exists?
    result = subprocess.run(['grep', '-q', 'def toggle_autonomous_manual',
                           '/home/molochzuhause/moloch/core/moloch_service.py'])
    if result.returncode == 0:
        print("✅ TEST C: toggle_autonomous_manual() existiert")
    else:
        print("❌ TEST C: Methode FEHLT!")
        return False

    return True

def main():
    print("\n" + "="*70)
    print("MANUAL-BUTTON FIX (Python-basiert)")
    print("="*70)

    # Apply fixes
    if not fix_service():
        return False
    if not fix_panel_proxy_init():
        return False
    if not fix_panel_apply_status():
        return False
    if not fix_panel_button_update():
        return False

    # Syntax check
    print("\n" + "="*60)
    print("SYNTAX CHECK")
    print("="*60)
    try:
        subprocess.run(['python3', '-m', 'py_compile',
                       '/home/molochzuhause/moloch/core/moloch_service.py'],
                      check=True, capture_output=True)
        print("✓ Service Syntax OK")
    except:
        print("✗ SERVICE SYNTAX ERROR!")
        subprocess.run(['git', 'checkout', '/home/molochzuhause/moloch/core/moloch_service.py'])
        return False

    try:
        subprocess.run(['python3', '-m', 'py_compile',
                       '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'],
                      check=True, capture_output=True)
        print("✓ Panel Syntax OK")
    except:
        print("✗ PANEL SYNTAX ERROR!")
        subprocess.run(['git', 'checkout', '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'])
        return False

    # Restart service
    print("\n" + "="*60)
    print("SERVICE NEUSTART")
    print("="*60)
    subprocess.run(['sudo', 'systemctl', 'restart', 'moloch.service'], check=True)
    print("✓ Service neu gestartet")

    # Run tests
    if not run_tests():
        print("\n❌ TESTS FAILED!")
        return False

    print("\n" + "="*70)
    print("✅ ALLE TESTS BESTANDEN!")
    print("="*70)
    print("\n>>> Panel NEUSTART erforderlich! <<<\n")
    return True

if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
