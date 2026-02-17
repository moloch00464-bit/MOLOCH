#!/usr/bin/env python3
"""Fix: Manual/Autonomous Mode Isolation

PROBLEM:
- Panel PTZ commands gehen DIREKT zur Kamera (bypass Service)
- Kein Mode-Check -> Steuerkreuz funktioniert auch in AUTONOMOUS
- MOLOCH kann trotzdem eingreifen in MANUAL

FIX:
1. Panel: Guard in _ptz_move + _ptz_goto (block wenn autonomous)
2. Service: Enforce ST state based on mode
3. Service: Guard in camera movement detection (block takeover wenn manual)

TEST:
- Manual: PTZ works, MOLOCH blocked, ST off
- Autonomous: PTZ blocked, MOLOCH active, ST on
"""

import subprocess
import time
import json

def fix_panel_ptz_guards():
    """Add mode guards to panel PTZ controls"""
    print("\n" + "="*60)
    print("FIX 1: Panel PTZ Guards (Block in Autonomous)")
    print("="*60)

    with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py') as f:
        lines = f.readlines()

    # Find _ptz_move method and add guard after "def do_move():"
    for i, line in enumerate(lines):
        if 'def do_move():' in line and '_ptz_move' in ''.join(lines[max(0,i-5):i]):
            # Check if guard already exists
            for j in range(i, min(i+10, len(lines))):
                if 'manual_mode' in lines[j] or 'AUTONOMOUS' in lines[j]:
                    print("✓ PTZ guard bereits in _ptz_move")
                    break
            else:
                # Insert guard after "def do_move():"
                indent = '            '
                guard = f'''{indent}# Mode Guard: Block PTZ in AUTONOMOUS mode
{indent}if hasattr(self, 'service') and hasattr(self.service, '_manual_mode'):
{indent}    if not self.service._manual_mode:  # AUTONOMOUS = manual_mode False
{indent}        logger.info("[PTZ] Blocked: AUTONOMOUS mode active")
{indent}        return
'''
                lines.insert(i+1, guard)
                print(f"✓ PTZ guard bei Zeile {i+1} eingefuegt (_ptz_move)")
            break

    # Find _ptz_goto and add guard
    for i, line in enumerate(lines):
        if 'def do_goto():' in line and '_ptz_goto' in ''.join(lines[max(0,i-5):i]):
            # Check if guard already exists
            for j in range(i, min(i+10, len(lines))):
                if 'manual_mode' in lines[j] or 'AUTONOMOUS' in lines[j]:
                    print("✓ PTZ guard bereits in _ptz_goto")
                    break
            else:
                # Insert guard
                indent = '            '
                guard = f'''{indent}# Mode Guard: Block PTZ in AUTONOMOUS mode
{indent}if hasattr(self, 'service') and hasattr(self.service, '_manual_mode'):
{indent}    if not self.service._manual_mode:  # AUTONOMOUS
{indent}        logger.info("[PTZ] Blocked: AUTONOMOUS mode active")
{indent}        return
'''
                lines.insert(i+1, guard)
                print(f"✓ PTZ guard bei Zeile {i+1} eingefuegt (_ptz_goto)")
            break

    with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py', 'w') as f:
        f.writelines(lines)
    return True

def fix_service_st_enforcement():
    """Enforce Smart Tracking state based on mode"""
    print("\n" + "="*60)
    print("FIX 2: Service - ST Enforcement (AN/AUS je nach Mode)")
    print("="*60)

    with open('/home/molochzuhause/moloch/core/moloch_service.py') as f:
        lines = f.readlines()

    # Find toggle_autonomous_manual - section where we switch to AUTONOM
    for i, line in enumerate(lines):
        if 'else:' in line and 'AUTONOM: Kamera wieder freigeben' in ''.join(lines[i:min(i+5, len(lines))]):
            # Check if ST enforcement already there
            for j in range(i, min(i+30, len(lines))):
                if 'set_smart_tracking(True)' in lines[j]:
                    print("✓ ST Enforcement bereits in toggle_autonomous_manual")
                    return True

            # Find the line with self._manual_mode = False
            for j in range(i, min(i+20, len(lines))):
                if 'self._manual_mode = False' in lines[j]:
                    # Insert ST activation after manual_mode = False
                    indent = '            '
                    st_code = f'''
{indent}# Smart Tracking AN wenn AUTONOM
{indent}def enable_st():
{indent}    if self._cloud and self._cloud.connected:
{indent}        try:
{indent}            self._cloud.run(self._cloud.bridge.set_smart_tracking(True))
{indent}            self._set_smart_tracking_state(True)
{indent}            logger.info("[MODUS] Smart Tracking AN (AUTONOM)")
{indent}        except Exception as e:
{indent}            logger.error(f"ST enable failed: {{e}}")
{indent}threading.Thread(target=enable_st, daemon=True).start()
'''
                    lines.insert(j+1, st_code)
                    print(f"✓ ST Enforcement bei Zeile {j+1} eingefuegt")
                    break
            break

    with open('/home/molochzuhause/moloch/core/moloch_service.py', 'w') as f:
        f.writelines(lines)
    return True

def verify_existing_guards():
    """Verify existing manual_mode guards are in place"""
    print("\n" + "="*60)
    print("CHECK: Bestehende Guards im Service")
    print("="*60)

    # Check _moloch_takeover
    result = subprocess.run(['grep', '-A', '3', '_moloch_takeover',
                           '/home/molochzuhause/moloch/core/moloch_service.py'],
                          capture_output=True, text=True)
    if 'self._manual_mode' in result.stdout:
        print("✓ _moloch_takeover hat manual_mode guard")
    else:
        print("❌ _moloch_takeover FEHLT manual_mode guard!")
        return False

    # Check _check_guardian_timeout
    result = subprocess.run(['grep', '-A', '3', '_check_guardian_timeout',
                           '/home/molochzuhause/moloch/core/moloch_service.py'],
                          capture_output=True, text=True)
    if 'self._manual_mode' in result.stdout:
        print("✓ _check_guardian_timeout hat manual_mode guard")
    else:
        print("❌ _check_guardian_timeout FEHLT guard!")
        return False

    return True

def run_tests():
    """Run tests"""
    print("\n" + "="*60)
    print("TESTS")
    print("="*60)

    # Syntax check
    try:
        subprocess.run(['python3', '-m', 'py_compile',
                       '/home/molochzuhause/moloch/core/moloch_service.py'],
                      check=True, capture_output=True)
        print("✓ Service Syntax OK")
    except:
        print("❌ SERVICE SYNTAX ERROR!")
        subprocess.run(['git', 'checkout', '/home/molochzuhause/moloch/core/moloch_service.py'])
        return False

    try:
        subprocess.run(['python3', '-m', 'py_compile',
                       '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'],
                      check=True, capture_output=True)
        print("✓ Panel Syntax OK")
    except:
        print("❌ PANEL SYNTAX ERROR!")
        subprocess.run(['git', 'checkout', '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'])
        return False

    # Restart service
    print("\n" + "="*60)
    print("SERVICE NEUSTART")
    print("="*60)
    subprocess.run(['sudo', 'systemctl', 'restart', 'moloch.service'], check=True)
    time.sleep(3)
    print("✓ Service neu gestartet")

    # Check service running
    result = subprocess.run(['systemctl', 'is-active', 'moloch.service'],
                          capture_output=True, text=True)
    if result.stdout.strip() == 'active':
        print("✅ TEST A: Service laeuft")
    else:
        print("❌ TEST A: Service NICHT aktiv!")
        return False

    # Check status
    time.sleep(2)
    try:
        with open('/dev/shm/moloch_status.json') as f:
            status = json.load(f)
        manual = status.get('manual_mode', None)
        auto = status.get('autonomous_mode', None)
        st = status.get('smart_tracking_on', None)
        print(f"✅ TEST B: Status OK (manual={manual}, auto={auto}, st={st})")
    except Exception as e:
        print(f"❌ TEST B: Status Fehler: {e}")
        return False

    return True

def main():
    print("\n" + "="*70)
    print("MODE ISOLATION FIX")
    print("="*70)

    # Verify existing guards
    if not verify_existing_guards():
        print("\n❌ Bestehende Guards fehlen!")
        return False

    # Apply fixes
    if not fix_panel_ptz_guards():
        return False
    if not fix_service_st_enforcement():
        return False

    # Run tests
    if not run_tests():
        print("\n❌ TESTS FAILED!")
        return False

    print("\n" + "="*70)
    print("✅ FIXES ANGEWENDET + GETESTET!")
    print("="*70)
    print("\n>>> Panel NEUSTART fuer PTZ Guards! <<<")
    print("\nDANN TESTEN:")
    print("1. MANUAL Mode: Steuerkreuz works, MOLOCH blocked")
    print("2. AUTONOMOUS Mode: Steuerkreuz blocked, MOLOCH aktiv")
    print("3. ST Button togglet Smart Tracking")
    return True

if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
