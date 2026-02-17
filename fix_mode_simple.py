#!/usr/bin/env python3
import subprocess, time, json

print("="*60)
print("MODE ISOLATION FIX")
print("="*60)

# FIX 1: Panel PTZ Guards
print("\nFIX 1: Panel PTZ Guards...")
with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py') as f:
    lines = f.readlines()

fixed = 0
for i, line in enumerate(lines):
    if 'def do_move():' in line and '_ptz_move' in ''.join(lines[max(0,i-5):i]):
        if 'manual_mode' not in ''.join(lines[i:i+10]):
            guard = '''            # Mode Guard: Block PTZ in AUTONOMOUS
            if hasattr(self, 'service') and hasattr(self.service, '_manual_mode'):
                if not self.service._manual_mode:
                    logger.info("[PTZ] Blocked: AUTONOMOUS mode")
                    return
'''
            lines.insert(i+1, guard)
            print("  PTZ guard in _ptz_move eingefuegt")
            fixed += 1
        else:
            print("  PTZ guard in _ptz_move bereits da")
        break

for i, line in enumerate(lines):
    if 'def do_goto():' in line and '_ptz_goto' in ''.join(lines[max(0,i-5):i]):
        if 'manual_mode' not in ''.join(lines[i:i+10]):
            guard = '''            # Mode Guard
            if hasattr(self, 'service') and hasattr(self.service, '_manual_mode'):
                if not self.service._manual_mode:
                    return
'''
            lines.insert(i+1, guard)
            print("  PTZ guard in _ptz_goto eingefuegt")
            fixed += 1
        else:
            print("  PTZ guard in _ptz_goto bereits da")
        break

with open('/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py', 'w') as f:
    f.writelines(lines)

# FIX 2: Service ST Enforcement
print("\nFIX 2: Service ST Enforcement...")
with open('/home/molochzuhause/moloch/core/moloch_service.py') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'AUTONOM: Kamera wieder freigeben' in line:
        # Find self._manual_mode = False
        for j in range(i, min(i+15, len(lines))):
            if 'self._manual_mode = False' in lines[j]:
                if 'set_smart_tracking(True)' not in ''.join(lines[j:j+15]):
                    st_code = '''
            # Smart Tracking AN
            def enable_st():
                if self._cloud and self._cloud.connected:
                    try:
                        self._cloud.run(self._cloud.bridge.set_smart_tracking(True))
                        self._set_smart_tracking_state(True)
                    except: pass
            threading.Thread(target=enable_st, daemon=True).start()
'''
                    lines.insert(j+1, st_code)
                    print("  ST Enforcement eingefuegt")
                    fixed += 1
                else:
                    print("  ST Enforcement bereits da")
                break
        break

with open('/home/molochzuhause/moloch/core/moloch_service.py', 'w') as f:
    f.writelines(lines)

# Syntax + Restart
print("\nSyntax Check...")
try:
    subprocess.run(['python3', '-m', 'py_compile', '/home/molochzuhause/moloch/core/moloch_service.py'], check=True, capture_output=True)
    subprocess.run(['python3', '-m', 'py_compile', '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'], check=True, capture_output=True)
    print("  Syntax OK")
except:
    print("  SYNTAX ERROR!")
    subprocess.run(['git', 'checkout', '/home/molochzuhause/moloch/core/'])
    exit(1)

print("\nService Restart...")
subprocess.run(['sudo', 'systemctl', 'restart', 'moloch.service'], check=True)
time.sleep(3)

result = subprocess.run(['systemctl', 'is-active', 'moloch.service'], capture_output=True, text=True)
if result.stdout.strip() == 'active':
    print("  Service laeuft")
else:
    print("  Service FEHLER!")
    exit(1)

print("\n" + "="*60)
print(f"FERTIG! {fixed} Fixes angewendet")
print("="*60)
print("\n>>> Panel NEUSTARTEN <<<\n")
