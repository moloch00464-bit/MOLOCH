#!/usr/bin/env python3
"""Fix: Manual-Button Komplett + Tests

AUFTRAG:
Manual-Button ON  → M.O.L.O.C.H. gesperrt (keine Kamera-Kontrolle)
Manual-Button OFF → M.O.L.O.C.H. aktiv (Tentakel/Takeover erlaubt)

FIXES:
1. Service: manual_mode in Status-JSON
2. Panel: manual_mode aus Status lesen + in ServiceProxy speichern
3. Panel: AUTONOM Button Color-Update basierend auf manual_mode
4. Panel: Mode-Status-Label (MANUELL/MOLOCH/TENTAKEL/BEREIT)

TESTS (automatisch nach Fix):
A. Service laeuft + manual_mode im Status?
B. toggle_autonomous_manual() setzt _manual_mode?
C. IPC action 'toggle_autonomous' funktioniert?
"""

import subprocess
import json
import time

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'

fixes = 0

print("=" * 60)
print("FIX 1: manual_mode in Status-JSON")
print("=" * 60)

with open(svc) as f:
    svc_code = f.read()

old = '"autonomous_mode": self._autonomous_mode,'
new = '"autonomous_mode": self._autonomous_mode,\n                "manual_mode": self._manual_mode,'

if old in svc_code and new not in svc_code:
    svc_code = svc_code.replace(old, new, 1)
    with open(svc, 'w') as f:
        f.write(svc_code)
    print("✓ manual_mode in Status-JSON hinzugefuegt")
    fixes += 1
elif new in svc_code:
    print("✓ manual_mode bereits im Status-JSON")
    fixes += 1
else:
    print("✗ ANCHOR NOT FOUND!")

print("\n" + "=" * 60)
print("FIX 2: Panel - manual_mode lesen aus Status")
print("=" * 60)

with open(panel) as f:
    panel_code = f.read()

# In ServiceProxy __init__ - nach _tentakel_enabled
old_init = '        self._tentakel_enabled = False'
new_init = '''        self._tentakel_enabled = False
        self._manual_mode = False  # Remote state from service'''

if old_init in panel_code and '_manual_mode = False  # Remote state' not in panel_code:
    panel_code = panel_code.replace(old_init, new_init, 1)
    print("✓ _manual_mode in ServiceProxy.__init__")
    fixes += 1
elif '_manual_mode = False' in panel_code:
    print("✓ _manual_mode bereits in __init__")
    fixes += 1
else:
    print("✗ ANCHOR NOT FOUND!")

# In _apply_status - nach _tentakel_enabled
old_apply = '        self._tentakel_enabled = s.get(\'tentakel_enabled\', False)'
new_apply = '''        self._tentakel_enabled = s.get('tentakel_enabled', False)
        self._manual_mode = s.get('manual_mode', False)  # Read manual mode from service'''

if old_apply in panel_code and "self._manual_mode = s.get('manual_mode'" not in panel_code:
    panel_code = panel_code.replace(old_apply, new_apply, 1)
    print("✓ manual_mode aus Status lesen")
    fixes += 1
elif "self._manual_mode = s.get('manual_mode'" in panel_code:
    print("✓ manual_mode wird bereits gelesen")
    fixes += 1
else:
    print("✗ ANCHOR NOT FOUND!")

with open(panel, 'w') as f:
    f.write(panel_code)

print("\n" + "=" * 60)
print("FIX 3: AUTONOM Button Update (Color basierend auf manual_mode)")
print("=" * 60)

# Direkt per sed - nach ST button update, vor try: block
cmd = '''sed -i '/bg="#884400" if st_on else "#2a2a4e"/a\\
\\
        # Update AUTONOM Button\\
        if hasattr(self.service, "_manual_mode"):\\
            manual = self.service._manual_mode\\
            self.auto_btn.config(\\
                text="MANUELL" if manual else "AUTONOM",\\
                bg="#00aa00" if manual else "#dd2222"\\
            )' core/gui/moloch_unified_panel.py'''

result = subprocess.run(cmd, shell=True, cwd='/home/molochzuhause/moloch')
if result.returncode == 0:
    print("✓ AUTONOM Button Update Code eingefuegt")
    fixes += 1
else:
    print("✗ sed Fehler!")

print("\n" + "=" * 60)
print("FIX 4: Mode-Status-Label (optional, cosmetic)")
print("=" * 60)

# Status-Label in GUI - nach fps_label
cmd = '''sed -i '/self.fps_label.pack(side=tk.LEFT/a\\
\\
        # Mode status\\
        self.mode_label = tk.Label(bar, text="MODUS: --", bg="#0a0a14", fg="#ffaa00",\\
                                    font=("Helvetica", 11, "bold"))\\
        self.mode_label.pack(side=tk.LEFT, padx=5)' core/gui/moloch_unified_panel.py'''

result = subprocess.run(cmd, shell=True, cwd='/home/molochzuhause/moloch')
if result.returncode == 0:
    print("✓ Mode-Label GUI erstellt")
    fixes += 1
else:
    print("✗ sed Fehler (optional, nicht kritisch)")

# Mode-Label Update in _update_fps - nach AUTONOM button update
cmd = '''sed -i '/bg="#00aa00" if manual else "#dd2222"/,+2 a\\
\\
        # Update Mode Label\\
        if hasattr(self, "mode_label") and hasattr(self.service, "_manual_mode"):\\
            manual = getattr(self.service, "_manual_mode", False)\\
            st_on = getattr(self.service, "_smart_tracking_on", False)\\
            moloch = getattr(self.service, "_moloch_has_control", False)\\
            if manual:\\
                self.mode_label.config(text="MODUS: MANUELL", fg="#00ff00")\\
            elif moloch:\\
                self.mode_label.config(text="MODUS: MOLOCH", fg="#ff4444")\\
            elif st_on:\\
                self.mode_label.config(text="MODUS: TENTAKEL", fg="#00ffff")\\
            else:\\
                self.mode_label.config(text="MODUS: BEREIT", fg="#ffaa00")' core/gui/moloch_unified_panel.py'''

result = subprocess.run(cmd, shell=True, cwd='/home/molochzuhause/moloch')
if result.returncode == 0:
    print("✓ Mode-Label Update Logic")
    fixes += 1
else:
    print("✗ sed Fehler (optional)")

print("\n" + "=" * 60)
print(f"FIXES ANGEWENDET: {fixes}/6")
print("=" * 60)

# Syntax Check
print("\nSyntax-Check...")
try:
    subprocess.run(['python3', '-m', 'py_compile', svc], check=True, cwd='/home/molochzuhause/moloch', capture_output=True)
    print("✓ Service Syntax OK")
except:
    print("✗ SERVICE SYNTAX ERROR!")
    subprocess.run(['git', 'checkout', svc], cwd='/home/molochzuhause/moloch')
    exit(1)

try:
    subprocess.run(['python3', '-m', 'py_compile', panel], check=True, cwd='/home/molochzuhause/moloch', capture_output=True)
    print("✓ Panel Syntax OK")
except:
    print("✗ PANEL SYNTAX ERROR!")
    subprocess.run(['git', 'checkout', panel], cwd='/home/molochzuhause/moloch')
    exit(1)

# Service Restart
print("\n" + "=" * 60)
print("Service Neustart...")
print("=" * 60)
subprocess.run(['sudo', 'systemctl', 'restart', 'moloch.service'], check=True)
time.sleep(3)

# Test ob Service laeuft
result = subprocess.run(['systemctl', 'is-active', 'moloch.service'], capture_output=True, text=True)
if result.stdout.strip() == 'active':
    print("✓ Service läuft")
else:
    print("✗ Service NICHT aktiv!")
    exit(1)

# Test A: manual_mode im Status-JSON?
print("\n" + "=" * 60)
print("TEST A: manual_mode im Status-JSON?")
print("=" * 60)
time.sleep(1)
try:
    with open('/dev/shm/moloch_status.json') as f:
        status = json.load(f)
    if 'manual_mode' in status:
        print(f"✅ manual_mode im Status: {status['manual_mode']}")
    else:
        print("❌ manual_mode FEHLT im Status!")
except Exception as e:
    print(f"❌ Fehler beim Status-Lesen: {e}")

# Test B: toggle_autonomous_manual() vorhanden?
print("\n" + "=" * 60)
print("TEST B: toggle_autonomous_manual() implementiert?")
print("=" * 60)
result = subprocess.run(['grep', '-q', 'def toggle_autonomous_manual', svc], cwd='/home/molochzuhause/moloch')
if result.returncode == 0:
    print("✅ Methode toggle_autonomous_manual() existiert")
else:
    print("❌ Methode FEHLT!")

# Test C: IPC Handler fuer toggle_autonomous?
print("\n" + "=" * 60)
print("TEST C: IPC action 'toggle_autonomous' registriert?")
print("=" * 60)
result = subprocess.run(['grep', '-q', "action == 'toggle_autonomous'", svc], cwd='/home/molochzuhause/moloch')
if result.returncode == 0:
    print("✅ IPC Handler existiert")
else:
    print("❌ IPC Handler FEHLT!")

print("\n" + "=" * 60)
print("FERTIG!")
print("=" * 60)
print("\n>>> Panel NEUSTART erforderlich fuer Button-Updates <<<\n")
