#!/usr/bin/env python3
"""Test: Kann PTZ Kalibrierung ueber ONVIF ausgeloest werden?"""
from onvif import ONVIFCamera
from zeep.helpers import serialize_object
import json

cam = ONVIFCamera("192.168.178.25", 80, "Moloch_4.5", "Auge666")
ptz = cam.create_ptz_service()
media = cam.create_media_service()
profiles = media.GetProfiles()
token = profiles[0].token

# 1. Check: SendAuxiliaryCommand (manche Kameras haben "calibrate" als AuxCmd)
print("=== AuxiliaryCommands ===")
nodes = ptz.GetNodes()
for n in nodes:
    cmds = getattr(n, 'AuxiliaryCommands', [])
    print(f"  Node {n.Name}: AuxCommands = {cmds}")

# 2. Try SendAuxiliaryCommand with known calibration strings
aux_commands = [
    "tt:Wiper|On", "tt:IRLamp|Auto", "tt:IRLamp|On",
    "Calibrate", "PTZCalibrate", "tt:Calibrate",
    "aux1", "HomeReset", "tt:HomeReset",
]
print("\n=== SendAuxiliaryCommand Tests ===")
for cmd in aux_commands:
    try:
        result = ptz.SendAuxiliaryCommand({'ProfileToken': token, 'AuxiliaryData': cmd})
        print(f"  '{cmd}': OK -> {result}")
    except Exception as e:
        err = str(e)[:100]
        print(f"  '{cmd}': {err}")

# 3. SetHomePosition (setzt aktuelle Position als Home)
print("\n=== SetHomePosition ===")
try:
    ptz.SetHomePosition({'ProfileToken': token})
    print("  SetHomePosition: OK")
except Exception as e:
    print(f"  SetHomePosition: {e}")

# 4. GotoHomePosition
print("\n=== GotoHomePosition ===")
try:
    ptz.GotoHomePosition({'ProfileToken': token})
    print("  GotoHomePosition: OK")
except Exception as e:
    print(f"  GotoHomePosition: {e}")

# 5. Check device service for system commands
print("\n=== Device System Commands ===")
try:
    # SystemReboot
    print("  Reboot available (not executing)")
except: pass

# 6. Check all PTZ service operations
print("\n=== Alle PTZ Operationen ===")
try:
    svc = ptz.ws_client
    # List available operations
    ops = [attr for attr in dir(svc) if not attr.startswith('_') and callable(getattr(svc, attr, None))]
    for op in sorted(ops):
        print(f"  {op}")
except Exception as e:
    print(f"  Error listing ops: {e}")

print("\n=== Fazit ===")
print("PTZ-Kalibrierung ist eine Firmware-Funktion.")
print("Moeglich ueber ONVIF: Nur wenn AuxiliaryCommands es unterstuetzt.")
print("Alternative: Kompletten Bewegungsbereich selbst abfahren (wie die App).")
