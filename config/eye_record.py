#!/usr/bin/env python3
"""M.O.L.O.C.H. Eye - Einzelaufzeichnung. Argument: Dauer in Sekunden."""
import json, sys, time, threading
from datetime import datetime
from onvif import ONVIFCamera
from zeep.plugins import HistoryPlugin
from lxml import etree

CAM_IP = "192.168.178.25"
duration = int(sys.argv[1]) if len(sys.argv) > 1 else 30
label = sys.argv[2] if len(sys.argv) > 2 else "recording"

def ts():
    return datetime.now().isoformat(timespec='milliseconds')

cam = ONVIFCamera(CAM_IP, 80, "Moloch_4.5", "Auge666")
media = cam.create_media_service()
ptz = cam.create_ptz_service()
profiles = media.GetProfiles()
token = profiles[0].token
history = HistoryPlugin()
pp = cam.create_pullpoint_service()
pp.zeep_client.plugins.append(history)

events = []
positions = []
ns_tt = '{http://www.onvif.org/ver10/schema}'
ns_wsnt = '{http://docs.oasis-open.org/wsn/b-2}'

# Initial position
st = ptz.GetStatus({'ProfileToken': token})
pos = st.Position
pt = getattr(pos, 'PanTilt', None)
zm = getattr(pos, 'Zoom', None)
p0 = {"timestamp": ts(), "pan": float(getattr(pt,'x',0)), "tilt": float(getattr(pt,'y',0)), "zoom": float(getattr(zm,'x',0))}
positions.append(p0)
print(f"[{ts()}] START: pan={p0['pan']:.1f} tilt={p0['tilt']:.1f} zoom={p0['zoom']:.2f}")
print(f"[{ts()}] Aufzeichnung: {duration}s -- {label}")

last_pan, last_tilt = p0["pan"], p0["tilt"]
t_end = time.time() + duration
ec = 0

while time.time() < t_end:
    # Events
    try:
        pp.PullMessages({'Timeout': 'PT1S', 'MessageLimit': 50})
        if history.last_received:
            raw = history.last_received.get('envelope')
            if raw is not None:
                for ne in raw.iter(f'{ns_wsnt}NotificationMessage'):
                    ec += 1
                    ev = {"timestamp": ts(), "topic": None, "source": {}, "data": {}, "utc_time": None, "op": None}
                    te = ne.find(f'{ns_wsnt}Topic')
                    if te is not None and te.text:
                        ev["topic"] = te.text.strip()
                    me = ne.find(f'.//{ns_tt}Message')
                    if me is not None:
                        ev["utc_time"] = me.get('UtcTime')
                        ev["op"] = me.get('PropertyOperation')
                        for si in me.findall(f'.//{ns_tt}Source/{ns_tt}SimpleItem'):
                            ev["source"][si.get('Name','')] = si.get('Value','')
                        for si in me.findall(f'.//{ns_tt}Data/{ns_tt}SimpleItem'):
                            ev["data"][si.get('Name','')] = si.get('Value','')
                    events.append(ev)
                    tshort = ev["topic"].split("/")[-1] if ev["topic"] else "?"
                    print(f"  [{ts()}] EVENT #{ec}: {tshort} | {ev['op']} | data={json.dumps(ev['data'])}")
    except:
        pass

    # PTZ
    try:
        st = ptz.GetStatus({'ProfileToken': token})
        pos = st.Position
        pt = getattr(pos, 'PanTilt', None)
        zm = getattr(pos, 'Zoom', None)
        cp = float(getattr(pt,'x',0))
        ct = float(getattr(pt,'y',0))
        cz = float(getattr(zm,'x',0))
        ms = getattr(st, 'MoveStatus', None)
        ms_pt = str(getattr(ms, 'PanTilt', None)) if ms else None
        if abs(cp - last_pan) > 0.5 or abs(ct - last_tilt) > 0.5:
            print(f"  [{ts()}] PTZ: pan={cp:.1f} tilt={ct:.1f} zoom={cz:.2f} [{ms_pt}]")
            last_pan, last_tilt = cp, ct
        positions.append({"timestamp": ts(), "pan": cp, "tilt": ct, "zoom": cz, "move_status": ms_pt})
    except:
        pass
    time.sleep(1)

# Final position
pf = positions[-1] if positions else {}
print(f"\n[{ts()}] ENDE: pan={pf.get('pan',0):.1f} tilt={pf.get('tilt',0):.1f}")
print(f"  Events: {ec}, Positionen: {len(positions)}")

pans = [p["pan"] for p in positions]
tilts = [p["tilt"] for p in positions]
print(f"  Pan range: [{min(pans):.1f} .. {max(pans):.1f}]")
print(f"  Tilt range: [{min(tilts):.1f} .. {max(tilts):.1f}]")

result = {
    "label": label,
    "timestamp": ts(),
    "duration_sec": duration,
    "start_position": positions[0] if positions else None,
    "end_position": positions[-1] if positions else None,
    "pan_range": {"min": min(pans), "max": max(pans)},
    "tilt_range": {"min": min(tilts), "max": max(tilts)},
    "events": events,
    "positions": positions,
    "event_count": ec,
    "position_count": len(positions)
}

outfile = f"C:/Users/49179/moloch/config/rec_{label}.json"
with open(outfile, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
print(f"  Gespeichert: {outfile}")
