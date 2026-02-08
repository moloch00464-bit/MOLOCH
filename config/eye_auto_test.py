#!/usr/bin/env python3
"""
M.O.L.O.C.H. Eye -- Automatisches Testprotokoll
Alle Tests werden direkt ueber ONVIF + HA API ausgeloest.
"""
import json, sys, time, os, threading, requests
from datetime import datetime
from collections import defaultdict
from onvif import ONVIFCamera
from zeep.plugins import HistoryPlugin
from lxml import etree

CAM_IP = "192.168.178.25"
CAM_PORT = 80
CAM_USER = "Moloch_4.5"
CAM_PASS = "Auge666"
HA_URL = "http://192.168.178.32:8123"
HA_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJmZDgyNWI1MDNjMTY0ZDY0YWY0Y2U0NjRkZjkyMTFlNiIsImlhdCI6MTc3MDU0MTg1MCwiZXhwIjoyMDg1OTAxODUwfQ.hlvHR8U3pZ-1kiLDw64YMHBqdROs4j2TsxnbcKhrj5Q"
HA_HEADERS = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
OUTPUT_DIR = "C:/Users/49179/moloch/config"

all_events = []
all_positions = []
test_results = []

def ts():
    return datetime.now().isoformat(timespec='milliseconds')

def serialize(obj):
    if obj is None: return None
    if isinstance(obj, (str, int, float, bool)): return obj
    if isinstance(obj, (list, tuple)): return [serialize(i) for i in obj]
    if isinstance(obj, dict): return {k: serialize(v) for k, v in obj.items()}
    try:
        from zeep.helpers import serialize_object
        s = serialize_object(obj)
        if isinstance(s, (dict, list)): return serialize(s)
        return s
    except: pass
    if hasattr(obj, '__dict__'):
        return {k: serialize(v) for k, v in obj.__dict__.items() if not k.startswith('_')} or str(obj)
    return str(obj)

def clean(d):
    if isinstance(d, dict):
        return {k: clean(v) for k, v in d.items() if k not in ('_value_1','_attr_1') and v is not None}
    if isinstance(d, list): return [clean(i) for i in d]
    return d


# ── Event Collector (Background Thread) ──────────────────────────────────

class EventCollector(threading.Thread):
    def __init__(self, cam):
        super().__init__(daemon=True)
        self.history = HistoryPlugin()
        self.pp = cam.create_pullpoint_service()
        self.pp.zeep_client.plugins.append(self.history)
        self.events = []
        self.running = True
        self.cam = cam

    def run(self):
        ns_tt = '{http://www.onvif.org/ver10/schema}'
        ns_wsnt = '{http://docs.oasis-open.org/wsn/b-2}'
        while self.running:
            try:
                self.pp.PullMessages({'Timeout': 'PT1S', 'MessageLimit': 50})
                if self.history.last_received:
                    raw = self.history.last_received.get('envelope')
                    if raw is not None:
                        for ne in raw.iter(f'{ns_wsnt}NotificationMessage'):
                            ev = {"timestamp": ts(), "topic": None, "source": {}, "data": {}}
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
                            self.events.append(ev)
                            tshort = ev["topic"].split("/")[-1] if ev["topic"] else "?"
                            print(f"      >> EVENT: {tshort} | {ev.get('op','')} | data={json.dumps(ev['data'])}")
            except Exception as e:
                if "terminated" in str(e).lower():
                    try:
                        self.pp = self.cam.create_pullpoint_service()
                        self.pp.zeep_client.plugins.append(self.history)
                    except:
                        pass
            time.sleep(0.3)

    def snapshot(self):
        return len(self.events)

    def events_since(self, idx):
        return self.events[idx:]

    def stop(self):
        self.running = False


# ── PTZ Helper ────────────────────────────────────────────────────────────

def get_pos():
    st = ptz.GetStatus({'ProfileToken': main_token})
    pos = st.Position
    pt = getattr(pos, 'PanTilt', None)
    zm = getattr(pos, 'Zoom', None)
    ms = getattr(st, 'MoveStatus', None)
    return {
        "pan": float(getattr(pt,'x',0)), "tilt": float(getattr(pt,'y',0)),
        "zoom": float(getattr(zm,'x',0)),
        "status": str(getattr(ms, 'PanTilt', None)) if ms else None
    }

def wait_idle(timeout=10):
    for _ in range(timeout * 2):
        p = get_pos()
        if p.get("status") == "IDLE" or p.get("status") == "None":
            return p
        time.sleep(0.5)
    return get_pos()

def record_pos(label):
    p = get_pos()
    p["timestamp"] = ts()
    p["label"] = label
    all_positions.append(p)
    return p


# ── Test Runner ───────────────────────────────────────────────────────────

def run_test(tid, name, fn):
    print(f"\n{'='*60}")
    print(f"  [{tid}] {name}")
    print(f"  [{ts()}]")
    print(f"{'='*60}")
    ei = ec.snapshot()
    t0 = ts()
    try:
        data = fn()
    except Exception as e:
        data = {"error": str(e)}
        print(f"  !! FEHLER: {e}")
    events = ec.events_since(ei)
    result = {"test_id": tid, "name": name, "start": t0, "end": ts(),
              "data": data, "events": events, "event_count": len(events)}
    test_results.append(result)
    print(f"  -> {len(events)} Events | Fertig")
    return result


# ============================================================================
#  PHASE 1 -- DISCOVERY (automatisch)
# ============================================================================

def test_1_1():
    caps = cam.devicemgmt.GetCapabilities({'Category': 'All'})
    svc = cam.devicemgmt.GetServiceCapabilities()
    summary = {}
    for s in ['Analytics','Device','Events','Imaging','Media','PTZ']:
        has = bool(getattr(caps, s, None))
        summary[s] = has
        print(f"    {s}: {'JA' if has else 'nein'}")
    return {"capabilities": clean(serialize(caps)), "service_capabilities": clean(serialize(svc)), "summary": summary}

def test_1_2():
    di = cam.devicemgmt.GetDeviceInformation()
    ni = cam.devicemgmt.GetNetworkInterfaces()
    print(f"    {di.Manufacturer} {di.Model} FW:{di.FirmwareVersion} SN:{di.SerialNumber}")
    for iface in ni:
        ip_cfg = getattr(getattr(iface,'IPv4',None),'Config',None)
        ip = ip_cfg.Manual[0].Address if ip_cfg and ip_cfg.Manual else '?'
        print(f"    {iface.Info.Name}: {ip} ({iface.Info.HwAddress})")
    return {"device_info": clean(serialize(di)), "network": clean(serialize(ni))}

def test_1_3():
    profs = media.GetProfiles()
    vsrc = media.GetVideoSources()
    uris = {}
    try: asrc = media.GetAudioSources()
    except: asrc = []
    for p in profs:
        vec = p.VideoEncoderConfiguration
        res = f"{vec.Resolution.Width}x{vec.Resolution.Height}" if vec else "?"
        print(f"    {p.Name} [{p.token}]: {res} {vec.Encoding if vec else '?'}")
        try:
            uri = media.GetStreamUri({'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}}, 'ProfileToken': p.token})
            uris[p.token] = uri.Uri
            print(f"      RTSP: {uri.Uri}")
        except Exception as e:
            uris[p.token] = str(e)
    print(f"    Video: {len(vsrc)}, Audio: {len(asrc)}")
    return {"profiles": clean(serialize(profs)), "video_sources": clean(serialize(vsrc)),
            "audio_sources": clean(serialize(asrc)), "stream_uris": uris}

def test_1_4():
    nodes = ptz.GetNodes()
    configs = ptz.GetConfigurations()
    for n in nodes:
        print(f"    Node: {n.Name} [{n.token}] Home:{n.HomeSupported} MaxPresets:{n.MaximumNumberOfPresets}")
        sp = getattr(n, 'SupportedPTZSpaces', None)
        if sp:
            for s in (getattr(sp,'AbsolutePanTiltPositionSpace',[]) or []):
                print(f"      AbsPT: X=[{s.XRange.Min}..{s.XRange.Max}] Y=[{s.YRange.Min}..{s.YRange.Max}]")
    for c in configs:
        ptl = getattr(c, 'PanTiltLimits', None)
        if ptl:
            r = ptl.Range
            print(f"    Limits: Pan=[{r.XRange.Min}..{r.XRange.Max}] Tilt=[{r.YRange.Min}..{r.YRange.Max}]")
    return {"nodes": clean(serialize(nodes)), "configurations": clean(serialize(configs))}

def test_1_5():
    print(f"    EventCollector aktiv: Events bisher={ec.snapshot()}")
    time.sleep(3)
    print(f"    Nach 3s: Events={ec.snapshot()}")
    return {"collector_active": True, "events_at_start": ec.snapshot()}


# ============================================================================
#  PHASE 2 -- AKTIVE TESTS (selbst ausgeloest)
# ============================================================================

def test_2_1_motion():
    """Motion/Person/Cell Detection - PTZ schnell bewegen um Szene zu aendern"""
    print(f"    Trigge Motion durch schnelle PTZ-Bewegung...")
    p0 = record_pos("motion_before")
    # Schnelle Hin-und-Her Bewegung
    ptz.ContinuousMove({'ProfileToken': main_token, 'Velocity': {'PanTilt': {'x': 1.0, 'y': 0.0}, 'Zoom': {'x': 0.0}}})
    time.sleep(1.5)
    ptz.Stop({'ProfileToken': main_token, 'PanTilt': True, 'Zoom': True})
    time.sleep(0.5)
    ptz.ContinuousMove({'ProfileToken': main_token, 'Velocity': {'PanTilt': {'x': -1.0, 'y': 0.0}, 'Zoom': {'x': 0.0}}})
    time.sleep(1.5)
    ptz.Stop({'ProfileToken': main_token, 'PanTilt': True, 'Zoom': True})
    print(f"    Warte 10s auf Events...")
    time.sleep(10)
    p1 = record_pos("motion_after")
    print(f"    Pos: pan={p1['pan']:.1f} tilt={p1['tilt']:.1f}")
    return {"pos_before": p0, "pos_after": p1}

def test_2_2_ptz_absolute():
    """AbsoluteMove zu 4 Ecken + Center"""
    positions_test = [
        ("center", 0.0, 0.0),
        ("rechts_oben", 100.0, 50.0),
        ("links_unten", -100.0, -50.0),
        ("links_oben", -100.0, 50.0),
        ("rechts_unten", 100.0, -50.0),
        ("center_final", 0.0, 0.0),
    ]
    results = {}
    for label, pan, tilt in positions_test:
        print(f"    AbsoluteMove -> {label} ({pan}, {tilt})...")
        ptz.AbsoluteMove({'ProfileToken': main_token, 'Position': {'PanTilt': {'x': pan, 'y': tilt}, 'Zoom': {'x': 0.0}}})
        time.sleep(3)
        p = wait_idle(5)
        record_pos(f"abs_{label}")
        results[label] = {"target": {"pan": pan, "tilt": tilt}, "actual": p}
        print(f"      Ist: pan={p['pan']:.1f} tilt={p['tilt']:.1f} [{p.get('status')}]")
    return results

def test_2_3_ptz_relative():
    """RelativeMove: +50pan, -50pan, +30tilt, -30tilt"""
    # Start bei 0,0
    ptz.AbsoluteMove({'ProfileToken': main_token, 'Position': {'PanTilt': {'x': 0.0, 'y': 0.0}, 'Zoom': {'x': 0.0}}})
    time.sleep(3)
    p0 = record_pos("rel_start")
    print(f"    Start: pan={p0['pan']:.1f} tilt={p0['tilt']:.1f}")

    moves = [
        ("pan+50", 50.0, 0.0),
        ("pan-100", -100.0, 0.0),
        ("tilt+30", 0.0, 30.0),
        ("tilt-60", 0.0, -60.0),
    ]
    results = {"start": p0}
    for label, dx, dy in moves:
        print(f"    RelativeMove {label} (dx={dx}, dy={dy})...")
        ptz.RelativeMove({'ProfileToken': main_token, 'Translation': {'PanTilt': {'x': dx, 'y': dy}, 'Zoom': {'x': 0.0}}})
        time.sleep(3)
        p = wait_idle(5)
        record_pos(f"rel_{label}")
        results[label] = p
        print(f"      Ist: pan={p['pan']:.1f} tilt={p['tilt']:.1f}")
    return results

def test_2_4_ptz_continuous():
    """ContinuousMove in alle 4 Richtungen"""
    ptz.AbsoluteMove({'ProfileToken': main_token, 'Position': {'PanTilt': {'x': 0.0, 'y': 0.0}, 'Zoom': {'x': 0.0}}})
    time.sleep(3)
    record_pos("cont_start")

    directions = [
        ("rechts", 0.5, 0.0), ("links", -0.5, 0.0),
        ("hoch", 0.0, 0.5), ("runter", 0.0, -0.5),
        ("rechts_hoch", 0.5, 0.5), ("links_runter", -0.5, -0.5),
    ]
    results = {}
    for label, vx, vy in directions:
        p_before = get_pos()
        ptz.ContinuousMove({'ProfileToken': main_token, 'Velocity': {'PanTilt': {'x': vx, 'y': vy}, 'Zoom': {'x': 0.0}}})
        time.sleep(2)
        ptz.Stop({'ProfileToken': main_token, 'PanTilt': True, 'Zoom': True})
        time.sleep(1)
        p_after = get_pos()
        record_pos(f"cont_{label}")
        dp = p_after['pan'] - p_before['pan']
        dt = p_after['tilt'] - p_before['tilt']
        results[label] = {"before": p_before, "after": p_after, "delta_pan": round(dp,1), "delta_tilt": round(dt,1)}
        print(f"    {label}: delta pan={dp:+.1f} tilt={dt:+.1f}")
    return results

def test_2_5_ptz_home():
    """GotoHomePosition"""
    ptz.AbsoluteMove({'ProfileToken': main_token, 'Position': {'PanTilt': {'x': 80.0, 'y': -40.0}, 'Zoom': {'x': 0.0}}})
    time.sleep(3)
    p_away = record_pos("home_away")
    print(f"    Weg von Home: pan={p_away['pan']:.1f} tilt={p_away['tilt']:.1f}")

    print(f"    GotoHomePosition...")
    try:
        ptz.GotoHomePosition({'ProfileToken': main_token})
        time.sleep(4)
        p_home = wait_idle(5)
        record_pos("home_arrived")
        print(f"    Home: pan={p_home['pan']:.1f} tilt={p_home['tilt']:.1f}")
        return {"away": p_away, "home": p_home, "supported": True}
    except Exception as e:
        print(f"    Home ERROR: {e}")
        return {"away": p_away, "error": str(e), "supported": False}

def test_2_6_presets():
    """SetPreset, GotoPreset, GetPresets"""
    results = {}

    # Bestehende Presets
    pre_existing = ptz.GetPresets({'ProfileToken': main_token})
    results["existing"] = len(pre_existing) if pre_existing else 0
    print(f"    Bestehende Presets: {results['existing']}")

    # Preset 1: Tuer (50, -20)
    ptz.AbsoluteMove({'ProfileToken': main_token, 'Position': {'PanTilt': {'x': 50.0, 'y': -20.0}, 'Zoom': {'x': 0.0}}})
    time.sleep(3)
    wait_idle()
    try:
        t1 = ptz.SetPreset({'ProfileToken': main_token, 'PresetName': 'moloch_tuer'})
        results["preset_1"] = {"token": str(t1), "name": "moloch_tuer", "pos": get_pos()}
        print(f"    Preset 1 'moloch_tuer' erstellt: token={t1}")
    except Exception as e:
        results["preset_1"] = {"error": str(e)}
        print(f"    Preset 1 ERROR: {e}")

    # Preset 2: Fenster (-80, 10)
    ptz.AbsoluteMove({'ProfileToken': main_token, 'Position': {'PanTilt': {'x': -80.0, 'y': 10.0}, 'Zoom': {'x': 0.0}}})
    time.sleep(3)
    wait_idle()
    try:
        t2 = ptz.SetPreset({'ProfileToken': main_token, 'PresetName': 'moloch_fenster'})
        results["preset_2"] = {"token": str(t2), "name": "moloch_fenster", "pos": get_pos()}
        print(f"    Preset 2 'moloch_fenster' erstellt: token={t2}")
    except Exception as e:
        results["preset_2"] = {"error": str(e)}
        print(f"    Preset 2 ERROR: {e}")

    # Preset 3: Uebersicht (0, 30)
    ptz.AbsoluteMove({'ProfileToken': main_token, 'Position': {'PanTilt': {'x': 0.0, 'y': 30.0}, 'Zoom': {'x': 0.0}}})
    time.sleep(3)
    wait_idle()
    try:
        t3 = ptz.SetPreset({'ProfileToken': main_token, 'PresetName': 'moloch_uebersicht'})
        results["preset_3"] = {"token": str(t3), "name": "moloch_uebersicht", "pos": get_pos()}
        print(f"    Preset 3 'moloch_uebersicht' erstellt: token={t3}")
    except Exception as e:
        results["preset_3"] = {"error": str(e)}
        print(f"    Preset 3 ERROR: {e}")

    # GotoPreset 1
    if "token" in results.get("preset_1", {}):
        print(f"    GotoPreset -> moloch_tuer...")
        ptz.GotoPreset({'ProfileToken': main_token, 'PresetToken': results["preset_1"]["token"]})
        time.sleep(4)
        p = wait_idle()
        results["goto_1"] = p
        print(f"      Ist: pan={p['pan']:.1f} tilt={p['tilt']:.1f}")

    # GotoPreset 2
    if "token" in results.get("preset_2", {}):
        print(f"    GotoPreset -> moloch_fenster...")
        ptz.GotoPreset({'ProfileToken': main_token, 'PresetToken': results["preset_2"]["token"]})
        time.sleep(4)
        p = wait_idle()
        results["goto_2"] = p
        print(f"      Ist: pan={p['pan']:.1f} tilt={p['tilt']:.1f}")

    # Finale Preset-Liste
    presets_final = ptz.GetPresets({'ProfileToken': main_token})
    results["final_presets"] = []
    for pr in (presets_final or []):
        pd = {"token": getattr(pr,'token','?'), "name": getattr(pr,'Name','?')}
        pos = getattr(pr, 'PTZPosition', None)
        if pos:
            pt = getattr(pos, 'PanTilt', None)
            zm = getattr(pos, 'Zoom', None)
            pd["pan"] = getattr(pt,'x',None)
            pd["tilt"] = getattr(pt,'y',None)
            pd["zoom"] = getattr(zm,'x',None)
        results["final_presets"].append(pd)
        print(f"    [{pd['token']}] {pd['name']}: pan={pd.get('pan')} tilt={pd.get('tilt')}")
    return results

def test_2_7_ha_ptz():
    """PTZ ueber HA API"""
    p0 = record_pos("ha_before")
    print(f"    HA PTZ: RelativeMove RIGHT+UP...")
    try:
        r = requests.post(f"{HA_URL}/api/services/onvif/ptz", headers=HA_HEADERS, json={
            "entity_id": "camera.cam_pt2_mainstream",
            "pan": "RIGHT", "tilt": "UP", "distance": 0.3, "speed": 0.5,
            "move_mode": "RelativeMove"
        }, timeout=10)
        print(f"    HTTP {r.status_code}")
    except Exception as e:
        print(f"    ERROR: {e}")
    time.sleep(3)
    p1 = record_pos("ha_after_right_up")
    print(f"    Nach RIGHT+UP: pan={p1['pan']:.1f} tilt={p1['tilt']:.1f}")

    print(f"    HA PTZ: RelativeMove LEFT+DOWN...")
    try:
        r = requests.post(f"{HA_URL}/api/services/onvif/ptz", headers=HA_HEADERS, json={
            "entity_id": "camera.cam_pt2_mainstream",
            "pan": "LEFT", "tilt": "DOWN", "distance": 0.3, "speed": 0.5,
            "move_mode": "RelativeMove"
        }, timeout=10)
    except: pass
    time.sleep(3)
    p2 = record_pos("ha_after_left_down")
    print(f"    Nach LEFT+DOWN: pan={p2['pan']:.1f} tilt={p2['tilt']:.1f}")
    return {"before": p0, "right_up": p1, "left_down": p2}

def test_2_8_ha_snapshot():
    """Snapshot ueber HA"""
    print(f"    HA: camera.snapshot...")
    try:
        r = requests.post(f"{HA_URL}/api/services/camera/snapshot", headers=HA_HEADERS, json={
            "entity_id": "camera.cam_pt2_mainstream",
            "filename": "/config/www/moloch_test_snapshot.jpg"
        }, timeout=10)
        print(f"    HTTP {r.status_code}")
        return {"status": r.status_code, "filename": "/config/www/moloch_test_snapshot.jpg"}
    except Exception as e:
        print(f"    ERROR: {e}")
        return {"error": str(e)}

def test_2_9_ha_entities():
    """Alle HA Entity-States nochmal abfragen"""
    keywords = ["cam_pt", "onvif"]
    try:
        r = requests.get(f"{HA_URL}/api/states", headers=HA_HEADERS, timeout=10)
        all_states = r.json()
        cam_ents = [e for e in all_states if any(k in e["entity_id"] for k in keywords)]
        for e in cam_ents:
            print(f"    {e['entity_id']}: {e['state']}")
        return {"entities": [{
            "entity_id": e["entity_id"], "state": e["state"],
            "attributes": e.get("attributes", {}),
            "last_changed": e.get("last_changed")
        } for e in cam_ents]}
    except Exception as e:
        return {"error": str(e)}

def test_2_10_reset():
    """Zurueck auf (0,0)"""
    ptz.AbsoluteMove({'ProfileToken': main_token, 'Position': {'PanTilt': {'x': 0.0, 'y': 0.0}, 'Zoom': {'x': 0.0}}})
    time.sleep(3)
    p = wait_idle()
    record_pos("final_reset")
    print(f"    Final: pan={p['pan']:.1f} tilt={p['tilt']:.1f}")
    return {"position": p}


# ============================================================================
#  MAIN
# ============================================================================

if __name__ == "__main__":
    print(f"\n  M.O.L.O.C.H. Eye -- Auto-Testprotokoll")
    print(f"  {ts()}")
    print(f"  Kamera: {CAM_IP}")
    print(f"  HA: {HA_URL}\n")

    cam = ONVIFCamera(CAM_IP, CAM_PORT, CAM_USER, CAM_PASS)
    media = cam.create_media_service()
    ptz = cam.create_ptz_service()
    profiles = media.GetProfiles()
    main_token = profiles[0].token
    print(f"  Verbunden. Profil: {profiles[0].Name} [{main_token}]")

    ec = EventCollector(cam)
    ec.start()
    time.sleep(2)
    print(f"  Event-Collector aktiv. ({ec.snapshot()} initial events)\n")

    # ── PHASE 1 ──
    print(f"{'#'*60}")
    print(f"  PHASE 1 -- DISCOVERY")
    print(f"{'#'*60}")

    run_test("1.1", "GetCapabilities + GetServiceCapabilities", test_1_1)
    run_test("1.2", "GetDeviceInformation + GetNetworkInterfaces", test_1_2)
    run_test("1.3", "Media: Profiles, Sources, StreamUri", test_1_3)
    run_test("1.4", "PTZ: Nodes, Configurations, Limits", test_1_4)
    run_test("1.5", "Event-Subscription pruefen", test_1_5)

    # ── PHASE 2 ──
    print(f"\n{'#'*60}")
    print(f"  PHASE 2 -- AKTIVE TESTS")
    print(f"{'#'*60}")

    run_test("2.1", "Motion/Person/Cell Detection Trigger", test_2_1_motion)
    run_test("2.2", "PTZ AbsoluteMove (4 Ecken + Center)", test_2_2_ptz_absolute)
    run_test("2.3", "PTZ RelativeMove", test_2_3_ptz_relative)
    run_test("2.4", "PTZ ContinuousMove (6 Richtungen)", test_2_4_ptz_continuous)
    run_test("2.5", "PTZ GotoHomePosition", test_2_5_ptz_home)
    run_test("2.6", "PTZ Presets (Set/Goto/Get)", test_2_6_presets)
    run_test("2.7", "PTZ ueber HA API", test_2_7_ha_ptz)
    run_test("2.8", "Snapshot ueber HA", test_2_8_ha_snapshot)
    run_test("2.9", "HA Entity-States", test_2_9_ha_entities)
    run_test("2.10", "Reset auf (0,0)", test_2_10_reset)

    ec.stop()

    # ── PHASE 3 -- SPEICHERN ──
    print(f"\n{'#'*60}")
    print(f"  PHASE 3 -- REPORT & SPEICHERN")
    print(f"{'#'*60}")

    all_ev = ec.events
    topic_counts = defaultdict(int)
    for ev in all_ev:
        topic_counts[ev.get("topic","?")] += 1

    # 1. eye_capabilities.json
    try:
        with open(os.path.join(OUTPUT_DIR, "eye_capabilities.json"), "r") as f:
            caps = json.load(f)
    except: caps = {}
    caps["auto_test"] = {
        "timestamp": ts(), "tests_run": len(test_results),
        "events_total": len(all_ev), "topics": dict(topic_counts),
        "test_summary": {t["test_id"]: {"name": t["name"], "events": t["event_count"]} for t in test_results}
    }
    with open(os.path.join(OUTPUT_DIR, "eye_capabilities.json"), "w") as f:
        json.dump(caps, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [1/4] eye_capabilities.json")

    # 2. eye_presets.json
    preset_test = next((t for t in test_results if t["test_id"] == "2.6"), None)
    presets = {
        "timestamp": ts(), "camera": f"SONOFF CAM-PT2 @ {CAM_IP}",
        "ptz_limits": {"pan": [-168.4, 174.4], "tilt": [-78.8, 101.3], "zoom": [0.0, 0.0]},
        "calibration_observed": {"pan": [-166.4, 126.1], "tilt": [-93.2, 84.9]},
        "presets": preset_test["data"].get("final_presets", []) if preset_test else [],
        "home_test": next((t["data"] for t in test_results if t["test_id"] == "2.5"), None)
    }
    with open(os.path.join(OUTPUT_DIR, "eye_presets.json"), "w") as f:
        json.dump(presets, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [2/4] eye_presets.json")

    # 3. eye_events.json
    events_out = {
        "timestamp": ts(), "total": len(all_ev), "topics": dict(topic_counts),
        "feature_map": {
            "motion_alarm": {"topic": "tns1:VideoSource/MotionAlarm", "count": topic_counts.get("tns1:VideoSource/MotionAlarm", 0)},
            "person_detection": {"topic": "tns1:RuleEngine/MyRuleDetector/PeopleDetect", "count": topic_counts.get("tns1:RuleEngine/MyRuleDetector/PeopleDetect", 0)},
            "cell_motion": {"topic": "tns1:RuleEngine/CellMotionDetector/Motion", "count": topic_counts.get("tns1:RuleEngine/CellMotionDetector/Motion", 0)},
        },
        "events": all_ev
    }
    with open(os.path.join(OUTPUT_DIR, "eye_events.json"), "w") as f:
        json.dump(events_out, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [3/4] eye_events.json")

    # 4. eye_report.json
    report = {
        "timestamp": ts(),
        "camera": {"model": "SONOFF CAM-PT2", "fw": "1.0.8", "ip": CAM_IP, "mac": "48:d0:1c:c4:cd:f7"},
        "tests": test_results,
        "positions": all_positions,
        "events_summary": {"total": len(all_ev), "topics": dict(topic_counts)},
    }
    with open(os.path.join(OUTPUT_DIR, "eye_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [4/4] eye_report.json")

    print(f"\n{'='*60}")
    print(f"  FERTIG! {ts()}")
    print(f"  Tests: {len(test_results)}")
    print(f"  Events: {len(all_ev)}")
    print(f"  Topics: {dict(topic_counts)}")
    print(f"  Positionen: {len(all_positions)}")
    print(f"{'='*60}")
