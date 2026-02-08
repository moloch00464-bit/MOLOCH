#!/usr/bin/env python3
"""
M.O.L.O.C.H. Eye -- Strukturiertes ONVIF Testprotokoll
Sonoff CAM-PT2 @ 192.168.178.25

Fuehrt Tests in fester Reihenfolge durch.
Jeder Test wartet auf ENTER, loggt Events + PTZ, speichert Ergebnis.
"""

import json, sys, time, os, threading, signal
from datetime import datetime
from collections import defaultdict

CAM_IP = "192.168.178.25"
CAM_PORT = 80
CAM_USER = "Moloch_4.5"
CAM_PASS = "Auge666"
OUTPUT_DIR = "C:/Users/49179/moloch/config"

# ---- Globals ----
event_log = []
ptz_log = []
test_results = []
running = True

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


# ============================================================================
#  EVENT COLLECTOR (background thread)
# ============================================================================

class EventCollector(threading.Thread):
    """Pulls ONVIF events in background, parses via raw XML."""

    def __init__(self, cam):
        super().__init__(daemon=True)
        from zeep.plugins import HistoryPlugin
        self.cam = cam
        self.history = HistoryPlugin()
        self.pullpoint = cam.create_pullpoint_service()
        self.pullpoint.zeep_client.plugins.append(self.history)
        self.events = []
        self.running = True
        self.event_count = 0

    def run(self):
        from lxml import etree
        ns_tt = '{http://www.onvif.org/ver10/schema}'
        ns_wsnt = '{http://docs.oasis-open.org/wsn/b-2}'

        while self.running:
            try:
                self.pullpoint.PullMessages({'Timeout': 'PT2S', 'MessageLimit': 100})
                if self.history.last_received:
                    raw = self.history.last_received.get('envelope')
                    if raw is not None:
                        for ne in raw.iter(f'{ns_wsnt}NotificationMessage'):
                            self.event_count += 1
                            ev = {"timestamp": ts(), "topic": None, "source": {}, "data": {},
                                  "utc_time": None, "property_operation": None}
                            te = ne.find(f'{ns_wsnt}Topic')
                            if te is not None and te.text:
                                ev["topic"] = te.text.strip()
                            me = ne.find(f'.//{ns_tt}Message')
                            if me is not None:
                                ev["utc_time"] = me.get('UtcTime')
                                ev["property_operation"] = me.get('PropertyOperation')
                                for si in me.findall(f'.//{ns_tt}Source/{ns_tt}SimpleItem'):
                                    ev["source"][si.get('Name','')] = si.get('Value','')
                                for si in me.findall(f'.//{ns_tt}Data/{ns_tt}SimpleItem'):
                                    ev["data"][si.get('Name','')] = si.get('Value','')
                            self.events.append(ev)
                            tshort = ev["topic"].split("/")[-1] if ev["topic"] else "?"
                            print(f"    >> EVENT: {tshort} | {ev['property_operation']} | data={json.dumps(ev['data'])}")
            except Exception as e:
                err = str(e)
                if "terminated" in err.lower() or "invalid" in err.lower():
                    try:
                        self.pullpoint = self.cam.create_pullpoint_service()
                        self.pullpoint.zeep_client.plugins.append(self.history)
                    except:
                        self.running = False
            time.sleep(0.5)

    def get_events_since(self, start_idx):
        return self.events[start_idx:]

    def stop(self):
        self.running = False


# ============================================================================
#  PTZ HELPER
# ============================================================================

def get_ptz_position(ptz_svc, profile_token):
    try:
        status = ptz_svc.GetStatus({'ProfileToken': profile_token})
        pos = status.Position
        pt = getattr(pos, 'PanTilt', None)
        zm = getattr(pos, 'Zoom', None)
        ms = getattr(status, 'MoveStatus', None)
        return {
            "pan": float(getattr(pt, 'x', 0)),
            "tilt": float(getattr(pt, 'y', 0)),
            "zoom": float(getattr(zm, 'x', 0)),
            "move_status": {
                "pan_tilt": str(getattr(ms, 'PanTilt', None)) if ms else None,
                "zoom": str(getattr(ms, 'Zoom', None)) if ms else None
            }
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
#  EINZELNE TESTS
# ============================================================================

def run_test(test_id, name, description, action_fn, wait_for_user=True):
    """Runs one structured test step."""
    print(f"\n{'='*60}")
    print(f"  TEST {test_id}: {name}")
    print(f"  {description}")
    print(f"{'='*60}")

    if wait_for_user:
        input(f"\n  >> ENTER druecken wenn bereit fuer Test {test_id}...")

    print(f"  [{ts()}] Test startet...")
    event_start = collector.event_count
    start_time = ts()

    result = {
        "test_id": test_id,
        "name": name,
        "description": description,
        "start_time": start_time,
        "end_time": None,
        "result": None,
        "events_captured": [],
        "ptz_before": None,
        "ptz_after": None,
        "data": {}
    }

    # PTZ position before
    result["ptz_before"] = get_ptz_position(ptz, main_token)

    # Execute test action
    try:
        test_data = action_fn()
        result["data"] = test_data or {}
    except Exception as e:
        result["data"] = {"error": str(e)}
        print(f"  !! FEHLER: {e}")

    # PTZ position after
    result["ptz_after"] = get_ptz_position(ptz, main_token)

    # Collect events
    result["events_captured"] = collector.get_events_since(event_start)
    result["end_time"] = ts()

    n_events = len(result["events_captured"])
    print(f"  [{ts()}] Test beendet. {n_events} Events erfasst.")

    test_results.append(result)
    return result


# ============================================================================
#  TESTPROTOKOLL - ALLE SCHRITTE
# ============================================================================

def test_01_capabilities():
    """GetCapabilities + GetServiceCapabilities"""
    caps = cam.devicemgmt.GetCapabilities({'Category': 'All'})
    svc_caps = cam.devicemgmt.GetServiceCapabilities()
    data = {
        "capabilities": clean(serialize(caps)),
        "service_capabilities": clean(serialize(svc_caps)),
        "summary": {}
    }
    for svc in ['Analytics','Device','Events','Imaging','Media','PTZ']:
        has = bool(getattr(caps, svc, None))
        data["summary"][svc] = has
        print(f"    {svc}: {'JA' if has else 'nein'}")
    return data

def test_02_device_info():
    """GetDeviceInformation + GetNetworkInterfaces"""
    di = cam.devicemgmt.GetDeviceInformation()
    ni = cam.devicemgmt.GetNetworkInterfaces()
    print(f"    {di.Manufacturer} {di.Model} FW:{di.FirmwareVersion}")
    print(f"    SN: {di.SerialNumber} HW: {di.HardwareId}")
    for iface in ni:
        ip_cfg = getattr(getattr(iface, 'IPv4', None), 'Config', None)
        ip = ip_cfg.Manual[0].Address if ip_cfg and ip_cfg.Manual else '?'
        print(f"    {iface.Info.Name}: {ip} ({iface.Info.HwAddress})")
    return {
        "device_info": clean(serialize(di)),
        "network_interfaces": clean(serialize(ni))
    }

def test_03_media():
    """Media: GetProfiles, GetVideoSources, GetAudioSources, GetStreamUri"""
    profiles = media.GetProfiles()
    vsrc = media.GetVideoSources()
    uris = {}
    try:
        asrc = media.GetAudioSources()
    except:
        asrc = []
    for p in profiles:
        vec = p.VideoEncoderConfiguration
        res = f"{vec.Resolution.Width}x{vec.Resolution.Height}" if vec else "?"
        print(f"    {p.Name} [{p.token}]: {res} {vec.Encoding if vec else '?'}")
        try:
            uri = media.GetStreamUri({
                'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
                'ProfileToken': p.token
            })
            uris[p.token] = uri.Uri
            print(f"      RTSP: {uri.Uri}")
        except Exception as e:
            uris[p.token] = f"ERROR: {e}"
    print(f"    VideoSources: {len(vsrc)}, AudioSources: {len(asrc)}")
    return {
        "profiles": clean(serialize(profiles)),
        "video_sources": clean(serialize(vsrc)),
        "audio_sources": clean(serialize(asrc)),
        "stream_uris": uris
    }

def test_04_ptz_config():
    """PTZ: GetNodes, GetConfigurations (Limits)"""
    nodes = ptz.GetNodes()
    configs = ptz.GetConfigurations()
    for n in nodes:
        print(f"    Node: {n.Name} [{n.token}]")
        print(f"      HomeSupported: {n.HomeSupported}, MaxPresets: {n.MaximumNumberOfPresets}")
        sp = getattr(n, 'SupportedPTZSpaces', None)
        if sp:
            for s in (getattr(sp, 'AbsolutePanTiltPositionSpace', []) or []):
                print(f"      AbsPanTilt: X=[{s.XRange.Min}..{s.XRange.Max}] Y=[{s.YRange.Min}..{s.YRange.Max}]")
            for s in (getattr(sp, 'ContinuousPanTiltVelocitySpace', []) or []):
                print(f"      ContPanTilt: X=[{s.XRange.Min}..{s.XRange.Max}] Y=[{s.YRange.Min}..{s.YRange.Max}]")
    for c in configs:
        ptl = getattr(c, 'PanTiltLimits', None)
        if ptl:
            r = ptl.Range
            print(f"    Config {c.Name}: Pan=[{r.XRange.Min}..{r.XRange.Max}] Tilt=[{r.YRange.Min}..{r.YRange.Max}]")
    return {
        "ptz_nodes": clean(serialize(nodes)),
        "ptz_configurations": clean(serialize(configs))
    }

def test_05_event_subscribe():
    """PullPointSubscription ist bereits aktiv - Status pruefen"""
    print(f"    PullPoint aktiv: {collector.running}")
    print(f"    Events bisher: {collector.event_count}")
    print(f"    Warte 5 Sekunden auf Initial-Events...")
    time.sleep(5)
    new_events = collector.event_count
    print(f"    Events nach 5s: {new_events}")
    return {
        "pullpoint_active": collector.running,
        "initial_events": collector.event_count
    }

def test_06_motion_detection():
    """Motion Detection Test - Bewegung vor der Kamera"""
    print(f"    Bewege dich jetzt vor der Kamera!")
    print(f"    Erfasse Events fuer 15 Sekunden...")
    e_start = collector.event_count
    time.sleep(15)
    new_events = collector.get_events_since(e_start)
    topics = defaultdict(int)
    for ev in new_events:
        t = ev.get("topic","?").split("/")[-1] if ev.get("topic") else "?"
        topics[t] += 1
    print(f"    {len(new_events)} Events erfasst:")
    for t, c in topics.items():
        print(f"      {t}: {c}x")
    return {"events": new_events, "topic_counts": dict(topics)}

def test_07_person_detection():
    """Person Detection Test - Person vor die Kamera stellen"""
    print(f"    Stelle dich klar sichtbar vor die Kamera!")
    print(f"    Erfasse Events fuer 15 Sekunden...")
    e_start = collector.event_count
    time.sleep(15)
    new_events = collector.get_events_since(e_start)
    people_events = [e for e in new_events if e.get("topic") and "PeopleDetect" in e["topic"]]
    print(f"    {len(new_events)} Events total, {len(people_events)} PeopleDetect")
    for pe in people_events:
        print(f"      State={pe['data'].get('State')} @ {pe.get('utc_time')}")
    return {"events": new_events, "people_events": people_events}

def test_08_ptz_app():
    """PTZ ueber Sonoff App steuern"""
    print(f"    Bewege die Kamera jetzt ueber die Sonoff App!")
    print(f"    Links, rechts, hoch, runter testen.")
    print(f"    Tracke Position fuer 30 Sekunden...")
    positions = []
    e_start = collector.event_count
    for i in range(15):
        pos = get_ptz_position(ptz, main_token)
        positions.append({"timestamp": ts(), **pos})
        if i > 0:
            prev = positions[-2]
            dp = abs(pos["pan"] - prev["pan"])
            dt = abs(pos["tilt"] - prev["tilt"])
            if dp > 0.5 or dt > 0.5:
                print(f"    [{ts()}] pan={pos['pan']:.1f} tilt={pos['tilt']:.1f} (d: p={pos['pan']-prev['pan']:+.1f} t={pos['tilt']-prev['tilt']:+.1f})")
        time.sleep(2)
    pans = [p["pan"] for p in positions if "error" not in p]
    tilts = [p["tilt"] for p in positions if "error" not in p]
    print(f"    Pan observed: [{min(pans):.1f} .. {max(pans):.1f}]")
    print(f"    Tilt observed: [{min(tilts):.1f} .. {max(tilts):.1f}]")
    return {"positions": positions, "events": collector.get_events_since(e_start)}

def test_09_ptz_ha():
    """PTZ ueber HA onvif.ptz Service steuern"""
    import requests
    HA_URL = "http://192.168.178.32:8123"
    TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJmZDgyNWI1MDNjMTY0ZDY0YWY0Y2U0NjRkZjkyMTFlNiIsImlhdCI6MTc3MDU0MTg1MCwiZXhwIjoyMDg1OTAxODUwfQ.hlvHR8U3pZ-1kiLDw64YMHBqdROs4j2TsxnbcKhrj5Q"
    HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

    print(f"    Teste PTZ ueber HA API...")
    pos_before = get_ptz_position(ptz, main_token)
    print(f"    Vor: pan={pos_before['pan']:.1f} tilt={pos_before['tilt']:.1f}")

    # RelativeMove RIGHT
    try:
        r = requests.post(f"{HA_URL}/api/services/onvif/ptz", headers=HEADERS, json={
            "entity_id": "camera.cam_pt2_mainstream",
            "pan": "RIGHT", "tilt": "UP", "distance": 0.3,
            "speed": 0.5, "move_mode": "RelativeMove"
        }, timeout=10)
        print(f"    HA PTZ call: {r.status_code}")
    except Exception as e:
        print(f"    HA PTZ error: {e}")

    time.sleep(3)
    pos_after = get_ptz_position(ptz, main_token)
    print(f"    Nach: pan={pos_after['pan']:.1f} tilt={pos_after['tilt']:.1f}")
    print(f"    Delta: pan={pos_after['pan']-pos_before['pan']:+.1f} tilt={pos_after['tilt']-pos_before['tilt']:+.1f}")

    return {"pos_before": pos_before, "pos_after": pos_after, "method": "HA onvif.ptz RelativeMove"}

def test_10_ptz_onvif_direct():
    """PTZ direkt ueber ONVIF: Absolute, Relative, Continuous, Home"""
    results_ptz = {}

    # Current position
    pos0 = get_ptz_position(ptz, main_token)
    print(f"    Start: pan={pos0['pan']:.1f} tilt={pos0['tilt']:.1f}")

    # AbsoluteMove to 0,0
    print(f"    AbsoluteMove -> (0, 0)...")
    ptz.AbsoluteMove({'ProfileToken': main_token, 'Position': {'PanTilt': {'x': 0.0, 'y': 0.0}, 'Zoom': {'x': 0.0}}})
    time.sleep(3)
    pos1 = get_ptz_position(ptz, main_token)
    print(f"    Position: pan={pos1['pan']:.1f} tilt={pos1['tilt']:.1f}")
    results_ptz["absolute_0_0"] = pos1

    # RelativeMove +50 pan
    print(f"    RelativeMove +50 pan...")
    ptz.RelativeMove({'ProfileToken': main_token, 'Translation': {'PanTilt': {'x': 50.0, 'y': 0.0}, 'Zoom': {'x': 0.0}}})
    time.sleep(3)
    pos2 = get_ptz_position(ptz, main_token)
    print(f"    Position: pan={pos2['pan']:.1f} tilt={pos2['tilt']:.1f}")
    results_ptz["relative_plus50"] = pos2

    # ContinuousMove left for 1.5s
    print(f"    ContinuousMove links 1.5s...")
    ptz.ContinuousMove({'ProfileToken': main_token, 'Velocity': {'PanTilt': {'x': -0.5, 'y': 0.0}, 'Zoom': {'x': 0.0}}})
    time.sleep(1.5)
    ptz.Stop({'ProfileToken': main_token, 'PanTilt': True, 'Zoom': True})
    time.sleep(1)
    pos3 = get_ptz_position(ptz, main_token)
    print(f"    Position: pan={pos3['pan']:.1f} tilt={pos3['tilt']:.1f}")
    results_ptz["continuous_left"] = pos3

    # GotoHomePosition
    print(f"    GotoHomePosition...")
    try:
        ptz.GotoHomePosition({'ProfileToken': main_token})
        time.sleep(3)
        pos4 = get_ptz_position(ptz, main_token)
        print(f"    Home: pan={pos4['pan']:.1f} tilt={pos4['tilt']:.1f}")
        results_ptz["home"] = pos4
    except Exception as e:
        print(f"    Home error: {e}")
        results_ptz["home"] = {"error": str(e)}

    return results_ptz

def test_11_presets():
    """PTZ Presets: SetPreset, GotoPreset, GetPresets"""
    results_pre = {}

    # Check existing presets
    presets = ptz.GetPresets({'ProfileToken': main_token})
    print(f"    Bestehende Presets: {len(presets) if presets else 0}")
    results_pre["existing_presets"] = clean(serialize(presets))

    # Move to known position and create preset
    print(f"    Fahre zu Pan=50, Tilt=-20...")
    ptz.AbsoluteMove({'ProfileToken': main_token, 'Position': {'PanTilt': {'x': 50.0, 'y': -20.0}, 'Zoom': {'x': 0.0}}})
    time.sleep(3)

    print(f"    Erstelle Preset 'moloch_test_1'...")
    try:
        preset_token = ptz.SetPreset({'ProfileToken': main_token, 'PresetName': 'moloch_test_1'})
        print(f"    Preset erstellt: {preset_token}")
        results_pre["set_preset_1"] = {"token": str(preset_token), "position": get_ptz_position(ptz, main_token)}
    except Exception as e:
        print(f"    SetPreset error: {e}")
        results_pre["set_preset_1"] = {"error": str(e)}

    # Move away
    ptz.AbsoluteMove({'ProfileToken': main_token, 'Position': {'PanTilt': {'x': -80.0, 'y': 30.0}, 'Zoom': {'x': 0.0}}})
    time.sleep(3)

    # Second preset
    print(f"    Erstelle Preset 'moloch_test_2' bei (-80, 30)...")
    try:
        preset_token2 = ptz.SetPreset({'ProfileToken': main_token, 'PresetName': 'moloch_test_2'})
        print(f"    Preset erstellt: {preset_token2}")
        results_pre["set_preset_2"] = {"token": str(preset_token2), "position": get_ptz_position(ptz, main_token)}
    except Exception as e:
        print(f"    SetPreset 2 error: {e}")
        results_pre["set_preset_2"] = {"error": str(e)}

    # GotoPreset 1
    if "token" in results_pre.get("set_preset_1", {}):
        print(f"    GotoPreset 'moloch_test_1'...")
        try:
            ptz.GotoPreset({'ProfileToken': main_token, 'PresetToken': results_pre["set_preset_1"]["token"]})
            time.sleep(3)
            pos = get_ptz_position(ptz, main_token)
            print(f"    Position: pan={pos['pan']:.1f} tilt={pos['tilt']:.1f}")
            results_pre["goto_preset_1"] = pos
        except Exception as e:
            print(f"    GotoPreset error: {e}")

    # Final GetPresets
    presets_final = ptz.GetPresets({'ProfileToken': main_token})
    results_pre["final_presets"] = []
    print(f"    Finale Presets: {len(presets_final) if presets_final else 0}")
    for pr in (presets_final or []):
        pd = {"token": getattr(pr, 'token', '?'), "name": getattr(pr, 'Name', '?')}
        pos = getattr(pr, 'PTZPosition', None)
        if pos:
            pt = getattr(pos, 'PanTilt', None)
            pd["pan"] = getattr(pt, 'x', None)
            pd["tilt"] = getattr(pt, 'y', None)
            zm = getattr(pos, 'Zoom', None)
            pd["zoom"] = getattr(zm, 'x', None)
        results_pre["final_presets"].append(pd)
        print(f"      [{pd['token']}] {pd['name']}: pan={pd.get('pan')} tilt={pd.get('tilt')}")

    return results_pre

def test_12_nightvision():
    """Nachtsicht / IR Test - Imaging Settings abfragen"""
    print(f"    Imaging Service check...")
    try:
        img = cam.create_imaging_service()
        settings = img.GetImagingSettings({'VideoSourceToken': 'VideoSourceToken'})
        data = clean(serialize(settings))
        print(f"    Brightness: {getattr(settings, 'Brightness', '?')}")
        print(f"    Contrast: {getattr(settings, 'Contrast', '?')}")
        print(f"    IrCutFilter: {getattr(settings, 'IrCutFilter', '?')}")
        return {"imaging_supported": True, "settings": data}
    except Exception as e:
        print(f"    Imaging nicht unterstuetzt: {e}")
        print(f"    -> Nachtsicht nur ueber Sonoff App steuerbar")
        return {"imaging_supported": False, "error": str(e)}

def test_13_cell_motion():
    """Cell Motion Detection Test"""
    print(f"    Langsame Bewegung im Randbereich des Bildes.")
    print(f"    Erfasse Events fuer 15 Sekunden...")
    e_start = collector.event_count
    time.sleep(15)
    new_events = collector.get_events_since(e_start)
    cell_events = [e for e in new_events if e.get("topic") and "CellMotion" in e["topic"]]
    motion_events = [e for e in new_events if e.get("topic") and "MotionAlarm" in e["topic"]]
    print(f"    Total: {len(new_events)}, CellMotion: {len(cell_events)}, MotionAlarm: {len(motion_events)}")
    return {"events": new_events, "cell_motion_events": cell_events, "motion_events": motion_events}

def test_14_ptz_patrol_app():
    """PTZ Patrol / Auto-Tracking ueber App testen"""
    print(f"    Aktiviere Patrol oder Auto-Tracking in der Sonoff App!")
    print(f"    Tracke Position fuer 30 Sekunden...")
    positions = []
    e_start = collector.event_count
    for i in range(15):
        pos = get_ptz_position(ptz, main_token)
        positions.append({"timestamp": ts(), **pos})
        if i > 0 and "error" not in pos:
            prev = positions[-2]
            dp = abs(pos["pan"] - prev["pan"])
            dt = abs(pos["tilt"] - prev["tilt"])
            if dp > 0.5 or dt > 0.5:
                print(f"    [{ts()}] pan={pos['pan']:.1f} tilt={pos['tilt']:.1f}")
        time.sleep(2)
    return {"positions": positions, "events": collector.get_events_since(e_start)}

def test_15_center_reset():
    """Kamera auf Ausgangsposition (0,0) zurueckfahren"""
    print(f"    AbsoluteMove -> (0, 0)...")
    ptz.AbsoluteMove({'ProfileToken': main_token, 'Position': {'PanTilt': {'x': 0.0, 'y': 0.0}, 'Zoom': {'x': 0.0}}})
    time.sleep(3)
    pos = get_ptz_position(ptz, main_token)
    print(f"    Endposition: pan={pos['pan']:.1f} tilt={pos['tilt']:.1f}")
    return {"final_position": pos}


# ============================================================================
#  MAIN
# ============================================================================

if __name__ == "__main__":
    from onvif import ONVIFCamera

    print(f"""
    +--------------------------------------------------+
    |  M.O.L.O.C.H. Eye -- Strukturiertes Testprotokoll|
    |  Sonoff CAM-PT2 @ {CAM_IP}              |
    |  {ts()}                      |
    +--------------------------------------------------+
    """)

    # Connect
    print("[INIT] Verbinde mit Kamera...")
    cam = ONVIFCamera(CAM_IP, CAM_PORT, CAM_USER, CAM_PASS)
    media = cam.create_media_service()
    ptz = cam.create_ptz_service()
    profiles = media.GetProfiles()
    main_token = profiles[0].token
    print(f"[INIT] Verbunden. Hauptprofil: {profiles[0].Name} [{main_token}]")

    # Start event collector
    print("[INIT] Starte Event-Collector...")
    collector = EventCollector(cam)
    collector.start()
    time.sleep(2)
    print(f"[INIT] Event-Collector aktiv. ({collector.event_count} initial events)")

    # =============================================
    #  PHASE 1 - VOR DEN TESTS
    # =============================================
    print(f"\n\n{'#'*60}")
    print(f"  PHASE 1 -- VOR DEN TESTS (ONVIF Discovery)")
    print(f"{'#'*60}")

    run_test("1.1", "GetCapabilities + GetServiceCapabilities",
             "ONVIF Faehigkeiten der Kamera abfragen", test_01_capabilities, wait_for_user=False)

    run_test("1.2", "GetDeviceInformation + GetNetworkInterfaces",
             "Geraeteinfos und Netzwerk", test_02_device_info, wait_for_user=False)

    run_test("1.3", "Media: Profiles, Sources, StreamUri",
             "Medienprofile und Stream-URIs", test_03_media, wait_for_user=False)

    run_test("1.4", "PTZ: Nodes, Configurations, Limits",
             "PTZ Konfiguration und Bewegungsgrenzen", test_04_ptz_config, wait_for_user=False)

    run_test("1.5", "PullPointSubscription Status",
             "Event-Subscription laeuft bereits", test_05_event_subscribe, wait_for_user=False)

    # =============================================
    #  PHASE 2 - WAEHREND DER TESTS
    # =============================================
    print(f"\n\n{'#'*60}")
    print(f"  PHASE 2 -- INTERAKTIVE TESTS")
    print(f"  Events werden durchgehend im Hintergrund geloggt.")
    print(f"{'#'*60}")

    run_test("2.1", "Motion Detection",
             "Laufe vor der Kamera hin und her. 15s Aufnahme.",
             test_06_motion_detection)

    run_test("2.2", "Person Detection",
             "Stelle dich klar sichtbar vor die Kamera. 15s Aufnahme.",
             test_07_person_detection)

    run_test("2.3", "Cell Motion Detection",
             "Langsame Bewegung am Bildrand. 15s Aufnahme.",
             test_13_cell_motion)

    run_test("2.4", "Nachtsicht / Imaging",
             "Imaging-Settings abfragen (IR, Brightness, Contrast).",
             test_12_nightvision, wait_for_user=False)

    run_test("2.5", "PTZ ueber Sonoff App",
             "Bewege die Kamera in der Sonoff App (alle Richtungen). 30s Tracking.",
             test_08_ptz_app)

    run_test("2.6", "PTZ ueber HA (onvif.ptz)",
             "PTZ-Steuerung ueber Home Assistant API testen.",
             test_09_ptz_ha, wait_for_user=False)

    run_test("2.7", "PTZ direkt ONVIF (Absolute/Relative/Continuous/Home)",
             "Alle PTZ-Bewegungsmodi direkt ueber ONVIF testen.",
             test_10_ptz_onvif_direct, wait_for_user=False)

    run_test("2.8", "PTZ Presets (Set/Goto/Get)",
             "Presets erstellen und anfahren ueber ONVIF.",
             test_11_presets, wait_for_user=False)

    run_test("2.9", "Patrol / Auto-Tracking (App)",
             "Patrol oder Auto-Tracking in App aktivieren. 30s Tracking.",
             test_14_ptz_patrol_app)

    run_test("2.10", "Reset: Kamera auf (0,0)",
             "Kamera zurueck auf Ausgangsposition.",
             test_15_center_reset, wait_for_user=False)

    # =============================================
    #  PHASE 3 - NACH DEN TESTS
    # =============================================
    print(f"\n\n{'#'*60}")
    print(f"  PHASE 3 -- REPORT & SPEICHERN")
    print(f"{'#'*60}")

    collector.stop()

    # Collect all events
    all_events = collector.events
    all_topics = defaultdict(int)
    for ev in all_events:
        t = ev.get("topic", "?")
        all_topics[t] += 1

    # Feature -> Event Mapping
    feature_map = {}
    topic_to_feature = {
        "tns1:VideoSource/MotionAlarm": "motion_alarm",
        "tns1:RuleEngine/MyRuleDetector/PeopleDetect": "person_detection",
        "tns1:RuleEngine/CellMotionDetector/Motion": "cell_motion_detection"
    }
    for topic, feat in topic_to_feature.items():
        evts = [e for e in all_events if e.get("topic") == topic]
        feature_map[feat] = {
            "onvif_topic": topic,
            "event_count": len(evts),
            "triggered": len(evts) > 0,
            "sample": evts[0] if evts else None
        }

    # Build 4 output files

    # 1. eye_capabilities.json
    caps_data = {}
    for tr in test_results:
        if tr["test_id"] in ("1.1", "1.2", "1.3", "1.4"):
            caps_data[tr["name"]] = tr["data"]

    # Merge with existing capabilities
    try:
        with open(os.path.join(OUTPUT_DIR, "eye_capabilities.json"), "r", encoding="utf-8") as f:
            existing_caps = json.load(f)
    except:
        existing_caps = {}

    existing_caps["test_protocol"] = {
        "timestamp": ts(),
        "phase1_discovery": caps_data,
        "feature_event_mapping": feature_map,
        "total_events_captured": len(all_events),
        "unique_topics": dict(all_topics),
        "ptz_tests": {
            tr["test_id"]: {"name": tr["name"], "data": tr["data"]}
            for tr in test_results if tr["test_id"].startswith("2.") and "ptz" in tr["name"].lower()
        },
        "access_report": {
            "onvif_local": [
                "RTSP Stream (main 1080p + minor 360p)",
                "PTZ: Absolute, Relative, Continuous, Stop, Home",
                "PTZ: SetPreset, GotoPreset, GetPresets",
                "Events: MotionAlarm, PeopleDetect, CellMotionDetector",
                "Device: GetInfo, Reboot, SetSystemDateAndTime",
                "Media: Profiles, Sources, StreamUri"
            ],
            "ha_local": [
                "camera.turn_on/off, snapshot, record, play_stream",
                "onvif.ptz (alle Move-Modi)",
                "binary_sensor: motion_alarm, person_detection, cell_motion",
                "button: reboot, set_time"
            ],
            "app_only": [
                "Imaging (Brightness, Contrast, Saturation)",
                "Nachtsicht-Modus umschalten",
                "WiFi-Konfiguration",
                "Firmware-Update",
                "Cloud-Speicher",
                "Audio Talkback",
                "Patrol-Routen konfigurieren",
                "Video-Qualitaet Presets"
            ]
        }
    }

    with open(os.path.join(OUTPUT_DIR, "eye_capabilities.json"), "w", encoding="utf-8") as f:
        json.dump(existing_caps, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [1/4] eye_capabilities.json gespeichert")

    # 2. eye_presets.json
    preset_test = next((tr for tr in test_results if tr["test_id"] == "2.8"), None)
    presets_data = {
        "timestamp": ts(),
        "camera": f"SONOFF CAM-PT2 @ {CAM_IP}",
        "ptz_limits": {
            "pan": {"min": -168.4, "max": 174.4},
            "tilt": {"min": -78.8, "max": 101.3},
            "zoom": {"min": 0.0, "max": 0.0}
        },
        "presets": preset_test["data"].get("final_presets", []) if preset_test else [],
        "preset_test_results": preset_test["data"] if preset_test else {},
        "home_position": None
    }
    home_test = next((tr for tr in test_results if tr["test_id"] == "2.7"), None)
    if home_test and "home" in home_test.get("data", {}):
        presets_data["home_position"] = home_test["data"]["home"]

    with open(os.path.join(OUTPUT_DIR, "eye_presets.json"), "w", encoding="utf-8") as f:
        json.dump(presets_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [2/4] eye_presets.json gespeichert")

    # 3. eye_events.json
    events_data = {
        "timestamp": ts(),
        "camera": f"SONOFF CAM-PT2 @ {CAM_IP}",
        "total_events": len(all_events),
        "topic_counts": dict(all_topics),
        "feature_event_mapping": feature_map,
        "all_events": all_events
    }
    with open(os.path.join(OUTPUT_DIR, "eye_events.json"), "w", encoding="utf-8") as f:
        json.dump(events_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [3/4] eye_events.json gespeichert")

    # 4. eye_report.json
    report = {
        "timestamp": ts(),
        "camera": {"model": "SONOFF CAM-PT2", "firmware": "1.0.8", "ip": CAM_IP,
                    "mac": "48:d0:1c:c4:cd:f7", "serial": "25370200016333"},
        "test_results": test_results,
        "all_events": all_events,
        "summary": {
            "tests_run": len(test_results),
            "events_total": len(all_events),
            "topics_seen": dict(all_topics)
        }
    }
    with open(os.path.join(OUTPUT_DIR, "eye_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [4/4] eye_report.json gespeichert")

    print(f"\n{'='*60}")
    print(f"  TESTPROTOKOLL ABGESCHLOSSEN")
    print(f"  {ts()}")
    print(f"  Tests: {len(test_results)}")
    print(f"  Events: {len(all_events)}")
    print(f"  Topics: {dict(all_topics)}")
    print(f"\n  Dateien:")
    print(f"    1. eye_capabilities.json")
    print(f"    2. eye_presets.json")
    print(f"    3. eye_events.json")
    print(f"    4. eye_report.json")
    print(f"{'='*60}")
