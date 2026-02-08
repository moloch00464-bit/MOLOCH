#!/usr/bin/env python3
"""Phase 2: ONVIF Discovery for Sonoff CAM-PT2 — M.O.L.O.C.H. Eye"""
import json, sys, traceback
from datetime import datetime

CAM_IP = "192.168.178.25"
CAM_PORT = 80
CAM_USER = "Moloch_4.5"
CAM_PASS = "Auge666"

results = {
    "scan_timestamp": datetime.now().isoformat(),
    "camera_ip": CAM_IP,
    "camera_port": CAM_PORT,
    "device_info": None,
    "capabilities": None,
    "service_capabilities": None,
    "network_interfaces": None,
    "media_profiles": None,
    "video_sources": None,
    "audio_sources": None,
    "stream_uris": {},
    "ptz_nodes": None,
    "ptz_configurations": None,
    "ptz_presets": {},
    "imaging_options": None,
    "imaging_settings": None,
    "event_properties": None,
    "errors": []
}

presets_result = {
    "scan_timestamp": datetime.now().isoformat(),
    "camera_ip": CAM_IP,
    "presets": {}
}

def serialize(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [serialize(i) for i in obj]
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    # Try zeep serialize first
    try:
        from zeep.helpers import serialize_object
        s = serialize_object(obj)
        if isinstance(s, dict):
            return {k: serialize(v) for k, v in s.items()}
        if isinstance(s, list):
            return [serialize(i) for i in s]
        return s
    except:
        pass
    if hasattr(obj, '__dict__'):
        d = {}
        for k, v in obj.__dict__.items():
            if not k.startswith('_'):
                d[k] = serialize(v)
        return d if d else str(obj)
    return str(obj)

try:
    from onvif import ONVIFCamera
    print(f"Connecting to {CAM_IP}:{CAM_PORT}...")
    cam = ONVIFCamera(CAM_IP, CAM_PORT, CAM_USER, CAM_PASS)
    print("Connected!\n")

    # [1] DeviceInformation
    print("[1/10] GetDeviceInformation...")
    try:
        di = cam.devicemgmt.GetDeviceInformation()
        results["device_info"] = serialize(di)
        print(f"  Manufacturer: {di.Manufacturer}")
        print(f"  Model: {di.Model}")
        print(f"  Firmware: {di.FirmwareVersion}")
        print(f"  Serial: {di.SerialNumber}")
        print(f"  HardwareId: {di.HardwareId}")
    except Exception as e:
        results["errors"].append(f"DeviceInfo: {e}")
        print(f"  ERROR: {e}")

    # [2] Capabilities
    print("\n[2/10] GetCapabilities...")
    try:
        caps = cam.devicemgmt.GetCapabilities({'Category': 'All'})
        results["capabilities"] = serialize(caps)
        for cat in ['Analytics','Device','Events','Imaging','Media','PTZ','Extension']:
            val = getattr(caps, cat, None)
            print(f"  {cat}: {'YES' if val else 'no'}")
    except Exception as e:
        results["errors"].append(f"Capabilities: {e}")
        print(f"  ERROR: {e}")

    # [3] NetworkInterfaces
    print("\n[3/10] GetNetworkInterfaces...")
    try:
        net = cam.devicemgmt.GetNetworkInterfaces()
        results["network_interfaces"] = serialize(net)
        print(f"  Interfaces: {len(net) if net else 0}")
    except Exception as e:
        results["errors"].append(f"NetworkInterfaces: {e}")
        print(f"  ERROR: {e}")

    # [4] Media Profiles + Stream URIs
    print("\n[4/10] Media: GetProfiles + GetStreamUri...")
    profiles = []
    try:
        media = cam.create_media_service()
        profiles = media.GetProfiles()
        results["media_profiles"] = serialize(profiles)
        print(f"  Profiles: {len(profiles)}")
        for p in profiles:
            print(f"    - {p.Name} (token={p.token})")
            try:
                uri = media.GetStreamUri({
                    'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
                    'ProfileToken': p.token
                })
                results["stream_uris"][p.token] = serialize(uri)
                print(f"      RTSP: {uri.Uri}")
            except Exception as e:
                results["errors"].append(f"StreamUri({p.token}): {e}")
                print(f"      StreamUri ERROR: {e}")
    except Exception as e:
        results["errors"].append(f"MediaProfiles: {e}")
        print(f"  ERROR: {e}")

    # [5] VideoSources
    print("\n[5/10] Media: GetVideoSources...")
    vsources = []
    try:
        vsources = media.GetVideoSources()
        results["video_sources"] = serialize(vsources)
        for vs in vsources:
            print(f"  - token={vs.token}, {vs.Resolution.Width}x{vs.Resolution.Height}")
    except Exception as e:
        results["errors"].append(f"VideoSources: {e}")
        print(f"  ERROR: {e}")

    # [6] AudioSources
    print("\n[6/10] Media: GetAudioSources...")
    try:
        asrc = media.GetAudioSources()
        results["audio_sources"] = serialize(asrc)
        print(f"  Audio sources: {len(asrc) if asrc else 0}")
    except Exception as e:
        results["errors"].append(f"AudioSources: {e}")
        print(f"  ERROR: {e}")

    # [7] PTZ
    print("\n[7/10] PTZ: Nodes, Configurations, Presets...")
    try:
        ptz = cam.create_ptz_service()

        # Nodes
        try:
            nodes = ptz.GetNodes()
            results["ptz_nodes"] = serialize(nodes)
            print(f"  Nodes: {len(nodes)}")
            for n in nodes:
                print(f"    - {n.Name} (token={n.token})")
                sp = getattr(n, 'SupportedPTZSpaces', None)
                if sp:
                    apt = getattr(sp, 'AbsolutePanTiltPositionSpace', None)
                    if apt:
                        for s in apt:
                            print(f"      AbsPanTilt: X=[{s.XRange.Min}..{s.XRange.Max}] Y=[{s.YRange.Min}..{s.YRange.Max}]")
                    azm = getattr(sp, 'AbsoluteZoomPositionSpace', None)
                    if azm:
                        for s in azm:
                            print(f"      AbsZoom: [{s.XRange.Min}..{s.XRange.Max}]")
                    cpt = getattr(sp, 'ContinuousPanTiltVelocitySpace', None)
                    if cpt:
                        for s in cpt:
                            print(f"      ContPanTilt: X=[{s.XRange.Min}..{s.XRange.Max}] Y=[{s.YRange.Min}..{s.YRange.Max}]")
                    czm = getattr(sp, 'ContinuousZoomVelocitySpace', None)
                    if czm:
                        for s in czm:
                            print(f"      ContZoom: [{s.XRange.Min}..{s.XRange.Max}]")
        except Exception as e:
            results["errors"].append(f"PTZNodes: {e}")
            print(f"  Nodes ERROR: {e}")

        # Configurations
        try:
            configs = ptz.GetConfigurations()
            results["ptz_configurations"] = serialize(configs)
            print(f"  Configurations: {len(configs)}")
            for c in configs:
                print(f"    - {c.Name} (token={c.token})")
                ptl = getattr(c, 'PanTiltLimits', None)
                if ptl:
                    r = ptl.Range
                    print(f"      PanTilt Limits: X=[{r.XRange.Min}..{r.XRange.Max}] Y=[{r.YRange.Min}..{r.YRange.Max}]")
                zl = getattr(c, 'ZoomLimits', None)
                if zl:
                    r = zl.Range
                    print(f"      Zoom Limits: [{r.XRange.Min}..{r.XRange.Max}]")
        except Exception as e:
            results["errors"].append(f"PTZConfigs: {e}")
            print(f"  Configs ERROR: {e}")

        # Presets
        for p in profiles:
            try:
                presets = ptz.GetPresets({'ProfileToken': p.token})
                pl = []
                for pr in (presets or []):
                    pd = {
                        "token": getattr(pr, 'token', str(pr)),
                        "name": getattr(pr, 'Name', None),
                        "pan": None, "tilt": None, "zoom": None
                    }
                    pos = getattr(pr, 'PTZPosition', None)
                    if pos:
                        pt = getattr(pos, 'PanTilt', None)
                        if pt:
                            pd["pan"] = getattr(pt, 'x', None)
                            pd["tilt"] = getattr(pt, 'y', None)
                        zm = getattr(pos, 'Zoom', None)
                        if zm:
                            pd["zoom"] = getattr(zm, 'x', None)
                    pl.append(pd)
                results["ptz_presets"][p.token] = pl
                presets_result["presets"][p.token] = pl
                print(f"  Presets for {p.Name}: {len(pl)}")
                for pr in pl:
                    print(f"    [{pr['token']}] {pr['name']}: pan={pr['pan']} tilt={pr['tilt']} zoom={pr['zoom']}")
            except Exception as e:
                results["errors"].append(f"Presets({p.token}): {e}")
                print(f"  Presets ERROR {p.Name}: {e}")

    except Exception as e:
        results["errors"].append(f"PTZService: {e}")
        print(f"  PTZ ERROR: {e}")

    # [8] Imaging
    print("\n[8/10] Imaging...")
    try:
        img = cam.create_imaging_service()
        for vs in vsources:
            try:
                settings = img.GetImagingSettings({'VideoSourceToken': vs.token})
                results["imaging_settings"] = serialize(settings)
                print(f"  Brightness={getattr(settings,'Brightness','?')} Contrast={getattr(settings,'Contrast','?')} Saturation={getattr(settings,'ColorSaturation','?')}")
            except Exception as e:
                results["errors"].append(f"ImagingSettings: {e}")
                print(f"  Settings ERROR: {e}")
            try:
                opts = img.GetOptions({'VideoSourceToken': vs.token})
                results["imaging_options"] = serialize(opts)
                print(f"  Imaging options retrieved")
            except Exception as e:
                results["errors"].append(f"ImagingOptions: {e}")
                print(f"  Options ERROR: {e}")
    except Exception as e:
        results["errors"].append(f"ImagingService: {e}")
        print(f"  Imaging ERROR: {e}")

    # [9] Events
    print("\n[9/10] Events: GetEventProperties...")
    try:
        evt = cam.create_events_service()
        props = evt.GetEventProperties()
        results["event_properties"] = serialize(props)
        print(f"  Event properties retrieved")
    except Exception as e:
        results["errors"].append(f"EventProperties: {e}")
        print(f"  ERROR: {e}")

    # [10] ServiceCapabilities
    print("\n[10/10] GetServiceCapabilities...")
    try:
        sc = cam.devicemgmt.GetServiceCapabilities()
        results["service_capabilities"] = serialize(sc)
        print(f"  Service capabilities retrieved")
    except Exception as e:
        results["errors"].append(f"ServiceCapabilities: {e}")
        print(f"  ERROR: {e}")

except Exception as e:
    results["errors"].append(f"Connection: {e}")
    print(f"FATAL: {e}")
    traceback.print_exc()

# Save
with open("C:/Users/49179/moloch/config/onvif_phase2_raw.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
with open("C:/Users/49179/moloch/config/eye_presets_raw.json", "w", encoding="utf-8") as f:
    json.dump(presets_result, f, indent=2, ensure_ascii=False, default=str)

print(f"\n{'='*50}")
print(f"Phase 2 complete. Errors: {len(results['errors'])}")
for e in results["errors"]:
    print(f"  ! {e}")
