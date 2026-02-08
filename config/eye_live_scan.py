#!/usr/bin/env python3
"""
M.O.L.O.C.H. Eye — Live ONVIF Test Protocol
Sonoff CAM-PT2 @ 192.168.178.25

Phase 1: Pre-test discovery
Phase 2: Live event monitoring + PTZ tracking
Phase 3: Report generation

Usage:
  python eye_live_scan.py              # Full run (Phase 1 + 2 + 3)
  python eye_live_scan.py --phase1     # Only discovery
  python eye_live_scan.py --monitor    # Only live monitor (Phase 2)
  python eye_live_scan.py --report     # Only generate report (Phase 3)
"""

import json, sys, time, os, traceback, signal, threading
from datetime import datetime, timezone
from collections import defaultdict

# ─── Config ───────────────────────────────────────────────────────────────────
CAM_IP = "192.168.178.25"
CAM_PORT = 80
CAM_USER = "Moloch_4.5"
CAM_PASS = "Auge666"
POLL_INTERVAL = 2  # seconds
OUTPUT_DIR = "C:/Users/49179/moloch/config"

# ─── Globals ──────────────────────────────────────────────────────────────────
running = True
event_log = []
ptz_log = []
imaging_log = []
discovery_data = {}

def timestamp():
    return datetime.now().isoformat(timespec='milliseconds')

def serialize(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [serialize(i) for i in obj]
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
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

def clean_dict(d):
    """Remove zeep internal keys."""
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items()
                if k not in ('_value_1', '_attr_1') and v is not None}
    if isinstance(d, list):
        return [clean_dict(i) for i in d]
    return d

def signal_handler(sig, frame):
    global running
    print(f"\n[{timestamp()}] STOP signal received. Finishing up...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — Pre-Test Discovery
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_discovery():
    global discovery_data
    from onvif import ONVIFCamera

    print(f"\n{'='*60}")
    print(f"  PHASE 1 — ONVIF Pre-Test Discovery")
    print(f"  {timestamp()}")
    print(f"{'='*60}\n")

    cam = ONVIFCamera(CAM_IP, CAM_PORT, CAM_USER, CAM_PASS)
    discovery_data = {
        "scan_timestamp": timestamp(),
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
        "ptz_status": None,
        "event_properties": None,
        "errors": []
    }

    # [1] GetCapabilities
    print("[1/6] GetCapabilities + GetServiceCapabilities...")
    try:
        caps = cam.devicemgmt.GetCapabilities({'Category': 'All'})
        discovery_data["capabilities"] = clean_dict(serialize(caps))
        for svc in ['Analytics','Device','Events','Imaging','Media','PTZ']:
            print(f"  {svc}: {'YES' if getattr(caps, svc, None) else 'no'}")
    except Exception as e:
        discovery_data["errors"].append(f"Capabilities: {e}")
        print(f"  ERROR: {e}")

    try:
        sc = cam.devicemgmt.GetServiceCapabilities()
        discovery_data["service_capabilities"] = clean_dict(serialize(sc))
        print("  ServiceCapabilities: OK")
    except Exception as e:
        discovery_data["errors"].append(f"ServiceCapabilities: {e}")
        print(f"  ServiceCapabilities ERROR: {e}")

    # [2] DeviceInformation + NetworkInterfaces
    print("\n[2/6] GetDeviceInformation + GetNetworkInterfaces...")
    try:
        di = cam.devicemgmt.GetDeviceInformation()
        discovery_data["device_info"] = clean_dict(serialize(di))
        print(f"  {di.Manufacturer} {di.Model} FW:{di.FirmwareVersion} SN:{di.SerialNumber}")
    except Exception as e:
        discovery_data["errors"].append(f"DeviceInfo: {e}")
        print(f"  ERROR: {e}")

    try:
        ni = cam.devicemgmt.GetNetworkInterfaces()
        discovery_data["network_interfaces"] = clean_dict(serialize(ni))
        for iface in ni:
            mac = iface.Info.HwAddress if hasattr(iface, 'Info') else '?'
            ip_cfg = getattr(getattr(iface, 'IPv4', None), 'Config', None)
            ip = 'unknown'
            if ip_cfg and ip_cfg.Manual:
                ip = ip_cfg.Manual[0].Address
            print(f"  {iface.Info.Name}: {ip} ({mac})")
    except Exception as e:
        discovery_data["errors"].append(f"NetworkInterfaces: {e}")
        print(f"  ERROR: {e}")

    # [3] Media
    print("\n[3/6] Media: Profiles, VideoSources, AudioSources, StreamUri...")
    try:
        media = cam.create_media_service()
        profiles = media.GetProfiles()
        discovery_data["media_profiles"] = clean_dict(serialize(profiles))
        print(f"  {len(profiles)} profiles:")
        for p in profiles:
            vec = p.VideoEncoderConfiguration
            res = f"{vec.Resolution.Width}x{vec.Resolution.Height}" if vec else "?"
            enc = vec.Encoding if vec else "?"
            print(f"    {p.Name} [{p.token}]: {res} {enc}")
            try:
                uri = media.GetStreamUri({
                    'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
                    'ProfileToken': p.token
                })
                discovery_data["stream_uris"][p.token] = serialize(uri.Uri)
                print(f"      RTSP: {uri.Uri}")
            except Exception as e:
                discovery_data["errors"].append(f"StreamUri({p.token}): {e}")

        vsrc = media.GetVideoSources()
        discovery_data["video_sources"] = clean_dict(serialize(vsrc))
        for v in vsrc:
            print(f"  VideoSource: {v.token} {v.Resolution.Width}x{v.Resolution.Height}@{v.Framerate}fps")

        try:
            asrc = media.GetAudioSources()
            discovery_data["audio_sources"] = clean_dict(serialize(asrc))
            print(f"  AudioSources: {len(asrc)}")
        except Exception as e:
            discovery_data["errors"].append(f"AudioSources: {e}")
            print(f"  AudioSources: {e}")

    except Exception as e:
        discovery_data["errors"].append(f"Media: {e}")
        print(f"  ERROR: {e}")

    # [4] PTZ
    print("\n[4/6] PTZ: Nodes, Configurations, Limits...")
    try:
        ptz = cam.create_ptz_service()
        nodes = ptz.GetNodes()
        discovery_data["ptz_nodes"] = clean_dict(serialize(nodes))
        for n in nodes:
            print(f"  Node: {n.Name} [{n.token}]")
            sp = getattr(n, 'SupportedPTZSpaces', None)
            if sp:
                apt = getattr(sp, 'AbsolutePanTiltPositionSpace', [])
                for s in (apt or []):
                    print(f"    AbsPanTilt: X=[{s.XRange.Min}..{s.XRange.Max}] Y=[{s.YRange.Min}..{s.YRange.Max}]")
                azm = getattr(sp, 'AbsoluteZoomPositionSpace', [])
                for s in (azm or []):
                    print(f"    AbsZoom: [{s.XRange.Min}..{s.XRange.Max}]")
            print(f"    HomeSupported: {n.HomeSupported}")
            print(f"    MaxPresets: {n.MaximumNumberOfPresets}")

        configs = ptz.GetConfigurations()
        discovery_data["ptz_configurations"] = clean_dict(serialize(configs))
        for c in configs:
            ptl = getattr(c, 'PanTiltLimits', None)
            if ptl:
                r = ptl.Range
                print(f"  Config {c.Name}: Pan=[{r.XRange.Min}..{r.XRange.Max}] Tilt=[{r.YRange.Min}..{r.YRange.Max}]")
            zl = getattr(c, 'ZoomLimits', None)
            if zl:
                print(f"    Zoom=[{zl.Range.XRange.Min}..{zl.Range.XRange.Max}]")

        # Initial PTZ Status
        try:
            status = ptz.GetStatus({'ProfileToken': profiles[0].token})
            discovery_data["ptz_status"] = clean_dict(serialize(status))
            pos = status.Position
            if pos:
                pt = getattr(pos, 'PanTilt', None)
                zm = getattr(pos, 'Zoom', None)
                print(f"  Current position: Pan={getattr(pt,'x','?')} Tilt={getattr(pt,'y','?')} Zoom={getattr(zm,'x','?')}")
        except Exception as e:
            discovery_data["errors"].append(f"PTZStatus: {e}")
            print(f"  Status: {e}")

    except Exception as e:
        discovery_data["errors"].append(f"PTZ: {e}")
        print(f"  ERROR: {e}")

    # [5] Event Properties
    print("\n[5/6] Event Service: GetEventProperties...")
    try:
        evt = cam.create_events_service()
        props = evt.GetEventProperties()
        discovery_data["event_properties"] = clean_dict(serialize(props))
        print(f"  FixedTopicSet: {props.FixedTopicSet}")
        print(f"  TopicDialects: {len(props.TopicExpressionDialect)}")
    except Exception as e:
        discovery_data["errors"].append(f"EventProperties: {e}")
        print(f"  ERROR: {e}")

    # [6] Imaging (erwartet: nicht unterstuetzt)
    print("\n[6/6] Imaging Service (expected: not supported)...")
    try:
        img = cam.create_imaging_service()
        vsrc_list = discovery_data.get("video_sources", [])
        if vsrc_list:
            token = vsrc_list[0].get("token", "VideoSourceToken") if isinstance(vsrc_list[0], dict) else "VideoSourceToken"
            settings = img.GetImagingSettings({'VideoSourceToken': token})
            discovery_data["imaging_settings"] = clean_dict(serialize(settings))
            print(f"  SUPPORTED! Brightness={getattr(settings,'Brightness','?')}")
        else:
            print("  No video source token available")
    except Exception as e:
        discovery_data["imaging_supported"] = False
        print(f"  Confirmed: {e}")

    # Save Phase 1
    out_path = os.path.join(OUTPUT_DIR, "eye_discovery_fresh.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(discovery_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Phase 1 saved to {out_path}")
    print(f"  Errors: {len(discovery_data['errors'])}")

    return cam


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — Live Event Monitor + PTZ Tracker
# ═══════════════════════════════════════════════════════════════════════════════

def extract_event_data(notification):
    """Parse a single ONVIF notification into a structured dict."""
    from zeep.helpers import serialize_object
    result = {
        "timestamp": timestamp(),
        "topic": None,
        "source": {},
        "data": {},
        "raw": None
    }
    try:
        # Topic — extract the _value_1 string from the zeep Topic object
        topic = getattr(notification, 'Topic', None)
        if topic:
            v1 = getattr(topic, '_value_1', None)
            if v1 and isinstance(v1, str):
                result["topic"] = v1
            else:
                # Fallback: serialize and try to extract
                ser = serialize_object(topic)
                if isinstance(ser, dict) and '_value_1' in ser:
                    result["topic"] = str(ser['_value_1'])
                else:
                    result["topic"] = str(ser)

        # Message — nested Message.Message pattern
        msg_wrapper = getattr(notification, 'Message', None)
        if not msg_wrapper:
            result["raw"] = str(serialize_object(notification))
            return result

        msg_inner = getattr(msg_wrapper, 'Message', msg_wrapper)

        # UtcTime
        utc_time = getattr(msg_inner, 'UtcTime', None)
        if utc_time:
            result["utc_time"] = str(utc_time)

        # PropertyOperation
        prop_op = getattr(msg_inner, 'PropertyOperation', None)
        if prop_op:
            result["property_operation"] = str(prop_op)

        # Source SimpleItems
        source = getattr(msg_inner, 'Source', None)
        if source:
            for item in (getattr(source, 'SimpleItem', []) or []):
                name = getattr(item, 'Name', None)
                value = getattr(item, 'Value', None)
                if name:
                    result["source"][str(name)] = str(value) if value is not None else None

        # Data SimpleItems
        data = getattr(msg_inner, 'Data', None)
        if data:
            for item in (getattr(data, 'SimpleItem', []) or []):
                name = getattr(item, 'Name', None)
                value = getattr(item, 'Value', None)
                if name:
                    result["data"][str(name)] = str(value) if value is not None else None

    except Exception as e:
        result["parse_error"] = str(e)
        try:
            from zeep.helpers import serialize_object
            result["raw"] = str(serialize_object(notification))
        except:
            result["raw"] = str(notification)

    return result


def phase2_monitor(cam=None):
    global running, event_log, ptz_log, imaging_log
    from onvif import ONVIFCamera

    print(f"\n{'='*60}")
    print(f"  PHASE 2 — Live Event Monitor")
    print(f"  {timestamp()}")
    print(f"  Polling every {POLL_INTERVAL}s — Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    if cam is None:
        print("Connecting to camera...")
        cam = ONVIFCamera(CAM_IP, CAM_PORT, CAM_USER, CAM_PASS)
        print("Connected!")

    # Create services
    media = cam.create_media_service()
    profiles = media.GetProfiles()
    main_profile_token = profiles[0].token
    ptz = cam.create_ptz_service()

    # Try imaging
    imaging = None
    try:
        imaging = cam.create_imaging_service()
    except:
        pass

    # Create PullPoint subscription with HistoryPlugin for raw XML access
    from zeep.plugins import HistoryPlugin
    from lxml import etree

    print(f"[{timestamp()}] Creating PullPoint subscription...")
    evt = cam.create_events_service()
    history = HistoryPlugin()

    pullpoint = None
    try:
        sub = cam.create_pullpoint_service()
        pullpoint = sub
        # Attach HistoryPlugin to capture raw XML (needed for Topic parsing)
        pullpoint.zeep_client.plugins.append(history)
        print(f"[{timestamp()}] PullPoint subscription active (with XML capture)!")
    except Exception as e:
        print(f"[{timestamp()}] PullPoint creation error: {e}")
        print(f"[{timestamp()}] Will monitor PTZ status only (no events)")

    # Get initial PTZ position
    last_ptz = None
    try:
        status = ptz.GetStatus({'ProfileToken': main_profile_token})
        pos = status.Position
        pt = getattr(pos, 'PanTilt', None)
        zm = getattr(pos, 'Zoom', None)
        last_ptz = {
            "pan": float(getattr(pt, 'x', 0)),
            "tilt": float(getattr(pt, 'y', 0)),
            "zoom": float(getattr(zm, 'x', 0))
        }
        print(f"[{timestamp()}] Initial PTZ: pan={last_ptz['pan']:.1f} tilt={last_ptz['tilt']:.1f} zoom={last_ptz['zoom']:.1f}")
        ptz_log.append({
            "timestamp": timestamp(),
            "event": "initial_position",
            "position": last_ptz.copy()
        })
    except Exception as e:
        print(f"[{timestamp()}] PTZ Status error: {e}")

    # Get initial imaging
    if imaging:
        try:
            img_settings = imaging.GetImagingSettings({'VideoSourceToken': 'VideoSourceToken'})
            initial_img = {
                "brightness": getattr(img_settings, 'Brightness', None),
                "contrast": getattr(img_settings, 'Contrast', None),
                "saturation": getattr(img_settings, 'ColorSaturation', None),
                "sharpness": getattr(img_settings, 'Sharpness', None),
                "ir_cut_filter": str(getattr(img_settings, 'IrCutFilter', None)),
            }
            imaging_log.append({"timestamp": timestamp(), "event": "initial", "settings": initial_img})
            print(f"[{timestamp()}] Initial Imaging: {initial_img}")
        except Exception as e:
            print(f"[{timestamp()}] Imaging not available: {e}")
            imaging = None

    poll_count = 0
    event_count = 0

    print(f"\n[{timestamp()}] === MONITORING ACTIVE ===")
    print(f"  Perform tests on the camera now (App, HA, etc.)")
    print(f"  Press Ctrl+C when done.\n")

    while running:
        poll_count += 1
        ts = timestamp()

        # ─── Pull Events (raw XML parsing via HistoryPlugin) ───
        if pullpoint:
            try:
                messages = pullpoint.PullMessages({
                    'Timeout': 'PT2S',
                    'MessageLimit': 100
                })
                # Parse events from raw XML (zeep can't handle mixed-content Topics)
                if history.last_received:
                    raw_xml = history.last_received.get('envelope')
                    if raw_xml is not None:
                        ns_tt = '{http://www.onvif.org/ver10/schema}'
                        ns_wsnt = '{http://docs.oasis-open.org/wsn/b-2}'
                        for notif_elem in raw_xml.iter(f'{ns_wsnt}NotificationMessage'):
                            event_count += 1
                            parsed = {
                                "timestamp": ts,
                                "topic": None,
                                "source": {},
                                "data": {},
                                "utc_time": None,
                                "property_operation": None
                            }
                            # Topic text
                            topic_el = notif_elem.find(f'{ns_wsnt}Topic')
                            if topic_el is not None and topic_el.text:
                                parsed["topic"] = topic_el.text.strip()
                            # Message element
                            msg_el = notif_elem.find(f'.//{ns_tt}Message')
                            if msg_el is not None:
                                parsed["utc_time"] = msg_el.get('UtcTime')
                                parsed["property_operation"] = msg_el.get('PropertyOperation')
                                for si in msg_el.findall(f'.//{ns_tt}Source/{ns_tt}SimpleItem'):
                                    parsed["source"][si.get('Name', '')] = si.get('Value', '')
                                for si in msg_el.findall(f'.//{ns_tt}Data/{ns_tt}SimpleItem'):
                                    parsed["data"][si.get('Name', '')] = si.get('Value', '')

                            event_log.append(parsed)
                            topic_short = parsed["topic"].split("/")[-1] if parsed["topic"] else "?"
                            data_str = json.dumps(parsed["data"]) if parsed["data"] else ""
                            src_str = json.dumps(parsed["source"]) if parsed["source"] else ""
                            print(f"  [{ts}] EVENT #{event_count}: {topic_short} | {parsed['property_operation'] or ''} | src={src_str} | data={data_str}")
            except Exception as e:
                err_str = str(e)
                if "PullMessagesFaultResponse" not in err_str and "Timeout" not in err_str:
                    if "terminated" in err_str.lower() or "invalid" in err_str.lower():
                        print(f"  [{ts}] PullPoint expired, recreating...")
                        try:
                            pullpoint = cam.create_pullpoint_service()
                            pullpoint.zeep_client.plugins.append(history)
                            print(f"  [{ts}] PullPoint renewed!")
                        except:
                            pullpoint = None
                            print(f"  [{ts}] PullPoint renewal failed")

        # ─── PTZ Position ───
        try:
            status = ptz.GetStatus({'ProfileToken': main_profile_token})
            pos = status.Position
            pt = getattr(pos, 'PanTilt', None)
            zm = getattr(pos, 'Zoom', None)
            current_ptz = {
                "pan": float(getattr(pt, 'x', 0)),
                "tilt": float(getattr(pt, 'y', 0)),
                "zoom": float(getattr(zm, 'x', 0))
            }

            # Check if position changed
            if last_ptz is None or (
                abs(current_ptz["pan"] - last_ptz["pan"]) > 0.5 or
                abs(current_ptz["tilt"] - last_ptz["tilt"]) > 0.5 or
                abs(current_ptz["zoom"] - last_ptz["zoom"]) > 0.01
            ):
                move_status = getattr(status, 'MoveStatus', None)
                ms = None
                if move_status:
                    ms = {
                        "pan_tilt": str(getattr(move_status, 'PanTilt', None)),
                        "zoom": str(getattr(move_status, 'Zoom', None))
                    }
                entry = {
                    "timestamp": ts,
                    "event": "position_changed",
                    "position": current_ptz.copy(),
                    "delta": {
                        "pan": round(current_ptz["pan"] - (last_ptz["pan"] if last_ptz else 0), 2),
                        "tilt": round(current_ptz["tilt"] - (last_ptz["tilt"] if last_ptz else 0), 2),
                        "zoom": round(current_ptz["zoom"] - (last_ptz["zoom"] if last_ptz else 0), 3)
                    },
                    "move_status": ms
                }
                ptz_log.append(entry)
                print(f"  [{ts}] PTZ MOVE: pan={current_ptz['pan']:.1f} tilt={current_ptz['tilt']:.1f} zoom={current_ptz['zoom']:.2f} (delta: p={entry['delta']['pan']:+.1f} t={entry['delta']['tilt']:+.1f})")
                last_ptz = current_ptz.copy()
        except Exception as e:
            if poll_count % 15 == 0:
                print(f"  [{ts}] PTZ poll error: {e}")

        # ─── Imaging Check (every 10 polls) ───
        if imaging and poll_count % 10 == 0:
            try:
                img_s = imaging.GetImagingSettings({'VideoSourceToken': 'VideoSourceToken'})
                current_img = {
                    "brightness": getattr(img_s, 'Brightness', None),
                    "contrast": getattr(img_s, 'Contrast', None),
                    "saturation": getattr(img_s, 'ColorSaturation', None),
                    "ir_cut_filter": str(getattr(img_s, 'IrCutFilter', None)),
                }
                if imaging_log:
                    last_img = imaging_log[-1].get("settings", {})
                    if current_img != {k: last_img.get(k) for k in current_img}:
                        imaging_log.append({"timestamp": ts, "event": "settings_changed", "settings": current_img})
                        print(f"  [{ts}] IMAGING CHANGED: {current_img}")
            except:
                pass

        # Status line every 30 polls
        if poll_count % 30 == 0:
            print(f"  [{ts}] ... poll #{poll_count}, events={event_count}, ptz_changes={len(ptz_log)-1}, running...")

        time.sleep(POLL_INTERVAL)

    # Save intermediate
    save_monitor_data()
    print(f"\n[{timestamp()}] Monitor stopped. Events={event_count}, PTZ changes={len(ptz_log)-1}")
    return cam


def save_monitor_data():
    """Save raw monitoring data."""
    monitor_data = {
        "scan_timestamp": timestamp(),
        "camera": f"{CAM_IP}:{CAM_PORT}",
        "events": event_log,
        "ptz_movements": ptz_log,
        "imaging_changes": imaging_log,
        "summary": {
            "total_events": len(event_log),
            "total_ptz_changes": len(ptz_log),
            "total_imaging_changes": len(imaging_log),
            "unique_event_topics": list(set(e.get("topic","?") for e in event_log if e.get("topic")))
        }
    }
    out_path = os.path.join(OUTPUT_DIR, "eye_monitor_raw.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(monitor_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Monitor data saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — Report Generation
# ═══════════════════════════════════════════════════════════════════════════════

def phase3_report():
    global event_log, ptz_log, imaging_log, discovery_data

    print(f"\n{'='*60}")
    print(f"  PHASE 3 — Report Generation")
    print(f"  {timestamp()}")
    print(f"{'='*60}\n")

    # Load monitor data if not in memory
    if not event_log and not ptz_log:
        try:
            with open(os.path.join(OUTPUT_DIR, "eye_monitor_raw.json"), "r") as f:
                raw = json.load(f)
                event_log = raw.get("events", [])
                ptz_log = raw.get("ptz_movements", [])
                imaging_log = raw.get("imaging_changes", [])
        except:
            print("  No monitor data found.")

    if not discovery_data:
        try:
            with open(os.path.join(OUTPUT_DIR, "eye_discovery_fresh.json"), "r") as f:
                discovery_data = json.load(f)
        except:
            try:
                with open(os.path.join(OUTPUT_DIR, "onvif_phase2_raw.json"), "r") as f:
                    discovery_data = json.load(f)
            except:
                print("  No discovery data found.")

    # ─── Feature → Event Mapping ───
    event_topics = defaultdict(list)
    for evt in event_log:
        topic = evt.get("topic", "unknown")
        event_topics[topic].append(evt)

    feature_map = {}
    for topic, events in event_topics.items():
        topic_short = topic.split("/")[-1] if topic else "unknown"
        topic_parts = topic.split("/") if topic else []

        # Classify
        feature = "unknown"
        if "MotionAlarm" in topic or "motion" in topic.lower():
            feature = "motion_detection"
        elif "CellMotionDetector" in topic:
            feature = "cell_motion_detection"
        elif "PersonDetect" in topic or "person" in topic.lower() or "ObjectDetect" in topic:
            feature = "person_detection"
        elif "VideoSource" in topic:
            feature = "video_source_event"
        elif "RuleEngine" in topic:
            feature = "rule_engine"
        elif "Tampering" in topic or "tamper" in topic.lower():
            feature = "tamper_detection"
        elif "AudioDetect" in topic or "audio" in topic.lower():
            feature = "audio_detection"
        elif "DayNight" in topic or "night" in topic.lower():
            feature = "day_night_switch"
        elif "PTZ" in topic.upper():
            feature = "ptz_event"
        else:
            feature = topic_short

        data_samples = [e.get("data", {}) for e in events[:5]]
        source_samples = [e.get("source", {}) for e in events[:5]]

        feature_map[feature] = {
            "onvif_topic": topic,
            "event_count": len(events),
            "first_seen": events[0].get("timestamp"),
            "last_seen": events[-1].get("timestamp"),
            "sample_data": data_samples,
            "sample_source": source_samples,
            "available_via_onvif": True,
            "note": None
        }

    # Add known HA features that might not have triggered
    ha_features = {
        "motion_alarm": {"ha_entity": "binary_sensor.cam_pt2_motion_alarm", "ha_service": None},
        "person_detection": {"ha_entity": "binary_sensor.cam_pt2_person_detection", "ha_service": None},
        "cell_motion_detection": {"ha_entity": "binary_sensor.cam_pt2_cell_motion_detection", "ha_service": None},
        "ptz_control": {"ha_entity": "camera.cam_pt2_mainstream", "ha_service": "onvif.ptz"},
        "reboot": {"ha_entity": "button.cam_pt2_reboot", "ha_service": None},
        "set_time": {"ha_entity": "button.cam_pt2_set_system_date_and_time", "ha_service": None},
        "camera_stream": {"ha_entity": "camera.cam_pt2_mainstream", "ha_service": "camera.play_stream"},
        "snapshot": {"ha_entity": "camera.cam_pt2_mainstream", "ha_service": "camera.snapshot"},
        "record": {"ha_entity": "camera.cam_pt2_mainstream", "ha_service": "camera.record"},
    }
    for feat, ha_info in ha_features.items():
        if feat not in feature_map:
            feature_map[feat] = {
                "onvif_topic": None,
                "event_count": 0,
                "available_via_onvif": feat in ("ptz_control", "camera_stream"),
                "available_via_ha": True,
                **ha_info,
                "note": "Not triggered during test" if feat not in feature_map else None
            }
        else:
            feature_map[feat].update(ha_info)

    # ─── PTZ Analysis ───
    ptz_summary = {
        "total_movements": len(ptz_log),
        "positions_recorded": [],
        "pan_range_observed": {"min": None, "max": None},
        "tilt_range_observed": {"min": None, "max": None},
    }
    for entry in ptz_log:
        pos = entry.get("position", {})
        ptz_summary["positions_recorded"].append({
            "timestamp": entry.get("timestamp"),
            "pan": pos.get("pan"),
            "tilt": pos.get("tilt"),
            "zoom": pos.get("zoom"),
            "event": entry.get("event")
        })
        p = pos.get("pan")
        t = pos.get("tilt")
        if p is not None:
            if ptz_summary["pan_range_observed"]["min"] is None or p < ptz_summary["pan_range_observed"]["min"]:
                ptz_summary["pan_range_observed"]["min"] = p
            if ptz_summary["pan_range_observed"]["max"] is None or p > ptz_summary["pan_range_observed"]["max"]:
                ptz_summary["pan_range_observed"]["max"] = p
        if t is not None:
            if ptz_summary["tilt_range_observed"]["min"] is None or t < ptz_summary["tilt_range_observed"]["min"]:
                ptz_summary["tilt_range_observed"]["min"] = t
            if ptz_summary["tilt_range_observed"]["max"] is None or t > ptz_summary["tilt_range_observed"]["max"]:
                ptz_summary["tilt_range_observed"]["max"] = t

    # ─── ONVIF vs App Report ───
    onvif_features = []
    app_only_features = []
    ha_only_features = []

    for feat, info in feature_map.items():
        if info.get("available_via_onvif"):
            onvif_features.append(feat)
        elif info.get("ha_entity") or info.get("ha_service"):
            ha_only_features.append(feat)
        else:
            app_only_features.append(feat)

    report = {
        "report_timestamp": timestamp(),
        "camera": {
            "model": discovery_data.get("device_info", {}).get("Model", "CAM-PT2"),
            "firmware": discovery_data.get("device_info", {}).get("FirmwareVersion", "?"),
            "ip": CAM_IP,
            "mac": "48:d0:1c:c4:cd:f7"
        },
        "test_summary": {
            "total_events_captured": len(event_log),
            "unique_topics": list(event_topics.keys()),
            "ptz_movements_tracked": len(ptz_log),
            "imaging_changes_tracked": len(imaging_log)
        },
        "feature_event_mapping": feature_map,
        "ptz_analysis": ptz_summary,
        "imaging_changes": imaging_log,
        "access_report": {
            "via_onvif": {
                "description": "Features accessible via ONVIF protocol (local, no cloud)",
                "features": onvif_features
            },
            "via_ha": {
                "description": "Features accessible via Home Assistant integration",
                "features": list(ha_features.keys())
            },
            "app_only": {
                "description": "Features that only work via Sonoff/eWeLink app (cloud required)",
                "features": [
                    "wifi_configuration",
                    "firmware_update",
                    "cloud_storage",
                    "sharing",
                    "push_notifications",
                    "imaging_settings_brightness_contrast",
                    "night_vision_mode_toggle",
                    "video_quality_preset_selection",
                    "audio_talk_back",
                    "preset_creation_management"
                ],
                "note": "These features require the Sonoff/eWeLink app and typically cloud connectivity"
            },
            "local_after_lockdown": {
                "description": "Features that work 100% local after internet is blocked",
                "features": [
                    "rtsp_stream_main (1920x1080)",
                    "rtsp_stream_minor (640x360)",
                    "ptz_control (absolute/relative/continuous)",
                    "motion_alarm_events",
                    "person_detection_events",
                    "cell_motion_detection_events",
                    "snapshot_via_ha",
                    "record_via_ha",
                    "reboot_via_ha",
                    "set_time_via_ha"
                ]
            }
        }
    }

    # ─── Save all 4 files ───

    # 1. eye_capabilities.json (updated)
    caps_path = os.path.join(OUTPUT_DIR, "eye_capabilities.json")
    try:
        with open(caps_path, "r", encoding="utf-8") as f:
            existing_caps = json.load(f)
        existing_caps["test_report"] = {
            "test_timestamp": timestamp(),
            "events_captured": len(event_log),
            "features_verified": list(feature_map.keys()),
            "access_report": report["access_report"]
        }
        with open(caps_path, "w", encoding="utf-8") as f:
            json.dump(existing_caps, f, indent=2, ensure_ascii=False, default=str)
        print(f"  [1/4] Updated {caps_path}")
    except Exception as e:
        print(f"  [1/4] Could not update capabilities: {e}")

    # 2. eye_presets.json (updated with observed positions)
    presets_path = os.path.join(OUTPUT_DIR, "eye_presets.json")
    try:
        with open(presets_path, "r", encoding="utf-8") as f:
            existing_presets = json.load(f)
        existing_presets["observed_positions"] = ptz_summary["positions_recorded"]
        existing_presets["pan_range_observed"] = ptz_summary["pan_range_observed"]
        existing_presets["tilt_range_observed"] = ptz_summary["tilt_range_observed"]
        with open(presets_path, "w", encoding="utf-8") as f:
            json.dump(existing_presets, f, indent=2, ensure_ascii=False, default=str)
        print(f"  [2/4] Updated {presets_path}")
    except Exception as e:
        print(f"  [2/4] Could not update presets: {e}")

    # 3. eye_events.json (full event log)
    events_path = os.path.join(OUTPUT_DIR, "eye_events.json")
    events_out = {
        "scan_timestamp": timestamp(),
        "camera": f"SONOFF CAM-PT2 @ {CAM_IP}",
        "total_events": len(event_log),
        "unique_topics": list(event_topics.keys()),
        "feature_event_mapping": feature_map,
        "events": event_log
    }
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump(events_out, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [3/4] Saved {events_path}")

    # 4. eye_report.json (full test report)
    report_path = os.path.join(OUTPUT_DIR, "eye_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [4/4] Saved {report_path}")

    # Print summary
    print(f"\n{'─'*60}")
    print(f"  TEST REPORT SUMMARY")
    print(f"{'─'*60}")
    print(f"  Events captured:        {len(event_log)}")
    print(f"  Unique event topics:    {len(event_topics)}")
    print(f"  PTZ movements tracked:  {len(ptz_log)}")
    print(f"  Imaging changes:        {len(imaging_log)}")
    print(f"\n  Via ONVIF (lokal):      {', '.join(onvif_features) or 'none triggered'}")
    print(f"  Via HA:                 {', '.join(ha_features.keys())}")
    print(f"  App-only:               imaging, presets, wifi, cloud, firmware")
    print(f"\n  Output files:")
    print(f"    1. eye_capabilities.json  (updated)")
    print(f"    2. eye_presets.json        (updated + observed positions)")
    print(f"    3. eye_events.json         (full event log)")
    print(f"    4. eye_report.json         (complete test report)")
    print(f"{'─'*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = sys.argv[1:]

    print(f"""
    +----------------------------------------------+
    |  M.O.L.O.C.H. Eye -- Live ONVIF Test Suite   |
    |  Sonoff CAM-PT2 @ {CAM_IP}          |
    |  {timestamp()}                  |
    +----------------------------------------------+
    """)

    if "--phase1" in args:
        phase1_discovery()
    elif "--monitor" in args:
        phase2_monitor()
    elif "--report" in args:
        phase3_report()
    else:
        # Full run
        cam = phase1_discovery()
        print(f"\n[{timestamp()}] Phase 1 complete. Starting live monitor...")
        print(f"[{timestamp()}] Perform your tests now. Press Ctrl+C when done.\n")
        phase2_monitor(cam)
        phase3_report()

    print(f"[{timestamp()}] M.O.L.O.C.H. Eye scan complete.")
