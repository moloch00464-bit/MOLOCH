#!/usr/bin/env python3
"""Quick debug: figure out the actual Topic structure"""
import json
from onvif import ONVIFCamera
from zeep.helpers import serialize_object

cam = ONVIFCamera("192.168.178.25", 80, "Moloch_4.5", "Auge666")
pullpoint = cam.create_pullpoint_service()

print("Pulling with 10s timeout...")
try:
    msgs = pullpoint.PullMessages({'Timeout': 'PT10S', 'MessageLimit': 50})
    notifs = msgs.NotificationMessage
    print(f"Got {len(notifs) if notifs else 0} notifications\n")

    if not notifs:
        # Trigger something - try subscription renewal
        print("No events. PullPoint may be stale.")
        print("Creating fresh pullpoint...")
        pullpoint2 = cam.create_pullpoint_service()
        msgs2 = pullpoint2.PullMessages({'Timeout': 'PT10S', 'MessageLimit': 50})
        notifs = msgs2.NotificationMessage
        print(f"Fresh pull: {len(notifs) if notifs else 0} notifications")

    if notifs:
        for i, n in enumerate(notifs):
            print(f"--- Event {i+1} ---")
            # Full serialization
            full_ser = serialize_object(n)
            print(f"Full keys: {list(full_ser.keys()) if isinstance(full_ser, dict) else type(full_ser)}")
            print(f"Full dump:\n{json.dumps(full_ser, indent=2, default=str)[:1000]}")

            # Direct attribute inspection
            topic = getattr(n, 'Topic', None)
            if topic:
                print(f"\nTopic type: {type(topic).__name__}")
                attrs = {a: getattr(topic, a, 'ERR') for a in dir(topic) if not a.startswith('__')}
                for k, v in attrs.items():
                    if not callable(v):
                        print(f"  topic.{k} = {repr(v)[:200]}")
            print()
    else:
        print("Still no events. Will check raw monitor file instead.")
        # Check the saved monitor data
        try:
            with open("eye_monitor_raw.json", "r") as f:
                data = json.load(f)
                evts = data.get("events", [])
                print(f"Monitor file has {len(evts)} events")
                if evts:
                    print(f"First event: {json.dumps(evts[0], indent=2, default=str)[:500]}")
        except:
            print("No monitor file yet.")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
