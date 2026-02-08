#!/usr/bin/env python3
"""Debug: Inspect raw ONVIF event structure from CAM-PT2"""
import json, sys
from onvif import ONVIFCamera
from zeep.helpers import serialize_object

cam = ONVIFCamera("192.168.178.25", 80, "Moloch_4.5", "Auge666")
pullpoint = cam.create_pullpoint_service()

print("Pulling events...")
try:
    messages = pullpoint.PullMessages({'Timeout': 'PT5S', 'MessageLimit': 10})
    print(f"CurrentTime: {messages.CurrentTime}")
    print(f"TerminationTime: {messages.TerminationTime}")

    notifs = messages.NotificationMessage
    print(f"Notifications: {len(notifs) if notifs else 0}\n")

    if notifs:
        for i, n in enumerate(notifs):
            print(f"=== Notification {i} ===")
            print(f"  type: {type(n)}")
            print(f"  dir:  {[a for a in dir(n) if not a.startswith('__')]}")

            # Topic
            topic = getattr(n, 'Topic', None)
            if topic:
                print(f"  Topic type: {type(topic)}")
                print(f"  Topic dir: {[a for a in dir(topic) if not a.startswith('__')]}")
                print(f"  Topic._value_1: {getattr(topic, '_value_1', 'N/A')}")
                print(f"  Topic str: {topic}")
                try:
                    print(f"  Topic serialized: {serialize_object(topic)}")
                except Exception as e:
                    print(f"  Topic serialize err: {e}")

            # Message
            msg = getattr(n, 'Message', None)
            if msg:
                print(f"  Message type: {type(msg)}")
                print(f"  Message dir: {[a for a in dir(msg) if not a.startswith('__')]}")
                # Inner message
                inner = getattr(msg, 'Message', msg)
                if inner:
                    print(f"  Inner type: {type(inner)}")
                    print(f"  Inner dir: {[a for a in dir(inner) if not a.startswith('__')]}")
                    src = getattr(inner, 'Source', None)
                    data = getattr(inner, 'Data', None)
                    utc = getattr(inner, 'UtcTime', None)
                    prop = getattr(inner, 'PropertyOperation', None)
                    print(f"  Source: {serialize_object(src) if src else None}")
                    print(f"  Data: {serialize_object(data) if data else None}")
                    print(f"  UtcTime: {utc}")
                    print(f"  PropertyOperation: {prop}")

                    # Try _value_1
                    v1 = getattr(inner, '_value_1', None)
                    if v1:
                        print(f"  Inner._value_1: {v1}")

            # Try full serialize
            try:
                full = serialize_object(n)
                print(f"  FULL SERIALIZED: {json.dumps(full, indent=4, default=str)}")
            except Exception as e:
                print(f"  Full serialize err: {e}")

            print()
    else:
        print("No notifications received.")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
