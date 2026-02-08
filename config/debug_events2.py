#!/usr/bin/env python3
"""Debug: Fresh subscription + inspect initial property events"""
import json, sys
from onvif import ONVIFCamera
from zeep.helpers import serialize_object
from lxml import etree

cam = ONVIFCamera("192.168.178.25", 80, "Moloch_4.5", "Auge666")

# Fresh subscription
print("Creating fresh PullPoint subscription...")
pullpoint = cam.create_pullpoint_service()

print("Pulling (with 10s timeout for initial state events)...")
try:
    messages = pullpoint.PullMessages({'Timeout': 'PT10S', 'MessageLimit': 50})

    notifs = messages.NotificationMessage
    print(f"Got {len(notifs) if notifs else 0} notifications\n")

    if notifs:
        for i, n in enumerate(notifs):
            print(f"=== Event {i+1} ===")

            # Topic - try _value_1 which should be the topic string
            topic = getattr(n, 'Topic', None)
            topic_str = "unknown"
            if topic:
                v1 = getattr(topic, '_value_1', None)
                if v1:
                    topic_str = str(v1)
                else:
                    topic_str = str(topic)
            print(f"  Topic: {topic_str}")

            # Message wrapper
            msg = getattr(n, 'Message', None)
            if msg:
                # The actual Message is nested inside
                inner = getattr(msg, 'Message', msg)

                # UtcTime
                utc = getattr(inner, 'UtcTime', None)
                print(f"  UtcTime: {utc}")

                # PropertyOperation
                prop_op = getattr(inner, 'PropertyOperation', None)
                print(f"  PropertyOp: {prop_op}")

                # Source
                source = getattr(inner, 'Source', None)
                if source:
                    items = getattr(source, 'SimpleItem', [])
                    for item in (items or []):
                        print(f"  Source: {item.Name}={item.Value}")

                # Data
                data = getattr(inner, 'Data', None)
                if data:
                    items = getattr(data, 'SimpleItem', [])
                    for item in (items or []):
                        print(f"  Data: {item.Name}={item.Value}")

            # Also try the full zeep serialization
            try:
                ser = serialize_object(n)
                # Clean out noise
                clean = json.dumps(ser, indent=2, default=str)
                if len(clean) < 2000:
                    print(f"  RAW: {clean}")
            except:
                pass
            print()

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
