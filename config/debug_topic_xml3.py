#!/usr/bin/env python3
"""Debug: Use zeep HistoryPlugin to capture raw XML from PullMessages"""
from onvif import ONVIFCamera
from zeep.plugins import HistoryPlugin
from lxml import etree

history = HistoryPlugin()

cam = ONVIFCamera("192.168.178.25", 80, "Moloch_4.5", "Auge666")
pullpoint = cam.create_pullpoint_service()

# Add history plugin to the actual zeep Client
pullpoint.zeep_client.plugins.append(history)

print("Pulling events...")
msgs = pullpoint.PullMessages({'Timeout': 'PT10S', 'MessageLimit': 50})
notifs = msgs.NotificationMessage
print(f"Notifications: {len(notifs) if notifs else 0}")

# Check history
if history.last_received:
    raw = history.last_received['envelope']
    xml_str = etree.tostring(raw, pretty_print=True, encoding='unicode')
    print(f"\n=== RAW XML ({len(xml_str)} chars) ===")
    print(xml_str[:4000])

    # Extract topics
    ns = {
        'wsnt': 'http://docs.oasis-open.org/wsn/b-2',
        'tt': 'http://www.onvif.org/ver10/schema',
        'tns1': 'http://www.onvif.org/ver10/topics'
    }
    for topic_elem in raw.iter('{http://docs.oasis-open.org/wsn/b-2}Topic'):
        print(f"\n  Topic text: '{topic_elem.text}'")
        print(f"  Topic attrib: {dict(topic_elem.attrib)}")

    for msg_elem in raw.iter('{http://www.onvif.org/ver10/schema}Message'):
        print(f"\n  Message attrib: {dict(msg_elem.attrib)}")
        for child in msg_elem:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            print(f"    {tag}:")
            for item in child:
                print(f"      {dict(item.attrib)}")
else:
    print("No history captured")
    print(f"history attributes: {dir(history)}")
