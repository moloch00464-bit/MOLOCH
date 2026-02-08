#!/usr/bin/env python3
"""Debug: Extract topic from raw XML via zeep transport plugin"""
import json
from onvif import ONVIFCamera
from zeep.helpers import serialize_object
from lxml import etree

cam = ONVIFCamera("192.168.178.25", 80, "Moloch_4.5", "Auge666")
pullpoint = cam.create_pullpoint_service()

# Get the raw SOAP client
ws_client = pullpoint.ws_client

# Enable raw XML capture via zeep plugin
class RawXMLPlugin:
    def __init__(self):
        self.last_response = None
    def ingress(self, envelope, http_headers, operation):
        self.last_response = envelope
        return envelope, http_headers

plugin = RawXMLPlugin()
ws_client.plugins.append(plugin)

print("Pulling events with XML capture...")
try:
    msgs = pullpoint.PullMessages({'Timeout': 'PT10S', 'MessageLimit': 50})
    notifs = msgs.NotificationMessage
    print(f"Got {len(notifs) if notifs else 0} notifications")

    if plugin.last_response is not None:
        xml_str = etree.tostring(plugin.last_response, pretty_print=True, encoding='unicode')
        print(f"\n=== RAW XML RESPONSE ===")
        print(xml_str[:5000])

        # Parse topics from XML
        ns = {
            'wsnt': 'http://docs.oasis-open.org/wsn/b-2',
            'tns1': 'http://www.onvif.org/ver10/topics',
            'tt': 'http://www.onvif.org/ver10/schema',
            'soap': 'http://www.w3.org/2003/05/soap-envelope'
        }
        topics = plugin.last_response.findall('.//wsnt:Topic', ns)
        print(f"\nFound {len(topics)} Topic elements:")
        for t in topics:
            print(f"  Topic text: '{t.text}'")
            print(f"  Topic attrib: {dict(t.attrib)}")

        # Parse Message elements
        messages = plugin.last_response.findall('.//tt:Message', ns)
        print(f"\nFound {len(messages)} Message elements:")
        for m in messages:
            print(f"  Message attrib: {dict(m.attrib)}")
            # Source
            for src in m.findall('.//tt:Source/tt:SimpleItem', ns):
                print(f"    Source: {dict(src.attrib)}")
            # Data
            for data in m.findall('.//tt:Data/tt:SimpleItem', ns):
                print(f"    Data: {dict(data.attrib)}")
    else:
        print("No XML response captured")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
