#!/usr/bin/env python3
"""Debug: Use zeep history plugin to capture raw XML"""
import json
from onvif import ONVIFCamera
from zeep.plugins import HistoryPlugin
from zeep.helpers import serialize_object
from lxml import etree

history = HistoryPlugin()

cam = ONVIFCamera("192.168.178.25", 80, "Moloch_4.5", "Auge666")

# Access the underlying zeep transport client and add plugin
# The pullpoint service creates its own client, so we need to hook in there
pullpoint = cam.create_pullpoint_service()

# Get the zeep Client object - it's on the ONVIFService
zeep_client = pullpoint.zeep_client
if hasattr(zeep_client, 'plugins'):
    zeep_client.plugins.append(history)
    print("Plugin added to zeep_client")
elif hasattr(pullpoint, 'ws_client'):
    # Try the ws_client's underlying zeep client
    inner = pullpoint.ws_client
    print(f"ws_client type: {type(inner)}")
    print(f"ws_client dir: {[a for a in dir(inner) if not a.startswith('_')]}")

# Alternative: Just do raw HTTP request to get the XML
import requests
from requests.auth import HTTPDigestAuth

print("\n=== Direct SOAP request ===")
# Build the SOAP envelope for PullMessages
soap_body = """<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tev="http://www.onvif.org/ver10/events/wsdl">
  <s:Header>
    <Security xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd" s:mustUnderstand="true">
    </Security>
  </s:Header>
  <s:Body>
    <tev:CreatePullPointSubscription>
      <tev:InitialTerminationTime>PT60S</tev:InitialTerminationTime>
    </tev:CreatePullPointSubscription>
  </s:Body>
</s:Envelope>"""

headers = {'Content-Type': 'application/soap+xml; charset=utf-8'}

try:
    # First create subscription
    r = requests.post(
        "http://192.168.178.25/onvif/events_service",
        data=soap_body,
        headers=headers,
        auth=HTTPDigestAuth("Moloch_4.5", "Auge666"),
        timeout=10
    )
    print(f"CreatePullPoint status: {r.status_code}")
    xml_resp = r.text
    print(f"Response:\n{xml_resp[:2000]}")

    # Parse the subscription reference
    root = etree.fromstring(r.content)
    ns = {'wsa': 'http://www.w3.org/2005/08/addressing'}
    addr_elem = root.find('.//wsa:Address', ns)
    if addr_elem is not None:
        sub_addr = addr_elem.text
        print(f"\nSubscription address: {sub_addr}")

        # Now PullMessages
        pull_body = f"""<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tev="http://www.onvif.org/ver10/events/wsdl">
  <s:Header>
  </s:Header>
  <s:Body>
    <tev:PullMessages>
      <tev:Timeout>PT10S</tev:Timeout>
      <tev:MessageLimit>50</tev:MessageLimit>
    </tev:PullMessages>
  </s:Body>
</s:Envelope>"""

        r2 = requests.post(
            sub_addr,
            data=pull_body,
            headers=headers,
            auth=HTTPDigestAuth("Moloch_4.5", "Auge666"),
            timeout=15
        )
        print(f"\nPullMessages status: {r2.status_code}")
        print(f"Response:\n{r2.text[:3000]}")
    else:
        print("Could not find subscription address")
        # Print all addresses found
        for elem in root.iter():
            if 'Address' in elem.tag:
                print(f"  Found: {elem.tag} = {elem.text}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
