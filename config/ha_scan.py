#!/usr/bin/env python3
"""Phase 1: HA API scan for CAM-PT2 entities — M.O.L.O.C.H. Eye"""
import requests, json, sys

HA_URL = "http://192.168.178.32:8123"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJmZDgyNWI1MDNjMTY0ZDY0YWY0Y2U0NjRkZjkyMTFlNiIsImlhdCI6MTc3MDU0MTg1MCwiZXhwIjoyMDg1OTAxODUwfQ.hlvHR8U3pZ-1kiLDw64YMHBqdROs4j2TsxnbcKhrj5Q"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

camera_keywords = ["sonoff", "cam_pt", "campt", "onvif", "192_168_178_25", "192.168.178.25", "eye", "moloch", "auge"]
result = {"ha_entities": [], "ha_services": [], "ha_device_info": None}

# 1. Get ALL states
print("=== Phase 1: HA API Scan ===")
try:
    r = requests.get(f"{HA_URL}/api/states", headers=HEADERS, timeout=15)
    r.raise_for_status()
    all_states = r.json()
    cam_entities = []
    for entity in all_states:
        eid = entity.get("entity_id", "").lower()
        friendly = entity.get("attributes", {}).get("friendly_name", "").lower()
        combined = eid + " " + friendly
        if any(kw in combined for kw in camera_keywords):
            cam_entities.append({
                "entity_id": entity["entity_id"],
                "state": entity["state"],
                "attributes": entity["attributes"],
                "last_changed": entity.get("last_changed"),
                "last_updated": entity.get("last_updated")
            })
    result["ha_entities"] = cam_entities
    print(f"Found {len(cam_entities)} camera entities out of {len(all_states)} total")
    for e in cam_entities:
        print(f"  - {e['entity_id']} ({e['state']})")
except Exception as ex:
    print(f"ERROR states: {ex}")

# 2. Entity registry (includes disabled)
try:
    r = requests.get(f"{HA_URL}/api/config/entity_registry", headers=HEADERS, timeout=15)
    if r.status_code == 200:
        all_reg = r.json()
        cam_reg = [e for e in all_reg if any(kw in (e.get("entity_id","") + e.get("original_name","") + e.get("unique_id","")).lower() for kw in camera_keywords)]
        result["ha_entity_registry"] = cam_reg
        print(f"\nEntity registry: {len(cam_reg)} entries (incl. disabled)")
        for e in cam_reg:
            disabled = e.get("disabled_by", None)
            print(f"  - {e['entity_id']} {'[DISABLED:'+str(disabled)+']' if disabled else '[ENABLED]'}")
    else:
        print(f"Entity registry: HTTP {r.status_code}")
except Exception as ex:
    print(f"Note entity registry: {ex}")

# 3. Device registry
try:
    r = requests.get(f"{HA_URL}/api/config/device_registry", headers=HEADERS, timeout=15)
    if r.status_code == 200:
        all_dev = r.json()
        cam_dev = [d for d in all_dev if any(kw in json.dumps(d).lower() for kw in camera_keywords)]
        result["ha_devices"] = cam_dev
        print(f"\nDevice registry: {len(cam_dev)} devices")
        for d in cam_dev:
            print(f"  - {d.get('name','?')} (id: {d.get('id','?')[:16]})")
    else:
        print(f"Device registry: HTTP {r.status_code}")
except Exception as ex:
    print(f"Note device registry: {ex}")

# 4. Services
try:
    r = requests.get(f"{HA_URL}/api/services", headers=HEADERS, timeout=15)
    r.raise_for_status()
    all_svc = r.json()
    cam_svc = [s for s in all_svc if s.get("domain") in ["camera", "onvif", "ptz", "stream"]]
    result["ha_services"] = cam_svc
    print(f"\nCamera services: {len(cam_svc)} domains")
    for s in cam_svc:
        print(f"  {s['domain']}: {list(s.get('services',{}).keys())}")
except Exception as ex:
    print(f"ERROR services: {ex}")

# 5. ONVIF config entries
try:
    r = requests.get(f"{HA_URL}/api/config/config_entries/entry", headers=HEADERS, timeout=15)
    if r.status_code == 200:
        entries = r.json()
        onvif_e = [e for e in entries if "onvif" in e.get("domain","").lower()]
        result["ha_config_entries_onvif"] = onvif_e
        print(f"\nONVIF config entries: {len(onvif_e)}")
        for e in onvif_e:
            print(f"  - {e.get('title','?')} domain={e.get('domain')} state={e.get('state')}")
except Exception as ex:
    print(f"Note config entries: {ex}")

with open("C:/Users/49179/moloch/config/ha_phase1_raw.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False, default=str)

print(f"\nPhase 1 saved to ha_phase1_raw.json")
