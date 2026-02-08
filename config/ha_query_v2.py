import requests, json, sys, os

HA_URL = "http://192.168.178.32:8123"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJmZDgyNWI1MDNjMTY0ZDY0YWY0Y2U0NjRkZjkyMTFlNiIsImlhdCI6MTc3MDU0MTg1MCwiZXhwIjoyMDg1OTAxODUwfQ.hlvHR8U3pZ-1kiLDw64YMHBqdROs4j2TsxnbcKhrj5Q"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

result = {"ha_entities": [], "ha_services": [], "ha_device_info": None}

camera_keywords = ["sonoff", "cam_pt", "campt", "onvif", "192_168_178_25", "192.168.178.25", "eye", "moloch", "auge"]

# 1. Get ALL states
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
    print(f"Found {len(cam_entities)} camera-related entities out of {len(all_states)} total")
    for e in cam_entities:
        print(f"  - {e['entity_id']} ({e['state']})")
except Exception as ex:
    print(f"ERROR getting states: {ex}")

# 2. Get entity registry
try:
    r = requests.get(f"{HA_URL}/api/config/entity_registry", headers=HEADERS, timeout=15)
    if r.status_code == 200:
        all_registry = r.json()
        cam_registry = [e for e in all_registry if any(kw in (e.get("entity_id","") + e.get("original_name","") + e.get("unique_id","")).lower() for kw in camera_keywords)]
        result["ha_entity_registry"] = cam_registry
        print(f"\nEntity registry: {len(cam_registry)} camera entries (incl. disabled)")
        for e in cam_registry:
            disabled = e.get("disabled_by", None)
            status = "[DISABLED: "+disabled+"]" if disabled else "[ENABLED]"
            print(f"  - {e['entity_id']} {status}")
    else:
        r2 = requests.post(f"{HA_URL}/api/template", headers=HEADERS, json={"template": "{{ states | list | length }}"}, timeout=10)
        print(f"Entity registry API returned {r.status_code}, template API: {r2.status_code}")
except Exception as ex:
    print(f"Note: Entity registry: {ex}")

# 3. Get device registry
try:
    r = requests.get(f"{HA_URL}/api/config/device_registry", headers=HEADERS, timeout=15)
    if r.status_code == 200:
        all_devices = r.json()
        cam_devices = [d for d in all_devices if any(kw in json.dumps(d).lower() for kw in camera_keywords)]
        result["ha_devices"] = cam_devices
        print(f"\nDevice registry: {len(cam_devices)} camera devices")
        for d in cam_devices:
            print(f"  - {d.get('name', 'unknown')} (id: {d.get('id', '?')[:12]}...)")
except Exception as ex:
    print(f"Note: Device registry: {ex}")

# 4. Get services
try:
    r = requests.get(f"{HA_URL}/api/services", headers=HEADERS, timeout=15)
    r.raise_for_status()
    all_services = r.json()
    cam_services = [s for s in all_services if s.get("domain") in ["camera", "onvif", "ptz", "stream"]]
    result["ha_services"] = cam_services
    print(f"\nCamera-related services: {len(cam_services)} domains")
    for s in cam_services:
        print(f"  Domain: {s['domain']} -> {list(s.get('services',{}).keys())}")
except Exception as ex:
    print(f"ERROR getting services: {ex}")

# 5. Get ONVIF config entries
try:
    r = requests.get(f"{HA_URL}/api/config/config_entries/entry", headers=HEADERS, timeout=15)
    if r.status_code == 200:
        entries = r.json()
        onvif_entries = [e for e in entries if "onvif" in e.get("domain","").lower() or "onvif" in json.dumps(e).lower()]
        result["ha_config_entries_onvif"] = onvif_entries
        print(f"\nONVIF config entries: {len(onvif_entries)}")
        for e in onvif_entries:
            print(f"  - {e.get('title', '?')} domain={e.get('domain')} state={e.get('state')}")
except Exception as ex:
    print(f"Note: Config entries: {ex}")

os.makedirs("C:/Users/49179/moloch/config", exist_ok=True)
with open("C:/Users/49179/moloch/config/ha_phase1_raw.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False, default=str)

print(f"\nPhase 1 complete. Saved to ha_phase1_raw.json")
print(f"Total entities found: {len(result.get('ha_entities', []))}")
