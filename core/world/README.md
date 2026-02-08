# M.O.L.O.C.H. World Inventory

## Philosophy

The World Inventory is M.O.L.O.C.H.'s **map of possibilities**.

It defines not just what hardware is connected NOW, but what COULD be connected. Every sensor, every output device, every interaction channel has a slot reserved - even if that slot is currently empty.

**Why placeholders?**
Because M.O.L.O.C.H. needs to know what it CAN become, not just what it IS. The inventory is a blueprint for growth. When Markus adds a temperature sensor, there's already a slot waiting for it. M.O.L.O.C.H. doesn't need to learn that temperature sensors exist - it already expected one.

## Design Principles

### 1. Slots, Not Dynamic Lists
Each category has exactly 10 slots. This is intentional:
- Prevents unbounded growth
- Forces prioritization
- Makes the system predictable
- Enables visual dashboards later

### 2. Three States
- **available**: Device is present and functional
- **placeholder**: Slot reserved for future device
- **empty**: Unused slot (truly empty)

Plus error states:
- **offline**: Device configured but not responding
- **error**: Device has encountered an error

### 3. No Polling
The inventory is a **static data structure**. It does not:
- Check if devices are actually connected
- Poll sensors for data
- Validate device status

That's the job of other systems (like `environment_watcher.py`).

### 4. Human-Editable
The inventory is a JSON file. Markus can edit it directly. M.O.L.O.C.H. adapts.

## Categories

### interaction (Input Channels)
How humans communicate TO M.O.L.O.C.H.:
- Terminal input (available)
- Voice input (placeholder)
- Gesture input (placeholder)
- Keyboard hotkeys (placeholder)
- REST API (placeholder)
- MQTT messages (placeholder)

### output (Output Channels)
How M.O.L.O.C.H. communicates TO humans:
- Terminal output (available)
- HDMI audio (available - via TTS)
- 3.5mm jack (available - via TTS)
- LED matrix (placeholder)
- OLED display (placeholder)
- MQTT messages (placeholder)
- Webhooks (placeholder)

### vision (Visual Perception)
What M.O.L.O.C.H. can see:
- AI Camera with Hailo (placeholder)
- Thermal camera (placeholder)
- USB webcam (placeholder)
- Depth sensor (placeholder)

### environment (Environmental Sensing)
What M.O.L.O.C.H. can sense about its surroundings:
- Room temperature (placeholder)
- Humidity (placeholder)
- Light level (placeholder)
- Motion/presence (placeholder)
- Air quality (placeholder)
- Sound level (placeholder)
- Barometric pressure (placeholder)

### compute (Processing Resources)
What M.O.L.O.C.H. can think with:
- Raspberry Pi 5 CPU (available)
- Hailo-10H NPU 40 TOPS (available)
- VideoCore VII GPU (available)
- NVMe SSD (available)
- System RAM (available)

### network (Connectivity)
How M.O.L.O.C.H. connects to the world:
- Gigabit Ethernet (available)
- WiFi 6 (available)
- Bluetooth (placeholder)
- Zigbee (placeholder)
- Z-Wave (placeholder)

## File Structure

```
~/moloch/core/world/
├── state/
│   └── world_inventory.json    # The inventory data
├── world_state.py              # Load/save logic
└── README.md                   # This file
```

## Usage

### Python API

```python
import sys
sys.path.insert(0, '~/moloch/core/world')

from world_state import get_world_inventory

# Get the inventory
inventory = get_world_inventory()

# Get summary
summary = inventory.get_summary()
print(f"Available: {summary['totals']['available']}")
print(f"Placeholder: {summary['totals']['placeholder']}")

# Get specific category
compute = inventory.get_category("compute")
print(compute["description"])

# Get available slots in a category
available_outputs = inventory.get_available_slots("output")
for slot_id, slot_data in available_outputs.items():
    print(f"{slot_id}: {slot_data['name']}")

# Update a slot (when hardware is added)
inventory.update_slot("vision", "slot_1", {
    "status": "available",
    "device": "/dev/video0"
})
inventory.save()
```

### Console Commands

From the M.O.L.O.C.H. console:
```
> /world
World Inventory Summary:
----------------------------------------
Version: 1.0.0
Phase:   2

Totals:
  Available:   12
  Placeholder: 22
  Empty:       26

> /inventory
World Inventory Details:
----------------------------------------

[INTERACTION] (6 defined)
  Input channels for human-to-M.O.L.O.C.H. communication
  Available: 1 | Placeholder: 5
    + Console Input

[OUTPUT] (7 defined)
  Output channels for M.O.L.O.C.H.-to-human communication
  Available: 3 | Placeholder: 4
    + Console Output
    + HDMI Audio
    + 3.5mm Jack
...
```

### Direct JSON Editing

```bash
# Edit the inventory
nano ~/moloch/core/world/state/world_inventory.json

# Validate JSON
python3 -m json.tool ~/moloch/core/world/state/world_inventory.json
```

## Slot Schema

Each slot has this structure:

```json
{
  "type": "sensor_type",
  "name": "Human-readable name",
  "status": "available|placeholder|empty|offline|error",
  "device": "/dev/xxx or null",
  "notes": "Optional notes",
  // Additional type-specific fields...
}
```

### Status Values

| Status | Meaning |
|--------|---------|
| `available` | Device is present and ready |
| `placeholder` | Slot reserved for future device |
| `empty` | Unused slot |
| `offline` | Device configured but not responding |
| `error` | Device has encountered an error |

## Growth Pattern

When Markus adds hardware:

1. **Physical**: Connect the device
2. **Detection**: `environment_watcher.py` detects new device
3. **Inventory Update**: Update the appropriate slot
4. **Verification**: Device is now "available"

Example: Adding a temperature sensor

```json
// Before (placeholder)
"slot_1": {
  "type": "room_temperature",
  "name": "Room Temperature Sensor",
  "status": "placeholder",
  "device": null
}

// After (available)
"slot_1": {
  "type": "room_temperature",
  "name": "Room Temperature Sensor",
  "status": "available",
  "device": "/dev/i2c-1",
  "address": "0x40",
  "model": "DHT22"
}
```

## Logging

World state operations are logged to:
```
~/moloch/logs/world.log
```

## Future Enhancements

- [ ] Auto-sync with environment_watcher detections
- [ ] Slot capability descriptions
- [ ] Hardware profiles (save/restore configurations)
- [ ] Dependency mapping (which outputs work with which inputs)
- [ ] Web-based inventory editor

## The Map

> "M.O.L.O.C.H. kennt seine Grenzen. Es weiß, was es werden könnte. Jeder leere Slot ist eine Einladung. Jeder Platzhalter ist ein Versprechen."

---

**The World Is Defined. The Possibilities Are Mapped.** 🗺️
