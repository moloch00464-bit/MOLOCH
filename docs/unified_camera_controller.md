# Unified Camera Controller

**Created:** 2026-02-07
**Camera Model:** Sonoff GK-200MP2-B (CAM-PT2)
**Firmware:** 1.0.8

---

## Overview

The `UnifiedCameraController` class provides a **single, unified interface** for ALL camera control operations in the M.O.L.O.C.H. system. This replaces direct ONVIF API calls and consolidates camera functionality.

**Key Principle:** No direct camera API calls should be made outside this class.

---

## Architecture

```
Application Code (Vision, GUI, etc.)
         ↓
  UnifiedCameraController
         ↓
    ONVIF Camera API
         ↓
  Sonoff GK-200MP2-B Camera
```

---

## File Locations

| Component | Path |
|-----------|------|
| **Main Controller** | `/home/molochzuhause/moloch/core/hardware/unified_camera_controller.py` |
| **Test Script** | `/home/molochzuhause/moloch/scripts/test_camera_controller.py` |
| **Capability Probe** | `/home/molochzuhause/moloch/scripts/probe_camera_capabilities.py` |

---

## Supported Features

### ✅ Available Features (via ONVIF)

| Feature | Status | Details |
|---------|--------|---------|
| **PTZ Control** | ✅ Fully Supported | Pan (-168.4° to +170.0°), Tilt (-78.0° to +78.8°) |
| **Absolute Movement** | ✅ Fully Supported | Move to specific pan/tilt position |
| **Relative Movement** | ✅ Fully Supported | Move by delta degrees |
| **Continuous Movement** | ✅ Fully Supported | Velocity-based control |
| **Position Tracking** | ✅ Fully Supported | Real-time position feedback |
| **Patrol Scanning** | ✅ Fully Supported | 360° automated scan pattern |
| **Audio Stream** | ✅ Available | G.711 A-law, 8kHz, mono via RTSP |
| **Status Query** | ✅ Fully Supported | Complete camera status |

### ❌ Unavailable Features (Hardware Limitations)

| Feature | Status | Reason |
|---------|--------|--------|
| **Night/IR Mode** | ❌ Not Supported | No ONVIF Imaging Service |
| **LED Control** | ❌ Not Supported | No ONVIF Device I/O Service |
| **Sleep/Privacy Mode** | ❌ Not Supported | No ONVIF Privacy API |
| **Mic Gain Control** | ❌ Not Supported | No ONVIF Audio Input Config |

**Note:** These features require proprietary HTTP API or mobile app access, which the camera does not expose via standard protocols.

---

## Usage Examples

### Basic Connection

```python
from core.hardware.unified_camera_controller import UnifiedCameraController

# Create controller
controller = UnifiedCameraController(
    camera_ip="192.168.178.25",
    username="CHANGE_ME",
    password="CHANGE_ME"
)

# Connect
if controller.connect():
    print("✓ Connected")
else:
    print("✗ Connection failed")
```

### PTZ Movement

```python
# Move to absolute position
controller.move_absolute(pan_deg=45.0, tilt_deg=10.0, speed=0.5)

# Move relative to current position
controller.move_relative(pan_delta=-30.0, tilt_delta=5.0)

# Continuous velocity-based movement
controller.continuous_move(vel_pan=0.3, vel_tilt=0.1, timeout_sec=2.0)

# Go to home position (0, 0)
controller.goto_home()

# Stop all movement
controller.stop()
```

### Position Tracking

```python
# Get current position
pos = controller.get_position()
print(f"Pan: {pos.pan:.1f}°, Tilt: {pos.tilt:.1f}°")
print(f"Zoom: {pos.zoom:.2f}, Moving: {pos.moving}")
```

### Person Tracking

```python
from core.hardware.unified_camera_controller import Detection

# Process detection from vision system
detection = Detection(
    person_id="person_1",
    bbox=(100, 100, 200, 300),  # x, y, w, h
    center_x=0.6,  # Normalized 0-1
    center_y=0.4,  # Normalized 0-1
    confidence=0.85
)

# Track the target
controller.process_detection(detection)

# Check if target lost and start patrol
controller.check_target_lost()
```

### Status Query

```python
# Get complete status
status = controller.get_status()

print(f"Model: {status.model}")
print(f"Firmware: {status.firmware}")
print(f"Position: Pan={status.position.pan:.1f}°, Tilt={status.position.tilt:.1f}°")
print(f"PTZ Available: {status.ptz_available}")
print(f"Night Mode Available: {status.night_mode_available}")

# Convert to dictionary
status_dict = status.to_dict()
```

### Audio Stream Access

```python
# Get RTSP URLs
main_stream = controller.get_rtsp_url("main")  # 1920x1080
sub_stream = controller.get_rtsp_url("sub")    # 640x360
audio_url = controller.get_audio_stream_url()

print(f"Main: {main_stream}")
print(f"Audio: G.711 A-law, 8kHz, mono")
```

### Control Modes

```python
# Set control mode
controller.enable_autonomous()    # AI control
controller.enable_manual()        # User control
controller.enable_safe_mode()     # No movement

# Acquire exclusive control
if controller.acquire_exclusive("VisionLab"):
    # Do manual control
    controller.move_absolute(0, 0)
    # Release when done
    controller.release_exclusive("VisionLab")
```

### Singleton Access

```python
from core.hardware.unified_camera_controller import get_camera_controller

# Get singleton instance (auto-connects)
controller = get_camera_controller()

# Use it
controller.goto_home()
```

---

## Testing

### Run Full Test Suite

```bash
cd /home/molochzuhause/moloch
python3 scripts/test_camera_controller.py
```

### Quick Test (Shorter Delays)

```bash
python3 scripts/test_camera_controller.py --quick
```

### PTZ Only

```bash
python3 scripts/test_camera_controller.py --ptz-only
```

### Status Query Only

```bash
python3 scripts/test_camera_controller.py --status
```

### Probe Camera Capabilities

```bash
python3 scripts/probe_camera_capabilities.py
```

---

## Test Results (2026-02-07)

**Success Rate:** 88.9% (8/9 tests passed)

| Test | Result | Notes |
|------|--------|-------|
| Connection | ✅ PASS | Successfully connected to camera |
| Status Query | ✅ PASS | All fields populated correctly |
| PTZ Relative Movement | ✅ PASS | All movements executed |
| PTZ Absolute Positioning | ⚠️ FAIL | Timing issue in quick mode only |
| Night Mode Control | ✅ PASS | Correctly reports as unsupported |
| LED Control | ✅ PASS | Correctly reports as unsupported |
| Sleep Mode | ✅ PASS | Correctly reports as unsupported |
| Mic Gain Control | ✅ PASS | Correctly reports as unsupported |
| Audio Stream | ✅ PASS | RTSP URLs generated correctly |
| Control Modes | ✅ PASS | Mode switching works |

**Note:** The absolute positioning test fails in `--quick` mode because the camera needs more time to reach far positions. In normal mode (without `--quick`), this test passes.

---

## Camera Specifications

### Hardware

- **Model:** Sonoff GK-200MP2-B (CAM-PT2)
- **Firmware:** 1.0.8
- **Manufacturer:** SONOFF
- **Serial:** 25370200016333

### PTZ Capabilities

- **Pan Range:** -168.4° to +170.0° (338.4° total)
- **Tilt Range:** -78.0° to +78.8° (156.8° total)
- **Pan Speed:** ~20°/second
- **Tilt Speed:** ~12°/second
- **Calibrated:** 2026-02-04

### Video Streams

| Stream | Resolution | FPS | Bitrate | Codec |
|--------|------------|-----|---------|-------|
| Main | 1920x1080 | 20 | 1500 kbps | H.264 |
| Sub | 640x360 | 20 | 1500 kbps | H.264 |

### Audio Stream

- **Codec:** G.711 A-law
- **Sample Rate:** 8000 Hz
- **Channels:** Mono (1)
- **Bitrate:** 64 kbps

### ONVIF Services

| Service | Status | Version |
|---------|--------|---------|
| Device Management | ✅ Available | 10 |
| Media | ✅ Available | 10 |
| PTZ | ✅ Available | 20 |
| Events | ✅ Available | 10 |
| Imaging | ❌ Not Available | - |
| Device I/O | ❌ Not Available | - |
| Analytics | ❌ Not Available | - |

---

## Integration Notes

### Replacing Old Code

When migrating existing code to use `UnifiedCameraController`:

**Before:**
```python
from core.hardware.sonoff_camera_controller import SonoffCameraController

camera = SonoffCameraController()
camera.connect()
camera.move_absolute(0, 0)
```

**After:**
```python
from core.hardware.unified_camera_controller import UnifiedCameraController

camera = UnifiedCameraController()
camera.connect()
camera.move_absolute(0, 0)
```

The API is mostly compatible. Key differences:

1. Class name changed: `SonoffCameraController` → `UnifiedCameraController`
2. Added explicit feature support flags in `get_status()`
3. Added stub methods for unsupported features (night mode, LED, sleep, mic gain)
4. Improved documentation and type hints

### Thread Safety

The controller uses internal locking for thread-safe operations:

- `_lock`: Protects camera state
- `_exclusive_lock`: Protects exclusive access control

Safe to call from multiple threads.

### Exclusive Access Pattern

```python
# Acquire exclusive control
if controller.acquire_exclusive("owner_id"):
    try:
        # Do exclusive operations
        controller.move_absolute(45, 10)
    finally:
        # Always release
        controller.release_exclusive("owner_id")
```

### Callbacks

```python
def on_position_changed(position):
    print(f"New position: {position.pan:.1f}°, {position.tilt:.1f}°")

def on_tracking_state_changed(state):
    print(f"Tracking state: {state.name}")

controller.on_position_update = on_position_changed
controller.on_state_change = on_tracking_state_changed
```

---

## Future Enhancements

### Possible Additions

1. **Reverse Engineering:** Analyze mobile app traffic to discover proprietary APIs for:
   - IR/Night mode control
   - LED brightness levels
   - Sleep/Privacy mode
   - Mic gain adjustment

2. **Cloud Integration:** If camera supports cloud control, add cloud API wrapper

3. **Video Recording:** Add local video recording capability

4. **Motion Detection:** Add ONVIF motion detection events (camera supports Events Service)

5. **Presets:** Although camera reports no preset support, may be worth investigating

### How to Add Proprietary API Support

If you discover HTTP endpoints for unsupported features:

1. Add HTTP client to `UnifiedCameraController.__init__`
2. Implement the stub methods (e.g., `set_night_mode`)
3. Update feature flags in `get_status()`
4. Add tests in `test_camera_controller.py`

Example:

```python
def set_night_mode(self, mode: NightMode) -> bool:
    """Set IR/Night mode via proprietary HTTP API."""
    try:
        endpoint = f"http://{self.camera_ip}/api/ir_mode"
        payload = {"mode": mode.name.lower()}
        response = requests.post(endpoint, json=payload,
                               auth=(self.username, self.password))
        return response.status_code == 200
    except Exception as e:
        self.logger.error(f"Set night mode failed: {e}")
        return False
```

---

## Troubleshooting

### Connection Issues

```python
# Check if camera is reachable
import subprocess
result = subprocess.run(['ping', '-c', '1', '192.168.178.25'])
if result.returncode != 0:
    print("Camera not reachable")

# Check ONVIF port
import socket
sock = socket.socket()
sock.settimeout(3)
result = sock.connect_ex(('192.168.178.25', 80))
if result != 0:
    print("ONVIF port 80 not open")
```

### PTZ Not Moving

```python
# Check control mode
if controller.mode == ControlMode.SAFE_MODE:
    print("Camera in SAFE_MODE - enable autonomous or manual")
    controller.enable_autonomous()

# Check exclusive access
if controller.exclusive_owner:
    print(f"PTZ locked by: {controller.exclusive_owner}")
```

### Position Drift

```python
# Update position before moving
controller.get_position()  # Force position update
controller.move_absolute(0, 0)  # Then move
```

---

## License

Part of the M.O.L.O.C.H. System
Copyright © 2026

---

## Contact

For issues or questions:
- Check logs: Look for `UnifiedCameraController` logger output
- Run diagnostic: `python3 scripts/probe_camera_capabilities.py`
- Run tests: `python3 scripts/test_camera_controller.py`
