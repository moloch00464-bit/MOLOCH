# ONVIF Diagnostic Scanner - Quick Guide

**Script:** `scripts/onvif_capability_scan.py`
**Created:** 2026-02-07

---

## Purpose

Direct ONVIF diagnostic tool that queries all camera capabilities without using the UnifiedCameraController. Produces detailed raw responses for troubleshooting and capability discovery.

---

## Usage

### Basic Scan (Default Camera)

```bash
cd ~/moloch
python3 scripts/onvif_capability_scan.py
```

### Custom Camera

```bash
python3 scripts/onvif_capability_scan.py --ip 192.168.1.100 --user admin --password mypass
```

### Save Output

```bash
python3 scripts/onvif_capability_scan.py > diagnostic_output.txt 2>&1
```

---

## What It Queries

### 1. Device Information
- Manufacturer, Model, Firmware
- System time and timezone
- Hostname
- Network interfaces (IP, MAC, MTU)

### 2. Available Services
- All ONVIF services with versions
- Service XAddr endpoints
- Service capabilities

### 3. Device Capabilities
- Discovery capabilities
- Security features
- System features
- Supported ONVIF versions

### 4. Media Profiles
- Video sources
- Video encoder configurations
- Video source configurations
- Stream details (codec, resolution, bitrate, FPS)

### 5. PTZ Capabilities
- PTZ configuration
- PTZ configuration options
- Current PTZ status (position, moving)
- PTZ presets
- PTZ nodes

### 6. Imaging Capabilities ⚠️
- **GetImagingSettings** (IR mode, brightness, contrast)
- **GetOptions** (available imaging options)
- **GetMoveOptions** (focus move options)
- **GetStatus** (imaging status)

**Result for Sonoff GK-200MP2-B:**
```
✗ Imaging Service NOT SUPPORTED by camera
  Features unavailable:
    - IR/Night mode control
    - Brightness/Contrast adjustment
    - Backlight compensation
    - Wide Dynamic Range
    - Day/Night filter control
```

### 7. Audio Capabilities
- **GetAudioSources**
- **GetAudioSourceConfigurations**
- **GetAudioEncoderConfigurations**
- **GetAudioOutputs**
- **GetAudioDecoderConfigurations**

**Result for Sonoff GK-200MP2-B:**
- ✓ 1 audio source (mono)
- ✓ G.711 encoding, 8kHz
- ✗ No audio outputs (backchannel)

### 8. Device I/O Capabilities ⚠️
- **GetRelayOutputs** (LED control)
- **GetDigitalInputs**
- **GetServiceCapabilities**

**Result for Sonoff GK-200MP2-B:**
```
✗ Device I/O Service NOT SUPPORTED by camera
  Features unavailable:
    - LED control
    - Relay control
    - Digital I/O
```

### 9. Events Capabilities
- **GetServiceCapabilities**
- **GetEventProperties**

**Result for Sonoff GK-200MP2-B:**
- ✓ Events service available
- ✓ Event properties available

---

## Key Findings for Sonoff GK-200MP2-B

### ✅ Supported (via ONVIF)

| Service | Status | Details |
|---------|--------|---------|
| Device Management | ✅ Available | Version 19.12 |
| Media | ✅ Available | Version 19.6 |
| PTZ | ✅ Available | Version 20.12 |
| Events | ✅ Available | Version 18.6 |

### ❌ Not Supported (No ONVIF Service)

| Service | Status | Impact |
|---------|--------|--------|
| Imaging | ❌ Not Available | Cannot control IR/Night mode, brightness, contrast |
| Device I/O | ❌ Listed but non-functional | Cannot control LED, relays, digital I/O |

**Note:** Device I/O service appears in the service list but returns "not supported" error when queried. This is a firmware quirk.

---

## Output Format

The script produces **formatted JSON output** for all responses:

```json
{
  "Manufacturer": "SONOFF",
  "Model": "CAM-PT2",
  "FirmwareVersion": "1.0.8",
  "SerialNumber": "25370200016333",
  "HardwareId": "CAM-PT2"
}
```

All ONVIF responses are serialized from Zeep objects to Python dictionaries, then formatted as JSON for readability.

---

## Sample Output Sections

### PTZ Status Example

```json
{
  "Position": {
    "PanTilt": {
      "x": 0.0,
      "y": 0.0,
      "space": "http://www.onvif.org/ver10/tptz/PanTiltSpaces/PositionGenericSpace"
    },
    "Zoom": {
      "x": 0.0,
      "space": "http://www.onvif.org/ver10/tptz/ZoomSpaces/PositionGenericSpace"
    }
  },
  "MoveStatus": {
    "PanTilt": "IDLE",
    "Zoom": "IDLE"
  },
  "UtcTime": "2026-02-07T14:13:26Z"
}
```

### Audio Source Example

```json
{
  "Channels": 1,
  "token": "AudioSourceToken",
  "_attr_1": {}
}
```

---

## Comparison with probe_camera_capabilities.py

| Feature | probe_camera_capabilities.py | onvif_capability_scan.py |
|---------|------------------------------|--------------------------|
| **Scope** | High-level overview | Detailed raw queries |
| **Output** | Summary | Full JSON responses |
| **Imaging Queries** | Basic check | GetSettings, GetOptions, GetMoveOptions, GetStatus |
| **Audio Queries** | Basic list | Full configurations and codecs |
| **PTZ Queries** | Basic status | Configuration, options, nodes, presets |
| **Error Handling** | Generic | Specific per-query |
| **Use Case** | Quick check | Deep troubleshooting |

---

## Troubleshooting with This Script

### Problem: Camera not responding

**Check:**
1. Connection section - does it connect?
2. Network interfaces - correct IP?
3. Services section - are services listed?

### Problem: PTZ not working

**Check:**
1. Section 5: PTZ Capabilities
2. Look for PTZ configuration and options
3. Verify PTZ status shows position updates

### Problem: No audio

**Check:**
1. Section 7: Audio Capabilities
2. Verify audio sources exist
3. Check audio encoder configurations

### Problem: Want to add LED control

**Check:**
1. Section 8: Device I/O Capabilities
2. If "NOT SUPPORTED" → need proprietary API
3. If relays exist → implement via GetRelayOutputs/SetRelayOutputSettings

---

## Integration with UnifiedCameraController

This diagnostic script is **independent** of UnifiedCameraController:

```
┌─────────────────────────────────┐
│  onvif_capability_scan.py       │  ← Direct ONVIF queries
│  (diagnostic)                   │     for troubleshooting
└─────────────────────────────────┘

┌─────────────────────────────────┐
│  UnifiedCameraController        │  ← Production camera control
│  (application code)             │
└─────────────────────────────────┘
```

Use the diagnostic script to:
- Verify camera capabilities
- Troubleshoot connection issues
- Discover new features
- Test ONVIF responses

Use UnifiedCameraController for:
- Production camera control
- Integration with vision pipeline
- Application logic

---

## Full Output Reference

Complete diagnostic output saved to:
```
~/moloch/docs/onvif_diagnostic_output.txt
```

**Size:** ~1,269 lines of detailed ONVIF responses

---

## Command Line Options

```bash
python3 scripts/onvif_capability_scan.py --help

Options:
  --ip IP           Camera IP address (default: 192.168.178.25)
  --user USER       ONVIF username (default: Moloch_4.5)
  --password PASS   ONVIF password
  --port PORT       ONVIF port (default: 80)
```

---

## Exit Codes

- **0** - Scan completed successfully
- **1** - Scan failed (connection error, exception)

---

## Dependencies

```bash
pip install onvif-zeep
```

**Version:** onvif-zeep 0.2.12+ (installed in user's environment)

---

## Related Files

| File | Purpose |
|------|---------|
| [scripts/onvif_capability_scan.py](../scripts/onvif_capability_scan.py) | Main diagnostic script |
| [docs/onvif_diagnostic_output.txt](onvif_diagnostic_output.txt) | Full scan output (1,269 lines) |
| [scripts/probe_camera_capabilities.py](../scripts/probe_camera_capabilities.py) | Quick capability probe |
| [scripts/test_camera_controller.py](../scripts/test_camera_controller.py) | UnifiedCameraController test suite |

---

## License

Part of the M.O.L.O.C.H. System
Copyright © 2026
