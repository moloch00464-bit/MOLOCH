#!/usr/bin/env python3
"""
ONVIF Capability Scanner - Direct Diagnostic Tool
==================================================

Comprehensive ONVIF diagnostic script that directly queries camera
capabilities without using UnifiedCameraController.

Queries:
- Device information
- Available services
- Service capabilities
- PTZ profiles and configurations
- Imaging settings and options
- Audio source configurations
- Media profiles

Prints raw ONVIF responses in formatted JSON.

Usage:
    python3 onvif_capability_scan.py
    python3 onvif_capability_scan.py --ip 192.168.178.25
"""

import sys
import json
import logging
import argparse
from typing import Any, Dict, Optional
from datetime import datetime

try:
    from onvif import ONVIFCamera
    from zeep.helpers import serialize_object
    from zeep.exceptions import Fault
    ONVIF_AVAILABLE = True
except ImportError:
    print("ERROR: onvif-zeep not installed")
    print("Install: pip install onvif-zeep")
    sys.exit(1)

# Camera defaults
DEFAULT_IP = "192.168.178.25"
DEFAULT_USERNAME = "Moloch_4.5"
DEFAULT_PASSWORD = "Auge666"
DEFAULT_PORT = 80

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Suppress verbose ONVIF logs
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str, char: str = "="):
    """Print formatted header."""
    width = 100
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)


def print_section(title: str):
    """Print formatted section."""
    print(f"\n{'─' * 100}")
    print(f"► {title}")
    print('─' * 100)


def format_json(obj: Any) -> str:
    """Format object as pretty JSON."""
    try:
        # Serialize zeep objects
        if hasattr(obj, '__class__') and 'zeep' in str(type(obj)):
            obj = serialize_object(obj)

        # Convert to JSON
        return json.dumps(obj, indent=2, default=str)
    except Exception as e:
        return str(obj)


def safe_query(func, *args, **kwargs) -> tuple[bool, Any, Optional[str]]:
    """
    Safely execute ONVIF query.

    Returns:
        (success, result, error_message)
    """
    try:
        result = func(*args, **kwargs)
        return True, result, None
    except Fault as e:
        return False, None, f"SOAP Fault: {e.message}"
    except Exception as e:
        return False, None, f"Error: {str(e)}"


class ONVIFScanner:
    """ONVIF capability scanner."""

    def __init__(self, ip: str, username: str, password: str, port: int = 80):
        """Initialize scanner."""
        self.ip = ip
        self.username = username
        self.password = password
        self.port = port
        self.camera: Optional[ONVIFCamera] = None

    def connect(self) -> bool:
        """Connect to camera."""
        print_section("CONNECTION")

        print(f"Camera IP:    {self.ip}")
        print(f"ONVIF Port:   {self.port}")
        print(f"Username:     {self.username}")
        print(f"Connecting...")

        try:
            self.camera = ONVIFCamera(
                self.ip,
                self.port,
                self.username,
                self.password
            )
            print("✓ Connected successfully")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    def scan_device_info(self):
        """Scan device information."""
        print_section("1. DEVICE INFORMATION")

        try:
            devicemgmt = self.camera.create_devicemgmt_service()

            # Get device information
            print("\n📋 Device Info:")
            success, device_info, error = safe_query(devicemgmt.GetDeviceInformation)
            if success:
                info_dict = serialize_object(device_info)
                print(format_json(info_dict))
            else:
                print(f"  ✗ Failed: {error}")

            # Get system date and time
            print("\n🕐 System Time:")
            success, sys_time, error = safe_query(devicemgmt.GetSystemDateAndTime)
            if success:
                time_dict = serialize_object(sys_time)
                print(format_json(time_dict))
            else:
                print(f"  ✗ Failed: {error}")

            # Get hostname
            print("\n🌐 Hostname:")
            success, hostname, error = safe_query(devicemgmt.GetHostname)
            if success:
                print(format_json(serialize_object(hostname)))
            else:
                print(f"  ✗ Failed: {error}")

            # Get network interfaces
            print("\n🔌 Network Interfaces:")
            success, interfaces, error = safe_query(devicemgmt.GetNetworkInterfaces)
            if success:
                for i, iface in enumerate(interfaces):
                    print(f"\n  Interface {i+1}:")
                    print(format_json(serialize_object(iface)))
            else:
                print(f"  ✗ Failed: {error}")

        except Exception as e:
            print(f"✗ Device info scan failed: {e}")

    def scan_services(self):
        """Scan available services."""
        print_section("2. AVAILABLE SERVICES")

        try:
            devicemgmt = self.camera.create_devicemgmt_service()

            print("\n📦 Services (IncludeCapability=False):")
            success, services, error = safe_query(devicemgmt.GetServices, False)
            if success:
                for i, service in enumerate(services):
                    service_dict = serialize_object(service)
                    print(f"\n  Service {i+1}:")
                    print(f"    Namespace: {service_dict.get('Namespace', 'N/A')}")
                    print(f"    XAddr:     {service_dict.get('XAddr', 'N/A')}")
                    print(f"    Version:   {service_dict.get('Version', 'N/A')}")
            else:
                print(f"  ✗ Failed: {error}")

            print("\n\n📦 Services (IncludeCapability=True):")
            success, services_cap, error = safe_query(devicemgmt.GetServices, True)
            if success:
                for i, service in enumerate(services_cap):
                    service_dict = serialize_object(service)
                    print(f"\n  Service {i+1}:")
                    print(format_json(service_dict))
            else:
                print(f"  ✗ Failed: {error}")

        except Exception as e:
            print(f"✗ Service scan failed: {e}")

    def scan_capabilities(self):
        """Scan device capabilities."""
        print_section("3. DEVICE CAPABILITIES")

        try:
            devicemgmt = self.camera.create_devicemgmt_service()

            print("\n🎯 Capabilities:")
            success, capabilities, error = safe_query(devicemgmt.GetCapabilities)
            if success:
                cap_dict = serialize_object(capabilities)
                print(format_json(cap_dict))
            else:
                print(f"  ✗ Failed: {error}")

        except Exception as e:
            print(f"✗ Capabilities scan failed: {e}")

    def scan_media_profiles(self):
        """Scan media profiles."""
        print_section("4. MEDIA PROFILES")

        try:
            media = self.camera.create_media_service()

            print("\n📹 Media Profiles:")
            success, profiles, error = safe_query(media.GetProfiles)
            if success:
                print(f"Found {len(profiles)} profile(s)\n")
                for i, profile in enumerate(profiles):
                    profile_dict = serialize_object(profile)
                    print(f"Profile {i+1}:")
                    print(format_json(profile_dict))
                    print()
            else:
                print(f"  ✗ Failed: {error}")

            # Get video sources
            print("\n📷 Video Sources:")
            success, video_sources, error = safe_query(media.GetVideoSources)
            if success:
                for i, source in enumerate(video_sources):
                    print(f"\n  Video Source {i+1}:")
                    print(format_json(serialize_object(source)))
            else:
                print(f"  ✗ Failed: {error}")

            # Get video source configurations
            print("\n⚙️  Video Source Configurations:")
            success, video_configs, error = safe_query(media.GetVideoSourceConfigurations)
            if success:
                for i, config in enumerate(video_configs):
                    print(f"\n  Config {i+1}:")
                    print(format_json(serialize_object(config)))
            else:
                print(f"  ✗ Failed: {error}")

            # Get video encoder configurations
            print("\n🎬 Video Encoder Configurations:")
            success, encoder_configs, error = safe_query(media.GetVideoEncoderConfigurations)
            if success:
                for i, config in enumerate(encoder_configs):
                    print(f"\n  Encoder {i+1}:")
                    print(format_json(serialize_object(config)))
            else:
                print(f"  ✗ Failed: {error}")

        except Exception as e:
            print(f"✗ Media profile scan failed: {e}")

    def scan_ptz_capabilities(self):
        """Scan PTZ capabilities."""
        print_section("5. PTZ CAPABILITIES")

        try:
            ptz = self.camera.create_ptz_service()
            media = self.camera.create_media_service()

            # Get profiles first
            success, profiles, error = safe_query(media.GetProfiles)
            if not success or not profiles:
                print("  ✗ No media profiles available")
                return

            profile_token = profiles[0].token
            print(f"Using profile: {profile_token}\n")

            # Get PTZ configuration
            print("🎮 PTZ Configuration:")
            success, ptz_config, error = safe_query(
                ptz.GetConfiguration,
                {'PTZConfigurationToken': profiles[0].PTZConfiguration.token}
            )
            if success:
                print(format_json(serialize_object(ptz_config)))
            else:
                print(f"  ✗ Failed: {error}")

            # Get PTZ configuration options
            print("\n\n⚙️  PTZ Configuration Options:")
            success, ptz_options, error = safe_query(
                ptz.GetConfigurationOptions,
                {'ConfigurationToken': profiles[0].PTZConfiguration.token}
            )
            if success:
                print(format_json(serialize_object(ptz_options)))
            else:
                print(f"  ✗ Failed: {error}")

            # Get current PTZ status
            print("\n\n📍 Current PTZ Status:")
            success, ptz_status, error = safe_query(
                ptz.GetStatus,
                {'ProfileToken': profile_token}
            )
            if success:
                print(format_json(serialize_object(ptz_status)))
            else:
                print(f"  ✗ Failed: {error}")

            # Get PTZ presets
            print("\n\n🔖 PTZ Presets:")
            success, presets, error = safe_query(
                ptz.GetPresets,
                {'ProfileToken': profile_token}
            )
            if success:
                if presets:
                    for i, preset in enumerate(presets):
                        print(f"\n  Preset {i+1}:")
                        print(format_json(serialize_object(preset)))
                else:
                    print("  No presets configured")
            else:
                print(f"  ✗ Failed: {error}")

            # Get PTZ nodes
            print("\n\n🔧 PTZ Nodes:")
            success, nodes, error = safe_query(ptz.GetNodes)
            if success:
                for i, node in enumerate(nodes):
                    print(f"\n  Node {i+1}:")
                    print(format_json(serialize_object(node)))
            else:
                print(f"  ✗ Failed: {error}")

        except Exception as e:
            print(f"✗ PTZ scan failed: {e}")

    def scan_imaging_capabilities(self):
        """Scan imaging capabilities."""
        print_section("6. IMAGING CAPABILITIES")

        print("Attempting to create Imaging Service...")

        try:
            imaging = self.camera.create_imaging_service()
            print("✓ Imaging service created successfully\n")

            # Get video source token
            media = self.camera.create_media_service()
            success, profiles, error = safe_query(media.GetProfiles)
            if not success or not profiles:
                print("  ✗ No media profiles available")
                return

            video_source_token = profiles[0].VideoSourceConfiguration.SourceToken
            print(f"Video Source Token: {video_source_token}\n")

            # Get imaging settings
            print("🎨 GetImagingSettings:")
            success, settings, error = safe_query(
                imaging.GetImagingSettings,
                {'VideoSourceToken': video_source_token}
            )
            if success:
                print(format_json(serialize_object(settings)))
            else:
                print(f"  ✗ Failed: {error}")

            # Get imaging options
            print("\n\n⚙️  GetOptions (Imaging):")
            success, options, error = safe_query(
                imaging.GetOptions,
                {'VideoSourceToken': video_source_token}
            )
            if success:
                print(format_json(serialize_object(options)))
            else:
                print(f"  ✗ Failed: {error}")

            # Get move options
            print("\n\n↔️  GetMoveOptions:")
            success, move_options, error = safe_query(
                imaging.GetMoveOptions,
                {'VideoSourceToken': video_source_token}
            )
            if success:
                print(format_json(serialize_object(move_options)))
            else:
                print(f"  ✗ Failed: {error}")

            # Get status
            print("\n\n📊 GetStatus:")
            success, status, error = safe_query(
                imaging.GetStatus,
                {'VideoSourceToken': video_source_token}
            )
            if success:
                print(format_json(serialize_object(status)))
            else:
                print(f"  ✗ Failed: {error}")

        except Exception as e:
            error_msg = str(e)
            if "doesn`t support service: imaging" in error_msg.lower():
                print("✗ Imaging Service NOT SUPPORTED by camera")
                print("  This camera does not implement ONVIF Imaging Service")
                print("  Features unavailable:")
                print("    - IR/Night mode control")
                print("    - Brightness/Contrast adjustment")
                print("    - Backlight compensation")
                print("    - Wide Dynamic Range")
                print("    - Day/Night filter control")
            else:
                print(f"✗ Imaging scan failed: {e}")

    def scan_audio_capabilities(self):
        """Scan audio capabilities."""
        print_section("7. AUDIO CAPABILITIES")

        try:
            media = self.camera.create_media_service()

            # Get audio sources
            print("🎤 GetAudioSources:")
            success, audio_sources, error = safe_query(media.GetAudioSources)
            if success:
                if audio_sources:
                    for i, source in enumerate(audio_sources):
                        print(f"\n  Audio Source {i+1}:")
                        print(format_json(serialize_object(source)))
                else:
                    print("  No audio sources found")
            else:
                print(f"  ✗ Failed: {error}")

            # Get audio source configurations
            print("\n\n⚙️  GetAudioSourceConfigurations:")
            success, audio_configs, error = safe_query(media.GetAudioSourceConfigurations)
            if success:
                if audio_configs:
                    for i, config in enumerate(audio_configs):
                        print(f"\n  Audio Source Config {i+1}:")
                        print(format_json(serialize_object(config)))
                else:
                    print("  No audio source configurations found")
            else:
                print(f"  ✗ Failed: {error}")

            # Get audio encoder configurations
            print("\n\n🎚️  GetAudioEncoderConfigurations:")
            success, encoder_configs, error = safe_query(media.GetAudioEncoderConfigurations)
            if success:
                if encoder_configs:
                    for i, config in enumerate(encoder_configs):
                        print(f"\n  Audio Encoder Config {i+1}:")
                        print(format_json(serialize_object(config)))
                else:
                    print("  No audio encoder configurations found")
            else:
                print(f"  ✗ Failed: {error}")

            # Get audio outputs
            print("\n\n🔊 GetAudioOutputs:")
            success, audio_outputs, error = safe_query(media.GetAudioOutputs)
            if success:
                if audio_outputs:
                    for i, output in enumerate(audio_outputs):
                        print(f"\n  Audio Output {i+1}:")
                        print(format_json(serialize_object(output)))
                else:
                    print("  No audio outputs found")
            else:
                print(f"  ✗ Failed: {error}")

            # Get audio decoder configurations
            print("\n\n🎧 GetAudioDecoderConfigurations:")
            success, decoder_configs, error = safe_query(media.GetAudioDecoderConfigurations)
            if success:
                if decoder_configs:
                    for i, config in enumerate(decoder_configs):
                        print(f"\n  Audio Decoder Config {i+1}:")
                        print(format_json(serialize_object(config)))
                else:
                    print("  No audio decoder configurations found")
            else:
                print(f"  ✗ Failed: {error}")

        except Exception as e:
            print(f"✗ Audio scan failed: {e}")

    def scan_deviceio_capabilities(self):
        """Scan Device I/O capabilities."""
        print_section("8. DEVICE I/O CAPABILITIES")

        print("Attempting to create Device I/O Service...")

        try:
            deviceio = self.camera.create_deviceio_service()
            print("✓ Device I/O service created successfully\n")

            # Get relay outputs
            print("🔌 GetRelayOutputs:")
            success, relays, error = safe_query(deviceio.GetRelayOutputs)
            if success:
                if relays:
                    for i, relay in enumerate(relays):
                        print(f"\n  Relay {i+1}:")
                        print(format_json(serialize_object(relay)))
                else:
                    print("  No relay outputs found")
            else:
                print(f"  ✗ Failed: {error}")

            # Get digital inputs
            print("\n\n📥 GetDigitalInputs:")
            success, inputs, error = safe_query(deviceio.GetDigitalInputs)
            if success:
                if inputs:
                    for i, inp in enumerate(inputs):
                        print(f"\n  Digital Input {i+1}:")
                        print(format_json(serialize_object(inp)))
                else:
                    print("  No digital inputs found")
            else:
                print(f"  ✗ Failed: {error}")

            # Get service capabilities
            print("\n\n🎯 GetServiceCapabilities:")
            success, capabilities, error = safe_query(deviceio.GetServiceCapabilities)
            if success:
                print(format_json(serialize_object(capabilities)))
            else:
                print(f"  ✗ Failed: {error}")

        except Exception as e:
            error_msg = str(e)
            if "doesn`t support service: deviceio" in error_msg.lower():
                print("✗ Device I/O Service NOT SUPPORTED by camera")
                print("  This camera does not implement ONVIF Device I/O Service")
                print("  Features unavailable:")
                print("    - LED control")
                print("    - Relay control")
                print("    - Digital I/O")
            else:
                print(f"✗ Device I/O scan failed: {e}")

    def scan_events_capabilities(self):
        """Scan Events capabilities."""
        print_section("9. EVENTS CAPABILITIES")

        try:
            events = self.camera.create_events_service()

            # Get service capabilities
            print("🎯 GetServiceCapabilities:")
            success, capabilities, error = safe_query(events.GetServiceCapabilities)
            if success:
                print(format_json(serialize_object(capabilities)))
            else:
                print(f"  ✗ Failed: {error}")

            # Get event properties
            print("\n\n📋 GetEventProperties:")
            success, properties, error = safe_query(events.GetEventProperties)
            if success:
                # Event properties contain complex nested structures
                print(format_json(serialize_object(properties)))
            else:
                print(f"  ✗ Failed: {error}")

        except Exception as e:
            print(f"✗ Events scan failed: {e}")

    def run_full_scan(self):
        """Run complete capability scan."""
        print_header("ONVIF CAPABILITY SCANNER", "=")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Connect
        if not self.connect():
            return False

        # Run all scans
        try:
            self.scan_device_info()
            self.scan_services()
            self.scan_capabilities()
            self.scan_media_profiles()
            self.scan_ptz_capabilities()
            self.scan_imaging_capabilities()
            self.scan_audio_capabilities()
            self.scan_deviceio_capabilities()
            self.scan_events_capabilities()

            # Summary
            print_header("SCAN COMPLETE", "=")
            print(f"Camera:    {self.ip}")
            print(f"Time:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Status:    ✓ Complete")
            print()

            return True

        except Exception as e:
            print(f"\n✗ Scan failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ONVIF Capability Scanner - Direct diagnostic tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--ip', default=DEFAULT_IP, help=f'Camera IP (default: {DEFAULT_IP})')
    parser.add_argument('--user', default=DEFAULT_USERNAME, help=f'Username (default: {DEFAULT_USERNAME})')
    parser.add_argument('--password', default=DEFAULT_PASSWORD, help='Password')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'ONVIF port (default: {DEFAULT_PORT})')

    args = parser.parse_args()

    # Create scanner
    scanner = ONVIFScanner(
        ip=args.ip,
        username=args.user,
        password=args.password,
        port=args.port
    )

    # Run scan
    success = scanner.run_full_scan()

    # Exit
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
