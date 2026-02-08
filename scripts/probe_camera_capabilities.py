#!/usr/bin/env python3
"""
Probe Sonoff Camera ONVIF Capabilities
========================================

Discovers all available ONVIF services and imaging capabilities
for the Sonoff GK-200MP2-B camera.

This script helps identify what features are actually available:
- Imaging settings (brightness, contrast, etc.)
- IR/Night mode control
- Day/Night filter
- Backlight compensation
- Wide Dynamic Range
- Audio input/output
"""

import logging
from onvif import ONVIFCamera
from zeep.helpers import serialize_object
import json

# Camera credentials
CAMERA_IP = "192.168.178.25"
USERNAME = "Moloch_4.5"
PASSWORD = "Auge666"
ONVIF_PORT = 80

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def probe_all_capabilities():
    """Probe all ONVIF capabilities of the camera."""

    print("=" * 80)
    print("SONOFF GK-200MP2-B CAMERA CAPABILITY PROBE")
    print("=" * 80)

    try:
        # Connect to camera
        print(f"\n[1] Connecting to {CAMERA_IP}...")
        camera = ONVIFCamera(CAMERA_IP, ONVIF_PORT, USERNAME, PASSWORD)
        print("✓ Connected successfully")

        # =====================================================================
        # Device Management Service
        # =====================================================================
        print("\n[2] Device Management Service")
        print("-" * 80)
        try:
            devicemgmt = camera.create_devicemgmt_service()

            # Get device information
            device_info = devicemgmt.GetDeviceInformation()
            print(f"  Manufacturer: {device_info.Manufacturer}")
            print(f"  Model: {device_info.Model}")
            print(f"  FirmwareVersion: {device_info.FirmwareVersion}")
            print(f"  SerialNumber: {device_info.SerialNumber}")
            print(f"  HardwareId: {device_info.HardwareId}")

            # Get available services
            print("\n  Available Services:")
            services = devicemgmt.GetServices(False)
            for service in services:
                print(f"    - {service.Namespace}: {service.XAddr}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

        # =====================================================================
        # Media Service
        # =====================================================================
        print("\n[3] Media Service")
        print("-" * 80)
        try:
            media_service = camera.create_media_service()

            # Get profiles
            profiles = media_service.GetProfiles()
            print(f"  Found {len(profiles)} media profile(s)")

            for i, profile in enumerate(profiles):
                print(f"\n  Profile {i+1}:")
                print(f"    Token: {profile.token}")
                print(f"    Name: {profile.Name}")

                # Video source configuration
                if profile.VideoSourceConfiguration:
                    print(f"    Video Source: {profile.VideoSourceConfiguration.SourceToken}")

                # Video encoder configuration
                if profile.VideoEncoderConfiguration:
                    enc = profile.VideoEncoderConfiguration
                    print(f"    Video Encoding: {enc.Encoding}")
                    print(f"    Resolution: {enc.Resolution.Width}x{enc.Resolution.Height}")
                    print(f"    Quality: {enc.Quality}")
                    print(f"    FrameRate: {enc.RateControl.FrameRateLimit}")
                    print(f"    Bitrate: {enc.RateControl.BitrateLimit}")

                # Audio configuration
                if profile.AudioEncoderConfiguration:
                    audio = profile.AudioEncoderConfiguration
                    print(f"    Audio Encoding: {audio.Encoding}")
                    print(f"    Sample Rate: {audio.SampleRate}")
                    print(f"    Bitrate: {audio.Bitrate}")

                # PTZ configuration
                if profile.PTZConfiguration:
                    print(f"    PTZ: Enabled")

        except Exception as e:
            print(f"  ✗ Error: {e}")

        # =====================================================================
        # Imaging Service (KEY FOR NEW FEATURES)
        # =====================================================================
        print("\n[4] Imaging Service (IR/Night Mode, LED, etc.)")
        print("-" * 80)
        try:
            imaging_service = camera.create_imaging_service()

            # Get video source token
            media_service = camera.create_media_service()
            profiles = media_service.GetProfiles()
            if profiles:
                video_source_token = profiles[0].VideoSourceConfiguration.SourceToken
                print(f"  Video Source Token: {video_source_token}")

                # Get imaging settings
                print("\n  Current Imaging Settings:")
                try:
                    settings = imaging_service.GetImagingSettings({'VideoSourceToken': video_source_token})
                    settings_dict = serialize_object(settings)
                    print(json.dumps(settings_dict, indent=4))
                except Exception as e:
                    print(f"    ✗ GetImagingSettings failed: {e}")

                # Get imaging options (what's available)
                print("\n  Available Imaging Options:")
                try:
                    options = imaging_service.GetOptions({'VideoSourceToken': video_source_token})
                    options_dict = serialize_object(options)
                    print(json.dumps(options_dict, indent=4))
                except Exception as e:
                    print(f"    ✗ GetOptions failed: {e}")

                # Check for specific features
                print("\n  Feature Support Check:")

                # Day/Night mode
                try:
                    if hasattr(settings, 'IrCutFilter'):
                        print(f"    ✓ IR Cut Filter: {settings.IrCutFilter}")
                    else:
                        print(f"    ✗ IR Cut Filter: Not available")
                except:
                    print(f"    ✗ IR Cut Filter: Not accessible")

                # Brightness
                try:
                    if hasattr(settings, 'Brightness'):
                        print(f"    ✓ Brightness: {settings.Brightness}")
                    else:
                        print(f"    ✗ Brightness: Not available")
                except:
                    print(f"    ✗ Brightness: Not accessible")

                # Wide Dynamic Range
                try:
                    if hasattr(settings, 'WideDynamicRange'):
                        print(f"    ✓ Wide Dynamic Range: {settings.WideDynamicRange}")
                    else:
                        print(f"    ✗ Wide Dynamic Range: Not available")
                except:
                    print(f"    ✗ Wide Dynamic Range: Not accessible")

        except Exception as e:
            print(f"  ✗ Imaging Service Error: {e}")

        # =====================================================================
        # Device I/O Service (for LED, relays, etc.)
        # =====================================================================
        print("\n[5] Device I/O Service (LED Control)")
        print("-" * 80)
        try:
            deviceio = camera.create_deviceio_service()

            # Get relay outputs
            try:
                relay_outputs = deviceio.GetRelayOutputs()
                print(f"  Relay Outputs: {len(relay_outputs)}")
                for i, relay in enumerate(relay_outputs):
                    relay_dict = serialize_object(relay)
                    print(f"    Relay {i+1}: {json.dumps(relay_dict, indent=6)}")
            except Exception as e:
                print(f"  ✗ GetRelayOutputs: {e}")

            # Get digital inputs
            try:
                digital_inputs = deviceio.GetDigitalInputs()
                print(f"  Digital Inputs: {len(digital_inputs)}")
            except Exception as e:
                print(f"  ✗ GetDigitalInputs: {e}")

        except Exception as e:
            print(f"  ✗ Device I/O Service not available: {e}")

        # =====================================================================
        # Analytics Service
        # =====================================================================
        print("\n[6] Analytics Service")
        print("-" * 80)
        try:
            analytics = camera.create_analytics_service()
            print("  ✓ Analytics service available")

            # Get supported analytics modules
            try:
                modules = analytics.GetSupportedAnalyticsModules()
                print(f"  Supported Modules: {serialize_object(modules)}")
            except Exception as e:
                print(f"  ✗ GetSupportedAnalyticsModules: {e}")

        except Exception as e:
            print(f"  ✗ Analytics Service not available: {e}")

        # =====================================================================
        # Events Service
        # =====================================================================
        print("\n[7] Events Service")
        print("-" * 80)
        try:
            events = camera.create_events_service()
            print("  ✓ Events service available")

            # Get event properties
            try:
                properties = events.GetEventProperties()
                props_dict = serialize_object(properties)
                print(f"  Event Properties: {json.dumps(props_dict, indent=4)}")
            except Exception as e:
                print(f"  ✗ GetEventProperties: {e}")

        except Exception as e:
            print(f"  ✗ Events Service not available: {e}")

        # =====================================================================
        # Audio Sources (for Mic Gain)
        # =====================================================================
        print("\n[8] Audio Configuration")
        print("-" * 80)
        try:
            media_service = camera.create_media_service()

            # Get audio sources
            try:
                audio_sources = media_service.GetAudioSources()
                print(f"  Audio Sources: {len(audio_sources)}")
                for i, source in enumerate(audio_sources):
                    source_dict = serialize_object(source)
                    print(f"    Source {i+1}: {json.dumps(source_dict, indent=6)}")
            except Exception as e:
                print(f"  ✗ GetAudioSources: {e}")

            # Get audio encoder configurations
            try:
                audio_configs = media_service.GetAudioEncoderConfigurations()
                print(f"  Audio Encoder Configurations: {len(audio_configs)}")
                for i, config in enumerate(audio_configs):
                    config_dict = serialize_object(config)
                    print(f"    Config {i+1}: {json.dumps(config_dict, indent=6)}")
            except Exception as e:
                print(f"  ✗ GetAudioEncoderConfigurations: {e}")

        except Exception as e:
            print(f"  ✗ Audio Configuration Error: {e}")

        print("\n" + "=" * 80)
        print("PROBE COMPLETE")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Fatal error during probe: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    probe_all_capabilities()
