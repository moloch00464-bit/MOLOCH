#!/usr/bin/env python3
"""
Start MOLOCH Autonomous Tracking
=================================

Starts the autonomous camera tracking system.
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.mpo.autonomous_tracker import AutonomousTracker
from core.hardware.camera import get_camera_controller
from context.system_autonomy import get_system_autonomy

print("=" * 80)
print("🤖 STARTING MOLOCH AUTONOMOUS TRACKING")
print("=" * 80)

# Get camera controller
print("\n📹 Step 1: Getting camera controller...")
camera = get_camera_controller()

if not camera.is_connected:
    print("   Connecting to camera...")
    if not camera.connect():
        print("❌ Failed to connect to camera!")
        sys.exit(1)

print(f"✅ Camera connected: {camera.is_connected}")

# Create tracker
print("\n🎯 Step 2: Creating autonomous tracker...")
tracker = AutonomousTracker(camera_controller=camera)
print(f"✅ Tracker created")

# Register with SystemAutonomy
print("\n🔗 Step 3: Registering with SystemAutonomy...")
autonomy = get_system_autonomy()
autonomy.register_tracker(tracker)
print("✅ Tracker registered")

# Start tracker
print("\n🚀 Step 4: Starting tracker thread...")
if tracker.start():
    print("✅ Tracker thread started!")
else:
    print("❌ Failed to start tracker!")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ AUTONOMOUS TRACKING ACTIVE!")
print("=" * 80)
print("\n📊 STATUS:")
print(f"   Tracker running: {tracker._running}")
print(f"   Tracker state: {tracker.state}")
print(f"   Camera connected: {camera.is_connected}")
print("\n👁️  MOLOCH is now watching and tracking!")
print("   Move in front of the camera to test it.")
print("\n⏹️  To stop: Close this script or run stop_autonomous_tracking.py")
print("=" * 80)

# Keep alive
try:
    import time
    while True:
        time.sleep(1)
        if not tracker._running:
            print("\n⚠️  Tracker stopped!")
            break
except KeyboardInterrupt:
    print("\n\n🛑 Stopping tracker...")
    tracker.stop()
    print("✅ Tracker stopped")
