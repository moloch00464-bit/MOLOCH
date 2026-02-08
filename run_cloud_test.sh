#!/bin/bash
# Camera Cloud Bridge Test Launcher
# Runs cloud bridge integration test with terminal output

cd /home/molochzuhause/moloch
python3 scripts/test_cloud_bridge.py --enable-cloud

echo ""
echo "========================================="
echo "Test beendet. Fenster schließen mit ENTER."
echo "========================================="
read
