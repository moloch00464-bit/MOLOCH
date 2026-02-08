#!/bin/bash
# Camera Controller Test Launcher (ohne Cloud)
# Runs camera controller integration test with terminal output

cd /home/molochzuhause/moloch
python3 scripts/test_camera_controller.py --quick

echo ""
echo "========================================="
echo "Test beendet. Fenster schließen mit ENTER."
echo "========================================="
read
