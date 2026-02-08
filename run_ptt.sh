#!/bin/bash
# PTT Launcher mit Crash-Logging
export DISPLAY=:0
cd /home/molochzuhause/moloch
exec python -m core.gui.push_to_talk >> /tmp/ptt_debug.log 2>&1
