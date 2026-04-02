#!/bin/bash
# M.O.L.O.C.H. PyGame Avatar-Auge starten
# Liest State aus /dev/shm/moloch_status.json (vom Service geschrieben)
#
# Steuerung:
#   ESC/Q  = Beenden
#   F      = Fullscreen Toggle

cd ~/moloch
exec python3 core/gui/avatar_pygame.py "$@"
