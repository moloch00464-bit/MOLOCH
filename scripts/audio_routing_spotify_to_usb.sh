#!/bin/bash
# Spotify-Audio auf USB-Sink umleiten (Sprint-2-Fix-5 2026-05-10)
#
# Use-Case: Markus' USB-Combi-Device (Mic + Speaker) — wenn Spotify
# spielt, soll Default-Sink auf USB sein, damit Mic-Loopback den Stream
# einfangen kann (Beat-Detection auf USB-Mic 48kHz).
#
# Aufruf via mic_mode_controller bei music.playing event, falls
# settings.audio.spotify_to_usb_speaker=true.
#
# Fail-soft: wenn USB-Sink nicht da, kein-op + Hinweis.

set -e

if ! command -v pactl >/dev/null 2>&1; then
    echo "[audio] pactl nicht installiert — skip"
    exit 0
fi

USB_SINK=$(pactl list short sinks 2>/dev/null | grep -iE "usb|combi" | head -1 | awk '{print $2}')

if [ -z "$USB_SINK" ]; then
    echo "[audio] kein USB-Sink gefunden — Spotify bleibt auf Default (HDMI)"
    pactl list short sinks | head -5
    exit 0
fi

CURRENT_DEFAULT=$(pactl get-default-sink 2>/dev/null || echo "")
if [ "$CURRENT_DEFAULT" = "$USB_SINK" ]; then
    echo "[audio] Default-Sink schon $USB_SINK"
    exit 0
fi

if pactl set-default-sink "$USB_SINK" 2>&1; then
    echo "[audio] Spotify-Default umgeleitet: $CURRENT_DEFAULT -> $USB_SINK"
else
    echo "[audio] set-default-sink failed — bleibt auf $CURRENT_DEFAULT"
    exit 1
fi
