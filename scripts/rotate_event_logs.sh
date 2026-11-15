#!/bin/bash
# Event-Log-Rotation fuer MOLOCH EventBus
# Hintergrund: EventBus schreibt ~1 GB/Tag Telemetrie (ptz_track 5Hz etc.)
# nach /mnt/moloch-data/event_logs (Symlink-Ziel von ~/moloch/logs/events).
# Ohne Rotation lief frueher /run (tmpfs) voll -> Login-Loop am 2026-05-11.
#
# Strategie:
#   1. Alle Tagesdateien ausser der heutigen -> gzip (spart ~95%)
#   2. Komprimierte Logs aelter als 14 Tage -> loeschen
#      (character_distiller destilliert taeglich, Rohdaten danach entbehrlich)
#
# Cron: taeglich 05:50 (vor watcher_agent 06:15)

set -u
LOG_DIR="/mnt/moloch-data/event_logs"
HEUTE="events_$(date +%F).jsonl"

[ -d "$LOG_DIR" ] || exit 0

# 1. Gestern + aelter komprimieren (heutige Datei wird live beschrieben — nie anfassen)
for f in "$LOG_DIR"/events_*.jsonl; do
    [ -f "$f" ] || continue
    [ "$(basename "$f")" = "$HEUTE" ] && continue
    gzip "$f" && echo "$(date -Is) rotiert: $(basename "$f").gz"
done

# 2. Retention: .gz aelter als 14 Tage loeschen
find "$LOG_DIR" -name "events_*.jsonl.gz" -mtime +14 -delete

exit 0
