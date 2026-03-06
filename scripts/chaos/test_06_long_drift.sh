#!/bin/bash
# Test 6/8: Long Drift — System-Monitoring (Threads, RAM, Temp)
# Dauer konfigurierbar via CHAOS_DRIFT_MINUTES (Default: 360 = 6 Stunden)
# Ggf. in Screen/tmux starten
DURATION_MINUTES=${CHAOS_DRIFT_MINUTES:-360}
LOG=~/moloch/logs/chaos_results.txt
echo "=== TEST 6/8: Long Drift ===" >> $LOG
echo "Konfigurierte Dauer: ${DURATION_MINUTES} Minuten" >> $LOG
date >> $LOG
echo "[6/8] Long Drift Monitor laeuft (${DURATION_MINUTES} Min)..."
for ((i=1; i<=DURATION_MINUTES; i++))
do
  echo "--- drift sample $i/$DURATION_MINUTES ---" >> $LOG
  date >> $LOG
  ps -eLf | wc -l >> $LOG
  free -m >> $LOG
  vcgencmd measure_temp >> $LOG
  sleep 60
done
echo "Long drift monitor finished" >> $LOG
echo "[6/8] Long Drift DONE"
