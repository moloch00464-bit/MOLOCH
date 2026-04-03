#!/bin/bash
# Test 4/8: NPU Load — 30 Minuten Temperatur + CPU Load loggen
LOG=~/moloch/logs/chaos_results.txt
echo "=== TEST 4/8: NPU Load ===" >> $LOG
date >> $LOG
echo "[4/8] NPU Load Monitor laeuft (30 Min)..."
for i in {1..180}
do
  vcgencmd measure_temp >> $LOG
  top -b -n1 | head -n 10 >> $LOG
  sleep 10
done
echo "NPU load monitor finished" >> $LOG
echo "[4/8] NPU Load DONE"
