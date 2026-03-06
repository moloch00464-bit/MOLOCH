#!/bin/bash
# Test 7/8: Mode Switch — 20x schnelles Umschalten manual/autonomous
LOG=~/moloch/logs/chaos_results.txt
echo "=== TEST 7/8: Mode Switch ===" >> $LOG
date >> $LOG
echo "[7/8] Mode Switch laeuft..."
for i in {1..20}
do
  curl -s -X POST http://localhost:8080/mode/manual >> $LOG
  sleep 1
  curl -s -X POST http://localhost:8080/mode/autonomous >> $LOG
  sleep 1
done
echo "Mode switch test finished" >> $LOG
echo "[7/8] Mode Switch DONE"
