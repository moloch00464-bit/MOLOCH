#!/bin/bash
# Test 1/8: PTZ Stress — 50 Runden schnelle Richtungswechsel
LOG=~/moloch/logs/chaos_results.txt
echo "=== TEST 1/8: PTZ Stress ===" >> $LOG
date >> $LOG
echo "[1/8] PTZ Stress laeuft..."
for i in {1..50}
do
  curl -s -X POST http://localhost:8080/ptz/left &
  curl -s -X POST http://localhost:8080/ptz/right &
  curl -s -X POST http://localhost:8080/ptz/up &
  curl -s -X POST http://localhost:8080/ptz/down &
  sleep 0.2
done
wait
echo "PTZ stress finished" >> $LOG
echo "[1/8] PTZ Stress DONE"
