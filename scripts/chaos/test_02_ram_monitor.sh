#!/bin/bash
# Test 2/8: RAM Monitor — 30 Minuten alle 10s RAM-Verbrauch loggen
LOG=~/moloch/logs/chaos_results.txt
echo "=== TEST 2/8: RAM Monitor ===" >> $LOG
date >> $LOG
echo "[2/8] RAM Monitor laeuft (30 Min)..."
for i in {1..180}
do
  ps -o rss,comm -C python3 >> $LOG
  sleep 10
done
echo "RAM monitor finished" >> $LOG
echo "[2/8] RAM Monitor DONE"
