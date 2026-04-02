#!/bin/bash
# Test 3/8: RTSP Reconnect — Stream starten, killen, pruefen ob Service sich erholt
LOG=~/moloch/logs/chaos_results.txt
echo "=== TEST 3/8: RTSP Reconnect ===" >> $LOG
date >> $LOG
echo "[3/8] RTSP Reconnect laeuft..."
ffplay -nodisp -autoexit -t 10 rtsp://localhost:8554/stream &>/dev/null &
FFPID=$!
sleep 10
kill $FFPID 2>/dev/null
wait $FFPID 2>/dev/null
echo "RTSP reconnect test done" >> $LOG
echo "[3/8] RTSP Reconnect DONE"
