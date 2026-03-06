#!/bin/bash
# Test 5/8: Status JSON Corruption — Testet ob Service korrupte/fehlende Status-Datei ueberlebt
# ACHTUNG: Destruktiv! Ueberschreibt und loescht moloch_status.json temporaer
LOG=~/moloch/logs/chaos_results.txt
STATUS=~/moloch/moloch_status.json
echo "=== TEST 5/8: Status JSON Corruption ===" >> $LOG
date >> $LOG
echo "[5/8] Status JSON Corruption laeuft..."

# Backup anlegen
if [ -f "$STATUS" ]; then
  cp "$STATUS" "${STATUS}.chaos_bak"
fi

echo "" > $STATUS
sleep 5
echo "INVALID DATA !!!" > $STATUS
sleep 5
rm -f $STATUS
sleep 5

# Backup wiederherstellen
if [ -f "${STATUS}.chaos_bak" ]; then
  mv "${STATUS}.chaos_bak" "$STATUS"
  echo "Status JSON restored from backup" >> $LOG
fi

echo "Status corruption test complete" >> $LOG
echo "[5/8] Status JSON Corruption DONE"
