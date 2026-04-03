#!/bin/bash
# Test 8/8: Face DB Reload — Leert Face-DB temporaer, stellt wieder her
# ACHTUNG: Destruktiv! Face Recognition faellt fuer ~10s aus
LOG=~/moloch/logs/chaos_results.txt
DB=~/moloch/data/face_embeddings.json
echo "=== TEST 8/8: Face DB Reload ===" >> $LOG
date >> $LOG
echo "[8/8] Face DB Reload laeuft..."

# Backup anlegen
cp "$DB" "${DB}.chaos_bak"
echo "{}" > $DB
sleep 10
# Wiederherstellen
mv "${DB}.chaos_bak" "$DB"

echo "Face DB reload test complete" >> $LOG
echo "[8/8] Face DB Reload DONE"
