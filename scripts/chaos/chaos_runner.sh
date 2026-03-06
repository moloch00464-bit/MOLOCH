#!/bin/bash
# M.O.L.O.C.H. Chaos Engineering Runner
# Startet alle 8 Tests sequenziell
LOG=~/moloch/logs/chaos_results.txt
mkdir -p ~/moloch/logs
echo "=== M.O.L.O.C.H CHAOS TEST START ===" >> $LOG
date >> $LOG
echo ""
echo "CHAOS RUNNER gestartet. Log: $LOG"
echo ""

cd "$(dirname "$0")"

bash test_01_ptz_stress.sh
bash test_02_ram_monitor.sh
bash test_03_rtsp_reconnect.sh
bash test_04_npu_load.sh
bash test_05_status_json_corruption.sh
bash test_06_long_drift.sh
bash test_07_mode_switch.sh
bash test_08_face_db_reload.sh

echo "=== CHAOS TEST COMPLETE ===" >> $LOG
date >> $LOG
echo ""
echo "ALLE TESTS ABGESCHLOSSEN. Ergebnisse in $LOG"
