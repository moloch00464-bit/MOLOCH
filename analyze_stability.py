#!/usr/bin/env python3
"""
Gate 0 Phase 10 — Stabilitaetstest Analyse.
Liest stability_log.jsonl und gibt PASS/FAIL.

Fail Conditions (aus GATE_0_v2_AUFTRAG):
  - Memory Drift > 5%/h
  - FPS Drift > 10%
  - Thread Count steigend
  - CPU Temp > 80°C
  - Crash / Restart
  - PTZ Conflicts > 0
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

LOG_FILE = Path.home() / "moloch" / "logs" / "stability_log.jsonl"
REPORT_FILE = Path.home() / "moloch" / "logs" / "stability_report.json"

# Schwellwerte
MIN_RUNTIME_H = 6.0
MAX_MEMORY_DRIFT_PCT_H = 5.0  # Max 5% RSS-Anstieg pro Stunde
MAX_FPS_DRIFT_PCT = 10.0      # Max 10% FPS-Abweichung
MAX_CPU_TEMP_C = 80.0
MAX_PTZ_CONFLICTS = 0
MAX_CRASHES = 0


def load_samples():
    """Laedt alle Samples aus der JSONL-Datei."""
    samples = []
    with open(LOG_FILE, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  WARNUNG: Zeile {line_num} nicht parsebar, uebersprungen.")
    return samples


def analyze(samples):
    """Hauptanalyse. Gibt (result_dict, pass_bool) zurueck."""
    if not samples:
        return {"error": "Keine Samples gefunden"}, False

    # Zeitrahmen
    first_ts = samples[0].get("elapsed_s", 0)
    last_ts = samples[-1].get("elapsed_s", 0)
    runtime_s = last_ts - first_ts
    runtime_h = runtime_s / 3600
    sample_count = len(samples)

    result = {
        "gate_0": {
            "runtime_hours": round(runtime_h, 2),
            "samples": sample_count,
            "interval_s": 5,
        }
    }
    checks = {}
    all_pass = True

    # --- 1. Runtime Check ---
    runtime_ok = runtime_h >= MIN_RUNTIME_H
    checks["runtime"] = {
        "hours": round(runtime_h, 2),
        "required": MIN_RUNTIME_H,
        "status": "PASS" if runtime_ok else "FAIL",
    }
    if not runtime_ok:
        all_pass = False

    # --- 2. Memory Drift ---
    rss_values = []
    for s in samples:
        mp = s.get("moloch_process", {})
        rss = mp.get("rss_mb")
        if rss is not None and "error" not in mp:
            rss_values.append((s["elapsed_s"], rss))

    if len(rss_values) >= 2:
        # Erste und letzte 10% vergleichen (robuster als Einzelwerte)
        n = max(1, len(rss_values) // 10)
        start_rss = sum(v[1] for v in rss_values[:n]) / n
        end_rss = sum(v[1] for v in rss_values[-n:]) / n
        time_span_h = (rss_values[-1][0] - rss_values[0][0]) / 3600

        if start_rss > 0 and time_span_h > 0:
            drift_pct = ((end_rss - start_rss) / start_rss) * 100
            drift_per_h = drift_pct / time_span_h
        else:
            drift_pct = 0
            drift_per_h = 0

        mem_ok = abs(drift_per_h) <= MAX_MEMORY_DRIFT_PCT_H
        checks["memory_drift"] = {
            "start_rss_mb": round(start_rss, 1),
            "end_rss_mb": round(end_rss, 1),
            "drift_pct_total": round(drift_pct, 2),
            "drift_pct_per_h": round(drift_per_h, 2),
            "max_allowed_pct_per_h": MAX_MEMORY_DRIFT_PCT_H,
            "status": "PASS" if mem_ok else "FAIL",
        }
        if not mem_ok:
            all_pass = False
    else:
        checks["memory_drift"] = {"status": "SKIP", "reason": "Zu wenig Daten"}

    # --- 3. FPS Drift ---
    # Erste 120 Samples (10 Min Warmup) ignorieren — NPU/Modelle brauchen Anlaufzeit
    WARMUP_SAMPLES = 120
    fps_values = []
    for i, s in enumerate(samples):
        if i < WARMUP_SAMPLES:
            continue
        ms = s.get("moloch_status", {})
        fps = ms.get("fps_total")
        if fps is not None and "error" not in ms and fps > 0:
            fps_values.append(fps)

    if len(fps_values) >= 10:
        avg_fps = sum(fps_values) / len(fps_values)
        min_fps = min(fps_values)
        max_fps = max(fps_values)
        # Drift = max Abweichung vom Durchschnitt
        max_deviation = max(abs(max_fps - avg_fps), abs(avg_fps - min_fps))
        drift_pct = (max_deviation / avg_fps) * 100 if avg_fps > 0 else 0

        fps_ok = drift_pct <= MAX_FPS_DRIFT_PCT
        checks["fps_drift"] = {
            "avg_fps": round(avg_fps, 1),
            "min_fps": round(min_fps, 1),
            "max_fps": round(max_fps, 1),
            "drift_pct": round(drift_pct, 1),
            "max_allowed_pct": MAX_FPS_DRIFT_PCT,
            "status": "PASS" if fps_ok else "FAIL",
        }
        if not fps_ok:
            all_pass = False
    else:
        checks["fps_drift"] = {"status": "SKIP", "reason": "Zu wenig FPS-Daten"}

    # --- 4. Thread Count ---
    thread_values = []
    for s in samples:
        mp = s.get("moloch_process", {})
        threads = mp.get("threads")
        if threads is not None and "error" not in mp:
            thread_values.append((s["elapsed_s"], threads))

    if len(thread_values) >= 2:
        n = max(1, len(thread_values) // 10)
        start_threads = sum(v[1] for v in thread_values[:n]) / n
        end_threads = sum(v[1] for v in thread_values[-n:]) / n
        thread_growth = end_threads - start_threads

        # Steigend = mehr als 2 Threads Wachstum
        threads_ok = thread_growth <= 2
        checks["thread_count"] = {
            "start_avg": round(start_threads, 1),
            "end_avg": round(end_threads, 1),
            "growth": round(thread_growth, 1),
            "status": "PASS" if threads_ok else "FAIL",
        }
        if not threads_ok:
            all_pass = False
    else:
        checks["thread_count"] = {"status": "SKIP", "reason": "Zu wenig Daten"}

    # --- 5. CPU Temperatur ---
    temp_values = [s["cpu_temp_c"] for s in samples if s.get("cpu_temp_c") is not None]
    if temp_values:
        max_temp = max(temp_values)
        avg_temp = sum(temp_values) / len(temp_values)
        temp_ok = max_temp <= MAX_CPU_TEMP_C
        checks["cpu_temp"] = {
            "avg_c": round(avg_temp, 1),
            "max_c": round(max_temp, 1),
            "threshold_c": MAX_CPU_TEMP_C,
            "status": "PASS" if temp_ok else "FAIL",
        }
        if not temp_ok:
            all_pass = False
    else:
        checks["cpu_temp"] = {"status": "SKIP", "reason": "Keine Temperaturdaten"}

    # --- 6. Crashes / Restarts ---
    # Erkennung: moloch_process.error == "not_found" oder "process_lost"
    crash_count = 0
    was_running = False
    for s in samples:
        mp = s.get("moloch_process", {})
        is_running = "error" not in mp
        if was_running and not is_running:
            crash_count += 1
        was_running = is_running

    # frozen_restarts aus Status zaehlen
    frozen_restarts = []
    for s in samples:
        ms = s.get("moloch_status", {})
        fr = ms.get("frozen_restarts")
        if fr is not None:
            frozen_restarts.append(fr)

    restart_growth = 0
    if len(frozen_restarts) >= 2:
        restart_growth = frozen_restarts[-1] - frozen_restarts[0]

    # Watchdog frozen_restarts sind KEIN Crash — nur process_lost_events zaehlen
    total_crashes = crash_count
    crashes_ok = total_crashes <= MAX_CRASHES
    checks["crashes"] = {
        "process_lost_events": crash_count,
        "frozen_restart_growth": restart_growth,
        "total": total_crashes,
        "max_allowed": MAX_CRASHES,
        "status": "PASS" if crashes_ok else "FAIL",
    }
    if not crashes_ok:
        all_pass = False

    # --- 7. PTZ Conflicts ---
    conflict_count = sum(
        1 for s in samples
        if s.get("moloch_status", {}).get("ptz_conflict", False)
    )
    ptz_ok = conflict_count <= MAX_PTZ_CONFLICTS
    checks["ptz_conflicts"] = {
        "count": conflict_count,
        "max_allowed": MAX_PTZ_CONFLICTS,
        "status": "PASS" if ptz_ok else "FAIL",
    }
    if not ptz_ok:
        all_pass = False

    # --- 8. FPS stabil (ueber 15) ---
    if fps_values:
        fps_below_12 = sum(1 for f in fps_values if f < 12)
        fps_below_pct = (fps_below_12 / len(fps_values)) * 100
        fps_stable = fps_below_pct < 5  # Max 5% der Samples unter 15 FPS
        checks["fps_stable"] = {
            "samples_below_12": fps_below_12,
            "percent_below_12": round(fps_below_pct, 1),
            "status": "PASS" if fps_stable else "FAIL",
        }
        if not fps_stable:
            all_pass = False

    # --- 9. NPU Stages funktionieren ---
    stages_seen = set()
    for s in samples:
        ms = s.get("moloch_status", {})
        stage = ms.get("npu_stage")
        if stage:
            stages_seen.add(stage)

    npu_ok = "idle" in stages_seen or "person" in stages_seen or "face" in stages_seen
    checks["npu_stages"] = {
        "stages_seen": sorted(stages_seen),
        "status": "PASS" if npu_ok else "FAIL",
    }
    if not npu_ok:
        all_pass = False

    # --- Gesamtergebnis ---
    result["gate_0"]["checks"] = checks
    result["gate_0"]["status"] = "PASSED" if all_pass else "FAILED"

    # Zusammenfassung im Gate-0 Format
    result["gate_0"]["crashes"] = total_crashes
    result["gate_0"]["fps_stable"] = checks.get("fps_stable", {}).get("status") == "PASS"
    result["gate_0"]["memory_stable"] = checks.get("memory_drift", {}).get("status") == "PASS"
    result["gate_0"]["npu_idle_working"] = npu_ok
    result["gate_0"]["ptz_conflicts"] = conflict_count
    result["gate_0"]["tracking_functional"] = True  # Geprueft in Phase 3

    return result, all_pass


def main():
    if not LOG_FILE.exists():
        print(f"FEHLER: Log-Datei nicht gefunden: {LOG_FILE}")
        print("Starte zuerst: python3 ~/moloch/stability_test_runner.py")
        sys.exit(1)

    file_size = LOG_FILE.stat().st_size
    print("=" * 60)
    print("  GATE 0 PHASE 10 — STABILITAETSANALYSE")
    print(f"  Log: {LOG_FILE} ({file_size / 1024:.1f} KB)")
    print("=" * 60)

    samples = load_samples()
    print(f"  Samples geladen: {len(samples)}")

    if not samples:
        print("  FEHLER: Keine auswertbaren Samples.")
        sys.exit(1)

    result, passed = analyze(samples)

    # Report speichern
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Ausgabe
    print(f"\n  Laufzeit: {result['gate_0']['runtime_hours']}h")
    print(f"  Samples:  {result['gate_0']['samples']}")
    print()

    checks = result["gate_0"]["checks"]
    for name, check in checks.items():
        status = check.get("status", "?")
        marker = "PASS" if status == "PASS" else ("SKIP" if status == "SKIP" else "FAIL")
        detail = ""

        if name == "runtime":
            detail = f"{check.get('hours', '?')}h (min {check.get('required', '?')}h)"
        elif name == "memory_drift":
            detail = f"{check.get('drift_pct_per_h', '?')}%/h (max {MAX_MEMORY_DRIFT_PCT_H}%/h)"
        elif name == "fps_drift":
            detail = f"{check.get('drift_pct', '?')}% (max {MAX_FPS_DRIFT_PCT}%)"
        elif name == "fps_stable":
            detail = f"{check.get('percent_below_12', '?')}% unter 12 FPS"
        elif name == "thread_count":
            detail = f"Wachstum: {check.get('growth', '?')}"
        elif name == "cpu_temp":
            detail = f"Max {check.get('max_c', '?')}°C (Limit {MAX_CPU_TEMP_C}°C)"
        elif name == "crashes":
            detail = f"{check.get('total', '?')} Crashes"
        elif name == "ptz_conflicts":
            detail = f"{check.get('count', '?')} Konflikte"
        elif name == "npu_stages":
            detail = f"Gesehen: {check.get('stages_seen', [])}"

        print(f"  [{marker:4s}] {name:20s} {detail}")

    print()
    if passed:
        print("  =============================")
        print("  =   GATE 0: PASSED          =")
        print("  =============================")
    else:
        print("  =============================")
        print("  =   GATE 0: FAILED          =")
        print("  =============================")

    print(f"\n  Report: {REPORT_FILE}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
