#!/usr/bin/env python3
"""Calibration Engine: Panel-Integration.

1. Bilderbuch-Tab (7. Tab im Notebook)
2. Observer Handler fuer calibration_result + calibration_status
3. IPC Steuerung (Start/Pause/Stop)
"""
import sys

panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    code = f.read()

fixes = 0

# --- FIX 1: Bilderbuch-Tab nach Debug-Tab hinzufuegen ---
old_tabs = """        self._build_tab_debug(self._perc_nb)

        # Perception config laden
        self._load_perception_config_to_gui()"""

new_tabs = """        self._build_tab_debug(self._perc_nb)
        self._build_tab_calibration(self._perc_nb)

        # Perception config laden
        self._load_perception_config_to_gui()"""

if old_tabs in code:
    code = code.replace(old_tabs, new_tabs)
    print('FIX 1: Bilderbuch Tab Registration - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# --- FIX 2: Observer Handler erweitern ---
old_observer = """        elif event == "cloud_status":
            pass  # Cloud status handled via cam_status"""

new_observer = """        elif event == "cloud_status":
            pass  # Cloud status handled via cam_status
        elif event == "calibration_result":
            self.root.after(0, lambda d=data: self._on_calibration_result(d))
        elif event == "calibration_status":
            self.root.after(0, lambda d=data: self._on_calibration_status(d))"""

if old_observer in code:
    code = code.replace(old_observer, new_observer)
    print('FIX 2: Observer Handler - OK')
    fixes += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

# --- FIX 3: Bilderbuch-Tab Methode + Handler (vor _build_tab_face) ---
old_tab_face = """    def _build_tab_face(self, notebook):
        \"\"\"Tab 1: Face-Parameter.\"\"\""""

new_tab_face = """    # =========================================================================
    # Tab 7: Bilderbuch-Kalibrierung
    # =========================================================================

    def _build_tab_calibration(self, notebook):
        \"\"\"Tab 7: Bilderbuch Training.\"\"\"
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Bilderbuch")

        # --- Obere Zeile: Phase + Steuerung ---
        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=5, pady=3)

        # Phase-Auswahl
        phase_frame = ttk.LabelFrame(top, text="Phase")
        phase_frame.pack(side=tk.LEFT, padx=(0, 10))
        self._cal_phase = tk.StringVar(value="emotions")
        tk.Radiobutton(phase_frame, text="Emotionen", variable=self._cal_phase,
                        value="emotions", bg="#1a1a2e", fg="white",
                        selectcolor="#2a2a4e", activebackground="#1a1a2e",
                        activeforeground="white").pack(anchor=tk.W)
        tk.Radiobutton(phase_frame, text="Gesten", variable=self._cal_phase,
                        value="gestures", bg="#1a1a2e", fg="white",
                        selectcolor="#2a2a4e", activebackground="#1a1a2e",
                        activeforeground="white").pack(anchor=tk.W)

        # Steuerung
        ctrl_frame = ttk.LabelFrame(top, text="Steuerung")
        ctrl_frame.pack(side=tk.LEFT, padx=(0, 10))
        self._cal_start_btn = tk.Button(
            ctrl_frame, text="START", bg="#00aa44", fg="white",
            font=("Helvetica", 10, "bold"), width=8,
            command=self._cal_start)
        self._cal_start_btn.pack(pady=1, padx=3)
        self._cal_pause_btn = tk.Button(
            ctrl_frame, text="PAUSE", bg="#aa8800", fg="white",
            font=("Helvetica", 10), width=8, state=tk.DISABLED,
            command=self._cal_pause)
        self._cal_pause_btn.pack(pady=1, padx=3)
        self._cal_stop_btn = tk.Button(
            ctrl_frame, text="STOP", bg="#aa0000", fg="white",
            font=("Helvetica", 10), width=8, state=tk.DISABLED,
            command=self._cal_stop)
        self._cal_stop_btn.pack(pady=1, padx=3)

        # Tempo
        tempo_frame = ttk.LabelFrame(top, text="Tempo")
        tempo_frame.pack(side=tk.LEFT)
        self._cal_speed = tk.IntVar(value=3)
        tk.Label(tempo_frame, text="Bilder/s:", bg="#1a1a2e", fg="#aaaaaa",
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=2)
        tk.Scale(tempo_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                 variable=self._cal_speed, bg="#1a1a2e", fg="white",
                 troughcolor="#2a2a4e", highlightthickness=0,
                 length=80).pack(side=tk.LEFT)

        # --- Fortschritt ---
        prog_frame = ttk.LabelFrame(tab, text="Fortschritt")
        prog_frame.pack(fill=tk.X, padx=5, pady=3)
        self._cal_progress = ttk.Progressbar(prog_frame, mode='determinate', length=300)
        self._cal_progress.pack(fill=tk.X, padx=5, pady=2)
        self._cal_count_label = tk.Label(prog_frame, text="0 / 0 Bilder",
                                          bg="#1a1a2e", fg="#aaaaaa",
                                          font=("Helvetica", 10))
        self._cal_count_label.pack()

        # --- Live-Ergebnis ---
        result_frame = ttk.LabelFrame(tab, text="Live-Ergebnis")
        result_frame.pack(fill=tk.X, padx=5, pady=3)
        self._cal_file_label = tk.Label(result_frame, text="---",
                                         bg="#1a1a2e", fg="white",
                                         font=("Helvetica", 9))
        self._cal_file_label.pack(anchor=tk.W, padx=5)
        self._cal_result_label = tk.Label(result_frame, text="Erkannt: ---",
                                           bg="#1a1a2e", fg="#aaaaaa",
                                           font=("Helvetica", 10, "bold"))
        self._cal_result_label.pack(anchor=tk.W, padx=5)

        # --- Zusammenfassung (Scrollbare Tabelle) ---
        summary_frame = ttk.LabelFrame(tab, text="Zusammenfassung")
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)
        self._cal_summary = tk.Text(summary_frame, height=8, bg="#0a0a14",
                                     fg="#cccccc", font=("Courier", 9),
                                     state=tk.DISABLED, wrap=tk.NONE)
        self._cal_summary.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        # Farb-Tags
        self._cal_summary.tag_configure("good", foreground="#00ff88")
        self._cal_summary.tag_configure("warn", foreground="#ffaa00")
        self._cal_summary.tag_configure("bad", foreground="#ff4444")
        self._cal_summary.tag_configure("header", foreground="#8888ff")

        # Interne Statistik
        self._cal_stats = {}

    def _cal_start(self):
        \"\"\"Kalibrierung starten.\"\"\"
        phase = self._cal_phase.get()
        speed = self._cal_speed.get()
        self._cal_stats = {}
        self._cal_start_btn.config(state=tk.DISABLED)
        self._cal_pause_btn.config(state=tk.NORMAL)
        self._cal_stop_btn.config(state=tk.NORMAL)
        self._cal_progress['value'] = 0
        self._cal_count_label.config(text="Starte...")
        self._cal_result_label.config(text="Erkannt: ---", fg="#aaaaaa")
        # Summary leeren
        self._cal_summary.config(state=tk.NORMAL)
        self._cal_summary.delete("1.0", tk.END)
        self._cal_summary.config(state=tk.DISABLED)
        # IPC
        self._send_cmd({
            "action": "calibration_start",
            "phase": phase,
            "speed": speed,
        })
        self._append_chat(f"[Bilderbuch] {phase.title()} Training gestartet (Tempo {speed}/s)", "system")

    def _cal_pause(self):
        \"\"\"Pausieren/Fortsetzen.\"\"\"
        self._send_cmd({"action": "calibration_pause"})
        current = self._cal_pause_btn.cget("text")
        if current == "PAUSE":
            self._cal_pause_btn.config(text="WEITER")
            self._append_chat("[Bilderbuch] Pausiert", "system")
        else:
            self._cal_pause_btn.config(text="PAUSE")
            self._append_chat("[Bilderbuch] Fortgesetzt", "system")

    def _cal_stop(self):
        \"\"\"Kalibrierung abbrechen.\"\"\"
        self._send_cmd({"action": "calibration_stop"})
        self._cal_start_btn.config(state=tk.NORMAL)
        self._cal_pause_btn.config(state=tk.DISABLED, text="PAUSE")
        self._cal_stop_btn.config(state=tk.DISABLED)
        self._append_chat("[Bilderbuch] Gestoppt", "system")

    def _on_calibration_result(self, data):
        \"\"\"Live-Ergebnis aus CalibrationEngine anzeigen.\"\"\"
        fname = data.get("file", "?")
        category = data.get("category", "?")
        detected = data.get("detected", "---")
        confidence = data.get("confidence", 0)
        correct = data.get("correct", False)
        progress = data.get("progress", (0, 1))

        # Progress Bar
        pct = (progress[0] / progress[1] * 100) if progress[1] > 0 else 0
        self._cal_progress['value'] = pct
        self._cal_count_label.config(
            text=f"{progress[0]:,} / {progress[1]:,} Bilder ({pct:.0f}%)")

        # Live-Ergebnis
        self._cal_file_label.config(text=f"{category}/{fname}")
        color = "#00ff88" if correct else "#ff4444"
        icon = "OK" if correct else "FALSCH"
        self._cal_result_label.config(
            text=f"Erkannt: {detected} ({confidence:.0%}) [{icon}]",
            fg=color)

        # Statistik aktualisieren
        if category not in self._cal_stats:
            self._cal_stats[category] = {"total": 0, "correct": 0, "conf_sum": 0.0}
        s = self._cal_stats[category]
        s["total"] += 1
        if correct:
            s["correct"] += 1
        s["conf_sum"] += confidence

        # Tabelle alle 50 Bilder aktualisieren (Performance)
        if progress[0] % 50 == 0 or progress[0] == progress[1]:
            self._update_cal_summary()

    def _on_calibration_status(self, data):
        \"\"\"Status-Updates (finished, error, etc.).\"\"\"
        status = data.get("status", "")

        if status == "finished":
            self._cal_start_btn.config(state=tk.NORMAL)
            self._cal_pause_btn.config(state=tk.DISABLED, text="PAUSE")
            self._cal_stop_btn.config(state=tk.DISABLED)
            rate = data.get("rate", 0)
            total = data.get("total", 0)
            correct = data.get("correct", 0)
            duration = data.get("duration", 0)
            self._update_cal_summary()
            self._append_chat(
                f"[Bilderbuch] FERTIG: {correct}/{total} ({rate:.1%}) in {duration:.0f}s",
                "system")

        elif status == "error":
            self._cal_start_btn.config(state=tk.NORMAL)
            self._cal_pause_btn.config(state=tk.DISABLED, text="PAUSE")
            self._cal_stop_btn.config(state=tk.DISABLED)
            msg = data.get("message", "Unbekannter Fehler")
            self._append_chat(f"[Bilderbuch] FEHLER: {msg}", "system")

    def _update_cal_summary(self):
        \"\"\"Zusammenfassungs-Tabelle aktualisieren.\"\"\"
        self._cal_summary.config(state=tk.NORMAL)
        self._cal_summary.delete("1.0", tk.END)

        header = f"{'Kategorie':<14} {'Total':>6} {'Rate':>7} {'Conf':>6}\\n"
        self._cal_summary.insert(tk.END, header, "header")
        self._cal_summary.insert(tk.END, "-" * 36 + "\\n", "header")

        total_all = 0
        correct_all = 0

        for cat in sorted(self._cal_stats.keys()):
            s = self._cal_stats[cat]
            total_all += s["total"]
            correct_all += s["correct"]
            rate = s["correct"] / s["total"] if s["total"] > 0 else 0
            avg_conf = s["conf_sum"] / s["total"] if s["total"] > 0 else 0

            tag = "good" if rate >= 0.80 else ("warn" if rate >= 0.60 else "bad")
            line = f"{cat:<14} {s['total']:>6} {rate:>6.0%} {avg_conf:>6.0%}\\n"
            self._cal_summary.insert(tk.END, line, tag)

        if total_all > 0:
            overall = correct_all / total_all
            tag = "good" if overall >= 0.80 else ("warn" if overall >= 0.60 else "bad")
            self._cal_summary.insert(tk.END, "-" * 36 + "\\n", "header")
            line = f"{'GESAMT':<14} {total_all:>6} {overall:>6.0%}\\n"
            self._cal_summary.insert(tk.END, line, tag)

        self._cal_summary.config(state=tk.DISABLED)

    def _build_tab_face(self, notebook):
        \"\"\"Tab 1: Face-Parameter.\"\"\""""

if old_tab_face in code:
    code = code.replace(old_tab_face, new_tab_face)
    print('FIX 3: Bilderbuch Tab + Handler Methoden - OK')
    fixes += 1
else:
    print('FIX 3: ANCHOR NOT FOUND!')

with open(panel, 'w') as f:
    f.write(code)

print(f'\nPanel: {fixes}/3 Fixes.')
if fixes < 3:
    print('PANEL INCOMPLETE!')
    sys.exit(1)

print('\n=== CALIBRATION PANEL KOMPLETT ===')
