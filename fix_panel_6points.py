#!/usr/bin/env python3
"""Fix: 6-Punkt Panel-Verbesserungen.

1. OFFLINE Status klarstellen (was ist offline?)
2. Leeren schwarzen Platz fuellen (System-Status)
3. Chat-Fenster kleiner
4. SCRFD/ArcFace Checkboxen periodisch syncen
5. API Indicator gross, oben-rechts, gruen/rot
6. Snapshot Button immer gruen, unabhaengig von ArcFace
"""
import sys

panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    code = f.read()

fixes = 0

# ============================================================
# FIX 1: OFFLINE -> "KEIN SERVICE" (klarstellen was offline ist)
# ============================================================
old_offline = '''        self.mode_label = tk.Label(bar, text="OFFLINE", bg="#0a0a14", fg="#888888",
                                   font=("Helvetica", 14, "bold"))
        self.mode_label.pack(side=tk.LEFT, padx=(0, 2))'''

new_offline = '''        self.mode_label = tk.Label(bar, text="KEIN SERVICE", bg="#0a0a14", fg="#ff4444",
                                   font=("Helvetica", 14, "bold"))
        self.mode_label.pack(side=tk.LEFT, padx=(0, 2))'''

if old_offline in code:
    code = code.replace(old_offline, new_offline)
    print('FIX 1a: OFFLINE -> KEIN SERVICE - OK')
    fixes += 1
else:
    print('FIX 1a: ANCHOR NOT FOUND!')

# FIX 1b: _update_cam_status - bessere Labels
old_cam_status = '''    def _update_cam_status(self, data):
        """Update status bar from cam_status event."""
        mode = data.get("mode", "offline")
        ctrl = data.get("ctrl_text", "")
        smart = data.get("smart", "--")
        ptz = data.get("ptz", "--")

        colors = {"moloch": "#00ff88", "tentakel": "#00d4ff",
                  "manual": "#aaaaaa", "offline": "#ff4444"}
        self.mode_label.config(text=ctrl or mode.upper(),
                               fg=colors.get(mode, "#888888"))'''

new_cam_status = '''    def _update_cam_status(self, data):
        """Update status bar from cam_status event."""
        mode = data.get("mode", "offline")
        ctrl = data.get("ctrl_text", "")
        smart = data.get("smart", "--")
        ptz = data.get("ptz", "--")

        # Klarere Status-Labels
        mode_labels = {
            "moloch": ctrl or "MOLOCH AKTIV",
            "tentakel": "TENTAKEL SCANNT",
            "manual": "MANUELL",
            "offline": "KAMERA OFFLINE",
        }
        colors = {"moloch": "#00ff88", "tentakel": "#00d4ff",
                  "manual": "#aaaaaa", "offline": "#ff4444"}
        display = mode_labels.get(mode, mode.upper())
        self.mode_label.config(text=display, fg=colors.get(mode, "#888888"))'''

if old_cam_status in code:
    code = code.replace(old_cam_status, new_cam_status)
    print('FIX 1b: Bessere Mode-Labels - OK')
    fixes += 1
else:
    print('FIX 1b: ANCHOR NOT FOUND!')

# ============================================================
# FIX 2: System-Status unter Perception Tabs
# ============================================================
# Nach dem Perception auto-save timer, System-Status Section einfuegen
old_autosave = '''        # Auto-Save Timer starten
        self.root.after(5000, self._perc_auto_save)'''

new_autosave = '''        # Auto-Save Timer starten
        self.root.after(5000, self._perc_auto_save)

        # --- System Status (unter Perception Tabs) ---
        sys_sep = ttk.Separator(model_frame, orient=tk.HORIZONTAL)
        sys_sep.pack(fill=tk.X, pady=(2, 1))
        sys_frame = tk.Frame(model_frame, bg="#0a0a14")
        sys_frame.pack(fill=tk.X)
        self._sys_cpu_label = tk.Label(sys_frame, text="CPU: --", bg="#0a0a14",
                                        fg="#888888", font=("Courier", 9), anchor=tk.W)
        self._sys_cpu_label.pack(fill=tk.X)
        self._sys_ram_label = tk.Label(sys_frame, text="RAM: --", bg="#0a0a14",
                                        fg="#888888", font=("Courier", 9), anchor=tk.W)
        self._sys_ram_label.pack(fill=tk.X)
        self._sys_tension_label = tk.Label(sys_frame, text="Tension: --", bg="#0a0a14",
                                            fg="#888888", font=("Courier", 9), anchor=tk.W)
        self._sys_tension_label.pack(fill=tk.X)
        self._sys_mode_label = tk.Label(sys_frame, text="Modus: --", bg="#0a0a14",
                                         fg="#888888", font=("Courier", 9), anchor=tk.W)
        self._sys_mode_label.pack(fill=tk.X)
        self._sys_npu_label = tk.Label(sys_frame, text="NPU: --", bg="#0a0a14",
                                        fg="#888888", font=("Courier", 9), anchor=tk.W)
        self._sys_npu_label.pack(fill=tk.X)
        self.root.after(2000, self._update_system_status)'''

if old_autosave in code:
    code = code.replace(old_autosave, new_autosave)
    print('FIX 2a: System Status Labels - OK')
    fixes += 1
else:
    print('FIX 2a: ANCHOR NOT FOUND!')

# FIX 2b: _update_system_status Methode einfuegen (vor _update_cam_status)
old_update_cam = '''    def _update_cam_status(self, data):'''

new_update_system = '''    def _update_system_status(self):
        """System-Status aktualisieren (CPU, RAM, Tension, Modus)."""
        if not self.running:
            return
        try:
            # CPU Temperatur
            try:
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    temp_c = int(f.read().strip()) / 1000
                color = "#00ff88" if temp_c < 60 else "#ffaa00" if temp_c < 75 else "#ff4444"
                self._sys_cpu_label.config(text=f"CPU: {temp_c:.0f} C", fg=color)
            except Exception:
                self._sys_cpu_label.config(text="CPU: n/a")

            # RAM
            try:
                with open("/proc/meminfo") as f:
                    lines = f.readlines()
                total = int(lines[0].split()[1]) // 1024
                avail = int(lines[2].split()[1]) // 1024
                used = total - avail
                pct = used * 100 // total if total > 0 else 0
                color = "#00ff88" if pct < 60 else "#ffaa00" if pct < 80 else "#ff4444"
                self._sys_ram_label.config(text=f"RAM: {used}/{total}MB ({pct}%)", fg=color)
            except Exception:
                self._sys_ram_label.config(text="RAM: n/a")

            # NPU Modelle
            if self.service:
                try:
                    with self.service._fps_lock:
                        active = list(self.service._active_ctx.keys())
                    npu_text = ", ".join(active) if active else "keine"
                    self._sys_npu_label.config(
                        text=f"NPU: {npu_text}",
                        fg="#00ff88" if active else "#888888")
                except Exception:
                    pass

            # Tension + Modus (aus face_state.json oder Personality)
            try:
                if os.path.exists(FACE_STATE_PATH):
                    with open(FACE_STATE_PATH) as f:
                        fs = json.load(f)
                    emotion = fs.get("emotion", "")
                    if emotion:
                        self._sys_tension_label.config(
                            text=f"Emotion: {emotion}", fg="#00d4ff")
                    mode = fs.get("personality_mode", "")
                    if mode:
                        mcolor = "#00ff88" if mode == "guardian" else "#ff4444"
                        self._sys_mode_label.config(
                            text=f"Modus: {mode.title()}", fg=mcolor)
            except Exception:
                pass

        except Exception:
            pass

        self.root.after(2000, self._update_system_status)

    def _update_cam_status(self, data):'''

if old_update_cam in code:
    code = code.replace(old_update_cam, new_update_system, 1)
    print('FIX 2b: _update_system_status Methode - OK')
    fixes += 1
else:
    print('FIX 2b: ANCHOR NOT FOUND!')

# ============================================================
# FIX 3: Chat-Fenster kleiner (height 6 -> 4)
# ============================================================
old_chat_height = '''            wrap="word", height=6, state="disabled",'''
new_chat_height = '''            wrap="word", height=4, state="disabled",'''

if old_chat_height in code:
    code = code.replace(old_chat_height, new_chat_height)
    print('FIX 3: Chat height 6 -> 4 - OK')
    fixes += 1
else:
    # Versuche andere Heights
    for h in range(3, 15):
        alt = f'wrap="word", height={h}, state="disabled",'
        if alt in code:
            code = code.replace(alt, 'wrap="word", height=4, state="disabled",')
            print(f'FIX 3: Chat height {h} -> 4 - OK')
            fixes += 1
            break
    else:
        print('FIX 3: ANCHOR NOT FOUND!')

# ============================================================
# FIX 4: Periodisches Model-Checkbox Sync (Belt-and-Suspenders)
# ============================================================
# In _update_fps (laeuft alle 500ms), Checkboxen nachjustieren
old_fps_end = '''        self.root.after(500, self._update_fps)

    def _update_npu_status(self):'''

new_fps_end = '''        # Periodisch Checkboxen synchronisieren (Belt-and-Suspenders)
        try:
            if not self._syncing:
                _map = [
                    (self.scrfd_var, self.service.scrfd_active),
                    (self.arcface_var, self.service.arcface_active),
                    (self.yolo_var, self.service.yolo_active),
                    (self.pose_var, self.service.pose_active),
                    (self.hand_lm_var, getattr(self.service, 'hand_active', False)),
                ]
                for var, active in _map:
                    if var.get() != active:
                        var.set(active)
        except Exception:
            pass

        self.root.after(500, self._update_fps)

    def _update_npu_status(self):'''

if old_fps_end in code:
    code = code.replace(old_fps_end, new_fps_end)
    print('FIX 4: Periodisches Checkbox Sync - OK')
    fixes += 1
else:
    print('FIX 4: ANCHOR NOT FOUND!')

# ============================================================
# FIX 5: API Indicator gross, oben-rechts, gruen/rot
# ============================================================
# 5a: Status-Bar Label hinzufuegen (nach status_label)
old_status_right = '''        self.status_label = tk.Label(bar, text="Initialisierung...", bg="#0a0a14",
                                      fg="#00ff88", font=("Helvetica", 12))
        self.status_label.pack(side=tk.RIGHT)'''

new_status_right = '''        # API Status Indikator (gross, rechts)
        self._api_dot = tk.Label(bar, text=" API ", bg="#ff4444", fg="white",
                                  font=("Helvetica", 11, "bold"),
                                  relief=tk.RAISED, padx=4, pady=1)
        self._api_dot.pack(side=tk.RIGHT, padx=(2, 0))

        self.status_label = tk.Label(bar, text="Initialisierung...", bg="#0a0a14",
                                      fg="#00ff88", font=("Helvetica", 12))
        self.status_label.pack(side=tk.RIGHT)'''

if old_status_right in code:
    code = code.replace(old_status_right, new_status_right)
    print('FIX 5a: API Dot in Status Bar - OK')
    fixes += 1
else:
    print('FIX 5a: ANCHOR NOT FOUND!')

# 5b: _update_api_indicator erweitern - auch Dot aktualisieren
old_api_update = '''    def _update_api_indicator(self):
        """API Token-Zaehler im Chat anzeigen."""
        if hasattr(self, '_api_indicator'):
            cost_approx = (self._api_tokens_in * 3 + self._api_tokens_out * 15) / 1_000_000
            self._api_indicator.config(
                text=f"API: {self._api_calls} Calls | "
                     f"{self._api_tokens_in + self._api_tokens_out:,} Tokens | "
                     f"~${cost_approx:.3f}")'''

new_api_update = '''    def _update_api_indicator(self):
        """API Token-Zaehler + Status Dot aktualisieren."""
        if hasattr(self, '_api_indicator'):
            cost_approx = (self._api_tokens_in * 3 + self._api_tokens_out * 15) / 1_000_000
            self._api_indicator.config(
                text=f"API: {self._api_calls} Calls | "
                     f"{self._api_tokens_in + self._api_tokens_out:,} Tokens | "
                     f"~${cost_approx:.3f}",
                fg="#00ff88")
        # Status Dot: gruen wenn API Key vorhanden und Claude Client aktiv
        if hasattr(self, '_api_dot'):
            if self.claude_client is not None:
                self._api_dot.config(bg="#00aa44", text=" API ")
            else:
                self._api_dot.config(bg="#ff4444", text=" API ")'''

if old_api_update in code:
    code = code.replace(old_api_update, new_api_update)
    print('FIX 5b: API Indicator erweitert - OK')
    fixes += 1
else:
    print('FIX 5b: ANCHOR NOT FOUND!')

# 5c: API Dot initial auf gruen setzen wenn Claude Client geladen wird
# Suche nach Claude Client init
old_claude_init = '''            self.claude_client = anthropic.Anthropic()'''
if old_claude_init in code:
    new_claude_init = '''            self.claude_client = anthropic.Anthropic()
            self.root.after(0, self._update_api_indicator)'''
    code = code.replace(old_claude_init, new_claude_init, 1)
    print('FIX 5c: API Dot Update nach Claude Init - OK')
    fixes += 1
else:
    # Versuche alternative
    old_claude2 = 'self.claude_client = anthropic.Anthropic('
    if old_claude2 in code:
        print('FIX 5c: Claude Init gefunden aber anderes Format - manuell pruefen')
    else:
        print('FIX 5c: ANCHOR NOT FOUND (Claude init)')

# ============================================================
# FIX 6: Snapshot Button immer gruen + Referenz speichern
# ============================================================
old_snap_btn = '''        tk.Button(ar_row, text="SNAP", bg="#00aa44", fg="white", width=6,
                  font=("Helvetica", 11, "bold"),
                  command=self._take_snapshot).pack(side=tk.LEFT, padx=1)'''

new_snap_btn = '''        self._snap_btn = tk.Button(ar_row, text="SNAP", bg="#00aa44", fg="white",
                                   width=6, font=("Helvetica", 11, "bold"),
                                   command=self._take_snapshot)
        self._snap_btn.pack(side=tk.LEFT, padx=1)'''

if old_snap_btn in code:
    code = code.replace(old_snap_btn, new_snap_btn)
    print('FIX 6a: Snapshot Button Referenz - OK')
    fixes += 1
else:
    print('FIX 6a: ANCHOR NOT FOUND!')

# FIX 6b: _take_snapshot - nur Bild speichern (Enrollment nur wenn ArcFace aktiv)
old_snap_enroll = '''            # Enrollment via IPC an Service delegieren
            result_path = "/tmp/moloch_snapshot_result.json"
            # Altes Ergebnis loeschen
            try:
                if os.path.exists(result_path):
                    os.remove(result_path)
            except Exception:
                pass

            self.root.after(0, lambda: self._append_chat(
                "[Snapshot] Enrollment laeuft (Service)...", "system"))
            self._send_cmd({"action": "snapshot_enroll"})

            # Auf Ergebnis warten (max 10s)
            for _ in range(20):
                time.sleep(0.5)
                try:
                    if os.path.exists(result_path):
                        with open(result_path, "r") as f:
                            result = json.load(f)
                        if result.get("success"):
                            msg = f"[Snapshot] {result['message']}"
                        else:
                            msg = f"[Snapshot] Fehler: {result['message']}"
                        self.root.after(0, lambda m=msg: self._append_chat(m, "system"))
                        return
                except Exception:
                    pass

            self.root.after(0, lambda: self._append_chat(
                "[Snapshot] Enrollment Timeout (keine Antwort vom Service)", "system"))'''

new_snap_enroll = '''            # Enrollment NUR wenn ArcFace aktiv (optional, Snapshot geht immer)
            if self.service and getattr(self.service, 'arcface_active', False):
                result_path = "/tmp/moloch_snapshot_result.json"
                try:
                    if os.path.exists(result_path):
                        os.remove(result_path)
                except Exception:
                    pass

                self.root.after(0, lambda: self._append_chat(
                    "[Snapshot] Enrollment laeuft (ArcFace aktiv)...", "system"))
                self._send_cmd({"action": "snapshot_enroll"})

                for _ in range(20):
                    time.sleep(0.5)
                    try:
                        if os.path.exists(result_path):
                            with open(result_path, "r") as f:
                                result = json.load(f)
                            if result.get("success"):
                                msg = f"[Snapshot] {result['message']}"
                            else:
                                msg = f"[Snapshot] Fehler: {result['message']}"
                            self.root.after(0, lambda m=msg: self._append_chat(m, "system"))
                            return
                    except Exception:
                        pass

                self.root.after(0, lambda: self._append_chat(
                    "[Snapshot] Enrollment Timeout", "system"))
            else:
                self.root.after(0, lambda: self._append_chat(
                    "[Snapshot] Bild gespeichert (kein Enrollment - ArcFace nicht aktiv)", "system"))'''

if old_snap_enroll in code:
    code = code.replace(old_snap_enroll, new_snap_enroll)
    print('FIX 6b: Snapshot ohne ArcFace-Abhaengigkeit - OK')
    fixes += 1
else:
    print('FIX 6b: ANCHOR NOT FOUND!')

# ============================================================
# Write + Verify
# ============================================================
with open(panel, 'w') as f:
    f.write(code)

compile(open(panel).read(), panel, 'exec')
print(f'\n{fixes} Fixes angewendet. Syntax OK.')

if fixes < 8:
    print(f'WARNUNG: Nur {fixes}/12 Fixes!')

print('\n=== 6-PUNKT PANEL-VERBESSERUNGEN ===')
print('1. OFFLINE -> KEIN SERVICE / KAMERA OFFLINE')
print('2. System-Status unter Perception Tabs')
print('3. Chat height 6 -> 4')
print('4. Periodisches Checkbox Sync')
print('5. API Indicator gross + gruen/rot Dot')
print('6. Snapshot immer gruen, Enrollment optional')
