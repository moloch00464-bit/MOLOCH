#!/usr/bin/env python3
"""Patch: Hand-Occlusion Controls im Unified Panel.

Fuegt hinzu:
- Toggle Button (AN/AUS)
- Status-Label ("HAND ERKANNT" / "")
- Parameter-Slider: Timeout (1-10s), Streak (1-10), Recency (0.5-5.0s)
- Status-Update in _update_npu_status() (alle 1s)
"""
import sys

panel_path = "/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py"
with open(panel_path, "r") as f:
    code = f.read()

changes = 0

# PATCH 1: Hand-Occlusion Section nach Pose-Modell-Section einfuegen
old_after_pose = """            model_frame, "Pose", self.pose_var, "pose",
            [("Conf", self.pose_conf_var, 0.1, 0.9),
             ("NMS", self.pose_nms_var, 0.1, 0.9)])

    def _build_model_section(self, parent, title, enabled_var, model_key, sliders):"""

new_after_pose = """            model_frame, "Pose", self.pose_var, "pose",
            [("Conf", self.pose_conf_var, 0.1, 0.9),
             ("NMS", self.pose_nms_var, 0.1, 0.9)])

        # --- Hand-Occlusion Controls ---
        hand_sep = ttk.Separator(model_frame, orient=tk.HORIZONTAL)
        hand_sep.pack(fill=tk.X, pady=(8, 4))

        hand_header = ttk.Frame(model_frame)
        hand_header.pack(fill=tk.X)

        self.hand_var = tk.BooleanVar(value=True)
        hand_cb = tk.Checkbutton(hand_header, text="Hand-Erkennung",
                                 variable=self.hand_var,
                                 bg="#0a0a14", fg="#e0e0e0", selectcolor="#2a2a4e",
                                 activebackground="#1a1a2e", font=("Helvetica", 9),
                                 command=self._on_hand_toggle)
        hand_cb.pack(side=tk.LEFT)

        self.hand_status_label = tk.Label(hand_header, text="", bg="#0a0a14",
                                          fg="#ff4444", font=("Helvetica", 9, "bold"))
        self.hand_status_label.pack(side=tk.RIGHT)

        # Timeout Slider
        hand_timeout_row = ttk.Frame(model_frame)
        hand_timeout_row.pack(fill=tk.X, padx=(15, 0))
        ttk.Label(hand_timeout_row, text="Timeout:", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.hand_timeout_var = tk.DoubleVar(value=5.0)
        self.hand_timeout_lbl = ttk.Label(hand_timeout_row, text="5.0s", width=4,
                                           font=("Helvetica", 11))
        self.hand_timeout_lbl.pack(side=tk.RIGHT)
        ttk.Scale(hand_timeout_row, from_=1.0, to=10.0, variable=self.hand_timeout_var,
                  command=self._on_hand_param_change).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        # Streak Slider
        hand_streak_row = ttk.Frame(model_frame)
        hand_streak_row.pack(fill=tk.X, padx=(15, 0))
        ttk.Label(hand_streak_row, text="Streak:", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.hand_streak_var = tk.DoubleVar(value=3.0)
        self.hand_streak_lbl = ttk.Label(hand_streak_row, text="3", width=4,
                                          font=("Helvetica", 11))
        self.hand_streak_lbl.pack(side=tk.RIGHT)
        ttk.Scale(hand_streak_row, from_=1.0, to=10.0, variable=self.hand_streak_var,
                  command=self._on_hand_param_change).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        # Recency Slider
        hand_recency_row = ttk.Frame(model_frame)
        hand_recency_row.pack(fill=tk.X, padx=(15, 0))
        ttk.Label(hand_recency_row, text="Recency:", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.hand_recency_var = tk.DoubleVar(value=2.0)
        self.hand_recency_lbl = ttk.Label(hand_recency_row, text="2.0s", width=4,
                                           font=("Helvetica", 11))
        self.hand_recency_lbl.pack(side=tk.RIGHT)
        ttk.Scale(hand_recency_row, from_=0.5, to=5.0, variable=self.hand_recency_var,
                  command=self._on_hand_param_change).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=3)

    def _build_model_section(self, parent, title, enabled_var, model_key, sliders):"""

if old_after_pose in code:
    code = code.replace(old_after_pose, new_after_pose, 1)
    changes += 1
    print("PATCH 1: Hand-Occlusion UI Section eingefuegt")
else:
    print("ERROR: Pose-Section Ende nicht gefunden")
    sys.exit(1)

# PATCH 2: Hand-Occlusion Toggle + Param-Change Methoden
# Fuege nach _update_npu_status() ein
old_phase5_header = """    # =========================================================================
    # Phase 5: PTZ + eWeLink
    # ========================================================================="""

new_hand_methods_then_phase5 = """    # =========================================================================
    # Hand-Occlusion Controls
    # =========================================================================

    def _on_hand_toggle(self):
        \"\"\"Toggle Hand-Occlusion Erkennung.\"\"\"
        if not self.service or not self.service._perception:
            return
        enabled = self.hand_var.get()
        pe = self.service._perception
        if enabled:
            # Alle Parameter synchronisieren
            pe._HAND_TIMEOUT = self.hand_timeout_var.get()
            pe._MIN_FACE_STREAK = int(self.hand_streak_var.get())
            pe._FACE_RECENCY = self.hand_recency_var.get()
        else:
            # Occlusion deaktivieren: Streak auf unmoeglich hohen Wert
            pe._MIN_FACE_STREAK = 999999
            pe._hand_occlusion = False
        logger.info(f"[PANEL] Hand-Occlusion: {'AN' if enabled else 'AUS'}")

    def _on_hand_param_change(self, *args):
        \"\"\"Hand-Occlusion Parameter aktualisieren.\"\"\"
        # Labels aktualisieren
        self.hand_timeout_lbl.config(text=f"{self.hand_timeout_var.get():.1f}s")
        self.hand_streak_lbl.config(text=f"{int(self.hand_streak_var.get())}")
        self.hand_recency_lbl.config(text=f"{self.hand_recency_var.get():.1f}s")
        # An PerceptionEngine weiterreichen
        if not self.service or not self.service._perception or not self.hand_var.get():
            return
        pe = self.service._perception
        pe._HAND_TIMEOUT = self.hand_timeout_var.get()
        pe._MIN_FACE_STREAK = int(self.hand_streak_var.get())
        pe._FACE_RECENCY = self.hand_recency_var.get()

    # =========================================================================
    # Phase 5: PTZ + eWeLink
    # ========================================================================="""

if old_phase5_header in code:
    code = code.replace(old_phase5_header, new_hand_methods_then_phase5, 1)
    changes += 1
    print("PATCH 2: Hand-Occlusion Toggle + Param Methoden eingefuegt")
else:
    print("ERROR: Phase 5 Header nicht gefunden")
    sys.exit(1)

# PATCH 3: Hand-Occlusion Status in _update_npu_status() aktualisieren
old_npu_update_end = """        self.root.after(1000, self._update_npu_status)

    # =========================================================================
    # Hand-Occlusion Controls
    # ========================================================================="""

new_npu_update_end = """        # Hand-Occlusion Status aktualisieren
        try:
            if self.service._perception:
                pe_state = self.service._perception.get_state()
                if pe_state.get("hand_occlusion"):
                    self.hand_status_label.config(text="HAND!", fg="#ff4444")
                else:
                    streak = pe_state.get("face_streak", 0)
                    if streak > 0:
                        self.hand_status_label.config(text=f"S:{streak}", fg="#888888")
                    else:
                        self.hand_status_label.config(text="", fg="#888888")
        except Exception:
            pass

        self.root.after(1000, self._update_npu_status)

    # =========================================================================
    # Hand-Occlusion Controls
    # ========================================================================="""

if old_npu_update_end in code:
    code = code.replace(old_npu_update_end, new_npu_update_end, 1)
    changes += 1
    print("PATCH 3: Hand-Status Update in _update_npu_status()")
else:
    print("ERROR: NPU update end nicht gefunden")
    sys.exit(1)

with open(panel_path, "w") as f:
    f.write(code)
print(f"\nPanel patched: {changes} changes")
print(f"\nHand-Occlusion Controls im Unified Panel:")
print(f"  - Checkbox: Hand-Erkennung AN/AUS (default AN)")
print(f"  - Status: 'HAND!' (rot) oder Streak-Counter")
print(f"  - Slider: Timeout (1-10s, default 5s)")
print(f"  - Slider: Streak (1-10 Frames, default 3)")
print(f"  - Slider: Recency (0.5-5.0s, default 2.0s)")
