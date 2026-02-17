#!/usr/bin/env python3
"""Perception Config: Panel-Umbau mit 6 Tabs.

_build_model_controls wird umgebaut:
- Oben: Modell-Checkboxen + FPS (wie bisher)
- Unten: ttk.Notebook mit 6 Tabs (Face, Hand, Pose, Global, NPU, Debug)
- Auto-Save alle 5s, dirty-Flag
- Hand-Occlusion Slider wandern in Hand-Tab
"""
import sys

panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    code = f.read()

fixes = 0

# ============================================================
# FIX 1: _build_model_controls komplett ersetzen
# ============================================================
old_build = '''    def _build_model_controls(self, parent):
        """Model checkboxes + threshold sliders on the right side."""
        model_frame = ttk.Frame(parent)
        model_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 0))

        ttk.Label(model_frame, text="MODELLE", style="Header.TLabel").pack(anchor=tk.W)

        # Model variables
        self.scrfd_var = tk.BooleanVar(value=False)
        self.arcface_var = tk.BooleanVar(value=False)
        self.yolo_var = tk.BooleanVar(value=False)
        self.pose_var = tk.BooleanVar(value=False)

        # Threshold variables
        self.scrfd_conf_var = tk.DoubleVar(value=0.40)
        self.scrfd_nms_var = tk.DoubleVar(value=0.40)
        self.arcface_thresh_var = tk.DoubleVar(value=0.60)
        self.yolo_conf_var = tk.DoubleVar(value=0.50)
        self.pose_conf_var = tk.DoubleVar(value=0.50)
        self.pose_nms_var = tk.DoubleVar(value=0.70)

        # SCRFD
        self._scrfd_fps = self._build_model_section(
            model_frame, "SCRFD Face", self.scrfd_var, "scrfd",
            [("Conf", self.scrfd_conf_var, 0.1, 0.9),
             ("NMS", self.scrfd_nms_var, 0.1, 0.9)])

        # ArcFace
        self._arcface_fps = self._build_model_section(
            model_frame, "ArcFace", self.arcface_var, "arcface",
            [("Thresh", self.arcface_thresh_var, 0.3, 0.9)])

        # YOLOv8m
        self._yolov8m_fps = self._build_model_section(
            model_frame, "YOLOv8m", self.yolo_var, "yolov8m",
            [("Conf", self.yolo_conf_var, 0.1, 0.9)])

        # Pose
        self._pose_fps = self._build_model_section(
            model_frame, "Pose", self.pose_var, "pose",
            [("Conf", self.pose_conf_var, 0.1, 0.9),
             ("NMS", self.pose_nms_var, 0.1, 0.9)])

        # Hand Landmark (braucht Pose als Dependency)
        self.hand_lm_var = tk.BooleanVar(value=False)
        self._hand_lm_fps = self._build_model_section(
            model_frame, "Hand LM", self.hand_lm_var, "hand_landmark", [])

        # --- Save Settings ---
        save_sep = ttk.Separator(model_frame, orient=tk.HORIZONTAL)
        save_sep.pack(fill=tk.X, pady=(8, 4))
        self._save_btn = tk.Button(model_frame, text="SAVE SETTINGS",
                                   bg="#00aa44", fg="white",
                                   font=("Helvetica", 10, "bold"),
                                   command=self._save_settings)
        self._save_btn.pack(fill=tk.X, pady=(2, 0))

        # --- Hand-Occlusion (Auto-Erkennung) ---
        hand_sep = ttk.Separator(model_frame, orient=tk.HORIZONTAL)
        hand_sep.pack(fill=tk.X, pady=(8, 4))

        hand_header = ttk.Frame(model_frame)
        hand_header.pack(fill=tk.X)

        self.hand_var = tk.BooleanVar(value=True)
        hand_cb = tk.Checkbutton(hand_header, text="Auto-Occlusion",
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
            side=tk.LEFT, fill=tk.X, expand=True, padx=3)'''

new_build = '''    def _build_model_controls(self, parent):
        """Model checkboxes (oben) + Perception Tabs (unten)."""
        model_frame = ttk.Frame(parent)
        model_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 0))

        ttk.Label(model_frame, text="MODELLE", style="Header.TLabel").pack(anchor=tk.W)

        # Model variables
        self.scrfd_var = tk.BooleanVar(value=False)
        self.arcface_var = tk.BooleanVar(value=False)
        self.yolo_var = tk.BooleanVar(value=False)
        self.pose_var = tk.BooleanVar(value=False)
        self.hand_lm_var = tk.BooleanVar(value=False)

        # Threshold variables (fuer IPC-Kompatibilitaet)
        self.scrfd_conf_var = tk.DoubleVar(value=0.40)
        self.scrfd_nms_var = tk.DoubleVar(value=0.40)
        self.arcface_thresh_var = tk.DoubleVar(value=0.60)
        self.yolo_conf_var = tk.DoubleVar(value=0.50)
        self.pose_conf_var = tk.DoubleVar(value=0.50)
        self.pose_nms_var = tk.DoubleVar(value=0.70)

        # Modell-Checkboxen (kompakt, ohne Slider)
        self._scrfd_fps = self._build_model_section(
            model_frame, "SCRFD Face", self.scrfd_var, "scrfd", [])
        self._arcface_fps = self._build_model_section(
            model_frame, "ArcFace", self.arcface_var, "arcface", [])
        self._yolov8m_fps = self._build_model_section(
            model_frame, "YOLOv8m", self.yolo_var, "yolov8m", [])
        self._pose_fps = self._build_model_section(
            model_frame, "Pose", self.pose_var, "pose", [])
        self._hand_lm_fps = self._build_model_section(
            model_frame, "Hand LM", self.hand_lm_var, "hand_landmark", [])

        # --- Save Settings ---
        save_sep = ttk.Separator(model_frame, orient=tk.HORIZONTAL)
        save_sep.pack(fill=tk.X, pady=(8, 4))
        self._save_btn = tk.Button(model_frame, text="SAVE SETTINGS",
                                   bg="#00aa44", fg="white",
                                   font=("Helvetica", 10, "bold"),
                                   command=self._save_settings)
        self._save_btn.pack(fill=tk.X, pady=(2, 0))

        # Hand-Occlusion Variablen (fuer IPC-Kompatibilitaet)
        self.hand_var = tk.BooleanVar(value=True)
        self.hand_status_label = tk.Label(model_frame, text="", bg="#0a0a14",
                                          fg="#ff4444", font=("Helvetica", 9, "bold"))
        self.hand_timeout_var = tk.DoubleVar(value=5.0)
        self.hand_streak_var = tk.DoubleVar(value=3.0)
        self.hand_recency_var = tk.DoubleVar(value=2.0)

        # ========== PERCEPTION TABS ==========
        tab_sep = ttk.Separator(model_frame, orient=tk.HORIZONTAL)
        tab_sep.pack(fill=tk.X, pady=(6, 2))

        self._perc_dirty = False
        self._perc_vars = {}
        self._build_perception_tabs(model_frame)

        # Auto-Save Timer starten
        self.after(5000, self._perc_auto_save)

    # ----------------------------------------------------------------
    # Perception Tabs
    # ----------------------------------------------------------------
    def _build_perception_tabs(self, parent):
        """6 Tabs fuer Perception-Parameter."""
        self._perc_nb = ttk.Notebook(parent)
        self._perc_nb.pack(fill=tk.BOTH, expand=True)

        self._build_tab_face(self._perc_nb)
        self._build_tab_hand(self._perc_nb)
        self._build_tab_pose(self._perc_nb)
        self._build_tab_global(self._perc_nb)
        self._build_tab_npu(self._perc_nb)
        self._build_tab_debug(self._perc_nb)

        # Perception config laden
        self._load_perception_config_to_gui()

    def _build_tab_face(self, notebook):
        """Tab 1: Face-Parameter."""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Face")
        self._perc_slider(tab, "Confidence", "face.confidence_threshold", 0.10, 0.90, 0.05, 0.40)
        self._perc_slider(tab, "NMS", "face.nms_threshold", 0.10, 0.90, 0.05, 0.40)
        self._perc_slider(tab, "Recognition", "face.recognition_threshold", 0.30, 0.90, 0.05, 0.60)
        self._perc_slider(tab, "Min Size", "face.min_size", 0.02, 0.30, 0.01, 0.08)
        self._perc_slider(tab, "Max Size", "face.max_size", 0.30, 1.00, 0.05, 0.65)
        self._perc_slider(tab, "Smoothing", "face.position_smoothing", 0.1, 1.0, 0.1, 0.5)
        self._perc_toggle(tab, "Landmarks", "face.landmarks_enabled", True)
        self._perc_toggle(tab, "Head Pose", "face.head_pose_enabled", True)

    def _build_tab_hand(self, notebook):
        """Tab 2: Hand-Parameter."""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Hand")
        self._perc_slider(tab, "Confidence", "hand.confidence_threshold", 0.10, 0.90, 0.05, 0.30)
        self._perc_slider(tab, "Smoothing", "hand.position_smoothing", 0.1, 1.0, 0.1, 0.5)
        self._perc_slider(tab, "Min Crop", "hand.min_crop_size", 80, 200, 10, 140)
        self._perc_slider(tab, "Max Crop", "hand.max_crop_size", 200, 400, 10, 300)
        self._perc_slider(tab, "Max Hands", "hand.max_hands", 1, 4, 1, 1)
        self._perc_toggle(tab, "Landmarks", "hand.landmarks_enabled", True)
        sep = ttk.Separator(tab, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(tab, text="Occlusion", font=("Helvetica", 9, "bold")).pack(anchor=tk.W)
        self._perc_slider(tab, "Timeout", "hand.occlusion_timeout", 1.0, 15.0, 0.5, 5.0)
        self._perc_slider(tab, "Streak", "hand.occlusion_streak", 1, 10, 1, 3)
        self._perc_slider(tab, "Recency", "hand.occlusion_recency", 0.5, 5.0, 0.5, 2.0)

    def _build_tab_pose(self, notebook):
        """Tab 3: Pose-Parameter."""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Pose")
        self._perc_slider(tab, "Confidence", "pose.confidence_threshold", 0.10, 0.90, 0.05, 0.50)
        self._perc_slider(tab, "NMS", "pose.nms_threshold", 0.10, 0.90, 0.05, 0.70)
        self._perc_slider(tab, "Smoothing", "pose.position_smoothing", 0.1, 1.0, 0.1, 0.5)
        self._perc_slider(tab, "Motion Sens.", "pose.motion_sensitivity", 0.1, 1.0, 0.1, 0.5)
        self._perc_slider(tab, "Max Detect.", "pose.max_detections", 1, 10, 1, 10)
        self._perc_toggle(tab, "Landmarks", "pose.landmarks_enabled", True)

    def _build_tab_global(self, notebook):
        """Tab 4: Globale Parameter."""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Global")
        self._perc_slider(tab, "Min Conf.", "global.min_confidence", 0.20, 0.80, 0.05, 0.50)
        self._perc_slider(tab, "Smoothing", "global.position_smoothing", 0.1, 1.0, 0.1, 0.5)
        self._perc_slider(tab, "Max Objects", "global.max_tracked_objects", 1, 5, 1, 3)
        self._perc_slider(tab, "Min Area", "global.min_bbox_area", 0.01, 0.20, 0.01, 0.08)
        self._perc_toggle(tab, "Filter Jitter", "global.filter_jitter", True)
        self._perc_toggle(tab, "Filter Small", "global.filter_small_objects", True)
        self._perc_toggle(tab, "Filter Outliers", "global.filter_outliers", True)

    def _build_tab_npu(self, notebook):
        """Tab 5: NPU / Perception Engine."""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="NPU")
        self._perc_slider(tab, "Max FPS", "npu.max_fps", 5, 30, 1, 15)
        # Power Mode Dropdown
        pm_row = ttk.Frame(tab)
        pm_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(pm_row, text="Power:", font=("Helvetica", 9)).pack(side=tk.LEFT)
        self._perc_vars["npu.power_mode"] = tk.StringVar(value="balanced")
        pm_cb = ttk.Combobox(pm_row, textvariable=self._perc_vars["npu.power_mode"],
                             values=["balanced", "performance", "low_power"],
                             state="readonly", width=12, font=("Helvetica", 9))
        pm_cb.pack(side=tk.RIGHT, padx=3)
        pm_cb.bind("<<ComboboxSelected>>", lambda e: self._perc_mark_dirty())
        self._perc_toggle(tab, "Perception", "npu.perception_enabled", True)
        sep = ttk.Separator(tab, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(tab, text="Scoring", font=("Helvetica", 9, "bold")).pack(anchor=tk.W)
        self._perc_slider(tab, "Swap Interval", "npu.swap_interval", 3.0, 30.0, 1.0, 10.0)
        self._perc_slider(tab, "Hysteresis", "npu.swap_hysteresis", 0.05, 0.50, 0.05, 0.15)
        self._perc_slider(tab, "Base SCRFD", "npu.base_scrfd", 0.0, 1.0, 0.1, 0.6)
        self._perc_slider(tab, "Base ArcFace", "npu.base_arcface", 0.0, 1.0, 0.1, 0.5)
        self._perc_slider(tab, "Base YOLO", "npu.base_yolov8m", 0.0, 1.0, 0.1, 0.4)
        self._perc_slider(tab, "Base Pose", "npu.base_pose", 0.0, 1.0, 0.1, 0.3)
        self._perc_slider(tab, "Base Hand", "npu.base_hand", 0.0, 1.0, 0.1, 0.2)

    def _build_tab_debug(self, notebook):
        """Tab 6: Debug/Display."""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Debug")
        self._perc_toggle(tab, "Show BBoxes", "debug.show_bboxes", True)
        self._perc_toggle(tab, "Show Landmarks", "debug.show_landmarks", True)
        self._perc_toggle(tab, "Show Names", "debug.show_names", True)
        self._perc_toggle(tab, "Show FPS", "debug.show_fps", True)
        self._perc_toggle(tab, "Show Confidence", "debug.show_confidence", True)
        self._perc_toggle(tab, "Show Head Pose", "debug.show_head_pose", True)
        self._perc_toggle(tab, "Show Skeleton", "debug.show_skeleton", True)
        self._perc_toggle(tab, "Show Hand Crop", "debug.show_hand_crop", False)
        sep = ttk.Separator(tab, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(tab, text="Display", font=("Helvetica", 9, "bold")).pack(anchor=tk.W)
        self._perc_toggle(tab, "Flip H", "debug.display_flip_h", False)
        self._perc_toggle(tab, "Flip V", "debug.display_flip_v", False)
        # Rotation Dropdown
        rot_row = ttk.Frame(tab)
        rot_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(rot_row, text="Rotation:", font=("Helvetica", 9)).pack(side=tk.LEFT)
        self._perc_vars["debug.display_rotation"] = tk.IntVar(value=0)
        rot_cb = ttk.Combobox(rot_row, textvariable=self._perc_vars["debug.display_rotation"],
                              values=[0, 90, 180, 270],
                              state="readonly", width=5, font=("Helvetica", 9))
        rot_cb.pack(side=tk.RIGHT, padx=3)
        rot_cb.bind("<<ComboboxSelected>>", lambda e: self._perc_mark_dirty())

    # ----------------------------------------------------------------
    # Perception Tab Helpers
    # ----------------------------------------------------------------
    def _perc_slider(self, parent, label, key, from_, to_, resolution, default):
        """Slider-Zeile: Label | Scale | Wert-Anzeige."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(1, 0))
        ttk.Label(row, text=f"{label}:", font=("Helvetica", 9)).pack(side=tk.LEFT)
        # Int oder Float?
        if isinstance(default, int) and isinstance(from_, int):
            var = tk.IntVar(value=default)
            fmt = "{}"
        else:
            var = tk.DoubleVar(value=float(default))
            fmt = "{:.2f}" if resolution < 0.1 else "{:.1f}"
        self._perc_vars[key] = var
        val_lbl = ttk.Label(row, text=fmt.format(default), width=5,
                            font=("Helvetica", 9))
        val_lbl.pack(side=tk.RIGHT)
        def on_change(v, lbl=val_lbl, f=fmt):
            lbl.configure(text=f.format(float(v)))
            self._perc_dirty = True
        ttk.Scale(row, from_=from_, to=to_, variable=var,
                  command=on_change).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=3)

    def _perc_toggle(self, parent, label, key, default):
        """Toggle-Zeile: Checkbutton."""
        var = tk.BooleanVar(value=default)
        self._perc_vars[key] = var
        cb = tk.Checkbutton(parent, text=label, variable=var,
                            bg="#0a0a14", fg="#e0e0e0", selectcolor="#2a2a4e",
                            activebackground="#1a1a2e", font=("Helvetica", 9),
                            command=self._perc_mark_dirty)
        cb.pack(anchor=tk.W)

    def _perc_mark_dirty(self):
        """Flag setzen: Aenderungen vorhanden."""
        self._perc_dirty = True

    def _perc_auto_save(self):
        """Alle 5s: wenn dirty, perception.json schreiben."""
        if self._perc_dirty:
            self._write_perception_config()
            self._perc_dirty = False
        self.after(5000, self._perc_auto_save)

    def _write_perception_config(self):
        """Alle _perc_vars in perception.json schreiben (atomic)."""
        perc_path = os.path.expanduser("~/moloch/config/perception.json")
        data = {"version": 1}
        for key, var in self._perc_vars.items():
            section, param = key.split(".", 1)
            if section not in data:
                data[section] = {}
            val = var.get()
            # BooleanVar gibt int zurueck, explizit zu bool
            if isinstance(var, tk.BooleanVar):
                val = bool(val)
            data[section][param] = val
        try:
            tmp = perc_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, perc_path)
            logger.info("[PERC-CFG] perception.json gespeichert")
        except Exception as e:
            logger.error(f"[PERC-CFG] Speichern fehlgeschlagen: {e}")

    def _load_perception_config_to_gui(self):
        """perception.json lesen und alle _perc_vars setzen."""
        perc_path = os.path.expanduser("~/moloch/config/perception.json")
        if not os.path.exists(perc_path):
            return
        try:
            with open(perc_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"[PERC-CFG] Laden fehlgeschlagen: {e}")
            return
        for key, var in self._perc_vars.items():
            section, param = key.split(".", 1)
            sec_data = data.get(section, {})
            if param in sec_data:
                try:
                    var.set(sec_data[param])
                except Exception:
                    pass
        # Sync: Face-Thresholds auch in die alten Vars (IPC-Kompatibilitaet)
        face = data.get("face", {})
        if "confidence_threshold" in face:
            self.scrfd_conf_var.set(face["confidence_threshold"])
        if "nms_threshold" in face:
            self.scrfd_nms_var.set(face["nms_threshold"])
        if "recognition_threshold" in face:
            self.arcface_thresh_var.set(face["recognition_threshold"])
        pose = data.get("pose", {})
        if "confidence_threshold" in pose:
            self.pose_conf_var.set(pose["confidence_threshold"])
        if "nms_threshold" in pose:
            self.pose_nms_var.set(pose["nms_threshold"])
        # Hand-Occlusion Vars sync
        hand = data.get("hand", {})
        if "occlusion_timeout" in hand:
            self.hand_timeout_var.set(hand["occlusion_timeout"])
        if "occlusion_streak" in hand:
            self.hand_streak_var.set(hand["occlusion_streak"])
        if "occlusion_recency" in hand:
            self.hand_recency_var.set(hand["occlusion_recency"])
        logger.info("[PERC-CFG] GUI aus perception.json geladen")'''

if old_build in code:
    code = code.replace(old_build, new_build)
    print('FIX 1: _build_model_controls mit Tabs - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

with open(panel, 'w') as f:
    f.write(code)

print(f'\nPanel: {fixes}/1 Fixes.')
if fixes < 1:
    print('PANEL INCOMPLETE!')
    sys.exit(1)

print('\n=== PERCEPTION TABS KOMPLETT ===')
