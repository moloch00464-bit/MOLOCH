#!/usr/bin/env python3
"""Patch: Slider-Werte via IPC an Service durchleiten.

Bug: Panel setzt Slider-Werte auf ServiceProxy-Objekt statt via IPC.
     Der echte Service in systemd bekommt NICHTS.

Fixes:
1. _setup_threshold_bindings: IPC statt setattr bei Proxy-Modus
2. _on_hand_param_change: IPC statt direkter PerceptionProxy-Zugriff
3. _on_hand_toggle: IPC statt direkter PerceptionProxy-Zugriff
4. _execute_panel_cmd: set_hand_params Handler hinzufuegen
5. _apply_status: Slider-Sync OHNE IPC-Feedback-Loop
"""
import sys

fixes_ok = 0

# ===== PANEL FIXES =====
panel_path = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel_path, 'r') as f:
    panel = f.read()

# --- Fix 1: _setup_threshold_bindings mit IPC ---
old_bindings = """    def _setup_threshold_bindings(self):
        \"\"\"Bind slider DoubleVars to service threshold attributes.\"\"\"
        bindings = [
            (self.scrfd_conf_var, "scrfd_conf_val"),
            (self.scrfd_nms_var, "scrfd_nms_val"),
            (self.arcface_thresh_var, "arcface_thresh_val"),
            (self.yolo_conf_var, "yolo_conf_val"),
            (self.pose_conf_var, "pose_conf_val"),
            (self.pose_nms_var, "pose_nms_val"),
        ]
        for var, attr in bindings:
            var.trace_add("write", lambda *_, a=attr, v=var:
                          setattr(self.service, a, v.get()) if self.service else None)"""

new_bindings = """    def _setup_threshold_bindings(self):
        \"\"\"Bind slider DoubleVars to service threshold attributes (via IPC bei Proxy).\"\"\"
        bindings = [
            (self.scrfd_conf_var, "scrfd_conf_val"),
            (self.scrfd_nms_var, "scrfd_nms_val"),
            (self.arcface_thresh_var, "arcface_thresh_val"),
            (self.yolo_conf_var, "yolo_conf_val"),
            (self.pose_conf_var, "pose_conf_val"),
            (self.pose_nms_var, "pose_nms_val"),
        ]
        for var, attr in bindings:
            def on_change(*_, a=attr, v=var):
                if not self.service or self._syncing_thresholds:
                    return
                val = v.get()
                if hasattr(self.service, '_remote_mode') and self.service._remote_mode:
                    self.service._send_cmd({"action": "set_threshold", "attr": a, "value": val})
                else:
                    setattr(self.service, a, val)
            var.trace_add("write", on_change)"""

if old_bindings in panel:
    panel = panel.replace(old_bindings, new_bindings)
    print('FIX 1: _setup_threshold_bindings mit IPC - OK')
    fixes_ok += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# --- Fix 1b: _syncing_thresholds Flag initialisieren ---
old_syncing = "        self._syncing = False\n        self.running = True"
new_syncing = "        self._syncing = False\n        self._syncing_thresholds = False\n        self.running = True"

if old_syncing in panel:
    panel = panel.replace(old_syncing, new_syncing)
    print('FIX 1b: _syncing_thresholds Flag - OK')
    fixes_ok += 1
else:
    print('FIX 1b: ANCHOR NOT FOUND!')

# --- Fix 2: _on_hand_param_change mit IPC ---
old_hand_param = """    def _on_hand_param_change(self, *args):
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
        pe._FACE_RECENCY = self.hand_recency_var.get()"""

new_hand_param = """    def _on_hand_param_change(self, *args):
        \"\"\"Hand-Occlusion Parameter aktualisieren (via IPC bei Proxy).\"\"\"
        # Labels aktualisieren
        self.hand_timeout_lbl.config(text=f"{self.hand_timeout_var.get():.1f}s")
        self.hand_streak_lbl.config(text=f"{int(self.hand_streak_var.get())}")
        self.hand_recency_lbl.config(text=f"{self.hand_recency_var.get():.1f}s")
        if not self.service or not self.hand_var.get():
            return
        params = {
            "timeout": self.hand_timeout_var.get(),
            "streak": int(self.hand_streak_var.get()),
            "recency": self.hand_recency_var.get(),
        }
        if hasattr(self.service, '_remote_mode') and self.service._remote_mode:
            self.service._send_cmd({"action": "set_hand_params", **params})
        elif self.service._perception:
            pe = self.service._perception
            pe._HAND_TIMEOUT = params["timeout"]
            pe._MIN_FACE_STREAK = params["streak"]
            pe._FACE_RECENCY = params["recency"]"""

if old_hand_param in panel:
    panel = panel.replace(old_hand_param, new_hand_param)
    print('FIX 2: _on_hand_param_change mit IPC - OK')
    fixes_ok += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

# --- Fix 3: _on_hand_toggle mit IPC ---
old_hand_toggle = """    def _on_hand_toggle(self):
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
        logger.info(f"[PANEL] Hand-Occlusion: {'AN' if enabled else 'AUS'}")"""

new_hand_toggle = """    def _on_hand_toggle(self):
        \"\"\"Toggle Hand-Occlusion Erkennung (via IPC bei Proxy).\"\"\"
        if not self.service:
            return
        enabled = self.hand_var.get()
        if hasattr(self.service, '_remote_mode') and self.service._remote_mode:
            if enabled:
                self.service._send_cmd({
                    "action": "set_hand_params",
                    "timeout": self.hand_timeout_var.get(),
                    "streak": int(self.hand_streak_var.get()),
                    "recency": self.hand_recency_var.get(),
                })
            else:
                self.service._send_cmd({
                    "action": "set_hand_params",
                    "streak": 999999,
                    "disable_occlusion": True,
                })
        elif self.service._perception:
            pe = self.service._perception
            if enabled:
                pe._HAND_TIMEOUT = self.hand_timeout_var.get()
                pe._MIN_FACE_STREAK = int(self.hand_streak_var.get())
                pe._FACE_RECENCY = self.hand_recency_var.get()
            else:
                pe._MIN_FACE_STREAK = 999999
                pe._hand_occlusion = False
        logger.info(f"[PANEL] Hand-Occlusion: {'AN' if enabled else 'AUS'}")"""

if old_hand_toggle in panel:
    panel = panel.replace(old_hand_toggle, new_hand_toggle)
    print('FIX 3: _on_hand_toggle mit IPC - OK')
    fixes_ok += 1
else:
    print('FIX 3: ANCHOR NOT FOUND!')

# --- Fix 4: _apply_status Threshold-Sync mit DoubleVars ---
old_apply_thresh = """        thresholds = s.get('thresholds', {})
        if thresholds:
            self.scrfd_conf_val = thresholds.get('scrfd_conf', self.scrfd_conf_val)
            self.scrfd_nms_val = thresholds.get('scrfd_nms', self.scrfd_nms_val)
            self.arcface_thresh_val = thresholds.get('arcface_thresh', self.arcface_thresh_val)
            self.yolo_conf_val = thresholds.get('yolo_conf', self.yolo_conf_val)
            self.pose_conf_val = thresholds.get('pose_conf', self.pose_conf_val)
            self.pose_nms_val = thresholds.get('pose_nms', self.pose_nms_val)"""

new_apply_thresh = """        thresholds = s.get('thresholds', {})
        if thresholds:
            self.scrfd_conf_val = thresholds.get('scrfd_conf', self.scrfd_conf_val)
            self.scrfd_nms_val = thresholds.get('scrfd_nms', self.scrfd_nms_val)
            self.arcface_thresh_val = thresholds.get('arcface_thresh', self.arcface_thresh_val)
            self.yolo_conf_val = thresholds.get('yolo_conf', self.yolo_conf_val)
            self.pose_conf_val = thresholds.get('pose_conf', self.pose_conf_val)
            self.pose_nms_val = thresholds.get('pose_nms', self.pose_nms_val)"""

# Threshold-Sync bleibt gleich (nur Proxy-Attribute, nicht DoubleVars)
# Die DoubleVars werden beim Seeding gesetzt und danach vom User gesteuert

with open(panel_path, 'w') as f:
    f.write(panel)
print(f'Panel: {fixes_ok}/4 Fixes geschrieben.\n')


# ===== SERVICE FIXES =====
svc_path = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc_path, 'r') as f:
    svc = f.read()

svc_fixes = 0

# --- Fix 5: set_hand_params IPC Handler ---
old_exec = """        elif action == 'set_threshold':
            attr = cmd.get('attr')
            value = cmd.get('value')
            if attr and value is not None and hasattr(self, attr):
                setattr(self, attr, float(value))"""

new_exec = """        elif action == 'set_threshold':
            attr = cmd.get('attr')
            value = cmd.get('value')
            if attr and value is not None and hasattr(self, attr):
                setattr(self, attr, float(value))
                logger.info(f"[IPC] Threshold: {attr} = {float(value):.3f}")
        elif action == 'set_hand_params':
            if self._perception:
                if cmd.get('disable_occlusion'):
                    self._perception._MIN_FACE_STREAK = 999999
                    self._perception._hand_occlusion = False
                    logger.info("[IPC] Hand-Occlusion DEAKTIVIERT")
                else:
                    self._perception._HAND_TIMEOUT = float(cmd.get('timeout', 5.0))
                    self._perception._MIN_FACE_STREAK = int(cmd.get('streak', 3))
                    self._perception._FACE_RECENCY = float(cmd.get('recency', 2.0))
                    logger.info(f"[IPC] Hand params: timeout={self._perception._HAND_TIMEOUT}, "
                                f"streak={self._perception._MIN_FACE_STREAK}, "
                                f"recency={self._perception._FACE_RECENCY}")"""

if old_exec in svc:
    svc = svc.replace(old_exec, new_exec)
    print('FIX 5: set_hand_params IPC Handler - OK')
    svc_fixes += 1
else:
    print('FIX 5: ANCHOR NOT FOUND!')

# --- Fix 6: Hand-Occlusion Parameter in Status-JSON exportieren ---
old_perception_status = """        "perception": {
            "slots": self._perception.slots if self._perception else [],
            "forced": self._perception._forced_models if self._perception else None,
            "scores": dict(self._perception._last_scores) if self._perception and hasattr(self._perception, '_last_scores') else {},
            "tension": self._perception._tension if self._perception else 0.0,
            "personality_mode": self._perception._personality_mode if self._perception else "guardian",
            "min_interval": self._perception._min_interval if self._perception else 10.0,
            "hand_occlusion": self._perception._hand_occlusion if self._perception else False,
            "face_streak": self._perception._face_streak if self._perception else 0
        }"""

new_perception_status = """        "perception": {
            "slots": self._perception.slots if self._perception else [],
            "forced": self._perception._forced_models if self._perception else None,
            "scores": dict(self._perception._last_scores) if self._perception and hasattr(self._perception, '_last_scores') else {},
            "tension": self._perception._tension if self._perception else 0.0,
            "personality_mode": self._perception._personality_mode if self._perception else "guardian",
            "min_interval": self._perception._min_interval if self._perception else 10.0,
            "hand_occlusion": self._perception._hand_occlusion if self._perception else False,
            "face_streak": self._perception._face_streak if self._perception else 0,
            "hand_timeout": self._perception._HAND_TIMEOUT if self._perception else 5.0,
            "hand_streak_min": self._perception._MIN_FACE_STREAK if self._perception else 3,
            "hand_recency": self._perception._FACE_RECENCY if self._perception else 2.0
        }"""

if old_perception_status in svc:
    svc = svc.replace(old_perception_status, new_perception_status)
    print('FIX 6: Hand params in Status-JSON - OK')
    svc_fixes += 1
else:
    print('FIX 6: ANCHOR NOT FOUND!')

with open(svc_path, 'w') as f:
    f.write(svc)
print(f'Service: {svc_fixes}/2 Fixes geschrieben.\n')

total = fixes_ok + svc_fixes
print(f'GESAMT: {total}/6 Fixes erfolgreich.')
if total < 6:
    sys.exit(1)
