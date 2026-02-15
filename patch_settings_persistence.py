#!/usr/bin/env python3
"""Settings Persistence: config/settings.json

Aenderungen:
1. moloch_service.py: SETTINGS_PATH + _load_settings() + _save_settings() + IPC Handler
2. moloch_unified_panel.py: SAVE SETTINGS Button + Handler
"""
import sys

# ============================================================
# TEIL 1: moloch_service.py
# ============================================================
svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# --- 1A: SETTINGS_PATH Konstante nach NPU_VISION_PAUSED ---
old_path = 'NPU_VISION_PAUSED = "/tmp/moloch_npu_vision_paused"'

new_path = '''NPU_VISION_PAUSED = "/tmp/moloch_npu_vision_paused"
SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.json")'''

if old_path in code:
    code = code.replace(old_path, new_path)
    print('1A: SETTINGS_PATH Konstante - OK')
    fixes += 1
else:
    print('1A: ANCHOR NOT FOUND!')

# --- 1B: _load_settings() Aufruf nach Threshold-Defaults ---
old_defaults = """        self.scrfd_conf_val = 0.40
        self.scrfd_nms_val = 0.40
        self.arcface_thresh_val = 0.60
        self.yolo_conf_val = 0.50
        self.pose_conf_val = 0.50
        self.pose_nms_val = 0.70"""

new_defaults = """        self.scrfd_conf_val = 0.40
        self.scrfd_nms_val = 0.40
        self.arcface_thresh_val = 0.60
        self.yolo_conf_val = 0.50
        self.pose_conf_val = 0.50
        self.pose_nms_val = 0.70

        # Settings aus config/settings.json laden (ueberschreibt Defaults)
        self._load_settings()"""

if old_defaults in code:
    code = code.replace(old_defaults, new_defaults)
    print('1B: _load_settings() Aufruf - OK')
    fixes += 1
else:
    print('1B: ANCHOR NOT FOUND!')

# --- 1C: Hand-Params nach Perception-Init anwenden ---
old_perception = """            self._perception = PerceptionEngine(personality_engine=_pe)
            logger.info(f"[INIT] Perception Engine bereit (Personality: {_pe.mode.value})")"""

new_perception = """            self._perception = PerceptionEngine(personality_engine=_pe)
            logger.info(f"[INIT] Perception Engine bereit (Personality: {_pe.mode.value})")
            # Gespeicherte Hand-Occlusion Params anwenden
            if hasattr(self, '_saved_hand_timeout'):
                self._perception._HAND_TIMEOUT = self._saved_hand_timeout
                self._perception._MIN_FACE_STREAK = self._saved_hand_streak
                self._perception._FACE_RECENCY = self._saved_hand_recency
                logger.info(f"[SETTINGS] Hand-Occlusion Params aus settings.json angewendet")"""

if old_perception in code:
    code = code.replace(old_perception, new_perception)
    print('1C: Hand-Params nach Perception-Init - OK')
    fixes += 1
else:
    print('1C: ANCHOR NOT FOUND!')

# --- 1D: IPC Handler save_settings in _execute_panel_cmd ---
old_ipc = """        elif action == 'set_hand_params':
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

new_ipc = """        elif action == 'set_hand_params':
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
                                f"recency={self._perception._FACE_RECENCY}")
        elif action == 'save_settings':
            # Audio + Camera Werte aus Panel uebernehmen
            _au = cmd.get('audio')
            if _au:
                self._saved_mic_gain = float(_au.get('mic_gain', 1.0))
                self._saved_agc = bool(_au.get('agc_enabled', False))
                self._saved_noise_gate = float(_au.get('noise_gate_db', -60.0))
            _cam = cmd.get('camera')
            if _cam:
                self._saved_ptz_speed = float(_cam.get('ptz_speed', 15.0))
                self._saved_led = bool(_cam.get('led_enabled', False))
                self._saved_ir = str(_cam.get('ir_mode', 'Aus'))
            self._save_settings()"""

if old_ipc in code:
    code = code.replace(old_ipc, new_ipc)
    print('1D: IPC Handler save_settings - OK')
    fixes += 1
else:
    print('1D: ANCHOR NOT FOUND!')

# --- 1E: _load_settings() + _save_settings() Methoden vor if __name__ ---
old_main = '''if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.setLevel(logging.INFO)

    print("M.O.L.O.C.H. Core Service - Phase 2")
    service = MolochService()
    service.init()
    service.start(blocking=True)'''

new_main = '''    # ----------------------------------------------------------------
    # Settings Persistence
    # ----------------------------------------------------------------
    def _load_settings(self):
        """Lade Settings aus config/settings.json (ueberschreibt Defaults)."""
        if not os.path.exists(SETTINGS_PATH):
            logger.info("[SETTINGS] Keine settings.json vorhanden - verwende Defaults")
            return
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"[SETTINGS] Lade settings.json (version={data.get('version', '?')})")
        except Exception as e:
            logger.warning(f"[SETTINGS] settings.json korrupt, verwende Defaults: {e}")
            return

        # Thresholds
        try:
            th = data.get("thresholds", {})
            if "scrfd_conf" in th:
                self.scrfd_conf_val = float(th["scrfd_conf"])
            if "scrfd_nms" in th:
                self.scrfd_nms_val = float(th["scrfd_nms"])
            if "arcface_thresh" in th:
                self.arcface_thresh_val = float(th["arcface_thresh"])
            if "yolo_conf" in th:
                self.yolo_conf_val = float(th["yolo_conf"])
            if "pose_conf" in th:
                self.pose_conf_val = float(th["pose_conf"])
            if "pose_nms" in th:
                self.pose_nms_val = float(th["pose_nms"])
            logger.info(f"[SETTINGS] Thresholds: scrfd={self.scrfd_conf_val}/{self.scrfd_nms_val} "
                        f"arc={self.arcface_thresh_val} yolo={self.yolo_conf_val} "
                        f"pose={self.pose_conf_val}/{self.pose_nms_val}")
        except Exception as e:
            logger.warning(f"[SETTINGS] Thresholds-Fehler: {e}")

        # Hand-Occlusion (gespeichert fuer spaeter, Perception Engine existiert noch nicht)
        try:
            ho = data.get("hand_occlusion", {})
            if ho:
                self._saved_hand_timeout = float(ho.get("timeout", 5.0))
                self._saved_hand_streak = int(ho.get("streak", 3))
                self._saved_hand_recency = float(ho.get("recency", 2.0))
                logger.info(f"[SETTINGS] Hand-Occlusion: timeout={self._saved_hand_timeout} "
                            f"streak={self._saved_hand_streak} recency={self._saved_hand_recency}")
        except Exception as e:
            logger.warning(f"[SETTINGS] Hand-Occlusion-Fehler: {e}")

        # Audio
        try:
            au = data.get("audio", {})
            if au:
                self._saved_mic_gain = float(au.get("mic_gain", 1.0))
                self._saved_agc = bool(au.get("agc_enabled", False))
                self._saved_noise_gate = float(au.get("noise_gate_db", -60.0))
                logger.info(f"[SETTINGS] Audio: gain={self._saved_mic_gain} "
                            f"agc={self._saved_agc} gate={self._saved_noise_gate}")
        except Exception as e:
            logger.warning(f"[SETTINGS] Audio-Fehler: {e}")

        # Camera
        try:
            cam = data.get("camera", {})
            if cam:
                self._saved_ptz_speed = float(cam.get("ptz_speed", 15.0))
                self._saved_led = bool(cam.get("led_enabled", False))
                self._saved_ir = str(cam.get("ir_mode", "Aus"))
                logger.info(f"[SETTINGS] Camera: speed={self._saved_ptz_speed} "
                            f"led={self._saved_led} ir={self._saved_ir}")
        except Exception as e:
            logger.warning(f"[SETTINGS] Camera-Fehler: {e}")

    def _save_settings(self):
        """Speichere aktuelle Settings nach config/settings.json (atomic write)."""
        data = {"version": 1}

        # Thresholds
        data["thresholds"] = {
            "scrfd_conf": round(self.scrfd_conf_val, 3),
            "scrfd_nms": round(self.scrfd_nms_val, 3),
            "arcface_thresh": round(self.arcface_thresh_val, 3),
            "yolo_conf": round(self.yolo_conf_val, 3),
            "pose_conf": round(self.pose_conf_val, 3),
            "pose_nms": round(self.pose_nms_val, 3),
        }

        # Hand-Occlusion
        if self._perception:
            data["hand_occlusion"] = {
                "timeout": round(self._perception._HAND_TIMEOUT, 1),
                "streak": self._perception._MIN_FACE_STREAK,
                "recency": round(self._perception._FACE_RECENCY, 1),
            }

        # Audio (aus gespeicherten Werten oder Defaults)
        data["audio"] = {
            "mic_gain": round(getattr(self, '_saved_mic_gain', 1.0), 2),
            "agc_enabled": getattr(self, '_saved_agc', False),
            "noise_gate_db": round(getattr(self, '_saved_noise_gate', -60.0), 1),
        }

        # Camera
        data["camera"] = {
            "ptz_speed": round(getattr(self, '_saved_ptz_speed', 15.0), 1),
            "led_enabled": getattr(self, '_saved_led', False),
            "ir_mode": getattr(self, '_saved_ir', "Aus"),
        }

        # Atomic write
        try:
            tmp = SETTINGS_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, SETTINGS_PATH)
            logger.info(f"[SETTINGS] Gespeichert: {SETTINGS_PATH}")
        except Exception as e:
            logger.error(f"[SETTINGS] Speichern fehlgeschlagen: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.setLevel(logging.INFO)

    print("M.O.L.O.C.H. Core Service - Phase 2")
    service = MolochService()
    service.init()
    service.start(blocking=True)'''

if old_main in code:
    code = code.replace(old_main, new_main)
    print('1E: _load_settings + _save_settings Methoden - OK')
    fixes += 1
else:
    print('1E: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

print(f'\nService: {fixes}/5 Fixes angewendet.')
if fixes < 5:
    print('SERVICE INCOMPLETE!')
    sys.exit(1)

# ============================================================
# TEIL 2: moloch_unified_panel.py
# ============================================================
panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    pcode = f.read()

pfixes = 0

# --- 2A: SAVE SETTINGS Button nach Hand LM, vor Hand-Occlusion ---
old_save_area = """        self._hand_lm_fps = self._build_model_section(
            model_frame, "Hand LM", self.hand_lm_var, "hand_landmark", [])

        # --- Hand-Occlusion (Auto-Erkennung) ---"""

new_save_area = """        self._hand_lm_fps = self._build_model_section(
            model_frame, "Hand LM", self.hand_lm_var, "hand_landmark", [])

        # --- Save Settings ---
        save_sep = ttk.Separator(model_frame, orient=tk.HORIZONTAL)
        save_sep.pack(fill=tk.X, pady=(8, 4))
        self._save_btn = tk.Button(model_frame, text="SAVE SETTINGS",
                                   bg="#00aa44", fg="white",
                                   font=("Helvetica", 10, "bold"),
                                   command=self._save_settings)
        self._save_btn.pack(fill=tk.X, pady=(2, 0))

        # --- Hand-Occlusion (Auto-Erkennung) ---"""

if old_save_area in pcode:
    pcode = pcode.replace(old_save_area, new_save_area)
    print('2A: SAVE SETTINGS Button - OK')
    pfixes += 1
else:
    print('2A: ANCHOR NOT FOUND!')

# --- 2B: _save_settings Handler - nach _on_hand_param_change oder _on_hand_toggle ---
# Suche nach _on_hand_toggle um die Methode danach einzufuegen
old_handler_anchor = """    def _on_hand_toggle(self):"""

new_handler_anchor = """    def _save_settings(self):
        \"\"\"Alle Settings persistent speichern via IPC.\"\"\"
        self._send_cmd({
            "action": "save_settings",
            "audio": {
                "mic_gain": self._mic_gain_var.get(),
                "agc_enabled": self._agc_var.get(),
                "noise_gate_db": self._noise_gate_var.get(),
            },
            "camera": {
                "ptz_speed": self.speed_var.get(),
                "led_enabled": self.led_var.get(),
                "ir_mode": self.ir_var.get(),
            },
        })
        # Visuelles Feedback
        self._save_btn.config(text="SAVED!", bg="#006622")
        self.after(2000, lambda: self._save_btn.config(text="SAVE SETTINGS", bg="#00aa44"))

    def _on_hand_toggle(self):"""

if old_handler_anchor in pcode:
    pcode = pcode.replace(old_handler_anchor, new_handler_anchor, 1)
    print('2B: _save_settings Handler - OK')
    pfixes += 1
else:
    print('2B: ANCHOR NOT FOUND!')

with open(panel, 'w') as f:
    f.write(pcode)

print(f'\nPanel: {pfixes}/2 Fixes angewendet.')
if pfixes < 2:
    print('PANEL INCOMPLETE!')
    sys.exit(1)

print('\n=== SETTINGS PERSISTENCE KOMPLETT ===')
