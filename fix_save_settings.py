#!/usr/bin/env python3
"""Fix: ALLE Panel-Einstellungen speichern UND beim Start laden.

1. Panel: _save_settings() erweitern (Hand-Occlusion, Thresholds)
2. Panel: _load_panel_settings() beim Start
3. Service: Erweiterte Daten aus save_settings IPC verarbeiten
4. Service: _save_settings() erweitern

HINWEIS: Voice wird NICHT gespeichert - MOLOCHs Stimme ist sein Wille!
"""
import sys

# ============================================================
# TEIL 1: Panel - _save_settings erweitern
# ============================================================
panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    code = f.read()

fixes = 0

# FIX 1A: _save_settings um alle Einstellungen erweitern
old_save = '''    def _save_settings(self):
        """Alle Settings persistent speichern via IPC."""
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
        })'''

new_save = '''    def _save_settings(self):
        """ALLE Settings persistent speichern via IPC."""
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
            "thresholds": {
                "scrfd_conf": self.scrfd_conf_var.get(),
                "scrfd_nms": self.scrfd_nms_var.get(),
                "arcface_thresh": self.arcface_thresh_var.get(),
                "yolo_conf": self.yolo_conf_var.get(),
                "pose_conf": self.pose_conf_var.get(),
                "pose_nms": self.pose_nms_var.get(),
            },
            "hand_occlusion": {
                "enabled": self.hand_var.get(),
                "timeout": self.hand_timeout_var.get(),
                "streak": self.hand_streak_var.get(),
                "recency": self.hand_recency_var.get(),
            },
        })'''

if old_save in code:
    code = code.replace(old_save, new_save)
    print('FIX 1A: _save_settings erweitert - OK')
    fixes += 1
else:
    print('FIX 1A: ANCHOR NOT FOUND!')

# FIX 1B: _load_panel_settings() Methode hinzufuegen (vor _save_settings)
old_save_start = '''    def _save_settings(self):
        """ALLE Settings persistent speichern via IPC."""'''

new_load_and_save = '''    def _load_panel_settings(self):
        """Gespeicherte Settings aus settings.json laden und anwenden."""
        import json
        settings_path = os.path.expanduser("~/moloch/config/settings.json")
        if not os.path.exists(settings_path):
            return
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Audio
            au = data.get("audio", {})
            if "mic_gain" in au:
                self._mic_gain_var.set(float(au["mic_gain"]))
            if "agc_enabled" in au:
                self._agc_var.set(bool(au["agc_enabled"]))
            if "noise_gate_db" in au:
                self._noise_gate_var.set(float(au["noise_gate_db"]))

            # Camera
            cam = data.get("camera", {})
            if "ptz_speed" in cam:
                self.speed_var.set(float(cam["ptz_speed"]))
            if "led_enabled" in cam:
                self.led_var.set(bool(cam["led_enabled"]))
            if "ir_mode" in cam:
                self.ir_var.set(str(cam["ir_mode"]))

            # Thresholds
            th = data.get("thresholds", {})
            if "scrfd_conf" in th:
                self.scrfd_conf_var.set(float(th["scrfd_conf"]))
            if "scrfd_nms" in th:
                self.scrfd_nms_var.set(float(th["scrfd_nms"]))
            if "arcface_thresh" in th:
                self.arcface_thresh_var.set(float(th["arcface_thresh"]))
            if "yolo_conf" in th:
                self.yolo_conf_var.set(float(th["yolo_conf"]))
            if "pose_conf" in th:
                self.pose_conf_var.set(float(th["pose_conf"]))
            if "pose_nms" in th:
                self.pose_nms_var.set(float(th["pose_nms"]))

            # Hand-Occlusion
            ho = data.get("hand_occlusion", {})
            if "enabled" in ho:
                self.hand_var.set(bool(ho["enabled"]))
            if "timeout" in ho:
                self.hand_timeout_var.set(float(ho["timeout"]))
            if "streak" in ho:
                self.hand_streak_var.set(float(ho["streak"]))
            if "recency" in ho:
                self.hand_recency_var.set(float(ho["recency"]))

            self._append_chat("[Settings] Einstellungen geladen", "system")
        except Exception as e:
            self._append_chat(f"[Settings] Laden fehlgeschlagen: {e}", "system")

    def _save_settings(self):
        """ALLE Settings persistent speichern via IPC."""'''

if old_save_start in code:
    code = code.replace(old_save_start, new_load_and_save)
    print('FIX 1B: _load_panel_settings() Methode - OK')
    fixes += 1
else:
    print('FIX 1B: ANCHOR NOT FOUND!')

# FIX 1C: _load_panel_settings() beim Start aufrufen (nach GUI init)
old_gui_start = '''        # Perception config laden
        self._load_perception_config_to_gui()'''

new_gui_start = '''        # Perception config laden
        self._load_perception_config_to_gui()

        # Gespeicherte Settings laden
        self._load_panel_settings()'''

if old_gui_start in code:
    code = code.replace(old_gui_start, new_gui_start)
    print('FIX 1C: _load_panel_settings() Aufruf - OK')
    fixes += 1
else:
    print('FIX 1C: ANCHOR NOT FOUND!')

with open(panel, 'w') as f:
    f.write(code)

print(f'\nPanel: {fixes}/3 Fixes.')

# ============================================================
# TEIL 2: Service - save_settings IPC + _save_settings erweitern
# ============================================================
svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code2 = f.read()

fixes2 = 0

# FIX 2A: IPC Handler - erweiterte Daten verarbeiten
old_ipc = '''        elif action == 'save_settings':
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
            self._save_settings()'''

new_ipc = '''        elif action == 'save_settings':
            # ALLE Werte aus Panel uebernehmen
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
            _th = cmd.get('thresholds')
            if _th:
                self.scrfd_conf_val = float(_th.get('scrfd_conf', self.scrfd_conf_val))
                self.scrfd_nms_val = float(_th.get('scrfd_nms', self.scrfd_nms_val))
                self.arcface_thresh_val = float(_th.get('arcface_thresh', self.arcface_thresh_val))
                self.yolo_conf_val = float(_th.get('yolo_conf', self.yolo_conf_val))
                self.pose_conf_val = float(_th.get('pose_conf', self.pose_conf_val))
                self.pose_nms_val = float(_th.get('pose_nms', self.pose_nms_val))
            _ho = cmd.get('hand_occlusion')
            if _ho and self._perception:
                self._perception._hand_occlusion = bool(_ho.get('enabled', True))
                self._perception._HAND_TIMEOUT = float(_ho.get('timeout', 5.0))
                self._perception._MIN_FACE_STREAK = int(_ho.get('streak', 3))
                self._perception._FACE_RECENCY = float(_ho.get('recency', 2.0))
            self._save_settings()
            logger.info("[SETTINGS] Alle Panel-Einstellungen gespeichert")'''

if old_ipc in code2:
    code2 = code2.replace(old_ipc, new_ipc)
    print('FIX 2A: IPC Handler erweitert - OK')
    fixes2 += 1
else:
    print('FIX 2A: ANCHOR NOT FOUND!')

# FIX 2B: (entfaellt - Voice ist MOLOCHs Wille, wird nicht gespeichert)
fixes2 += 1  # Zaehlt trotzdem als "erledigt"
print('FIX 2B: Voice NICHT gespeichert (MOLOCHs Wille) - OK')

# FIX 2C: _save_settings() - Hand-Occlusion enabled hinzufuegen
old_hand_save = '''        # Hand-Occlusion
        if hasattr(self, "_perception") and self._perception:
            data["hand_occlusion"] = {
                "timeout": round(self._perception._HAND_TIMEOUT, 1),
                "streak": self._perception._MIN_FACE_STREAK,
                "recency": round(self._perception._FACE_RECENCY, 1),
            }'''

new_hand_save = '''        # Hand-Occlusion
        if hasattr(self, "_perception") and self._perception:
            data["hand_occlusion"] = {
                "enabled": bool(self._perception._hand_occlusion),
                "timeout": round(self._perception._HAND_TIMEOUT, 1),
                "streak": self._perception._MIN_FACE_STREAK,
                "recency": round(self._perception._FACE_RECENCY, 1),
            }'''

if old_hand_save in code2:
    code2 = code2.replace(old_hand_save, new_hand_save)
    print('FIX 2C: Hand-Occlusion enabled in _save_settings - OK')
    fixes2 += 1
else:
    print('FIX 2C: ANCHOR NOT FOUND!')

# FIX 2D: (entfaellt - Voice ist MOLOCHs Wille, wird nicht geladen)
fixes2 += 1  # Zaehlt trotzdem als "erledigt"
print('FIX 2D: Voice NICHT geladen (MOLOCHs Wille) - OK')

with open(svc, 'w') as f:
    f.write(code2)

print(f'\nService: {fixes2}/4 Fixes.')

# Syntax-Check
total = fixes + fixes2
compile(open(panel).read(), panel, 'exec')
compile(open(svc).read(), svc, 'exec')
print('\nSyntax OK (beide Dateien)')

if total < 7:
    print(f'\n!!! INCOMPLETE: {total}/7 Fixes !!!')
    sys.exit(1)

print('\n=== SAVE SETTINGS FIX KOMPLETT ===')
print('Panel laedt jetzt settings.json beim Start')
print('SAVE speichert: Audio, Camera, Thresholds, Hand-Occlusion (KEINE Voice)')
