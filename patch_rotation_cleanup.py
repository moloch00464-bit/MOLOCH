#!/usr/bin/env python3
"""Patch: Tentakel Swap + GUI FIFO entfernen, NUR Perception Engine entscheidet.

Aenderungen in moloch_service.py:
1. Tentakel Swap Block im Inference Loop ENTFERNEN
2. _swap_to_arcface() und _swap_to_yolov8m() Methoden ENTFERNEN
3. State-Variablen (face_mode, swap_lock etc.) aus __init__ ENTFERNEN
4. _all_models_off() Face-Mode-Zeilen BEREINIGEN
5. Release-Code Face-Mode-Reset BEREINIGEN
6. _model_order aus configure/unconfigure ENTFERNEN
7. toggle_model() FIFO -> perception.force_models()
8. IPC Handler: toggle_model -> force_models
9. MAX_CONCURRENT_MODELS Konstante ENTFERNEN
"""

import sys

path = "/home/molochzuhause/moloch/core/moloch_service.py"
with open(path, "r") as f:
    code = f.read()

original = code  # Backup fuer Vergleich
changes = 0

# =========================================================================
# 1. MAX_CONCURRENT_MODELS Konstante entfernen
# =========================================================================
old = "MAX_CONCURRENT_MODELS = 2\n"
if old in code:
    code = code.replace(old, "", 1)
    changes += 1
    print("1. MAX_CONCURRENT_MODELS entfernt")
else:
    print("WARN 1: MAX_CONCURRENT_MODELS nicht gefunden")

# =========================================================================
# 2. State-Variablen aus __init__ entfernen
# =========================================================================
old = """        # Dynamic Model Swap
        self._face_mode_active = False
        self._face_seen_count = 0
        self._face_lost_time = 0
        self._swapping_models = False
        self._swap_lock = threading.Lock()
        self._FACE_MODE_FRAMES = 3
        self._FACE_MODE_TIMEOUT = 5.0
        self._FACE_MODE_STARTUP_DELAY = 5.0"""

if old in code:
    code = code.replace(old, "", 1)
    changes += 1
    print("2. State-Variablen aus __init__ entfernt")
else:
    print("WARN 2: State-Variablen nicht gefunden")

# =========================================================================
# 3. _model_order aus __init__ entfernen
# =========================================================================
old = """        self._model_order = []\n"""
if old in code:
    code = code.replace(old, "", 1)
    changes += 1
    print("3. _model_order aus __init__ entfernt")
else:
    print("WARN 3: _model_order init nicht gefunden")

# =========================================================================
# 4. _model_order aus _configure_model entfernen
# =========================================================================
old = """                if name not in self._model_order:
                    self._model_order.append(name)"""
if old in code:
    code = code.replace(old, "", 1)
    changes += 1
    print("4. _model_order aus _configure_model entfernt")
else:
    print("WARN 4: _model_order in configure nicht gefunden")

# =========================================================================
# 5. _model_order aus _unconfigure_model entfernen
# =========================================================================
old = """                if name in self._model_order:
                    self._model_order.remove(name)"""
if old in code:
    code = code.replace(old, "", 1)
    changes += 1
    print("5. _model_order aus _unconfigure_model entfernt")
else:
    print("WARN 5: _model_order in unconfigure nicht gefunden")

# =========================================================================
# 6. Tentakel Swap Block im Inference Loop ENTFERNEN
# =========================================================================
old = """            # === Dynamischer Modell-Swap: Face -> ArcFace, kein Face -> YOLOv8m ===
            _swap_allowed = (self._moloch_has_control and not self._swapping_models
                             and hasattr(self, '_takeover_time')
                             and time.time() - self._takeover_time > self._FACE_MODE_STARTUP_DELAY)
            if _swap_allowed:
                if face_detected:
                    self._face_seen_count += 1
                    self._face_lost_time = 0
                    if not self._face_mode_active and self._face_seen_count >= self._FACE_MODE_FRAMES:
                        threading.Thread(target=self._swap_to_arcface, daemon=True).start()
                else:
                    self._face_seen_count = 0
                    if self._face_mode_active:
                        if self._face_lost_time == 0:
                            self._face_lost_time = time.time()
                        elif time.time() - self._face_lost_time > self._FACE_MODE_TIMEOUT:
                            threading.Thread(target=self._swap_to_yolov8m, daemon=True).start()

"""

if old in code:
    code = code.replace(old, "\n", 1)
    changes += 1
    print("6. Tentakel Swap Block aus Inference Loop entfernt")
else:
    print("WARN 6: Tentakel Swap Block nicht gefunden")

# =========================================================================
# 7. _swap_to_arcface + _swap_to_yolov8m Methoden ENTFERNEN
# =========================================================================
old = """    # =========================================================================
    # Model Swap (ArcFace <-> YOLOv8m)
    # =========================================================================

    def _swap_to_arcface(self):
        \"\"\"YOLOv8m -> ArcFace swap (MOLOCH will wissen WER da ist).\"\"\"
        with self._swap_lock:
            if self._face_mode_active or self._swapping_models:
                return
            self._swapping_models = True
        try:
            logger.info("[TENTAKEL] Face erkannt -> Swap YOLOv8m -> ArcFace")
            self._unconfigure_model("yolov8m")
            self.yolo_active = False
            self._notify("model_toggle", {"yolov8m": False})
            time.sleep(0.2)
            self._configure_model("arcface")
            self.arcface_active = True
            self._notify("model_toggle", {"arcface": True})
            self._face_mode_active = True
            logger.info("[TENTAKEL] ArcFace aktiv - Gesichtserkennung laeuft")
        except Exception as e:
            logger.error(f"Swap to ArcFace failed: {e}")
        finally:
            with self._swap_lock:
                self._swapping_models = False

    def _swap_to_yolov8m(self):
        \"\"\"ArcFace -> YOLOv8m swap (kein Face mehr, braucht Person-Detection).\"\"\"
        with self._swap_lock:
            if not self._face_mode_active or self._swapping_models:
                return
            self._swapping_models = True
        try:
            logger.info("[TENTAKEL] Face verloren -> Swap ArcFace -> YOLOv8m")
            self._unconfigure_model("arcface")
            self.arcface_active = False
            self._notify("model_toggle", {"arcface": False})
            time.sleep(0.2)
            self._configure_model("yolov8m")
            self.yolo_active = True
            self._notify("model_toggle", {"yolov8m": True})
            self._face_mode_active = False
            logger.info("[TENTAKEL] YOLOv8m aktiv - Person-Detection laeuft")
        except Exception as e:
            logger.error(f"Swap to YOLOv8m failed: {e}")
        finally:
            with self._swap_lock:
                self._swapping_models = False
"""

if old in code:
    code = code.replace(old, "", 1)
    changes += 1
    print("7. _swap_to_arcface + _swap_to_yolov8m Methoden entfernt")
else:
    print("WARN 7: Swap-Methoden nicht gefunden")

# =========================================================================
# 8. _all_models_off() Face-Mode-Zeilen bereinigen
# =========================================================================
old = """        self._models_preloaded = False
        self._face_mode_active = False
        self._face_seen_count = 0
        self._face_lost_time = 0
        self.scrfd_active = False"""

new = """        self._models_preloaded = False
        self.scrfd_active = False"""

if old in code:
    code = code.replace(old, new, 1)
    changes += 1
    print("8. _all_models_off() Face-Mode-Zeilen bereinigt")
else:
    print("WARN 8: _all_models_off Face-Mode nicht gefunden")

# =========================================================================
# 9. Release-Code Face-Mode-Reset bereinigen
# =========================================================================
old = """            # Face mode reset
            self._face_mode_active = False
            self._face_seen_count = 0
            self._face_lost_time = 0

            # Position-Tracking"""

new = """            # Position-Tracking"""

if old in code:
    code = code.replace(old, new, 1)
    changes += 1
    print("9. Release-Code Face-Mode-Reset bereinigt")
else:
    print("WARN 9: Release Face-Mode-Reset nicht gefunden")

# =========================================================================
# 10. toggle_model() FIFO -> perception.force_models()
# =========================================================================
old = """    def toggle_model(self, model_key, enabled):
        \"\"\"Toggle model on/off mit FIFO eviction (fuer GUI-Buttons).\"\"\"
        active_map = {"scrfd": "scrfd_active", "arcface": "arcface_active",
                      "yolov8m": "yolo_active", "pose": "pose_active"}

        # ArcFace braucht SCRFD
        if model_key == "arcface" and enabled and not self.scrfd_active:
            self.toggle_model("scrfd", True)

        # SCRFD aus -> ArcFace auch aus
        if model_key == "scrfd" and not enabled and self.arcface_active:
            self.toggle_model("arcface", False)

        attr = active_map.get(model_key)
        if not attr:
            return
        setattr(self, attr, enabled)

        if enabled:
            evicted = False
            if len(self._active_ctx) >= MAX_CONCURRENT_MODELS:
                oldest = self._model_order[0] if self._model_order else None
                if oldest:
                    logger.info(f"NPU-Limit: {oldest} wird deaktiviert fuer {model_key} (FIFO)")
                    self._update_status(f"Wechsel: {oldest} -> {model_key}...")
                    oldest_attr = active_map.get(oldest)
                    if oldest_attr:
                        setattr(self, oldest_attr, False)
                    if oldest == "scrfd" and self.arcface_active:
                        self.arcface_active = False
                        self._unconfigure_model("arcface")
                        self._notify("model_toggle", {"arcface": False})
                    self._unconfigure_model(oldest)
                    self._notify("model_toggle", {oldest: False})
                    evicted = True

            def do_cfg():
                try:
                    if evicted:
                        time.sleep(0.2)
                    self._configure_model(model_key)
                    self._update_status("RTSP + NPU aktiv")
                except Exception as e:
                    crash_log = os.path.expanduser("~/moloch/logs/panel_crash.log")
                    crash_info = (f"\\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                                  f"TOGGLE CRASH: {model_key}\\n{traceback.format_exc()}\\n")
                    logger.error(crash_info)
                    try:
                        with open(crash_log, "a") as f:
                            f.write(crash_info)
                    except Exception:
                        pass
                    self._update_status(f"CRASH: {model_key}")
            threading.Thread(target=do_cfg, daemon=True).start()
        else:
            self._unconfigure_model(model_key)
            with self._fps_lock:
                self._fps[model_key] = 0
        self._notify("model_toggle", {model_key: enabled})"""

new = """    def toggle_model(self, model_key, enabled):
        \"\"\"Toggle model on/off via Perception Engine force_models().\"\"\"
        if not self._perception:
            logger.warning(f"[TOGGLE] Perception Engine nicht verfuegbar, ignoriere {model_key}={enabled}")
            return

        active_map = {"scrfd": "scrfd_active", "arcface": "arcface_active",
                      "yolov8m": "yolo_active", "pose": "pose_active"}
        if model_key not in active_map:
            return

        # Aktuelle gewuenschte Modelle ermitteln
        current = set(self._active_ctx.keys())
        if enabled:
            wanted = current | {model_key}
            # ArcFace braucht SCRFD
            if "arcface" in wanted and "scrfd" not in wanted:
                wanted.add("scrfd")
        else:
            wanted = current - {model_key}
            # SCRFD weg -> ArcFace auch weg
            if model_key == "scrfd":
                wanted.discard("arcface")

        if wanted:
            self._perception.force_models(list(wanted))
            logger.info(f"[TOGGLE] force_models({list(wanted)}) via Panel")
        else:
            # Alles aus -> zurueck zu Auto-Scoring
            self._perception.force_models(None)
            logger.info("[TOGGLE] Alle Modelle aus -> Perception Auto-Modus")"""

if old in code:
    code = code.replace(old, new, 1)
    changes += 1
    print("10. toggle_model() FIFO ersetzt durch perception.force_models()")
else:
    print("WARN 10: toggle_model() FIFO nicht gefunden")

# =========================================================================
# 11. IPC Handler: force_models Aktion hinzufuegen
# =========================================================================
old = """        if action == 'toggle_model':
            model = cmd.get('model')
            enabled = cmd.get('enabled', False)
            if model:
                self.toggle_model(model, enabled)"""

new = """        if action == 'toggle_model':
            model = cmd.get('model')
            enabled = cmd.get('enabled', False)
            if model:
                self.toggle_model(model, enabled)
        elif action == 'force_models':
            models = cmd.get('models')  # List[str] oder None
            if self._perception:
                self._perception.force_models(models)
                logger.info(f"[IPC] force_models({models})")"""

if old in code:
    code = code.replace(old, new, 1)
    changes += 1
    print("11. IPC Handler: force_models Aktion hinzugefuegt")
else:
    print("WARN 11: IPC toggle_model Handler nicht gefunden")

# =========================================================================
# Ergebnis
# =========================================================================
print(f"\n{'='*60}")
print(f"Aenderungen: {changes}/11")

if changes < 8:
    print("FEHLER: Zu wenige Aenderungen, Patch wird NICHT geschrieben!")
    sys.exit(1)

with open(path, "w") as f:
    f.write(code)
print(f"Patch geschrieben: {path}")

# Syntax-Check
import py_compile
try:
    py_compile.compile(path, doraise=True)
    print("Syntax-Check: OK")
except py_compile.PyCompileError as e:
    print(f"SYNTAX FEHLER: {e}")
    # Rollback
    with open(path, "w") as f:
        f.write(original)
    print("ROLLBACK: Original wiederhergestellt!")
    sys.exit(1)

print("\nZusammenfassung:")
print("  - Tentakel Swap (Inference Loop + Methoden): ENTFERNT")
print("  - GUI FIFO (toggle_model + _model_order): ENTFERNT")
print("  - toggle_model() -> nutzt perception.force_models()")
print("  - IPC: force_models Aktion hinzugefuegt")
print("  - State-Variablen bereinigt")
print("  - NUR Perception Engine entscheidet ueber Modell-Rotation")
