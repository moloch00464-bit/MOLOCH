#!/usr/bin/env python3
"""Fix: RTSP Watchdog - Auto-Reconnect bei Stream-Verlust.

Problem: Kamera-Stream friert ein wenn Kamera stromlos war.
Panel zeigt Frozen-Frame, kein Auto-Reconnect.

Fix:
1. RTSP Watchdog prueft alle 5s ob neue Frames kommen
2. Nach 3x Fehler (15s) -> Stream tot, Reconnect alle 10s
3. Panel zeigt KAMERA OFFLINE statt Frozen-Frame
4. Auto-Reconnect im Hintergrund
5. Stream-Status Indikator: gruen/gelb/rot
"""
import sys

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# FIX 1: Init-Variablen fuer RTSP Watchdog hinzufuegen
# Nach self._frame_lock = threading.Lock()
old_init = '''        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._annotated_lock = threading.Lock()'''

new_init = '''        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._annotated_lock = threading.Lock()

        # RTSP Watchdog
        self._rtsp_last_frame_time = 0
        self._rtsp_stream_alive = False
        self._rtsp_reconnecting = False
        self._rtsp_cap = None'''

if old_init in code:
    code = code.replace(old_init, new_init)
    print('FIX 1: RTSP Watchdog Init-Variablen - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# FIX 2: _start_rtsp mit Watchdog + Reconnect Logic
old_rtsp = '''    def _start_rtsp(self):
        """Starte RTSP Background Reader."""
        def rtsp_reader():
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"
            )
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                self._update_status(f"RTSP FEHLER: {RTSP_URL}")
                return

            self._update_status("RTSP aktiv")

            while self.running:
                grabbed = cap.grab()
                if grabbed:
                    ret, frame = cap.retrieve()
                    if ret and frame is not None:
                        frame = cv2.resize(frame, (self.PREVIEW_W, self.PREVIEW_H))
                        with self._frame_lock:
                            self._latest_frame = frame
                else:
                    time.sleep(0.1)

            cap.release()

        threading.Thread(target=rtsp_reader, daemon=True, name="RTSPReader").start()'''

new_rtsp = '''    def _start_rtsp(self):
        """Starte RTSP Background Reader + Watchdog."""
        def rtsp_connect():
            """RTSP Verbindung aufbauen."""
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"
            )
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap if cap.isOpened() else None

        def rtsp_reader():
            """RTSP Reader mit Auto-Reconnect."""
            self._rtsp_cap = rtsp_connect()

            if not self._rtsp_cap:
                self._update_status(f"RTSP FEHLER: {RTSP_URL}")
                self._rtsp_stream_alive = False
                return

            self._update_status("RTSP aktiv")
            self._rtsp_stream_alive = True
            self._rtsp_last_frame_time = time.time()
            consecutive_fails = 0

            while self.running:
                if self._rtsp_reconnecting:
                    # Watchdog triggered reconnect
                    if self._rtsp_cap:
                        self._rtsp_cap.release()
                    logger.info("[RTSP] Reconnect triggered by watchdog...")
                    self._rtsp_cap = rtsp_connect()
                    if self._rtsp_cap:
                        logger.info("[RTSP] Stream restored")
                        self._rtsp_stream_alive = True
                        self._rtsp_last_frame_time = time.time()
                        consecutive_fails = 0
                    else:
                        logger.warning("[RTSP] Reconnect failed, retry in 10s")
                        time.sleep(10)
                        continue
                    self._rtsp_reconnecting = False

                grabbed = self._rtsp_cap.grab()
                if grabbed:
                    ret, frame = self._rtsp_cap.retrieve()
                    if ret and frame is not None:
                        frame = cv2.resize(frame, (self.PREVIEW_W, self.PREVIEW_H))
                        with self._frame_lock:
                            self._latest_frame = frame
                        self._rtsp_last_frame_time = time.time()
                        consecutive_fails = 0
                        if not self._rtsp_stream_alive:
                            self._rtsp_stream_alive = True
                            logger.info("[RTSP] Stream alive")
                    else:
                        consecutive_fails += 1
                else:
                    consecutive_fails += 1
                    time.sleep(0.1)

                # Local failure detection (parallel zu Watchdog)
                if consecutive_fails > 50:  # 5 Sekunden
                    logger.warning("[RTSP] Local failure detection - trigger reconnect")
                    self._rtsp_stream_alive = False
                    self._rtsp_reconnecting = True
                    consecutive_fails = 0

            if self._rtsp_cap:
                self._rtsp_cap.release()

        def rtsp_watchdog():
            """Watchdog: Stream-Health pruefen alle 5s."""
            fail_count = 0
            while self.running:
                time.sleep(5)
                if not self.running:
                    break

                time_since_frame = time.time() - self._rtsp_last_frame_time

                if time_since_frame > 5 and self._rtsp_stream_alive:
                    fail_count += 1
                    logger.warning(f"[RTSP Watchdog] No frames for {time_since_frame:.1f}s (fail {fail_count}/3)")

                    if fail_count >= 3:
                        logger.error("[RTSP Watchdog] Stream DEAD - triggering reconnect")
                        self._rtsp_stream_alive = False
                        self._rtsp_reconnecting = True
                        fail_count = 0
                else:
                    fail_count = 0

        threading.Thread(target=rtsp_reader, daemon=True, name="RTSPReader").start()
        threading.Thread(target=rtsp_watchdog, daemon=True, name="RTSPWatchdog").start()'''

if old_rtsp in code:
    code = code.replace(old_rtsp, new_rtsp)
    print('FIX 2: RTSP Reader + Watchdog mit Auto-Reconnect - OK')
    fixes += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

# FIX 3: Stream-Status in Status-JSON schreiben
old_status_json = '''            status = {
                "scrfd_active": self.scrfd_active,
                "arcface_active": self.arcface_active,
                "yolo_active": self.yolo_active,
                "pose_active": self.pose_active,
                "hand_active": self.hand_active,
                "npu_paused": self._npu_paused,
                "active_models": list(self._active_ctx.keys()),
                "autonomous_mode": self._autonomous_mode,
                "moloch_has_control": self._moloch_has_control,
                "tentakel_enabled": getattr(self, '_tentakel_enabled', False),'''

new_status_json = '''            status = {
                "scrfd_active": self.scrfd_active,
                "arcface_active": self.arcface_active,
                "yolo_active": self.yolo_active,
                "pose_active": self.pose_active,
                "hand_active": self.hand_active,
                "npu_paused": self._npu_paused,
                "active_models": list(self._active_ctx.keys()),
                "autonomous_mode": self._autonomous_mode,
                "moloch_has_control": self._moloch_has_control,
                "tentakel_enabled": getattr(self, '_tentakel_enabled', False),
                "rtsp_alive": self._rtsp_stream_alive,
                "rtsp_reconnecting": self._rtsp_reconnecting,'''

if old_status_json in code:
    code = code.replace(old_status_json, new_status_json)
    print('FIX 3: RTSP Status in Status-JSON - OK')
    fixes += 1
else:
    print('FIX 3: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

compile(open(svc).read(), svc, 'exec')
print(f'\n{fixes} Fixes. Syntax OK.')
print('\nNaechster Schritt: Panel-Indikator hinzufuegen (separate fix_rtsp_panel_indicator.py)')
