#!/usr/bin/env python3
"""Fix: Frozen Frame Watchdog v2 - Hash-basierte Erkennung + Reconnect.

Bug: Stream friert ein nach Kamera-Reconnect, Modelle laufen auf altem Bild.
     Watchdog greift nicht weil:
     1. _last_frame_write nur in _write_shm() aktualisiert, nicht im Reader
     2. Kein Reconnect bei grab()-Failure (nur sleep+retry mit totem cap)
     3. Thread-Leak: Watchdog startet neuen Reader ohne alten zu stoppen

Fix:
1. Hash-basierte Frozen-Frame-Erkennung (10 identische Frames = Reconnect)
2. grab()-Failure -> release + neuer VideoCapture
3. _start_rtsp() stoppt alten Reader-Thread via Event bevor neuer startet
4. _last_frame_write wird im Reader aktualisiert (nicht nur in _write_shm)
5. Log-Zeile bei jedem Reconnect
"""
import sys

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# ==========================================================================
# FIX 1: Init-Variablen fuer Hash-Detection + Thread-Stop
# ==========================================================================
old_init = """        # Frozen Frame Watchdog
        self._last_frame_write = time.time()
        self._frozen_restart_count = 0"""

new_init = """        # Frozen Frame Watchdog
        self._last_frame_write = time.time()
        self._frozen_restart_count = 0
        self._rtsp_frame_hash = None
        self._rtsp_identical_count = 0
        self._rtsp_stop_reader = threading.Event()"""

if old_init in code:
    code = code.replace(old_init, new_init)
    print('FIX 1: Init-Variablen (Hash + Stop-Event) - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')
    # Checke ob schon gepatcht
    if '_rtsp_frame_hash' in code:
        print('       -> Bereits gepatcht!')

# ==========================================================================
# FIX 2: _start_rtsp komplett ersetzen
# ==========================================================================
old_rtsp = """    def _start_rtsp(self):
        \"\"\"Starte RTSP Background Reader.\"\"\"
        def rtsp_reader():
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"
            )
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            self._rtsp_cap = cap  # Fuer Watchdog-Zugriff
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

        threading.Thread(target=rtsp_reader, daemon=True, name="RTSPReader").start()"""

new_rtsp = """    def _start_rtsp(self):
        \"\"\"Starte RTSP Background Reader (mit Frozen-Frame-Detection + Auto-Reconnect).\"\"\"
        # Alten Reader-Thread stoppen
        if hasattr(self, '_rtsp_stop_reader'):
            self._rtsp_stop_reader.set()

        # Alten Cap freigeben
        if hasattr(self, '_rtsp_cap') and self._rtsp_cap is not None:
            try:
                self._rtsp_cap.release()
            except Exception:
                pass
            self._rtsp_cap = None

        # Frische State-Variablen
        self._rtsp_stop_reader = threading.Event()
        self._rtsp_frame_hash = None
        self._rtsp_identical_count = 0
        stop_event = self._rtsp_stop_reader

        def _rtsp_connect():
            \"\"\"Neuen RTSP VideoCapture erstellen.\"\"\"
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"
            )
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        def rtsp_reader():
            cap = _rtsp_connect()
            self._rtsp_cap = cap

            if not cap.isOpened():
                self._update_status(f"RTSP FEHLER: {RTSP_URL}")
                return

            self._update_status("RTSP aktiv")
            self._last_frame_write = time.time()
            identical_count = 0
            prev_hash = None

            while self.running and not stop_event.is_set():
                grabbed = cap.grab()
                if grabbed:
                    ret, frame = cap.retrieve()
                    if ret and frame is not None:
                        frame = cv2.resize(frame, (self.PREVIEW_W, self.PREVIEW_H))

                        # Frozen Frame Detection via Hash (jeden 20. Pixel samplen)
                        frame_hash = hash(frame[::20, ::20].tobytes())
                        if frame_hash == prev_hash:
                            identical_count += 1
                            if identical_count >= 10:
                                self._frozen_restart_count += 1
                                logger.warning(
                                    f"[RTSP] {identical_count} identische Frames erkannt - "
                                    f"Stream frozen! Reconnect #{self._frozen_restart_count}..."
                                )
                                cap.release()
                                time.sleep(2)
                                cap = _rtsp_connect()
                                self._rtsp_cap = cap
                                identical_count = 0
                                prev_hash = None
                                if cap.isOpened():
                                    logger.info("[RTSP] Stream wiederhergestellt (frozen-detect)")
                                    self._last_frame_write = time.time()
                                else:
                                    logger.warning("[RTSP] Reconnect fehlgeschlagen, retry...")
                                continue
                        else:
                            prev_hash = frame_hash
                            identical_count = 0

                        with self._frame_lock:
                            self._latest_frame = frame
                        self._last_frame_write = time.time()
                    else:
                        time.sleep(0.05)
                else:
                    # grab() fehlgeschlagen - Stream offline
                    self._frozen_restart_count += 1
                    logger.warning(
                        f"[RTSP] grab() fehlgeschlagen - Reconnect "
                        f"#{self._frozen_restart_count} in 2s..."
                    )
                    cap.release()
                    time.sleep(2)
                    cap = _rtsp_connect()
                    self._rtsp_cap = cap
                    identical_count = 0
                    prev_hash = None
                    if cap.isOpened():
                        logger.info("[RTSP] Stream wiederhergestellt (grab-fail)")
                        self._last_frame_write = time.time()
                    else:
                        logger.warning("[RTSP] Reconnect fehlgeschlagen, retry in 5s...")
                        time.sleep(5)

            cap.release()

        threading.Thread(target=rtsp_reader, daemon=True, name="RTSPReader").start()"""

if old_rtsp in code:
    code = code.replace(old_rtsp, new_rtsp)
    print('FIX 2: _start_rtsp mit Hash-Detection + Auto-Reconnect - OK')
    fixes += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

# ==========================================================================
# FIX 3: _frozen_frame_watchdog - Thread-Leak fixen
#         (Kein manuelles cap.release mehr, _start_rtsp handelt das jetzt)
# ==========================================================================
old_watchdog = """    def _frozen_frame_watchdog(self):
        \"\"\"Erkennt eingefrorene Frames und startet RTSP Stream neu.\"\"\"
        while self.running:
            try:
                time.sleep(10)  # Alle 10 Sekunden pruefen

                frame_age = time.time() - self._last_frame_write

                if frame_age > 30:  # Frame aelter als 30 Sekunden
                    self._frozen_restart_count += 1
                    logger.warning(
                        f"[WATCHDOG] Frame eingefroren seit {frame_age:.0f}s! "
                        f"Restart #{self._frozen_restart_count}"
                    )

                    # RTSP Stream neu verbinden
                    try:
                        if hasattr(self, '_rtsp_cap') and self._rtsp_cap is not None:
                            try:
                                self._rtsp_cap.release()
                            except Exception:
                                pass
                        self._start_rtsp()
                        logger.info("[WATCHDOG] RTSP Stream neu gestartet")
                        self._last_frame_write = time.time()
                    except Exception as e:
                        logger.error(f"[WATCHDOG] RTSP Reconnect Error: {e}")

                    # Max 5 Versuche, danach loggen und warten
                    if self._frozen_restart_count >= 5:
                        logger.error("[WATCHDOG] 5 Reconnects fehlgeschlagen, warte 60s")
                        time.sleep(60)
                        self._frozen_restart_count = 0

            except Exception as e:
                logger.error(f"[WATCHDOG] Error: {e}")"""

new_watchdog = """    def _frozen_frame_watchdog(self):
        \"\"\"Backup-Watchdog: Erkennt wenn Reader-Thread kein Frame mehr liefert.

        Primaere Detection ist jetzt Hash-basiert im Reader selbst (10 identische
        Frames -> Reconnect). Dieser Watchdog ist Fallback fuer den Fall dass
        der Reader-Thread komplett haengt oder crashed.
        \"\"\"
        while self.running:
            try:
                time.sleep(10)

                frame_age = time.time() - self._last_frame_write

                if frame_age > 30:
                    logger.warning(
                        f"[WATCHDOG] Kein neuer Frame seit {frame_age:.0f}s! "
                        f"Reader haengt - starte RTSP komplett neu..."
                    )

                    try:
                        self._start_rtsp()
                        logger.info("[WATCHDOG] RTSP Stream neu gestartet")
                    except Exception as e:
                        logger.error(f"[WATCHDOG] RTSP Reconnect Error: {e}")

                    # Max 5 Versuche, danach loggen und warten
                    if self._frozen_restart_count >= 5:
                        logger.error("[WATCHDOG] 5 Reconnects fehlgeschlagen, warte 60s")
                        time.sleep(60)
                        self._frozen_restart_count = 0

            except Exception as e:
                logger.error(f"[WATCHDOG] Error: {e}")"""

if old_watchdog in code:
    code = code.replace(old_watchdog, new_watchdog)
    print('FIX 3: Watchdog Thread-Leak Fix (kein manuelles cap.release) - OK')
    fixes += 1
else:
    print('FIX 3: ANCHOR NOT FOUND!')

# ==========================================================================
# Schreiben + Syntax-Check
# ==========================================================================
if fixes == 0:
    print('\nKEINE FIXES ANGEWENDET! Abbruch.')
    sys.exit(1)

with open(svc, 'w') as f:
    f.write(code)

compile(open(svc).read(), svc, 'exec')
print(f'\n{fixes}/3 Fixes angewendet. Syntax OK.')
print('Neustart: sudo systemctl restart moloch')
