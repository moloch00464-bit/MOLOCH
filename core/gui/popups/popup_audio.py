#!/usr/bin/env python3
"""
M.O.L.O.C.H. Audio Popup — ReSpeaker Lite
============================================

Eigenstaendiges Toplevel-Fenster fuer Audio-Einstellungen.
ReSpeaker Lite Voice Assistant Kit per USB am Pi5.
Audio laeuft ueber PipeWire/WirePlumber.

Features:
- ReSpeaker Source ID via wpctl status
- Mic Gain Slider (0.0 - 3.0) mit wpctl set-volume
- Noise Gate Slider (-80 bis -20 dB)
- AGC Checkbox
- VU Meter (200px Canvas, 100ms Update, pw-record PCM)
- MIC TEST (3s Aufnahme + Wiedergabe mit Countdown)
- SAVE Button (settings persistent speichern)

Importiert NUR panel_styles und tkinter.
"""

import logging
import math
import os
import signal
import struct
import subprocess
import threading
import time
import tkinter as tk

from core.gui.panel_styles import (
    BG_DARK, BG_FRAME, BG_BUTTON, BG_INPUT,
    FG_WHITE, FG_LABEL, FG_DIM,
    ACCENT_GREEN, ACCENT_CYAN,
    STATUS_GREEN, STATUS_YELLOW, STATUS_RED,
    FONT_TITLE, FONT_LABEL, FONT_BUTTON, FONT_SMALL,
)

logger = logging.getLogger("moloch.popup_audio")

# VU Meter Konstanten
VU_UPDATE_MS = 100
VU_WIDTH = 200
VU_HEIGHT = 18
VU_CHUNK_SIZE = 3200  # 100ms @ 16kHz 16bit mono


class AudioPopup:
    """Audio Settings als eigenstaendiges Toplevel-Fenster."""

    def __init__(self, parent, service_proxy):
        """
        Args:
            parent: Parent-Widget (fuer Toplevel)
            service_proxy: ServiceProxy Instanz fuer Commands/Status
        """
        self.parent = parent
        self.service = service_proxy

        # ReSpeaker PipeWire Node-ID (gecacht)
        self._respeaker_source_id = None

        # VU Monitor State
        self._vu_process = None
        self._vu_monitor_running = False
        self._vu_after_id = None
        self._current_rms_db = -80.0

        # Mic Test State
        self._mic_test_running = False
        self._countdown_after_id = None

        # Flag: User hat Slider angefasst -> poll_status darf nicht ueberschreiben
        self._user_touched_gain = False
        self._user_touched_gate = False

        # Toplevel erstellen
        self.win = tk.Toplevel(parent)
        self.win.title("Audio \u2014 ReSpeaker Lite")
        self.win.configure(bg=BG_DARK)
        self.win.geometry("400x520")
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Variablen
        self._gain_var = tk.DoubleVar(value=1.0)
        self._noise_gate_var = tk.DoubleVar(value=-60.0)
        self._agc_var = tk.BooleanVar(value=False)

        # GUI aufbauen
        self._build_title()
        self._build_gain_slider()
        self._build_agc_checkbox()
        self._build_noise_gate_slider()
        self._build_vu_meter()
        self._build_mic_test()
        self._build_save_button()

        # ReSpeaker suchen und aktuelle Werte laden
        self._find_respeaker_source_id()
        self._load_current_values()

        # VU Monitor starten
        self._start_vu_monitor()

    # =========================================================================
    # ReSpeaker Source ID finden
    # =========================================================================

    def _find_respeaker_source_id(self):
        """ReSpeaker PipeWire Source Node-ID via wpctl status finden."""
        if self._respeaker_source_id:
            return self._respeaker_source_id
        try:
            result = subprocess.run(
                ["wpctl", "status"], capture_output=True, text=True, timeout=5)
            in_sources = False
            for line in result.stdout.splitlines():
                if "Sources:" in line or "Quellen:" in line:
                    in_sources = True
                    continue
                if in_sources and ("Sinks:" in line or "Senken:" in line):
                    break
                if in_sources and "ReSpeaker" in line and "Analog" in line:
                    # Format: "  *   59. ReSpeaker Lite Analog Stereo"
                    parts = line.strip().lstrip("*").strip().split(".")
                    if parts:
                        node_id = parts[0].strip()
                        if node_id.isdigit():
                            self._respeaker_source_id = node_id
                            logger.info(f"[AUDIO] ReSpeaker source ID: {node_id}")
                            return node_id
        except Exception as e:
            logger.error(f"[AUDIO] wpctl status failed: {e}")
        return None

    # =========================================================================
    # Titel
    # =========================================================================

    def _build_title(self):
        """Titel-Label oben."""
        tk.Label(
            self.win, text="Audio \u2014 ReSpeaker Lite",
            bg=BG_DARK, fg=FG_WHITE, font=FONT_TITLE,
        ).pack(pady=(10, 5))

    # =========================================================================
    # Mic Gain Slider (0.0 - 3.0)
    # =========================================================================

    def _build_gain_slider(self):
        """Horizontaler Slider fuer Mic Gain (0.0 - 3.0)."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=15, pady=5)

        row = tk.Frame(frame, bg=BG_DARK)
        row.pack(fill=tk.X)

        tk.Label(
            row, text="Mic Gain:", bg=BG_DARK, fg=FG_LABEL, font=FONT_LABEL,
        ).pack(side=tk.LEFT)

        self._gain_label = tk.Label(
            row, text="1.00", width=5,
            bg=BG_DARK, fg=STATUS_YELLOW, font=FONT_LABEL,
        )
        self._gain_label.pack(side=tk.RIGHT)

        self._gain_slider = tk.Scale(
            frame, from_=0.0, to=3.0, resolution=0.01,
            orient=tk.HORIZONTAL, variable=self._gain_var,
            bg=BG_FRAME, fg=FG_WHITE, troughcolor=BG_INPUT,
            highlightthickness=0, font=FONT_SMALL,
            showvalue=False,
            command=self._on_gain_changed,
        )
        self._gain_slider.pack(fill=tk.X)

    def _on_gain_changed(self, value):
        """Gain geaendert — Label updaten und via wpctl setzen."""
        val = float(value)
        self._gain_label.config(text=f"{val:.2f}")
        self._user_touched_gain = True

        def apply():
            node_id = self._find_respeaker_source_id()
            if node_id:
                try:
                    subprocess.run(
                        ["wpctl", "set-volume", node_id, f"{val:.2f}"],
                        capture_output=True, timeout=3)
                except Exception as e:
                    logger.error(f"[AUDIO] Set gain failed: {e}")

        threading.Thread(target=apply, daemon=True).start()

    # =========================================================================
    # AGC Checkbox
    # =========================================================================

    def _build_agc_checkbox(self):
        """Checkbox fuer Automatic Gain Control."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=15, pady=5)

        self._agc_cb = tk.Checkbutton(
            frame, text="AGC (Automatic Gain Control)",
            variable=self._agc_var,
            bg=BG_DARK, fg=FG_WHITE,
            selectcolor=BG_FRAME,
            activebackground=BG_DARK,
            activeforeground=FG_WHITE,
            font=FONT_LABEL,
            command=self._on_agc_changed,
        )
        self._agc_cb.pack(anchor=tk.W)

    def _on_agc_changed(self):
        """AGC geaendert — an Service senden."""
        self.service._write_command("action", {
            "action": "set_audio",
            "agc_enabled": self._agc_var.get(),
        })

    # =========================================================================
    # Noise Gate Slider (-80 bis -20 dB)
    # =========================================================================

    def _build_noise_gate_slider(self):
        """Horizontaler Slider fuer Noise Gate (-80 bis -20 dB)."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=15, pady=5)

        row = tk.Frame(frame, bg=BG_DARK)
        row.pack(fill=tk.X)

        tk.Label(
            row, text="Noise Gate:", bg=BG_DARK, fg=FG_LABEL, font=FONT_LABEL,
        ).pack(side=tk.LEFT)

        self._noise_gate_label = tk.Label(
            row, text="-60 dB", width=7,
            bg=BG_DARK, fg=STATUS_YELLOW, font=FONT_LABEL,
        )
        self._noise_gate_label.pack(side=tk.RIGHT)

        self._noise_gate_slider = tk.Scale(
            frame, from_=-80, to=-20, resolution=1,
            orient=tk.HORIZONTAL, variable=self._noise_gate_var,
            bg=BG_FRAME, fg=FG_WHITE, troughcolor=BG_INPUT,
            highlightthickness=0, font=FONT_SMALL,
            showvalue=False,
            command=self._on_noise_gate_changed,
        )
        self._noise_gate_slider.pack(fill=tk.X)

    def _on_noise_gate_changed(self, value):
        """Noise Gate geaendert — Label updaten und an Service senden."""
        val = float(value)
        self._noise_gate_label.config(text=f"{val:.0f} dB")
        self._user_touched_gate = True
        self.service._write_command("action", {
            "action": "set_audio",
            "noise_gate_db": val,
        })

    # =========================================================================
    # VU Meter (200px Canvas, 100ms Update, pw-record PCM)
    # =========================================================================

    def _build_vu_meter(self):
        """Canvas-Balken fuer Audio-Pegel (gruen/gelb/rot) + dB Label."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=15, pady=(10, 5))

        row = tk.Frame(frame, bg=BG_DARK)
        row.pack(fill=tk.X)

        tk.Label(
            row, text="VU:", bg=BG_DARK, fg=FG_LABEL, font=FONT_LABEL,
        ).pack(side=tk.LEFT)

        self._vu_db_label = tk.Label(
            row, text="-- dB", width=8,
            bg=BG_DARK, fg=FG_DIM, font=FONT_LABEL,
        )
        self._vu_db_label.pack(side=tk.RIGHT)

        self._vu_canvas = tk.Canvas(
            frame, width=VU_WIDTH, height=VU_HEIGHT,
            bg=BG_INPUT, highlightthickness=1, highlightbackground=FG_DIM,
        )
        self._vu_canvas.pack(fill=tk.X, pady=(2, 0))

    def _start_vu_monitor(self):
        """VU Monitor starten: pw-record liest PCM, berechnet RMS."""
        if self._vu_monitor_running:
            return

        node_id = self._find_respeaker_source_id()
        if not node_id:
            logger.warning("[AUDIO] Kein ReSpeaker gefunden — VU Meter inaktiv")
            self._vu_db_label.config(text="no mic", fg=STATUS_RED)
            return

        self._vu_monitor_running = True

        def monitor():
            try:
                self._vu_process = subprocess.Popen(
                    ["pw-record", "--target", node_id,
                     "--channels", "1", "--rate", "16000",
                     "--format", "s16", "-"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

                while self._vu_monitor_running and self._vu_process.poll() is None:
                    data = self._vu_process.stdout.read(VU_CHUNK_SIZE)
                    if not data or len(data) < 4:
                        continue
                    # s16le PCM parsen
                    n_samples = len(data) // 2
                    samples = struct.unpack(f"<{n_samples}h", data[:n_samples * 2])
                    # RMS berechnen
                    rms = math.sqrt(sum(s * s for s in samples) / n_samples) if n_samples > 0 else 0
                    self._current_rms_db = 20 * math.log10(max(rms, 1) / 32768.0)
            except Exception as e:
                logger.error(f"[AUDIO] VU monitor error: {e}")
            finally:
                self._vu_monitor_running = False
                if self._vu_process:
                    try:
                        self._vu_process.terminate()
                    except Exception:
                        pass
                    self._vu_process = None

        threading.Thread(target=monitor, daemon=True).start()

        # Canvas-Update Timer starten
        self._update_vu_canvas()

    def _stop_vu_monitor(self):
        """VU Monitor stoppen."""
        self._vu_monitor_running = False
        if self._vu_process:
            try:
                self._vu_process.terminate()
                self._vu_process.wait(timeout=2)
            except Exception:
                pass
            self._vu_process = None

    def _update_vu_canvas(self):
        """VU Meter Canvas alle 100ms neu zeichnen."""
        rms_db = self._current_rms_db

        # dB auf Pixel mappen (-80dB=0, 0dB=volle Breite)
        canvas_w = self._vu_canvas.winfo_width()
        if canvas_w < 10:
            canvas_w = VU_WIDTH
        px = max(0, min(canvas_w, int((rms_db + 80) * canvas_w / 80)))

        self._vu_canvas.delete("all")
        if px > 0:
            # Farbe: gruen < -20dB, gelb -20 bis -6dB, rot > -6dB
            if rms_db < -20:
                color = STATUS_GREEN
            elif rms_db < -6:
                color = STATUS_YELLOW
            else:
                color = STATUS_RED
            self._vu_canvas.create_rectangle(
                0, 0, px, VU_HEIGHT, fill=color, outline="")

        # dB Label
        if rms_db > -79:
            if rms_db < -20:
                lbl_color = STATUS_GREEN
            elif rms_db < -6:
                lbl_color = STATUS_YELLOW
            else:
                lbl_color = STATUS_RED
            self._vu_db_label.config(text=f"{rms_db:.0f} dB", fg=lbl_color)
        else:
            self._vu_db_label.config(text="-- dB", fg=FG_DIM)

        # Naechstes Update
        self._vu_after_id = self.win.after(VU_UPDATE_MS, self._update_vu_canvas)

    # =========================================================================
    # MIC TEST (3s Aufnahme + Wiedergabe mit Countdown)
    # =========================================================================

    def _build_mic_test(self):
        """MIC TEST Button mit Status-Label."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=15, pady=(10, 5))

        self._btn_mic_test = tk.Button(
            frame, text="MIC TEST", width=14,
            bg=BG_BUTTON, fg=ACCENT_CYAN, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._on_mic_test,
        )
        self._btn_mic_test.pack()

        self._lbl_mic_status = tk.Label(
            frame, text="", bg=BG_DARK, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_mic_status.pack(pady=(3, 0))

    def _on_mic_test(self):
        """3-Sekunden Mic Test starten mit Countdown."""
        if self._mic_test_running:
            return

        node_id = self._find_respeaker_source_id()
        if not node_id:
            self._lbl_mic_status.config(text="Kein ReSpeaker!", fg=STATUS_RED)
            return

        self._mic_test_running = True
        self._btn_mic_test.config(state=tk.DISABLED)

        # VU Monitor stoppen (haelt pw-record offen!)
        vu_was_running = self._vu_monitor_running
        if vu_was_running:
            self._stop_vu_monitor()

        # Countdown starten: 3s Aufnahme
        self._mic_test_countdown(3, node_id, vu_was_running)

    def _mic_test_countdown(self, remaining, node_id, restart_vu):
        """Countdown auf dem Button waehrend Aufnahme."""
        if remaining > 0:
            self._btn_mic_test.config(text=f"REC {remaining}s")
            self._lbl_mic_status.config(text="Aufnahme laeuft...", fg=STATUS_YELLOW)

            if remaining == 3:
                # Aufnahme im Thread starten
                threading.Thread(
                    target=self._do_mic_test, args=(node_id, restart_vu),
                    daemon=True).start()

            self._countdown_after_id = self.win.after(
                1000, self._mic_test_countdown, remaining - 1, node_id, restart_vu)
        else:
            self._btn_mic_test.config(text="PLAY...")
            self._lbl_mic_status.config(text="Wiedergabe...", fg=ACCENT_CYAN)

    def _do_mic_test(self, node_id, restart_vu):
        """Aufnahme + Wiedergabe im Hintergrund-Thread."""
        test_path = "/tmp/moloch_mic_test.wav"

        # Kurz warten bis VU Monitor wirklich gestoppt
        time.sleep(0.3)

        try:
            # 3 Sekunden aufnehmen
            proc = subprocess.Popen(
                ["pw-record", "--target", node_id,
                 "--channels", "1", "--rate", "16000", test_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=3)
        except Exception as e:
            logger.error(f"[AUDIO] Mic Test Aufnahme failed: {e}")
            self.win.after(0, self._mic_test_error, f"Aufnahme-Fehler: {e}")
            return

        # Pruefen ob Aufnahme existiert
        if not os.path.exists(test_path) or os.path.getsize(test_path) < 1000:
            self.win.after(0, self._mic_test_error, "Keine Aufnahme erstellt")
            return

        # Wiedergabe
        try:
            subprocess.run(["pw-play", test_path], timeout=10, capture_output=True)
        except Exception:
            try:
                subprocess.run(["aplay", test_path], timeout=10, capture_output=True)
            except Exception as e:
                logger.error(f"[AUDIO] Mic Test Wiedergabe failed: {e}")

        # Fertig — UI updaten
        self.win.after(0, self._mic_test_done, restart_vu)

    def _mic_test_done(self, restart_vu):
        """Mic Test abgeschlossen — Button freigeben, VU wieder starten."""
        self._mic_test_running = False
        self._btn_mic_test.config(state=tk.NORMAL, text="MIC TEST")
        self._lbl_mic_status.config(text="Test abgeschlossen", fg=ACCENT_GREEN)
        self.win.after(3000, lambda: self._lbl_mic_status.config(text="", fg=FG_DIM))

        # VU Monitor wieder starten
        if restart_vu:
            self._start_vu_monitor()

    def _mic_test_error(self, msg):
        """Mic Test Fehler anzeigen."""
        self._mic_test_running = False
        self._btn_mic_test.config(state=tk.NORMAL, text="MIC TEST")
        self._lbl_mic_status.config(text=msg, fg=STATUS_RED)
        self.win.after(4000, lambda: self._lbl_mic_status.config(text="", fg=FG_DIM))

    # =========================================================================
    # SAVE Button
    # =========================================================================

    def _build_save_button(self):
        """SAVE Button — speichert Audio-Einstellungen persistent."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=15, pady=(10, 15))

        self._btn_save = tk.Button(
            frame, text="SAVE", width=14,
            bg=BG_BUTTON, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._on_save,
        )
        self._btn_save.pack()

        self._lbl_save = tk.Label(
            frame, text="", bg=BG_DARK, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_save.pack(pady=(3, 0))

    def _on_save(self):
        """Audio-Einstellungen persistent speichern."""
        self.service._write_command("action", {
            "action": "save_settings",
            "audio": {
                "mic_gain": self._gain_var.get(),
                "noise_gate_db": self._noise_gate_var.get(),
                "agc_enabled": self._agc_var.get(),
            },
        })
        self._lbl_save.config(text="Gespeichert!", fg=ACCENT_GREEN)
        self.win.after(2000, lambda: self._lbl_save.config(text="", fg=FG_DIM))

    # =========================================================================
    # Werte laden (einmalig beim Oeffnen)
    # =========================================================================

    def _load_current_values(self):
        """Aktuelle Werte vom Service lesen und Slider/Checkbox setzen."""
        status = self.service.read_status()
        if not status:
            return

        audio = status.get("audio")
        if not isinstance(audio, dict):
            return

        # Gain (0.0 - 3.0)
        raw_gain = audio.get("mic_gain")
        if raw_gain is not None and not isinstance(raw_gain, (dict, list)):
            try:
                gain = max(0.0, min(3.0, float(raw_gain)))
                self._gain_var.set(gain)
                self._gain_label.config(text=f"{gain:.2f}")
            except (TypeError, ValueError):
                pass

        # Noise Gate (-80 bis -20)
        raw_ng = audio.get("noise_gate_db")
        if raw_ng is not None and not isinstance(raw_ng, (dict, list)):
            try:
                ng = max(-80.0, min(-20.0, float(raw_ng)))
                self._noise_gate_var.set(ng)
                self._noise_gate_label.config(text=f"{ng:.0f} dB")
            except (TypeError, ValueError):
                pass

        # AGC
        raw_agc = audio.get("agc_enabled")
        if raw_agc is not None:
            try:
                self._agc_var.set(bool(raw_agc))
            except (TypeError, ValueError):
                pass

    # =========================================================================
    # Schliessen
    # =========================================================================

    def _on_close(self):
        """Fenster sauber schliessen — Timer und VU Monitor stoppen."""
        # VU Canvas Timer stoppen
        if self._vu_after_id is not None:
            self.win.after_cancel(self._vu_after_id)
            self._vu_after_id = None

        # Countdown Timer stoppen
        if self._countdown_after_id is not None:
            self.win.after_cancel(self._countdown_after_id)
            self._countdown_after_id = None

        # VU Monitor Prozess stoppen
        self._stop_vu_monitor()

        self.win.destroy()
