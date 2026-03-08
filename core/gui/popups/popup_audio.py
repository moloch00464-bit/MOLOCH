#!/usr/bin/env python3
"""
M.O.L.O.C.H. Audio Popup — WiFi-Mic + ReSpeaker Lite
======================================================

Eigenstaendiges Toplevel-Fenster fuer Audio-Einstellungen.
Zeigt ESP32 WiFi-Mic Status UND USB ReSpeaker Lite (PipeWire).

WiFi-Mic Sektion (oben):
- ESP32 Verbindungsstatus: IP, Ping, Latenz
- Audio-Source Anzeige: ESP32 WiFi oder USB ReSpeaker
- Samplerate-Umschaltung 16kHz / 48kHz via HTTP
- Mikrofon-Pegel aus ESP32 Status
- Health/Buffer Status

USB ReSpeaker Sektion (unten):
- Mic Gain Slider (wpctl), AGC, Noise Gate
- VU Meter (pw-record PCM), MIC TEST

Update nur wenn Popup offen, max 2 Hz Refresh.
Importiert NUR panel_styles und tkinter.
"""

import json
import logging
import math
import os
import signal
import struct
import subprocess
import tempfile
import threading
import time
import tkinter as tk
from urllib.request import urlopen, Request
from urllib.error import URLError

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

# WiFi-Mic Konstanten
ESP32_IP = "10.42.0.2"
ESP32_BASE_URL = f"http://{ESP32_IP}"
WIFI_UPDATE_MS = 500  # 2 Hz Refresh

# Settings-Pfad
SETTINGS_PATH = os.path.expanduser("~/moloch/config/settings.json")


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

        # Debounced Save Timer
        self._save_after_id = None

        # WiFi-Mic State
        self._wifi_after_id = None
        self._wifi_status = {}  # Letzter ESP32 Status
        self._wifi_ping_ms = -1.0  # Letzte Ping-Latenz (-1 = offline)
        self._wifi_connected = False
        self._wifi_poll_running = False
        self._current_samplerate = 16000
        self._wifi_mic_ref = None  # WiFiMic Singleton fuer Buffer/Amplitude
        try:
            from core.audio.wifi_mic import get_wifi_mic
            self._wifi_mic_ref = get_wifi_mic()
        except Exception:
            pass

        # Toplevel erstellen
        self.win = tk.Toplevel(parent)
        self.win.attributes('-topmost', True)
        self.win.transient(parent)
        self.win.title("Audio \u2014 WiFi-Mic + ReSpeaker")
        self.win.configure(bg=BG_DARK)
        self.win.geometry("420x740")
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Variablen
        self._gain_var = tk.DoubleVar(value=1.0)
        self._noise_gate_var = tk.DoubleVar(value=-60.0)
        self._agc_var = tk.BooleanVar(value=False)

        # GUI aufbauen — WiFi-Mic oben, USB unten
        self._build_wifi_section()
        self._build_separator()
        self._build_status_label()
        self._build_gain_slider()
        self._build_agc_checkbox()
        self._build_noise_gate_slider()
        self._build_vu_meter()
        self._build_mic_test()

        # ReSpeaker suchen und Status-Label updaten
        self._find_respeaker_source_id()
        self._update_status_label()

        # Aktuelle Werte laden
        self._load_current_values()

        # VU Monitor starten
        self._start_vu_monitor()

        # WiFi-Mic Status-Poll starten (2 Hz, nur bei offenem Popup)
        self._start_wifi_poll()

    # =========================================================================
    # WiFi-Mic Sektion (ESP32-S3 via HTTP/Ping)
    # =========================================================================

    def _build_wifi_section(self):
        """WiFi-Mic Status-Sektion oben im Popup."""
        # Titel
        tk.Label(
            self.win, text="ESP32 WiFi-Mic",
            bg=BG_DARK, fg=ACCENT_CYAN, font=FONT_TITLE,
        ).pack(pady=(10, 3))

        container = tk.Frame(self.win, bg=BG_FRAME, bd=1, relief=tk.GROOVE)
        container.pack(fill=tk.X, padx=12, pady=(0, 5))

        # Zeile 1: Verbindungsstatus + IP + Latenz
        row1 = tk.Frame(container, bg=BG_FRAME)
        row1.pack(fill=tk.X, padx=8, pady=(6, 2))

        self._wifi_conn_dot = tk.Label(
            row1, text="\u25cf", bg=BG_FRAME, fg=STATUS_RED, font=FONT_LABEL,
        )
        self._wifi_conn_dot.pack(side=tk.LEFT)

        self._wifi_conn_label = tk.Label(
            row1, text=f"  {ESP32_IP}  --  Offline",
            bg=BG_FRAME, fg=FG_DIM, font=FONT_LABEL,
        )
        self._wifi_conn_label.pack(side=tk.LEFT)

        self._wifi_ping_label = tk.Label(
            row1, text="-- ms",
            bg=BG_FRAME, fg=FG_DIM, font=FONT_LABEL,
        )
        self._wifi_ping_label.pack(side=tk.RIGHT)

        # Zeile 2: Audio-Source + Samplerate
        row2 = tk.Frame(container, bg=BG_FRAME)
        row2.pack(fill=tk.X, padx=8, pady=2)

        tk.Label(
            row2, text="Source:", bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        ).pack(side=tk.LEFT)

        self._wifi_source_label = tk.Label(
            row2, text="--", bg=BG_FRAME, fg=FG_DIM, font=FONT_LABEL,
        )
        self._wifi_source_label.pack(side=tk.LEFT, padx=(5, 0))

        # Samplerate Button
        self._btn_samplerate = tk.Button(
            row2, text="16 kHz", width=8,
            bg=BG_BUTTON, fg=ACCENT_CYAN, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._on_toggle_samplerate,
        )
        self._btn_samplerate.pack(side=tk.RIGHT)

        tk.Label(
            row2, text="Rate:", bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        ).pack(side=tk.RIGHT, padx=(0, 5))

        # Zeile 3: Streaming + RSSI + Uptime
        row3 = tk.Frame(container, bg=BG_FRAME)
        row3.pack(fill=tk.X, padx=8, pady=2)

        self._wifi_stream_label = tk.Label(
            row3, text="Stream: --",
            bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._wifi_stream_label.pack(side=tk.LEFT)

        self._wifi_rssi_label = tk.Label(
            row3, text="RSSI: --",
            bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._wifi_rssi_label.pack(side=tk.RIGHT)

        # Zeile 4: Health + FW Version
        row4 = tk.Frame(container, bg=BG_FRAME)
        row4.pack(fill=tk.X, padx=8, pady=(2, 6))

        self._wifi_health_label = tk.Label(
            row4, text="Health: --",
            bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._wifi_health_label.pack(side=tk.LEFT)

        self._wifi_fw_label = tk.Label(
            row4, text="FW: --",
            bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._wifi_fw_label.pack(side=tk.RIGHT)

        # Zeile 5: Buffer-Fuellstand + Pegel-Balken
        row5 = tk.Frame(container, bg=BG_FRAME)
        row5.pack(fill=tk.X, padx=8, pady=(2, 2))

        self._wifi_buf_label = tk.Label(
            row5, text="Buf: --",
            bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._wifi_buf_label.pack(side=tk.LEFT)

        self._wifi_pegel_label = tk.Label(
            row5, text="-- dB",
            bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._wifi_pegel_label.pack(side=tk.RIGHT)

        # Zeile 6: Amplitude-Balken (Canvas)
        self._wifi_vu_canvas = tk.Canvas(
            container, width=180, height=12,
            bg=BG_INPUT, highlightthickness=0,
        )
        self._wifi_vu_canvas.pack(padx=8, pady=(0, 6))

    def _build_separator(self):
        """Trennlinie zwischen WiFi und USB Sektion."""
        sep = tk.Frame(self.win, bg=FG_DIM, height=1)
        sep.pack(fill=tk.X, padx=15, pady=5)

        tk.Label(
            self.win, text="USB ReSpeaker (PipeWire)",
            bg=BG_DARK, fg=FG_LABEL, font=FONT_SMALL,
        ).pack(pady=(0, 3))

    # =========================================================================
    # WiFi-Mic Poll (2 Hz, nur bei offenem Popup)
    # =========================================================================

    def _start_wifi_poll(self):
        """WiFi-Mic Status-Abfrage starten (Ping + HTTP, 2 Hz)."""
        self._wifi_poll_running = True
        self._do_wifi_poll()

    def _do_wifi_poll(self):
        """Einen WiFi-Poll-Zyklus starten (Background-Thread)."""
        if not self._wifi_poll_running:
            return

        def poll_thread():
            ping_ms = self._ping_esp32()
            status = {}
            if ping_ms >= 0:
                status = self._fetch_esp32_status()
            # Ergebnis an Tkinter Main-Thread uebergeben
            try:
                self.win.after(0, self._update_wifi_ui, ping_ms, status)
            except Exception:
                pass  # Fenster schon geschlossen

        threading.Thread(target=poll_thread, daemon=True).start()

    def _ping_esp32(self) -> float:
        """Ping an ESP32, gibt Latenz in ms zurueck (-1 bei Timeout)."""
        try:
            result = subprocess.run(
                ["ping", "-c1", "-W1", ESP32_IP],
                capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                # "time=1.23 ms" aus der Ausgabe parsen
                for part in result.stdout.split():
                    if part.startswith("time="):
                        return float(part.split("=")[1])
        except Exception:
            pass
        return -1.0

    def _fetch_esp32_status(self) -> dict:
        """HTTP GET /audio/status vom ESP32 holen."""
        try:
            req = Request(f"{ESP32_BASE_URL}/audio/status")
            with urlopen(req, timeout=1.5) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return {}

    def _update_wifi_ui(self, ping_ms: float, status: dict):
        """WiFi-Mic UI-Elemente updaten (wird im Main-Thread aufgerufen)."""
        self._wifi_ping_ms = ping_ms
        if status:
            self._wifi_status = status
        self._wifi_connected = ping_ms >= 0 and bool(status)

        # Verbindungsstatus
        if self._wifi_connected:
            self._wifi_conn_dot.config(fg=STATUS_GREEN)
            self._wifi_conn_label.config(
                text=f"  {ESP32_IP}  --  Verbunden",
                fg=STATUS_GREEN,
            )
            self._wifi_ping_label.config(
                text=f"{ping_ms:.1f} ms", fg=STATUS_GREEN,
            )
        elif ping_ms >= 0:
            # Ping OK, HTTP fehlgeschlagen
            self._wifi_conn_dot.config(fg=STATUS_YELLOW)
            self._wifi_conn_label.config(
                text=f"  {ESP32_IP}  --  HTTP Fehler",
                fg=STATUS_YELLOW,
            )
            self._wifi_ping_label.config(
                text=f"{ping_ms:.1f} ms", fg=STATUS_YELLOW,
            )
        else:
            self._wifi_conn_dot.config(fg=STATUS_RED)
            self._wifi_conn_label.config(
                text=f"  {ESP32_IP}  --  Offline",
                fg=STATUS_RED,
            )
            self._wifi_ping_label.config(text="-- ms", fg=FG_DIM)

        # Audio-Source Anzeige
        if self._wifi_connected:
            streaming = status.get("streaming", False)
            if streaming:
                self._wifi_source_label.config(
                    text="ESP32 WiFi", fg=STATUS_GREEN,
                )
            else:
                self._wifi_source_label.config(
                    text="ESP32 (gestoppt)", fg=STATUS_YELLOW,
                )
        else:
            self._wifi_source_label.config(
                text="USB ReSpeaker (Fallback)", fg=STATUS_YELLOW,
            )

        # Samplerate Button aktualisieren
        if status:
            rate = status.get("rate", 16000)
            self._current_samplerate = rate
            rate_text = f"{rate // 1000} kHz"
            mode = status.get("mode", "")
            self._btn_samplerate.config(text=rate_text)

        # Streaming + RSSI
        if status:
            streaming = status.get("streaming", False)
            mode = status.get("mode", "--")
            self._wifi_stream_label.config(
                text=f"Stream: {'AN' if streaming else 'AUS'} ({mode})",
                fg=STATUS_GREEN if streaming else STATUS_YELLOW,
            )
            rssi = status.get("wifi_rssi", 0)
            rssi_color = STATUS_GREEN if rssi > -60 else (
                STATUS_YELLOW if rssi > -75 else STATUS_RED)
            self._wifi_rssi_label.config(
                text=f"RSSI: {rssi} dBm", fg=rssi_color,
            )
        else:
            self._wifi_stream_label.config(text="Stream: --", fg=FG_DIM)
            self._wifi_rssi_label.config(text="RSSI: --", fg=FG_DIM)

        # Health + FW
        if status:
            uptime = status.get("uptime_s", 0)
            heap = status.get("free_heap", 0)
            # Uptime formatieren
            if uptime >= 3600:
                up_str = f"{uptime // 3600}h{(uptime % 3600) // 60}m"
            elif uptime >= 60:
                up_str = f"{uptime // 60}m{uptime % 60}s"
            else:
                up_str = f"{uptime}s"
            self._wifi_health_label.config(
                text=f"Up: {up_str}  Heap: {heap // 1024}kB",
                fg=STATUS_GREEN if heap > 50000 else STATUS_YELLOW,
            )
            fw = status.get("fw_version", "--")
            self._wifi_fw_label.config(text=f"FW: v{fw}", fg=FG_LABEL)
        else:
            self._wifi_health_label.config(text="Health: --", fg=FG_DIM)
            self._wifi_fw_label.config(text="FW: --", fg=FG_DIM)

        # Buffer-Fuellstand + Amplitude aus WiFi-Mic Singleton
        self._update_wifi_buffer_and_level()

        # Naechster Poll
        if self._wifi_poll_running:
            self._wifi_after_id = self.win.after(WIFI_UPDATE_MS, self._do_wifi_poll)

    def _update_wifi_buffer_and_level(self):
        """Buffer-Fuellstand und Live-Amplitude aus WiFi-Mic aktualisieren."""
        if not self._wifi_mic_ref:
            self._wifi_buf_label.config(text="Buf: n/a", fg=FG_DIM)
            self._wifi_pegel_label.config(text="-- dB", fg=FG_DIM)
            self._wifi_vu_canvas.delete("all")
            return

        try:
            mic_status = self._wifi_mic_ref.get_status()
            buf_bytes = mic_status.get("buf_16k_bytes", 0)
            buf_pct = min(100, int(buf_bytes / 640))  # 64000 max = 100%
            connected = mic_status.get("connected_16k", False)

            if connected:
                self._wifi_buf_label.config(
                    text=f"Buf: {buf_bytes}B ({buf_pct}%)",
                    fg=STATUS_GREEN if buf_pct < 80 else STATUS_YELLOW,
                )
            else:
                self._wifi_buf_label.config(text="Buf: --", fg=FG_DIM)
                self._wifi_pegel_label.config(text="-- dB", fg=FG_DIM)
                self._wifi_vu_canvas.delete("all")
                return

            # Amplitude: kurzen Chunk lesen (nicht-destruktiv via peek)
            # Wir lesen 10ms (320 Bytes) fuer RMS-Berechnung
            chunk = self._wifi_mic_ref.get_audio_chunk(rate=16000, duration_ms=10)
            if len(chunk) >= 4:
                n_samples = len(chunk) // 2
                samples = struct.unpack(f"<{n_samples}h", chunk[:n_samples * 2])
                rms = math.sqrt(sum(s * s for s in samples) / n_samples)
                rms_db = 20 * math.log10(max(rms, 1) / 32768.0)
            else:
                rms_db = -80.0

            # Pegel-Label
            if rms_db > -79:
                if rms_db < -20:
                    lbl_color = STATUS_GREEN
                elif rms_db < -6:
                    lbl_color = STATUS_YELLOW
                else:
                    lbl_color = STATUS_RED
                self._wifi_pegel_label.config(
                    text=f"{rms_db:.0f} dB", fg=lbl_color)
            else:
                self._wifi_pegel_label.config(text="-- dB", fg=FG_DIM)

            # Amplitude-Balken zeichnen
            canvas_w = self._wifi_vu_canvas.winfo_width()
            if canvas_w < 10:
                canvas_w = 180
            px = max(0, min(canvas_w, int((rms_db + 80) * canvas_w / 80)))
            self._wifi_vu_canvas.delete("all")
            if px > 0:
                if rms_db < -20:
                    color = STATUS_GREEN
                elif rms_db < -6:
                    color = STATUS_YELLOW
                else:
                    color = STATUS_RED
                self._wifi_vu_canvas.create_rectangle(
                    0, 0, px, 12, fill=color, outline="")

        except Exception:
            pass  # Polling darf nicht sterben

    # =========================================================================
    # Samplerate umschalten (16kHz / 48kHz)
    # =========================================================================

    def _on_toggle_samplerate(self):
        """Samplerate zwischen 16kHz und 48kHz umschalten via HTTP POST."""
        new_rate = 48000 if self._current_samplerate == 16000 else 16000
        self._btn_samplerate.config(text="...", state=tk.DISABLED)

        def switch_thread():
            success = False
            try:
                req = Request(
                    f"{ESP32_BASE_URL}/audio/mode?rate={new_rate}",
                    method="POST",
                )
                with urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        success = True
            except Exception as e:
                logger.error(f"[AUDIO] Samplerate switch failed: {e}")

            def update_ui():
                if success:
                    self._current_samplerate = new_rate
                    self._btn_samplerate.config(
                        text=f"{new_rate // 1000} kHz",
                        state=tk.NORMAL,
                    )
                    # In settings.json persistieren
                    self._save_audio_settings()
                else:
                    self._btn_samplerate.config(
                        text=f"{self._current_samplerate // 1000} kHz",
                        state=tk.NORMAL,
                    )

            try:
                self.win.after(0, update_ui)
            except Exception:
                pass

        threading.Thread(target=switch_thread, daemon=True).start()

    # =========================================================================
    # ReSpeaker Source ID finden
    # =========================================================================

    def _find_respeaker_source_id(self):
        """ReSpeaker PipeWire Source Node-ID via wpctl status finden.

        Parst die Sources-Sektion und bevorzugt die aktive (*) Source.
        Format: "  *   59. ReSpeaker Lite Analog Stereo  [vol: 1.00]"
        """
        if self._respeaker_source_id:
            return self._respeaker_source_id
        try:
            result = subprocess.run(
                ["wpctl", "status"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                logger.error("[AUDIO] wpctl status returncode != 0")
                return None

            in_sources = False
            active_id = None
            fallback_id = None

            # Sektions-Keywords die Sources beenden
            section_ends = (
                "endpoints:", "Sinks:", "Senken:", "Streams:",
                "Filters:", "Devices:", "Ger\u00e4te:",
            )

            for line in result.stdout.splitlines():
                # Tree-Zeichen entfernen fuer sauberes Parsen
                clean = line.replace("\u2502", "").replace("\u251c", "") \
                            .replace("\u2500", "").replace("\u2514", "") \
                            .replace("\u2502", "").replace("\u251c", "") \
                            .replace("\u2500", "").replace("\u2514", "")

                stripped = clean.strip()

                # Sources-Sektion erkennen
                if "Sources:" in stripped or "Quellen:" in stripped:
                    in_sources = True
                    continue

                # Ende der Sources-Sektion
                if in_sources and stripped:
                    if any(kw in stripped for kw in section_ends):
                        break

                if not in_sources:
                    continue

                if "ReSpeaker" not in line:
                    continue

                # Aktive Source hat * vor der Node-ID
                is_active = "*" in line.split("ReSpeaker")[0]

                # Node-ID: Zahl vor dem ersten Punkt
                # "  *   59. ReSpeaker Lite Analog Stereo" -> 59
                parts = stripped.lstrip("*").strip().split(".")
                if parts and parts[0].strip().isdigit():
                    node_id = parts[0].strip()
                    if is_active:
                        active_id = node_id
                    elif fallback_id is None:
                        fallback_id = node_id

            # Aktive Source bevorzugen
            chosen = active_id or fallback_id
            if chosen:
                self._respeaker_source_id = chosen
                logger.info(
                    f"[AUDIO] ReSpeaker source ID: {chosen}"
                    f" (aktiv={'ja' if active_id else 'nein'})")
                return chosen

            logger.warning("[AUDIO] ReSpeaker in wpctl status nicht gefunden")
        except Exception as e:
            logger.error(f"[AUDIO] wpctl status failed: {e}")
        return None

    # =========================================================================
    # Status-Label (USB ReSpeaker)
    # =========================================================================

    def _build_status_label(self):
        """Status-Label: zeigt ReSpeaker Node oder Fehler."""
        self._status_label = tk.Label(
            self.win, text="Suche ReSpeaker...",
            bg=BG_DARK, fg=FG_DIM, font=FONT_LABEL,
        )
        self._status_label.pack(pady=(3, 3))

    def _update_status_label(self):
        """Status-Label nach Erkennung updaten."""
        if self._respeaker_source_id:
            self._status_label.config(
                text=f"ReSpeaker Lite (Node {self._respeaker_source_id})",
                fg=STATUS_GREEN,
            )
        else:
            self._status_label.config(
                text="Kein USB-Mikrofon",
                fg=STATUS_RED,
            )

    # =========================================================================
    # Mic Gain Slider (0.0 - 3.0)
    # =========================================================================

    def _build_gain_slider(self):
        """Horizontaler Slider fuer Mic Gain (0.0 - 3.0)."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=15, pady=3)

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

        # Hinweis falls wpctl-Wert abweicht
        self._wpctl_hint = tk.Label(
            frame, text="", bg=BG_DARK, fg=FG_DIM, font=FONT_SMALL,
        )
        self._wpctl_hint.pack(anchor=tk.W)

    def _on_gain_changed(self, value):
        """Gain geaendert — Label updaten, wpctl setzen, persistent speichern."""
        val = float(value)
        self._gain_label.config(text=f"{val:.2f}")
        # wpctl-Hinweis zuruecksetzen wenn User Slider bewegt
        self._wpctl_hint.config(text="")

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
        self._save_audio_settings()

    # =========================================================================
    # AGC Checkbox
    # =========================================================================

    def _build_agc_checkbox(self):
        """Checkbox fuer Automatic Gain Control."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=15, pady=3)

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
        """AGC geaendert — sofort an Service senden und persistent speichern."""
        self.service._write_command("action", {
            "action": "set_audio",
            "agc_enabled": self._agc_var.get(),
        })
        self._save_audio_settings()

    # =========================================================================
    # Noise Gate Slider (-80 bis -20 dB)
    # =========================================================================

    def _build_noise_gate_slider(self):
        """Horizontaler Slider fuer Noise Gate (-80 bis -20 dB)."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=15, pady=3)

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
        """Noise Gate geaendert — Label updaten, Service senden, persistent speichern."""
        val = float(value)
        self._noise_gate_label.config(text=f"{val:.0f} dB")
        self.service._write_command("action", {
            "action": "set_audio",
            "noise_gate_db": val,
        })
        self._save_audio_settings()

    # =========================================================================
    # VU Meter (200px Canvas, 100ms Update, pw-record PCM)
    # =========================================================================

    def _build_vu_meter(self):
        """Canvas-Balken fuer Audio-Pegel (gruen/gelb/rot) + dB Label."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=15, pady=(5, 3))

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
        frame.pack(fill=tk.X, padx=15, pady=(5, 10))

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
    # Werte laden (einmalig beim Oeffnen)
    # =========================================================================

    def _load_current_values(self):
        """Gespeicherte Werte aus settings.json laden, wpctl-Gain vergleichen."""
        audio = {}

        # Aus settings.json lesen
        try:
            if os.path.exists(SETTINGS_PATH):
                with open(SETTINGS_PATH, "r") as f:
                    data = json.load(f)
                audio = data.get("audio", {})
                if not isinstance(audio, dict):
                    audio = {}
        except Exception as e:
            logger.error(f"[AUDIO] settings.json lesen failed: {e}")

        # Gain (0.0 - 3.0)
        saved_gain = None
        raw_gain = audio.get("mic_gain")
        if raw_gain is not None and not isinstance(raw_gain, (dict, list)):
            try:
                saved_gain = max(0.0, min(3.0, float(raw_gain)))
                self._gain_var.set(saved_gain)
                self._gain_label.config(text=f"{saved_gain:.2f}")
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

        # Gespeicherte Samplerate laden
        raw_rate = audio.get("wifi_samplerate")
        if raw_rate in (16000, 48000):
            self._current_samplerate = raw_rate
            self._btn_samplerate.config(text=f"{raw_rate // 1000} kHz")

        # wpctl-Gain lesen und vergleichen
        wpctl_gain = self._read_wpctl_gain()
        if wpctl_gain is not None and saved_gain is not None:
            if abs(wpctl_gain - saved_gain) > 0.02:
                self._wpctl_hint.config(
                    text=f"wpctl aktuell: {wpctl_gain:.2f}"
                         f" (Config: {saved_gain:.2f})",
                    fg=STATUS_YELLOW,
                )
                logger.info(
                    f"[AUDIO] wpctl Gain {wpctl_gain:.2f} weicht von "
                    f"Config {saved_gain:.2f} ab")

    # =========================================================================
    # wpctl Gain lesen
    # =========================================================================

    def _read_wpctl_gain(self):
        """Aktuellen Gain-Wert via wpctl get-volume lesen."""
        node_id = self._find_respeaker_source_id()
        if not node_id:
            return None
        try:
            result = subprocess.run(
                ["wpctl", "get-volume", node_id],
                capture_output=True, text=True, timeout=3)
            if result.returncode != 0:
                return None
            # Format: "Volume: 1.00" oder "Volume: 1.00 [MUTED]"
            for part in result.stdout.strip().split():
                try:
                    return float(part)
                except ValueError:
                    continue
        except Exception as e:
            logger.error(f"[AUDIO] wpctl get-volume failed: {e}")
        return None

    # =========================================================================
    # Settings persistent speichern (debounced 300ms)
    # =========================================================================

    def _save_audio_settings(self):
        """Save nach 300ms Debounce ausloesen (verhindert Schreibflut bei Slider)."""
        if self._save_after_id is not None:
            self.win.after_cancel(self._save_after_id)
        self._save_after_id = self.win.after(300, self._do_save_audio_settings)

    def _do_save_audio_settings(self):
        """Audio-Sektion in settings.json atomar schreiben (tmp + rename)."""
        self._save_after_id = None
        try:
            # Bestehende settings.json lesen
            data = {}
            if os.path.exists(SETTINGS_PATH):
                with open(SETTINGS_PATH, "r") as f:
                    data = json.load(f)

            # Nur audio-Sektion updaten
            data["audio"] = {
                "mic_gain": round(self._gain_var.get(), 2),
                "noise_gate_db": round(self._noise_gate_var.get(), 1),
                "agc_enabled": self._agc_var.get(),
                "wifi_samplerate": self._current_samplerate,
            }

            # Atomar schreiben: tmp + rename
            dir_path = os.path.dirname(SETTINGS_PATH)
            fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2)
                    f.write("\n")
                os.replace(tmp_path, SETTINGS_PATH)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            logger.debug("[AUDIO] Audio-Settings in settings.json gespeichert")
        except Exception as e:
            logger.error(f"[AUDIO] settings.json speichern failed: {e}")

    # =========================================================================
    # Schliessen
    # =========================================================================

    def _on_close(self):
        """Fenster sauber schliessen — alle Timer und Prozesse stoppen."""
        # WiFi-Poll stoppen
        self._wifi_poll_running = False
        if self._wifi_after_id is not None:
            self.win.after_cancel(self._wifi_after_id)
            self._wifi_after_id = None

        # Ausstehende Settings sofort speichern
        if self._save_after_id is not None:
            self.win.after_cancel(self._save_after_id)
            self._save_after_id = None
            self._do_save_audio_settings()

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
