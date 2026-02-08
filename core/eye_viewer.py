#!/usr/bin/env python3
"""
M.O.L.O.C.H. Eye Viewer - USB Direct Mode
==========================================

Direkte Verbindung zum Grove Vision AI V2 (Himax) über USB.
Liefert SOWOHL Kamerabild ALS AUCH Face Detection Bounding Boxes.

USB: /dev/ttyACM0 @ 921600 baud
Befehl: AT+INVOKE=-1,0,0 (kontinuierliche Inference mit Bild)
"""

import sys
import time
import json
import base64
import threading
from io import BytesIO

import serial
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# ============================================================================
# Configuration
# ============================================================================

SERIAL_PORT = "/dev/ttyACM1"
SERIAL_BAUD = 921600
WINDOW_TITLE = "M.O.L.O.C.H. Eye"

# ============================================================================
# USB Camera Interface
# ============================================================================

class USBCamera:
    """Direct interface to Grove Vision AI V2 via USB."""

    def __init__(self, port=SERIAL_PORT, baud=SERIAL_BAUD):
        self.port = port
        self.baud = baud
        self.serial = None
        self.connected = False
        self.last_inference = None
        self.lock = threading.Lock()

    def connect(self):
        """Connect to Himax via USB."""
        try:
            self.serial = serial.Serial(self.port, self.baud, timeout=2)
            time.sleep(0.5)
            self.serial.reset_input_buffer()
            self.connected = True
            return True
        except Exception as e:
            print(f"USB Fehler: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from camera."""
        if self.serial:
            try:
                self.serial.close()
            except:
                pass
        self.serial = None
        self.connected = False

    def get_inference(self):
        """Get inference from Himax with image."""
        if not self.serial or not self.connected:
            return None

        with self.lock:
            try:
                self.serial.reset_input_buffer()
                # AT+INVOKE=-1,0,0 = continuous inference with image
                self.serial.write(b'AT+INVOKE=-1,0,0\r\n')

                # Read response - need enough data for base64 image
                data = b''
                start = time.time()
                while time.time() - start < 2.0:
                    if self.serial.in_waiting:
                        chunk = self.serial.read(self.serial.in_waiting)
                        data += chunk
                        # Check if we have complete JSON with image
                        if b'"image":' in data and b'"INVOKE"' in data:
                            # Count braces to find complete JSON
                            text = data.decode('utf-8', errors='ignore')
                            if self._has_complete_json(text):
                                break
                    else:
                        time.sleep(0.02)

                return self._parse_response(data.decode('utf-8', errors='ignore'))

            except Exception as e:
                print(f"Inference Fehler: {e}")
                return None

    def _has_complete_json(self, text):
        """Check if we have a complete JSON object."""
        for line in text.split('\n'):
            if '"INVOKE"' in line and '"image":' in line:
                start = line.find('{"type"')
                if start >= 0:
                    brace_count = 0
                    for c in line[start:]:
                        if c == '{':
                            brace_count += 1
                        elif c == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                return True
        return False

    def _parse_response(self, response):
        """Parse Himax response to extract detections and image."""
        result = {"num_detections": 0, "detections": [], "image": None}

        # Find JSON lines in response
        for line in response.split('\n'):
            line = line.strip()
            if '"INVOKE"' not in line or '"image":' not in line:
                continue

            # Find start of JSON object
            start_idx = line.find('{"type"')
            if start_idx == -1:
                continue

            # Find matching closing brace
            brace_count = 0
            end_idx = start_idx
            for i, c in enumerate(line[start_idx:]):
                if c == '{':
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = start_idx + i + 1
                        break

            try:
                data = json.loads(line[start_idx:end_idx])

                if data.get('name') == 'INVOKE':
                    resp_data = data.get('data', {})
                    boxes = resp_data.get('boxes', [])
                    result['num_detections'] = len(boxes)

                    for box in boxes:
                        if isinstance(box, list) and len(box) >= 5:
                            result['detections'].append({
                                'x': box[0],
                                'y': box[1],
                                'w': box[2],
                                'h': box[3],
                                'score': box[4]
                            })

                    # Extract image if present
                    image_b64 = resp_data.get('image')
                    if image_b64:
                        try:
                            # Fix padding if needed
                            missing = len(image_b64) % 4
                            if missing:
                                image_b64 += '=' * (4 - missing)
                            image_data = base64.b64decode(image_b64)
                            result['image'] = image_data
                        except Exception as e:
                            print(f"Image decode error: {e}")

                    self.last_inference = result
                    return result

            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
                continue

        return self.last_inference if self.last_inference else result


# ============================================================================
# GUI Viewer
# ============================================================================

class EyeViewer:
    """M.O.L.O.C.H. Eye Viewer - USB Direct Mode."""

    def __init__(self):
        self.camera = USBCamera()
        self.running = False
        self.frame_times = []
        self.photo = None

        # Create main window
        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.configure(bg='#1a1a2e')
        self.root.geometry("520x550")

        self._create_widgets()
        self._bind_events()

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1a1a2e')

        # Title
        title_label = tk.Label(
            main_frame,
            text="👁 M.O.L.O.C.H. EYE",
            font=('Courier', 18, 'bold'),
            bg='#1a1a2e',
            fg='#00ff88'
        )
        title_label.pack(pady=(0, 5))

        # Mode indicator
        self.mode_label = tk.Label(
            main_frame,
            text=f"🔌 USB Direct Mode: {SERIAL_PORT}",
            font=('Courier', 9),
            bg='#1a1a2e',
            fg='#00aaff'
        )
        self.mode_label.pack(pady=(0, 10))

        # Main canvas
        self.canvas_frame = tk.Frame(main_frame, bg='#0f0f1a', bd=2, relief=tk.SUNKEN)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=480,
            height=360,
            bg='#0f0f1a',
            highlightthickness=0
        )
        self.canvas.pack(padx=5, pady=5)

        # Info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(10, 5))

        self.fps_label = tk.Label(
            info_frame,
            text="Inference: --",
            font=('Courier', 10),
            bg='#1a1a2e',
            fg='#00aaff'
        )
        self.fps_label.pack(side=tk.LEFT)

        self.status_label = tk.Label(
            info_frame,
            text="Stopped",
            font=('Courier', 10),
            bg='#1a1a2e',
            fg='#888888'
        )
        self.status_label.pack(side=tk.RIGHT)

        # Detection info
        self.detection_label = tk.Label(
            main_frame,
            text="Detections: --",
            font=('Courier', 11, 'bold'),
            bg='#1a1a2e',
            fg='#00ff88'
        )
        self.detection_label.pack(fill=tk.X, pady=(5, 5))

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        self.connect_btn = tk.Button(
            button_frame,
            text="▶ START",
            font=('Courier', 11, 'bold'),
            bg='#16213e',
            fg='#00ff88',
            activebackground='#1f4068',
            activeforeground='#00ff88',
            bd=0,
            padx=20,
            pady=8,
            command=self.toggle_stream
        )
        self.connect_btn.pack(expand=True, fill=tk.X)

        self._show_placeholder()

    def _bind_events(self):
        self.root.bind('<space>', lambda e: self.toggle_stream())
        self.root.bind('<q>', lambda e: self.quit())
        self.root.bind('<Escape>', lambda e: self.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def _show_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            240, 180,
            text="👁\n\nUSB Direct Mode\n\n"
                 f"Grove Vision AI V2\n"
                 f"{SERIAL_PORT} @ {SERIAL_BAUD}\n\n"
                 "[SPACE] Start/Stop\n[Q] Quit",
            font=('Courier', 11),
            fill='#00ff88',
            justify=tk.CENTER
        )

    def toggle_stream(self):
        if self.running:
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self):
        self.status_label.config(text="Connecting...", fg='#ffaa00')
        self.root.update()

        if not self.camera.connect():
            self.status_label.config(text="USB nicht erreichbar!", fg='#ff4444')
            return

        self.running = True
        self.frame_times = []
        self.status_label.config(text="Streaming...", fg='#00ff88')
        self.connect_btn.config(text="⏹ STOP", fg='#ff4444')

        self._update_frame()

    def stop_stream(self):
        self.running = False
        self.camera.disconnect()
        self.connect_btn.config(text="▶ START", fg='#00ff88')
        self.status_label.config(text="Stopped", fg='#888888')
        self._show_placeholder()

    def _update_frame(self):
        if not self.running:
            return

        inference = self.camera.get_inference()

        if inference:
            num_detections = inference.get('num_detections', 0)
            detections = inference.get('detections', [])
            image_data = inference.get('image')

            self._draw_frame(detections, num_detections, image_data)
            self._update_fps()

            # Update detection label
            if num_detections > 0:
                self.detection_label.config(
                    text=f"👤 {num_detections} Face(s) detected",
                    fg='#00ff88'
                )
            else:
                self.detection_label.config(
                    text="Scanning...",
                    fg='#888888'
                )

        # Schedule next update
        if self.running:
            self.root.after(100, self._update_frame)

    def _draw_frame(self, detections, num_detections, image_data=None):
        self.canvas.delete("all")

        # Draw camera image if available
        if image_data:
            try:
                img = Image.open(BytesIO(image_data))
                # Resize to fit canvas
                img = img.resize((480, 360), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            except Exception as e:
                print(f"Image display error: {e}")
                self._draw_grid()
        else:
            self._draw_grid()

        # Draw detection boxes ON TOP of image
        for det in detections:
            x = det.get('x', 0)
            y = det.get('y', 0)
            w = det.get('w', 50)
            h = det.get('h', 50)
            score = det.get('score', 0)

            # Scale coordinates to canvas (480x360 from 480x480 or similar)
            # Himax typically outputs center-x, center-y, width, height
            scale_x = 480 / 480
            scale_y = 360 / 480
            cx = int(x * scale_x)
            cy = int(y * scale_y)
            bw = int(w * scale_x)
            bh = int(h * scale_y)

            x1, y1 = cx - bw // 2, cy - bh // 2
            x2, y2 = cx + bw // 2, cy + bh // 2

            # Draw green bounding box
            self.canvas.create_rectangle(x1, y1, x2, y2, outline='#00ff88', width=3)

            # Draw label background
            self.canvas.create_rectangle(x1, y1 - 20, x1 + 80, y1, fill='#00ff88', outline='')
            self.canvas.create_text(x1 + 40, y1 - 10, text=f"Face {score}%", font=('Courier', 9, 'bold'), fill='#000000')

        # Status overlay
        status_text = f"USB: {num_detections} det"
        self.canvas.create_text(10, 10, text=status_text, font=('Courier', 9), fill='#00aaff', anchor=tk.NW)

    def _draw_grid(self):
        """Draw grid background when no image."""
        self.canvas.create_rectangle(0, 0, 480, 360, fill='#0a0a15', outline='')
        for i in range(0, 480, 40):
            self.canvas.create_line(i, 0, i, 360, fill='#1a1a2e', width=1)
        for i in range(0, 360, 40):
            self.canvas.create_line(0, i, 480, i, fill='#1a1a2e', width=1)

    def _update_fps(self):
        now = time.time()
        self.frame_times.append(now)
        self.frame_times = [t for t in self.frame_times if now - t < 1.0]
        fps = len(self.frame_times)
        self.fps_label.config(text=f"Inference: {fps} FPS")

    def quit(self):
        self.running = False
        self.camera.disconnect()
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 50)
    print("M.O.L.O.C.H. Eye Viewer - USB Direct Mode")
    print("=" * 50)
    print()
    print(f"USB: {SERIAL_PORT} @ {SERIAL_BAUD} baud")
    print("Grove Vision AI V2 (Himax)")
    print("AT+INVOKE=-1,0,0 (Bild + Face Detection)")
    print()
    print("Controls:")
    print("  SPACE  - Start/Stop")
    print("  Q/ESC  - Quit")
    print()

    viewer = EyeViewer()
    viewer.run()


if __name__ == "__main__":
    main()
