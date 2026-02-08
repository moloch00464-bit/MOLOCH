#!/usr/bin/env python3
"""
M.O.L.O.C.H. Face Training Capture Tool

Beautiful GUI for capturing face training photos.
Uses XIAO Vision AI camera via serial AT commands.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import sys
import time
import base64
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageTk
import io

# Add moloch to path
sys.path.insert(0, str(Path.home() / "moloch"))

from core.vision.xiao_vision import get_vision

# Configuration
TRAINING_DIR = Path.home() / "moloch" / "data" / "faces" / "train"
PHOTO_TARGET = 30  # Target number of photos per profile

# M.O.L.O.C.H. Colors (matching dashboard)
COLORS = {
    'bg': '#0a0a0f',
    'panel': '#12121a',
    'accent': '#1a1a2e',
    'primary': '#00d4ff',
    'ok': '#00ff88',
    'warning': '#ffaa00',
    'error': '#ff4444',
    'text': '#e0e0e0',
    'dim': '#666666'
}


class FaceCaptureApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("M.O.L.O.C.H. Face Training")
        self.root.geometry("900x700")
        self.root.configure(bg=COLORS['bg'])

        # State
        self.current_profile = None
        self.photo_count = 0
        self.preview_running = False
        self.xiao = None
        self.last_image = None

        self._setup_ui()
        self._load_profiles()
        self._connect_camera()

    def _setup_ui(self):
        """Setup the user interface."""
        # Title
        title = tk.Label(
            self.root,
            text="◉ M.O.L.O.C.H. FACE TRAINING",
            font=("Monospace", 20, "bold"),
            fg=COLORS['primary'],
            bg=COLORS['bg']
        )
        title.pack(pady=20)

        # Main container
        main_frame = tk.Frame(self.root, bg=COLORS['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left panel - Profile selection
        left_panel = tk.Frame(main_frame, bg=COLORS['panel'], width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        tk.Label(
            left_panel,
            text="PROFILE",
            font=("Monospace", 12, "bold"),
            fg=COLORS['primary'],
            bg=COLORS['panel']
        ).pack(pady=15)

        # Profile listbox
        self.profile_listbox = tk.Listbox(
            left_panel,
            font=("Monospace", 11),
            bg=COLORS['accent'],
            fg=COLORS['text'],
            selectbackground=COLORS['primary'],
            selectforeground=COLORS['bg'],
            height=10,
            borderwidth=0,
            highlightthickness=1,
            highlightcolor=COLORS['primary']
        )
        self.profile_listbox.pack(fill=tk.X, padx=15, pady=5)
        self.profile_listbox.bind('<<ListboxSelect>>', self._on_profile_select)

        # New profile entry
        tk.Label(
            left_panel,
            text="Neues Profil:",
            font=("Monospace", 10),
            fg=COLORS['dim'],
            bg=COLORS['panel']
        ).pack(pady=(20, 5))

        self.new_profile_entry = tk.Entry(
            left_panel,
            font=("Monospace", 11),
            bg=COLORS['accent'],
            fg=COLORS['text'],
            insertbackground=COLORS['primary'],
            borderwidth=0
        )
        self.new_profile_entry.pack(fill=tk.X, padx=15, pady=5)

        self._create_button(
            left_panel, "➕ Erstellen", self._create_profile,
            COLORS['ok']
        ).pack(fill=tk.X, padx=15, pady=10)

        # Camera status
        self.camera_status = tk.Label(
            left_panel,
            text="Kamera: ...",
            font=("Monospace", 9),
            fg=COLORS['dim'],
            bg=COLORS['panel']
        )
        self.camera_status.pack(pady=(30, 5))

        # Right panel - Preview and capture
        right_panel = tk.Frame(main_frame, bg=COLORS['panel'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Preview area
        self.preview_label = tk.Label(
            right_panel,
            text="Kamera Vorschau\n\nVerbinde mit XIAO...",
            font=("Monospace", 14),
            fg=COLORS['dim'],
            bg=COLORS['accent'],
            width=80,
            height=20
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Status bar
        status_frame = tk.Frame(right_panel, bg=COLORS['panel'])
        status_frame.pack(fill=tk.X, padx=15)

        self.status_label = tk.Label(
            status_frame,
            text="Bereit",
            font=("Monospace", 12),
            fg=COLORS['text'],
            bg=COLORS['panel']
        )
        self.status_label.pack(side=tk.LEFT)

        self.counter_label = tk.Label(
            status_frame,
            text="Fotos: 0 / 30",
            font=("Monospace", 12, "bold"),
            fg=COLORS['primary'],
            bg=COLORS['panel']
        )
        self.counter_label.pack(side=tk.RIGHT)

        # Progress bar
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Custom.Horizontal.TProgressbar",
                        troughcolor=COLORS['accent'],
                        background=COLORS['primary'])

        self.progress = ttk.Progressbar(
            right_panel,
            length=400,
            mode='determinate',
            maximum=PHOTO_TARGET,
            style="Custom.Horizontal.TProgressbar"
        )
        self.progress.pack(fill=tk.X, padx=15, pady=10)

        # Capture buttons
        button_frame = tk.Frame(right_panel, bg=COLORS['panel'])
        button_frame.pack(fill=tk.X, padx=15, pady=15)

        self._create_button(
            button_frame, "📷 FOTO (Space)", self._capture_single,
            COLORS['primary'], width=18
        ).pack(side=tk.LEFT, padx=5)

        self._create_button(
            button_frame, "🔄 5x AUTO", self._capture_burst,
            COLORS['ok'], width=18
        ).pack(side=tk.LEFT, padx=5)

        self._create_button(
            button_frame, "✖ Beenden", self._quit,
            COLORS['warning'], width=18
        ).pack(side=tk.RIGHT, padx=5)

        # Keyboard bindings
        self.root.bind('<space>', lambda e: self._capture_single())
        self.root.bind('<c>', lambda e: self._capture_burst())
        self.root.bind('<q>', lambda e: self._quit())
        self.root.bind('<Escape>', lambda e: self._quit())

    def _create_button(self, parent, text, command, color, width=None):
        """Create a styled button."""
        btn = tk.Button(
            parent,
            text=text,
            font=("Monospace", 11, "bold"),
            fg=COLORS['bg'],
            bg=color,
            activebackground=color,
            activeforeground=COLORS['bg'],
            borderwidth=0,
            padx=15,
            pady=8,
            cursor="hand2",
            command=command
        )
        if width:
            btn.configure(width=width)
        return btn

    def _connect_camera(self):
        """Connect to XIAO camera."""
        try:
            self.xiao = get_vision()
            if self.xiao.connected:
                mode = self.xiao.connection_mode
                port = self.xiao.serial_port if mode == "serial" else "WiFi"
                self.camera_status.config(
                    text=f"Kamera: {mode}\n({port})",
                    fg=COLORS['ok']
                )
                self.preview_label.config(text="Profil auswählen\nzum Starten")
            else:
                self.camera_status.config(
                    text="Kamera: OFFLINE",
                    fg=COLORS['error']
                )
                self.preview_label.config(
                    text="Kamera nicht gefunden!\n\nXIAO nicht verbunden?"
                )
        except Exception as e:
            self.camera_status.config(
                text=f"Kamera: FEHLER",
                fg=COLORS['error']
            )
            self.preview_label.config(text=f"Kamera Fehler:\n{str(e)}")

    def _load_profiles(self):
        """Load existing profiles."""
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        self.profile_listbox.delete(0, tk.END)

        for d in sorted(TRAINING_DIR.iterdir()):
            if d.is_dir():
                count = len(list(d.glob("*.jpg")))
                self.profile_listbox.insert(tk.END, f"{d.name} ({count})")

    def _on_profile_select(self, event):
        """Handle profile selection."""
        selection = self.profile_listbox.curselection()
        if not selection:
            return

        item = self.profile_listbox.get(selection[0])
        profile_name = item.split(" (")[0]
        self.current_profile = TRAINING_DIR / profile_name

        self.photo_count = len(list(self.current_profile.glob("*.jpg")))
        self._update_counter()

        self.status_label.config(text=f"Profil: {profile_name}")
        self._start_preview()

    def _create_profile(self):
        """Create a new profile."""
        name = self.new_profile_entry.get().strip()
        if not name:
            messagebox.showwarning("Fehler", "Bitte Profilnamen eingeben!")
            return

        # Sanitize name
        name = "".join(c for c in name if c.isalnum() or c in "_-")

        profile_dir = TRAINING_DIR / name
        profile_dir.mkdir(parents=True, exist_ok=True)

        self.new_profile_entry.delete(0, tk.END)
        self._load_profiles()

        # Select the new profile
        for i in range(self.profile_listbox.size()):
            if self.profile_listbox.get(i).startswith(name):
                self.profile_listbox.selection_set(i)
                self._on_profile_select(None)
                break

    def _update_counter(self):
        """Update the photo counter."""
        self.counter_label.config(text=f"Fotos: {self.photo_count} / {PHOTO_TARGET}")
        self.progress['value'] = min(self.photo_count, PHOTO_TARGET)

        if self.photo_count >= PHOTO_TARGET:
            self.counter_label.config(fg=COLORS['ok'])
        else:
            self.counter_label.config(fg=COLORS['primary'])

    def _start_preview(self):
        """Start camera preview."""
        if self.preview_running:
            return

        if not self.xiao or not self.xiao.connected:
            self.preview_label.config(text="Kamera nicht verbunden!")
            return

        self.preview_running = True
        self.preview_label.config(text="Starte Kamera...")

        # Start preview thread
        threading.Thread(target=self._preview_loop, daemon=True).start()

    def _preview_loop(self):
        """Camera preview loop using XIAO."""
        try:
            while self.preview_running:
                # Get inference with image from XIAO
                result = self.xiao.get_inference(with_image=True)

                if result and result.get("image"):
                    # Decode base64 image
                    img_data = base64.b64decode(result["image"])
                    img = Image.open(io.BytesIO(img_data))

                    # Store for capture
                    self.last_image = img.copy()

                    # Resize for preview (XIAO gives 480x480)
                    img = img.resize((420, 420), Image.Resampling.LANCZOS)

                    # Draw face detection boxes if any
                    detections = result.get("detections", [])
                    if detections:
                        from PIL import ImageDraw
                        draw = ImageDraw.Draw(img)
                        # Scale factor for 480->420
                        scale = 420 / 480
                        for det in detections:
                            x, y, w, h = det['x'], det['y'], det['w'], det['h']
                            x1, y1 = int(x * scale), int(y * scale)
                            x2, y2 = int((x + w) * scale), int((y + h) * scale)
                            draw.rectangle([x1, y1, x2, y2], outline='#00ff88', width=2)

                    # Convert for Tkinter
                    photo = ImageTk.PhotoImage(img)

                    # Update label (thread-safe)
                    self.root.after(0, lambda p=photo: self._update_preview(p))

                time.sleep(0.15)  # ~6-7 FPS (XIAO is slower)

        except Exception as e:
            self.root.after(0, lambda: self.preview_label.config(
                text=f"Kamera Fehler:\n{str(e)}"
            ))

    def _update_preview(self, photo):
        """Update preview image (called from main thread)."""
        self.preview_label.config(image=photo, text="")
        self.preview_label.image = photo  # Keep reference

    def _capture_single(self):
        """Capture a single photo."""
        if not self.current_profile:
            messagebox.showwarning("Fehler", "Bitte erst Profil auswählen!")
            return

        self._capture_photo()

    def _capture_photo(self):
        """Capture and save a photo."""
        if not self.last_image:
            self.status_label.config(text="✗ Kein Bild verfügbar")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = self.current_profile / f"face_{timestamp}.jpg"

        try:
            # Save the last captured image (full resolution from XIAO)
            self.last_image.save(str(filename), "JPEG", quality=95)

            self.photo_count += 1
            self._update_counter()
            self._load_profiles()  # Refresh counts

            # Flash effect
            self.preview_label.config(bg=COLORS['ok'])
            self.root.after(100, lambda: self.preview_label.config(bg=COLORS['accent']))

            self.status_label.config(text=f"✓ Foto {self.photo_count} gespeichert")

        except Exception as e:
            self.status_label.config(text=f"Fehler: {str(e)[:30]}")

    def _capture_burst(self):
        """Capture 5 photos with countdown."""
        if not self.current_profile:
            messagebox.showwarning("Fehler", "Bitte erst Profil auswählen!")
            return

        threading.Thread(target=self._burst_capture_thread, daemon=True).start()

    def _burst_capture_thread(self):
        """Burst capture in background thread."""
        for i in range(5):
            for countdown in range(3, 0, -1):
                self.root.after(0, lambda c=countdown, n=i+1:
                    self.status_label.config(text=f"Foto {n}/5 in {c}..."))
                time.sleep(1)

            self.root.after(0, self._capture_photo)
            time.sleep(0.5)

        self.root.after(0, lambda: self.status_label.config(text="5 Fotos aufgenommen!"))

    def _quit(self):
        """Quit the application."""
        self.preview_running = False
        self.root.destroy()

    def run(self):
        """Run the application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    # Ensure we have PIL
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("Installing Pillow...")
        import subprocess
        subprocess.run(["pip3", "install", "Pillow"])
        from PIL import Image, ImageTk

    app = FaceCaptureApp()
    app.run()

    print("\n" + "=" * 50)
    print("Training starten mit:")
    print("  python3 ~/moloch/scripts/train_faces.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
