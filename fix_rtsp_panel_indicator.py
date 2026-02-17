#!/usr/bin/env python3
"""Fix: RTSP Stream Indikator im Panel.

Zeigt Stream-Status:
- Gruen: Stream live
- Gelb: Reconnecting...
- Rot: Stream tot, "KAMERA OFFLINE" overlay
"""

panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    code = f.read()

fixes = 0

# FIX 1: Init-Variablen fuer Stream-Indikator
old_init = '''        self._canvas_image_id = None
        self._photo = None
        self._syncing = False'''

new_init = '''        self._canvas_image_id = None
        self._photo = None
        self._syncing = False

        # RTSP Stream Indikator
        self._stream_indicator_id = None
        self._stream_offline_text_id = None'''

if old_init in code:
    code = code.replace(old_init, new_init)
    print('FIX 1: Stream-Indikator Init - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# FIX 2: Stream-Status Indikator im Display Loop
# Finde den Abschnitt wo der Frame auf Canvas gezeichnet wird
old_display = '''        if frame is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (self.PREVIEW_W, self.PREVIEW_H))
                self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                if self._canvas_image_id is None:
                    self._canvas_image_id = self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
                else:
                    self.preview_canvas.itemconfig(self._canvas_image_id, image=self._photo)
            except Exception:
                pass

        self._display_after_id = self.root.after(
            1000 // self.DISPLAY_FPS, self._display_loop)'''

new_display = '''        if frame is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (self.PREVIEW_W, self.PREVIEW_H))
                self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                if self._canvas_image_id is None:
                    self._canvas_image_id = self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
                else:
                    self.preview_canvas.itemconfig(self._canvas_image_id, image=self._photo)
            except Exception:
                pass

        # RTSP Stream Status Indikator
        try:
            rtsp_alive = getattr(self.service, '_rtsp_stream_alive', True)
            rtsp_reconnecting = getattr(self.service, '_rtsp_reconnecting', False)

            # Indikator-Farbe
            if not rtsp_alive:
                color = "#ff4444"  # Rot
                status_text = "OFFLINE"
            elif rtsp_reconnecting:
                color = "#ffaa00"  # Gelb
                status_text = "RECONNECT"
            else:
                color = "#00aa44"  # Gruen
                status_text = None

            # Indikator (Oval unten rechts)
            x, y = self.PREVIEW_W - 15, self.PREVIEW_H - 15
            if self._stream_indicator_id is None:
                self._stream_indicator_id = self.preview_canvas.create_oval(
                    x-8, y-8, x+8, y+8, fill=color, outline="")
            else:
                self.preview_canvas.itemconfig(self._stream_indicator_id, fill=color)

            # "KAMERA OFFLINE" Overlay wenn Stream tot
            if not rtsp_alive:
                if self._stream_offline_text_id is None:
                    self._stream_offline_text_id = self.preview_canvas.create_text(
                        self.PREVIEW_W // 2, self.PREVIEW_H // 2,
                        text="KAMERA OFFLINE", font=("Consolas", 24, "bold"),
                        fill="#ff4444", anchor=tk.CENTER)
            else:
                if self._stream_offline_text_id is not None:
                    self.preview_canvas.delete(self._stream_offline_text_id)
                    self._stream_offline_text_id = None
        except Exception:
            pass

        self._display_after_id = self.root.after(
            1000 // self.DISPLAY_FPS, self._display_loop)'''

if old_display in code:
    code = code.replace(old_display, new_display)
    print('FIX 2: Stream-Status Indikator + OFFLINE Overlay - OK')
    fixes += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

with open(panel, 'w') as f:
    f.write(code)

compile(open(panel).read(), panel, 'exec')
print(f'\n{fixes} Fixes. Syntax OK.')
