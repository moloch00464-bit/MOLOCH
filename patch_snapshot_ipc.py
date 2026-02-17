#!/usr/bin/env python3
"""Snapshot Enrollment via IPC.

Problem: Panel (Remote-Modus) hat keinen NPU-Zugriff -> _run_model() = None.
Fix: Panel sendet IPC "snapshot_enroll", Service macht SCRFD+ArcFace+Save.
     Ergebnis via /tmp/moloch_snapshot_result.json zurueck ans Panel.
"""
import sys

# ============================================================
# TEIL 1: moloch_service.py - IPC Handler + Enrollment-Methode
# ============================================================
svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# --- 1A: IPC Handler nach reload_face_db ---
old_ipc = """        elif action == 'reload_face_db':
            self._reload_face_db()"""

new_ipc = """        elif action == 'reload_face_db':
            self._reload_face_db()
        elif action == 'snapshot_enroll':
            threading.Thread(target=self._snapshot_enroll, daemon=True).start()"""

if old_ipc in code:
    code = code.replace(old_ipc, new_ipc)
    print('1A: IPC Handler snapshot_enroll - OK')
    fixes += 1
else:
    print('1A: ANCHOR NOT FOUND!')

# --- 1B: _snapshot_enroll Methode vor _reload_face_db ---
old_reload = """    def _reload_face_db(self):"""

new_reload = """    def _snapshot_enroll(self):
        \"\"\"Snapshot-Enrollment: SCRFD + ArcFace auf aktuellem Frame, Embedding speichern.\"\"\"
        result_path = "/tmp/moloch_snapshot_result.json"
        try:
            # Frame holen
            frame = None
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()
            if frame is None:
                self._write_snapshot_result(result_path, False, "Kein Frame verfuegbar")
                return

            fh, fw = frame.shape[:2]

            # SCRFD muss aktiv sein (oder temporaer laden)
            _had_scrfd = "scrfd" in self._active_ctx
            _had_arcface = "arcface" in self._active_ctx

            if not _had_scrfd:
                self._configure_model("scrfd")
                import time as _t; _t.sleep(0.3)
            if not _had_arcface:
                self._configure_model("arcface")
                import time as _t; _t.sleep(0.3)

            # SCRFD Inference
            input_640 = cv2.resize(frame, (640, 640))
            input_rgb = cv2.cvtColor(input_640, cv2.COLOR_BGR2RGB)
            outputs = self._run_model("scrfd", input_rgb)
            if not outputs:
                self._write_snapshot_result(result_path, False, "SCRFD Inference fehlgeschlagen")
                return

            out_names = self._output_names["scrfd"]
            raw_outputs = [outputs[n] for n in out_names]
            faces = decode_scrfd(raw_outputs, score_thresh=0.4)

            if not faces:
                self._write_snapshot_result(result_path, False, "Kein Gesicht erkannt")
                return

            # Groesstes Face
            largest = max(faces, key=lambda f: (f[0][2]-f[0][0]) * (f[0][3]-f[0][1]))
            box = largest[0]

            # Crop mit 20% Margin
            x1 = max(0, int(box[0] * fw))
            y1 = max(0, int(box[1] * fh))
            x2 = min(fw, int(box[2] * fw))
            y2 = min(fh, int(box[3] * fh))
            bw, bh = x2 - x1, y2 - y1
            mx, my = int(bw * 0.2), int(bh * 0.2)
            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(fw, x2 + mx)
            y2 = min(fh, y2 + my)

            if x2 <= x1 or y2 <= y1:
                self._write_snapshot_result(result_path, False, "Face-Crop ungueltig")
                return

            # ArcFace Embedding
            crop = frame[y1:y2, x1:x2]
            crop_112 = cv2.resize(crop, (112, 112))
            crop_rgb = cv2.cvtColor(crop_112, cv2.COLOR_BGR2RGB)

            arc_outputs = self._run_model("arcface", crop_rgb)
            if not arc_outputs:
                self._write_snapshot_result(result_path, False, "ArcFace Inference fehlgeschlagen")
                return

            emb_key = self._output_names["arcface"][0]
            embedding = arc_outputs[emb_key].flatten()
            embedding = normalize_arcface(embedding)

            # In face_embeddings.json speichern
            existing_db = {}
            if os.path.exists(FACE_DB_PATH):
                with open(FACE_DB_PATH, "r") as f:
                    existing_db = json.load(f)

            if "Markus" in existing_db:
                old_emb = np.array(existing_db["Markus"], dtype=np.float32)
                old_norm = np.linalg.norm(old_emb)
                if old_norm > 0:
                    old_emb = old_emb / old_norm
                combined = (old_emb * 0.7) + (embedding * 0.3)
                combined = combined / np.linalg.norm(combined)
                existing_db["Markus"] = combined.tolist()
                msg = "Markus-Embedding aktualisiert (70/30 gewichtet)"
            else:
                existing_db["Markus"] = embedding.tolist()
                msg = "Markus-Embedding NEU gespeichert"

            os.makedirs(os.path.dirname(FACE_DB_PATH), exist_ok=True)
            with open(FACE_DB_PATH, "w") as f:
                json.dump(existing_db, f)

            self._reload_face_db()
            logger.info(f"[SNAPSHOT] {msg}")
            self._write_snapshot_result(result_path, True, msg)

        except Exception as e:
            logger.error(f"[SNAPSHOT] Enrollment fehlgeschlagen: {e}")
            self._write_snapshot_result(result_path, False, str(e))

    def _write_snapshot_result(self, path, success, message):
        \"\"\"Ergebnis fuer Panel schreiben.\"\"\"
        try:
            import time as _t
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"success": success, "message": message, "ts": _t.time()}, f)
            os.replace(tmp, path)
        except Exception:
            pass

    def _reload_face_db(self):"""

if old_reload in code:
    code = code.replace(old_reload, new_reload, 1)
    print('1B: _snapshot_enroll Methode - OK')
    fixes += 1
else:
    print('1B: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

print(f'\nService: {fixes}/2 Fixes.')
if fixes < 2:
    print('SERVICE INCOMPLETE!')
    sys.exit(1)

# ============================================================
# TEIL 2: moloch_unified_panel.py - Snapshot via IPC
# ============================================================
panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    pcode = f.read()

pfixes = 0

# --- Komplette _take_snapshot ersetzen ---
old_snap = '''    def _take_snapshot(self):
        """Snapshot: Frame IMMER speichern, ArcFace-Enrollment optional."""
        def do_snapshot():
            if not self.service:
                self.root.after(0, lambda: self._append_chat(
                    "System: Service nicht verbunden", "system"))
                return

            self.root.after(0, lambda: self._append_chat(
                "[Snapshot] Erfasse Frame...", "system"))

            # Get current frame
            frame = None
            try:
                with self.service._frame_lock:
                    if self.service._latest_frame is not None:
                        frame = self.service._latest_frame.copy()
            except Exception:
                pass

            if frame is None:
                self.root.after(0, lambda: self._append_chat(
                    "[Snapshot] Kein Frame verfuegbar", "system"))
                return

            fh, fw = frame.shape[:2]

            # Frame IMMER als JPG speichern
            snap_dir = os.path.expanduser("~/moloch/data/snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            snap_path = os.path.join(snap_dir, f"snap_{ts}.jpg")
            cv2.imwrite(snap_path, frame)
            self.root.after(0, lambda p=snap_path: self._append_chat(
                f"[Snapshot] Bild gespeichert: {p}", "system"))
            logger.info(f"[SNAPSHOT] Frame saved: {snap_path}")

            # ArcFace-Enrollment nur wenn beide Modelle aktiv
            _has_scrfd = "scrfd" in self.service._active_ctx
            _has_arcface = "arcface" in self.service._active_ctx
            if not (_has_scrfd and _has_arcface):
                self.root.after(0, lambda: self._append_chat(
                    "[Snapshot] Kein Enrollment (SCRFD/ArcFace nicht aktiv)", "system"))
                return

            # Run SCRFD
            self.root.after(0, lambda: self._append_chat(
                "[Snapshot] Face Detection...", "system"))
            try:
                from core.perception.hailo_postprocess import decode_scrfd, normalize_arcface
                input_640 = cv2.resize(frame, (640, 640))
                input_rgb = cv2.cvtColor(input_640, cv2.COLOR_BGR2RGB)

                outputs = self.service._run_model("scrfd", input_rgb)
                if not outputs:
                    self.root.after(0, lambda: self._append_chat(
                        "[Snapshot] SCRFD Inference fehlgeschlagen", "system"))
                    return

                out_names = self.service._output_names["scrfd"]
                raw_outputs = [outputs[n] for n in out_names]
                faces = decode_scrfd(raw_outputs, score_thresh=0.4)

                if not faces:
                    self.root.after(0, lambda: self._append_chat(
                        "[Snapshot] Kein Gesicht erkannt!", "system"))
                    return

            except Exception as e:
                self.root.after(0, lambda: self._append_chat(
                    f"[Snapshot] SCRFD Fehler: {e}", "system"))
                return

            # Find largest face
            largest = max(faces, key=lambda f: (f[0][2]-f[0][0]) * (f[0][3]-f[0][1]))
            box = largest[0]

            # Crop face with 20% margin (map to original frame)
            x1 = max(0, int(box[0] * fw))
            y1 = max(0, int(box[1] * fh))
            x2 = min(fw, int(box[2] * fw))
            y2 = min(fh, int(box[3] * fh))
            bw, bh = x2 - x1, y2 - y1
            mx, my = int(bw * 0.2), int(bh * 0.2)
            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(fw, x2 + mx)
            y2 = min(fh, y2 + my)

            if x2 <= x1 or y2 <= y1:
                self.root.after(0, lambda: self._append_chat(
                    "[Snapshot] Face-Crop ungueltig", "system"))
                return

            # ArcFace embedding
            self.root.after(0, lambda: self._append_chat(
                "[Snapshot] ArcFace Embedding...", "system"))
            try:
                crop = frame[y1:y2, x1:x2]
                crop_112 = cv2.resize(crop, (112, 112))
                crop_rgb = cv2.cvtColor(crop_112, cv2.COLOR_BGR2RGB)

                arc_outputs = self.service._run_model("arcface", crop_rgb)
                if not arc_outputs:
                    self.root.after(0, lambda: self._append_chat(
                        "[Snapshot] ArcFace Inference fehlgeschlagen", "system"))
                    return

                emb_key = self.service._output_names["arcface"][0]
                embedding = arc_outputs[emb_key].flatten()
                embedding = normalize_arcface(embedding)

            except Exception as e:
                self.root.after(0, lambda: self._append_chat(
                    f"[Snapshot] ArcFace Fehler: {e}", "system"))
                return

            # Save to face_embeddings.json
            db_path = os.path.expanduser("~/moloch/data/face_embeddings.json")
            try:
                existing_db = {}
                if os.path.exists(db_path):
                    with open(db_path, "r") as f:
                        existing_db = json.load(f)

                if "Markus" in existing_db:
                    old_emb = np.array(existing_db["Markus"], dtype=np.float32)
                    old_norm = np.linalg.norm(old_emb)
                    if old_norm > 0:
                        old_emb = old_emb / old_norm
                    combined = (old_emb * 0.7) + (embedding * 0.3)
                    combined = combined / np.linalg.norm(combined)
                    existing_db["Markus"] = combined.tolist()
                    msg = "[Snapshot] Markus-Embedding aktualisiert (gewichtet)"
                else:
                    existing_db["Markus"] = embedding.tolist()
                    msg = "[Snapshot] Markus-Embedding NEU gespeichert"

                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                with open(db_path, "w") as f:
                    json.dump(existing_db, f)

                if hasattr(self.service, '_reload_face_db'):
                    self.service._reload_face_db()

                self.root.after(0, lambda m=msg: self._append_chat(m, "system"))
                logger.info(f"[SNAPSHOT] Markus embedding saved to {db_path}")

            except Exception as e:
                self.root.after(0, lambda: self._append_chat(
                    f"[Snapshot] Speichern fehlgeschlagen: {e}", "system"))

        threading.Thread(target=do_snapshot, daemon=True).start()'''

new_snap = '''    def _take_snapshot(self):
        """Snapshot: Frame als JPG + Enrollment via IPC (Service macht NPU-Arbeit)."""
        def do_snapshot():
            if not self.service:
                self.root.after(0, lambda: self._append_chat(
                    "System: Service nicht verbunden", "system"))
                return

            self.root.after(0, lambda: self._append_chat(
                "[Snapshot] Erfasse Frame...", "system"))

            # Frame holen und als JPG speichern
            frame = None
            try:
                with self.service._frame_lock:
                    if self.service._latest_frame is not None:
                        frame = self.service._latest_frame.copy()
            except Exception:
                pass

            if frame is None:
                self.root.after(0, lambda: self._append_chat(
                    "[Snapshot] Kein Frame verfuegbar", "system"))
                return

            snap_dir = os.path.expanduser("~/moloch/data/snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            snap_path = os.path.join(snap_dir, f"snap_{ts}.jpg")
            cv2.imwrite(snap_path, frame)
            self.root.after(0, lambda p=snap_path: self._append_chat(
                f"[Snapshot] Bild gespeichert: {p}", "system"))
            logger.info(f"[SNAPSHOT] Frame saved: {snap_path}")

            # Enrollment via IPC an Service delegieren
            result_path = "/tmp/moloch_snapshot_result.json"
            # Altes Ergebnis loeschen
            try:
                if os.path.exists(result_path):
                    os.remove(result_path)
            except Exception:
                pass

            self.root.after(0, lambda: self._append_chat(
                "[Snapshot] Enrollment laeuft (Service)...", "system"))
            self._send_cmd({"action": "snapshot_enroll"})

            # Auf Ergebnis warten (max 10s)
            for _ in range(20):
                time.sleep(0.5)
                try:
                    if os.path.exists(result_path):
                        with open(result_path, "r") as f:
                            result = json.load(f)
                        if result.get("success"):
                            msg = f"[Snapshot] {result['message']}"
                        else:
                            msg = f"[Snapshot] Fehler: {result['message']}"
                        self.root.after(0, lambda m=msg: self._append_chat(m, "system"))
                        return
                except Exception:
                    pass

            self.root.after(0, lambda: self._append_chat(
                "[Snapshot] Enrollment Timeout (keine Antwort vom Service)", "system"))

        threading.Thread(target=do_snapshot, daemon=True).start()'''

if old_snap in pcode:
    pcode = pcode.replace(old_snap, new_snap)
    print('2A: _take_snapshot via IPC - OK')
    pfixes += 1
else:
    print('2A: ANCHOR NOT FOUND!')

with open(panel, 'w') as f:
    f.write(pcode)

print(f'\nPanel: {pfixes}/1 Fixes.')
if pfixes < 1:
    print('PANEL INCOMPLETE!')
    sys.exit(1)

print('\n=== SNAPSHOT IPC KOMPLETT ===')
