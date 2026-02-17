#!/usr/bin/env python3
"""Fix: Snapshot IMMER speichern, ArcFace optional.

Problem: Snapshot blockiert wenn SCRFD/ArcFace nicht aktiv.
Fix: Frame immer als JPG speichern, Enrollment nur wenn Modelle da.
"""
import sys

panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    code = f.read()

# === Komplette _take_snapshot Methode ersetzen ===
old_method = '''    def _take_snapshot(self):
        """Take snapshot, run SCRFD+ArcFace, save embedding as Markus."""
        def do_snapshot():
            if not self.service:
                self.root.after(0, lambda: self._append_chat(
                    "System: Service nicht verbunden", "system"))
                return

            # Check models active
            if "scrfd" not in self.service._active_ctx:
                self.root.after(0, lambda: self._append_chat(
                    "System: SCRFD nicht aktiv! Checkbox aktivieren.", "system"))
                return
            if "arcface" not in self.service._active_ctx:
                self.root.after(0, lambda: self._append_chat(
                    "System: ArcFace nicht aktiv! Checkbox aktivieren.", "system"))
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
            box = largest[0]  # normalized xyxy in 640x640 space

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
                # Load existing
                existing_db = {}
                if os.path.exists(db_path):
                    with open(db_path, "r") as f:
                        existing_db = json.load(f)

                # Average with existing Markus embedding if present
                if "Markus" in existing_db:
                    old_emb = np.array(existing_db["Markus"], dtype=np.float32)
                    old_norm = np.linalg.norm(old_emb)
                    if old_norm > 0:
                        old_emb = old_emb / old_norm
                    # Weighted average: 70% existing + 30% new
                    combined = (old_emb * 0.7) + (embedding * 0.3)
                    combined = combined / np.linalg.norm(combined)
                    existing_db["Markus"] = combined.tolist()
                    msg = "[Snapshot] Markus-Embedding aktualisiert (gewichtet)"
                else:
                    existing_db["Markus"] = embedding.tolist()
                    msg = "[Snapshot] Markus-Embedding NEU gespeichert"

                # Ensure data dir exists
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                with open(db_path, "w") as f:
                    json.dump(existing_db, f)

                # Reload service face DB
                if hasattr(self.service, '_reload_face_db'):
                    self.service._reload_face_db()

                self.root.after(0, lambda m=msg: self._append_chat(m, "system"))
                logger.info(f"[SNAPSHOT] Markus embedding saved to {db_path}")

            except Exception as e:
                self.root.after(0, lambda: self._append_chat(
                    f"[Snapshot] Speichern fehlgeschlagen: {e}", "system"))

        threading.Thread(target=do_snapshot, daemon=True).start()'''

new_method = '''    def _take_snapshot(self):
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

if old_method in code:
    code = code.replace(old_method, new_method)
    with open(panel, 'w') as f:
        f.write(code)
    print('Snapshot Fix - OK')
else:
    print('ANCHOR NOT FOUND!')
    sys.exit(1)
