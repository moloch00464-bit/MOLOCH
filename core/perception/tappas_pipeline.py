#!/usr/bin/env python3
"""
TappasPipeline — GStreamer/TAPPAS Multi-Model NPU Pipeline.

Gate 0.5 Phase 3.1: Ersetzt inference_engine.py + model_orchestrator.py
durch native TAPPAS-Pipeline mit Model Scheduler.

Pipeline:
  rtspsrc → YOLO_WRAPPER(letterbox→hailonet→yolo_postproc)
          → SCRFD_WRAPPER(letterbox→hailonet→scrfd_postproc)
          → hailotracker
          → FACE_CROPPER(face_align→hailonet_arcface→arcface_postproc)
          → identity_callback → hailooverlay → appsink

Alle Modelle teilen sich den NPU via vdevice-group-id=SHARED (Model Scheduler).

Nutzung:
    # Feature-Flag: MOLOCH_USE_TAPPAS=1
    from core.perception.tappas_pipeline import TappasPipeline
    pipeline = TappasPipeline()
    pipeline.start()
    detections = pipeline.get_detections()
    pframe = pipeline.get_current_pframe()
    pipeline.stop()
"""

import os
import time
import struct
import threading
import logging
from typing import List, Dict, Optional

import cv2
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import hailo

from core.perception.perception_frame import PerceptionFrame, estimate_distance
from core.moloch_event_bus import get_event_bus, PRIO_PERCEPTION

logger = logging.getLogger("TappasPipeline")

# --- Modell-Pfade (SSD2) ---
YOLO_HEF = "/mnt/moloch-data/hailo/models/yolov8m_h10.hef"
SCRFD_HEF = "/mnt/moloch-data/hailo/models/scrfd_10g.hef"
ARCFACE_HEF = "/mnt/moloch-data/hailo/models/arcface_mobilefacenet.hef"
FACE_ATTR_HEF = "/mnt/moloch-data/hailo/models/face_attr_resnet_v1_18.hef"

# --- Postprocess SOs ---
YOLO_POSTPROCESS_SO = "/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so"
YOLO_POSTPROCESS_FUNC = "filter_letterbox"
SCRFD_POSTPROCESS_SO = "/usr/local/hailo/resources/so/libscrfd.so"
SCRFD_POSTPROCESS_FUNC = "scrfd_10g_letterbox"
SCRFD_CONFIG_JSON = "/usr/local/hailo/resources/json/scrfd.json"
ARCFACE_POSTPROCESS_SO = "/usr/local/hailo/resources/so/libface_recognition_post.so"
ARCFACE_POSTPROCESS_FUNC = "filter"
FACE_ATTR_POSTPROCESS_SO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "lib", "libface_attributes_fc2.so")
FACE_ATTR_POSTPROCESS_FUNC = "filter"
FACE_ALIGN_SO = "/usr/local/hailo/resources/so/libvms_face_align.so"
FACE_CROP_SO = "/usr/local/hailo/resources/so/libvms_croppers.so"
FACE_CROP_FUNC = "face_recognition"
WHOLE_BUFFER_SO = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so"

VDEVICE_GROUP_ID = "SHARED"

# YOLO Klassen-Whitelist (nur diese werden verarbeitet, Rest ignoriert)
YOLO_ALLOWED_CLASSES = {"person"}

# NPU Model-Scheduler Zustände
SCHED_YOLO_ONLY = "YOLO_ONLY"    # Niemand da → nur YOLO aktiv
SCHED_YOLO_SCRFD = "YOLO_SCRFD"  # Person erkannt → YOLO + SCRFD
SCHED_ALL_ACTIVE = "ALL_ACTIVE"   # Gesicht sichtbar → alle Modelle
SCHED_COOLDOWN_DOWN = 3.0         # Sekunden ohne Daten bis Downgrade

# IPC: Frame-Preview fuer Panel (gleicher Weg wie InferenceEngine → IPCRouter)
SHM_FRAME_PATH = "/dev/shm/moloch_frame"
SHM_PREVIEW_W = 640
SHM_PREVIEW_H = 360


def _build_rtsp_url() -> str:
    """RTSP-URL aus Environment-Variablen zusammenbauen."""
    # Zuerst: fertige URL aus ENV
    url = os.environ.get("MOLOCH_RTSP_URL", "")
    if url:
        return url
    # Fallback: Einzelteile
    host = os.environ.get("MOLOCH_CAMERA_HOST", "192.168.178.25")
    user = os.environ.get("MOLOCH_CAMERA_USER", "")
    pw = os.environ.get("MOLOCH_CAMERA_PASS", "")
    if user and pw:
        return f"rtsp://{user}:{pw}@{host}:554/av_stream/ch0"
    return f"rtsp://{host}:554/av_stream/ch0"


class TappasPipeline:
    """GStreamer/TAPPAS Multi-Model NPU Pipeline.

    Gleiche Output-Struktur wie InferenceEngine, damit moloch_service.py
    beide nutzen kann (Feature-Flag MOLOCH_USE_TAPPAS=1).
    """

    def __init__(self, rtsp_url: str = None, face_db: dict = None,
                 width: int = 1280, height: int = 720):
        """
        Args:
            rtsp_url: RTSP Stream URL (oder None fuer ENV-Variable)
            face_db: Face-Embedding-DB {name: np.array} fuer ArcFace-Matching
            width: Pipeline-Breite (nach Source-Scale)
            height: Pipeline-Hoehe (nach Source-Scale)
        """
        self._rtsp_url = rtsp_url or _build_rtsp_url()
        self._width = width
        self._height = height

        # Face DB fuer Embedding-Matching
        self._face_db = face_db or {}
        self._face_db_lock = threading.Lock()

        # State
        self._pipeline = None
        self._loop = None
        self._loop_thread = None
        self._running = False

        # Detections (thread-safe, letzter Frame)
        self._lock = threading.Lock()
        self._detections: List[Dict] = []
        self._current_pframe = PerceptionFrame()
        self._annotated_frame: Optional[np.ndarray] = None

        # FPS Tracking
        self._fps_lock = threading.Lock()
        self._frame_count = 0
        self._fps_start = 0.0
        self._fps_last_count = 0
        self._fps_last_time = 0.0
        self._current_fps = 0.0

        # SHM IPC Sequenznummer
        self._shm_seq = 0

        # Face Attributes Cache (befuellt von _on_face_attr_buffer Probe)
        self._face_attr_cache = {}  # {"gender": "M"|"F", "smiling": True|False}
        self._face_attr_lock = threading.Lock()

        # --- Model-Active-Flags (TAPPAS = immer aktiv, Panel liest diese) ---
        self.scrfd_active = True
        self.arcface_active = True
        self.yolo_active = True
        self.hand_active = False   # Hand-Modell nicht in TAPPAS Pipeline
        self.pose_active = False   # Pose-Modell nicht in TAPPAS Pipeline

        # --- Threshold-Werte (Panel setzt diese, TAPPAS managed intern) ---
        self.scrfd_conf_val = 0.30
        self.scrfd_nms_val = 0.45
        self.arcface_thresh_val = 0.65
        self.yolo_conf_val = 0.30
        self.pose_conf_val = 0.30
        self.hand_conf_val = 0.30

        # --- Feature-Flags (Panel/Settings lesen/schreiben diese) ---
        self._learner_flash = False
        self._hand_occlusion_enabled = False

        # --- Live-Enrollment ---
        self._enroll_active = False
        self._enroll_done = False
        self._enroll_name = ""
        self._enroll_target = 20
        self._enroll_candidates = []  # (score, embedding_copy)
        self._enroll_min_score = 0.50
        self._enroll_diversity = 0.85  # Cosine-Sim Schwelle
        self._enroll_lock = threading.Lock()

        # Event Bus fuer Action Bridge
        self._event_bus = get_event_bus()
        self._last_person_state = False  # Fuer target_lost Erkennung

        # --- NPU Model-Scheduler ---
        self._sched_mode = SCHED_YOLO_ONLY
        self._sched_person_last_seen = 0.0
        self._sched_face_last_seen = 0.0
        self._sched_lock = threading.Lock()

        # hailonet Referenzen fuer pass-through Steuerung (nach start() gesetzt)
        self._scrfd_hailonet_el = None
        self._arcface_hailonet_el = None

        # GStreamer einmal initialisieren
        if not Gst.is_initialized():
            Gst.init(None)

    # =====================================================================
    # Public API
    # =====================================================================

    def start(self):
        """Pipeline starten. Blockiert NICHT — laeuft in eigenem Thread."""
        if self._running:
            logger.warning("Pipeline laeuft bereits")
            return

        logger.info("Starte TAPPAS Multi-Model Pipeline...")
        logger.info(f"  RTSP: {self._rtsp_url.split('@')[1] if '@' in self._rtsp_url else self._rtsp_url}")
        logger.info(f"  Modelle: YOLO + SCRFD + ArcFace (vdevice-group-id={VDEVICE_GROUP_ID})")
        logger.info(f"  YOLO filter: person-only active (allowed={YOLO_ALLOWED_CLASSES})")

        pipeline_str = self._build_pipeline_string()

        try:
            self._pipeline = Gst.parse_launch(pipeline_str)
        except GLib.Error as e:
            logger.error(f"Pipeline-Erstellen fehlgeschlagen: {e}")
            raise RuntimeError(f"GStreamer Pipeline Error: {e}")

        # Identity Callback (Pad-Probe fuer Detection-Auswertung)
        identity = self._pipeline.get_by_name("identity_callback")
        if identity is None:
            raise RuntimeError("identity_callback Element nicht in Pipeline gefunden")
        pad = identity.get_static_pad("src")
        pad.add_probe(Gst.PadProbeType.BUFFER, self._on_buffer, None)

        # Face Attributes Probe: Tensor NACH hailonet, VOR Aggregator abgreifen
        # fattr_output_q ist die Queue zwischen hailonet und aggregator
        fattr_out_q = self._pipeline.get_by_name("fattr_output_q")
        if fattr_out_q:
            fattr_src_pad = fattr_out_q.get_static_pad("src")
            if fattr_src_pad:
                fattr_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_face_attr_buffer, None)
                logger.info("[FACE-ATTR] Pad-Probe auf fattr_output_q src registriert")
        else:
            logger.warning("[FACE-ATTR] fattr_output_q Element nicht gefunden")

        # appsink — Frames abholen damit Pipeline nicht blockiert
        appsink = self._pipeline.get_by_name("sink")
        if appsink:
            appsink.connect("new-sample", self._on_appsink_sample)

        # hailonet Referenzen fuer pass-through Steuerung durch Scheduler
        self._scrfd_hailonet_el = self._pipeline.get_by_name("scrfd_hailonet")
        self._arcface_hailonet_el = self._pipeline.get_by_name("arcface_hailonet")
        if self._scrfd_hailonet_el:
            logger.info("[NPU-SCHED] scrfd_hailonet gefunden — pass-through Steuerung aktiv")
        if self._arcface_hailonet_el:
            logger.info("[NPU-SCHED] arcface_hailonet gefunden — pass-through Steuerung aktiv")

        # Bus fuer Fehler/EOS
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        # FPS Reset
        self._fps_start = time.time()
        self._fps_last_time = self._fps_start
        self._fps_last_count = 0
        self._frame_count = 0

        # Pipeline starten
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("Pipeline konnte nicht gestartet werden")

        self._running = True

        # GLib MainLoop in eigenem Thread
        self._loop = GLib.MainLoop()
        self._loop_thread = threading.Thread(
            target=self._run_loop, name="TappasPipeline-GLib", daemon=True
        )
        self._loop_thread.start()

        logger.info("TAPPAS Pipeline gestartet")

    def stop(self):
        """Pipeline sauber beenden. Funktioniert auch wenn _running schon False (z.B. nach Bus-Error)."""
        was_running = self._running
        self._running = False

        if was_running:
            logger.info("Stoppe TAPPAS Pipeline...")
        else:
            logger.info("Raeume tote TAPPAS Pipeline auf...")

        if self._loop and self._loop.is_running():
            self._loop.quit()

        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5.0)

        self._pipeline = None
        self._loop = None
        self._loop_thread = None

        # SHM-Frame loeschen damit Panel "Kein Signal" zeigt statt Frozen Frame
        self._cleanup_shm()

        logger.info("TAPPAS Pipeline gestoppt + aufgeraeumt")

    def is_running(self) -> bool:
        return self._running

    def get_detections(self) -> List[Dict]:
        """Letzte Detections als Liste von Dicts.

        Returns:
            [{"class": "person"/"face", "bbox": [x1, y1, x2, y2],
              "confidence": float, "embedding": np.array or None,
              "track_id": int or None}]

        BBox-Koordinaten sind normalisiert (0.0-1.0).
        """
        with self._lock:
            return list(self._detections)

    def get_current_pframe(self) -> PerceptionFrame:
        """Letzten aggregierten PerceptionFrame zurueckgeben.

        Kompatibel mit InferenceEngine.get_current_pframe().
        """
        with self._lock:
            return self._current_pframe

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """Letzten annotierten Frame (RGB, mit Overlay) zurueckgeben."""
        with self._lock:
            return self._annotated_frame

    def get_fps(self) -> dict:
        """FPS-Snapshot zurueckgeben. Kompatibel mit InferenceEngine.get_fps()."""
        with self._fps_lock:
            return {
                "scrfd": self._current_fps,
                "arcface": self._current_fps,
                "yolov8m": self._current_fps,
                "total": self._current_fps,
                # Modelle laufen parallel in Pipeline — gleiche Frame-Rate
            }

    def get_npu_sched_mode(self) -> str:
        """Aktueller NPU Scheduler-Modus: YOLO_ONLY / YOLO_SCRFD / ALL_ACTIVE."""
        with self._sched_lock:
            return self._sched_mode

    def _update_npu_scheduler(self, has_person: bool, has_face: bool):
        """Scheduler-Modus basierend auf aktuellen Detections aktualisieren."""
        now = time.time()
        if has_person:
            self._sched_person_last_seen = now
        if has_face:
            self._sched_face_last_seen = now

        person_recent = (now - self._sched_person_last_seen) < SCHED_COOLDOWN_DOWN
        face_recent = (now - self._sched_face_last_seen) < SCHED_COOLDOWN_DOWN

        if face_recent:
            new_mode = SCHED_ALL_ACTIVE
        elif person_recent:
            new_mode = SCHED_YOLO_SCRFD
        else:
            new_mode = SCHED_YOLO_ONLY

        with self._sched_lock:
            if self._sched_mode != new_mode:
                logger.info(f"[NPU-SCHED] {self._sched_mode} → {new_mode}")
                self._sched_mode = new_mode

                # hailonet pass-through steuern (spart echte NPU-Zyklen)
                scrfd_pt = (new_mode == SCHED_YOLO_ONLY)
                arcface_pt = (new_mode != SCHED_ALL_ACTIVE)
                if self._scrfd_hailonet_el:
                    try:
                        self._scrfd_hailonet_el.set_property("pass-through", scrfd_pt)
                    except Exception as e:
                        logger.warning(f"[NPU-SCHED] scrfd pass-through Fehler: {e}")
                if self._arcface_hailonet_el:
                    try:
                        self._arcface_hailonet_el.set_property("pass-through", arcface_pt)
                    except Exception as e:
                        logger.warning(f"[NPU-SCHED] arcface pass-through Fehler: {e}")

                # Model-Active-Flags fuer GUI-Checkboxen + Status-JSON
                self.scrfd_active = not scrfd_pt
                self.arcface_active = not arcface_pt

    def reload_face_db(self, face_db: dict = None):
        """Face-DB aktualisieren.

        Wenn face_db=None (Service ruft ohne Parameter auf), wird die DB
        aus data/face_embeddings.json geladen (gleicher Weg wie InferenceEngine).
        """
        if face_db is None:
            face_db = self._load_face_db_from_disk()
        with self._face_db_lock:
            self._face_db = face_db
        logger.info(f"Face-DB aktualisiert: {len(self._face_db)} Personen")

    # =====================================================================
    # Live-Enrollment
    # =====================================================================

    def start_enrollment(self, name: str, n: int = 20):
        """Live-Enrollment starten: sammelt Embeddings aus GStreamer-Stream + Teachen/Snapshots.

        Ablauf:
        1. Sofort aktiv setzen (Panel sieht "laeuft")
        2. Background-Thread: Teachen-JSONs + Batch-Pipeline + Live-Stream
        3. Diversity-Filter waehlt beste 20 aus allen Kandidaten

        Embeddings kommen aus dem GLEICHEN Pipeline-Pfad wie die Erkennung
        (libvms_face_align.so → hailonet arcface → postprocess).

        Args:
            name: Personenname (z.B. "Markus")
            n: Maximale Anzahl Kandidaten zum Sammeln (beste 20 werden gespeichert)
        """
        with self._enroll_lock:
            if self._enroll_active:
                logger.warning("[ENROLLMENT] Laeuft bereits!")
                return
            self._enroll_name = name.lower()
            self._enroll_target = max(n, 60)
            self._enroll_candidates = []
            self._enroll_active = True
            self._enroll_done = False

        # Batch-Verarbeitung im Background-Thread (blockiert nicht IPC)
        threading.Thread(
            target=self._enrollment_batch_worker,
            args=(name,),
            daemon=True,
            name="EnrollmentBatch",
        ).start()

        logger.info(f"[ENROLLMENT] Gestartet fuer '{name}': "
                     f"Batch-Worker + Live-Stream (Score >{self._enroll_min_score})")

    def _enrollment_batch_worker(self, name: str):
        """Background-Worker: Teachen-JSONs Embeddings laden + Status loggen.

        Laeuft parallel zur Live-Sammlung aus _on_buffer.
        Teachen-JSON Embeddings (von frueheren Enrollments) werden thread-safe
        zu _enroll_candidates hinzugefuegt.

        Batch-Bildverarbeitung via separate GStreamer-Pipeline ist NICHT moeglich
        (Hailo-10H erlaubt nur EIN VDevice — Live-Pipeline belegt es).
        """
        try:
            # 1. Offline-Embeddings aus Teachen-JSONs laden
            offline_candidates = self._load_teachen_embeddings(name)
            if offline_candidates:
                with self._enroll_lock:
                    self._enroll_candidates.extend(offline_candidates)
                logger.info(f"[ENROLLMENT] {len(offline_candidates)} Teachen-JSON Embeddings geladen")

            # 2. Status der Bilder ohne Embeddings loggen
            self._log_enrollment_image_status(name)

            # Pruefen ob Target schon erreicht (Offline + Live zusammen)
            with self._enroll_lock:
                count = len(self._enroll_candidates)
                target = self._enroll_target

            if count >= target:
                self._finish_enrollment()

        except Exception as e:
            logger.error(f"[ENROLLMENT] Batch-Worker Fehler: {e}")
            import traceback
            traceback.print_exc()

    def _load_teachen_embeddings(self, name: str) -> list:
        """Embeddings aus Teachen-JSONs laden (von frueheren Enrollments gespeichert).

        Durchsucht /mnt/moloch-data/Teachen/<YYYY-MM-DD>/*.json nach Eintraegen
        mit passender Person und gespeichertem 'embedding' Feld.

        Returns: Liste von (score, embedding) Tupeln
        """
        import json
        import glob

        teachen_dir = "/mnt/moloch-data/Teachen"
        snap_dir = os.path.expanduser("~/moloch/snapshots")
        candidates = []
        name_lower = name.lower()
        teachen_files = 0
        snap_files = 0

        # Teachen-Ordner: Alle Tage durchsuchen
        if os.path.isdir(teachen_dir):
            for day_dir in os.listdir(teachen_dir):
                day_path = os.path.join(teachen_dir, day_dir)
                if not os.path.isdir(day_path) or not day_dir.startswith("20"):
                    continue
                for fname in os.listdir(day_path):
                    if not fname.lower().endswith(".json"):
                        continue
                    # Filename-Match: Name muss im Dateinamen vorkommen
                    if name_lower not in fname.lower():
                        continue
                    teachen_files += 1
                    json_path = os.path.join(day_path, fname)
                    try:
                        with open(json_path, 'r') as f:
                            meta = json.load(f)
                        emb_list = meta.get("embedding")
                        if emb_list and isinstance(emb_list, list):
                            emb = np.array(emb_list, dtype=np.float32)
                            norm = np.linalg.norm(emb)
                            if norm > 0:
                                emb = emb / norm
                            score = meta.get("confidence", 0.5)
                            candidates.append((score, emb))
                    except Exception:
                        pass

        logger.info(f"[ENROLLMENT] Teachen: {teachen_files} JSONs fuer '{name}', "
                     f"davon {len(candidates)} mit Embedding. "
                     f"Snapshots: {snap_files} Bilder")
        return candidates

    def _save_embeddings_to_teachen(self, name: str, selected: list):
        """Nach Enrollment: GStreamer-Embeddings in Teachen-JSONs zurueckschreiben.

        Damit koennen kuenftige Enrollments diese Embeddings wiederverwenden.
        Schreibt NUR in JSONs die zum Namen passen UND noch kein Embedding haben.
        """
        import json

        teachen_dir = "/mnt/moloch-data/Teachen"
        name_lower = name.lower()
        written = 0

        if not os.path.isdir(teachen_dir):
            return

        # Bestes Embedding (hoechster Score) zum Zurueckschreiben
        if not selected:
            return
        best_emb = selected[0][1]  # Score-sortiert, erstes = bestes

        for day_dir in os.listdir(teachen_dir):
            day_path = os.path.join(teachen_dir, day_dir)
            if not os.path.isdir(day_path) or not day_dir.startswith("20"):
                continue
            for fname in os.listdir(day_path):
                if not fname.lower().endswith(".json"):
                    continue
                if name_lower not in fname.lower():
                    continue
                json_path = os.path.join(day_path, fname)
                try:
                    with open(json_path, 'r') as f:
                        meta = json.load(f)
                    if meta.get("embedding"):
                        continue  # Schon vorhanden, nicht ueberschreiben
                    meta["embedding"] = best_emb.tolist()
                    meta["embedding_learned"] = True
                    with open(json_path, 'w') as f:
                        json.dump(meta, f, indent=2, ensure_ascii=False)
                    written += 1
                except Exception:
                    pass

        if written:
            logger.info(f"[ENROLLMENT] {written} Teachen-JSONs mit Embedding aktualisiert")

    def _log_enrollment_image_status(self, name: str):
        """Bilder aus Teachen + Snapshots zaehlen und Status loggen.

        HINWEIS: Bilder ohne Embeddings koennen NICHT als separate Batch-Pipeline
        verarbeitet werden, da der Hailo-10H nur EIN VDevice erlaubt (Live-Pipeline
        belegt es). Embeddings fuer neue Bilder werden beim naechsten Enrollment
        ueber den Live-Stream + _save_embeddings_to_teachen() in die JSONs geschrieben.
        """
        import json as _json
        name_lower = name.lower()
        teachen_without = 0
        snap_count = 0

        teachen_dir = "/mnt/moloch-data/Teachen"
        if os.path.isdir(teachen_dir):
            for day_dir in os.listdir(teachen_dir):
                day_path = os.path.join(teachen_dir, day_dir)
                if not os.path.isdir(day_path) or not day_dir.startswith("20"):
                    continue
                for fname in os.listdir(day_path):
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    if name_lower not in fname.lower():
                        continue
                    json_path = os.path.join(day_path, fname.rsplit(".", 1)[0] + ".json")
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r') as f:
                                meta = _json.load(f)
                            if meta.get("embedding"):
                                continue
                        except Exception:
                            pass
                    teachen_without += 1

        snap_dir = os.path.expanduser("~/moloch/snapshots")
        if os.path.isdir(snap_dir):
            for fname in os.listdir(snap_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    snap_count += 1

        if teachen_without > 0 or snap_count > 0:
            logger.info(f"[ENROLLMENT] {teachen_without} Teachen-Bilder ohne Embedding, "
                         f"{snap_count} Snapshots — Batch-NPU nicht moeglich "
                         f"(Single-VDevice), Embeddings werden ueber Live-Stream gesammelt")

    def get_enrollment_status(self) -> dict:
        """Enrollment-Status fuer IPC/Panel."""
        with self._enroll_lock:
            return {
                "active": self._enroll_active,
                "done": self._enroll_done,
                "name": self._enroll_name,
                "collected": len(self._enroll_candidates),
                "target": self._enroll_target,
            }

    def _collect_enrollment_embedding(self, embedding: np.ndarray, score: float):
        """Embedding-Kandidat fuer Enrollment sammeln (aus _on_buffer Callback).

        Filtert nach Score und Limit, ruft _finish_enrollment wenn genug da sind.
        """
        with self._enroll_lock:
            if not self._enroll_active:
                return
            if score < self._enroll_min_score:
                return
            # Embedding kopieren (Buffer wird wiederverwendet)
            emb = embedding.copy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            self._enroll_candidates.append((score, emb))
            count = len(self._enroll_candidates)
            target = self._enroll_target

        logger.info(f"[ENROLLMENT] Embedding #{count} gesammelt (score={score:.3f})")

        # Genug Kandidaten? → Enrollment abschliessen
        if count >= target:
            self._finish_enrollment()

    def _finish_enrollment(self):
        """Enrollment abschliessen: Diversitaets-Selektion + Face-DB speichern.

        Kandidaten koennen aus 2 Quellen stammen:
        1. Teachen-JSONs (Offline, von frueheren Enrollments via GStreamer)
        2. Live-Stream (GStreamer _on_buffer Callback)
        Diversity-Filter waehlt die besten 20 aus allen Kandidaten.
        """
        import json

        with self._enroll_lock:
            if not self._enroll_active:
                return
            self._enroll_active = False
            self._enroll_done = True
            name = self._enroll_name
            candidates = list(self._enroll_candidates)
            diversity_thresh = self._enroll_diversity

        if not candidates:
            logger.warning("[ENROLLMENT] Keine Embeddings gesammelt!")
            return

        n_total = len(candidates)

        # Nach Score sortieren (hoechster zuerst)
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Greedy diverse selection
        selected = []
        for score, emb in candidates:
            if len(selected) >= 20:
                break
            is_diverse = all(
                float(np.dot(emb, sel_emb)) < diversity_thresh
                for _, sel_emb in selected
            )
            if is_diverse:
                selected.append((score, emb))

        # Falls nicht genug diverse: Rest nach Score auffuellen
        if len(selected) < 20:
            selected_embs = {id(emb) for _, emb in selected}
            for score, emb in candidates:
                if len(selected) >= 20:
                    break
                if id(emb) not in selected_embs:
                    selected.append((score, emb))

        logger.info(f"[ENROLLMENT] {n_total} Kandidaten → "
                     f"{len(selected)} diverse Embeddings ausgewaehlt fuer '{name}'")

        # Face-DB laden, alte Eintraege entfernen, neue speichern
        embeddings_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "face_embeddings.json"
        )
        db = {}
        if os.path.exists(embeddings_path):
            try:
                with open(embeddings_path, 'r') as f:
                    db = json.load(f)
            except Exception:
                pass

        # Alte Eintraege fuer diesen Namen entfernen
        old_keys = [k for k in db if k.split('#')[0].lower() == name]
        for k in old_keys:
            del db[k]
        logger.info(f"[ENROLLMENT] {len(old_keys)} alte '{name}' Eintraege entfernt")

        # Neues Haupt-Embedding + Varianten speichern
        db[name] = selected[0][1].tolist()
        for i, (score, emb) in enumerate(selected[1:]):
            db[f"{name}#snap_{i}"] = emb.tolist()

        # Atomar speichern
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        tmp = embeddings_path + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(db, f, indent=1, ensure_ascii=False)
        os.replace(tmp, embeddings_path)

        logger.info(f"[ENROLLMENT] Face-DB gespeichert: {len(selected)} Embeddings fuer '{name}'")

        # Embeddings in Teachen-JSONs zurueckschreiben (fuer kuenftige Enrollments)
        self._save_embeddings_to_teachen(name, selected)

        # Face-DB neu laden (Best-Match)
        self.reload_face_db()

    def sync_flags_from_npu(self):
        """No-op: TAPPAS Pipeline hat alle Modelle permanent aktiv."""
        pass

    def reset_fps(self):
        """FPS-Counter zuruecksetzen (z.B. nach Stream-Restart)."""
        with self._fps_lock:
            self._frame_count = 0
            self._fps_start = time.time()
            self._fps_last_count = 0
            self._fps_last_time = self._fps_start
            self._current_fps = 0.0

    # =====================================================================
    # Face-DB von Disk laden
    # =====================================================================

    def _load_face_db_from_disk(self) -> dict:
        """Face-Embeddings aus data/face_embeddings.json laden.

        Gruppiert nach Person (Name vor '#') und bildet Durchschnitt.
        z.B. "Markus", "Markus#snap_1", "Markus#train_42_0" → ein Embedding fuer "Markus".
        Returns: {person_name: np.array} oder {} bei Fehler.
        """
        import json
        embeddings_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "face_embeddings.json"
        )
        if not os.path.exists(embeddings_path):
            logger.warning(f"Face-DB nicht gefunden: {embeddings_path}")
            return {}
        try:
            with open(embeddings_path, 'r') as f:
                raw = json.load(f)
            # Gruppiere nach Person (Name vor '#')
            groups = {}
            for key, emb_list in raw.items():
                person = key.split('#')[0].lower()
                emb = np.array(emb_list, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                if person not in groups:
                    groups[person] = []
                groups[person].append(emb)
            # Alle Embeddings pro Person speichern (Best-Match statt Mean)
            db = {}
            for person, embs in groups.items():
                db[person] = embs  # Liste von normalisierten Embeddings
            total_embs = sum(len(e) for e in groups.values())
            logger.info(f"Face-DB geladen: {len(db)} Personen aus {total_embs} Embeddings ({embeddings_path})")
            return db
        except Exception as e:
            logger.error(f"Face-DB laden fehlgeschlagen: {e}")
            return {}

    # =====================================================================
    # GStreamer Pipeline String
    # =====================================================================

    def _build_pipeline_string(self) -> str:
        """Komplette Multi-Model Pipeline (YOLO + SCRFD + Tracker + ArcFace)."""

        # --- Source: RTSP → H264 depay → decode → scale → RGB ---
        # retry=5: rtspsrc versucht 5x reconnect bei Verbindungsverlust
        # timeout=5000000: 5s Timeout (Microsekunden)
        source = (
            f'rtspsrc location="{self._rtsp_url}" name=source latency=300 protocols=tcp '
            f'retry=5 timeout=5000000 tcp-timeout=5000000 ! '
            f'rtph264depay name=source_depay ! '
            f'queue name=source_queue_decode leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! '
            f'avdec_h264 name=source_decode ! '
            f'queue name=source_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'videoscale name=source_videoscale n-threads=2 ! '
            f'queue name=source_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'videoconvert n-threads=3 name=source_convert qos=false ! '
            f'video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width={self._width}, height={self._height} '
        )

        # --- Stage 1: YOLO Person Detection ---
        yolo_inner = (
            f'queue name=yolo_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'videoscale name=yolo_videoscale n-threads=2 qos=false ! '
            f'queue name=yolo_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'video/x-raw, pixel-aspect-ratio=1/1 ! '
            f'videoconvert name=yolo_videoconvert n-threads=2 ! '
            f'queue name=yolo_hailonet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailonet name=yolo_hailonet hef-path={YOLO_HEF} batch-size=1 '
            f'vdevice-group-id={VDEVICE_GROUP_ID} '
            f'nms-score-threshold=0.3 nms-iou-threshold=0.45 '
            f'output-format-type=HAILO_FORMAT_TYPE_FLOAT32 '
            f'force-writable=true ! '
            f'queue name=yolo_hailofilter_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailofilter name=yolo_hailofilter so-path={YOLO_POSTPROCESS_SO} '
            f'function-name={YOLO_POSTPROCESS_FUNC} qos=false ! '
            f'queue name=yolo_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
        )

        yolo_wrapper = (
            f'queue name=yolo_wrapper_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailocropper name=yolo_wrapper_crop so-path={WHOLE_BUFFER_SO} function-name=create_crops '
            f'use-letterbox=true resize-method=inter-area internal-offset=true '
            f'hailoaggregator name=yolo_wrapper_agg '
            f'yolo_wrapper_crop. ! queue name=yolo_wrapper_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0 ! yolo_wrapper_agg.sink_0 '
            f'yolo_wrapper_crop. ! {yolo_inner} ! yolo_wrapper_agg.sink_1 '
            f'yolo_wrapper_agg. ! queue name=yolo_wrapper_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
        )

        # --- Stage 2: SCRFD Face Detection ---
        scrfd_inner = (
            f'queue name=scrfd_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'videoscale name=scrfd_videoscale n-threads=2 qos=false ! '
            f'queue name=scrfd_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'video/x-raw, pixel-aspect-ratio=1/1 ! '
            f'videoconvert name=scrfd_videoconvert n-threads=2 ! '
            f'queue name=scrfd_hailonet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailonet name=scrfd_hailonet hef-path={SCRFD_HEF} batch-size=1 '
            f'vdevice-group-id={VDEVICE_GROUP_ID} '
            f'force-writable=true ! '
            f'queue name=scrfd_hailofilter_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailofilter name=scrfd_hailofilter so-path={SCRFD_POSTPROCESS_SO} '
            f'function-name={SCRFD_POSTPROCESS_FUNC} config-path={SCRFD_CONFIG_JSON} qos=false ! '
            f'queue name=scrfd_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
        )

        scrfd_wrapper = (
            f'queue name=scrfd_wrapper_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailocropper name=scrfd_wrapper_crop so-path={WHOLE_BUFFER_SO} function-name=create_crops '
            f'use-letterbox=true resize-method=inter-area internal-offset=true '
            f'hailoaggregator name=scrfd_wrapper_agg '
            f'scrfd_wrapper_crop. ! queue name=scrfd_wrapper_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0 ! scrfd_wrapper_agg.sink_0 '
            f'scrfd_wrapper_crop. ! {scrfd_inner} ! scrfd_wrapper_agg.sink_1 '
            f'scrfd_wrapper_agg. ! queue name=scrfd_wrapper_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
        )

        # --- Stage 3: Tracker + Face Cropper (face_align + ArcFace) ---
        tracker = (
            f'hailotracker name=hailo_face_tracker class-id=-1 '
            f'kalman-dist-thr=0.7 iou-thr=0.8 init-iou-thr=0.9 '
            f'keep-new-frames=2 keep-tracked-frames=6 keep-lost-frames=8 '
            f'keep-past-metadata=true qos=false ! '
            f'queue name=tracker_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
        )

        arcface_inner = (
            f'hailofilter so-path={FACE_ALIGN_SO} name=face_align_hailofilter use-gst-buffer=true qos=false ! '
            f'queue name=face_align_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'queue name=arcface_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'videoscale name=arcface_videoscale n-threads=2 qos=false ! '
            f'queue name=arcface_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'video/x-raw, pixel-aspect-ratio=1/1 ! '
            f'videoconvert name=arcface_videoconvert n-threads=2 ! '
            f'queue name=arcface_hailonet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailonet name=arcface_hailonet hef-path={ARCFACE_HEF} batch-size=1 '
            f'vdevice-group-id={VDEVICE_GROUP_ID} '
            f'force-writable=true ! '
            f'queue name=arcface_hailofilter_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailofilter name=arcface_hailofilter so-path={ARCFACE_POSTPROCESS_SO} '
            f'function-name={ARCFACE_POSTPROCESS_FUNC} qos=false ! '
            f'queue name=arcface_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
        )

        face_cropper = (
            f'queue name=face_crop_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailocropper name=face_cropper so-path={FACE_CROP_SO} function-name={FACE_CROP_FUNC} '
            f'use-letterbox=true no-scaling-bbox=true internal-offset=true resize-method=bilinear '
            f'hailoaggregator name=face_crop_agg '
            f'face_cropper. ! queue name=face_crop_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0 ! face_crop_agg.sink_0 '
            f'face_cropper. ! {arcface_inner} ! face_crop_agg.sink_1 '
            f'face_crop_agg. ! queue name=face_crop_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
        )

        # --- Stage 4: Face Attributes (gender/smiling via face_attr_resnet_v1_18) ---
        # Zweiter Face-Cropper: gleiche Detections, resize auf 178x218, kein Postprocess
        face_attr_inner = (
            f'queue name=fattr_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'videoscale name=fattr_videoscale n-threads=2 qos=false ! '
            f'queue name=fattr_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'video/x-raw, pixel-aspect-ratio=1/1 ! '
            f'videoconvert name=fattr_videoconvert n-threads=2 ! '
            f'queue name=fattr_hailonet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailonet name=fattr_hailonet hef-path={FACE_ATTR_HEF} batch-size=1 '
            f'vdevice-group-id={VDEVICE_GROUP_ID} '
            f'force-writable=true ! '
            f'queue name=fattr_hailofilter_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailofilter name=fattr_hailofilter so-path={FACE_ATTR_POSTPROCESS_SO} '
            f'function-name={FACE_ATTR_POSTPROCESS_FUNC} qos=false ! '
            f'queue name=fattr_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
        )

        face_attr_cropper = (
            f'queue name=fattr_crop_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailocropper name=fattr_cropper so-path={FACE_CROP_SO} function-name={FACE_CROP_FUNC} '
            f'use-letterbox=true no-scaling-bbox=true internal-offset=true resize-method=bilinear '
            f'hailoaggregator name=fattr_crop_agg '
            f'fattr_cropper. ! queue name=fattr_crop_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0 ! fattr_crop_agg.sink_0 '
            f'fattr_cropper. ! {face_attr_inner} ! fattr_crop_agg.sink_1 '
            f'fattr_crop_agg. ! queue name=fattr_crop_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
        )

        # --- Callback + Overlay + appsink ---
        callback_and_sink = (
            f'queue name=cb_q leaky=downstream max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! '
            f'identity name=identity_callback ! '
            f'queue name=overlay_q leaky=downstream max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! '
            f'hailooverlay name=hailo_overlay ! '
            f'queue name=sink_convert_q leaky=downstream max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! '
            f'videoconvert n-threads=2 qos=false ! '
            f'video/x-raw, format=RGB ! '
            f'appsink name=sink emit-signals=true drop=true max-buffers=1 sync=false'
        )

        return (
            f'{source} ! '
            f'{yolo_wrapper} ! '
            f'{scrfd_wrapper} ! '
            f'{tracker} ! '
            f'{face_cropper} ! '
            f'{face_attr_cropper} ! '
            f'{callback_and_sink}'
        )

    # =====================================================================
    # GStreamer Callbacks
    # =====================================================================

    def _on_face_attr_buffer(self, pad, info, user_data):
        """Pad-Probe nach fattr_hailofilter — liest HAILO_CLASSIFICATION (gender/smiling)."""
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        try:
            roi = hailo.get_roi_from_buffer(buffer)
            classifications = roi.get_objects_typed(hailo.HAILO_CLASSIFICATION)
            if not classifications:
                return Gst.PadProbeReturn.OK

            gender = None
            smiling = False
            self._fattr_probe_count = getattr(self, '_fattr_probe_count', 0) + 1

            for c in classifications:
                label = c.get_label()
                if label == "Male":
                    gender = "M"
                elif label == "Female":
                    gender = "F"
                elif label == "Smiling":
                    smiling = True

            if gender is not None:
                with self._face_attr_lock:
                    self._face_attr_cache = {"gender": gender, "smiling": smiling}

            if self._fattr_probe_count % 100 == 1:
                cls_info = [(c.get_label(), round(c.get_confidence(), 2))
                            for c in classifications]
                logger.info(f"[FACE-ATTR] #{self._fattr_probe_count}: gender={gender} "
                            f"smiling={smiling} attrs={cls_info[:6]}")

        except Exception as e:
            self._fattr_err_count = getattr(self, '_fattr_err_count', 0) + 1
            if self._fattr_err_count % 100 == 1:
                logger.error(f"[FACE-ATTR] Probe-Fehler: {e}")
        return Gst.PadProbeReturn.OK

    def _on_buffer(self, pad, info, user_data):
        """Pad-Probe auf identity element — extrahiert Detections + baut PerceptionFrame."""
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        roi = hailo.get_roi_from_buffer(buffer)
        hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        detections = []
        persons = []
        faces = []
        best_face_conf = 0.0
        best_face_bbox = None
        face_id = None
        face_similarity = 0.0

        for det in hailo_detections:
            label = det.get_label()
            conf = det.get_confidence()
            bbox = det.get_bbox()

            # YOLO-Klassenfilter: nur erlaubte Klassen durchlassen
            if label != "face" and label not in YOLO_ALLOWED_CLASSES:
                continue

            # Threshold-Filterung (Panel-Slider Werte anwenden)
            if label in YOLO_ALLOWED_CLASSES and conf < self.yolo_conf_val:
                continue
            if label == "face" and conf < self.scrfd_conf_val:
                continue

            # Normalisierte BBox [0.0-1.0]
            x1 = bbox.xmin()
            y1 = bbox.ymin()
            x2 = bbox.xmax()
            y2 = bbox.ymax()

            entry = {
                "class": label,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "embedding": None,
                "track_id": None,
            }

            # Track-ID aus hailotracker
            track_ids = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if track_ids:
                entry["track_id"] = track_ids[0].get_id()

            if label == "person":
                persons.append(entry)
            elif label == "face":
                # ArcFace Embedding extrahieren
                embeddings = det.get_objects_typed(hailo.HAILO_MATRIX)
                if embeddings:
                    emb_data = np.array(embeddings[0].get_data(), dtype=np.float32)
                    entry["embedding"] = emb_data

                    # Live-Enrollment: Embedding sammeln wenn aktiv
                    if self._enroll_active:
                        self._collect_enrollment_embedding(emb_data, conf)

                    # Face-Matching gegen DB
                    matched_name, matched_sim = self._match_face(emb_data)
                    # Debug-Log (alle 50 Frames): SCRFD-Score vs ArcFace-Similarity klar trennen
                    self._match_log_count = getattr(self, '_match_log_count', 0) + 1
                    if self._match_log_count % 50 == 1:
                        logger.info(f"[FACE-MATCH] SCRFD={conf:.3f} ArcFace={matched_sim:.3f} "
                                    f"thresh={self.arcface_thresh_val:.2f} → "
                                    f"{'✓ ' + matched_name if matched_name else '✗ kein Match'} "
                                    f"(db={len(self._face_db)} emb={len(emb_data)})")
                    if matched_name:
                        entry["face_id"] = matched_name
                        entry["face_similarity"] = matched_sim

                # Face Attributes aus Cache (befuellt von _on_face_attr_buffer Probe)
                with self._face_attr_lock:
                    if self._face_attr_cache:
                        entry["gender"] = self._face_attr_cache.get("gender")
                        entry["smiling"] = self._face_attr_cache.get("smiling", False)

                faces.append(entry)

                if conf > best_face_conf:
                    best_face_conf = conf
                    best_face_bbox = (x1, y1, x2, y2)

            detections.append(entry)

        # Bestes Face-Match fuer PerceptionFrame
        for f in faces:
            if f.get("face_similarity", 0) > face_similarity:
                face_id = f.get("face_id")
                face_similarity = f.get("face_similarity", 0)

        # --- NPU Model-Scheduler: Modus aktualisieren + Ergebnisse unterdrücken ---
        self._update_npu_scheduler(len(persons) > 0, len(faces) > 0)
        sched_mode = self.get_npu_sched_mode()
        if sched_mode == SCHED_YOLO_ONLY:
            # SCRFD/ArcFace Ergebnisse ignorieren (Pipeline laeuft, Daten werden verworfen)
            faces = []
            best_face_conf = 0.0
            best_face_bbox = None
            face_id = None
            face_similarity = 0.0
        elif sched_mode == SCHED_YOLO_SCRFD:
            # ArcFace Matching ignorieren
            face_id = None
            face_similarity = 0.0

        # PerceptionFrame bauen
        pf = self._build_pframe(persons, faces, best_face_conf,
                                best_face_bbox, face_id, face_similarity)

        # Thread-safe update (Frame kommt aus appsink — NACH hailooverlay)
        with self._lock:
            self._detections = detections
            self._current_pframe = pf

        # --- Perception Events auf Event Bus publishen (fuer Action Bridge) ---
        self._publish_perception_events(persons, faces, face_id, face_similarity)

        # FPS Tracking
        self._update_fps()

        return Gst.PadProbeReturn.OK

    def _publish_perception_events(self, persons: list, faces: list,
                                       face_id: Optional[str], face_similarity: float):
        """Perception-Events auf Event Bus publishen (fuer Action Bridge FSM).

        Events:
          perception.person_detected  — Person erkannt (YOLO)
          perception.face_confirmed   — Gesicht erkannt (SCRFD)
          perception.owner_detected   — Owner erkannt (ArcFace Match)
          perception.target_lost      — Keine Person mehr im Frame
        """
        has_person = len(persons) > 0 or len(faces) > 0

        if has_person:
            # Beste Person-BBox fuer Tracking
            best_bbox = [0, 0, 0, 0]
            best_conf = 0.0
            if persons:
                best = max(persons, key=lambda p: p["confidence"])
                best_bbox = best["bbox"]
                best_conf = best["confidence"]

            self._event_bus.publish(
                event_type="perception.person_detected",
                payload={"confidence": best_conf, "bbox": best_bbox,
                         "count": len(persons)},
                source="tappas_pipeline",
                priority=PRIO_PERCEPTION,
            )

            # Face confirmed (SCRFD hat Gesicht gefunden)
            if faces:
                best_face = max(faces, key=lambda f: f["confidence"])
                self._event_bus.publish(
                    event_type="perception.face_confirmed",
                    payload={"confidence": best_face["confidence"],
                             "bbox": best_face["bbox"],
                             "similarity": face_similarity or 0.0},
                    source="tappas_pipeline",
                    priority=PRIO_PERCEPTION,
                )

            # Owner detected (ArcFace Match ueber Threshold)
            if face_id and face_similarity >= self.arcface_thresh_val:
                self._event_bus.publish(
                    event_type="perception.owner_detected",
                    payload={"name": face_id, "similarity": face_similarity},
                    source="tappas_pipeline",
                    priority=PRIO_PERCEPTION,
                )

        # Target lost: War Person da, jetzt nicht mehr
        if self._last_person_state and not has_person:
            self._event_bus.publish(
                event_type="perception.target_lost",
                payload={"reason": "no_detection"},
                source="tappas_pipeline",
                priority=PRIO_PERCEPTION,
            )

        self._last_person_state = has_person

    def _on_appsink_sample(self, appsink):
        """appsink Callback — annotiertes Frame (MIT BBoxen von hailooverlay) extrahieren."""
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        caps = sample.get_caps()
        if buf is None or caps is None:
            return Gst.FlowReturn.OK

        struct = caps.get_structure(0)
        width = struct.get_value("width")
        height = struct.get_value("height")

        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK
        try:
            data = np.frombuffer(mapinfo.data, dtype=np.uint8).copy()
        finally:
            buf.unmap(mapinfo)

        expected = width * height * 3
        if len(data) != expected:
            return Gst.FlowReturn.OK

        frame = data.reshape(height, width, 3)

        # Thread-safe: annotiertes Frame speichern
        with self._lock:
            self._annotated_frame = frame

        # SHM IPC: Annotiertes Frame (MIT BBoxen) fuer Panel Preview
        self._write_shm_frame(frame)

        return Gst.FlowReturn.OK

    def _on_bus_message(self, bus, message):
        """GStreamer Bus Messages verarbeiten. Raeumt auf bei ERROR/EOS."""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"[BUS] GStreamer Fehler: {err}")
            logger.error(f"[BUS] Debug: {debug}")
            self._running = False
            # Pipeline auf NULL setzen damit NPU-Resources freigegeben werden
            if self._pipeline:
                try:
                    self._pipeline.set_state(Gst.State.NULL)
                except Exception:
                    pass
            # SHM loeschen damit Panel "Kein Signal" zeigt
            self._cleanup_shm()
            if self._loop and self._loop.is_running():
                self._loop.quit()
        elif t == Gst.MessageType.EOS:
            logger.warning("[BUS] End-of-Stream — Pipeline beendet")
            self._running = False
            if self._pipeline:
                try:
                    self._pipeline.set_state(Gst.State.NULL)
                except Exception:
                    pass
            self._cleanup_shm()
            if self._loop and self._loop.is_running():
                self._loop.quit()
        return True

    def _run_loop(self):
        """GLib MainLoop in eigenem Thread ausfuehren."""
        try:
            self._loop.run()
        except Exception as e:
            logger.error(f"GLib MainLoop Fehler: {e}")
        finally:
            self._running = False

    # =====================================================================
    # Frame-Extraktion
    # =====================================================================

    def _extract_frame(self, buffer, pad) -> Optional[np.ndarray]:
        """Extrahiert numpy Frame (RGB) aus GStreamer Buffer."""
        caps = pad.get_current_caps()
        if caps is None:
            return None
        struct = caps.get_structure(0)
        width = struct.get_value("width")
        height = struct.get_value("height")

        success, mapinfo = buffer.map(Gst.MapFlags.READ)
        if not success:
            return None
        try:
            data = np.frombuffer(mapinfo.data, dtype=np.uint8).copy()
        finally:
            buffer.unmap(mapinfo)

        expected = width * height * 3
        if len(data) != expected:
            return None
        return data.reshape(height, width, 3)

    # =====================================================================
    # Face Matching
    # =====================================================================

    def _match_face(self, embedding: np.ndarray) -> tuple:
        """Face-Embedding gegen DB matchen.

        Returns:
            (name, similarity) oder (None, 0.0)
        """
        with self._face_db_lock:
            if not self._face_db:
                return (None, 0.0)

            # Embedding normalisieren
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            best_name = None
            best_sim = 0.0
            threshold = self.arcface_thresh_val  # Panel-Slider Wert

            for name, db_embs in self._face_db.items():
                # db_embs ist Liste von Embeddings (Best-Match statt Mean)
                if isinstance(db_embs, np.ndarray) and db_embs.ndim == 1:
                    db_embs = [db_embs]  # Rueckwaerts-Kompatibilitaet
                for db_emb in db_embs:
                    sim = float(np.dot(embedding, db_emb))
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name

            if best_sim >= threshold:
                logger.debug(f"[FACE-MATCH] {best_name} sim={best_sim:.3f} (thresh={threshold:.2f})")
                return (best_name, best_sim)
            logger.debug(f"[FACE-MATCH] KEIN Match: best={best_name} sim={best_sim:.3f} < thresh={threshold:.2f}")
            return (None, best_sim)

    # =====================================================================
    # PerceptionFrame Builder
    # =====================================================================

    def _build_pframe(self, persons: list, faces: list, best_face_conf: float,
                      best_face_bbox: tuple, face_id: str, face_similarity: float) -> PerceptionFrame:
        """Baut PerceptionFrame aus Detections — kompatibel mit InferenceEngine."""
        pf = PerceptionFrame()
        pf.timestamp = time.time()

        # Person Detection
        pf.person_detected = len(persons) > 0 or len(faces) > 0
        pf.person_count = len(persons) if persons else (1 if faces else 0)

        # Distanz aus groesster Person-BBox
        if persons:
            biggest = max(persons, key=lambda d: (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]))
            bbox = biggest["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            pf.distance_ratio = area
            pf.distance = estimate_distance(area)
        elif best_face_bbox:
            area = (best_face_bbox[2] - best_face_bbox[0]) * (best_face_bbox[3] - best_face_bbox[1])
            pf.distance_ratio = area
            pf.distance = estimate_distance(area)

        # Face Detection
        pf.face_detected = len(faces) > 0
        pf.face_count = len(faces)
        if faces:
            pf.face_confidence = best_face_conf
            pf.face_bbox = best_face_bbox

        # Face Recognition
        if face_id:
            pf.face_id = face_id.lower()
            pf.face_similarity = face_similarity

        # Face Attributes (bestes Face mit Attributen)
        if faces:
            best_attr_face = max(faces, key=lambda f: f.get("confidence", 0))
            if best_attr_face.get("gender"):
                pf.gender = best_attr_face["gender"]
            if best_attr_face.get("smiling") is not None:
                pf.emotion = "happy" if best_attr_face["smiling"] else "neutral"

        # Active Models (Scheduler-basiert: logisch aktiv, nicht nur NPU-seitig)
        sched = self.get_npu_sched_mode()
        if sched == SCHED_ALL_ACTIVE:
            pf.active_models = ["yolov8m", "scrfd", "arcface"]
        elif sched == SCHED_YOLO_SCRFD:
            pf.active_models = ["yolov8m", "scrfd"]
        else:
            pf.active_models = ["yolov8m"]

        return pf

    # =====================================================================
    # FPS
    # =====================================================================

    def _update_fps(self):
        """FPS alle 2 Sekunden aktualisieren."""
        with self._fps_lock:
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._fps_last_time
            if elapsed >= 2.0:
                frames = self._frame_count - self._fps_last_count
                self._current_fps = frames / elapsed
                self._fps_last_time = now
                self._fps_last_count = self._frame_count

    # =====================================================================
    # SHM IPC (Panel Preview)
    # =====================================================================

    def _cleanup_shm(self):
        """SHM-Frame loeschen damit Panel sofort 'Kein Signal' zeigt."""
        for path in [SHM_FRAME_PATH, SHM_FRAME_PATH + '.tmp']:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

    def _write_shm_frame(self, frame: np.ndarray):
        """Frame nach /dev/shm/moloch_frame schreiben (RGB direkt, kein BGR-Umweg).

        GStreamer liefert RGB, resize auf Preview-Groesse hier in Python.
        24-Byte Header: h, w, c, seq (uint32 LE) + timestamp (float64 LE).
        """
        try:
            h, w = frame.shape[:2]
            if h != SHM_PREVIEW_H or w != SHM_PREVIEW_W:
                frame = cv2.resize(frame, (SHM_PREVIEW_W, SHM_PREVIEW_H))
                h, w = SHM_PREVIEW_H, SHM_PREVIEW_W
            c = frame.shape[2] if len(frame.shape) > 2 else 1
            self._shm_seq = (self._shm_seq + 1) & 0xFFFFFFFF
            ts = time.monotonic()
            header = struct.pack('<IIIId', h, w, c, self._shm_seq, ts)
            with open(SHM_FRAME_PATH + '.tmp', 'wb') as f:
                f.write(header)
                f.write(frame.tobytes())
            os.rename(SHM_FRAME_PATH + '.tmp', SHM_FRAME_PATH)
        except Exception:
            pass
