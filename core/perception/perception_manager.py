#!/usr/bin/env python3
"""
M.O.L.O.C.H. Perception Manager
===============================
Unified perception stack for Hailo-10H NPU.

Modes:
    PERSON_TRACKING - YOLOv8 person detection
    FACE_RECOGNITION - Face detection + embedding matching
    GESTURE_MODE - Hand/pose detection
    EMOTION_MODE - Emotion classification (requires stable face)

Only ONE mode active at a time.
Mode switch only via MPO.
"""

import time
import json
import logging
import threading
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Hailo model directory (secondary SSD)
HAILO_MODELS_DIR = "/mnt/moloch-data/hailo/models"


class PerceptionMode(Enum):
    IDLE = "idle"
    PERSON_TRACKING = "person_tracking"
    FACE_RECOGNITION = "face_recognition"
    GESTURE_MODE = "gesture_mode"
    EMOTION_MODE = "emotion_mode"


@dataclass
class PerceptionResult:
    """Result from perception inference."""
    mode: PerceptionMode
    timestamp: float
    inference_time_ms: float
    fps: float
    detections: List[Dict] = field(default_factory=list)
    recognition_result: Optional[str] = None
    confidence: float = 0.0
    error: str = ""
    
    def to_log(self) -> str:
        ts = datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S.%f")[:-3]
        n_det = len(self.detections)
        return f"[PERCEPTION {ts}] mode={self.mode.value} det={n_det} inf={self.inference_time_ms:.1f}ms fps={self.fps:.1f}"


@dataclass
class FaceEmbedding:
    """Stored face embedding."""
    name: str
    embedding: np.ndarray
    created_at: float
    samples: int = 1


class PerceptionManager:
    """
    Unified perception manager for Hailo-10H.
    
    Models:
        - Person: yolov8m_h10.hef
        - Face Detection: scrfd_10g.hef
        - Face Embedding: arcface_mobilefacenet.hef
        - Pose: yolov8s_pose_h10.hef
    """
    
    # Model paths (using secondary SSD)
    MODELS = {
        "person": f"{HAILO_MODELS_DIR}/yolov8m_h10.hef",
        "face_detect": "/usr/local/hailo/resources/models/hailo10h/scrfd_10g.hef",
        "face_embed": "/usr/local/hailo/resources/models/hailo10h/arcface_mobilefacenet.hef",
        "pose": f"{HAILO_MODELS_DIR}/yolov8s_pose_h10.hef",
    }
    
    # Thresholds
    PERSON_CONFIDENCE = 0.6
    FACE_CONFIDENCE = 0.5
    FACE_MATCH_THRESHOLD = 0.6  # Cosine similarity
    ENROLL_FRAMES = 10
    
    # Database path
    EMBEDDINGS_PATH = Path("/home/molochzuhause/moloch/data/face_embeddings.json")
    
    def __init__(self):
        self._mode = PerceptionMode.IDLE
        self._lock = threading.Lock()
        self._vdevice = None
        self._current_hef = None
        self._current_model_name = None
        self._infer_model = None
        
        # FPS tracking
        self._frame_times: List[float] = []
        self._last_inference_time = 0.0
        
        # Face database
        self._face_db: Dict[str, FaceEmbedding] = {}
        self._load_face_db()
        
        # Enrollment state
        self._enrolling = False
        self._enroll_name = ""
        self._enroll_embeddings: List[np.ndarray] = []
        
        # Callbacks
        self._on_detection: Optional[Callable] = None
        self._on_recognition: Optional[Callable] = None
        
        logger.info("[PERCEPTION] Manager initialized")
    
    def _load_face_db(self):
        """Load face embeddings from disk."""
        if self.EMBEDDINGS_PATH.exists():
            try:
                with open(self.EMBEDDINGS_PATH) as f:
                    data = json.load(f)
                for name, item in data.items():
                    self._face_db[name] = FaceEmbedding(
                        name=name,
                        embedding=np.array(item["embedding"]),
                        created_at=item["created_at"],
                        samples=item.get("samples", 1)
                    )
                logger.info(f"[PERCEPTION] Loaded {len(self._face_db)} face embeddings")
            except Exception as e:
                logger.error(f"[PERCEPTION] Failed to load face DB: {e}")
    
    def _save_face_db(self):
        """Save face embeddings to disk."""
        self.EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for name, fe in self._face_db.items():
            data[name] = {
                "embedding": fe.embedding.tolist(),
                "created_at": fe.created_at,
                "samples": fe.samples
            }
        with open(self.EMBEDDINGS_PATH, "w") as f:
            json.dump(data, f)
        logger.info(f"[PERCEPTION] Saved {len(data)} face embeddings")
    
    def connect(self) -> bool:
        """Connect to Hailo device."""
        with self._lock:
            if self._vdevice is not None:
                return True
            try:
                from hailo_platform import VDevice
                self._vdevice = VDevice()
                logger.info("[PERCEPTION] Connected to Hailo device")
                return True
            except Exception as e:
                logger.error(f"[PERCEPTION] Failed to connect to Hailo: {e}")
                return False
    
    def disconnect(self):
        """Disconnect from Hailo device."""
        with self._lock:
            self._unload_model()
            if self._vdevice:
                del self._vdevice
                self._vdevice = None
            logger.info("[PERCEPTION] Disconnected from Hailo")
    
    def _load_model(self, model_name: str) -> bool:
        """Load HEF model onto Hailo."""
        if model_name not in self.MODELS:
            logger.error(f"[PERCEPTION] Unknown model: {model_name}")
            return False
        
        model_path = self.MODELS[model_name]
        if not Path(model_path).exists():
            logger.error(f"[PERCEPTION] Model not found: {model_path}")
            return False
        
        # Unload current model if different
        if self._current_model_name == model_name:
            return True
        
        self._unload_model()
        
        try:
            from hailo_platform import HEF
            
            self._current_hef = HEF(model_path)
            self._infer_model = self._vdevice.create_infer_model(model_path)
            self._infer_model.set_batch_size(1)
            self._current_model_name = model_name
            
            logger.info(f"[PERCEPTION] Loaded model: {model_name} ({model_path})")
            return True
        except Exception as e:
            logger.error(f"[PERCEPTION] Failed to load model {model_name}: {e}")
            return False
    
    def _unload_model(self):
        """Unload current model."""
        if self._infer_model:
            del self._infer_model
            self._infer_model = None
        if self._current_hef:
            del self._current_hef
            self._current_hef = None
        self._current_model_name = None
    
    def set_mode(self, mode: PerceptionMode) -> bool:
        """Set perception mode. Loads appropriate model."""
        if mode == self._mode:
            return True
        
        old_mode = self._mode
        logger.info(f"[PERCEPTION MODE] {old_mode.value} -> {mode.value}")
        
        with self._lock:
            if not self._vdevice:
                if not self.connect():
                    return False
            
            # Load model for mode
            model_map = {
                PerceptionMode.PERSON_TRACKING: "person",
                PerceptionMode.FACE_RECOGNITION: "face_detect",
                PerceptionMode.GESTURE_MODE: "pose",
                PerceptionMode.EMOTION_MODE: "face_detect",  # Start with face detection
            }
            
            if mode in model_map:
                if not self._load_model(model_map[mode]):
                    return False
            
            self._mode = mode
            return True
    
    @property
    def mode(self) -> PerceptionMode:
        return self._mode
    
    def _update_fps(self):
        """Update FPS calculation."""
        now = time.time()
        self._frame_times.append(now)
        # Keep last 30 frames
        self._frame_times = [t for t in self._frame_times if now - t < 1.0]
    
    def _get_fps(self) -> float:
        """Get current FPS."""
        if len(self._frame_times) < 2:
            return 0.0
        return len(self._frame_times)
    
    def process_frame(self, frame: np.ndarray) -> PerceptionResult:
        """Process a frame according to current mode."""
        result = PerceptionResult(
            mode=self._mode,
            timestamp=time.time(),
            inference_time_ms=0,
            fps=self._get_fps()
        )
        
        if self._mode == PerceptionMode.IDLE:
            return result
        
        if not self._vdevice or not self._infer_model:
            result.error = "not_connected"
            return result
        
        start_time = time.time()
        
        try:
            if self._mode == PerceptionMode.PERSON_TRACKING:
                result = self._process_person_tracking(frame, result)
            elif self._mode == PerceptionMode.FACE_RECOGNITION:
                result = self._process_face_recognition(frame, result)
            elif self._mode == PerceptionMode.GESTURE_MODE:
                result = self._process_gesture(frame, result)
            elif self._mode == PerceptionMode.EMOTION_MODE:
                result = self._process_emotion(frame, result)
        except Exception as e:
            result.error = str(e)
            logger.error(f"[PERCEPTION] Inference error: {e}")
        
        result.inference_time_ms = (time.time() - start_time) * 1000
        self._update_fps()
        result.fps = self._get_fps()
        
        logger.debug(result.to_log())
        return result
    
    def _preprocess_yolo(self, frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """Preprocess frame for YOLO models."""
        import cv2
        # Resize
        resized = cv2.resize(frame, target_size)
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to 0-1
        normalized = rgb.astype(np.float32) / 255.0
        # Add batch dimension and transpose to NCHW
        batch = np.expand_dims(normalized.transpose(2, 0, 1), axis=0)
        return batch
    
    def _postprocess_yolo_persons(self, output: np.ndarray, frame_shape: Tuple[int, int],
                                   conf_threshold: float = 0.6) -> List[Dict]:
        """Postprocess YOLO output for person detection."""
        detections = []
        h, w = frame_shape[:2]
        
        # Handle Hailo NMS output format (list of 80 class arrays)
        if isinstance(output, list):
            # Class 0 is person in COCO
            person_dets = output[0] if len(output) > 0 else np.array([])
            
            for det in person_dets:
                if len(det) >= 5:
                    x1, y1, x2, y2, conf = det[:5]
                    if conf >= conf_threshold:
                        detections.append({
                            "class": "person",
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": float(conf)
                        })
        
        # Sort by confidence, keep top detections
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections[:10]
    
    def _process_person_tracking(self, frame: np.ndarray, result: PerceptionResult) -> PerceptionResult:
        """Run person tracking inference."""
        with self._lock:
            # Preprocess
            input_data = self._preprocess_yolo(frame)
            
            # Run inference
            with self._infer_model.configure() as configured:
                bindings = configured.create_bindings()
                bindings.input().set_buffer(input_data)
                configured.run([bindings], timeout=5000)
                output = bindings.output().get_buffer()
            
            # Postprocess
            result.detections = self._postprocess_yolo_persons(output, frame.shape, self.PERSON_CONFIDENCE)
            
            if result.detections:
                result.confidence = result.detections[0]["confidence"]
                if self._on_detection:
                    self._on_detection(result.detections)
        
        return result
    
    def _preprocess_scrfd(self, frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """Preprocess frame for SCRFD face detection."""
        import cv2
        resized = cv2.resize(frame, target_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) - 127.5) / 128.0
        batch = np.expand_dims(normalized.transpose(2, 0, 1), axis=0)
        return batch
    
    def _postprocess_scrfd(self, output: np.ndarray, frame_shape: Tuple[int, int],
                           conf_threshold: float = 0.5) -> List[Dict]:
        """Postprocess SCRFD output for face detection."""
        detections = []
        h, w = frame_shape[:2]
        
        # SCRFD output format varies, handle common cases
        if isinstance(output, (list, tuple)):
            for det in output[0] if len(output) > 0 else []:
                if len(det) >= 5:
                    x1, y1, x2, y2, conf = det[:5]
                    if conf >= conf_threshold:
                        # Denormalize if needed
                        if x2 <= 1.0:
                            x1, x2 = x1 * w, x2 * w
                            y1, y2 = y1 * h, y2 * h
                        detections.append({
                            "class": "face",
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": float(conf)
                        })
        
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections[:5]
    
    def _extract_face_embedding(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Extract face embedding using ArcFace."""
        import cv2
        
        # Crop face region with margin
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Add 20% margin
        margin_x = int((x2 - x1) * 0.2)
        margin_y = int((y2 - y1) * 0.2)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
        
        # Load embedding model if not loaded
        if self._current_model_name != "face_embed":
            if not self._load_model("face_embed"):
                return None
        
        # Preprocess for ArcFace (112x112)
        face_resized = cv2.resize(face_crop, (112, 112))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = (face_rgb.astype(np.float32) - 127.5) / 128.0
        face_batch = np.expand_dims(face_normalized.transpose(2, 0, 1), axis=0)
        
        # Run inference
        with self._infer_model.configure() as configured:
            bindings = configured.create_bindings()
            bindings.input().set_buffer(face_batch)
            configured.run([bindings], timeout=5000)
            embedding = bindings.output().get_buffer()
        
        # Normalize embedding
        embedding = embedding.flatten()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def _match_face(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Match embedding against database."""
        if not self._face_db:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        for name, fe in self._face_db.items():
            # Cosine similarity
            similarity = np.dot(embedding, fe.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        if best_similarity >= self.FACE_MATCH_THRESHOLD:
            return best_match, best_similarity
        return None, best_similarity
    
    def _process_face_recognition(self, frame: np.ndarray, result: PerceptionResult) -> PerceptionResult:
        """Run face recognition pipeline."""
        # First detect faces
        if self._current_model_name != "face_detect":
            self._load_model("face_detect")
        
        with self._lock:
            input_data = self._preprocess_scrfd(frame)
            
            with self._infer_model.configure() as configured:
                bindings = configured.create_bindings()
                bindings.input().set_buffer(input_data)
                configured.run([bindings], timeout=5000)
                output = bindings.output().get_buffer()
            
            result.detections = self._postprocess_scrfd(output, frame.shape, self.FACE_CONFIDENCE)
        
        if not result.detections:
            return result
        
        # Get embedding for largest face
        largest_face = max(result.detections, 
                         key=lambda d: (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]))
        
        embedding = self._extract_face_embedding(frame, largest_face["bbox"])
        if embedding is None:
            return result
        
        # Check if enrolling
        if self._enrolling:
            self._enroll_embeddings.append(embedding)
            if len(self._enroll_embeddings) >= self.ENROLL_FRAMES:
                self._complete_enrollment()
            result.recognition_result = f"enrolling:{self._enroll_name}:{len(self._enroll_embeddings)}/{self.ENROLL_FRAMES}"
        else:
            # Match against database
            match_name, similarity = self._match_face(embedding)
            if match_name:
                result.recognition_result = match_name
                result.confidence = similarity
                if self._on_recognition:
                    self._on_recognition(match_name, similarity)
            else:
                result.recognition_result = "unknown"
                result.confidence = similarity
        
        # Reload face detection model for next frame
        self._load_model("face_detect")
        return result
    
    def enroll_face(self, name: str) -> bool:
        """Start face enrollment for a person."""
        if self._enrolling:
            logger.warning(f"[PERCEPTION] Already enrolling {self._enroll_name}")
            return False
        
        self._enrolling = True
        self._enroll_name = name
        self._enroll_embeddings = []
        logger.info(f"[PERCEPTION] Starting enrollment for: {name}")
        return True
    
    def _complete_enrollment(self):
        """Complete face enrollment - average embeddings."""
        if not self._enroll_embeddings:
            return
        
        # Average all embeddings
        avg_embedding = np.mean(self._enroll_embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        # Save to database
        self._face_db[self._enroll_name] = FaceEmbedding(
            name=self._enroll_name,
            embedding=avg_embedding,
            created_at=time.time(),
            samples=len(self._enroll_embeddings)
        )
        
        self._save_face_db()
        logger.info(f"[PERCEPTION] Enrolled {self._enroll_name} with {len(self._enroll_embeddings)} samples")
        
        self._enrolling = False
        self._enroll_name = ""
        self._enroll_embeddings = []
    
    def cancel_enrollment(self):
        """Cancel ongoing enrollment."""
        if self._enrolling:
            logger.info(f"[PERCEPTION] Enrollment cancelled for {self._enroll_name}")
            self._enrolling = False
            self._enroll_name = ""
            self._enroll_embeddings = []
    
    def delete_face(self, name: str) -> bool:
        """Delete a face from the database."""
        if name in self._face_db:
            del self._face_db[name]
            self._save_face_db()
            logger.info(f"[PERCEPTION] Deleted face: {name}")
            return True
        return False
    
    def list_faces(self) -> List[str]:
        """List all enrolled faces."""
        return list(self._face_db.keys())
    
    def _process_gesture(self, frame: np.ndarray, result: PerceptionResult) -> PerceptionResult:
        """Run gesture/pose detection."""
        with self._lock:
            input_data = self._preprocess_yolo(frame)
            
            with self._infer_model.configure() as configured:
                bindings = configured.create_bindings()
                bindings.input().set_buffer(input_data)
                configured.run([bindings], timeout=5000)
                output = bindings.output().get_buffer()
            
            # Pose output includes keypoints
            # For now just detect if person is present
            result.detections = self._postprocess_yolo_persons(output, frame.shape, 0.5)
            
            if result.detections:
                # TODO: Parse keypoints for gesture recognition
                result.recognition_result = "pose_detected"
                result.confidence = result.detections[0]["confidence"]
        
        return result
    
    def _process_emotion(self, frame: np.ndarray, result: PerceptionResult) -> PerceptionResult:
        """Run emotion detection (requires stable face first)."""
        # First detect face
        result = self._process_face_recognition(frame, result)
        
        if not result.detections:
            result.recognition_result = "no_face"
            return result
        
        # TODO: Add emotion classification model
        # For now, mark as face detected
        result.recognition_result = f"face:{result.recognition_result or 'detected'}"
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get perception manager status."""
        return {
            "mode": self._mode.value,
            "connected": self._vdevice is not None,
            "active_model": self._current_model_name,
            "fps": self._get_fps(),
            "enrolled_faces": len(self._face_db),
            "enrolling": self._enrolling,
            "enroll_progress": f"{len(self._enroll_embeddings)}/{self.ENROLL_FRAMES}" if self._enrolling else None
        }
    
    def set_detection_callback(self, callback: Callable):
        """Set callback for detections."""
        self._on_detection = callback
    
    def set_recognition_callback(self, callback: Callable):
        """Set callback for face recognition."""
        self._on_recognition = callback
    
    def recognize_face(self, frame: np.ndarray) -> Tuple[Optional[str], float]:
        """
        One-shot face recognition.
        Returns (name, confidence) or (None, 0) if no match.
        """
        old_mode = self._mode
        self.set_mode(PerceptionMode.FACE_RECOGNITION)
        
        result = self.process_frame(frame)
        
        # Restore mode
        if old_mode != PerceptionMode.FACE_RECOGNITION:
            self.set_mode(old_mode)
        
        if result.recognition_result and result.recognition_result not in ["unknown", "no_face"]:
            if not result.recognition_result.startswith("enrolling"):
                return result.recognition_result, result.confidence
        return None, result.confidence


# Singleton instance
_perception_manager: Optional[PerceptionManager] = None

def get_perception_manager() -> PerceptionManager:
    """Get singleton perception manager."""
    global _perception_manager
    if _perception_manager is None:
        _perception_manager = PerceptionManager()
    return _perception_manager
