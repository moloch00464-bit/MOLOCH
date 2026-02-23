#!/usr/bin/env python3
"""
M.O.L.O.C.H. Perception Buffer
================================
Ring-Buffer der letzten N Perception Frames.

Ermoeglicht:
- Trend-Erkennung (Emotion wechselt, Person naehert sich)
- Glaettung (einzelne Ausreisser werden ignoriert)
- Zeitliche Muster (wie lange ist jemand schon da?)
"""

import time
import threading
import logging
from collections import deque
from typing import Optional, Dict, List

from core.perception.perception_frame import PerceptionFrame

_logger = logging.getLogger("PerceptionBuffer")

# Default: 30 Frames (~1s bei 30 FPS, ~2s bei 15 FPS)
_DEFAULT_SIZE = 30


class PerceptionBuffer:
    """Ring-Buffer fuer Perception Frames mit Trend-Analyse."""

    def __init__(self, size: int = _DEFAULT_SIZE):
        self._buffer: deque = deque(maxlen=size)
        self._lock = threading.Lock()
        self._size = size

    def push(self, frame: PerceptionFrame):
        """Neuen Frame in den Buffer schieben."""
        with self._lock:
            self._buffer.append(frame)

    @property
    def latest(self) -> Optional[PerceptionFrame]:
        """Neuester Frame (oder None)."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def get_recent(self, n: int = 5) -> List[PerceptionFrame]:
        """Die letzten N Frames."""
        with self._lock:
            frames = list(self._buffer)
        return frames[-n:]

    # =========================================================================
    # Glaettung: Mehrheitsentscheidung ueber letzte N Frames
    # =========================================================================

    def smoothed_person_detected(self, window: int = 5) -> bool:
        """Person erkannt in Mehrheit der letzten N Frames?"""
        frames = self.get_recent(window)
        if not frames:
            return False
        count = sum(1 for f in frames if f.person_detected)
        return count > len(frames) // 2

    def smoothed_face_id(self, window: int = 5) -> Optional[str]:
        """Haeufigste Face-ID in den letzten N Frames (ignoriert None)."""
        frames = self.get_recent(window)
        ids = [f.face_id for f in frames if f.face_id is not None]
        if not ids:
            return None
        # Haeufigste ID
        from collections import Counter
        most_common = Counter(ids).most_common(1)
        return most_common[0][0] if most_common else None

    def smoothed_emotion(self, window: int = 10) -> Optional[str]:
        """Dominante Emotion ueber N Frames (filtert Einzelausreisser)."""
        frames = self.get_recent(window)
        emotions = [f.emotion for f in frames if f.emotion is not None]
        if not emotions:
            return None
        from collections import Counter
        most_common = Counter(emotions).most_common(1)
        # Nur zurueckgeben wenn mindestens 40% der Frames diese Emotion haben
        name, count = most_common[0]
        if count >= len(frames) * 0.4:
            return name
        return None

    def smoothed_distance(self, window: int = 5) -> str:
        """Dominante Distanz ueber N Frames."""
        frames = self.get_recent(window)
        distances = [f.distance for f in frames if f.distance != "none"]
        if not distances:
            return "none"
        from collections import Counter
        return Counter(distances).most_common(1)[0][0]

    # =========================================================================
    # Trend-Erkennung
    # =========================================================================

    def trend_approaching(self, window: int = 10) -> bool:
        """Person naehert sich (distance_ratio steigt stetig)."""
        frames = self.get_recent(window)
        if len(frames) < 3:
            return False
        ratios = [f.distance_ratio for f in frames if f.distance_ratio > 0]
        if len(ratios) < 3:
            return False
        # Steigend wenn mindestens 60% der Deltas positiv
        deltas = [ratios[i+1] - ratios[i] for i in range(len(ratios)-1)]
        positive = sum(1 for d in deltas if d > 0.001)
        return positive > len(deltas) * 0.6

    def trend_leaving(self, window: int = 10) -> bool:
        """Person entfernt sich (distance_ratio sinkt stetig)."""
        frames = self.get_recent(window)
        if len(frames) < 3:
            return False
        ratios = [f.distance_ratio for f in frames if f.distance_ratio > 0]
        if len(ratios) < 3:
            return False
        deltas = [ratios[i+1] - ratios[i] for i in range(len(ratios)-1)]
        negative = sum(1 for d in deltas if d < -0.001)
        return negative > len(deltas) * 0.6

    def trend_emotion_change(self, window: int = 15) -> Optional[str]:
        """Erkennt Emotionswechsel: erste Haelfte vs. zweite Haelfte.

        Returns:
            "neutral->happy", "happy->angry" etc. oder None
        """
        frames = self.get_recent(window)
        if len(frames) < 6:
            return None

        mid = len(frames) // 2
        first_half = [f.emotion for f in frames[:mid] if f.emotion]
        second_half = [f.emotion for f in frames[mid:] if f.emotion]

        if not first_half or not second_half:
            return None

        from collections import Counter
        emo_before = Counter(first_half).most_common(1)[0][0]
        emo_after = Counter(second_half).most_common(1)[0][0]

        if emo_before != emo_after:
            return f"{emo_before}->{emo_after}"
        return None

    def presence_duration(self) -> float:
        """Wie lange ist kontinuierlich jemand erkannt (Sekunden)?

        Zaehlt rueckwaerts ab dem neuesten Frame bis zum ersten Frame
        ohne Person-Detection.
        """
        with self._lock:
            frames = list(self._buffer)
        if not frames:
            return 0.0

        # Rueckwaerts iterieren
        start_ts = frames[-1].timestamp
        for f in reversed(frames):
            if not f.anyone_present:
                break
            start_ts = f.timestamp

        return frames[-1].timestamp - start_ts if frames else 0.0

    def absence_duration(self) -> float:
        """Wie lange ist kontinuierlich NIEMAND erkannt (Sekunden)?"""
        with self._lock:
            frames = list(self._buffer)
        if not frames:
            return 0.0

        if frames[-1].anyone_present:
            return 0.0

        start_ts = frames[-1].timestamp
        for f in reversed(frames):
            if f.anyone_present:
                break
            start_ts = f.timestamp

        return frames[-1].timestamp - start_ts

    def avg_pose_energy(self, window: int = 10) -> float:
        """Durchschnittliche Pose-Energie (Bewegungsintensitaet)."""
        frames = self.get_recent(window)
        energies = [f.pose_energy for f in frames if f.pose_energy > 0]
        if not energies:
            return 0.0
        return sum(energies) / len(energies)

    def get_trends(self) -> Dict:
        """Alle Trend-Daten aggregiert (fuer Core Integrator)."""
        return {
            "approaching": self.trend_approaching(),
            "leaving": self.trend_leaving(),
            "emotion_change": self.trend_emotion_change(),
            "presence_duration": round(self.presence_duration(), 1),
            "absence_duration": round(self.absence_duration(), 1),
            "avg_pose_energy": round(self.avg_pose_energy(), 3),
            "smoothed_person": self.smoothed_person_detected(),
            "smoothed_face_id": self.smoothed_face_id(),
            "smoothed_emotion": self.smoothed_emotion(),
            "smoothed_distance": self.smoothed_distance(),
        }


# Singleton
_instance: Optional[PerceptionBuffer] = None

def get_perception_buffer() -> PerceptionBuffer:
    """Singleton-Zugriff auf den PerceptionBuffer."""
    global _instance
    if _instance is None:
        _instance = PerceptionBuffer()
    return _instance
