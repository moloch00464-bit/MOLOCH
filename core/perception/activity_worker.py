#!/usr/bin/env python3
"""
ActivityWorker — Aktivitaetserkennung via r3d_18 (Kinetics-400).

Erkennt was die Person gerade tut: geht, sitzt, steht, winkt, tippt, ...
Arbeitet mit einem Ringpuffer der letzten 16 Frames (112x112).

Input:  [112, 112, 48] uint8 — 16 RGB-Frames channel-gestapelt (16*3=48)
Output: [400] float32 — Kinetics-400 Klassen-Scores
FPS:    ~55 (Batch=1), laeuft alle 30 Frames (~1.5s bei 20fps)
Modell: /mnt/moloch-data/hailo/models/r3d_18.hef
"""

import os
import logging
import threading
import numpy as np
import cv2
from collections import deque
from typing import Optional, Dict

from core.perception.vision_workers import (
    BaseWorker, WorkItem, WorkerResult, create_configured_model,
    INFERENCE_TIMEOUT_MS
)

logger = logging.getLogger("ActivityWorker")

MODEL_DIR = "/mnt/moloch-data/hailo/models"
ACTIVITY_HEF = os.path.join(MODEL_DIR, "r3d_18.hef")

CLIP_FRAMES = 16        # r3d_18 braucht genau 16 Frames
FRAME_SIZE  = 112       # 112x112 Input
CHANNELS    = CLIP_FRAMES * 3  # = 48

# Kinetics-400 Labels (Index 0-399)
# Vollstaendige Liste — nur die Moloch-relevanten sind hier mit Deutung versehen.
# Quelle: Standard Kinetics-400 Klassen-Reihenfolge.
KINETICS_400 = [
    "abseiling", "air drumming", "answering questions", "applauding", "applying cream",
    "archery", "arm wrestling", "arranging flowers", "assembling computer", "auctioning",
    "baby waking up", "baking cookies", "balloon blowing", "bandaging", "barbequing",
    "bartending", "beatboxing", "bee keeping", "belly dancing", "bench pressing",
    "bending back", "bending metal", "biking through snow", "blasting sand", "blowing glass",
    "blowing leaves", "blowing nose", "blowing out candles", "bobsledding", "bookbinding",
    "bouncing on trampoline", "bowling", "braiding hair", "breading or breadcrumbing",
    "breakdancing", "brush painting", "brushing hair", "brushing teeth", "building cabinet",
    "building shed", "bungee jumping", "busking", "canoeing or kayaking", "capoeira",
    "carrying baby", "cartwheeling", "carving pumpkin", "catching fish", "catching or throwing baseball",
    "catching or throwing frisbee", "catching or throwing softball", "celebrating", "checkmate",
    "cheerleading", "chopping wood", "clapping", "clay pottery making", "clean and jerk",
    "cleaning floor", "cleaning gutters", "cleaning pool", "cleaning shoes", "cleaning toilet",
    "cleaning windows", "climbing a rope", "climbing ladder", "climbing tree", "contact juggling",
    "cooking chicken", "cooking egg", "cooking on campfire", "cooking sausages", "counting money",
    "country line dancing", "cracking knuckles", "crossing river", "crying", "curling hair",
    "cutting nails", "cutting pineapple", "cutting watermelon", "dancing ballet", "dancing charleston",
    "dancing gangnam style", "dancing macarena", "deadlifting", "decorating the christmas tree",
    "digging", "disc golfing", "diving cliff", "dodgeball", "doing aerobics", "doing laundry",
    "doing nails", "drawing", "dribbling basketball", "drinking", "drinking beer", "drinking shots",
    "driving car", "driving tractor", "drop kicking", "drumming fingers", "dunking basketball",
    "dying hair", "eating burger", "eating cake", "eating carrots", "eating chips", "eating doughnuts",
    "eating hotdog", "eating ice cream", "eating spaghetti", "eating watermelon", "egg hunting",
    "exercising arm", "exercising with an exercise ball", "extinguishing fire", "faceplanting",
    "feeding birds", "feeding fish", "feeding goats", "filling eyebrows", "finger snapping",
    "fixing hair", "flipping pancake", "fly tying", "folding clothes", "folding napkins",
    "folding paper", "front raises", "frying vegetables", "garbage collecting", "gargling",
    "getting a haircut", "getting a tattoo", "giving or receiving award", "golf chipping",
    "golf driving", "golf putting", "grinding meat", "grooming dog", "grooming horse",
    "gymnastics tumbling", "hammer throw", "headbanging", "headbutting", "high jump",
    "high kick", "hitting baseball", "hockey stop", "hopscotch", "hoverboarding",
    "hugging", "hula hooping", "hurdling", "hurling (sport)", "ice climbing",
    "ice fishing", "ice skating", "ironing", "javelin throw", "jetskiing",
    "jogging", "juggling balls", "juggling fire", "juggling soccer ball", "jumping into pool",
    "jumpstyle dancing", "kicking field goal", "kicking soccer ball", "kissing", "kitesurfing",
    "knitting", "krumping", "laughing", "laying bricks", "long jump",
    "lunge", "making a cake", "making a sandwich", "making bed", "making jewelry",
    "making pizza", "making snowman", "making sushi", "making tea", "marching",
    "massaging back", "massaging feet", "massaging legs", "massaging person's head", "milking cow",
    "mopping floor", "motorcycling", "moving furniture", "mowing lawn", "news anchoring",
    "opening bottle", "opening present", "paragliding", "parasailing", "parkour",
    "passing American football (in game)", "passing American football (not in game)", "peeling apples",
    "peeling potatoes", "petting animal (not cat)", "petting cat", "picking fruit",
    "planting trees", "plastering", "playing accordion", "playing badminton", "playing bagpipes",
    "playing basketball", "playing bass guitar", "playing cards", "playing cello",
    "playing chess", "playing clarinet", "playing controller", "playing cricket",
    "playing cymbals", "playing didgeridoo", "playing drums", "playing flute",
    "playing guitar", "playing harmonica", "playing harp", "playing ice hockey",
    "playing keyboard", "playing kickball", "playing monopoly", "playing organ",
    "playing paintball", "playing piano", "playing poker", "playing recorder",
    "playing rugby", "playing saxophone", "playing squash or racquetball", "playing tennis",
    "playing trombone", "playing trumpet", "playing ukulele", "playing violin",
    "playing volleyball", "playing xylophone", "pole vault", "presenting weather forecast",
    "pull ups", "pumping fist", "pumping gas", "push up", "pushing car",
    "pushing cart", "pushing wheelchair", "reading book", "reading newspaper",
    "recording music", "riding a bike", "riding camel", "riding elephant",
    "riding mechanical bull", "riding mountain bike", "riding mule", "riding or walking with horse",
    "riding scooter", "riding unicycle", "ripping paper", "robot dancing",
    "rock climbing", "rock scissors paper", "roller skating", "running on treadmill",
    "sailing", "salsa dancing", "sanding floor", "scrambling eggs", "scuba diving",
    "setting table", "shaking hands", "shaking head", "sharpening knives", "sharpening pencil",
    "shaving head", "shaving legs", "shearing sheep", "shining shoes", "shooting basketball",
    "shooting goal (soccer)", "shot put", "shoveling snow", "shredding paper",
    "shuffling cards", "side kick", "sign language interpreting", "singing", "situp",
    "skateboarding", "ski jumping", "skiing (not slalom or crosscountry)", "skiing crosscountry",
    "skiing slalom", "skipping rope", "skydiving", "slacklining", "slapping",
    "sled dog racing", "smoking", "smoking hookah", "snatch weight lifting", "sneezing",
    "sniffing", "snowboarding", "snowkiting", "snowmobiling", "somersaulting",
    "spinning poi", "spray painting", "springboard diving", "squat", "sticking tongue out",
    "stomping grapes", "stretching arm", "stretching leg", "strumming guitar",
    "surfing crowd", "surfing water", "sweeping floor", "swimming backstroke",
    "swimming breast stroke", "swimming butterfly stroke", "swing dancing",
    "swinging legs", "swinging on something", "sword fighting", "tai chi",
    "taking a shower", "tango dancing", "tap dancing", "tapping guitar",
    "tapping pen", "tasting beer", "tasting food", "testifying", "texting",
    "throwing axe", "throwing ball", "throwing discus", "tickling", "tobogganing",
    "tossing coin", "tossing salad", "training dog", "trapezing", "trimming or shaving beard",
    "trimming trees", "triple jump", "tying bow tie", "tying knot (not on a boat)",
    "tying tie", "unboxing", "unloading truck", "using computer", "using remote controller (not gaming)",
    "using segway", "vault", "waiting in line", "walking the dog", "washing dishes",
    "washing feet", "washing hair", "washing hands", "water skiing", "water sliding",
    "watering plants", "waxing back", "waxing chest", "waxing eyebrows", "waxing legs",
    "weaving basket", "welding", "whistling", "windsurfing", "wrapping present",
    "wrestling", "writing", "yawning", "yoga", "zumba",
]

# Fuer Moloch besonders interessante Aktivitaeten (Teilstring-Match)
RELEVANT_KEYWORDS = {
    "walking", "jogging", "running", "sitting", "standing",
    "waving", "clapping", "typing", "texting", "using computer",
    "reading", "drinking", "eating", "laughing", "crying",
    "sleeping", "stretching", "yoga", "hugging", "kissing",
}


class ActivityWorker(BaseWorker):
    """Erkennt Aktivitaeten aus einem rollenden 16-Frame-Clip.

    Ringpuffer haelt die letzten 16 Frames (112x112 RGB).
    Inference laeuft alle 30 Frames (~1.5s bei 20fps).
    """

    def __init__(self):
        super().__init__(name="ActivityWorker", max_queue=2)
        self._model = None
        self._out_names = []
        self._out_shapes = {}
        self._frame_buffer: deque = deque(maxlen=CLIP_FRAMES)
        self._buffer_lock = threading.Lock()
        self._min_conf = 0.4  # IPC-steuerbar: Mindest-Score fuer Activity

    def _load_models(self, vdevice):
        if not os.path.exists(ACTIVITY_HEF):
            raise FileNotFoundError(f"r3d_18 HEF fehlt: {ACTIVITY_HEF}")
        _, self._model, _, self._out_names, self._out_shapes = \
            create_configured_model(vdevice, ACTIVITY_HEF)
        logger.info("[ActivityWorker] r3d_18 geladen — Outputs: %s shapes: %s",
                    self._out_names, self._out_shapes)

    def push_frame(self, frame_rgb: np.ndarray):
        """Frame in Ringpuffer eintragen (von aussen aufrufbar)."""
        small = cv2.resize(frame_rgb, (FRAME_SIZE, FRAME_SIZE))
        with self._buffer_lock:
            self._frame_buffer.append(small)

    def _build_clip_input(self) -> Optional[np.ndarray]:
        """16 Frames zu [112, 112, 48] uint8 stapeln."""
        with self._buffer_lock:
            frames = list(self._frame_buffer)
        if len(frames) < CLIP_FRAMES:
            return None
        # Kanäle stapeln: Frame0_R, Frame0_G, Frame0_B, Frame1_R, ...
        clip = np.concatenate([f for f in frames[-CLIP_FRAMES:]], axis=2)
        return np.ascontiguousarray(clip, dtype=np.uint8)

    def _process(self, item: WorkItem) -> WorkerResult:
        # Frame in Puffer (falls item.frame vorhanden)
        if item.frame is not None and item.frame.size > 0:
            self.push_frame(item.frame)

        clip = self._build_clip_input()
        if clip is None:
            return WorkerResult(
                worker_name="ActivityWorker",
                frame_id=item.frame_id,
                timestamp=item.timestamp,
                success=False,
                data={"activity": None, "reason": f"Puffer noch nicht voll ({len(self._frame_buffer)}/{CLIP_FRAMES})"}
            )

        bindings = self._model.create_bindings()
        bindings.input().set_buffer(clip)

        bufs = {}
        for name in self._out_names:
            buf = np.empty(self._out_shapes[name], dtype=np.float32)
            bindings.output(name).set_buffer(buf)
            bufs[name] = buf

        self._model.run([bindings], INFERENCE_TIMEOUT_MS)

        scores = bufs[self._out_names[0]].flatten().copy()
        top5_idx = np.argsort(scores)[::-1][:5]

        top_label = KINETICS_400[top5_idx[0]] if top5_idx[0] < len(KINETICS_400) else f"klasse_{top5_idx[0]}"
        top_score = float(scores[top5_idx[0]])

        top5 = [
            {
                "label": KINETICS_400[i] if i < len(KINETICS_400) else f"klasse_{i}",
                "score": float(scores[i])
            }
            for i in top5_idx
        ]

        # Relevant-Flag: nur Moloch-relevante Aktivitaeten weiterleiten
        # r3d_18 gibt sonst "snowmobiling", "riding mule" etc. aus — das ist physisch falsch
        relevant = any(kw in top_label for kw in RELEVANT_KEYWORDS) and top_score >= self._min_conf

        if relevant:
            logger.info("[ActivityWorker] %s (%.2f)", top_label, top_score)

        return WorkerResult(
            worker_name="ActivityWorker",
            frame_id=item.frame_id,
            timestamp=item.timestamp,
            success=True,
            data={
                "activity": top_label if relevant else None,
                "score": top_score if relevant else 0.0,
                "relevant": relevant,
                "top5": top5,
            }
        )


_instance: Optional[ActivityWorker] = None


def get_activity_worker() -> ActivityWorker:
    """Singleton-Getter."""
    global _instance
    if _instance is None:
        _instance = ActivityWorker()
    return _instance
