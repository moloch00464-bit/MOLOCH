#!/usr/bin/env python3
"""
M.O.L.O.C.H. Capability Monitor
=================================
Scannt beim Systemstart welche Module, Modelle und Hardware
tatsaechlich verfuegbar sind. Generiert system_capabilities.json
neu und vergleicht mit der vorherigen Version.

Bei Aenderungen: bus.publish("capability.changed", {added:[], removed:[]})

Singleton: get_capability_monitor()
"""

import json
import logging
import os
import subprocess
import importlib
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

logger = logging.getLogger("moloch.capability_monitor")

MOLOCH_ROOT = Path.home() / "moloch"
CAPS_PATH = MOLOCH_ROOT / "config" / "system_capabilities.json"
HEF_DIRS = [
    Path("/mnt/moloch-data/hailo/models"),
    MOLOCH_ROOT / "models" / "hailo",
]
VOICE_DIR = MOLOCH_ROOT / "models" / "voices"

# Modul-Packages und ihre Python-Pfade
MODULE_PACKAGES = {
    "core": [
        "action_bridge", "moloch_event_bus", "moloch_service", "core_integrator",
        "perception_engine", "ptz_tracker", "ptz_arbiter", "arbitration",
        "camera_manager", "led_controller", "spotify_controller",
        "voice_pipeline", "tts", "moloch_sprache", "keyword_handler",
        "teachen", "longterm_memory", "ipc_router", "status",
        "inference_engine", "model_orchestrator", "calibration_engine",
        "dashboard", "timeline", "eye_viewer", "einpraegen",
        "environment_watcher", "cloud_controller", "capability_monitor",
    ],
    "awareness": ["room_map", "motion_analyzer", "activity_analyzer", "context_evaluator"],
    "personality": ["tension_integrator", "mood_engine", "behavior_rules", "personality_engine"],
    "autonomy": ["decision_engine", "atmosphere_controller", "homeostasis", "night_cycle"],
    "memory": ["episodic_memory", "persistent_memory", "person_reid", "vector_memory"],
    "music": ["spotify_bridge", "music_memory"],
    "perception": [
        "tappas_pipeline", "perception_buffer", "perception_frame",
        "perception_manager", "hailo_postprocess", "spatial_learning", "model_health",
    ],
    "hardware": ["camera", "hailo_manager", "thermal_manager", "camera_cloud_bridge", "ptz_calibration"],
    "speech": ["hailo_whisper", "audio_pipeline"],
}

# HEF-Modelle: Name → Task-Mapping
HEF_TASKS = {
    "yolov8m_h10": "person_detection",
    "scrfd_10g": "face_detection",
    "arcface_mobilefacenet": "face_recognition",
    "yolov8s_pose_h10": "pose_estimation",
    "yolov8m_pose_h10": "pose_estimation",
    "yolov11m_h10": "object_detection",
    "yolov5n_seg_h10": "segmentation",
    "face_attr_resnet_v1_18": "face_attributes",
    "hand_landmark_lite": "hand_tracking",
    "resnet_v1_50_h10": "classification",
}

# Aktive Modelle in der TAPPAS-Pipeline
TAPPAS_ACTIVE = {"yolov8m_h10", "scrfd_10g", "arcface_mobilefacenet", "yolov8s_pose_h10"}


EVOLUTION_LOG_PATH = MOLOCH_ROOT / "logs" / "moloch_evolution_log.json"


class CapabilityMonitor:
    """Scannt System-Faehigkeiten und erkennt Aenderungen."""

    def __init__(self):
        self._previous = self._load_previous()
        self._current = None
        self._subscribe_capability_events()

    def _subscribe_capability_events(self):
        """Abonniert capability.* Events und schreibt sie ins evolution_log."""
        try:
            from core.moloch_event_bus import get_event_bus
            get_event_bus().subscribe("capability.*", self._on_capability_event)
            logger.info("[CAP] Subscribt auf capability.* Events")
        except Exception as e:
            logger.debug(f"[CAP] Bus-Subscribe fehlgeschlagen (noch nicht bereit): {e}")

    def _on_capability_event(self, event):
        """Capability-Event ins evolution_log schreiben."""
        try:
            entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "event_type": getattr(event, "event_type", str(event)),
                "payload": getattr(event, "payload", {}),
            }
            log = []
            if EVOLUTION_LOG_PATH.exists():
                try:
                    log = json.loads(EVOLUTION_LOG_PATH.read_text())
                    if not isinstance(log, list):
                        log = []
                except (json.JSONDecodeError, OSError):
                    log = []
            log.append(entry)
            # Nur letzte 500 Eintraege behalten
            if len(log) > 500:
                log = log[-500:]
            EVOLUTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            EVOLUTION_LOG_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"[CAP] evolution_log schreiben fehlgeschlagen: {e}")

    def _load_previous(self) -> dict:
        """Laedt die vorherige system_capabilities.json."""
        if CAPS_PATH.exists():
            try:
                return json.loads(CAPS_PATH.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"[CAP] Vorherige capabilities unlesbar: {e}")
        return {}

    def scan(self) -> dict:
        """Scannt alle Subsysteme und baut die aktuelle Capabilities-Map."""
        logger.info("[CAP] Scanne System-Faehigkeiten...")
        t0 = time.time()

        caps = {
            "_meta": {
                "description": "M.O.L.O.C.H. System Capabilities — Was Moloch kann",
                "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": "1.1",
            },
            "vision_models": self._scan_hef_models(),
            "hardware": self._scan_hardware(),
            "abilities": self._scan_abilities(),
            "gates": self._scan_gates(),
            "modules": self._scan_modules(),
            "pipeline": self._scan_pipeline(),
        }

        self._current = caps
        dt = (time.time() - t0) * 1000
        logger.info(f"[CAP] Scan abgeschlossen in {dt:.0f}ms")
        return caps

    def _scan_hef_models(self) -> list:
        """Findet alle HEF-Dateien auf dem System."""
        found = {}  # name → {path, size_mb}
        for hef_dir in HEF_DIRS:
            if not hef_dir.exists():
                continue
            for hef in hef_dir.glob("*.hef"):
                name = hef.stem
                if name not in found:
                    found[name] = {
                        "name": name,
                        "task": HEF_TASKS.get(name, "unknown"),
                        "size_mb": round(hef.stat().st_size / (1024 * 1024), 1),
                        "active": name in TAPPAS_ACTIVE,
                    }

        return sorted(found.values(), key=lambda m: (not m["active"], m["name"]))

    def _scan_hardware(self) -> dict:
        """Prueft Hardware-Verfuegbarkeit."""
        hw = {
            "brain": {"device": "Raspberry Pi 5", "ram_gb": 4, "ip": "192.168.178.24"},
            "npu": self._check_npu(),
            "camera": self._check_camera(),
            "audio_input": self._check_audio_input(),
            "audio_output": "HDMI-1 via PipeWire",
            "storage": self._check_storage(),
        }
        return hw

    def _check_npu(self) -> dict:
        """Hailo NPU erreichbar?"""
        npu = {"device": "Hailo-10H", "tops": 40, "ram_gb": 8, "online": False}
        try:
            result = subprocess.run(
                ["hailortcli", "fw-control", "identify"],
                capture_output=True, text=True, timeout=5,
            )
            npu["online"] = result.returncode == 0
        except Exception:
            pass
        return npu

    def _check_camera(self) -> dict:
        """Kamera-Info (statisch, RTSP-Check waere zu langsam fuer Startup)."""
        return {
            "device": "Sonoff CAM-PT2",
            "ip": "192.168.178.25",
            "resolution": "1920x1080",
            "fps": 20,
            "protocols": ["RTSP", "ONVIF"],
            "ptz": True,
        }

    def _check_audio_input(self) -> list:
        """Vorhandene Audio-Eingaenge."""
        devices = []
        try:
            result = subprocess.run(
                ["arecord", "-l"], capture_output=True, text=True, timeout=3,
            )
            if "SmartMic" in result.stdout or "smartmic" in result.stdout.lower():
                devices.append("SmartMic BT")
            if "ReSpeaker" in result.stdout or "respeaker" in result.stdout.lower():
                devices.append("ReSpeaker Lite USB")
            # Fallback: wenn nichts erkannt aber Geraete da
            if not devices and "card" in result.stdout.lower():
                devices.append("audio_device_detected")
        except Exception:
            devices = ["SmartMic BT", "ReSpeaker Lite USB"]  # Annahme
        return devices

    def _check_storage(self) -> list:
        """Mount-Points pruefen."""
        mounts = []
        for mount, fs, role in [
            ("/home/molochzuhause/moloch", "ext4", "Code + Configs"),
            ("/mnt/moloch-data", "NTFS", "AI-Modelle + Qdrant DB"),
        ]:
            p = Path(mount)
            if p.exists() and p.is_dir():
                mounts.append({"mount": mount, "fs": fs, "role": role, "available": True})
            else:
                mounts.append({"mount": mount, "fs": fs, "role": role, "available": False})
        return mounts

    def _scan_abilities(self) -> dict:
        """Leitet Faehigkeiten aus vorhandenen Modulen + Modellen ab."""
        modules = self._scan_modules()
        hefs = {m["name"] for m in self._scan_hef_models()}
        voices = self._scan_voices()

        return {
            "sehen": {
                "person_erkennen": "yolov8m_h10" in hefs,
                "gesicht_erkennen": "scrfd_10g" in hefs,
                "besitzer_identifizieren": "arcface_mobilefacenet" in hefs,
                "pose_schaetzen": any(h for h in hefs if "pose" in h),
                "hand_tracking": "hand_landmark_lite" in hefs,
                "segmentierung": "yolov5n_seg_h10" in hefs,
            },
            "hoeren": {
                "sprache_zu_text": "hailo_whisper" in modules.get("speech", []),
                "engine": "hailo_whisper",
                "sprache": "de",
            },
            "sprechen": {
                "tts": "tts" in modules.get("core", []),
                "engine": "piper",
                "stimmen": voices,
            },
            "bewegen": {
                "ptz_steuerung": "camera" in modules.get("hardware", []),
                "tracking": "ptz_tracker" in modules.get("core", []),
                "suchverhalten": "ptz_arbiter" in modules.get("core", []),
                "park_position": True,
            },
            "musik": {
                "spotify_playback": "spotify_controller" in modules.get("core", []),
                "emotionale_musikwahl": "spotify_bridge" in modules.get("music", []),
            },
            "gedaechtnis": {
                "episodisch": "episodic_memory" in modules.get("memory", []),
                "langzeit": "longterm_memory" in modules.get("core", []),
                "personen_reid": "person_reid" in modules.get("memory", []),
                "musik_assoziationen": "music_memory" in modules.get("music", []),
                "vektor_db": "qdrant",
            },
            "selbststeuerung": {
                "entscheidungen": "decision_engine" in modules.get("autonomy", []),
                "atmosphaere": "atmosphere_controller" in modules.get("autonomy", []),
                "selbstueberwachung": "homeostasis" in modules.get("autonomy", []),
                "nachtverarbeitung": "night_cycle" in modules.get("autonomy", []),
            },
        }

    def _scan_voices(self) -> list:
        """Piper Voice-Modelle finden."""
        if not VOICE_DIR.exists():
            return []
        voices = []
        for f in sorted(VOICE_DIR.glob("*.onnx")):
            # de_DE-thorsten-low.onnx → thorsten-low
            name = f.stem
            if name.startswith("de_DE-"):
                name = name[6:]
            voices.append(name)
        return voices

    def _scan_gates(self) -> dict:
        """Gate-Status (statisch, wird manuell in CLAUDE.md gepflegt)."""
        # Gates aendern sich nicht automatisch — aus vorheriger Version uebernehmen
        if self._previous and "gates" in self._previous:
            return self._previous["gates"]
        return {
            "gate_0": {"name": "Vier Inseln verdrahtet", "status": "PASS", "date": "2026-03-01"},
            "gate_0_5": {"name": "TAPPAS Pipeline", "status": "PASS", "date": "2026-03-05"},
            "gate_1": {"name": "Action Bridge + Tracking", "status": "AKTIV", "date": "2026-03-06"},
            "gate_2": {"name": "Identity (ReID + Qdrant)", "status": "GEPLANT"},
            "gate_3": {"name": "Awareness (Timing/Behaviour)", "status": "GEPLANT"},
            "gate_4": {"name": "Personality (Tension/Mood)", "status": "GEPLANT"},
            "gate_5": {"name": "Night Cycle (Dreaming)", "status": "GEPLANT"},
        }

    def _scan_modules(self) -> dict:
        """Prueft welche Python-Module importierbar sind."""
        result = {}
        for package, modules in MODULE_PACKAGES.items():
            importable = []
            for mod_name in modules:
                if package == "core":
                    import_path = f"core.{mod_name}"
                else:
                    import_path = f"core.{package}.{mod_name}"
                try:
                    importlib.import_module(import_path)
                    importable.append(mod_name)
                except Exception:
                    pass
            result[package] = importable
        return result

    def _scan_pipeline(self) -> dict:
        """TAPPAS Pipeline-Konfiguration."""
        tappas_flag = os.environ.get("MOLOCH_USE_TAPPAS", "0")
        return {
            "type": "TAPPAS/GStreamer" if tappas_flag == "1" else "InferenceEngine (Legacy)",
            "feature_flag": f"MOLOCH_USE_TAPPAS={tappas_flag}",
            "fps": 20 if tappas_flag == "1" else 10,
            "active_models": sorted(TAPPAS_ACTIVE) if tappas_flag == "1" else [],
            "scheduling": "vdevice-group-id=SHARED",
        }

    def compare_and_publish(self) -> Tuple[List[str], List[str]]:
        """Vergleicht aktuelle mit vorheriger Version, publisht Aenderungen."""
        if not self._current:
            self.scan()

        old_keys = self._flatten_capabilities(self._previous)
        new_keys = self._flatten_capabilities(self._current)

        added = sorted(new_keys - old_keys)
        removed = sorted(old_keys - new_keys)

        if added or removed:
            logger.info(f"[CAP] Aenderungen erkannt: +{len(added)} -{len(removed)}")
            for a in added:
                logger.info(f"[CAP]   + {a}")
            for r in removed:
                logger.info(f"[CAP]   - {r}")

            try:
                from core.moloch_event_bus import get_event_bus
                get_event_bus().publish(
                    event_type="capability.changed",
                    payload={"added": added, "removed": removed},
                    source="capability_monitor",
                    priority=3,
                )
            except Exception as e:
                logger.warning(f"[CAP] Event Bus publish fehlgeschlagen: {e}")
        else:
            logger.info("[CAP] Keine Aenderungen seit letztem Start")

        return added, removed

    def _flatten_capabilities(self, caps: dict) -> Set[str]:
        """Erzeugt ein flaches Set von Capability-Keys fuer Vergleich."""
        keys = set()
        if not caps:
            return keys

        # Module
        for package, modules in caps.get("modules", {}).items():
            for mod in modules:
                keys.add(f"module:{package}.{mod}")

        # Vision-Modelle
        for model in caps.get("vision_models", []):
            name = model.get("name", "") if isinstance(model, dict) else str(model)
            keys.add(f"hef:{name}")

        # Abilities (nur boolean True-Werte)
        for category, abilities in caps.get("abilities", {}).items():
            if isinstance(abilities, dict):
                for ability, value in abilities.items():
                    if value is True:
                        keys.add(f"ability:{category}.{ability}")

        # Hardware
        hw = caps.get("hardware", {})
        if isinstance(hw.get("npu"), dict) and hw["npu"].get("online"):
            keys.add("hw:npu_online")
        for storage in hw.get("storage", []):
            if isinstance(storage, dict) and storage.get("available", True):
                keys.add(f"hw:storage:{storage.get('mount', '?')}")

        return keys

    def save(self):
        """Speichert die aktuellen Capabilities als JSON."""
        if not self._current:
            self.scan()
        try:
            CAPS_PATH.parent.mkdir(parents=True, exist_ok=True)
            CAPS_PATH.write_text(
                json.dumps(self._current, indent=2, ensure_ascii=False) + "\n"
            )
            logger.info(f"[CAP] Gespeichert: {CAPS_PATH}")
        except OSError as e:
            logger.error(f"[CAP] Speichern fehlgeschlagen: {e}")

    def run(self) -> Tuple[List[str], List[str]]:
        """Kompletter Durchlauf: Scan → Vergleich → Speichern → Publish."""
        self.scan()
        added, removed = self.compare_and_publish()
        self.save()
        return added, removed

    def get_status(self) -> dict:
        """Status fuer Health-Checks."""
        if not self._current:
            return {"scanned": False}
        modules = self._current.get("modules", {})
        total = sum(len(v) for v in modules.values())
        hefs = len(self._current.get("vision_models", []))
        return {"scanned": True, "modules": total, "hef_models": hefs}


# --- Singleton ---
_instance = None


def get_capability_monitor() -> CapabilityMonitor:
    global _instance
    if _instance is None:
        _instance = CapabilityMonitor()
    return _instance
