"""Anwesenheits-Ausloeser fuer proaktive Saetze: Markus kommt, Markus geht, Unbekannter.

Reine Logik ohne I/O ausser dem Lesen der Konfiguration. Der Aufrufer (voice_pipeline)
fuettert alle paar Sekunden, was die Kamera sieht, und bekommt hoechstens ein Ereignis
zurueck. Die Gesichtserkennung flackert (Similarity 0.4-0.5, zeitweise "unbekannt"),
deshalb wird ueberall entprellt statt auf Einzelbilder zu reagieren.

Regeln (config/proactive.json, wird bei Aenderung neu gelesen):
    enabled           Hauptschalter
    quiet_from/to     Nachtruhe (Stunde), in der nichts ausgeloest wird
    max_per_10min     Obergrenze ueber alle Ereignisse
    events.<name>     cooldown_s und die Entprell-Zeiten des Ereignisses
"""

import json
import os
import time
from collections import deque
from datetime import datetime
from typing import Callable, Dict, Optional

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "proactive.json",
)

DEFAULTS: Dict = {
    "enabled": True,
    "owner": "markus",
    "quiet_from": 23,
    "quiet_to": 7,
    "max_per_10min": 4,
    "events": {
        # owner_confirm_s: so lange muss Markus durchgehend erkannt sein
        # min_absence_s:   so lange muss er vorher weg gewesen sein
        "markus_arrived": {"enabled": True, "cooldown_s": 300, "owner_confirm_s": 4, "min_absence_s": 120},
        # leave_after_s:   so lange nicht erkannt = gegangen; min_presence_s: vorher mindestens da
        "markus_left": {"enabled": True, "cooldown_s": 300, "leave_after_s": 45, "min_presence_s": 60},
        # sustain_s: Unbekannter durchgehend im Bild; owner_absent_s: Markus so lange nicht erkannt
        "unknown_person": {"enabled": True, "cooldown_s": 180, "sustain_s": 8, "owner_absent_s": 20},
    },
}

REASONS = {
    "markus_arrived": ("Markus ist gerade wieder vor die Kamera gekommen, nach etwa {min} Minuten "
                       "Abwesenheit. Begruesse ihn kurz."),
    "markus_left": "Markus ist gerade aus dem Bild gegangen, nach etwa {min} Minuten. Ein kurzer Satz hinterher.",
    "unknown_person": ("Eine unbekannte Person ist im Bild, Markus ist nicht zu sehen. "
                       "Sprich sie knapp und wachsam an."),
}

# Ersetzt im Spontan-Prompt den Satz "Du siehst Markus gerade ueber deine Kamera."
SITUATION = {
    "markus_arrived": "Markus ist in diesem Moment wieder in dein Kamerabild gekommen.",
    "markus_left": "Markus ist in diesem Moment aus deinem Kamerabild gegangen. Du siehst ihn nicht mehr.",
    "unknown_person": "Du siehst gerade eine unbekannte Person. Markus ist NICHT im Bild.",
}


def _merge(base: Dict, over: Dict) -> Dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path: str = CONFIG_PATH) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _merge(DEFAULTS, json.load(f))
    except (OSError, ValueError):
        return _merge(DEFAULTS, {})


def _is_quiet(hour: int, start: int, end: int) -> bool:
    if start == end:
        return False
    return (start <= hour < end) if start < end else (hour >= start or hour < end)


class PresenceTriggers:
    """Entprellte Anwesenheits-Ereignisse mit Sperrzeiten und Obergrenze."""

    def __init__(self, config_path: str = CONFIG_PATH,
                 clock: Callable[[], float] = time.time,
                 hour: Callable[[], int] = lambda: datetime.now().hour):
        self._path = config_path
        self._clock = clock
        self._hour = hour
        self._cfg = load_config(config_path)
        self._cfg_mtime = self._mtime()
        self._started = clock()
        self._owner_since: Optional[float] = None    # seit wann durchgehend erkannt
        self._owner_last: Optional[float] = None     # zuletzt erkannt
        self._owner_present = False                  # entprellter Zustand
        self._present_since: Optional[float] = None
        self._absent_since: Optional[float] = self._started
        self._unknown_since: Optional[float] = None
        self._last_fired: Dict[str, float] = {}
        self._fired = deque(maxlen=32)               # Zeitpunkte aller Ereignisse

    # --- Konfiguration ---

    def _mtime(self):
        """Aenderungsmarke der Konfiguration (Zeitstempel + Groesse; None wenn sie fehlt)."""
        try:
            st = os.stat(self._path)
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def config(self) -> Dict:
        m = self._mtime()
        if m != self._cfg_mtime:
            self._cfg, self._cfg_mtime = load_config(self._path), m
        return self._cfg

    # --- Kern ---

    def update(self, face_id: Optional[str], face_detected: bool,
               person_detected: bool) -> Optional[Dict]:
        """Einen Beobachtungsschritt verarbeiten. Gibt ein Ereignis zurueck oder None.

        Ereignis: {"event": name, "reason": Text fuers LLM, "situation": Text fuers LLM}
        """
        now = self._clock()
        cfg = self.config()
        ev = cfg["events"]
        fid = (face_id or "").strip().lower()
        owner_seen = fid == str(cfg.get("owner", "markus")).lower()
        unknown_seen = bool(face_detected) and not owner_seen
        candidate: Optional[str] = None
        minutes = 0

        # Markus: durchgehende Erkennung -> anwesend; lange nicht erkannt -> gegangen
        if owner_seen:
            self._owner_last = now
            if self._owner_since is None:
                self._owner_since = now
            if (not self._owner_present
                    and now - self._owner_since >= ev["markus_arrived"]["owner_confirm_s"]):
                absent_for = now - (self._absent_since if self._absent_since is not None else now)
                self._owner_present = True
                self._present_since = now
                self._absent_since = None
                if absent_for >= ev["markus_arrived"]["min_absence_s"]:
                    candidate, minutes = "markus_arrived", int(absent_for // 60)
        else:
            self._owner_since = None
            if (self._owner_present and self._owner_last is not None
                    and now - self._owner_last >= ev["markus_left"]["leave_after_s"]):
                present_for = self._owner_last - (self._present_since or self._owner_last)
                self._owner_present = False
                self._absent_since = self._owner_last
                self._present_since = None
                if present_for >= ev["markus_left"]["min_presence_s"]:
                    candidate, minutes = "markus_left", int(present_for // 60)

        # Unbekannter: nur wenn Markus eine Weile nicht erkannt wurde (Erkennung flackert)
        owner_gone = (self._owner_last is None
                      or now - self._owner_last >= ev["unknown_person"]["owner_absent_s"])
        if unknown_seen and owner_gone and not self._owner_present:
            if self._unknown_since is None:
                self._unknown_since = now
            if candidate is None and now - self._unknown_since >= ev["unknown_person"]["sustain_s"]:
                candidate = "unknown_person"
        else:
            self._unknown_since = None

        if candidate is None or not self._allowed(candidate, now, cfg):
            return None
        self._last_fired[candidate] = now
        self._fired.append(now)
        if candidate == "unknown_person":
            self._unknown_since = None   # Sperrzeit regelt die naechste Meldung
        return {
            "event": candidate,
            "reason": REASONS[candidate].format(min=max(1, minutes)),
            "situation": SITUATION[candidate],
        }

    def _allowed(self, name: str, now: float, cfg: Dict) -> bool:
        if not cfg.get("enabled", True) or not cfg["events"][name].get("enabled", True):
            return False
        if _is_quiet(self._hour(), int(cfg["quiet_from"]), int(cfg["quiet_to"])):
            return False
        last = self._last_fired.get(name)
        if last is not None and now - last < cfg["events"][name]["cooldown_s"]:
            return False
        recent = sum(1 for t in self._fired if now - t < 600)
        return recent < int(cfg["max_per_10min"])

    def state(self) -> Dict:
        return {"owner_present": self._owner_present,
                "unknown_watch": self._unknown_since is not None,
                "fired_10min": sum(1 for t in self._fired if self._clock() - t < 600)}
