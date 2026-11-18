"""Begruessung bekannter Menschen (nicht Markus).

Erkennt die Kamera eine Person aus config/begruessungen.json einige Sekunden stabil,
sagt Moloch den Begruessungssatz (Pi-Piper), schreibt ihn in den Chatverlauf und meldet
ein Ereignis in GET /live ("begruessung"). pc/body_bridge.py spielt dazu den Reachy-Clip.
Danach Ruhe fuer pause_h Stunden pro Person.

Laeuft als Thread im HTTP-Chat-Server (nur eine Instanz, sonst doppelte Begruessung).
Wunsch Markus 2026-11-18: Rebecca (beste Freundin, Klingonisch) soll eine Begruessung mit
Freude bekommen, wenn Moloch sie zum ersten Mal sieht.
"""

import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(_ROOT, "config", "begruessungen.json")
STATUS_PATH = "/dev/shm/moloch_status.json"
STATE_PATH = "/dev/shm/moloch_begruessung.json"   # ueberlebt Dienst-Neustarts, nicht Reboots
POLL_S = 1.0
EREIGNIS_GUELTIG_S = 30.0


def lade_config(path: str = CONFIG_PATH) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def person_im_bild(st: Optional[dict], alter_s: float) -> Optional[str]:
    """Kleingeschriebener Name der erkannten Person, None bei Kamera aus/niemand/unbekannt."""
    if not st:
        return None
    try:
        from core.awareness.sinne import kamera_zustand
        if kamera_zustand(st, alter_s)["ok"] is not True:
            return None
    except Exception:
        return None
    if not st.get("person_detected"):
        return None
    m = re.match(r"[A-Za-zÄÖÜäöüß]+", str(st.get("face_id") or "").strip())
    return m.group(0).lower() if m else None


class Begruesser:
    """Zustand der Begruessung; schritt() einmal pro Sekunde mit dem aktuellen Status aufrufen."""

    def __init__(self, sprechen: Callable[[str], Any], merken: Callable[[str], Any],
                 jetzt: Callable[[], float] = time.time, state_path: Optional[str] = None):
        self._sprechen = sprechen
        self._merken = merken
        self._jetzt = jetzt
        self._state_path = state_path
        self._gesehen: Optional[tuple] = None   # (name, seit)
        self._zuletzt: Dict[str, float] = {}
        self.ereignis: Optional[dict] = None
        if state_path:
            try:
                with open(state_path, "r") as f:
                    self._zuletzt = {k: float(v) for k, v in (json.load(f) or {}).items()}
            except (OSError, ValueError, TypeError, AttributeError):
                pass

    def schritt(self, st: Optional[dict], alter_s: float, cfg: dict) -> Optional[dict]:
        personen = cfg.get("personen") or {}
        name = person_im_bild(st, alter_s)
        now = self._jetzt()
        if not cfg.get("aktiv", True) or name not in personen:
            self._gesehen = None
            return None
        if not self._gesehen or self._gesehen[0] != name:
            self._gesehen = (name, now)
        if now - self._gesehen[1] < float(cfg.get("stabil_s", 3)):
            return None   # Gesichtserkennung flackert - erst nach ein paar Sekunden glauben
        zuletzt = self._zuletzt.get(name)
        if zuletzt is not None and now - zuletzt < float(cfg.get("pause_h", 6)) * 3600:
            return None
        self._zuletzt[name] = now
        self._speichern()
        p = personen[name]
        anzeige = p.get("anzeige") or f"Hallo {name.capitalize()}!"
        self.ereignis = {"kennung": f"{name}-{int(now)}", "person": name, "ts": now,
                         "clip": p.get("clip") or "welcoming1", "text": anzeige}
        logger.info(f"[BEGRUESSUNG] {name} erkannt -> {anzeige!r}, Clip {self.ereignis['clip']}")
        for f, text in ((self._merken, anzeige), (self._sprechen, p.get("sprechen") or anzeige)):
            try:
                f(text)
            except Exception as e:
                logger.warning(f"[BEGRUESSUNG] {getattr(f, '__name__', f)} fehlgeschlagen: {e}")
        return self.ereignis

    def fuer_live(self) -> Optional[dict]:
        e = self.ereignis
        if e and self._jetzt() - e["ts"] < EREIGNIS_GUELTIG_S:
            return dict(e)
        return None

    def _speichern(self) -> None:
        if not self._state_path:
            return
        try:
            tmp = self._state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._zuletzt, f)
            os.replace(tmp, self._state_path)
        except OSError:
            pass


def _sprechen(text: str) -> None:
    """Pi-Piper im Hintergrund (speak() blockiert, solange gesprochen wird)."""
    def _run():
        from core.personality.personality_engine import get_personality_engine
        get_personality_engine().speak(text)
    threading.Thread(target=_run, daemon=True, name="begruessung-tts").start()


def _merken(text: str) -> None:
    from core.longterm_memory import get_memory
    get_memory().save_message("moloch", text, source="begruessung")


_instanz: Optional[Begruesser] = None


def starte() -> None:
    """Begruessungs-Thread starten (einmal pro Prozess)."""
    global _instanz
    if _instanz is not None:
        return
    _instanz = Begruesser(_sprechen, _merken, state_path=STATE_PATH)

    def _loop():
        while True:
            try:
                cfg = lade_config()
                if cfg.get("aktiv", True) and cfg.get("personen"):
                    with open(STATUS_PATH, "r") as f:
                        st = json.load(f)
                    _instanz.schritt(st, time.time() - os.path.getmtime(STATUS_PATH), cfg)
            except (OSError, ValueError):
                pass
            except Exception as e:
                logger.warning(f"[BEGRUESSUNG] Fehler: {e}")
            time.sleep(POLL_S)

    threading.Thread(target=_loop, daemon=True, name="begruessung").start()
    logger.info("[BEGRUESSUNG] Waechter gestartet")


def fuer_live() -> Optional[dict]:
    return _instanz.fuer_live() if _instanz else None


def klingonisch_zeile(cfg: Optional[dict] = None) -> str:
    """Feste Klingonisch-Saetze fuer den Prompt - das 7B-Modell erfindet sonst Woerter."""
    saetze = (cfg if cfg is not None else lade_config()).get("klingonisch") or {}
    if not saetze:
        return ""
    return ("[KLINGONISCH] Nur diese Saetze verwenden, nichts dazuerfinden: "
            + "; ".join(f"{k} = {v.get('bedeutung', '')}" for k, v in saetze.items()))
