#!/usr/bin/env python3
"""
Somatischer Zustand - technischer Koerperzustand aus echter Telemetrie (Issue #5).

Erste Nervenschleife: Temperatur -> Zustand -> Koerper -> Rueckmeldung.
Messung, Deutung und Ausdruck bleiben getrennt (NERVENSYSTEM_PLAN.md, Grundsatz 4):

    Messung   watchdog.cpu_temp im Status-JSON (Celsius, System-Watchdog alle 3 s)
    Deutung   SomaTracker.update() -> ok | strained | distressed | recovering | unknown
    Ausdruck  core/body/expression.py (Kanal "soma"), nicht hier

Regeln:
- Kein frischer Messwert -> "unknown", nie der letzte Stand (Grundsatz 1 und 2).
- Hysterese gegen Flattern an der Schwelle.
- Jeder Zustandswechsel bekommt eine neue Kennung. Ueber sie ordnet der Pi die
  Rueckmeldung des Koerpers dem Ausloeser zu (ACCEPTED / EXECUTED, Grundsatz 3).
- Kuehlung: laeuft der Pi-Kuehler (cooling_device0 > 0), ist das ein Verdacht auf den
  Noctua (soma["kuehlung"]). Nur Verdacht: dessen Drehzahl misst niemand.
- Testvorgabe: /dev/shm/moloch_soma_test.json {"cpu_temp_c": 78, "until": <epoch>,
  optional "pi_kuehler_stufe": 2}
  ersetzt den Messwert bis "until" und wird als quelle="simulation" gefuehrt.
  Sie wirkt NUR auf diesen Zustand - Watchdog und Thermal-Manager lesen weiter die
  echte Temperatur, Schutzmassnahmen loest eine Simulation also nicht aus.

Reine Logik ohne Thread. Laeuft im Prozess, der /live ausliefert (Chat-Server).
"""

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, Optional

ZUSTAENDE = ("ok", "strained", "distressed", "recovering", "unknown")
TEST_OVERRIDE_PATH = "/dev/shm/moloch_soma_test.json"
# Pi-5-Active-Cooler (0 = aus). Der Noctua (GPIO18, scripts/fan_control.py) haelt die CPU
# normal unter 50 °C, dort springt der Pi-Kuehler erst an. Laeuft er, kuehlt der Noctua
# vermutlich nicht. Die Noctua-Drehzahl selbst wird NICHT gemessen - darum nur "Verdacht".
PI_KUEHLER_PATH = "/sys/class/thermal/cooling_device0/cur_state"


@dataclass
class SomaConfig:
    warn_c: float = 70.0        # wie CPU_TEMP_WARN im System-Watchdog
    kritisch_c: float = 80.0    # wie CPU_TEMP_CRITICAL
    hysterese_c: float = 5.0
    max_alter_s: float = 15.0   # aelter = unbekannt
    erholung_s: float = 30.0    # so lange "recovering" nach dem Abklingen
    bestaetigung_s: float = 20.0  # ohne Rueckmeldung danach: unbestaetigt


class SomaTracker:

    def __init__(self, cfg: Optional[SomaConfig] = None, clock: Callable[[], float] = time.time):
        self.cfg = cfg or SomaConfig()
        self._clock = clock
        self._level = "ok"           # ok | strained | distressed (ohne Erholung/Unbekannt)
        self._zustand = "unknown"
        self._seit = clock()
        self._kennung = uuid.uuid4().hex[:8]
        self._erholung_bis = 0.0
        self._letzter: Dict = {}

    # === Messwert besorgen ===

    @staticmethod
    def read_override(path: str = TEST_OVERRIDE_PATH, now: Optional[float] = None) -> Optional[Dict]:
        """Gueltige Testvorgabe oder None."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            until = float(data.get("until", 0))
            temp = float(data["cpu_temp_c"])
            stufe = data.get("pi_kuehler_stufe")
            stufe = int(stufe) if stufe is not None else None
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if until <= (now if now is not None else time.time()):
            return None
        return {"cpu_temp_c": temp, "until": until, "pi_kuehler_stufe": stufe}

    @staticmethod
    def kuehlung(stufe: Optional[int]) -> Dict:
        """Befund zur Kuehlung aus der Stufe des Pi-Kuehlers (None = nicht lesbar)."""
        if stufe is None:
            return {"pi_kuehler_stufe": None, "verdacht_noctua": None,
                    "befund": "Pi-Kuehler-Stufe nicht lesbar"}
        if stufe <= 0:
            return {"pi_kuehler_stufe": 0, "verdacht_noctua": False, "befund": None}
        return {"pi_kuehler_stufe": stufe, "verdacht_noctua": True,
                "befund": f"Pi-Kuehler laeuft (Stufe {stufe}) - normal haelt der Noctua ihn aus. "
                          "Verdacht: Noctua kuehlt nicht (ausgefallen, blockiert, Kabel). "
                          "Noctua-Drehzahl wird nicht gemessen."}

    @staticmethod
    def read_pi_kuehler(path: str = PI_KUEHLER_PATH) -> Optional[int]:
        try:
            with open(path, "r") as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return None

    def update_from_status(self, status: Optional[Dict], status_mtime: Optional[float],
                           override_path: str = TEST_OVERRIDE_PATH,
                           mono: Callable[[], float] = time.monotonic,
                           kuehler_path: str = PI_KUEHLER_PATH) -> Dict:
        """Status-JSON auswerten. Testvorgabe hat Vorrang und ist als Simulation markiert."""
        now = self._clock()
        ov = self.read_override(override_path, now)
        if ov:
            out = self.update(ov["cpu_temp_c"], now, "simulation")
            out["kuehlung"] = self.kuehlung(ov["pi_kuehler_stufe"] if ov["pi_kuehler_stufe"] is not None
                                            else self.read_pi_kuehler(kuehler_path))
            return out
        temp, gemessen = None, status_mtime
        if isinstance(status, dict):
            wd = status.get("watchdog") or {}
            temp = wd.get("cpu_temp")
            if not isinstance(temp, (int, float)) or temp <= 0 or wd.get("running") is False:
                temp = None      # Watchdog startet mit 0.0 bzw. laeuft nicht - kein Messwert
            # last_check ist time.monotonic() im Service-Prozess. Unter Linux ist das
            # CLOCK_MONOTONIC und in jedem Prozess gleich: so zaehlt das Alter der
            # Messung selbst, nicht das der Datei (die schreibt der Service auch dann
            # weiter, wenn der Watchdog-Thread haengt).
            lc = wd.get("last_check")
            if isinstance(lc, (int, float)) and lc > 0:
                gemessen = now - max(0.0, mono() - float(lc))
        out = self.update(temp, gemessen, "watchdog.cpu_temp")
        out["kuehlung"] = self.kuehlung(self.read_pi_kuehler(kuehler_path))
        return out

    # === Deutung ===

    def _level_for(self, temp: float) -> str:
        c, prev = self.cfg, self._level
        if temp >= c.kritisch_c or (prev == "distressed" and temp > c.kritisch_c - c.hysterese_c):
            return "distressed"
        if temp >= c.warn_c or (prev in ("strained", "distressed") and temp > c.warn_c - c.hysterese_c):
            return "strained"
        return "ok"

    def update(self, temp_c: Optional[float], gemessen_ts: Optional[float], quelle: str) -> Dict:
        """Neuen Messwert verarbeiten und den aktuellen Zustand liefern."""
        now = self._clock()
        frisch = (temp_c is not None and gemessen_ts is not None
                  and 0 <= now - gemessen_ts <= self.cfg.max_alter_s)
        if not frisch:
            zustand, ursache = "unknown", "kein frischer Temperaturwert"
        else:
            prev_level = self._level
            self._level = self._level_for(float(temp_c))
            if self._level == "ok" and prev_level in ("strained", "distressed"):
                self._erholung_bis = now + self.cfg.erholung_s
            if self._level == "distressed":
                zustand, ursache = "distressed", f"CPU {temp_c:.1f} °C (kritisch ab {self.cfg.kritisch_c:.0f} °C)"
            elif self._level == "strained":
                zustand, ursache = "strained", f"CPU {temp_c:.1f} °C (warm ab {self.cfg.warn_c:.0f} °C)"
            elif now < self._erholung_bis:
                zustand, ursache = "recovering", f"CPU wieder {temp_c:.1f} °C"
            else:
                zustand, ursache = "ok", f"CPU {temp_c:.1f} °C"
        if zustand != self._zustand:
            self._zustand, self._seit = zustand, now
            self._kennung = uuid.uuid4().hex[:8]
        self._letzter = {
            "zustand": self._zustand,
            "ursache": ursache,
            "kennung": self._kennung,
            "seit": round(self._seit, 1),
            "quelle": quelle if frisch else None,
            "messwert": ({"cpu_temp_c": round(float(temp_c), 1), "gemessen": round(gemessen_ts, 1),
                          "alter_s": round(now - gemessen_ts, 1)} if frisch else None),
            "gueltig_bis": round(gemessen_ts + self.cfg.max_alter_s, 1) if frisch else None,
        }
        return dict(self._letzter)


def bestaetigung(soma: Dict, report: Optional[Dict], now: float, timeout_s: float = 20.0) -> Dict:
    """Rueckmeldung des Koerpers dem aktuellen Zustand zuordnen.

    stufe:
      AUSSTEHEND      noch keine passende Rueckmeldung, Zeitlimit laeuft
      UNBESTAETIGT    Zeitlimit abgelaufen, keine passende Rueckmeldung (Schweigen ist ein Befund)
      ACCEPTED        Koerper hat den Zustand gelesen, aber nichts ausgefuehrt (offline, Not-Aus ...)
      EXECUTED        Controller hat die Pose mit Soma-Kanal gesendet bzw. haelt sie
    VERIFIED gibt es hier nicht: dafuer braeuchte es eine unabhaengige Beobachtung
    (z. B. Sonoff-Kamera), keine Selbstauskunft der Bridge.
    Eine Rueckmeldung mit anderer Kennung zaehlt nie - auch keine verspaetete.
    """
    if not report or report.get("kennung") != soma.get("kennung"):
        offen = now - float(soma.get("seit") or now)
        return {"stufe": "UNBESTAETIGT" if offen > timeout_s else "AUSSTEHEND",
                "kennung": soma.get("kennung")}
    ergebnis = report.get("ergebnis")
    # "sent"/"unchanged": die Pose mit Soma-Kanal ging raus bzw. steht schon so.
    # Alles andere (offline, estop, ueberschrieben, neutral, idle) = nur gelesen.
    stufe = "EXECUTED" if ergebnis in ("sent", "unchanged") else "ACCEPTED"
    return {"stufe": stufe, "kennung": soma.get("kennung"), "ergebnis": ergebnis,
            "alter_s": round(now - float(report.get("ts") or now), 1)}


_tracker: Optional[SomaTracker] = None


def get_soma_tracker() -> SomaTracker:
    global _tracker
    if _tracker is None:
        _tracker = SomaTracker()
    return _tracker


def write_test_override(cpu_temp_c: float, dauer_s: float = 60.0, path: str = TEST_OVERRIDE_PATH,
                        pi_kuehler_stufe: Optional[int] = None) -> None:
    """Testvorgabe setzen (Abnahme Stufe 2). Atomar geschrieben, verfaellt selbst."""
    data = {"cpu_temp_c": float(cpu_temp_c), "until": time.time() + float(dauer_s)}
    if pi_kuehler_stufe is not None:
        data["pi_kuehler_stufe"] = int(pi_kuehler_stufe)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
