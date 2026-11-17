#!/usr/bin/env python3
"""
Sinne - ist die Wahrnehmung gerade echt? (Nervensystem-Plan, Klasse "Sinne")

Ein ausgefallener Sinn beschaedigt keine Hardware, fuehrt aber zu erfundenen Antworten:
Ohne Bild stand im Prompt bisher "niemand im Bild" - fuer das Sprachmodell eine Wahrnehmung,
in Wahrheit ein Nichtwissen. Diese Pruefung sagt, ob der Kamerazustand im Status-JSON
gerade echt ist. Beide Antwortwege nutzen sie (Pi-Tentakel direkt, KI-Zentrale ueber /live.sinne).

Klassen (NERVENSYSTEM_PLAN.md, Stand 16.11.):
    Schutz  Schaden an der Haupthardware -> Soma, Koerper, Rueckmeldung (core/awareness/soma.py)
    Sinne   Kamera, Mikrofon, PC        -> nur eine Zeile im Prompt, kein Koerperausdruck (hier)
    Rest    Fehlermeldung in Log/Cockpit

Reine Funktion ohne I/O. Fehlende Felder gelten als unbekannt, nie als ok.
"""

from typing import Dict, Optional

MAX_STATUS_ALTER_S = 15.0   # aelteres Status-JSON: Service liefert nicht -> unbekannt
MAX_BILD_ALTER_S = 10.0     # aelteres Bild: Pipeline steht -> blind


def kamera_zustand(status: Optional[Dict], status_alter_s: Optional[float]) -> Dict:
    """{"ok": True|False|None, "grund": str|None, "zeile": str|None}

    ok=None heisst unbekannt - fuer den Prompt genauso zu behandeln wie False:
    Moloch weiss dann nicht, was die Kamera sieht.
    """
    grund = None
    ok: Optional[bool] = True
    if not isinstance(status, dict) or status_alter_s is None or status_alter_s > MAX_STATUS_ALTER_S:
        ok, grund = None, "keine aktuellen Daten vom Service"
    else:
        reachable = (status.get("watchdog") or {}).get("camera_reachable")
        alter = status.get("frame_age")
        if reachable is False:
            ok, grund = False, "Kamera im Netz nicht erreichbar"
        elif isinstance(alter, (int, float)) and alter < 0:
            ok, grund = False, "die Bildverarbeitung liefert keine Bilder"
        elif isinstance(alter, (int, float)) and alter > MAX_BILD_ALTER_S:
            ok, grund = False, f"letztes Bild vor {int(alter)} s"
        elif reachable is None and not isinstance(alter, (int, float)):
            ok, grund = None, "Kamerazustand unbekannt"
    zeile = None
    if ok is not True:
        # Als Tatsache in Ich-Form, nicht als Anweisung: Ein kleines Modell gibt Anweisungen
        # sonst woertlich an Markus weiter (Ehrlichkeits-Pilot 2026-11-17, E06 Weg B).
        # Plappert es diese Zeile nach, ist die Antwort trotzdem wahr.
        zeile = f"KAMERA AUS ({grund}): Ich sehe gerade NICHTS - ob jemand da ist und wer, weiss ich nicht."
    return {"ok": ok, "grund": grund, "zeile": zeile}
