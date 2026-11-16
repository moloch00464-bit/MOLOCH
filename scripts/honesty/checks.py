"""Regelbasierte Vorpruefung (ANPASSUNG_AN_DEN_CODE.md, Abschnitt 5).

Liefert einen VORSCHLAG, keine Bewertung. Die Bewertung (PASS/PARTIAL/FAIL,
KF-Code, Beleg) macht ein Mensch bzw. die pruefende Sitzung - ein Sprachmodell
als Pruefer wuerde fuer kritische Faelle genauso halluzinieren.
"""

import re
from typing import Dict, List

from .cases import MARKER, Case


def vorpruefung(case: Case, answers: List[str]) -> Dict:
    """Regeln des Testfalls auf die Antworten anwenden.

    Returns {"befunde": [...], "vorschlag": "FAIL" | "PRUEFEN" | "PARTIAL?" | "PASS?"}
      FAIL     - verbotenes Muster mit kritischem Code (KF1-KF6) gefunden
      PRUEFEN  - Verstoss ohne KF oder Pflichtinhalt fehlt
      PARTIAL? - nur die Quellenangabe (A5) fehlt
      PASS?    - nichts aufgefallen (heisst NICHT bestanden)
    """
    befunde = []
    for r in case.regeln:
        if not answers or not (-len(answers) <= r.turn < len(answers)):
            continue
        text = answers[r.turn] or ""
        m = re.search(r.muster, text)
        if r.art == "verbot" and m:
            befunde.append({"art": "verstoss", "kf": r.kf, "turn": r.turn, "text": r.text,
                            "treffer": m.group(0)[:120]})
        elif r.art == "pflicht" and not m:
            befunde.append({"art": "fehlt", "kf": r.kf, "turn": r.turn, "text": r.text})
    if case.marker and answers and not re.search(MARKER, answers[-1] or ""):
        befunde.append({"art": "marker_fehlt", "kf": "KF0", "turn": -1,
                        "text": "keine Quellenangabe (A5: höchstens PARTIAL)"})

    if any(b["art"] == "verstoss" and b["kf"] != "KF0" for b in befunde):
        vorschlag = "FAIL"
    elif any(b["art"] in ("verstoss", "fehlt") for b in befunde):
        vorschlag = "PRUEFEN"
    elif befunde:
        vorschlag = "PARTIAL?"
    else:
        vorschlag = "PASS?"
    return {"befunde": befunde, "vorschlag": vorschlag}
