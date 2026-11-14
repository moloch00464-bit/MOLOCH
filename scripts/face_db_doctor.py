#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face-DB-Doktor: prueft data/face_embeddings.json auf innere Widersprueche.

Ohne Optionen nur lesend. Zeigt pro Person und Eintragsart (Basis, snap,
teach, cl, ...) wie gut die Embeddings zueinander passen, und meldet:
  - verdaechtige Namen (Fuellwoerter wie "nicht" - Fehl-Enrollment per Sprache)
  - gleiche Person in verschiedener Schreibweise ("markus" / "Markus")
  - Gruppen derselben Person, die NICHT zueinander passen
    (Hinweis auf unterschiedliche Vorverarbeitung beim Enrollment)
  - einzelne Ausreisser innerhalb einer Person

Aufraeumen nur ausdruecklich und immer mit Backup:
    python3 scripts/face_db_doctor.py                              # nur Bericht
    python3 scripts/face_db_doctor.py --remove nicht               # zeigt, was wegfiele
    python3 scripts/face_db_doctor.py --remove nicht --apply       # schreibt (Backup daneben)
    python3 scripts/face_db_doctor.py --remove-kind markus:snap --apply

WICHTIG: Vor --apply den Service stoppen (sudo systemctl stop moloch). Sonst
ueberschreibt Continuous Learning die Datei mit seinem alten Stand.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "face_embeddings.json")

SAME_PERSON_MIN = 0.40     # darunter passen zwei Gruppen derselben Person nicht zusammen
OUTLIER_MAX = 0.30         # bester Treffer zu den eigenen Eintraegen darunter = Ausreisser
SUSPICIOUS_NAMES = {
    "nicht", "unbekannt", "wohl", "auch", "ja", "nein", "doch", "schon", "mal", "halt",
    "eben", "nun", "so", "nur", "noch", "sehr", "klar", "kein", "mein", "der", "die", "das",
    "gut", "ok", "okay", "ein", "eine",
}


def split_key(key: str) -> Tuple[str, str]:
    """'Markus#teach_20260401' -> ('Markus', 'teach'); 'markus' -> ('markus', 'basis')."""
    if "#" not in key:
        return key, "basis"
    name, rest = key.split("#", 1)
    return name, rest.split("_", 1)[0] or "?"


def _unit(vectors: List[List[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def analyze(db: Dict[str, List[float]]) -> Dict:
    """Bericht als Datenstruktur (fuer Tests und Ausgabe)."""
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    bad_shape = []
    dims = defaultdict(int)
    for key, emb in db.items():
        if not isinstance(emb, list) or not emb:
            bad_shape.append(key)
            continue
        dims[len(emb)] += 1
        groups[split_key(key)].append(key)
    main_dim = max(dims, key=dims.get) if dims else 0
    for key, emb in db.items():
        if isinstance(emb, list) and emb and len(emb) != main_dim:
            bad_shape.append(key)
    usable = {g: [k for k in ks if k not in bad_shape] for g, ks in groups.items()}
    usable = {g: ks for g, ks in usable.items() if ks}
    vec = {g: _unit([db[k] for k in ks]) for g, ks in usable.items()}

    group_rows = []
    for (name, kind), ks in sorted(usable.items()):
        v = vec[(name, kind)]
        intra = None
        if len(ks) > 1:
            s = v @ v.T
            intra = float(s[~np.eye(len(ks), dtype=bool)].mean())
        group_rows.append({"name": name, "kind": kind, "count": len(ks), "intra": intra})

    # Personen ueber Gross-/Kleinschreibung zusammenfassen
    persons: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for g in usable:
        persons[g[0].lower()].append(g)

    findings = []
    for g in sorted({g[0] for g in usable}):
        if g.lower() in SUSPICIOUS_NAMES:
            n = sum(len(ks) for (name, _), ks in usable.items() if name == g)
            findings.append({"type": "verdaechtiger_name", "name": g, "count": n})
    for low, gs in sorted(persons.items()):
        spellings = sorted({g[0] for g in gs})
        if len(spellings) > 1:
            findings.append({"type": "schreibweise", "names": spellings})
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                cross = float((vec[gs[i]] @ vec[gs[j]].T).mean())
                if cross < SAME_PERSON_MIN:
                    findings.append({"type": "inkompatibel", "a": "%s#%s" % gs[i],
                                     "b": "%s#%s" % gs[j], "sim": round(cross, 2)})
        keys = [k for g in gs for k in usable[g]]
        if len(keys) > 2:
            allv = _unit([db[k] for k in keys])
            s = allv @ allv.T
            np.fill_diagonal(s, -1.0)
            for k, best in zip(keys, s.max(axis=1)):
                if best < OUTLIER_MAX:
                    findings.append({"type": "ausreisser", "key": k, "best": round(float(best), 2)})
    for k in sorted(set(bad_shape)):
        findings.append({"type": "defekt", "key": k})

    return {"entries": len(db), "dim": main_dim, "groups": group_rows, "findings": findings}


def select_removals(db: Dict, names: List[str], kinds: List[str]) -> List[str]:
    """Schluessel, die --remove NAME / --remove-kind NAME:ART treffen (exakte Schreibweise)."""
    wanted_kinds = set()
    for spec in kinds:
        if ":" not in spec:
            raise ValueError(f"--remove-kind braucht NAME:ART, nicht '{spec}'")
        wanted_kinds.add(tuple(spec.split(":", 1)))
    out = []
    for key in db:
        name, kind = split_key(key)
        if name in names or (name, kind) in wanted_kinds:
            out.append(key)
    return out


def write_db(path: str, db: Dict) -> str:
    """Backup neben die Datei legen, dann atomar schreiben. Returns: Backup-Pfad."""
    backup = f"{path}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
    with open(path, "rb") as src, open(backup, "wb") as dst:
        dst.write(src.read())
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=1, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return backup


def print_report(rep: Dict) -> None:
    print(f"{rep['entries']} Eintraege, Dimension {rep['dim']}\n")
    print(f"{'Person':<16}{'Art':<10}{'Anzahl':>7}  Aehnlichkeit untereinander")
    for g in rep["groups"]:
        intra = "-" if g["intra"] is None else f"{g['intra']:.2f}"
        print(f"{g['name']:<16}{g['kind']:<10}{g['count']:>7}  {intra}")
    print()
    if not rep["findings"]:
        print("Keine Auffaelligkeiten.")
        return
    for f in rep["findings"]:
        t = f["type"]
        if t == "verdaechtiger_name":
            print(f"[!] Name '{f['name']}' ({f['count']} Eintraege) ist ein Fuellwort - "
                  f"vermutlich Fehl-Enrollment per Sprache.")
        elif t == "schreibweise":
            print(f"[!] Gleiche Person in mehreren Schreibweisen: {', '.join(f['names'])}")
        elif t == "inkompatibel":
            print(f"[!] {f['a']} und {f['b']} passen nicht zusammen (Aehnlichkeit {f['sim']}). "
                  f"Hinweis auf unterschiedliche Vorverarbeitung beim Enrollment.")
        elif t == "ausreisser":
            print(f"[ ] Ausreisser: {f['key']} (bester Treffer zur eigenen Person {f['best']})")
        elif t == "defekt":
            print(f"[!] Defekter Eintrag: {f['key']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Face-DB pruefen und gezielt aufraeumen")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--remove", action="append", default=[], metavar="NAME",
                    help="alle Eintraege dieser Person entfernen (exakte Schreibweise)")
    ap.add_argument("--remove-kind", action="append", default=[], metavar="NAME:ART",
                    help="nur eine Eintragsart entfernen, z.B. markus:snap")
    ap.add_argument("--apply", action="store_true", help="wirklich schreiben (mit Backup)")
    args = ap.parse_args(argv)

    with open(args.db, "r", encoding="utf-8") as f:
        db = json.load(f)
    print_report(analyze(db))

    if not (args.remove or args.remove_kind):
        return 0
    doomed = select_removals(db, args.remove, args.remove_kind)
    print(f"\nEntfernen: {len(doomed)} von {len(db)} Eintraegen")
    if not doomed:
        return 0
    if len(doomed) == len(db):
        print("ABBRUCH: das wuerde die gesamte Face-DB leeren.")
        return 2
    if not args.apply:
        print("Nur Vorschau. Mit --apply schreiben (Service vorher stoppen).")
        return 0
    for k in doomed:
        del db[k]
    backup = write_db(args.db, db)
    print(f"Geschrieben. Backup: {backup}\nDanach Service starten, damit die DB neu geladen wird.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
