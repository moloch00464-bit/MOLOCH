#!/usr/bin/env python3
"""
test_blind_comparison — Blindvergleich Logging.

Loggt fuer jede der letzten N User-Fragen zwei Antwort-Varianten:
  A: Original aus Konversations-Log (volle Pipeline)
  B: Minimale Antwort via Qwen-lokal (POST /api/generate, kein Kontext)

Schreibt nach logs/blind_comparison_YYYY-MM-DD.log

Gate: 14 Operationstage (wie Phase 6) — ueberspringen mit --force

Verwendung:
  python3 tests/test_blind_comparison.py
  python3 tests/test_blind_comparison.py --force
  python3 tests/test_blind_comparison.py --force --n 10
"""

import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# --- Pfade ---
PROJECT_ROOT = Path(__file__).parent.parent
JOURNAL_DIR  = Path("/mnt/moloch-data/memory/journal")
CONV_DIR     = Path("/mnt/moloch-data/memory/conversations")
LOGS_DIR     = PROJECT_ROOT / "logs"

GATE_OPERATION_DAYS = 14
QWEN_URL            = "http://localhost:8000/api/generate"
QWEN_MODEL          = "qwen2.5:1.5b"
QWEN_TIMEOUT_S      = 20

# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def count_operation_days() -> int:
    try:
        days = len([
            f for f in os.listdir(JOURNAL_DIR)
            if re.match(r'\d{4}-\d{2}-\d{2}\.jsonl', f)
        ])
        archive = JOURNAL_DIR / "archive"
        if archive.is_dir():
            days += len([
                f for f in os.listdir(archive)
                if re.match(r'\d{4}-\d{2}-\d{2}\.jsonl', f)
            ])
        return days
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Konversations-Snippets laden
# ---------------------------------------------------------------------------

def load_recent_pairs(n: int) -> list[dict]:
    """
    Laedt die letzten n User-Fragen + Moloch-Antworten aus den neuesten
    Konversations-Dateien. Gibt Liste von {"user": ..., "moloch": ..., "ts": ...} zurueck.
    """
    pairs: list[dict] = []
    if not CONV_DIR.is_dir():
        return pairs

    files = sorted(CONV_DIR.glob("*.json"), reverse=True)
    for conv_file in files:
        if len(pairs) >= n:
            break
        try:
            with open(conv_file) as f:
                entries = json.load(f)
        except Exception:
            continue

        if not isinstance(entries, list):
            continue

        # Paare aus aufeinanderfolgenden user/moloch Eintraegen extrahieren
        for i in range(len(entries) - 1):
            a = entries[i]
            b = entries[i + 1]
            a_sender = a.get("sender") or a.get("role") or ""
            b_sender = b.get("sender") or b.get("role") or ""
            a_text   = a.get("text") or a.get("content") or ""
            b_text   = b.get("text") or b.get("content") or ""

            is_user_then_moloch = (
                a_sender in ("user", "markus")
                and b_sender in ("moloch", "assistant")
            )
            if is_user_then_moloch and a_text.strip() and b_text.strip():
                pairs.append({
                    "ts":     a.get("ts", ""),
                    "user":   a_text.strip(),
                    "moloch": b_text.strip(),
                    "source": conv_file.name,
                })
                if len(pairs) >= n:
                    break

    # Neueste zuerst (bereits in umgekehrter Datei-Reihenfolge geladen)
    return pairs[:n]


# ---------------------------------------------------------------------------
# Minimale Qwen-Antwort
# ---------------------------------------------------------------------------

def qwen_minimal(user_question: str) -> str:
    """
    Fragt Qwen lokal ohne Kontext: nur System-Profil + User-Frage.
    Gibt Antwort-String zurueck oder Fehlertext.
    """
    payload = {
        "model":  QWEN_MODEL,
        "prompt": user_question,
        "system": (
            "Du bist Moloch, eine KI auf einem Raspberry Pi. "
            "Antworte kurz und direkt auf Deutsch."
        ),
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 200},
    }
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        QWEN_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=QWEN_TIMEOUT_S) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "").strip()
    except urllib.error.URLError as e:
        return f"[QWEN NICHT ERREICHBAR: {e}]"
    except Exception as e:
        return f"[FEHLER: {e}]"


# ---------------------------------------------------------------------------
# Report schreiben
# ---------------------------------------------------------------------------

def write_report(pairs: list[dict], log_path: Path) -> None:
    sep = "=" * 72

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{sep}\n")
        f.write(f"Blindvergleich — {datetime.now().isoformat()}\n")
        f.write(f"{len(pairs)} Paare — Modell B: {QWEN_MODEL} (kein Kontext)\n")
        f.write(f"{sep}\n\n")

        for i, pair in enumerate(pairs, 1):
            f.write(f"--- Paar {i}/{len(pairs)} | {pair['ts']} | {pair['source']} ---\n")
            f.write(f"USER:  {pair['user'][:300]}\n\n")
            f.write(f"A (Original / volle Pipeline):\n  {pair['moloch'][:600]}\n\n")
            f.write(f"B (Minimal / kein Kontext):\n  {pair.get('minimal', '[nicht generiert]')[:600]}\n\n")

            # Kurz-Diff
            a_len = len(pair['moloch'])
            b_raw = pair.get('minimal', '')
            b_len = len(b_raw)
            ich_a = pair['moloch'].lower().count("ich ")
            ich_b = b_raw.lower().count("ich ")
            f.write(f"  Laenge A={a_len}Z / B={b_len}Z | 'ich' A={ich_a} B={ich_b}\n")
            f.write("\n")

        f.write(f"{sep}\n")

    print(f"  Report: {log_path}")


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Moloch Blindvergleich")
    parser.add_argument(
        "--force", action="store_true",
        help="Gate ueberspringen (fuer manuelle Ausfuehrung)"
    )
    parser.add_argument(
        "--n", type=int, default=5,
        help="Anzahl Konversationspaare (default: 5)"
    )
    args = parser.parse_args()

    days = count_operation_days()

    print(f"\n=== Blindvergleich Gate ===")
    print(f"  Operationstage: {days}/{GATE_OPERATION_DAYS}")

    if not args.force and days < GATE_OPERATION_DAYS:
        print(
            f"\n  Gate nicht offen: {days}/{GATE_OPERATION_DAYS} Tage.\n"
            f"  Noch {GATE_OPERATION_DAYS - days} Tag(e) noetig.\n"
            f"  Tipp: --force um Gate zu ueberspringen.\n"
        )
        sys.exit(0)

    if args.force and days < GATE_OPERATION_DAYS:
        print(f"  --force gesetzt — Gate uebersprungen.")

    print(f"\n  Lade letzte {args.n} Konversationspaare...")
    pairs = load_recent_pairs(args.n)

    if not pairs:
        print("  Keine Konversationspaare gefunden. Abbruch.")
        sys.exit(1)

    print(f"  {len(pairs)} Paare geladen. Generiere minimale Antworten (Qwen)...\n")

    for i, pair in enumerate(pairs, 1):
        user_q = pair["user"][:200]
        print(f"  [{i}/{len(pairs)}] {user_q[:80]}...")
        pair["minimal"] = qwen_minimal(user_q)
        minimal_preview = pair["minimal"][:100].replace("\n", " ")
        print(f"          B: {minimal_preview}...")

    log_path = LOGS_DIR / f"blind_comparison_{datetime.now().strftime('%Y-%m-%d')}.log"
    write_report(pairs, log_path)
    print(f"\n=== Fertig. {len(pairs)} Paare verglichen. ===\n")


if __name__ == "__main__":
    main()
