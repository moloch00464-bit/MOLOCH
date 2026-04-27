"""Setzt media_stream_mic = Allow fuer alle MOLOCH-Cockpit-URLs in Chrome's Prefs.

Chrome MUSS komplett geschlossen sein wenn das Skript laeuft, sonst
ueberschreibt Chrome die Prefs beim naechsten Save.

Run:
    python pc\\fix_chrome_mic_prefs.py
"""
import json
import os
import shutil
import time
from pathlib import Path

PREFS = Path(os.environ["LOCALAPPDATA"]) / "Google/Chrome/User Data/Default/Preferences"
URLS = [
    "http://localhost:9000,*",
    "http://localhost:11800,*",
    "http://localhost:11700,*",
    "https://moloch.local:9443,*",
    "https://192.168.178.30:9443,*",
    "http://192.168.178.30:9100,*",
    "http://192.168.178.20:11800,*",
    "http://192.168.178.20:11700,*",
]


def _windows_filetime() -> str:
    """Chrome stores timestamps as 100-nanosecond intervals since 1601-01-01."""
    return str(int((time.time() + 11644473600) * 10_000_000))


def main() -> int:
    if not PREFS.exists():
        print(f"[mic] Chrome Prefs not found at {PREFS}")
        return 1

    backup = PREFS.with_suffix(PREFS.suffix + f".bak_{int(time.time())}")
    shutil.copy2(PREFS, backup)
    print(f"[mic] backup -> {backup}")

    with PREFS.open("r", encoding="utf-8") as f:
        prefs = json.load(f)

    profile = prefs.setdefault("profile", {})
    cs = profile.setdefault("content_settings", {})
    excs = cs.setdefault("exceptions", {})
    mic = excs.setdefault("media_stream_mic", {})

    ts = _windows_filetime()
    added = 0
    skipped = 0
    for u in URLS:
        cur = mic.get(u)
        if isinstance(cur, dict) and cur.get("setting") == 1:
            print(f"  already allow: {u}")
            skipped += 1
            continue
        mic[u] = {
            "setting": 1,
            "last_modified": ts,
            "last_used": ts,
            "last_visit": ts,
        }
        print(f"  ALLOW added: {u}")
        added += 1

    if added == 0:
        print(f"[mic] keine Aenderung noetig (alle {skipped} URLs schon Allow)")
        return 0

    tmp = PREFS.with_suffix(PREFS.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(prefs, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, PREFS)
    print(f"[mic] {added} URLs neu auf Allow gesetzt, Prefs geschrieben.")
    print("[mic] WICHTIG: Chrome muss komplett zu sein wenn das Skript laeuft!")
    print("[mic] Chrome jetzt neu starten - alle MOLOCH-URLs haben Mic-Permission.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
