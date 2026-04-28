#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weekly_compactor — Woechentlicher Journal-Verdichter (Phase 4e Gate).

Laeuft jeden Sonntag 23:00 (via systemd-Timer), gates sich selbst wenn < 7 Tage
unarchivierte JSONL-Tagesdateien im Journal-Verzeichnis liegen.

Ablauf:
  1. is_ready(): >= 7 unarchivierte JSONL-Tagesdateien?
  2. compact(): letzte 7 Tage lesen, via DeepSeek (oder Qwen-lokal Fallback)
                zu 3-5 Themen zusammenfassen
  3. Themen als weekly_character_theme_YYYY-WW.md schreiben
  4. Verarbeitete Dateien nach archive/ verschieben
  5. phase_gates.json aktualisieren (days_collected, armed, last_run)

Storage:
  /mnt/moloch-data/memory/journal/                       — Tages-JSONL (YYYY-MM-DD.jsonl)
  /mnt/moloch-data/memory/journal/archive/               — verarbeitete Dateien
  /mnt/moloch-data/memory/themes/                        — weekly_character_theme_*.md
  /home/molochzuhause/moloch/config/phase_gates.json     — Phase-Gate-Config

Aufruf:
  python3 -m core.memory.weekly_compactor
  oder
  python3 /home/molochzuhause/moloch/core/memory/weekly_compactor.py
"""

import json
import logging
import os
import re
import shutil
import socket
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("WeeklyCompactor")

# --------------------------------------------------------------------------
# Pfade
# --------------------------------------------------------------------------
JOURNAL_DIR = "/mnt/moloch-data/memory/journal"
ARCHIVE_DIR = os.path.join(JOURNAL_DIR, "archive")
THEMES_DIR = "/mnt/moloch-data/memory/themes"
PHASE_GATES_PATH = "/home/molochzuhause/moloch/config/phase_gates.json"
API_KEYS_PATH = "/home/molochzuhause/moloch/config/api_keys.json"

# Regex fuer Tages-JSONL-Dateien (keine scored_*, keine _state.json)
DAILY_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\.jsonl$")

REQUIRED_DAYS = 7
DEEPSEEK_TIMEOUT = 60
QWEN_TIMEOUT = 90
QWEN_URL = "http://localhost:8000/api/generate"
QWEN_MODEL = "qwen2.5:1.5b"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_write_json(path: str, data) -> None:
    """Atomar schreiben (tempfile + os.replace, NTFS-Fallback)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp_path, path)
        except OSError:
            with open(tmp_path, "r", encoding="utf-8") as f_src:
                content = f_src.read()
            with open(path, "w", encoding="utf-8") as f_dst:
                f_dst.write(content)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        logger.error(f"[COMPACTOR] JSON-Schreiben fehlgeschlagen ({path}): {e}")
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _list_daily_files(journal_dir: str) -> List[str]:
    """Listet alle Tages-JSONL-Dateien im Journal-Verzeichnis (sortiert nach Datum)."""
    if not os.path.isdir(journal_dir):
        return []
    entries = []
    for name in os.listdir(journal_dir):
        if DAILY_RE.match(name):
            entries.append(name)
    entries.sort()
    return entries


# --------------------------------------------------------------------------
# is_ready
# --------------------------------------------------------------------------
def is_ready(journal_dir: str = JOURNAL_DIR) -> Tuple[bool, int]:
    """
    Prueft ob genug unarchivierte Tagesdateien vorliegen.

    Returns:
        (ready, n_days) — ready=True wenn n_days >= REQUIRED_DAYS.
    """
    files = _list_daily_files(journal_dir)
    n = len(files)
    return (n >= REQUIRED_DAYS, n)


# --------------------------------------------------------------------------
# LLM-Adapter
# --------------------------------------------------------------------------
def _load_deepseek_key() -> Optional[Dict[str, str]]:
    if not os.path.isfile(API_KEYS_PATH):
        return None
    try:
        with open(API_KEYS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        ds = data.get("deepseek") or {}
        if ds.get("api_key"):
            return {
                "api_key": ds["api_key"],
                "base_url": ds.get("base_url", "https://api.deepseek.com/v1"),
                "model": ds.get("model", "deepseek-chat"),
            }
    except Exception as e:
        logger.warning(f"[COMPACTOR] api_keys.json lesen fehlgeschlagen: {e}")
    return None


def _call_deepseek(prompt: str, cfg: Dict[str, str]) -> Optional[str]:
    """Ruft DeepSeek Chat-API auf. Gibt Antwort-String zurueck oder None bei Fehler."""
    try:
        url = cfg["base_url"].rstrip("/") + "/chat/completions"
        body = {
            "model": cfg["model"],
            "messages": [
                {"role": "system",
                 "content": "Du bist ein analytischer Assistent. Fasse Wochen-Events kurz, sachlich, deutsch zusammen."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 600,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg['api_key']}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=DEEPSEEK_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        choices = payload.get("choices") or []
        if not choices:
            return None
        msg = choices[0].get("message") or {}
        return msg.get("content")
    except Exception as e:
        logger.warning(f"[COMPACTOR] DeepSeek-Aufruf fehlgeschlagen: {e}")
        return None


def _call_qwen_local(prompt: str) -> Optional[str]:
    """Fallback: lokales Qwen via hailo-ollama auf localhost:8000."""
    try:
        body = {
            "model": QWEN_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 600},
        }
        req = urllib.request.Request(
            QWEN_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=QWEN_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("response")
    except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
        logger.warning(f"[COMPACTOR] Qwen-Lokal nicht erreichbar: {e}")
        return None
    except Exception as e:
        logger.warning(f"[COMPACTOR] Qwen-Lokal-Aufruf fehlgeschlagen: {e}")
        return None


def default_llm_fn(prompt: str) -> Optional[str]:
    """
    Standard-LLM-Aufruf: erst DeepSeek, dann Qwen-Lokal als Fallback.
    """
    cfg = _load_deepseek_key()
    if cfg:
        out = _call_deepseek(prompt, cfg)
        if out:
            return out
        logger.info("[COMPACTOR] DeepSeek lieferte nichts, Fallback auf Qwen-Lokal.")
    return _call_qwen_local(prompt)


# --------------------------------------------------------------------------
# compact
# --------------------------------------------------------------------------
def _read_jsonl(path: str) -> List[Dict]:
    """Liest eine JSONL-Datei. Robust gegen einzelne kaputte Zeilen."""
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"[COMPACTOR] {path} lesen fehlgeschlagen: {e}")
    return out


def _summarize_events(events: List[Dict]) -> Tuple[str, List[str]]:
    """
    Bildet einen kompakten Prompt-Block aus Events (max ~2000 Zeichen)
    und gibt zusaetzlich die Liste aller Event-IDs zurueck.

    Wir kuerzen pro Event auf das Wesentliche: type | tension_delta | zone | interpretation.
    Wenn Volumen zu gross: Sampling (jeden N-ten Eintrag).
    """
    all_ids: List[str] = []
    lines: List[str] = []
    for ev in events:
        eid = str(ev.get("event_id", ""))
        if eid:
            all_ids.append(eid)
        etype = ev.get("type", "?")
        td = ev.get("tension_delta", 0.0)
        zone = ev.get("zone") or ""
        interp = (ev.get("interpretation") or "").strip().replace("\n", " ")
        if len(interp) > 120:
            interp = interp[:117] + "..."
        zone_part = f" zone={zone}" if zone else ""
        lines.append(f"{eid} {etype} td={td:+.2f}{zone_part} :: {interp}")

    joined = "\n".join(lines)
    if len(joined) <= 2000:
        return joined, all_ids

    # Sampling: jeden N-ten Eintrag, bis < 2000.
    step = max(2, (len(joined) // 2000) + 1)
    sampled = lines[::step]
    joined = "\n".join(sampled)
    while len(joined) > 2000 and len(sampled) > 5:
        step += 1
        sampled = lines[::step]
        joined = "\n".join(sampled)
    return joined, all_ids


def _build_prompt(summary_block: str, day_files: List[str]) -> str:
    """Baut den Prompt fuer den LLM."""
    range_hint = ""
    if day_files:
        first = day_files[0].replace(".jsonl", "")
        last = day_files[-1].replace(".jsonl", "")
        range_hint = f" Zeitraum: {first} bis {last}."
    return (
        "Analysiere die folgenden Charakter-Events einer Woche.{range_hint}\n"
        "Identifiziere 3 bis 5 wiederkehrende Themen/Muster.\n"
        "Antworte AUSSCHLIESSLICH im Format (eine Zeile pro Thema):\n"
        "- Thema: <kurze Beschreibung in einem Satz>\n\n"
        "Events (gekuerzt):\n{summary}\n"
    ).format(range_hint=range_hint, summary=summary_block)


def _parse_themes(llm_text: str) -> List[str]:
    """Extrahiert Themen aus LLM-Antwort. Akzeptiert '- Thema: ...' und '- ...'."""
    if not llm_text:
        return []
    themes: List[str] = []
    for line in llm_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^[-*]\s*(?:Thema\s*:\s*)?(.+)$", line, re.IGNORECASE)
        if m:
            t = m.group(1).strip().rstrip(".")
            if t:
                themes.append(t)
        else:
            m2 = re.match(r"^\d+[\.\)]\s*(?:Thema\s*:\s*)?(.+)$", line, re.IGNORECASE)
            if m2:
                t = m2.group(1).strip().rstrip(".")
                if t:
                    themes.append(t)
    return themes[:5]


def _iso_week_tag(day_files: List[str]) -> str:
    """Gibt YYYY-WW Tag basierend auf juengster verarbeiteter Datei."""
    if not day_files:
        return datetime.now(timezone.utc).strftime("%G-W%V")
    last = day_files[-1].replace(".jsonl", "")
    try:
        d = datetime.strptime(last, "%Y-%m-%d")
        return d.strftime("%G-W%V")
    except ValueError:
        return datetime.now(timezone.utc).strftime("%G-W%V")


def compact(journal_dir: str = JOURNAL_DIR,
            output_dir: str = THEMES_DIR,
            llm_fn: Optional[Callable[[str], Optional[str]]] = None) -> Tuple[str, List[str]]:
    """
    Kompaktiert die letzten 7 Tage Journal-Events zu Wochen-Themen.

    Returns:
        (md_path, processed_files) — Pfad zur geschriebenen Markdown-Datei
        und Liste der Tages-Dateinamen die verarbeitet wurden.
    """
    if llm_fn is None:
        llm_fn = default_llm_fn

    files = _list_daily_files(journal_dir)
    if len(files) < REQUIRED_DAYS:
        raise RuntimeError(
            f"compact() aufgerufen aber nur {len(files)} Tage vorhanden (>= {REQUIRED_DAYS} erforderlich)"
        )

    last7 = files[-REQUIRED_DAYS:]

    all_events: List[Dict] = []
    for fname in last7:
        path = os.path.join(journal_dir, fname)
        all_events.extend(_read_jsonl(path))

    summary_block, all_ids = _summarize_events(all_events)
    prompt = _build_prompt(summary_block, last7)

    llm_text = llm_fn(prompt) or ""
    themes = _parse_themes(llm_text)
    if not themes:
        themes = ["Keine klaren Themen extrahierbar (LLM-Antwort leer oder nicht parsebar)"]
        logger.warning("[COMPACTOR] Themen-Parsing leer — Fallback-Eintrag.")

    week_tag = _iso_week_tag(last7)
    md_path = os.path.join(output_dir, f"weekly_character_theme_{week_tag}.md")
    os.makedirs(output_dir, exist_ok=True)

    md_lines: List[str] = []
    md_lines.append(f"# Weekly Character Theme {week_tag}")
    md_lines.append("")
    md_lines.append(f"- Generiert: {_utc_iso()}")
    md_lines.append(f"- Verarbeitete Tage: {', '.join(f.replace('.jsonl', '') for f in last7)}")
    md_lines.append(f"- Events gesamt: {len(all_events)}")
    md_lines.append(f"- Event-IDs: {all_ids[0] if all_ids else 'n/a'} .. {all_ids[-1] if all_ids else 'n/a'}")
    md_lines.append("")
    md_lines.append("## Themen")
    md_lines.append("")
    for i, t in enumerate(themes, 1):
        md_lines.append(f"{i}. {t}")
    md_lines.append("")
    md_lines.append("## Roh-Antwort des LLM")
    md_lines.append("")
    md_lines.append("```")
    md_lines.append((llm_text or "").strip() or "(keine Antwort)")
    md_lines.append("```")
    md_lines.append("")

    # Atomar schreiben (tempfile + os.replace, NTFS-Fallback)
    tmp_path = md_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        f.flush()
        os.fsync(f.fileno())
    try:
        os.replace(tmp_path, md_path)
    except OSError:
        with open(tmp_path, "r", encoding="utf-8") as f_src:
            content = f_src.read()
        with open(md_path, "w", encoding="utf-8") as f_dst:
            f_dst.write(content)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    logger.info(f"[COMPACTOR] Wochen-Themen geschrieben: {md_path}")
    return md_path, last7


# --------------------------------------------------------------------------
# archive
# --------------------------------------------------------------------------
def archive(journal_dir: str, processed_files: List[str]) -> List[str]:
    """
    Verschiebt verarbeitete Tagesdateien (und scored_*-Pendants) nach archive/.

    Returns:
        Liste der tatsaechlich verschobenen Dateinamen.
    """
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    moved: List[str] = []

    for fname in processed_files:
        src = os.path.join(journal_dir, fname)
        if os.path.isfile(src):
            dst = os.path.join(ARCHIVE_DIR, fname)
            try:
                shutil.move(src, dst)
                moved.append(fname)
            except Exception as e:
                logger.warning(f"[COMPACTOR] {fname} nicht verschoben: {e}")

        scored = f"scored_{fname}"
        scored_src = os.path.join(journal_dir, scored)
        if os.path.isfile(scored_src):
            scored_dst = os.path.join(ARCHIVE_DIR, scored)
            try:
                shutil.move(scored_src, scored_dst)
                moved.append(scored)
            except Exception as e:
                logger.warning(f"[COMPACTOR] {scored} nicht verschoben: {e}")

    return moved


# --------------------------------------------------------------------------
# update_gate
# --------------------------------------------------------------------------
def _default_phase_gates() -> Dict:
    return {
        "phase4e": {
            "required_days": REQUIRED_DAYS,
            "days_collected": 0,
            "armed": False,
            "armed_at": None,
            "last_run": None,
        },
        "phase6": {
            "required_operation_days": 14,
            "operation_days": 0,
            "armed": False,
            "armed_at": None,
            "last_validation": None,
            "last_result": None,
        },
    }


def _load_phase_gates(path: str) -> Dict:
    if not os.path.isfile(path):
        return _default_phase_gates()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        defaults = _default_phase_gates()
        for k, v in defaults.items():
            if k not in data:
                data[k] = v
            else:
                for sub_k, sub_v in v.items():
                    data[k].setdefault(sub_k, sub_v)
        return data
    except Exception as e:
        logger.warning(f"[COMPACTOR] phase_gates.json lesen fehlgeschlagen ({e}) — Defaults.")
        return _default_phase_gates()


def update_gate(phase_gates_path: str, days_collected: int,
                ran_compaction: bool = False) -> Dict:
    """
    Aktualisiert phase_gates.json mit Phase-4e-Status.

    Args:
        phase_gates_path: Zielpfad
        days_collected: Anzahl unarchivierter Tagesdateien VOR Lauf
        ran_compaction: True wenn compact() ausgefuehrt wurde

    Returns:
        Das geschriebene Gate-Dict.
    """
    gates = _load_phase_gates(phase_gates_path)
    p4e = gates.setdefault("phase4e", _default_phase_gates()["phase4e"])
    p4e["required_days"] = REQUIRED_DAYS
    p4e["days_collected"] = int(days_collected)
    now_iso = _utc_iso()

    if days_collected >= REQUIRED_DAYS:
        if not p4e.get("armed"):
            p4e["armed"] = True
            p4e["armed_at"] = now_iso
        if ran_compaction:
            p4e["last_run"] = now_iso
    else:
        p4e.setdefault("armed", False)

    _safe_write_json(phase_gates_path, gates)
    return gates


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    ready, n = is_ready(JOURNAL_DIR)
    if not ready:
        print(f"Phase 4e: {n}/{REQUIRED_DAYS} Tage — noch nicht scharf")
        update_gate(PHASE_GATES_PATH, days_collected=n, ran_compaction=False)
        return 0

    try:
        md_path, processed = compact(JOURNAL_DIR, THEMES_DIR, default_llm_fn)
    except Exception as e:
        logger.error(f"[COMPACTOR] compact() fehlgeschlagen: {e}")
        update_gate(PHASE_GATES_PATH, days_collected=n, ran_compaction=False)
        return 1

    moved = archive(JOURNAL_DIR, processed)
    logger.info(f"[COMPACTOR] Archiviert: {len(moved)} Dateien")

    # Nach Archivierung: days_collected neu zaehlen (i.d.R. < REQUIRED_DAYS)
    _, n_after = is_ready(JOURNAL_DIR)
    update_gate(PHASE_GATES_PATH, days_collected=n_after, ran_compaction=True)

    print(f"Phase 4e: Kompaktierung abgeschlossen — {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
