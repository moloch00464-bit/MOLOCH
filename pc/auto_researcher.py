"""MOLOCH Auto-Researcher (PC-Side, Phase 2 Synthese-Plan).

3-Stufen-Hybrid (DeepSeek + Gemini):
  Stufe 1 (Default, immer aktiv):
    - Hailo SDK Releases polling (GitHub API)
    - Ollama Model-Registry check
    - LoRA-Checkpoint-Vergleich (lokal)
    - Output: staging/research_proposals/YYYY-MM-DD.md (1 File pro Tag)

  Stufe 2 (Markus aktiviert via Cockpit-Toggle "Auto-Deploy fuer X Tage"):
    - State persistent in %LOCALAPPDATA%/moloch_pc_state/auto_researcher.json
    - Wenn auto_deploy_enabled_until_ts > now -> deployed in staging/auto_deploy/
    - Opus-Veto via pr-review-toolkit:code-reviewer Subagent
    - Bei PASS: Apply auf Production
    - Bei FAIL: Logs, Markus-Notification

  Stufe 3 (Markus-Stop):
    - Loescht auto_deploy_enabled_until_ts
    - Zurueck auf Stufe 1

Aktuell implementiert: NUR Stufe 1.
Stufe 2 + 3: TODO (kommt mit Cockpit-Sub-Tab "Forschung" + State-Toggle).

CLI:
  python -m pc.auto_researcher              # einmalig laufen, Proposal schreiben
  python -m pc.auto_researcher --json       # nur JSON-Output
  python -m pc.auto_researcher --force      # auch wenn schon ein Proposal heute existiert

NEVER 5: requests timeout. NEVER 6: atomic write. NEVER 8: kein shell.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("auto-researcher")

# Repo-Pfad (env-overridable)
REPO_ROOT = Path(os.environ.get("MOLOCH_REPO", r"C:\Users\49179\moloch_repo"))
STAGING_DIR = REPO_ROOT / "staging" / "research_proposals"
STAGING_DIR.mkdir(parents=True, exist_ok=True)

# State-File (Stufe 2/3 toggle)
_LOCAL_APPDATA = os.environ.get("LOCALAPPDATA")
if _LOCAL_APPDATA:
    _STATE_DIR = Path(_LOCAL_APPDATA) / "moloch_pc_state"
else:
    _STATE_DIR = Path.home() / "moloch_pc_state"
_STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = _STATE_DIR / "auto_researcher.json"

# Quellen
HAILO_GITHUB_API = "https://api.github.com/repos/hailo-ai/hailo_ai_sw_suite/releases"
OLLAMA_LOCAL_API = os.environ.get("MOLOCH_OLLAMA_URL", "http://localhost:11434/api/tags")
LORA_CHECKPOINT_DIR = REPO_ROOT / "lora_checkpoints"

HTTP_TIMEOUT = 10
USER_AGENT = "MOLOCH-AutoResearcher/1.0"


@dataclass
class Finding:
    """Ein einzelner Forschungs-Befund."""
    source: str  # "hailo" | "ollama" | "lora"
    title: str
    description: str
    severity: str  # "info" | "low" | "medium" | "high" | "critical"
    suggested_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Proposal:
    """Tagesvorschlag — alle Findings aggregiert."""
    date: str
    generated_at: str
    findings: List[Finding] = field(default_factory=list)
    sources_checked: List[str] = field(default_factory=list)
    sources_failed: List[str] = field(default_factory=list)


def check_hailo_releases() -> List[Finding]:
    """Pollt GitHub-API fuer Hailo SDK Releases.

    Returnt Findings wenn neueres Release als bekannte Version verfuegbar.
    NEVER 5: timeout=10s.
    """
    findings: List[Finding] = []
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
            r = client.get(HAILO_GITHUB_API)
            if r.status_code != 200:
                logger.warning(f"hailo github api: HTTP {r.status_code}")
                return findings
            releases = r.json()
            if not isinstance(releases, list) or not releases:
                return findings
            latest = releases[0]
            tag = latest.get("tag_name", "?")
            published = latest.get("published_at", "?")
            url = latest.get("html_url", "")
            findings.append(
                Finding(
                    source="hailo",
                    title=f"Hailo SDK Release {tag}",
                    description=f"Latest GitHub release: {tag} (published {published})",
                    severity="info",
                    suggested_action=f"Markus pruefen ob Update sinnvoll. Release-Notes: {url}",
                    metadata={"tag_name": tag, "published_at": published, "url": url},
                )
            )
    except Exception as e:
        logger.info(f"hailo check failed: {e}")
    return findings


def check_ollama_models() -> List[Finding]:
    """Pollt lokale Ollama-Instanz fuer installierte Modelle.

    Returnt Findings ueber alte/grosse Modelle die ggf. updatable sind.
    """
    findings: List[Finding] = []
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            r = client.get(OLLAMA_LOCAL_API)
            if r.status_code != 200:
                logger.info(f"ollama not running: HTTP {r.status_code}")
                return findings
            data = r.json()
            models = data.get("models", [])
            for m in models:
                name = m.get("name", "?")
                size_mb = m.get("size", 0) / (1024 * 1024)
                modified = m.get("modified_at", "?")
                # Heuristik: Modelle aelter als 90 Tage als update-Kandidaten
                findings.append(
                    Finding(
                        source="ollama",
                        title=f"Ollama-Modell {name}",
                        description=f"Size {size_mb:.0f} MB, last modified {modified}",
                        severity="info",
                        suggested_action=f"Pruefen ob neuere Version: ollama pull {name}",
                        metadata={"name": name, "size_mb": size_mb, "modified_at": modified},
                    )
                )
    except Exception as e:
        logger.info(f"ollama check failed: {e}")
    return findings


def check_lora_checkpoints() -> List[Finding]:
    """Vergleicht LoRA-Checkpoints lokal — neueste Adapter-Welle?"""
    findings: List[Finding] = []
    try:
        if not LORA_CHECKPOINT_DIR.exists():
            return findings
        checkpoints = sorted(LORA_CHECKPOINT_DIR.glob("*.safetensors"))
        if not checkpoints:
            return findings
        latest = checkpoints[-1]
        size_mb = latest.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
        findings.append(
            Finding(
                source="lora",
                title=f"Latest LoRA-Checkpoint {latest.name}",
                description=f"Size {size_mb:.0f} MB, mtime {mtime}, total {len(checkpoints)} checkpoints",
                severity="info",
                suggested_action="Pruefen ob aktuell aktiv via curl http://localhost:11600/health",
                metadata={"name": latest.name, "size_mb": size_mb, "mtime": mtime, "total_checkpoints": len(checkpoints)},
            )
        )
    except Exception as e:
        logger.info(f"lora check failed: {e}")
    return findings


def generate_proposal(force: bool = False) -> Optional[Proposal]:
    """Generiert tagesaktuellen Proposal.

    Returnt None wenn schon einer fuer heute existiert (es sei denn force=True).
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = STAGING_DIR / f"{today}.md"
    if path.exists() and not force:
        logger.info(f"Proposal fuer {today} existiert bereits: {path}")
        return None

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    proposal = Proposal(date=today, generated_at=now_iso)

    sources = [
        ("hailo", check_hailo_releases),
        ("ollama", check_ollama_models),
        ("lora", check_lora_checkpoints),
    ]
    for src_name, fn in sources:
        try:
            findings = fn()
            proposal.findings.extend(findings)
            proposal.sources_checked.append(src_name)
        except Exception as e:
            logger.warning(f"source {src_name} failed: {e}")
            proposal.sources_failed.append(src_name)

    return proposal


def render_markdown(proposal: Proposal) -> str:
    """Markdown-Render fuer staging/research_proposals/YYYY-MM-DD.md."""
    lines = []
    lines.append(f"# Auto-Researcher Proposal {proposal.date}")
    lines.append("")
    lines.append(f"Generated: {proposal.generated_at}")
    lines.append("")
    lines.append(f"Sources checked: {', '.join(proposal.sources_checked)}")
    if proposal.sources_failed:
        lines.append(f"Sources failed: {', '.join(proposal.sources_failed)}")
    lines.append("")
    lines.append(f"Total findings: {len(proposal.findings)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    by_source: Dict[str, List[Finding]] = {}
    for f in proposal.findings:
        by_source.setdefault(f.source, []).append(f)

    for source, findings in by_source.items():
        lines.append(f"## {source.upper()} ({len(findings)} findings)")
        lines.append("")
        for f in findings:
            lines.append(f"### {f.title}")
            lines.append(f"- **Severity:** {f.severity}")
            lines.append(f"- **Description:** {f.description}")
            lines.append(f"- **Action:** {f.suggested_action}")
            if f.metadata:
                lines.append(f"- **Metadata:** `{json.dumps(f.metadata, ensure_ascii=False)}`")
            lines.append("")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("**Markus-Action:** Approve / Reject jedes Finding via Cockpit-Sub-Tab 'Forschung' (Phase 2 TODO).")
    lines.append("Aktuell: nur Stufe 1 (passiv). Auto-Deploy (Stufe 2) braucht Markus-Toggle im Cockpit.")
    return "\n".join(lines) + "\n"


def write_proposal(proposal: Proposal) -> Path:
    """Atomic write des Proposals nach staging/research_proposals/."""
    today = proposal.date
    path = STAGING_DIR / f"{today}.md"

    md = render_markdown(proposal)
    fd, tmp = tempfile.mkstemp(
        dir=str(STAGING_DIR),
        prefix=path.name + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(md)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info(f"Proposal geschrieben: {path}")
    return path


def get_state() -> Dict[str, Any]:
    """Liest auto_researcher.json (Stufe 2/3-Toggle-State)."""
    if not STATE_FILE.exists():
        return {
            "version": 1,
            "auto_deploy_enabled_until_ts": 0.0,
            "last_proposal_date": None,
            "last_proposal_path": None,
            "stage": 1,
        }
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "auto_deploy_enabled_until_ts": 0.0, "stage": 1}


def update_state(updates: Dict[str, Any]) -> None:
    """Atomic write der state-File."""
    state = get_state()
    state.update(updates)
    fd, tmp = tempfile.mkstemp(dir=str(_STATE_DIR), prefix=STATE_FILE.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(tmp, str(STATE_FILE))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def current_stage() -> int:
    """Returnt 1 (passiv) oder 2 (auto-deploy aktiv) basierend auf Toggle + Cutoff."""
    state = get_state()
    until = state.get("auto_deploy_enabled_until_ts", 0.0)
    if until and time.time() < float(until):
        return 2
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="MOLOCH Auto-Researcher Stufe 1")
    parser.add_argument("--json", action="store_true", help="JSON-Output statt Markdown-Write")
    parser.add_argument("--force", action="store_true", help="auch wenn heutiger Proposal existiert")
    parser.add_argument("--stage", action="store_true", help="aktuelle Stufe ausgeben")
    args = parser.parse_args()

    if args.stage:
        print(f"Current stage: {current_stage()}")
        print(f"State file: {STATE_FILE}")
        print(json.dumps(get_state(), indent=2, ensure_ascii=False))
        return 0

    proposal = generate_proposal(force=args.force)
    if proposal is None:
        logger.info("Kein neuer Proposal — heutiger existiert (oder --force fuer Override)")
        return 0

    if args.json:
        out = {
            "date": proposal.date,
            "generated_at": proposal.generated_at,
            "sources_checked": proposal.sources_checked,
            "sources_failed": proposal.sources_failed,
            "findings": [asdict(f) for f in proposal.findings],
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        path = write_proposal(proposal)
        update_state({
            "last_proposal_date": proposal.date,
            "last_proposal_path": str(path),
            "last_run_ts": time.time(),
        })
        print(f"OK: {path} ({len(proposal.findings)} findings, sources: {','.join(proposal.sources_checked)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
