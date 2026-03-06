"""
ReviewerAgent — Qualitätskontrolle und Gate-Konformität.
Pipeline-Step 5: Letzter Check bevor Markus freigibt.

Prüft:
- Gate-Konformität (keine Gate-2-Features in Gate-1-Code)
- Code-Qualität (Dateigröße, Komplexität)
- Architektur-Regeln (SoC, Fail Isolation, IPC only)
- Performance-Indikatoren
"""

from .base_agent import BaseDevAgent, Task, TaskResult, TaskStatus
from pathlib import Path
from typing import List, Dict


class ReviewerAgent(BaseDevAgent):
    AGENT_NAME = "reviewer"
    PROJECT_ROOT = Path.home() / "moloch"

    # Architektur-Regeln aus Engineering-Regeln
    FORBIDDEN_PATTERNS = {
        "direkter_hardware_call": [
            "import RPi.GPIO",
            "GPIO.setup",
            "GPIO.output",
        ],
        "direkte_kopplung": [
            "from core.hardware.camera import",  # Nur über ServiceProxy/IPC!
            "from core.hardware.autonomous_tracker import",
        ],
        "cloud_dependency": [
            "requests.get('https://api.openai",
            "requests.post('https://api.anthropic",
        ],
    }

    # Maximale Dateigröße (Regel: Pi5 hat 4GB RAM)
    MAX_FILE_LINES = 1500  # moloch_service.py ist schon 1400+ und "Monolith"

    def execute(self, task: Task) -> TaskResult:
        """
        Review-Checkliste:
        1. Gate-Konformität
        2. Architektur-Regeln
        3. Dateigröße / Komplexität
        4. Freigabe-Empfehlung
        """
        findings = []
        warnings = []
        approvals = []

        for f in task.target_files:
            fp = self.PROJECT_ROOT / f
            if not fp.exists():
                findings.append(f"MISSING: {f}")
                continue

            if fp.suffix != ".py":
                approvals.append(f"{f}: kein Python, übersprungen")
                continue

            content = fp.read_text(errors="ignore")
            lines = content.split("\n")

            # 1. Dateigröße prüfen
            if len(lines) > self.MAX_FILE_LINES:
                warnings.append(
                    f"{f}: {len(lines)} Zeilen — über {self.MAX_FILE_LINES} Limit. "
                    f"Refactoring empfohlen (SoC-Regel)."
                )

            # 2. Forbidden Patterns prüfen
            for category, patterns in self.FORBIDDEN_PATTERNS.items():
                for pattern in patterns:
                    if pattern in content:
                        findings.append(
                            f"ARCHITEKTUR-VERLETZUNG in {f}: "
                            f"{category} — Pattern '{pattern}' gefunden"
                        )

            # 3. Gate-Konformität: keine Gate-2+ Features
            gate_violations = self._check_gate_conformity(f, content, task.gate)
            findings.extend(gate_violations)

            # 4. Basis-Qualität
            quality = self._check_quality(f, content, lines)
            warnings.extend(quality["warnings"])
            approvals.extend(quality["approvals"])

        # Ergebnis
        has_blockers = bool(findings)
        status = TaskStatus.FAIL if has_blockers else TaskStatus.PASS

        verdict = "BLOCKED" if has_blockers else ("APPROVED_WITH_WARNINGS" if warnings else "APPROVED")

        result = self._make_result(
            task,
            status,
            summary=f"Review: {verdict} — {len(findings)} Blocker, {len(warnings)} Warnings",
            details={
                "verdict": verdict,
                "blockers": findings,
                "warnings": warnings,
                "approvals": approvals,
                "recommendation": self._recommendation(findings, warnings),
            },
            errors=findings,
            next_action="operator_approval" if not has_blockers else "debugger_agent",
        )
        self.log_result(result)
        return result

    def _check_gate_conformity(self, filename: str, content: str, current_gate: str) -> List[str]:
        """Prüft ob Code nur Features des aktuellen Gates enthält."""
        violations = []

        # Gate 2+ Keywords die in Gate 1 nichts zu suchen haben
        if current_gate in ("gate_1", "gate_0"):
            gate2_patterns = [
                ("qdrant", "Gate 2: Qdrant Memory"),
                ("episodic_memory", "Gate 2: Episodisches Gedächtnis"),
                ("person_reid", "Gate 2: Person ReID"),
            ]
            for pattern, description in gate2_patterns:
                if pattern in content.lower():
                    violations.append(
                        f"GATE-VERLETZUNG in {filename}: "
                        f"'{pattern}' ist {description}, gehört nicht in {current_gate}"
                    )
        return violations

    def _check_quality(self, filename: str, content: str, lines: List[str]) -> Dict:
        """Basis-Qualitätsprüfung."""
        warnings = []
        approvals = []

        # Docstring vorhanden?
        if '"""' in content or "'''" in content:
            approvals.append(f"{filename}: Docstrings vorhanden")
        else:
            warnings.append(f"{filename}: Keine Docstrings — Dokumentation fehlt")

        # try/except vorhanden? (Fail Isolation Regel)
        if "try:" in content and "except" in content:
            approvals.append(f"{filename}: Exception Handling vorhanden")
        elif len(lines) > 50:
            warnings.append(f"{filename}: Kein Exception Handling — Fail Isolation?")

        # Logging vorhanden?
        if "logging" in content or "logger" in content:
            approvals.append(f"{filename}: Logging integriert")
        elif len(lines) > 100:
            warnings.append(f"{filename}: Kein Logging — Debug wird schwierig")

        return {"warnings": warnings, "approvals": approvals}

    def _recommendation(self, findings: List, warnings: List) -> str:
        """Generiert Freigabe-Empfehlung."""
        if findings:
            return (
                f"NICHT FREIGEBEN. {len(findings)} Architektur-Verletzungen "
                f"müssen zuerst behoben werden."
            )
        if warnings:
            return (
                f"BEDINGT FREIGEBEN. {len(warnings)} Warnings sollten "
                f"in einem Folge-Commit adressiert werden."
            )
        return "FREIGEBEN. Code ist sauber und Gate-konform."
