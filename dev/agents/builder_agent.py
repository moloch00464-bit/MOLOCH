"""
BuilderAgent — Implementiert Code nach Architektur-Vorgabe.
Pipeline-Step 2: Prüft ob Architect-Spec umgesetzt wurde.

In der Praxis: Claude Code schreibt den Code, BuilderAgent verifiziert
dass die richtigen Dateien existieren und syntaktisch korrekt sind.
"""

import subprocess
import sys
from .base_agent import BaseDevAgent, Task, TaskResult, TaskStatus
from pathlib import Path
from typing import Dict, Any


class BuilderAgent(BaseDevAgent):
    AGENT_NAME = "builder"
    PROJECT_ROOT = Path.home() / "moloch"

    def execute(self, task: Task, architect_spec: Dict[str, Any] = None) -> TaskResult:
        """
        Verifiziert Implementation:
        - Existieren alle geforderten Dateien?
        - Sind sie syntaktisch korrekt (py_compile)?
        - Haben sie sich seit Architect-Analyse verändert?
        """
        issues = []
        verified = []

        all_files = task.target_files
        if architect_spec:
            all_files = list(set(
                all_files
                + architect_spec.get("files_to_modify", [])
                + architect_spec.get("files_to_create", [])
            ))

        for f in all_files:
            fp = self.PROJECT_ROOT / f
            if not fp.exists():
                issues.append(f"MISSING: {f} existiert nicht")
                continue

            # Syntax-Check für Python-Dateien
            if fp.suffix == ".py":
                ok, err = self._syntax_check(fp)
                if ok:
                    verified.append(f)
                else:
                    issues.append(f"SYNTAX_ERROR in {f}: {err}")
            else:
                verified.append(f)

        # Import-Check: kann das Modul importiert werden?
        import_errors = []
        for f in verified:
            if f.endswith(".py") and f.startswith("core/"):
                module = f.replace("/", ".").replace(".py", "")
                ok, err = self._import_check(module)
                if not ok:
                    import_errors.append(f"IMPORT_ERROR {module}: {err}")

        all_issues = issues + import_errors
        status = TaskStatus.PASS if not all_issues else TaskStatus.FAIL

        result = self._make_result(
            task,
            status,
            summary=f"{len(verified)} files OK, {len(all_issues)} issues",
            details={
                "verified": verified,
                "issues": all_issues,
                "import_errors": import_errors,
            },
            errors=all_issues,
            next_action="tester_agent" if status == TaskStatus.PASS else "debugger_agent",
        )
        self.log_result(result)
        return result

    def _syntax_check(self, filepath: Path) -> tuple:
        """py_compile Syntax-Check ohne Import."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(filepath)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return True, ""
            return False, result.stderr.strip()[-200:]
        except Exception as e:
            return False, str(e)

    def _import_check(self, module: str) -> tuple:
        """Versucht Import — fängt fehlende Dependencies."""
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import {module}"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.PROJECT_ROOT),
            )
            if result.returncode == 0:
                return True, ""
            return False, result.stderr.strip()[-200:]
        except Exception as e:
            return False, str(e)
