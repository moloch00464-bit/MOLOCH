"""
ArchitectAgent — Übersetzt Feature-Ideen in konkrete Code-Spezifikationen.
Pipeline-Step 1: Bevor Code geschrieben wird, muss klar sein WAS und WO.
"""

from .base_agent import BaseDevAgent, Task, TaskResult, TaskStatus
from pathlib import Path
from typing import List, Dict


class ArchitectAgent(BaseDevAgent):
    AGENT_NAME = "architect"
    PROJECT_ROOT = Path.home() / "moloch"

    def execute(self, task: Task) -> TaskResult:
        """
        Analysiert Task und erstellt Code-Spezifikation:
        - Welche Dateien müssen geändert werden?
        - Welche neuen Module braucht es?
        - Welche Tests sind nötig?
        - Welche Abhängigkeiten bestehen?
        """
        spec = {
            "feature": task.feature,
            "gate": task.gate,
            "analysis": {},
        }

        # 1. Prüfe ob target_files existieren
        existing = []
        missing = []
        for f in task.target_files:
            fp = self.PROJECT_ROOT / f
            if fp.exists():
                existing.append(f)
                spec["analysis"][f] = {
                    "exists": True,
                    "size_bytes": fp.stat().st_size,
                    "lines": sum(1 for _ in open(fp, errors="ignore")),
                }
            else:
                missing.append(f)
                spec["analysis"][f] = {"exists": False, "action": "CREATE"}

        # 2. Prüfe ob Tests existieren
        test_dir = self.PROJECT_ROOT / "tests"
        has_tests = test_dir.exists() and any(test_dir.glob("test_*.py"))

        # 3. Prüfe CLAUDE.md
        claude_md = self.PROJECT_ROOT / "CLAUDE.md"
        has_claude_md = claude_md.exists()

        spec["files_to_modify"] = existing
        spec["files_to_create"] = missing
        spec["test_infrastructure"] = has_tests
        spec["claude_md_present"] = has_claude_md
        spec["dependencies"] = self._find_imports(existing)

        # 4. Generiere Claude Code Auftrag (max 3 Sätze!)
        spec["claude_code_task"] = self._generate_task_brief(task, existing, missing)

        result = self._make_result(
            task,
            TaskStatus.PASS,
            summary=f"Architektur-Spec erstellt: {len(existing)} modify, {len(missing)} create",
            details=spec,
            artifacts=[str(self.LOG_DIR / f"{task.task_id}_{self.AGENT_NAME}.json")],
            next_action="builder_agent",
        )
        self.log_result(result)
        return result

    def _find_imports(self, files: List[str]) -> Dict[str, List[str]]:
        """Findet Imports in bestehenden Dateien — zeigt Abhängigkeiten."""
        deps = {}
        for f in files:
            fp = self.PROJECT_ROOT / f
            if not fp.exists():
                continue
            imports = []
            try:
                for line in open(fp, errors="ignore"):
                    line = line.strip()
                    if line.startswith("from core.") or line.startswith("import core."):
                        imports.append(line)
                    elif line.startswith("from .") or line.startswith("import ."):
                        imports.append(line)
            except Exception:
                pass
            if imports:
                deps[f] = imports
        return deps

    def _generate_task_brief(self, task: Task, existing, missing) -> str:
        """Erzeugt einen Claude-Code-konformen Kurzauftrag."""
        parts = ["Lies CLAUDE.md."]
        if existing:
            parts.append(f"In {', '.join(existing)}: {task.description}")
        if missing:
            parts.append(f"Erstelle {', '.join(missing)} für {task.feature}.")
        return " ".join(parts)
