"""
DebuggerAgent — Analysiert Fehler und schlägt Fixes vor.
Pipeline-Step 4: Wird nur aktiviert wenn TesterAgent FAIL meldet.

Liest Logs, Decision-Ringbuffer, analysiert Tracebacks.
"""

import re
from .base_agent import BaseDevAgent, Task, TaskResult, TaskStatus
from pathlib import Path
from typing import List, Dict, Any


class DebuggerAgent(BaseDevAgent):
    AGENT_NAME = "debugger"
    PROJECT_ROOT = Path.home() / "moloch"
    LOG_DIRS = [
        Path.home() / "moloch" / "logs",
        Path.home() / "moloch" / "logs" / "events",
        Path.home() / "moloch" / "logs" / "dev_agents",
    ]

    def execute(self, task: Task, test_result: Dict[str, Any] = None) -> TaskResult:
        """
        Analysiert Fehler:
        1. Parsed Test-Errors für Tracebacks
        2. Scannt relevante Logfiles
        3. Prüft Decision-Ringbuffer der Action Bridge
        4. Generiert Root-Cause-Analyse + Fix-Vorschlag
        """
        analysis = {
            "tracebacks": [],
            "log_errors": [],
            "root_causes": [],
            "suggested_fixes": [],
        }

        # 1. Test-Errors parsen
        if test_result and "tests" in test_result:
            for t in test_result["tests"]:
                if t.get("result") == "FAIL":
                    tb = self._extract_traceback(t.get("output", ""))
                    if tb:
                        analysis["tracebacks"].append({
                            "test": t["test"],
                            "traceback": tb,
                            "error_type": self._classify_error(tb),
                        })

        # 2. Logfiles nach Errors scannen
        for log_dir in self.LOG_DIRS:
            if log_dir.exists():
                for logfile in sorted(log_dir.glob("*.log"))[-5:]:
                    errors = self._scan_logfile(logfile)
                    if errors:
                        analysis["log_errors"].extend(errors)

        # 3. Root-Cause bestimmen
        analysis["root_causes"] = self._determine_root_causes(analysis)

        # 4. Fix-Vorschläge generieren
        analysis["suggested_fixes"] = self._suggest_fixes(analysis)

        has_findings = bool(analysis["tracebacks"] or analysis["log_errors"])
        status = TaskStatus.PASS if has_findings else TaskStatus.FAIL

        result = self._make_result(
            task,
            status,
            summary=f"{len(analysis['tracebacks'])} Tracebacks, "
                    f"{len(analysis['root_causes'])} Root Causes, "
                    f"{len(analysis['suggested_fixes'])} Fix-Vorschläge",
            details=analysis,
            next_action="builder_agent" if analysis["suggested_fixes"] else "reviewer_agent",
        )
        self.log_result(result)
        return result

    def _extract_traceback(self, output: str) -> str:
        """Extrahiert Python-Traceback aus Output."""
        lines = output.split("\n")
        tb_start = None
        for i, line in enumerate(lines):
            if "Traceback" in line:
                tb_start = i
        if tb_start is not None:
            return "\n".join(lines[tb_start:])
        # Kein Traceback, aber Error?
        for line in lines:
            if "Error:" in line or "Exception:" in line:
                return line.strip()
        return ""

    def _classify_error(self, traceback: str) -> str:
        """Klassifiziert Error-Typ."""
        if "ImportError" in traceback or "ModuleNotFoundError" in traceback:
            return "IMPORT_ERROR"
        if "AttributeError" in traceback:
            return "ATTRIBUTE_ERROR"
        if "TypeError" in traceback:
            return "TYPE_ERROR"
        if "KeyError" in traceback:
            return "KEY_ERROR"
        if "FileNotFoundError" in traceback:
            return "FILE_NOT_FOUND"
        if "TimeoutError" in traceback or "TIMEOUT" in traceback:
            return "TIMEOUT"
        if "AssertionError" in traceback or "assert" in traceback.lower():
            return "ASSERTION_FAILED"
        return "UNKNOWN"

    def _scan_logfile(self, filepath: Path, max_lines: int = 500) -> List[Dict]:
        """Scannt Logfile nach ERROR/CRITICAL Einträgen."""
        errors = []
        try:
            lines = filepath.read_text(errors="ignore").split("\n")[-max_lines:]
            for line in lines:
                if any(kw in line.upper() for kw in ["ERROR", "CRITICAL", "FATAL", "EXCEPTION"]):
                    errors.append({
                        "file": filepath.name,
                        "line": line.strip()[:200],
                    })
        except Exception:
            pass
        return errors[-10:]  # Max 10 pro File

    def _determine_root_causes(self, analysis: Dict) -> List[str]:
        """Leitet Root Causes aus Tracebacks ab."""
        causes = []
        for tb in analysis["tracebacks"]:
            etype = tb.get("error_type", "UNKNOWN")
            if etype == "IMPORT_ERROR":
                causes.append(f"Modul-Import fehlgeschlagen in {tb['test']} — Abhängigkeit fehlt oder Pfad falsch")
            elif etype == "ATTRIBUTE_ERROR":
                causes.append(f"Attribut nicht gefunden in {tb['test']} — API-Änderung oder falscher Zugriff")
            elif etype == "TIMEOUT":
                causes.append(f"Timeout in {tb['test']} — Event-Loop blockiert oder Deadlock")
            elif etype == "ASSERTION_FAILED":
                causes.append(f"Assertion fehlgeschlagen in {tb['test']} — Logik-Fehler oder falsche Erwartung")
            else:
                causes.append(f"{etype} in {tb['test']}")
        return causes

    def _suggest_fixes(self, analysis: Dict) -> List[Dict]:
        """Generiert Fix-Vorschläge basierend auf Root Causes."""
        fixes = []
        for tb in analysis["tracebacks"]:
            etype = tb.get("error_type", "UNKNOWN")
            traceback_text = tb.get("traceback", "")

            # Datei aus Traceback extrahieren
            file_match = re.search(r'File "([^"]+)"', traceback_text)
            target_file = file_match.group(1) if file_match else "unbekannt"

            fix = {
                "target_file": target_file,
                "error_type": etype,
                "action": "UNKNOWN",
            }

            if etype == "IMPORT_ERROR":
                fix["action"] = "Import-Pfad prüfen oder fehlende Dependency installieren"
            elif etype == "ATTRIBUTE_ERROR":
                fix["action"] = "API-Kompatibilität prüfen — hat sich die Schnittstelle geändert?"
            elif etype == "TIMEOUT":
                fix["action"] = "Blockierenden Call finden — async oder Thread-Isolation nötig?"
            elif etype == "ASSERTION_FAILED":
                fix["action"] = "Erwarteten vs. tatsächlichen Wert vergleichen"

            fixes.append(fix)
        return fixes
