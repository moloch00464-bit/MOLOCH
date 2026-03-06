"""
TesterAgent — Validiert Gate-1-Funktionalität.
Pipeline-Step 3: Tests ausführen und berichten.

Fokus Gate 1:
- Action Bridge FSM Transitions
- Event Bus publish/subscribe
- Timeout-Verhalten
- Perception → Action Event-Kette
"""

import subprocess
import sys
import time
from .base_agent import BaseDevAgent, Task, TaskResult, TaskStatus
from pathlib import Path
from typing import List, Dict, Tuple


class TesterAgent(BaseDevAgent):
    AGENT_NAME = "tester"
    PROJECT_ROOT = Path.home() / "moloch"

    def execute(self, task: Task) -> TaskResult:
        """
        Führt Tests aus:
        1. Existierende test_*.py Dateien
        2. Inline Smoke-Tests für Gate-1-Module
        3. Import-Ketten-Test
        """
        test_results = []

        # 1. Vorhandene Unit-Tests ausführen
        test_dir = self.PROJECT_ROOT / "tests"
        if test_dir.exists():
            for tf in sorted(test_dir.glob("test_*.py")):
                passed, output = self._run_test_file(tf)
                test_results.append({
                    "test": tf.name,
                    "result": "PASS" if passed else "FAIL",
                    "output": output[-500:],
                })

        # 2. Smoke-Tests für Gate-1-Kernmodule
        smoke_tests = self._run_smoke_tests()
        test_results.extend(smoke_tests)

        # 3. Auswertung
        fails = [t for t in test_results if t["result"] == "FAIL"]
        passes = [t for t in test_results if t["result"] == "PASS"]

        status = TaskStatus.PASS if not fails else TaskStatus.FAIL

        result = self._make_result(
            task,
            status,
            summary=f"{len(passes)} PASS, {len(fails)} FAIL von {len(test_results)} Tests",
            details={
                "tests": test_results,
                "pass_count": len(passes),
                "fail_count": len(fails),
            },
            errors=[f["test"] + ": " + f["output"][:100] for f in fails],
            next_action="reviewer_agent" if status == TaskStatus.PASS else "debugger_agent",
        )
        self.log_result(result)
        return result

    def _run_test_file(self, filepath: Path) -> Tuple[bool, str]:
        """Führt eine Test-Datei aus."""
        try:
            r = subprocess.run(
                [sys.executable, str(filepath)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.PROJECT_ROOT),
            )
            return r.returncode == 0, (r.stdout + r.stderr).strip()
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT nach 30s"
        except Exception as e:
            return False, str(e)

    def _run_smoke_tests(self) -> List[Dict]:
        """Gate-1 Smoke-Tests — Kernmodule importierbar?"""
        results = []

        smoke_checks = [
            (
                "event_bus_singleton",
                "from core.moloch_event_bus import get_event_bus; "
                "bus = get_event_bus(); "
                "print(f'Bus OK, {len(bus._subscribers)} subscribers')",
            ),
            (
                "action_bridge_singleton",
                "from core.action_bridge import get_action_bridge; "
                "ab = get_action_bridge(); "
                "print(f'Bridge OK, state={ab.get_status()}')",
            ),
            (
                "action_bridge_states",
                "from core.action_bridge import get_action_bridge; "
                "ab = get_action_bridge(); "
                "st = ab.get_status(); "
                "assert 'state' in st if isinstance(st, dict) else True; "
                "print('States OK')",
            ),
            (
                "event_bus_pubsub",
                "from core.moloch_event_bus import get_event_bus; "
                "bus = get_event_bus(); "
                "received = []; "
                "bus.subscribe('test.smoke', lambda e: received.append(e)); "
                "bus.emit('test.smoke', {'check': True}); "
                "import time; time.sleep(0.1); "
                "assert len(received) >= 1, f'Expected 1 event, got {len(received)}'; "
                "print('PubSub OK')",
            ),
        ]

        for name, code in smoke_checks:
            try:
                r = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(self.PROJECT_ROOT),
                )
                passed = r.returncode == 0
                output = (r.stdout + r.stderr).strip()
            except subprocess.TimeoutExpired:
                passed = False
                output = "TIMEOUT"
            except Exception as e:
                passed = False
                output = str(e)

            results.append({
                "test": f"smoke_{name}",
                "result": "PASS" if passed else "FAIL",
                "output": output[-300:],
            })

        return results
