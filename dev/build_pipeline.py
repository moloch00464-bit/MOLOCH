#!/usr/bin/env python3
"""
M.O.L.O.C.H. Build Pipeline — Orchestriert das Agenten-Team.
Temporär für Gate 1–3, wird danach entfernt.

Usage:
    python3 dev/build_pipeline.py --feature "Action Bridge FSM" \
        --files core/action_bridge.py core/moloch_event_bus.py \
        --gate gate_1 \
        --description "MANUAL State + owner_lost Transition hinzufügen"

    python3 dev/build_pipeline.py --smoke  # Nur Smoke-Tests
    python3 dev/build_pipeline.py --review --files core/action_bridge.py  # Nur Review
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Dev-Agents Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent.parent))

from dev.agents.base_agent import Task, TaskStatus
from dev.agents.architect_agent import ArchitectAgent
from dev.agents.builder_agent import BuilderAgent
from dev.agents.tester_agent import TesterAgent
from dev.agents.debugger_agent import DebuggerAgent
from dev.agents.reviewer_agent import ReviewerAgent


class BuildPipeline:
    """
    5-Step Build Pipeline:
    Architect → Builder → Tester → Debugger (bei FAIL) → Reviewer
    """

    def __init__(self):
        self.architect = ArchitectAgent()
        self.builder = BuilderAgent()
        self.tester = TesterAgent()
        self.debugger = DebuggerAgent()
        self.reviewer = ReviewerAgent()
        self.results = []
        self.log_dir = Path.home() / "moloch" / "logs" / "dev_agents"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run_full(self, task: Task) -> dict:
        """Komplette Pipeline durchlaufen."""
        print(f"\n{'='*60}")
        print(f"  M.O.L.O.C.H. BUILD PIPELINE")
        print(f"  Feature: {task.feature}")
        print(f"  Gate: {task.gate}")
        print(f"{'='*60}\n")

        pipeline_start = time.time()
        pipeline_result = {
            "task": task.to_dict(),
            "steps": [],
            "final_verdict": "UNKNOWN",
        }

        # Step 1: Architect
        print("[1/5] ARCHITECT — Analysiere Spezifikation...")
        arch_result = self.architect.execute(task)
        pipeline_result["steps"].append(arch_result.to_dict())
        self._print_step_result("ARCHITECT", arch_result)

        if arch_result.status == TaskStatus.FAIL:
            pipeline_result["final_verdict"] = "BLOCKED_AT_ARCHITECTURE"
            return self._finalize(pipeline_result, pipeline_start)

        # Step 2: Builder (Verification)
        print("[2/5] BUILDER — Verifiziere Implementation...")
        build_result = self.builder.execute(task, arch_result.details)
        pipeline_result["steps"].append(build_result.to_dict())
        self._print_step_result("BUILDER", build_result)

        if build_result.status == TaskStatus.FAIL:
            # Direkt zum Debugger
            print("[  !] Builder FAIL → Debugger wird aktiviert...")
            debug_result = self.debugger.execute(task, build_result.details)
            pipeline_result["steps"].append(debug_result.to_dict())
            self._print_step_result("DEBUGGER", debug_result)
            pipeline_result["final_verdict"] = "BLOCKED_AT_BUILD"
            return self._finalize(pipeline_result, pipeline_start)

        # Step 3: Tester
        print("[3/5] TESTER — Führe Tests aus...")
        test_result = self.tester.execute(task)
        pipeline_result["steps"].append(test_result.to_dict())
        self._print_step_result("TESTER", test_result)

        if test_result.status == TaskStatus.FAIL:
            # Step 4: Debugger (nur bei FAIL)
            print("[4/5] DEBUGGER — Analysiere Fehler...")
            debug_result = self.debugger.execute(task, test_result.details)
            pipeline_result["steps"].append(debug_result.to_dict())
            self._print_step_result("DEBUGGER", debug_result)
            pipeline_result["final_verdict"] = "BLOCKED_AT_TEST"
            return self._finalize(pipeline_result, pipeline_start)
        else:
            print("[4/5] DEBUGGER — Übersprungen (keine Fehler)")

        # Step 5: Reviewer
        print("[5/5] REVIEWER — Qualitätskontrolle...")
        review_result = self.reviewer.execute(task)
        pipeline_result["steps"].append(review_result.to_dict())
        self._print_step_result("REVIEWER", review_result)

        pipeline_result["final_verdict"] = review_result.details.get("verdict", "UNKNOWN")
        return self._finalize(pipeline_result, pipeline_start)

    def run_smoke_only(self, task: Task) -> dict:
        """Nur Smoke-Tests, keine vollständige Pipeline."""
        print("\n  SMOKE TEST MODE\n")
        result = self.tester.execute(task)
        self._print_step_result("SMOKE", result)
        return result.to_dict()

    def run_review_only(self, task: Task) -> dict:
        """Nur Code-Review, keine Tests."""
        print("\n  REVIEW MODE\n")
        result = self.reviewer.execute(task)
        self._print_step_result("REVIEW", result)
        return result.to_dict()

    def _print_step_result(self, step: str, result):
        """Kompakte Ausgabe pro Pipeline-Step."""
        icon = "✅" if result.status == TaskStatus.PASS else "❌"
        print(f"  {icon} {step}: {result.status.value} — {result.summary}")
        if result.errors:
            for e in result.errors[:3]:
                print(f"     ⚠ {e[:120]}")
        if result.next_action:
            print(f"     → next: {result.next_action}")
        print()

    def _finalize(self, pipeline_result: dict, start_time: float) -> dict:
        """Pipeline-Ergebnis abschließen und speichern."""
        duration = time.time() - start_time
        pipeline_result["duration_seconds"] = round(duration, 1)
        pipeline_result["timestamp"] = time.time()

        # Zusammenfassung
        print(f"{'='*60}")
        print(f"  ERGEBNIS: {pipeline_result['final_verdict']}")
        print(f"  Dauer: {duration:.1f}s")
        print(f"{'='*60}\n")

        # Speichern
        task_id = pipeline_result["task"]["task_id"]
        outfile = self.log_dir / f"pipeline_{task_id}.json"
        outfile.write_text(
            json.dumps(pipeline_result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Log: {outfile}\n")

        return pipeline_result


def main():
    parser = argparse.ArgumentParser(
        description="M.O.L.O.C.H. Build Pipeline — Agenten-Team Orchestrator"
    )
    parser.add_argument("--feature", type=str, default="Gate 1 Check",
                        help="Feature-Name")
    parser.add_argument("--description", type=str, default="",
                        help="Feature-Beschreibung")
    parser.add_argument("--files", nargs="*", default=[],
                        help="Ziel-Dateien (relativ zu ~/moloch/)")
    parser.add_argument("--gate", type=str, default="gate_1",
                        help="Aktuelles Gate (default: gate_1)")
    parser.add_argument("--task-id", type=str, default=None,
                        help="Task-ID (default: auto)")
    parser.add_argument("--smoke", action="store_true",
                        help="Nur Smoke-Tests")
    parser.add_argument("--review", action="store_true",
                        help="Nur Code-Review")

    args = parser.parse_args()

    task_id = args.task_id or f"task_{int(time.time())}"

    task = Task(
        task_id=task_id,
        feature=args.feature,
        description=args.description or args.feature,
        gate=args.gate,
        target_files=args.files,
    )

    pipeline = BuildPipeline()

    if args.smoke:
        result = pipeline.run_smoke_only(task)
    elif args.review:
        result = pipeline.run_review_only(task)
    else:
        result = pipeline.run_full(task)

    # Exit-Code basierend auf Ergebnis
    verdict = result.get("final_verdict", result.get("status", "UNKNOWN"))
    if verdict in ("APPROVED", "APPROVED_WITH_WARNINGS", "PASS"):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
