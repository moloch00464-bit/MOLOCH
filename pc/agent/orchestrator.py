"""MOLOCH Orchestrator (Welle 21 Phase 2).

DeepSeek-LLM mit function-calling-Loop. User-Query rein -> Tool-Use-Iterations
-> finale Antwort. Tools werden via Pi-Bridge dispatched.

Aufruf:
  python -m pc.agent.orchestrator "wieviel P-Bands aufm WGT 2026?"
  python -m pc.agent.orchestrator --max-iter 3 "spiel was Hartes"

Konfig:
  --mock        Erzwingt MockBridge (Pi-unabhaengig)
  --max-iter N  Max Tool-Use-Iterations (default 5)
  --verbose     Tool-Calls + Token-Counts ausgeben

NEVER 5: requests-timeout. NEVER 7: kein Runtime-State committen.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

from pc.agent import deepseek_client
from pc.agent import local_llm_client
from pc.agent import token_budget
from pc.agent import tool_protocol
from pc.agent.pi_tool_bridge import MockBridge, ToolBridge, get_bridge
from pc.agent.safety import Context, Safety

logger = logging.getLogger("orchestrator")

DEFAULT_MAX_ITER = 5
SYSTEM_PROMPT = (
    "Du bist Moloch — anatomisches AI-System auf Markus' Pi+PC. "
    "Du hast Zugriff auf Tools fuer Web-Suche, URL-Fetch, Spotify, "
    "Mood/Zone und Hardware-Aktoren. Nutze Tools wenn noetig — recherchiere "
    "selbst statt zu raten. Antworte deutsch, knapp, direkt. Kein 'natuerlich gerne'. "
    "Wenn ein Tool fehlt, sag das klar."
)
# LLM-Backends mit gleicher Schnittstelle (complete / extract_message / extract_usage)
BACKENDS = {"deepseek": deepseek_client, "local": local_llm_client}
# Lokale Grenzen: wenige Iterationen, Wanduhr-Limit, Tool-Ergebnis gekuerzt
# (Prompt-Verarbeitung ~37 Tok/s auf dieser CPU — 1500 Zeichen kosten schon ~10 s)
LOCAL_MAX_ITER = 4
LOCAL_WALL_S = float(os.environ.get("MOLOCH_LOCAL_WALL_S", "120"))
LOCAL_TOOL_RESULT_CHARS = 1500
# Das Modell schreibt sonst Dialoge weiter oder zitiert seinen Prompt
LOCAL_STOP = ["\nMarkus:", "\nStil:", "\nMoloch:", "\nUser:"]


class Orchestrator:
    def __init__(
        self,
        bridge: Optional[ToolBridge] = None,
        max_iter: int = DEFAULT_MAX_ITER,
        system_prompt: str = SYSTEM_PROMPT,
        verbose: bool = False,
        backend: str = "deepseek",
        safety: Optional[Safety] = None,
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend muss eines von {sorted(BACKENDS)} sein")
        self.bridge = bridge or get_bridge()
        self.max_iter = max_iter
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.backend = backend
        self.llm = BACKENDS[backend]
        self.tools = self.bridge.get_catalog()
        if not self.tools:
            logger.warning("[orch] Tool-Catalog leer — LLM antwortet ohne Tools")
        # Lokal noch OHNE Tools: der ganze Katalog kostet ~2000 Prompt-Token (~50 s auf
        # der CPU) und 7B-Modelle rufen Tools unzuverlaessig. Kommt mit dem Tool-Protokoll
        # (Regel-Router + Schema + Validator), siehe PC_BRAIN_ARCHITEKTUR.md Etappe 2.
        self.llm_tools = self.tools if backend == "deepseek" else None
        self.index = tool_protocol.catalog_index(self.tools)
        self.safety = safety or (Safety() if backend == "local" else None)

    def _run_local(self, user_query: str, ctx: Optional[Context] = None) -> Dict[str, Any]:
        """Lokaler Pfad: Regel-Router -> JSON-Protokoll -> Validator -> Safety -> Dispatch."""
        t0 = time.time()
        intent, names = tool_protocol.route(user_query, list(self.index))
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        tool_calls_log: List[Dict[str, Any]] = []
        total_tokens = 0

        def result(answer: str, iterations: int) -> Dict[str, Any]:
            # Rollen-Praefix ("Moloch:") entfernen, den das Modell gern mitschreibt
            answer = re.sub(r"^\s*(moloch|antwort)\s*:\s*", "", answer or "", flags=re.IGNORECASE).strip()
            return {"answer": answer, "iterations": iterations, "tool_calls": tool_calls_log,
                    "total_tokens": total_tokens, "intent": intent,
                    "duration_s": round(time.time() - t0, 1)}

        if not names:
            # Smalltalk: frei antworten, ohne JSON-Zwang (bessere Sprache, kein Tool-Fehlgriff)
            messages.append({"role": "user", "content": user_query})
            response = self.llm.complete(messages, stop=LOCAL_STOP)
            total_tokens += self.llm.extract_usage(response).get("total_tokens", 0)
            return result((self.llm.extract_message(response).get("content") or "").strip(), 1)

        # Regel-Aufloesung: der Code holt/tut es, das Modell formuliert nur (kein Raten,
        # keine vertauschte Richtung, ein Modelldurchlauf statt zwei)
        resolved = tool_protocol.resolve(user_query, intent, names)
        if resolved:
            tool, args = resolved
            decision = self.safety.check(tool, args, ctx)
            if decision.ok:
                bridge_result = self.bridge.dispatch(tool, decision.args)
            else:
                bridge_result = {"result": None,
                                 "error": f"nicht ausgefuehrt ({decision.action}): {decision.reason}"}
            used = decision.args or args
            if self.verbose:
                logger.info(f"[orch] regel intent={intent} {tool}({used}) -> {decision.action} {decision.reason}")
            tool_calls_log.append({"name": tool, "params": used, "safety": decision.action,
                                   "reason": decision.reason, "result": bridge_result, "by": "regel"})
            hinweis = ""
            if decision.action == "clamp":
                hinweis = (f" Der Wunsch wurde begrenzt: verlangt {json.dumps(args, ensure_ascii=False)}, "
                           f"ausgefuehrt {json.dumps(used, ensure_ascii=False)}. Sag das.")
            messages.append({"role": "user", "content": user_query})
            # Daten als eigene System-Nachricht NACH der Frage: der Persona-Praefix bleibt im
            # Cache, und das Modell wiederholt die Rohdaten seltener (zusaetzlich: stop).
            messages.append({"role": "system", "content": (
                f"Bereits erledigt: {tool} {json.dumps(used, ensure_ascii=False)}\n"
                f"Ergebnis: {json.dumps(bridge_result, ensure_ascii=False)[:LOCAL_TOOL_RESULT_CHARS]}\n"
                "Antworte Markus jetzt in ein bis zwei Saetzen in deinem Ton, in der Ich-Form. "
                "Nutze nur dieses Ergebnis, erfinde nichts dazu. Uebersetze keine Eigennamen und "
                "keine Zonen (guardian, shadow, berserker). Gib keine Rohdaten und keine Klammern wieder. "
                "Wurde etwas nicht ausgefuehrt, sag das ehrlich." + hinweis
            )})
            response = self.llm.complete(
                messages, temperature=0.3,
                stop=LOCAL_STOP + ["\n[", "[Ergebnis", "[Ausgef", "Ergebnis:", "=>"])
            total_tokens += self.llm.extract_usage(response).get("total_tokens", 0)
            return result((self.llm.extract_message(response).get("content") or "").strip(), 1)

        # Keine Regel greift: Modell waehlt per JSON-Protokoll.
        # Werkzeug-Block als zweite System-Nachricht: der Persona-Praefix bleibt im Cache
        messages.append({"role": "system", "content": tool_protocol.prompt_block(names, self.index)})
        messages.append({"role": "user", "content": user_query})
        fmt = tool_protocol.schema(names)
        repaired = False
        max_iter = min(self.max_iter, LOCAL_MAX_ITER)

        for iteration in range(1, max_iter + 1):
            if time.time() - t0 > LOCAL_WALL_S:
                return result("[lokales Zeitlimit erreicht]", iteration - 1)
            response = self.llm.complete(messages, response_format=fmt, max_tokens=140)
            total_tokens += self.llm.extract_usage(response).get("total_tokens", 0)
            raw = self.llm.extract_message(response).get("content") or ""
            parsed = tool_protocol.parse(raw, names, self.index)
            if self.verbose:
                logger.info(f"[orch] iter {iteration} intent={intent} roh={raw[:160]!r}")

            if parsed.error:
                if repaired:
                    return result("[lokales Modell lieferte kein gueltiges Format]", iteration)
                repaired = True
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"FEHLER: {parsed.error} Antworte erneut."})
                continue
            if parsed.tool is None:
                return result(parsed.say, iteration)

            decision = self.safety.check(parsed.tool, parsed.args, ctx)
            if decision.ok:
                bridge_result = self.bridge.dispatch(parsed.tool, decision.args)
            else:
                bridge_result = {"result": None,
                                 "error": f"nicht ausgefuehrt ({decision.action}): {decision.reason}"}
            if self.verbose:
                logger.info(f"[orch]   tool_call: {parsed.tool}({decision.args or parsed.args}) "
                            f"-> {decision.action} {decision.reason}")
            tool_calls_log.append({"name": parsed.tool, "params": decision.args or parsed.args,
                                   "safety": decision.action, "reason": decision.reason,
                                   "result": bridge_result})
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": (
                f"TOOL-ERGEBNIS {parsed.tool}: "
                f"{json.dumps(bridge_result, ensure_ascii=False)[:LOCAL_TOOL_RESULT_CHARS]}\n"
                "Antworte jetzt Markus (tool = null, done = true)."
            )})
        return result("[max iterations reached without final answer]", max_iter)

    def run(self, user_query: str, ctx: Optional[Context] = None) -> Dict[str, Any]:
        """Loop bis finale Antwort. Returns {answer, iterations, tool_calls, total_tokens}."""
        if self.backend == "local":
            return self._run_local(user_query, ctx)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]
        tool_calls_log: List[Dict[str, Any]] = []
        total_tokens = 0

        cloud = self.backend == "deepseek"
        for iteration in range(1, self.max_iter + 1):
            # Welle 21 Phase 5: Per-Loop-Budget-Check (nur Cloud — lokal kostet nichts)
            if cloud and total_tokens >= token_budget.PER_LOOP_DEFAULT:
                logger.warning(
                    f"[orch] Per-Loop-Budget exhausted ({total_tokens} >= "
                    f"{token_budget.PER_LOOP_DEFAULT}) — abort"
                )
                return {
                    "answer": "[token-budget exhausted — kuerz die Frage oder reset Limit]",
                    "iterations": iteration - 1,
                    "tool_calls": tool_calls_log,
                    "total_tokens": total_tokens,
                }
            if cloud and token_budget.is_over_daily_cap():
                logger.warning("[orch] Daily-Cap exceeded — Cloud-Calls blocked")
                return {
                    "answer": "[daily token-cap erreicht — bitte morgen weiter oder MOLOCH_DAILY_TOKEN_HARD_CAP env hochsetzen]",
                    "iterations": iteration - 1,
                    "tool_calls": tool_calls_log,
                    "total_tokens": total_tokens,
                }
            if self.verbose:
                logger.info(f"[orch] iter {iteration}/{self.max_iter}")
            response = self.llm.complete(messages, tools=self.llm_tools)
            usage = self.llm.extract_usage(response)
            total_tokens += usage.get("total_tokens", 0)
            assistant_msg = self.llm.extract_message(response)
            if self.verbose and response.get("timing"):
                t = response["timing"]
                logger.info(
                    f"[orch]   lokal: {t['total_s']:.1f}s gesamt, Prompt {t['prompt_s']:.1f}s, "
                    f"{t['tok_per_s']:.1f} Tok/s"
                )
            messages.append(assistant_msg)
            tool_calls = assistant_msg.get("tool_calls") or []
            if not tool_calls:
                # Finale Antwort
                return {
                    "answer": assistant_msg.get("content", ""),
                    "iterations": iteration,
                    "tool_calls": tool_calls_log,
                    "total_tokens": total_tokens,
                }
            for tc in tool_calls:
                tname = tc.get("function", {}).get("name", "?")
                raw_args = tc.get("function", {}).get("arguments", "{}")
                try:
                    targs = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    targs = {}
                if self.verbose:
                    logger.info(f"[orch]   tool_call: {tname}({targs})")
                bridge_result = self.bridge.dispatch(tname, targs)
                tool_calls_log.append(
                    {"name": tname, "params": targs, "result": bridge_result}
                )
                # Tool-Result als string fuer LLM
                tool_content = json.dumps(bridge_result, ensure_ascii=False)[:5000]
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "content": tool_content,
                    }
                )
        return {
            "answer": "[max iterations reached without final answer]",
            "iterations": self.max_iter,
            "tool_calls": tool_calls_log,
            "total_tokens": total_tokens,
        }


def main(argv: List[str]) -> int:
    # Windows-cp1252-Fallback: stdout auf UTF-8 reconfigure damit Emojis
    # (✅, etc.) nicht UnicodeEncodeError werfen
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="MOLOCH Orchestrator (Welle 21)")
    parser.add_argument("query", nargs="+", help="User-Query")
    parser.add_argument("--mock", action="store_true",
                        help="Erzwingt MockBridge (kein Pi-Bezug)")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="local",
                        help="local = Ollama auf diesem PC (Default), deepseek = Cloud")
    args = parser.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    user_query = " ".join(args.query)
    bridge = MockBridge() if args.mock else get_bridge()
    orch = Orchestrator(bridge=bridge, max_iter=args.max_iter, verbose=args.verbose,
                        backend=args.backend)
    result = orch.run(user_query)
    print("=" * 60)
    print(f"ANTWORT (nach {result['iterations']} iter, {result['total_tokens']} tokens):")
    print(result["answer"])
    if result["tool_calls"] and args.verbose:
        print("\nTOOL-CALLS:")
        for tc in result["tool_calls"]:
            print(f"  - {tc['name']}({tc['params']}) -> "
                  f"{'OK' if tc['result'].get('error') is None else 'ERR: ' + str(tc['result'].get('error'))[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
