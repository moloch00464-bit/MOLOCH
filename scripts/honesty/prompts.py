"""Prompt-Bau fuer die zwei Wege, ueber die Moloch heute antwortet.

Beide Wege werden mit den PRODUKTIVEN Funktionen gebaut, nur ihre Datenquellen
werden durch Fixtures ersetzt. So landet jede Aenderung am Produktivcode
automatisch im Test.

Weg A - Pi-Chat (Tentakel): LocalLLMBridge._generate_tentacle() wird mit einem
    Ersatz-"self" aufgerufen, dessen HTTP-Client die fertige Ollama-Anfrage nur
    einfaengt und nicht sendet. Verlauf: was _build_local_context_snippet() aus
    get_recent_messages(n=2) macht (gekuerzt, als "VORHER" im System-Prompt).
Weg B - KI-Zentrale lokal: Persona wie GET /system_prompt, Verlauf als
    Nachrichten (bis 6 Turns), Kontextblock aus pc/ki_zentrale._context_block().

Bekannte Abweichungen von der Produktion (bewusst):
- Weg A ohne Klassifizierung durch den Chat-Server (prompt_type=None). Fragen,
  die dort als Web-Recherche erkannt wuerden, gingen in der Produktion an
  Suchdienst + Cloud; das loest der Test absichtlich NICHT aus.
- Weg B ohne Faehigkeits-Block (_build_capability_snippet im Chat-Server) und
  ohne den 10-Minuten-Cache der Persona.
- ThreeBrain-Zustand (Drift, Regeln, Journal) leer: haengt am Geraet, nicht am Test.
"""

import contextlib
import json
import os
import sys
import tempfile
import types
from typing import Dict, List, Optional
from unittest import mock

from . import fixtures as fx

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACE_ATTR_OVERRIDE = "/dev/shm/moloch_test_face_attr_override.json"


class NetworkBlocked(RuntimeError):
    """Beim Prompt-Bau darf kein Netz- oder Prozessaufruf passieren."""


class _Captured(Exception):
    pass


def _forbid(*_a, **_k):
    raise NetworkBlocked("Prompt-Bau hat versucht, Netz oder Prozess zu nutzen")


@contextlib.contextmanager
def _sandbox(memory, status: dict, telemetry: Optional[List[str]]):
    """Produktive Datenquellen fuer die Dauer des Prompt-Baus durch Fixtures ersetzen."""
    if os.path.exists(FACE_ATTR_OVERRIDE):
        # Diese Datei liest der Produktivcode mit festem Pfad - sie wuerde den Test verfaelschen.
        raise RuntimeError(f"{FACE_ATTR_OVERRIDE} existiert - erst entfernen, dann testen")
    import core.autonomy.local_llm_bridge as b
    import core.longterm_memory as lm

    with tempfile.TemporaryDirectory() as tmp:
        status_path = os.path.join(tmp, "moloch_status.json")
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f)
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(b, "_STATUS_JSON_PATH", status_path))
            st.enter_context(mock.patch.object(b, "_load_tentacle_cfg",
                                               lambda: {"enabled": True, "timeout_sec": 90}))
            st.enter_context(mock.patch.object(b, "_build_telemetry_footer",
                                               lambda: fx.telemetry_footer(telemetry)))
            st.enter_context(mock.patch.object(b, "_build_threebrain_state_snippet",
                                               lambda *a, **k: ""))
            st.enter_context(mock.patch.object(lm, "get_memory", lambda: memory))
            st.enter_context(mock.patch("subprocess.run", _forbid))
            st.enter_context(mock.patch("requests.post", _forbid))
            st.enter_context(mock.patch("requests.get", _forbid))
            st.enter_context(mock.patch("requests.Session.request", _forbid))
            yield b


def _append_system(messages: List[Dict], text: str) -> None:
    if not text:
        return
    if messages and messages[0]["role"] == "system":
        messages[0] = {"role": "system", "content": messages[0]["content"] + "\n\n" + text}
    else:
        messages.insert(0, {"role": "system", "content": text})


def build_weg_a(question: str, memory, status: dict, telemetry: Optional[List[str]] = None,
                extra: Optional[List[str]] = None, model: str = "qwen2.5:7b-instruct") -> Dict:
    """Ollama-Anfrage, wie der Pi-Chat sie fuer diese Frage stellen wuerde."""
    captured: Dict = {}

    class _Http:
        def post(self, url, json=None, timeout=None, **_k):
            captured["json"] = json
            raise _Captured()

    with _sandbox(memory, status, telemetry) as b:
        fake = types.SimpleNamespace(
            _tentacle_backoff_until=0.0, _tentacle_fail_count=0, _last_provider=None,
            _http=_Http(),
            _is_tentacle_running=lambda cfg: True,
            _discover_tentacle_model=lambda cfg: model,
            _tentacle_url=lambda cfg: "http://ehrlichkeitspruefung.invalid",
        )
        b.LocalLLMBridge._generate_tentacle(fake, question, "", 160, prompt_type=None)

    if "json" not in captured:
        raise RuntimeError("Weg A: Produktivcode hat keine Anfrage gebaut (Tentakel aus?)")
    req = captured["json"]
    messages = [dict(m) for m in req["messages"]]
    _append_system(messages, fx.extra_block(extra))
    return {"weg": "A", "model": req["model"], "messages": messages,
            "options": dict(req.get("options") or {}),
            "kanal_zusatz": bool(extra)}


def _import_zentrale():
    pc_dir = os.path.join(ROOT, "pc")
    if pc_dir not in sys.path:
        sys.path.insert(0, pc_dir)
    import ki_zentrale  # braucht fastapi, httpx, psutil (PC-Umgebung)
    return ki_zentrale


def _zentrale_options() -> Dict:
    with open(os.path.join(ROOT, "pc", "ki_rollen.json"), encoding="utf-8") as f:
        roles = json.load(f)
    opts = roles.get("chat_optionen") or {}
    return {
        "model": roles.get("haupthirn") or "qwen2.5:7b-instruct",
        "turns": int(opts.get("verlauf_turns", 6)),
        "options": {
            "num_ctx": int(opts.get("num_ctx", 4096)),
            "num_predict": int(opts.get("num_predict", 160)),
            "temperature": float(opts.get("temperature", 0.3)),
            "stop": ["<|im_start|>", "<|im_end|>"],
        },
    }


def build_weg_b(question: str, memory, status: dict, live: Optional[dict],
                history: List[Dict[str, str]], extra: Optional[List[str]] = None) -> Dict:
    """Ollama-Anfrage, wie der lokale Chat der KI-Zentrale sie stellen wuerde.

    history: bisherige Turns dieses Tests [{"sender": "user"|"moloch", "text": ...}]
    live:    None = Pi nicht erreichbar
    """
    kz = _import_zentrale()
    cfg = _zentrale_options()
    with _sandbox(memory, status, None) as b:
        import json as _json
        with open(b._PROFILES_PATH, encoding="utf-8") as f:
            profile = (_json.load(f).get("profiles", {}) or {}).get("tentacle") or {}
        persona = profile.get("system", "")
        if profile.get("include_live_context", True):
            persona += b._build_local_context_snippet()
        mctx = memory.get_memory_context_minimal()
        if mctx:
            persona += "\n\n--- MEMORY ---\n" + mctx
        hits = memory.recall_hits(question, limit=3)
        recall = {"count": len(hits), "hits": hits, "block": memory.recall(question, limit=3)}

    messages = [{"role": "system", "content": persona}]
    for h in history[-cfg["turns"] * 2:]:
        messages.append({"role": "user" if h["sender"] == "user" else "assistant",
                         "content": h["text"]})
    context = kz._context_block(question, live, recall)
    if extra:
        context += "\n" + fx.extra_block(extra)
    messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": question})
    return {"weg": "B", "model": cfg["model"], "messages": messages,
            "options": cfg["options"], "kanal_zusatz": bool(extra),
            "erinnerungen": len(hits)}
