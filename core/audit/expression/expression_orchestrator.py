"""
ExpressionOrchestrator — Lifecycle fuer alle 5 Expression-Module

API:
  start_all_expressions()    — beim Service-Boot (vom service-Agent gerufen)
  stop_all_expressions()     — graceful shutdown
  get_expression_state()     — Status aller Module fuer audit_state.layers.expression

Schreibt periodisch (30s) atomic nach /dev/shm/expression_state.json damit
audit_orchestrator (separater Subprocess, eigener Singleton) den Live-State sieht.
"""
import json
import logging
import os
import tempfile
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("expression.orchestrator")

_lock = threading.RLock()
_started: bool = False
_modules: Dict[str, Any] = {}
_start_ts: float = 0.0

# Cross-Prozess-State-File (audit_orchestrator liest hier)
EXPRESSION_STATE_PATH = "/dev/shm/expression_state.json"
WRITER_INTERVAL_S = 30.0
_writer_thread: Optional[threading.Thread] = None
_writer_stop = threading.Event()


def _atomic_write_state(state: Dict[str, Any]) -> bool:
    """Atomic via tempfile + os.replace (NEVER 6)."""
    try:
        d = os.path.dirname(EXPRESSION_STATE_PATH)
        fd, tmp = tempfile.mkstemp(dir=d, prefix="expression_state.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False)
            os.replace(tmp, EXPRESSION_STATE_PATH)
            return True
        except Exception:
            try: os.unlink(tmp)
            except OSError: pass
            raise
    except Exception as e:
        logger.debug(f"_atomic_write_state Fehler: {e}")
        return False


def _writer_loop():
    """Background-Thread schreibt expression_state alle 30s."""
    logger.info(f"expression-state-writer gestartet (Intervall {WRITER_INTERVAL_S}s)")
    # Sofort 1× schreiben damit audit nicht 30s wartet
    _atomic_write_state(get_expression_state())
    while not _writer_stop.wait(timeout=WRITER_INTERVAL_S):
        try:
            _atomic_write_state(get_expression_state())
        except Exception as e:
            logger.debug(f"writer-loop Fehler: {e}")
    logger.info("expression-state-writer gestoppt")


def _safe_get(module_name: str, getter_path: str):
    """Best-effort: importiert Modul + Singleton-Getter."""
    try:
        mod = __import__(f"core.audit.expression.{module_name}", fromlist=[getter_path])
        getter = getattr(mod, getter_path, None)
        if getter is None:
            return None
        return getter()
    except Exception as e:
        logger.debug(f"_safe_get({module_name}.{getter_path}): {e}")
        return None


def start_all_expressions() -> Dict[str, bool]:
    """Startet alle 5 Expression-Module. Gibt Status pro Modul zurueck."""
    global _started, _modules, _start_ts
    with _lock:
        if _started:
            return {name: True for name in _modules}
        _modules = {}
        registry = [
            ("tension_to_fan", "get_tension_to_fan"),
            ("mood_to_spotify", "get_mood_to_spotify"),
            ("zone_to_led", "get_zone_to_led"),
            ("berserker_strobo", "get_berserker_strobo"),
            ("tension_to_tts_volume", "get_tension_to_tts_volume"),
        ]
        results: Dict[str, bool] = {}
        for module_name, getter in registry:
            instance = _safe_get(module_name, getter)
            if instance is None:
                results[module_name] = False
                continue
            try:
                ok = instance.start() if hasattr(instance, "start") else True
                _modules[module_name] = instance
                results[module_name] = bool(ok)
            except Exception as e:
                logger.warning(f"start_all_expressions: {module_name}.start() fehlgeschlagen: {e}")
                results[module_name] = False
        _started = True
        _start_ts = time.time()
        ok_count = sum(1 for v in results.values() if v)
        logger.info(f"ExpressionOrchestrator: {ok_count}/{len(registry)} Module gestartet")
    # State-Writer ausserhalb des _lock starten (Thread-Start kann blockieren)
    global _writer_thread
    if _writer_thread is None or not _writer_thread.is_alive():
        _writer_stop.clear()
        _writer_thread = threading.Thread(target=_writer_loop, name="expression-state-writer", daemon=True)
        _writer_thread.start()
    return results


def stop_all_expressions() -> Dict[str, bool]:
    """Graceful shutdown aller Module."""
    global _started, _modules
    with _lock:
        if not _started:
            return {}
        results: Dict[str, bool] = {}
        for name, instance in list(_modules.items()):
            try:
                if hasattr(instance, "stop"):
                    instance.stop()
                results[name] = True
            except Exception as e:
                logger.debug(f"stop_all_expressions: {name}.stop() Fehler: {e}")
                results[name] = False
        _started = False
        _modules = {}
        logger.info("ExpressionOrchestrator: alle Module gestoppt")
    # Writer stoppen ausserhalb _lock
    _writer_stop.set()
    return results


def get_expression_state() -> Dict[str, Any]:
    """Status aller 5 Module — wird von audit_state.layers.expression genutzt."""
    with _lock:
        state: Dict[str, Any] = {
            "started": _started,
            "start_ts": _start_ts,
            "uptime_s": (time.time() - _start_ts) if _started and _start_ts else 0.0,
            "module_count": len(_modules),
            "modules": {},
        }
        alive_count = 0
        for name, instance in _modules.items():
            try:
                module_state = instance.get_state() if hasattr(instance, "get_state") else {"alive": True}
            except Exception as e:
                module_state = {"alive": False, "error": str(e)}
            state["modules"][name] = module_state
            if module_state.get("alive"):
                alive_count += 1
        state["alive_count"] = alive_count
        state["health"] = "ok" if alive_count == len(_modules) and _started else (
            "degraded" if alive_count > 0 else "down"
        )
        return state


def is_started() -> bool:
    with _lock:
        return _started
