"""
ExpressionOrchestrator — Lifecycle fuer alle 5 Expression-Module

API:
  start_all_expressions()    — beim Service-Boot (vom service-Agent gerufen)
  stop_all_expressions()     — graceful shutdown
  get_expression_state()     — Status aller Module fuer audit_state.layers.expression
"""
import logging
import threading
import time
from typing import Any, Dict

logger = logging.getLogger("expression.orchestrator")

_lock = threading.RLock()
_started: bool = False
_modules: Dict[str, Any] = {}
_start_ts: float = 0.0


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
