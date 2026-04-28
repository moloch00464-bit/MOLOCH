#!/usr/bin/env python3
"""
M.O.L.O.C.H. Event Bus — Asyncio Pub/Sub mit Priority-Queues
=============================================================

Erweitert den bestehenden MolochEventBus aus action_bridge.py:
- Priority-Queues (0=critical bis 9=logging)
- Event-Schema: {timestamp, event_type, priority, source, payload}
- Event-Deduplication innerhalb 1s
- Disk-Writes in separatem Thread
- Sync + Async Interface (kompatibel mit bestehendem Threading-Code)

Singleton: get_event_bus() — ersetzt den alten Bus aus action_bridge.py.

Author: M.O.L.O.C.H. System (Gate 1)
"""

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from queue import PriorityQueue, Empty
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("MolochEventBus")

# Disk-Write Pfad
EVENT_LOG_DIR = os.path.expanduser("~/moloch/logs/events")

# ============================================================
# JSONL-PERSIST (Phase 5 Task 5c-V0)
# RAM-Disk-Persist fuer Crash-Recovery + Cross-Session Diagnose
# ============================================================
import json as _json
import os as _os

_EVENT_BUS_JSONL = "/dev/shm/event_bus.jsonl"
_seq_counter = 0           # monoton steigende Sequenz-Nr (Modul-Level)
_seq_lock = threading.Lock()


# ============================================================
# EVENT SCHEMA
# ============================================================

@dataclass(order=True)
class MolochEvent:
    """
    Standardisiertes Event-Schema.

    priority: 0=critical, 1=perception, 2=action, 5=info, 9=logging
    """
    priority: int = field(compare=True)
    timestamp: float = field(default_factory=time.time, compare=False)
    event_type: str = field(default="unknown", compare=False)
    source: str = field(default="unknown", compare=False)
    payload: dict = field(default_factory=dict, compare=False)

    # Internes Feld fuer Dedup-Hash
    _dedup_key: str = field(default="", compare=False, repr=False)

    def __post_init__(self):
        """Dedup-Key aus event_type + source + payload generieren."""
        raw = f"{self.event_type}:{self.source}:{json.dumps(self.payload, sort_keys=True, default=str)}"
        self._dedup_key = hashlib.md5(raw.encode()).hexdigest()

    def to_dict(self) -> dict:
        """Event als Dict (fuer IPC, Logging, Disk)."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "priority": self.priority,
            "source": self.source,
            "payload": self.payload,
        }


# Priority-Konstanten
PRIO_CRITICAL = 0
PRIO_PERCEPTION = 1
PRIO_ACTION = 2
PRIO_BRIDGE = 3
PRIO_SYSTEM = 4
PRIO_INFO = 5
PRIO_DEBUG = 8
PRIO_LOGGING = 9


# ============================================================
# DISK WRITER — Separater Thread fuer persistente Event-Logs
# ============================================================

class _DiskWriter:
    """Schreibt Events asynchron auf Disk in separatem Thread."""

    def __init__(self, log_dir: str = EVENT_LOG_DIR):
        self._log_dir = log_dir
        self._queue: List[dict] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._flush_interval = 5.0  # Sekunden zwischen Disk-Writes

    def start(self):
        if self._running:
            return
        os.makedirs(self._log_dir, exist_ok=True)
        self._running = True
        self._thread = threading.Thread(
            target=self._write_loop, daemon=True, name="EventBusDisk"
        )
        self._thread.start()
        logger.debug("[DISK-WRITER] Gestartet")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        # Letzte Events noch schreiben
        self._flush()

    def enqueue(self, event_dict: dict):
        """Event in Write-Queue einreihen (non-blocking)."""
        with self._lock:
            self._queue.append(event_dict)

    def _write_loop(self):
        while self._running:
            time.sleep(self._flush_interval)
            self._flush()

    def _flush(self):
        """Alle gepufferten Events auf Disk schreiben."""
        with self._lock:
            if not self._queue:
                return
            batch = self._queue.copy()
            self._queue.clear()

        # Dateiname nach Datum
        date_str = time.strftime("%Y-%m-%d")
        filepath = os.path.join(self._log_dir, f"events_{date_str}.jsonl")
        try:
            with open(filepath, "a") as f:
                for evt in batch:
                    f.write(json.dumps(evt, default=str) + "\n")
        except Exception as e:
            logger.error(f"[DISK-WRITER] Schreibfehler: {e}")


# ============================================================
# EVENT BUS — Asyncio Pub/Sub mit Priority-Queues
# ============================================================

class MolochEventBus:
    """
    Zentraler Event Bus mit Priority-Queues und Deduplication.

    Bietet sowohl sync (publish/subscribe) als auch async (emit/listen)
    Interfaces fuer Kompatibilitaet mit bestehendem Threading-Code.

    Priority-Levels:
      0 = CRITICAL (System-Notfall, Shutdown)
      1 = PERCEPTION (Kamera, NPU, Sensoren)
      2 = ACTION (PTZ, TTS, LED)
      3 = BRIDGE (FSM-Transitionen)
      4 = SYSTEM (Health, Thermal)
      5 = INFO (Status-Updates)
      8 = DEBUG
      9 = LOGGING

    Deduplication: Identische Events (gleicher Typ+Source+Payload)
    werden innerhalb von 1 Sekunde ignoriert.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Subscriber: topic -> [(priority, callback), ...]
        self._subscribers: Dict[str, List[tuple]] = {}
        self._sub_lock = threading.Lock()

        # Async Subscriber: topic -> [async_callback, ...]
        self._async_subscribers: Dict[str, List[Callable]] = {}
        self._async_lock = threading.Lock()

        # Deduplication: dedup_key -> letzter Timestamp
        self._dedup_cache: Dict[str, float] = {}
        self._dedup_window = 1.0  # Sekunden
        self._dedup_lock = threading.Lock()

        # Event-Log (Ringbuffer fuer Debug/Panel)
        self._event_log: List[dict] = []
        self._max_log = 200

        # Disk Writer
        self._disk_writer = _DiskWriter()
        self._disk_writer.start()

        # Silence-Level: 0=normal, 1=reduced (nur PRIO<=2), 2=silent (nur PRIO 0)
        self._silence_level = 0

        # Statistik
        self._stats = {
            "total_published": 0,
            "total_deduplicated": 0,
            "total_delivered": 0,
            "total_errors": 0,
            "total_silenced": 0,
        }

        logger.info("[EVENT-BUS] Initialisiert (Priority 0-9, Dedup 1s, Disk-Writer aktiv)")

    # ============================================================
    # SUBSCRIBE — Sync Callbacks
    # ============================================================

    def subscribe(self, topic: str, callback: Callable, priority: int = 5):
        """
        Subscriber registrieren (synchron).

        Niedrigere Priority = frueher aufgerufen.
        Kompatibel mit bestehendem action_bridge.py Code.

        Args:
            topic: Event-Typ zum Abonnieren (z.B. "perception.person_detected")
            callback: Funktion die mit MolochEvent-Dict aufgerufen wird
            priority: Aufruf-Reihenfolge (0=zuerst, 9=zuletzt)
        """
        with self._sub_lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            self._subscribers[topic].append((priority, callback))
            self._subscribers[topic].sort(key=lambda x: x[0])
        logger.debug(f"[EVENT-BUS] Subscribe: {topic} prio={priority}")

    def unsubscribe(self, topic: str, callback: Callable):
        """Subscriber entfernen."""
        with self._sub_lock:
            if topic in self._subscribers:
                self._subscribers[topic] = [
                    (p, cb) for p, cb in self._subscribers[topic] if cb is not callback
                ]

    # ============================================================
    # SUBSCRIBE — Async Callbacks
    # ============================================================

    def subscribe_async(self, topic: str, callback: Callable):
        """
        Async Subscriber registrieren.

        Callback muss eine async def Funktion sein.
        Wird via asyncio.create_task() aufgerufen.
        """
        with self._async_lock:
            if topic not in self._async_subscribers:
                self._async_subscribers[topic] = []
            self._async_subscribers[topic].append(callback)
        logger.debug(f"[EVENT-BUS] Async-Subscribe: {topic}")

    def unsubscribe_async(self, topic: str, callback: Callable):
        """Async Subscriber entfernen."""
        with self._async_lock:
            if topic in self._async_subscribers:
                self._async_subscribers[topic] = [
                    cb for cb in self._async_subscribers[topic] if cb is not callback
                ]

    # ============================================================
    # PUBLISH — Synchron (Thread-safe, fuer bestehenden Code)
    # ============================================================

    def publish(self, event_type: str, payload: dict = None,
                source: str = "unknown", priority: int = 5,
                # Rueckwaerts-kompatibel: action_bridge nutzt topic= und data=
                topic: str = None, data: dict = None) -> bool:
        """
        Event synchron publishen. Alle Subscriber werden aufgerufen.

        Rueckgabe: True wenn zugestellt, False wenn dedupliziert.

        Kompatibilitaet: topic/data werden auf event_type/payload gemappt.
        """
        # Rueckwaerts-Kompatibilitaet mit action_bridge.py Aufrufen
        if topic is not None:
            event_type = topic
        if data is not None:
            payload = data

        event = MolochEvent(
            priority=priority,
            timestamp=time.time(),
            event_type=event_type,
            source=source,
            payload=payload or {},
        )

        # --- DEDUPLICATION ---
        if self._is_duplicate(event):
            self._stats["total_deduplicated"] += 1
            return False

        # --- SILENCE-LEVEL FILTER ---
        if self._silence_level == 2 and priority > PRIO_CRITICAL:
            self._stats["total_silenced"] += 1
            return False
        if self._silence_level == 1 and priority > PRIO_ACTION:
            self._stats["total_silenced"] += 1
            return False

        self._stats["total_published"] += 1
        event_dict = event.to_dict()

        # Event-Log (Ringbuffer)
        self._event_log.append(event_dict)
        if len(self._event_log) > self._max_log:
            self._event_log.pop(0)

        # Disk-Write (async via Thread)
        self._disk_writer.enqueue(event_dict)

        # --- JSONL-PERSIST (Phase 5 Task 5c-V0) ---
        # Schreibt jeden publish() in /dev/shm/event_bus.jsonl mit Sequence-Nummer.
        # Persist-Fehler darf Dispatch NICHT stoppen.
        try:
            global _seq_counter
            with _seq_lock:
                _seq_counter += 1
                seq_now = _seq_counter
            entry = {
                "seq": seq_now,
                "topic": event_type,
                "payload": event_dict.get("payload", {}),
                "ts": event_dict.get("timestamp", time.time()),
            }
            with open(_EVENT_BUS_JSONL, "a") as _f:
                _f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
                _f.flush()
            # File-Rotation: alle 100 Events pruefen, max 500 Zeilen
            if seq_now % 100 == 0:
                try:
                    with open(_EVENT_BUS_JSONL, "r") as _f:
                        _lines = _f.readlines()
                    if len(_lines) > 500:
                        with open(_EVENT_BUS_JSONL, "w") as _f:
                            _f.writelines(_lines[-300:])
                except Exception:
                    pass
        except Exception:
            pass  # Persist-Fehler darf Dispatch NICHT stoppen

        # --- SYNC SUBSCRIBER ---
        with self._sub_lock:
            subs = list(self._subscribers.get(event_type, []))
        for _prio, callback in subs:
            try:
                callback(event_dict)
                self._stats["total_delivered"] += 1
            except Exception as e:
                self._stats["total_errors"] += 1
                logger.error(f"[EVENT-BUS] Subscriber-Fehler {event_type}: {e}")

        # --- ASYNC SUBSCRIBER ---
        with self._async_lock:
            async_subs = list(self._async_subscribers.get(event_type, []))
        if async_subs:
            self._dispatch_async(async_subs, event_dict)

        return True

    # Alias fuer Kompatibilitaet
    emit = publish

    # ============================================================
    # ASYNC DISPATCH
    # ============================================================

    def _dispatch_async(self, callbacks: List[Callable], event_dict: dict):
        """Async Subscriber in laufender Event-Loop dispatchen."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Kein Event-Loop aktiv — async Callbacks ueberspringen
            return
        for cb in callbacks:
            try:
                loop.create_task(cb(event_dict))
            except Exception as e:
                self._stats["total_errors"] += 1
                logger.error(f"[EVENT-BUS] Async-Fehler: {e}")

    # ============================================================
    # DEDUPLICATION
    # ============================================================

    def _is_duplicate(self, event: MolochEvent) -> bool:
        """Prueft ob identisches Event innerhalb von 1s bereits gesendet wurde."""
        now = time.time()
        key = event._dedup_key

        with self._dedup_lock:
            # Alte Eintraege aufraeumen (aelter als 2x Window)
            expired = [k for k, t in self._dedup_cache.items() if now - t > self._dedup_window * 2]
            for k in expired:
                del self._dedup_cache[k]

            # Duplikat?
            if key in self._dedup_cache:
                if now - self._dedup_cache[key] < self._dedup_window:
                    return True

            # Neuer Eintrag
            self._dedup_cache[key] = now
            return False

    # ============================================================
    # PUBLIC API — Query & Debug
    # ============================================================

    def get_recent_events(self, count: int = 20) -> List[dict]:
        """Letzte N Events fuer Debug/Panel."""
        return self._event_log[-count:]

    def get_stats(self) -> dict:
        """Bus-Statistiken."""
        return {
            **self._stats,
            "subscribers": {topic: len(subs) for topic, subs in self._subscribers.items()},
            "async_subscribers": {topic: len(subs) for topic, subs in self._async_subscribers.items()},
            "dedup_cache_size": len(self._dedup_cache),
        }

    def get_subscriber_count(self) -> int:
        """Gesamtzahl aller Subscriber."""
        total = sum(len(subs) for subs in self._subscribers.values())
        total += sum(len(subs) for subs in self._async_subscribers.values())
        return total

    def set_silence_level(self, level: int):
        """Silence-Level setzen: 0=normal, 1=reduced (PRIO<=2), 2=silent (nur CRITICAL)."""
        self._silence_level = max(0, min(2, level))
        logger.info(f"[EVENT-BUS] Silence-Level: {self._silence_level}")

    @property
    def silence_level(self) -> int:
        return self._silence_level

    # ============================================================
    # LIFECYCLE
    # ============================================================

    def stop(self):
        """Bus sauber herunterfahren (Disk-Writer flushen)."""
        self._disk_writer.stop()
        logger.info(f"[EVENT-BUS] Gestoppt. Stats: {self._stats}")


# ============================================================
# SINGLETON
# ============================================================

def get_event_bus() -> MolochEventBus:
    """Singleton-Zugriff auf den Event Bus."""
    return MolochEventBus()


# ============================================================
# JSONL-RECOVERY (Phase 5 Task 5c-V0)
# ============================================================

def get_last_events(n: int = 20, topic_filter: str = None) -> list:
    """
    Letzte N Events aus /dev/shm/event_bus.jsonl lesen.

    Verwendung: Crash-Recovery (Subscriber kann nach Neustart Stand wiederherstellen)
    und Cross-Session Diagnose ("was ist in der letzten Stunde passiert?").

    Args:
        n: Maximale Anzahl Events zurueckgeben.
        topic_filter: Wenn gesetzt, nur Events mit passendem topic.

    Returns:
        Liste von dicts mit Keys: seq, topic, payload, ts.
        Leere Liste falls Datei nicht existiert oder Fehler auftritt.
    """
    try:
        with open(_EVENT_BUS_JSONL, "r") as f:
            lines = f.readlines()
        events = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                events.append(_json.loads(ln))
            except Exception:
                continue
        if topic_filter:
            events = [e for e in events if e.get("topic") == topic_filter]
        return events[-n:]
    except Exception:
        return []
