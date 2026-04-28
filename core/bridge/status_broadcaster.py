#!/usr/bin/env python3
"""
status_broadcaster — UDS-Socket Notifier fuer Status-Updates + Frame-Events.

Singleton. Subscriber verbinden sich mit /tmp/moloch_notify.sock (SOCK_STREAM).
Bei Status-Write oder neuem Frame: 1-Byte b"S" an alle Clients gesendet.
Subscriber lesen dann /dev/shm/moloch_status.json direkt (kein Frame durch Socket).

Warum: ersetzt busy-polling (Panel, GUI) durch eventgesteuerte Notification.
RAM-overhead: ~1KB pro Subscriber-Connection, kein Prozess-Fork.
"""

import logging
import os
import select
import socket
import threading
from typing import List

logger = logging.getLogger("StatusBroadcaster")
NOTIFY_SOCK = "/tmp/moloch_notify.sock"
_SIGNAL_BYTE = b"S"


class StatusBroadcaster:
    def __init__(self):
        self._lock = threading.Lock()
        self._clients: List[socket.socket] = []
        self._server: socket.socket = None
        self._running = False
        self._thread: threading.Thread = None

    def start(self):
        """Server-Thread starten. Idempotent."""
        if self._running:
            return
        try:
            if os.path.exists(NOTIFY_SOCK):
                os.unlink(NOTIFY_SOCK)
            self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._server.bind(NOTIFY_SOCK)
            self._server.listen(8)
            self._server.setblocking(False)
            self._running = True
            self._thread = threading.Thread(
                target=self._accept_loop, daemon=True, name="StatusBroadcaster"
            )
            self._thread.start()
            logger.info(f"[BROADCAST] UDS-Socket bereit: {NOTIFY_SOCK}")
        except Exception as e:
            logger.warning(f"[BROADCAST] Start fehlgeschlagen: {e}")

    def stop(self):
        self._running = False
        try:
            self._server.close()
        except Exception:
            pass

    def notify(self):
        """1-Byte-Signal an alle Subscriber senden. Thread-safe, kein Block."""
        dead = []
        with self._lock:
            clients = list(self._clients)
        for c in clients:
            try:
                c.sendall(_SIGNAL_BYTE)
            except Exception:
                dead.append(c)
        if dead:
            with self._lock:
                for d in dead:
                    if d in self._clients:
                        self._clients.remove(d)
                    try:
                        d.close()
                    except Exception:
                        pass

    def _accept_loop(self):
        while self._running:
            try:
                r, _, _ = select.select([self._server], [], [], 1.0)
                if r:
                    conn, _ = self._server.accept()
                    conn.setblocking(False)
                    with self._lock:
                        self._clients.append(conn)
                    logger.debug("[BROADCAST] Neuer Subscriber")
            except Exception:
                pass


_instance: StatusBroadcaster = None
_instance_lock = threading.Lock()


def get_broadcaster() -> StatusBroadcaster:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = StatusBroadcaster()
    return _instance
