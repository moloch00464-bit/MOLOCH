"""SSRF-Schutz fuer search_proxy /fetch und browser_proxy /open.

Beide Proxys laufen auf 0.0.0.0 und holen URLs, die ein LLM aus Webinhalten
ableitet. Ohne Filter erreichen sie localhost, den Router, die Kamera
(192.168.178.25) und den Pi. Erlaubt sind nur global routbare Adressen.
"""
import ipaddress
import socket
import urllib.parse

_cache: dict = {}


def is_public_host(host: str) -> bool:
    """True, wenn alle aufgeloesten Adressen des Hosts global routbar sind."""
    if not host:
        return False
    host = host.strip("[]").lower()
    if host in _cache:
        return _cache[host]
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        ok = bool(infos) and all(
            ipaddress.ip_address(info[4][0].split("%")[0]).is_global for info in infos
        )
    except (socket.gaierror, ValueError, UnicodeError):
        ok = False
    if len(_cache) > 1024:
        _cache.clear()
    _cache[host] = ok
    return ok


def check_public_url(url: str) -> None:
    """Wirft ValueError fuer Nicht-HTTP(S)-URLs und interne Ziele."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"ungueltige URL: {url[:200]}")
    if not is_public_host(parsed.hostname):
        raise ValueError(f"internes Ziel blockiert: {parsed.hostname}")
