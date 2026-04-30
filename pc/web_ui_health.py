"""MOLOCH Web-UI-Health-Check (Welle 12, PC-Side).

Prueft Web-Interfaces auf Robustheit damit Markus nicht wieder 'blödes
Scheißmikrofoneinstellungen wieder verschwindet'-Erlebnisse hat.

Geprueft:
  https_cert_valid     mkcert-Cert auf Pi nicht abgelaufen
  https_cert_days_left Tage bis Cert-Ablauf (warn <30, fail <7)
  pi_cockpit_https     :9443 reachable + 200
  pi_cockpit_http      :9100 reachable + 200
  ssl_tunnel_localhost :9000 reachable (= SSH-Tunnel auf Pi-9100)
  mkcert_root_ca       Root-CA im Win-Cert-Store
  mic_secure_context   Welche URL hat HTTPS-Mikrofon-Zugang?

Output: layer.web_ui Status + URL-Empfehlung.

POSTet alle 5 Min an Pi audit-Orchestrator.

CLI:
  python pc/web_ui_health.py --once
  python pc/web_ui_health.py --interval-s 300
  python pc/web_ui_health.py --recommend  # nur URL-Empfehlung ausgeben

NEVER-Regeln respektiert.
"""
import argparse
import json
import logging
import os
import socket
import ssl
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("web-ui-health")

PI_BASE_HTTP = os.environ.get("MOLOCH_PI_CHAT", "http://192.168.178.30:9100")
PI_BASE_HTTPS = os.environ.get("MOLOCH_PI_CHAT_HTTPS", "https://192.168.178.30:9443")
PI_HOST = os.environ.get("MOLOCH_PI_HOST", "192.168.178.30")
TUNNEL_LOCAL = "http://localhost:9000"
DEFAULT_INTERVAL_S = 300
TIMEOUT_S = 6
HEADERS = {"Content-Type": "application/json"}
STATE_DIR = Path.home() / "moloch_logs" / "audit"
STATE_DIR.mkdir(parents=True, exist_ok=True)


def safe_json_write(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, str(path))
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def probe_https_cert(host: str, port: int = 9443) -> dict:
    """SSL-Cert-Validity + Days-Left."""
    info = {}
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with socket.create_connection((host, port), timeout=5) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert(binary_form=False) or {}
                # Note: with verify_mode=NONE, getpeercert may be empty.
                # Use binary cert + cryptography for parsing if needed.
                der = ssock.getpeercert(binary_form=True)
                info["handshake_ok"] = True
                info["cert_bytes"] = len(der) if der else 0
        # Parse via openssl-style fallback
        try:
            from cryptography import x509
            from cryptography.hazmat.backends import default_backend
            with socket.create_connection((host, port), timeout=5) as sock:
                with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                    der = ssock.getpeercert(binary_form=True)
                    cert_obj = x509.load_der_x509_certificate(der, default_backend())
                    not_after = cert_obj.not_valid_after_utc
                    info["valid_until"] = not_after.isoformat()
                    delta = not_after - datetime.now(timezone.utc)
                    info["days_left"] = delta.days
                    info["subject"] = cert_obj.subject.rfc4514_string()
        except ImportError:
            info["cert_parse"] = "cryptography lib fehlt — pip install cryptography"
    except (socket.timeout, ssl.SSLError, OSError) as e:
        info["handshake_ok"] = False
        info["error"] = str(e)[:80]
    return info


def probe_url(url: str, timeout: int = 4, verify: bool = False) -> dict:
    info = {"url": url}
    try:
        r = requests.get(url, timeout=timeout, verify=verify)
        info["ok"] = r.status_code == 200
        info["status_code"] = r.status_code
        info["latency_ms"] = int(r.elapsed.total_seconds() * 1000)
    except requests.RequestException as e:
        info["ok"] = False
        info["error"] = str(e)[:80]
    return info


def probe_mkcert_root() -> dict:
    """mkcert Root-CA im Win-Cert-Store?"""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-ChildItem -Path Cert:\\CurrentUser\\Root | Where-Object { $_.Subject -like '*mkcert*' } | Select -ExpandProperty Subject"],
        capture_output=True, text=True, timeout=10,
    )
    subj = out.stdout.strip()
    return {"installed": bool(subj), "subject": subj or "fehlt"}


def derive_mic_url() -> dict:
    """Welche URL bietet sicheren Mikrofon-Zugang?"""
    # Web Speech API + getUserMedia brauchen secure context = HTTPS oder localhost
    # Reihenfolge: HTTPS-direct (best) > localhost-tunnel > HTTP (kein Mic)
    candidates = [
        ("https://192.168.178.30:9443/", "Pi-HTTPS-Cockpit (mkcert)"),
        ("http://localhost:9000/", "SSH-Tunnel (localhost = secure context)"),
    ]
    out = []
    for url, label in candidates:
        verify = url.startswith("https://")
        info = probe_url(url, verify=False)  # mkcert validation lassen wir Browser machen
        if info.get("ok"):
            out.append({"url": url, "label": label, "secure": True})
    return {
        "candidates": out,
        "recommended": out[0]["url"] if out else None,
        "note": "Browser-Settings: einmal 'Mikrofon zulassen' klicken bei erstem Klick auf Mic-Icon",
    }


def collect() -> dict:
    started = time.time()
    cert_info = probe_https_cert(PI_HOST, 9443)
    pi_https = probe_url(PI_BASE_HTTPS + "/health")
    pi_http = probe_url(PI_BASE_HTTP + "/health")
    tunnel = probe_url(TUNNEL_LOCAL + "/health")
    mkcert_root = probe_mkcert_root()
    mic = derive_mic_url()

    data = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "https_cert": cert_info,
        "pi_https_9443": pi_https,
        "pi_http_9100": pi_http,
        "tunnel_localhost_9000": tunnel,
        "mkcert_root_ca": mkcert_root,
        "mic_secure_context": mic,
    }

    issues = []
    status = "PASS"
    if cert_info.get("days_left", 999) < 7:
        status = "FAIL"
        issues.append(f"cert_expires_<7d ({cert_info.get('days_left')}d)")
    elif cert_info.get("days_left", 999) < 30:
        status = "WARN"
        issues.append(f"cert_expires_soon ({cert_info.get('days_left')}d)")
    if not cert_info.get("handshake_ok"):
        status = "FAIL"
        issues.append("https_handshake_fail")
    if not pi_https.get("ok") and not pi_http.get("ok"):
        status = "FAIL"
        issues.append("pi_cockpit_offline_completely")
    elif not pi_https.get("ok"):
        status = "WARN"
        issues.append("pi_https_down (no mic)")
    if not mkcert_root.get("installed"):
        status = "WARN"
        issues.append("mkcert_root_ca_missing (browser-cert-warning)")
    if not mic["recommended"]:
        status = "FAIL"
        issues.append("no_secure_mic_context_available")

    data["status"] = status
    data["issues"] = issues
    data["duration_s"] = round(time.time() - started, 2)
    score = max(0, 6 - len(issues))
    return {"score": score, "max": 6, "status": status, "detail": data}


def post_layer(payload: dict) -> bool:
    try:
        r = requests.post(
            f"{PI_BASE_HTTP}/mailbox/audit/web_ui",
            headers=HEADERS, json=payload, timeout=TIMEOUT_S,
        )
        if r.status_code == 200:
            return True
        logger.warning(f"[post] HTTP {r.status_code}: {r.text[:120]}")
    except requests.RequestException as e:
        logger.warning(f"[post] {e}")
    return False


def tick() -> dict:
    started = time.time()
    payload = collect()
    posted = post_layer(payload)
    state = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(time.time() - started, 2),
        "payload": payload,
        "posted": posted,
    }
    safe_json_write(STATE_DIR / "web_ui_health_last.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description="MOLOCH Web-UI-Health (Welle 12)")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-s", type=int, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--recommend", action="store_true", help="Nur URL-Empfehlung ausgeben")
    args = parser.parse_args()

    if args.json:
        last = STATE_DIR / "web_ui_health_last.json"
        print(last.read_text(encoding="utf-8") if last.exists() else "{}")
        return

    if args.recommend:
        rec = derive_mic_url()
        if rec["recommended"]:
            print(f"Mikrofon-tauglich: {rec['recommended']}")
            for c in rec["candidates"]:
                print(f"  - {c['label']}: {c['url']}")
        else:
            print("KEINE mic-taugliche URL erreichbar!")
        return

    if args.once:
        state = tick()
        print(f"[once] status={state['payload']['status']} score={state['payload']['score']}/{state['payload']['max']} posted={state['posted']}")
        cert = state['payload']['detail']['https_cert']
        if cert.get('days_left'):
            print(f"       cert valid {cert['days_left']} more days (until {cert.get('valid_until','?')})")
        if state['payload']['detail'].get('issues'):
            print(f"       issues: {state['payload']['detail']['issues']}")
        return

    logger.info(f"Web-UI-Health: Loop alle {args.interval_s}s")
    while True:
        try:
            state = tick()
            logger.info(f"tick status={state['payload']['status']} posted={state['posted']}")
        except Exception as e:
            logger.exception(f"tick fail: {e}")
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
