"""KI-Zentrale: lokale KI auf dem PC steuern und mit Moloch reden.

Laeuft im Dashboard-Prozess (:11700), eingehaengt per app.include_router().

    GET  /zentrale          Seite
    GET  /api/ki/status     Ollama, Modelle mit Rollen, RAM, Molochs Zustand vom Pi
    POST /api/ki/model      {"model", "action": "load"|"unload"}     nur am PC selbst
    POST /api/ki/ollama     {"action": "start"|"stop"}               nur am PC selbst
    POST /api/ki/chat       {"message", "mode": "local"|"pi"}
                            local: NDJSON-Stream direkt vom PC-Hirn (Ollama),
                                   Persona vom Pi (/system_prompt), Live-Zustand vom Pi (/live)
                            pi:    bisheriger Weg ueber den Pi (/chat, Routing entscheidet der Pi)
    GET  /api/ki/verlauf    letzte Eintraege aus dem PC-Chatlog

Rollen und Haupthirn stehen in ki_rollen.json. Der Chatlog liegt in
~/moloch_zentrale/chat_log.jsonl (Verlauf fuer den Prompt); lokale Turns werden
zusaetzlich per POST /memory/turn ins gemeinsame Pi-Gedaechtnis gespiegelt.
"""
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx
import psutil
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

OLLAMA_URL = os.environ.get("MOLOCH_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
PI_URL = os.environ.get("MOLOCH_PI_TUNNEL_URL", "http://localhost:9000").rstrip("/")
OLLAMA_TASK = os.environ.get("MOLOCH_OLLAMA_TASK", "MolochOllama")

HERE = Path(__file__).resolve().parent
ROLES_FILE = HERE / "ki_rollen.json"
PAGE_FILE = HERE / "ki_zentrale.html"
LOG_DIR = Path(os.environ.get("MOLOCH_ZENTRALE_DIR", str(Path.home() / "moloch_zentrale")))
LOG_FILE = LOG_DIR / "chat_log.jsonl"

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}
ENV_KEYS = ("OLLAMA_HOST", "OLLAMA_NUM_GPU", "OLLAMA_KEEP_ALIVE",
            "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_NUM_PARALLEL", "OLLAMA_MODELS")
FALLBACK_PERSONA = (
    "Du bist MOLOCH, ein trockener, kumpelhafter Hausgeist mit Kamera und Mikrofon. "
    "Markus ist dein Boss und Kumpel. Deutsch, 1-3 kurze Saetze, kein Markdown, "
    "keine Assistenten-Floskeln."
)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

router = APIRouter()
_persona = {"text": None, "ts": 0.0}
_chat_lock = asyncio.Lock()  # CPU-only: immer nur ein lokaler Denk-Job


# ---------------------------------------------------------------- Hilfen

def _load_roles() -> dict:
    try:
        return json.loads(ROLES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"haupthirn": "qwen2.5:7b-instruct", "chat_optionen": {}, "modelle": {}}


def _role_of(roles: dict, name: str) -> dict:
    m = roles.get("modelle", {})
    if name in m:
        return m[name]
    alt = name[:-7] if name.endswith(":latest") else name + ":latest"
    return m.get(alt, {})


def _keep(value) -> object:
    """Ollama nimmt Zahlen (Sekunden) oder Go-Dauern ("5m"). "-1" als String waere ungueltig."""
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return value


def _require_local(request: Request) -> Optional[JSONResponse]:
    """Steuer-Endpunkte nur fuer die Zentrale selbst am PC.

    Die Client-IP allein reicht nicht: jede Webseite im Browser des PCs kann
    127.0.0.1:11700 per no-cors-POST erreichen (oder per DNS-Rebinding).
    Deshalb zusaetzlich Host- und Origin-Header pruefen und JSON-Content-Type
    verlangen - der erzwingt einen CORS-Preflight, und den beantwortet die App nicht.
    """
    deny = JSONResponse({"error": "Steuerung nur direkt am PC erlaubt"}, status_code=403)
    if (request.client.host if request.client else "") not in LOCAL_HOSTS:
        return deny
    host_hdr = request.headers.get("host", "").lower()
    if host_hdr.rsplit(":", 1)[0].strip("[]") not in LOCAL_HOSTS:
        return deny
    origin = request.headers.get("origin")
    if origin is not None and origin.lower() != f"http://{host_hdr}":
        return deny
    if not request.headers.get("content-type", "").lower().startswith("application/json"):
        return deny
    return None


def _is_embedding(roles: dict, name: str) -> bool:
    return _role_of(roles, name).get("typ") == "embedding" or "minilm" in name or "embed" in name


async def _get_json(client: httpx.AsyncClient, url: str, timeout: float = 3.0):
    try:
        r = await client.get(url, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _ollama_env() -> dict:
    """Ollama-Variablen aus der Registry (das, was ein neu gestartetes Ollama sieht)."""
    out: dict = {}
    try:
        import winreg
    except ImportError:
        return out
    places = (
        (winreg.HKEY_CURRENT_USER, "Environment", "benutzer"),
        (winreg.HKEY_LOCAL_MACHINE,
         r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", "system"),
    )
    for hive, sub, scope in places:
        try:
            with winreg.OpenKey(hive, sub) as key:
                for name in ENV_KEYS:
                    try:
                        val, _ = winreg.QueryValueEx(key, name)
                        out.setdefault(name, {})[scope] = str(val)
                    except OSError:
                        pass
        except OSError:
            pass
    return out


def _ollama_procs() -> list:
    procs = []
    for p in psutil.process_iter(["pid", "name", "memory_info"]):
        name = (p.info.get("name") or "")
        if name.lower().startswith("ollama"):
            mem = p.info.get("memory_info")
            procs.append({"pid": p.info["pid"], "name": name,
                          "rss_mb": int(mem.rss / 1e6) if mem else None})
    return procs


def _log(entry: dict) -> None:
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _read_log(n: int) -> list:
    if n <= 0 or not LOG_FILE.exists():
        return []
    try:
        lines = LOG_FILE.read_text(encoding="utf-8").splitlines()[-n:]
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _mood(live: Optional[dict]) -> dict:
    if not live:
        return {"online": False}
    core = live.get("core") or {}
    return {
        "online": True,
        "zone": core.get("zone"),
        "tension": core.get("tension"),
        "dominance": core.get("dominance"),
        "presence": core.get("presence"),
        "tageszeit": core.get("time_period"),
        "person": bool(live.get("person_detected")),
        "gesicht": live.get("face_id"),
        "fps": live.get("fps_total"),
    }


def _live_snippet(live: Optional[dict]) -> str:
    m = _mood(live)
    if not m["online"]:
        return "[LIVE] Pi nicht erreichbar - du weisst gerade nicht, was die Kamera sieht."
    parts = [f"Zone={m['zone'] or '?'}"]
    if isinstance(m["tension"], (int, float)):
        parts.append(f"Tension={m['tension']:+.2f}")
    if m["tageszeit"]:
        parts.append(f"Tageszeit={m['tageszeit']}")
    if m["gesicht"]:
        parts.append(f"im Bild erkannt: {m['gesicht']}")
    elif m["person"]:
        parts.append("Person im Bild, Gesicht nicht erkannt")
    else:
        parts.append("niemand im Bild")
    return "[LIVE] " + ", ".join(parts)


async def _persona_text(client: httpx.AsyncClient) -> str:
    if _persona["text"] and time.time() - _persona["ts"] < 600:
        return _persona["text"]
    data = await _get_json(client, f"{PI_URL}/system_prompt", timeout=5.0)
    if data and data.get("system"):
        _persona.update(text=data["system"], ts=time.time())
    return _persona["text"] or FALLBACK_PERSONA


async def _mirror_to_pi(client: httpx.AsyncClient, user_text: str, reply: str) -> bool:
    """Lokalen Turn ins gemeinsame Pi-Gedaechtnis schreiben (POST /memory/turn, Pi b02f0f2).

    Best effort: ist der Pi weg, bleibt der Turn nur im PC-Log.
    """
    try:
        for sender, text in (("markus", user_text), ("moloch", reply)):
            r = await client.post(f"{PI_URL}/memory/turn", timeout=5.0,
                                  json={"sender": sender, "text": text[:4000],
                                        "source": "pc_zentrale"})
            if r.status_code != 200:
                return False
        return True
    except Exception:
        return False


def _line(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


# ---------------------------------------------------------------- Routen

@router.get("/zentrale")
def zentrale_page():
    try:
        return HTMLResponse(PAGE_FILE.read_text(encoding="utf-8"))
    except OSError:
        return HTMLResponse("<h1>ki_zentrale.html fehlt</h1>", status_code=500)


@router.get("/api/ki/status")
async def ki_status():
    roles = _load_roles()
    async with httpx.AsyncClient() as client:
        ver, tags, ps, live = await asyncio.gather(
            _get_json(client, f"{OLLAMA_URL}/api/version"),
            _get_json(client, f"{OLLAMA_URL}/api/tags"),
            _get_json(client, f"{OLLAMA_URL}/api/ps"),
            _get_json(client, f"{PI_URL}/live", timeout=3.0),
        )
    loaded = {m.get("name"): m for m in (ps or {}).get("models", [])}
    brain = roles.get("haupthirn")
    models = []
    for m in (tags or {}).get("models", []):
        name = m.get("name", "")
        info = _role_of(roles, name)
        run = loaded.get(name)
        details = m.get("details") or {}
        models.append({
            "name": name,
            "groesse_gb": round(m.get("size", 0) / 1e9, 1),
            "parameter": details.get("parameter_size"),
            "quant": details.get("quantization_level"),
            "rolle": info.get("rolle", "ohne Rolle"),
            "aufgabe": info.get("aufgabe", ""),
            "haupthirn": name == brain,
            "geladen": run is not None,
            "ram_gb": round(run.get("size", 0) / 1e9, 1) if run else 0,
            "bis": run.get("expires_at") if run else None,
        })
    models.sort(key=lambda x: (not x["haupthirn"], not x["geladen"], x["name"]))
    vm = psutil.virtual_memory()
    try:
        disk = psutil.disk_usage(str(Path.home()))
        disk_free = round(disk.free / 1e9, 1)
    except OSError:
        disk_free = None
    # Prozessliste und Registry blockieren kurz -> aus dem Event-Loop nehmen (Streaming)
    procs, env = await asyncio.gather(asyncio.to_thread(_ollama_procs),
                                      asyncio.to_thread(_ollama_env))
    return {
        "ts": time.time(),
        "ollama": {
            "online": ver is not None,
            "version": (ver or {}).get("version"),
            "url": OLLAMA_URL,
            "prozesse": procs,
            "env": env,
        },
        "haupthirn": brain,
        "haupthirn_installiert": any(m["name"] == brain for m in models),
        "modelle": models,
        "system": {
            "ram_gesamt_gb": round(vm.total / 1e9, 1),
            "ram_belegt_gb": round(vm.used / 1e9, 1),
            "ram_prozent": vm.percent,
            "cpu_prozent": psutil.cpu_percent(interval=None),
            "platte_frei_gb": disk_free,
        },
        "moloch": _mood(live),
        "denkt": _chat_lock.locked(),
    }


@router.post("/api/ki/model")
async def ki_model(request: Request):
    deny = _require_local(request)
    if deny:
        return deny
    body = await request.json()
    model = str(body.get("model") or "")
    action = body.get("action")
    if action not in ("load", "unload"):
        return JSONResponse({"error": "action muss load oder unload sein"}, status_code=400)
    roles = _load_roles()
    async with httpx.AsyncClient() as client:
        tags = await _get_json(client, f"{OLLAMA_URL}/api/tags")
        if tags is None:
            return JSONResponse({"error": "Ollama laeuft nicht"}, status_code=503)
        if model not in {m.get("name") for m in tags.get("models", [])}:
            return JSONResponse({"error": f"{model} ist nicht installiert"}, status_code=404)
        keep = _keep(_role_of(roles, model).get("keep_alive", "5m")) if action == "load" else 0
        # Embedding-Modelle koennen kein /api/generate
        if _is_embedding(roles, model):
            url, payload = f"{OLLAMA_URL}/api/embed", {"model": model, "input": "", "keep_alive": keep}
        else:
            url, payload = f"{OLLAMA_URL}/api/generate", {"model": model, "prompt": "", "keep_alive": keep}
        t0 = time.time()
        try:
            r = await client.post(url, json=payload, timeout=240.0)
        except Exception as e:
            return JSONResponse({"error": f"Ollama: {str(e)[:150]}"}, status_code=502)
    ok = r.status_code == 200
    return {"ok": ok, "model": model, "action": action,
            "sekunden": round(time.time() - t0, 1),
            "error": None if ok else r.text[:200]}


@router.post("/api/ki/ollama")
async def ki_ollama(request: Request):
    deny = _require_local(request)
    if deny:
        return deny
    action = (await request.json()).get("action")
    if action == "start":
        async with httpx.AsyncClient() as client:
            if await _get_json(client, f"{OLLAMA_URL}/api/version", timeout=1.5):
                return {"ok": True, "info": "Ollama laeuft schon"}
        res = await asyncio.to_thread(
            subprocess.run, ["schtasks", "/run", "/tn", OLLAMA_TASK],
            capture_output=True, text=True, encoding="oem", errors="replace",
            creationflags=_NO_WINDOW, timeout=20)
        if res.returncode != 0:
            return JSONResponse({"error": (res.stderr or res.stdout).strip()[:200]}, status_code=500)
        async with httpx.AsyncClient() as client:
            for _ in range(30):
                if await _get_json(client, f"{OLLAMA_URL}/api/version", timeout=1.0):
                    return {"ok": True, "info": "Ollama gestartet"}
                await asyncio.sleep(1)
        return JSONResponse({"error": "Task gestartet, Ollama antwortet nach 30 s noch nicht"},
                            status_code=504)
    if action == "stop":
        killed = []
        for p in psutil.process_iter(["pid", "name"]):
            name = (p.info.get("name") or "").lower()
            if name.startswith("ollama"):
                try:
                    p.kill()
                    killed.append(p.info["pid"])
                except psutil.Error:
                    pass
        return {"ok": True, "info": f"{len(killed)} Ollama-Prozess(e) beendet"}
    return JSONResponse({"error": "action muss start oder stop sein"}, status_code=400)


@router.get("/api/ki/verlauf")
def ki_verlauf(n: int = 60):
    return {"messages": _read_log(max(0, min(n, 500)))}


@router.post("/api/ki/chat")
async def ki_chat(request: Request):
    body = await request.json()
    msg = (body.get("message") or "").strip()[:2000]
    mode = body.get("mode", "local")
    if not msg:
        return JSONResponse({"error": "leere Nachricht"}, status_code=400)

    if mode == "pi":
        t0 = time.time()
        async with httpx.AsyncClient() as client:
            try:
                r = await client.post(f"{PI_URL}/chat", json={"text": msg}, timeout=120.0)
                data = r.json() if r.status_code == 200 else {"error": f"Pi {r.status_code}"}
            except Exception as e:
                data = {"error": f"Pi nicht erreichbar: {str(e)[:120]}"}
        secs = round(time.time() - t0, 1)
        text = (data.get("text") or "").strip()
        _log({"who": "markus", "text": msg, "mode": "pi"})
        if text:
            _log({"who": "moloch", "text": text, "mode": "pi",
                  "model": data.get("provider"), "secs": secs})
        return {"text": text, "provider": data.get("provider"),
                "secs": secs, "error": data.get("error")}

    if _chat_lock.locked():
        return JSONResponse({"error": "Moloch denkt noch an der letzten Antwort"}, status_code=409)

    roles = _load_roles()
    model = roles.get("haupthirn") or "qwen2.5:7b-instruct"
    opts = roles.get("chat_optionen") or {}
    turns = int(opts.get("verlauf_turns", 6))

    async def stream():
        async with _chat_lock:
            async with httpx.AsyncClient() as client:
                persona, live = await asyncio.gather(
                    _persona_text(client),
                    _get_json(client, f"{PI_URL}/live", timeout=3.0),
                )
                # Fester Anfang (Persona + Verlauf), wechselnder Live-Kontext ans Ende:
                # so kann Ollama den verarbeiteten Praefix wiederverwenden.
                messages = [{"role": "system", "content": persona}]
                for e in _read_log(turns * 2):
                    role = "user" if e.get("who") == "markus" else "assistant"
                    if e.get("text"):
                        messages.append({"role": role, "content": e["text"]})
                messages.append({"role": "user", "content": f"{msg}\n\n{_live_snippet(live)}"})
                _log({"who": "markus", "text": msg, "mode": "local"})

                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "keep_alive": _keep(_role_of(roles, model).get("keep_alive", "-1")),
                    "options": {
                        "num_ctx": int(opts.get("num_ctx", 4096)),
                        "num_predict": int(opts.get("num_predict", 160)),
                        "temperature": float(opts.get("temperature", 0.7)),
                    },
                }
                t0 = time.time()
                first = None
                text = ""
                stats: dict = {}
                timeout = httpx.Timeout(connect=5.0, read=240.0, write=10.0, pool=5.0)
                try:
                    async with client.stream("POST", f"{OLLAMA_URL}/api/chat",
                                             json=payload, timeout=timeout) as r:
                        if r.status_code != 200:
                            err = (await r.aread()).decode("utf-8", "replace")[:200]
                            yield _line({"error": f"Ollama {r.status_code}: {err}"})
                            return
                        async for raw in r.aiter_lines():
                            if not raw.strip():
                                continue
                            d = json.loads(raw)
                            if d.get("error"):
                                yield _line({"error": d["error"]})
                                return
                            piece = (d.get("message") or {}).get("content", "")
                            if piece:
                                if first is None:
                                    first = time.time() - t0
                                text += piece
                                yield _line({"t": piece})
                            if d.get("done"):
                                stats = d
                except httpx.ConnectError:
                    yield _line({"error": "Ollama laeuft nicht - oben rechts starten"})
                    return
                except Exception as e:
                    yield _line({"error": f"{type(e).__name__}: {str(e)[:150]}"})
                    return

                text = text.strip()
                secs = round(time.time() - t0, 1)
                mirrored = False
                if text:
                    _log({"who": "moloch", "text": text, "mode": "local",
                          "model": model, "secs": secs})
                    mirrored = await _mirror_to_pi(client, msg, text)
                ev_n, ev_d = stats.get("eval_count"), stats.get("eval_duration")
                yield _line({
                    "done": True,
                    "gedaechtnis": mirrored,
                    "model": model,
                    "secs": secs,
                    "erstes_wort_s": round(first, 1) if first is not None else None,
                    "tok_s": round(ev_n / (ev_d / 1e9), 1) if ev_n and ev_d else None,
                    "laden_s": round(stats.get("load_duration", 0) / 1e9, 1),
                })

    return StreamingResponse(stream(), media_type="application/x-ndjson")
