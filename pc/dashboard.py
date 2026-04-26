"""MOLOCH Cowork Dashboard — PC-Side overview of both AIs.

Single auto-refreshing HTML page on :11700 showing:
- PC adapter (Qwen + LoRA) on :11600
- Pi-Side cockpit live status (via SSH-Tunnel localhost:9000 -> Pi:9100)
- Sample pool counts + training status
- Persoenlichkeits-Drift-Snapshot
"""
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

PORT = int(os.environ.get("MOLOCH_DASHBOARD_PORT", "11700"))
PROXY_URL = os.environ.get("MOLOCH_PROXY_URL", "http://localhost:11600")
PI_TUNNEL_URL = os.environ.get("MOLOCH_PI_TUNNEL_URL", "http://localhost:9000")
ADAPTERS_DIR = Path(
    os.environ.get("MOLOCH_ADAPTERS", str(Path.home() / "moloch_adapters"))
)
TRAINING_STATUS_FILE = ADAPTERS_DIR / "training_status.json"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


app = FastAPI(title="MOLOCH Dashboard", lifespan=lifespan)


async def _safe_get(client: httpx.AsyncClient, url: str, timeout: float = 2.0):
    t0 = time.time()
    try:
        r = await client.get(url, timeout=timeout)
        dt = int((time.time() - t0) * 1000)
        if r.status_code != 200:
            return None, dt
        try:
            return r.json(), dt
        except Exception:
            return {"_text": r.text[:200]}, dt
    except Exception:
        return None, int((time.time() - t0) * 1000)


def _list_adapters() -> list[str]:
    if not ADAPTERS_DIR.exists():
        return []
    found = [
        d.name for d in ADAPTERS_DIR.iterdir()
        if d.is_dir() and re.match(r"v\d+$", d.name)
    ]
    return sorted(found, key=lambda v: int(v[1:]))


def _read_json_file(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@app.get("/api/state")
async def api_state():
    async with httpx.AsyncClient() as client:
        pc_health, pc_dt = await _safe_get(client, f"{PROXY_URL}/health")
        pc_list, _ = await _safe_get(client, f"{PROXY_URL}/list")
        pi_live, pi_dt = await _safe_get(client, f"{PI_TUNNEL_URL}/live")
        pi_personality, _ = await _safe_get(client, f"{PI_TUNNEL_URL}/personality")
        pi_status, _ = await _safe_get(client, f"{PI_TUNNEL_URL}/status")
        pi_pool, _ = await _safe_get(client, f"{PI_TUNNEL_URL}/feedback_stats")

    adapters = _list_adapters()
    latest = adapters[-1] if adapters else None
    latest_log = _read_json_file(ADAPTERS_DIR / latest / "training_log.json") if latest else None
    training = _read_json_file(TRAINING_STATUS_FILE)

    return {
        "ts": time.time(),
        "pc_proxy": {
            "url": PROXY_URL,
            "online": pc_health is not None,
            "latency_ms": pc_dt,
            "health": pc_health,
            "list": pc_list,
        },
        "adapters": {
            "versions": adapters,
            "latest": latest,
            "latest_log": latest_log,
        },
        "training": training,
        "pi": {
            "url": PI_TUNNEL_URL,
            "online": pi_live is not None,
            "latency_ms": pi_dt,
            "live": pi_live,
            "personality": pi_personality,
            "status": pi_status,
            "pool": pi_pool,
        },
    }


HTML_PAGE = """<!doctype html>
<html lang="de"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOLOCH Cowork Dashboard</title>
<style>
:root{--bg:#0a0a0d;--fg:#e6e6ee;--mute:#7a7a8a;--card:#13131a;
      --border:#26262f;--ok:#5dc36b;--warn:#e6b84d;--err:#ff7676;
      --pi:#3673ce;--pc:#7e3bce;--comm:#c93838}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;padding:12px}
h1{font:600 16px system-ui;margin-bottom:10px;letter-spacing:1px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;max-width:1400px;margin:auto}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px;min-width:0}
.card h2{font:600 12px system-ui;color:var(--mute);text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px}
.card.pi h2{color:var(--pi)}.card.pc h2{color:var(--pc)}.card.comm h2{color:var(--comm)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:12.5px}
.kv .k{color:var(--mute);white-space:nowrap}
.kv .v{font-variant-numeric:tabular-nums;text-align:right;word-break:break-all}
.kv .v.ok{color:var(--ok)}.kv .v.err{color:var(--err)}.kv .v.warn{color:var(--warn)}
.full{grid-column:1/-1}
pre{font:11px/1.4 monospace;color:var(--mute);max-height:240px;overflow:auto;
    white-space:pre-wrap;background:#0a0a0d;padding:6px;border-radius:4px;border:1px solid var(--border)}
.pulse{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:middle}
.pulse.ok{background:var(--ok);box-shadow:0 0 6px var(--ok);animation:pulse 1.5s infinite}
.pulse.err{background:var(--err)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
footer{margin-top:10px;color:var(--mute);font-size:11px;text-align:center}
.bar{display:flex;gap:6px;align-items:center;font-size:12px;margin-top:4px}
.bar .arrow{color:var(--mute)}
</style></head><body>
<h1>🎛 MOLOCH COWORK DASHBOARD</h1>
<div class="grid">
  <div class="card pc"><h2><span class="pulse" id="pc-pulse"></span>PC ADAPTER (Qwen 1.5B + LoRA, :11600)</h2>
    <div class="kv" id="pc-kv">…</div>
  </div>
  <div class="card pi"><h2><span class="pulse" id="pi-pulse"></span>Pi LIVE (NPU + Sense, :9100 via Tunnel)</h2>
    <div class="kv" id="pi-kv">…</div>
  </div>
  <div class="card pc full"><h2>ADAPTER POOL & TRAINING</h2>
    <div class="kv" id="train-kv">…</div>
  </div>
  <div class="card pi"><h2>Pi PERSOENLICHKEIT (Drift)</h2>
    <div class="kv" id="pers-kv">…</div>
  </div>
  <div class="card comm"><h2>SAMPLE-POOL (Pi /feedback_stats)</h2>
    <div class="kv" id="pool-kv">…</div>
  </div>
  <div class="card full"><h2>RAW STATE (debug, klick Header zum collapsen)</h2>
    <details><summary style="cursor:pointer;color:#7a7a8a;font-size:11px">show raw</summary>
    <pre id="raw">…</pre></details>
  </div>
</div>
<footer>Auto-refresh 5s — last: <span id="ts">--</span></footer>
<script>
const $=id=>document.getElementById(id);
function fmt(v){return (v===null||v===undefined||v==='')?'—':String(v)}
function fix(v,n){return (typeof v==='number')?v.toFixed(n):'—'}
function setKv(elId, pairs){
  const el=$(elId); el.innerHTML='';
  for(const [k,v,cls] of pairs){
    const dk=document.createElement('div');dk.className='k';dk.textContent=k;
    const dv=document.createElement('div');dv.className='v'+(cls?' '+cls:'');dv.textContent=fmt(v);
    el.appendChild(dk);el.appendChild(dv);
  }
}
async function tick(){
  try{
    const s=await(await fetch('/api/state')).json();
    $('pc-pulse').className='pulse '+(s.pc_proxy.online?'ok':'err');
    $('pi-pulse').className='pulse '+(s.pi.online?'ok':'err');

    const ph=s.pc_proxy.health||{};
    const lst=s.pc_proxy.list||{};
    setKv('pc-kv',[
      ['Status', s.pc_proxy.online?'online':'OFFLINE', s.pc_proxy.online?'ok':'err'],
      ['Active Adapter', ph.adapter||'(none)'],
      ['Base Model', ph.base],
      ['Versions vorhanden', (lst.adapters||[]).join(', ')||'—'],
      ['Latency', s.pc_proxy.latency_ms+' ms'],
    ]);

    const pl=s.pi.live||{};
    const wd=pl.watchdog||{};
    setKv('pi-kv',[
      ['Status', s.pi.online?'online':'OFFLINE', s.pi.online?'ok':'err'],
      ['FPS', fix(pl.fps_total,1)],
      ['Person/Face', (pl.person_detected?'P':'·')+'/'+(pl.face_detected?'F':'·')],
      ['Face-ID', pl.face_id||'—'],
      ['Face-Conf', fix(pl.face_confidence,2)],
      ['CPU', fix(wd.cpu_temp,1)+'°C'],
      ['RAM', fix(wd.ram_percent,0)+'%'],
      ['Tunnel-Latency', s.pi.latency_ms+' ms'],
    ]);

    const log=s.adapters.latest_log||{};
    const tr=s.training;
    const trKv=[
      ['Versions', (s.adapters.versions||[]).join(', ')||'(noch keine)'],
      ['Latest', s.adapters.latest||'—'],
    ];
    if(log && log.final_loss!==undefined){
      trKv.push(['Last Loss', fix(log.final_loss,4)]);
      trKv.push(['Last Duration', fix(log.duration_seconds,1)+' s']);
      const raw=log.samples_used_raw||log.samples_used;
      const eff=log.samples_used_effective||log.samples_used;
      trKv.push(['Last Samples', raw+' raw → '+eff+' effective']);
      if(log.sample_weights) trKv.push(['Last Weights', JSON.stringify(log.sample_weights)]);
    }
    if(tr){
      trKv.push(['LIVE Training', tr.state||'running', 'warn']);
      trKv.push(['Step', (tr.step||'?')+' / '+(tr.total_steps||'?')]);
      if(tr.loss!==undefined && tr.loss!==null) trKv.push(['Loss (curr)', fix(tr.loss,4)]);
      if(tr.eta_seconds!==undefined) trKv.push(['ETA', fix(tr.eta_seconds,0)+' s']);
      if(tr.elapsed_seconds!==undefined) trKv.push(['Elapsed', fix(tr.elapsed_seconds,0)+' s']);
    } else {
      trKv.push(['LIVE Training', 'idle', 'ok']);
    }
    setKv('train-kv', trKv);

    const core=pl.core||{};
    setKv('pers-kv',[
      ['Tension', fix(core.tension,3)],
      ['Dominance', fix(core.dominance,3)],
      ['Zone', core.zone],
      ['Presence', fix(core.presence,2)],
      ['Time-Period', core.time_period],
      ['Berserker', core.berserker_active?'YES':'no', core.berserker_active?'err':'ok'],
      ['Owner-Confirmed', core.owner_confirmed?'YES':'no', core.owner_confirmed?'ok':'warn'],
    ]);

    const pool=s.pi.pool||{};
    setKv('pool-kv',[
      ['Total', pool.total],
      ['Approved', pool.approved, 'ok'],
      ['Pending', pool.pending, 'warn'],
      ['Rejected', pool.rejected],
      ['Critic', pool.critic],
      ['Thumbs Up', pool.thumbs_up],
      ['Thumbs Down', pool.thumbs_down],
    ]);

    $('raw').textContent=JSON.stringify(s,null,2);
    $('ts').textContent=new Date().toLocaleTimeString();
  }catch(e){
    $('ts').textContent='ERR: '+e.message;
  }
}
tick(); setInterval(tick, 5000);
</script></body></html>
"""


@app.get("/")
def root():
    return HTMLResponse(HTML_PAGE)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
