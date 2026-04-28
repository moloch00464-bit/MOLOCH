"""MOLOCH Cowork Dashboard — PC-Side overview of both AIs.

Single auto-refreshing HTML page on :11700 showing:
- PC adapter (Qwen + LoRA) on :11600
- Pi-Side cockpit live status (via SSH-Tunnel localhost:9000 -> Pi:9100)
- Sample pool counts + live trend (last 60min)
- Training status (live step/loss/eta wenn lora_trainer laeuft)
- Identity panel (live system_prompt + drift baselines + active rules)
- Persoenlichkeits-Drift-Snapshot
"""
import json
import os
import re
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

PORT = int(os.environ.get("MOLOCH_DASHBOARD_PORT", "11700"))
PROXY_URL = os.environ.get("MOLOCH_PROXY_URL", "http://localhost:11600")
PI_TUNNEL_URL = os.environ.get("MOLOCH_PI_TUNNEL_URL", "http://localhost:9000")
ADAPTERS_DIR = Path(
    os.environ.get("MOLOCH_ADAPTERS", str(Path.home() / "moloch_adapters"))
)
TRAINING_STATUS_FILE = ADAPTERS_DIR / "training_status.json"

# In-memory rolling pool snapshots — at 5s polling that's ~60 min of history.
POOL_HISTORY: deque = deque(maxlen=720)


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


def _record_pool(pool: Optional[dict]) -> None:
    if not pool:
        return
    POOL_HISTORY.append({
        "ts": time.time(),
        "total": pool.get("total"),
        "critic": pool.get("critic"),
        "pending": pool.get("pending_review", pool.get("pending")),
        "approved": pool.get("approved"),
    })


@app.get("/api/state")
async def api_state():
    async with httpx.AsyncClient() as client:
        pc_health, pc_dt = await _safe_get(client, f"{PROXY_URL}/health")
        pc_list, _ = await _safe_get(client, f"{PROXY_URL}/list")
        pi_live, pi_dt = await _safe_get(client, f"{PI_TUNNEL_URL}/live")
        pi_personality, _ = await _safe_get(client, f"{PI_TUNNEL_URL}/personality")
        pi_status, _ = await _safe_get(client, f"{PI_TUNNEL_URL}/status")
        pi_pool, _ = await _safe_get(client, f"{PI_TUNNEL_URL}/feedback_stats")
        pi_prompt, _ = await _safe_get(client, f"{PI_TUNNEL_URL}/system_prompt", timeout=4.0)

    _record_pool(pi_pool)

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
            "system_prompt": pi_prompt,
        },
        "pool_history": list(POOL_HISTORY),
    }


@app.post("/api/chat")
async def api_chat(request: Request):
    body = await request.json()
    msg = (body.get("message") or "").strip()
    if not msg:
        return {"error": "empty message"}
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                f"{PI_TUNNEL_URL}/chat",
                json={"text": msg},
                timeout=30.0,
            )
            if r.status_code == 200:
                return r.json()
            return {"error": f"Pi {r.status_code}", "raw": r.text[:200]}
        except Exception as e:
            return {"error": str(e)[:120]}


HTML_PAGE = """<!doctype html>
<html lang="de"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOLOCH Cowork Dashboard</title>
<style>
:root{--bg:#0a0a0d;--fg:#e6e6ee;--mute:#7a7a8a;--card:#13131a;
      --border:#26262f;--ok:#5dc36b;--warn:#e6b84d;--err:#ff7676;
      --pi:#3673ce;--pc:#7e3bce;--comm:#c93838;--ident:#56b3a8}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;padding:12px}
h1{font:600 16px system-ui;margin-bottom:10px;letter-spacing:1px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;max-width:1500px;margin:auto}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px;min-width:0}
.card h2{font:600 12px system-ui;color:var(--mute);text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px}
.card.pi h2{color:var(--pi)}.card.pc h2{color:var(--pc)}.card.comm h2{color:var(--comm)}.card.ident h2{color:var(--ident)}
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
.legend{display:flex;gap:14px;font-size:11px;margin:6px 0 4px;color:var(--mute);flex-wrap:wrap}
.legend span{display:inline-flex;align-items:center;gap:4px}
.legend i{display:inline-block;width:12px;height:3px;background:#fff;border-radius:2px}
.prompt-box{font:11.5px/1.45 monospace;color:#cfd0e0;background:#0a0a0d;padding:8px;
  border-radius:4px;border:1px solid var(--border);max-height:220px;overflow-y:auto;
  white-space:pre-wrap;word-break:break-word}
.rule{padding:5px 7px;background:#1a1c23;border-left:3px solid var(--ident);
  border-radius:0 4px 4px 0;margin-bottom:4px;font-size:12px}
.rule .t{color:var(--mute);font-size:11px}.rule .b{color:var(--fg);font-style:italic}
.tag{display:inline-block;padding:1px 6px;background:var(--border);border-radius:10px;
  font-size:10.5px;color:var(--mute);margin-right:4px}
.chat-log{font:12px/1.6 monospace;background:#0a0a0d;border:1px solid var(--border);
  border-radius:4px;padding:8px;height:220px;overflow-y:auto}
.chat-msg{margin-bottom:5px}
.chat-msg .who{font-weight:600}
.chat-msg.markus .who{color:var(--warn)}
.chat-msg.moloch .who{color:var(--ok)}
.chat-msg.system .who{color:var(--mute)}
.chat-msg .body{color:var(--fg)}
.chat-row{display:flex;gap:6px;margin-top:8px;align-items:center}
.chat-row input{flex:1;background:#13131a;border:1px solid var(--border);color:var(--fg);
  padding:6px 10px;border-radius:4px;font:13px system-ui;outline:none}
.chat-row input:focus{border-color:#444}
.chat-row button{padding:6px 14px;background:#1a3a1a;border:1px solid #2a4a2a;
  color:var(--ok);border-radius:4px;cursor:pointer;font:12px system-ui;white-space:nowrap}
.chat-row button:hover{border-color:var(--ok)}
#mic-btn{background:#1a2a3a;border-color:#2a4a6a;color:#6699ee;padding:6px 10px;font-size:16px}
#mic-btn.active{background:#3a1414;border-color:var(--err);color:var(--err);animation:pulse-mic .7s infinite}
@keyframes pulse-mic{0%,100%{opacity:1}50%{opacity:.5}}
.tts-bar{display:flex;gap:10px;align-items:center;margin-bottom:8px;font-size:12px;flex-wrap:wrap}
.tts-toggle{padding:3px 12px;border-radius:10px;border:1px solid;cursor:pointer;font:11px system-ui}
.tts-toggle.on{background:#1a3a1a;border-color:var(--ok);color:var(--ok)}
.tts-toggle.off{background:#2a1414;border-color:var(--err);color:var(--err)}
.tts-voice{color:var(--mute);font-size:11px}
.zone-pill{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600}
.zone-pill.guardian{background:#1a2a4a;color:#6699ee}
.zone-pill.shadow{background:#2a1a4a;color:#aa88ee}
.zone-pill.berserker{background:#4a1a1a;color:#ff8888}
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

  <div class="card comm full"><h2>SAMPLE-POOL TREND (rolling, max 60 min)</h2>
    <div class="legend">
      <span><i style="background:#cfd0e0"></i>total</span>
      <span><i style="background:#5dc36b"></i>approved</span>
      <span><i style="background:#e6b84d"></i>pending</span>
      <span><i style="background:#7e3bce"></i>critic</span>
    </div>
    <svg id="pool-svg" width="100%" height="140" preserveAspectRatio="none" viewBox="0 0 1000 140" style="background:#0a0a0d;border:1px solid var(--border);border-radius:4px"></svg>
    <div class="kv" id="pool-kv" style="margin-top:8px">…</div>
  </div>

  <div class="card pi"><h2>Pi PERSOENLICHKEIT (Drift)</h2>
    <div class="kv" id="pers-kv">…</div>
  </div>

  <div class="card ident"><h2>IDENTITY (live system_prompt + active rules)</h2>
    <div class="kv" id="ident-kv" style="margin-bottom:8px">…</div>
    <div id="ident-rules"></div>
    <details style="margin-top:8px">
      <summary style="cursor:pointer;color:var(--mute);font-size:11px">Live System-Prompt anzeigen (3.5k chars)</summary>
      <div class="prompt-box" id="ident-prompt">…</div>
    </details>
  </div>

  <div class="card pi" style="grid-column:1/-1;display:grid;grid-template-columns:320px 1fr;gap:10px;align-items:start">
    <div>
      <h2 style="color:var(--pi);font:600 12px system-ui;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px">MOLOCH AVATAR (live)</h2>
      <iframe src="http://localhost:11800" style="width:320px;height:320px;border:1px solid var(--border);border-radius:6px;background:#08080c" scrolling="no" title="Moloch Avatar"></iframe>
    </div>
    <div style="display:flex;flex-direction:column;gap:8px">
      <h2 style="color:var(--comm);font:600 12px system-ui;text-transform:uppercase;letter-spacing:.5px">ZONE / STIMMUNG</h2>
      <div class="kv" id="mood-kv">…</div>
    </div>
  </div>

  <div class="card comm full"><h2>CHAT + SPRACHAUSGABE</h2>
    <div class="tts-bar">
      <button class="tts-toggle on" id="tts-btn" onclick="toggleTTS()">TTS AN</button>
      <span class="zone-pill guardian" id="tts-zone">guardian</span>
      <span class="tts-voice" id="tts-params">pitch=1.00 rate=0.95 vol=0.80</span>
      <span id="tts-speaking" style="color:var(--ok);font-size:11px;display:none">&#9654; spricht...</span>
    </div>
    <div class="chat-log" id="chat-log">
      <div class="chat-msg system"><span class="who">SYS:</span> <span class="body" style="color:var(--mute)">Chat bereit — Moloch-Antworten werden vorgelesen</span></div>
    </div>
    <div class="chat-row">
      <button id="mic-btn" onclick="toggleMic()" title="Sprechen (de-DE)">🎤</button>
      <input type="text" id="chat-input" placeholder="Sprechen oder tippen..."
             onkeydown="if(event.key==='Enter')sendChat()">
      <button onclick="sendChat()" id="chat-send">Senden</button>
    </div>
    <div id="mic-status" style="font-size:11px;color:var(--mute);margin-top:4px;min-height:16px"></div>
  </div>

  <div class="card full"><h2>RAW STATE (debug)</h2>
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

function drawPoolChart(history){
  const svg=$('pool-svg');
  svg.innerHTML='';
  if(!history || history.length<2) return;
  const W=1000, H=140, PAD=8;
  // find max for normalization
  let maxV=0;
  for(const h of history){
    maxV=Math.max(maxV, h.total||0, h.approved||0, h.pending||0, h.critic||0);
  }
  if(maxV<=0) maxV=1;
  const t0=history[0].ts, tN=history[history.length-1].ts;
  const dt=Math.max(1, tN-t0);
  function x(t){return PAD + (W-2*PAD) * (t-t0)/dt}
  function y(v){return H-PAD - (H-2*PAD) * ((v||0)/maxV)}
  const series=[
    {key:'total',color:'#cfd0e0',width:1.5},
    {key:'critic',color:'#7e3bce',width:1.5},
    {key:'pending',color:'#e6b84d',width:1.5},
    {key:'approved',color:'#5dc36b',width:2},
  ];
  // grid lines
  const NS='http://www.w3.org/2000/svg';
  for(let i=0;i<=4;i++){
    const yy=PAD + (H-2*PAD)*i/4;
    const ln=document.createElementNS(NS,'line');
    ln.setAttribute('x1',PAD); ln.setAttribute('x2',W-PAD);
    ln.setAttribute('y1',yy); ln.setAttribute('y2',yy);
    ln.setAttribute('stroke','#26262f'); ln.setAttribute('stroke-width','1');
    svg.appendChild(ln);
  }
  // y-max label
  const lbl=document.createElementNS(NS,'text');
  lbl.setAttribute('x',W-PAD-2); lbl.setAttribute('y',PAD+10);
  lbl.setAttribute('fill','#7a7a8a'); lbl.setAttribute('font-size','10');
  lbl.setAttribute('text-anchor','end'); lbl.textContent='max '+maxV;
  svg.appendChild(lbl);
  // window label (minutes)
  const winLbl=document.createElementNS(NS,'text');
  winLbl.setAttribute('x',PAD); winLbl.setAttribute('y',H-3);
  winLbl.setAttribute('fill','#7a7a8a'); winLbl.setAttribute('font-size','10');
  winLbl.textContent='window '+Math.round(dt/60)+' min, '+history.length+' samples';
  svg.appendChild(winLbl);
  // lines
  for(const s of series){
    let d='';
    for(let i=0;i<history.length;i++){
      const px=x(history[i].ts), py=y(history[i][s.key]);
      d += (i===0?'M ':'L ') + px.toFixed(1) + ' ' + py.toFixed(1) + ' ';
    }
    const path=document.createElementNS(NS,'path');
    path.setAttribute('d',d); path.setAttribute('fill','none');
    path.setAttribute('stroke',s.color); path.setAttribute('stroke-width',s.width);
    svg.appendChild(path);
  }
}

// === SHARED STATE ===
let ttsEnabled = true;
let _zone = 'guardian', _tension = 0, _dominance = 0;
let _ttsActive = false;

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

    drawPoolChart(s.pool_history||[]);
    const pool=s.pi.pool||{};
    setKv('pool-kv',[
      ['Total', pool.total],
      ['Approved', pool.approved, 'ok'],
      ['Pending', (pool.pending_review!==undefined?pool.pending_review:pool.pending), 'warn'],
      ['Rejected', pool.rejected],
      ['Critic', pool.critic],
      ['Thumbs Up', pool.thumbs_up],
      ['Thumbs Down', pool.thumbs_down],
    ]);

    const core=pl.core||{};
    _zone      = core.zone     || 'guardian';
    _tension   = typeof core.tension   === 'number' ? core.tension   : 0;
    _dominance = typeof core.dominance === 'number' ? core.dominance : 0;
    if (!_ttsActive) _updateTtsDisplay(_zone, _voiceParams(_zone, _tension, _dominance));

    const zoneColMap = {guardian:'#3673ce', shadow:'#7e3bce', berserker:'#c93838'};
    const zoneCol = zoneColMap[_zone] || '#3673ce';
    const moodDesc = {
      guardian: _tension > 0.4 ? 'Wachsam, leicht angespannt' : _tension < -0.2 ? 'Entspannt, ruhig' : 'Ruhig, aufmerksam',
      shadow:   _tension < -0.3 ? 'Zurückgezogen, still' : 'Beobachtend, kühl',
      berserker:'AUFGEWÜHLT — hohe Spannung',
    }[_zone] || '—';
    const faceStr2 = pl.face_id && pl.face_confidence > 0.4
      ? `${pl.face_id} (${Math.round((pl.face_confidence||0)*100)}%)`
      : (pl.person_detected ? 'Person (unklar)' : 'Niemand');
    setKv('mood-kv',[
      ['Zone',      `${_zone.toUpperCase()}`, _zone==='berserker'?'err':_zone==='shadow'?'warn':'ok'],
      ['Stimmung',  moodDesc],
      ['Tension',   ((_tension>=0?'+':'')+_tension.toFixed(3))],
      ['Dominance', ((_dominance>=0?'+':'')+_dominance.toFixed(3))],
      ['Presence',  fix(core.presence,2)],
      ['Berserker', core.berserker_active?'AKTIV':'nein', core.berserker_active?'err':'ok'],
      ['Gesicht',   faceStr2],
      ['FPS',       fix(pl.fps_total,1)],
    ]);
    setKv('pers-kv',[
      ['Tension', fix(core.tension,3)],
      ['Dominance', fix(core.dominance,3)],
      ['Zone', core.zone],
      ['Presence', fix(core.presence,2)],
      ['Time-Period', core.time_period],
      ['Berserker', core.berserker_active?'YES':'no', core.berserker_active?'err':'ok'],
      ['Owner-Confirmed', core.owner_confirmed?'YES':'no', core.owner_confirmed?'ok':'warn'],
    ]);

    // Identity card
    const sp=s.pi.system_prompt||{};
    const pers=s.pi.personality||{};
    const drift=(pers.drift||{}).rolling||{};
    const patch=(pers.patch||{}).state||{};
    const active=(pers.patch||{}).active||[];
    setKv('ident-kv',[
      ['Prompt-Length', sp.length?sp.length+' chars':'—'],
      ['Drift Mood-Baseline', fix(drift.mood_baseline,3)],
      ['Drift Energy-Baseline', fix(drift.energy_baseline,3)],
      ['Drift Dominance-Baseline', fix(drift.dominance_baseline,3)],
      ['Behavior-Rules active', patch.active_count],
      ['Behavior-Rules pending', patch.pending_count],
      ['Behavior-Rules rejected', patch.rejected_count],
    ]);
    const rulesEl=$('ident-rules'); rulesEl.innerHTML='';
    for(const r of active){
      const div=document.createElement('div'); div.className='rule';
      const t=document.createElement('div'); t.className='t';
      t.textContent='Trigger: '+r.trigger;
      const b=document.createElement('div'); b.className='b';
      b.textContent='-> '+r.behavior;
      div.appendChild(t); div.appendChild(b); rulesEl.appendChild(div);
    }
    if(active.length===0){
      const empty=document.createElement('div'); empty.className='tag';
      empty.textContent='no active rules'; rulesEl.appendChild(empty);
    }
    $('ident-prompt').textContent = sp.system || '(no system_prompt fetched)';

    $('raw').textContent=JSON.stringify(s,null,2);
    $('ts').textContent=new Date().toLocaleTimeString();
  }catch(e){
    $('ts').textContent='ERR: '+e.message;
  }
}
tick(); setInterval(tick, 5000);

// === MIKROFON (STT) ===
const _SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let _recognition = null, _micActive = false;

if (_SR) {
  _recognition = new _SR();
  _recognition.lang = 'de-DE';
  _recognition.continuous = false;
  _recognition.interimResults = true;

  _recognition.onstart = () => {
    $('mic-status').textContent = '🔴 Höre zu...';
  };
  _recognition.onresult = (e) => {
    const r = e.results[e.results.length - 1];
    const txt = r[0].transcript;
    $('chat-input').value = txt;
    $('mic-status').textContent = (r.isFinal ? '✓ ' : '…') + txt;
    if (r.isFinal) {
      _micActive = false;
      _updateMicBtn();
      sendChat();
    }
  };
  _recognition.onerror = (e) => {
    $('mic-status').textContent = 'Mikrofon-Fehler: ' + e.error;
    _micActive = false; _updateMicBtn();
  };
  _recognition.onend = () => {
    if (_micActive) { _micActive = false; _updateMicBtn(); }
    if (!$('chat-input').value) $('mic-status').textContent = '';
  };
} else {
  setTimeout(() => {
    if ($('mic-btn')) {
      $('mic-btn').title = 'SpeechRecognition nicht verfügbar — Chrome verwenden';
      $('mic-btn').style.opacity = '0.4';
    }
  }, 500);
}

function _updateMicBtn() {
  const btn = $('mic-btn');
  if (!btn) return;
  if (_micActive) {
    btn.textContent = '⏹';
    btn.className = 'active';
  } else {
    btn.textContent = '🎤';
    btn.className = '';
  }
}

function toggleMic() {
  if (!_recognition) {
    $('mic-status').textContent = 'Kein SpeechRecognition — Chrome verwenden';
    return;
  }
  if (_micActive) {
    _recognition.stop();
    _micActive = false;
  } else {
    // TTS stoppen wenn Mikro startet — nicht sprechen waehrend zuhoeren
    if (window.speechSynthesis) speechSynthesis.cancel();
    $('tts-speaking').style.display = 'none';
    _ttsActive = false;
    $('chat-input').value = '';
    $('mic-status').textContent = '';
    _recognition.start();
    _micActive = true;
  }
  _updateMicBtn();
}

// === TTS + CHAT ===
if (window.speechSynthesis) {
  window.speechSynthesis.onvoiceschanged = () => {};
}

function toggleTTS() {
  ttsEnabled = !ttsEnabled;
  const btn = $('tts-btn');
  btn.textContent = ttsEnabled ? 'TTS AN' : 'TTS AUS';
  btn.className = 'tts-toggle ' + (ttsEnabled ? 'on' : 'off');
  if (!ttsEnabled && window.speechSynthesis) speechSynthesis.cancel();
}

function _voiceParams(zone, tension, dominance) {
  const pitchBase = {guardian:1.00, shadow:0.82, berserker:1.15}[zone] || 1.00;
  const rateBase  = {guardian:0.95, shadow:0.88, berserker:1.22}[zone] || 0.95;
  return {
    pitch:  Math.max(0.5, Math.min(2.0, pitchBase  + dominance * 0.06)),
    rate:   Math.max(0.5, Math.min(2.0, rateBase   + tension   * 0.18)),
    volume: Math.max(0.3, Math.min(1.0, 0.78       + dominance * 0.14)),
  };
}

function _updateTtsDisplay(zone, vp) {
  const zEl = $('tts-zone');
  zEl.textContent = zone;
  zEl.className = 'zone-pill ' + (zone || 'guardian');
  $('tts-params').textContent =
    `pitch=${vp.pitch.toFixed(2)} rate=${vp.rate.toFixed(2)} vol=${vp.volume.toFixed(2)}`;
}

function speakMoloch(text) {
  if (!ttsEnabled || !window.speechSynthesis || !text) return;
  // Zeilenumbrueche zu Pausen machen, nicht abbrechen
  const cleaned = text.replace(/\\n+/g, ' ').replace(/\\s{2,}/g, ' ').trim();
  if (!cleaned) return;
  const vp = _voiceParams(_zone, _tension, _dominance);
  _updateTtsDisplay(_zone, vp);
  const utt = new SpeechSynthesisUtterance(cleaned);
  utt.lang = 'de-DE';
  utt.pitch = vp.pitch;
  utt.rate  = vp.rate;
  utt.volume = vp.volume;
  const voices = speechSynthesis.getVoices();
  const deVoice = voices.find(v => v.lang === 'de-DE' && v.localService)
               || voices.find(v => v.lang.startsWith('de'));
  if (deVoice) utt.voice = deVoice;
  utt.onstart = () => { $('tts-speaking').style.display = ''; _ttsActive = true; };
  utt.onend = utt.onerror = () => { $('tts-speaking').style.display = 'none'; _ttsActive = false; };
  speechSynthesis.speak(utt);
}

function appendChat(who, text, cls) {
  const log = $('chat-log');
  const msg = document.createElement('div');
  msg.className = 'chat-msg ' + (cls || who.toLowerCase());
  const ts = new Date().toLocaleTimeString('de-DE', {hour:'2-digit',minute:'2-digit',second:'2-digit'});
  msg.innerHTML = `<span class="who">[${ts}] ${who}:</span> <span class="body">${text.replace(/&/g,'&amp;').replace(/</g,'&lt;')}</span>`;
  log.appendChild(msg);
  log.scrollTop = log.scrollHeight;
}

async function sendChat() {
  const input = $('chat-input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  const btn = $('chat-send');
  btn.disabled = true;
  btn.textContent = '...';
  appendChat('Markus', msg, 'markus');
  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg}),
    });
    const data = await r.json();
    const reply = data.response || data.text || data.reply || data.message
               || (data.error ? '[Fehler: ' + data.error + ']' : JSON.stringify(data));
    appendChat('MOLOCH', reply, 'moloch');
    speakMoloch(reply);
  } catch(e) {
    appendChat('SYS', 'Verbindungsfehler: ' + e.message, 'system');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Senden';
  }
}
</script></body></html>
"""


@app.get("/")
def root():
    return HTMLResponse(HTML_PAGE)


@app.get("/panel")
def panel():
    p = Path(__file__).parent.parent / "docs" / "main_panel_mockup.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Panel nicht gefunden</h1>", status_code=404)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
