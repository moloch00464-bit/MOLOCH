"""MOLOCH Avatar Service — visuelles Live-Fenster auf :11800.

Three.js low-poly Creature, mood-getrieben:
- Material-Color folgt Zone (Guardian=blau, Shadow=lila, Berserker=rot)
- Mesh scale + emissive Pulse folgt Tension
- Pose Rotation/Tilt folgt Dominance
- Presence beeinflusst Eigen-Glow
- Particle-Aura drumrum (Three.js Points)
- Watchdog-Warning -> roter Flash-Overlay

Plus integriertes System-HUD: FPS / RAM / NPU-Worker / Watchdog / Zone-Label.

Datenquelle: Pi `/state_full` via SSH-Tunnel (localhost:9000), plus
PC `:11600/health` fuer aktiven Adapter. Polling 1s, Render 60fps mit
smoother Interpolation.
"""
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

PORT = int(os.environ.get("MOLOCH_AVATAR_PORT", "11800"))
PROXY_URL = os.environ.get("MOLOCH_PROXY_URL", "http://localhost:11600")
PI_TUNNEL_URL = os.environ.get("MOLOCH_PI_TUNNEL_URL", "http://localhost:9000")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


app = FastAPI(title="MOLOCH Avatar", lifespan=lifespan)
# Preventive: cockpit on Pi may fetch /api/state directly some day (not via iframe).
# Same-origin iframe-embed works without this; CORS is purely future-proofing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000", "http://192.168.178.20:9000",
                   "http://192.168.178.30:9100", "https://192.168.178.30:9443"],
    allow_methods=["GET"],
)


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
            return None, dt
    except Exception:
        return None, int((time.time() - t0) * 1000)


@app.get("/api/state")
async def api_state():
    async with httpx.AsyncClient() as client:
        pi_full, pi_dt = await _safe_get(client, f"{PI_TUNNEL_URL}/state_full", timeout=3.0)
        pc_health, pc_dt = await _safe_get(client, f"{PROXY_URL}/health")

    return {
        "ts": time.time(),
        "pi": {"online": pi_full is not None, "latency_ms": pi_dt, "state": pi_full},
        "pc": {"online": pc_health is not None, "latency_ms": pc_dt, "health": pc_health},
    }


HTML_PAGE = """<!doctype html>
<html lang="de"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOLOCH Avatar</title>
<style>
  :root{--bg:#08080c;--fg:#e6e6ee;--mute:#7a7a8a;--ok:#5dc36b;--warn:#e6b84d;--err:#ff7676;
        --guardian:#3673ce;--shadow:#7e3bce;--berserker:#c93838}
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{height:100%;background:var(--bg);color:var(--fg);
    font:13px/1.4 system-ui,sans-serif;overflow:hidden}
  #scene{position:fixed;inset:0;z-index:0}
  .hud{position:fixed;z-index:5;font:600 11px/1.3 system-ui;letter-spacing:.4px;
    color:var(--mute);text-transform:uppercase;pointer-events:none}
  .hud .v{color:var(--fg);font-variant-numeric:tabular-nums}
  .panel{background:rgba(13,13,18,.78);border:1px solid #26262f;border-radius:8px;
    padding:8px 10px;backdrop-filter:blur(4px)}
  #hud-tl{top:12px;left:12px}
  #hud-tr{top:12px;right:12px;text-align:right}
  #hud-tc{top:12px;left:50%;transform:translateX(-50%);font:700 14px/1 system-ui;
    letter-spacing:6px;text-transform:uppercase;text-align:center}
  #hud-tc .zone{padding:6px 14px;border-radius:6px;background:rgba(13,13,18,.78);
    border:1px solid #26262f;display:inline-block;color:var(--fg)}
  #hud-tc.guardian .zone{box-shadow:0 0 16px rgba(54,115,206,.6);border-color:var(--guardian)}
  #hud-tc.shadow .zone{box-shadow:0 0 16px rgba(126,59,206,.6);border-color:var(--shadow)}
  #hud-tc.berserker .zone{box-shadow:0 0 22px rgba(201,56,56,.8);border-color:var(--berserker)}
  #hud-bl{bottom:12px;left:12px;max-width:360px}
  #hud-br{bottom:12px;right:12px}
  .bar{display:inline-block;width:90px;height:6px;background:#26262f;border-radius:3px;
    overflow:hidden;vertical-align:middle;margin-left:6px}
  .bar>i{display:block;height:100%;background:var(--ok);transition:width .3s}
  .bar.warn>i{background:var(--warn)} .bar.err>i{background:var(--err)}
  .row{margin:3px 0;display:flex;gap:8px;align-items:center;justify-content:flex-start}
  #hud-tr .row{justify-content:flex-end}
  .pill{display:inline-block;padding:2px 7px;border-radius:10px;background:#26262f;
    font-size:10px;color:var(--mute);margin:1px 2px}
  .pill.on{background:#1f3a1f;color:var(--ok)} .pill.off{background:#2a1414;color:var(--err)}
  .toast{margin-top:4px;padding:5px 8px;background:#2a1414;border-left:3px solid var(--err);
    border-radius:0 4px 4px 0;color:#ffc8c8;font-size:11px;animation:fadein .3s}
  @keyframes fadein{from{opacity:0;transform:translateX(-6px)}to{opacity:1;transform:none}}
  #flash{position:fixed;inset:0;z-index:4;pointer-events:none;background:radial-gradient(
    circle at center,rgba(255,118,118,.15),transparent 60%);opacity:0;transition:opacity .2s}
  #flash.on{opacity:1}
  #status{position:fixed;bottom:6px;left:50%;transform:translateX(-50%);
    color:var(--mute);font-size:10px;z-index:5;pointer-events:none}
</style></head>
<body>
<canvas id="scene"></canvas>
<div id="flash"></div>

<div id="hud-tc" class="hud guardian"><div class="zone">ZONE GUARDIAN</div></div>

<div id="hud-tl" class="hud panel">
  <div class="row">RAM <span class="bar"><i id="bar-ram" style="width:0%"></i></span> <span class="v" id="ram">—</span></div>
  <div class="row">CPU <span class="bar"><i id="bar-cpu" style="width:0%"></i></span> <span class="v" id="cpu">—</span></div>
  <div class="row">Tension <span class="v" id="tension">—</span></div>
  <div class="row">Dominance <span class="v" id="dominance">—</span></div>
  <div class="row">Presence <span class="v" id="presence">—</span></div>
</div>

<div id="hud-tr" class="hud panel">
  <div class="row">FPS <span class="bar"><i id="bar-fps" style="width:0%"></i></span> <span class="v" id="fps">—</span></div>
  <div class="row">Tunnel-Latenz <span class="v" id="lat">—</span> ms</div>
  <div class="row">PC-Adapter <span class="v" id="adapter">—</span></div>
  <div class="row">Pi-Provider <span class="v" id="provider">—</span></div>
</div>

<div id="hud-br" class="hud panel">
  <div>NPU-Worker</div>
  <div id="workers"></div>
</div>

<div id="hud-bl" class="hud">
  <div id="warnings"></div>
</div>

<div id="status">MOLOCH AVATAR · :11800 · auto-poll 1s</div>

<script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
<script>
// --- Scene setup ---
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true, powerPreference:'high-performance'});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x08080c);
scene.fog = new THREE.Fog(0x08080c, 4, 14);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 100);
camera.position.set(0, 0.4, 5.5);
camera.lookAt(0, 0, 0);

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- Lights ---
scene.add(new THREE.AmbientLight(0x404060, 0.7));
const key = new THREE.DirectionalLight(0xffffff, 0.6); key.position.set(3, 4, 5); scene.add(key);
const rim = new THREE.DirectionalLight(0xff5050, 0.3); rim.position.set(-4, -2, -3); scene.add(rim);

// --- Avatar: stylized low-poly Moloch face ---
// Group hierarchy so we can tilt/scale the head as one unit.
const head = new THREE.Group();
scene.add(head);

// Skull base — slightly elongated dodecahedron (12 faces, looks like a stylized head)
const skullGeo = new THREE.DodecahedronGeometry(1.0, 0);
skullGeo.scale(0.85, 1.15, 0.85);  // longer face, narrower sides
const basePositions = skullGeo.attributes.position.array.slice(); // for subtle vertex-breathing
const skullMat = new THREE.MeshPhongMaterial({
  color: 0x3673ce, emissive: 0x102040, emissiveIntensity: 0.45,
  flatShading: true, shininess: 25, specular: 0x222244
});
const skull = new THREE.Mesh(skullGeo, skullMat);
head.add(skull);

// Wireframe edges (shares skull geo - tracks vertex breathing)
const skullWireMat = new THREE.MeshBasicMaterial({color: 0xffffff, wireframe: true,
                                                   transparent: true, opacity: 0.16});
const skullWire = new THREE.Mesh(skullGeo, skullWireMat);
head.add(skullWire);

// --- Eyes (sockets, iris, pupil) ---
// Eye whites (dark recessed sphere as socket)
const eyeWhiteGeo = new THREE.SphereGeometry(0.16, 14, 10);
const eyeWhiteMat = new THREE.MeshPhongMaterial({color: 0x080810, emissive: 0x040408,
                                                  emissiveIntensity: 0.2, shininess: 60});
const leftEye = new THREE.Mesh(eyeWhiteGeo, eyeWhiteMat);
leftEye.position.set(-0.30, 0.18, 0.78);
head.add(leftEye);
const rightEye = new THREE.Mesh(eyeWhiteGeo, eyeWhiteMat);
rightEye.position.set(0.30, 0.18, 0.78);
head.add(rightEye);

// Iris (zone-colored, glows; one material per side so they could differ)
const irisGeo = new THREE.SphereGeometry(0.10, 14, 10);
const leftIrisMat = new THREE.MeshBasicMaterial({color: 0x3673ce});
const rightIrisMat = new THREE.MeshBasicMaterial({color: 0x3673ce});
const leftIris = new THREE.Mesh(irisGeo, leftIrisMat);
leftIris.position.set(-0.30, 0.18, 0.86);
head.add(leftIris);
const rightIris = new THREE.Mesh(irisGeo, rightIrisMat);
rightIris.position.set(0.30, 0.18, 0.86);
head.add(rightIris);

// Pupil (black, scales with tension - dilation)
const pupilGeo = new THREE.SphereGeometry(0.05, 10, 8);
const pupilMat = new THREE.MeshBasicMaterial({color: 0x000000});
const leftPupil = new THREE.Mesh(pupilGeo, pupilMat);
leftPupil.position.set(-0.30, 0.18, 0.92);
head.add(leftPupil);
const rightPupil = new THREE.Mesh(pupilGeo, pupilMat);
rightPupil.position.set(0.30, 0.18, 0.92);
head.add(rightPupil);

// Brow ridge above eyes (gives it a brow line)
const browGeo = new THREE.BoxGeometry(0.85, 0.06, 0.10);
const browMat = new THREE.MeshPhongMaterial({color: 0x141420, shininess: 8});
const brow = new THREE.Mesh(browGeo, browMat);
brow.position.set(0, 0.36, 0.74);
head.add(brow);

// Mouth (thin angular line; rotates with dominance for smile/frown, scales with tension)
const mouthGeo = new THREE.BoxGeometry(0.50, 0.05, 0.10);
const mouthMat = new THREE.MeshPhongMaterial({color: 0x100808, emissive: 0x080404,
                                               emissiveIntensity: 0.3, shininess: 20});
const mouth = new THREE.Mesh(mouthGeo, mouthMat);
mouth.position.set(0, -0.45, 0.68);
head.add(mouth);

// Cheekbones (angular planes for face definition)
const cheekGeo = new THREE.BoxGeometry(0.12, 0.32, 0.10);
const cheekMat = new THREE.MeshPhongMaterial({color: 0x1a1a28, shininess: 10});
const leftCheek = new THREE.Mesh(cheekGeo, cheekMat);
leftCheek.position.set(-0.55, -0.10, 0.42);
leftCheek.rotation.z = 0.35;
head.add(leftCheek);
const rightCheek = new THREE.Mesh(cheekGeo, cheekMat);
rightCheek.position.set(0.55, -0.10, 0.42);
rightCheek.rotation.z = -0.35;
head.add(rightCheek);

// --- Particle Aura ---
const PCOUNT = 180;
const pgeo = new THREE.BufferGeometry();
const ppos = new Float32Array(PCOUNT * 3);
const pvel = new Float32Array(PCOUNT * 3);
for (let i=0; i<PCOUNT; i++) {
  const r = 1.6 + Math.random() * 1.5;
  const th = Math.random() * Math.PI * 2;
  const ph = Math.acos(2*Math.random() - 1);
  ppos[i*3+0] = r * Math.sin(ph) * Math.cos(th);
  ppos[i*3+1] = r * Math.sin(ph) * Math.sin(th);
  ppos[i*3+2] = r * Math.cos(ph);
  pvel[i*3+0] = (Math.random()-0.5) * 0.004;
  pvel[i*3+1] = (Math.random()-0.5) * 0.004;
  pvel[i*3+2] = (Math.random()-0.5) * 0.004;
}
pgeo.setAttribute('position', new THREE.BufferAttribute(ppos, 3));
const pmat = new THREE.PointsMaterial({color: 0x6688cc, size: 0.05, transparent: true, opacity: 0.7});
const particles = new THREE.Points(pgeo, pmat);
scene.add(particles);

// --- Color helpers ---
const COL_GUARDIAN = new THREE.Color(0x3673ce);
const COL_SHADOW   = new THREE.Color(0x7e3bce);
const COL_BERSERKER= new THREE.Color(0xc93838);
const _scratchCol  = new THREE.Color();  // never let zoneColor() return a constant

function zoneColor(zone, guardianInf, shadowInf) {
  if (zone === 'berserker') return _scratchCol.copy(COL_BERSERKER);
  if (zone === 'shadow')    return _scratchCol.copy(COL_SHADOW);
  if (zone === 'guardian')  return _scratchCol.copy(COL_GUARDIAN);
  if (typeof guardianInf === 'number') {
    const g = Math.max(0, Math.min(1, guardianInf));
    return _scratchCol.copy(COL_GUARDIAN).lerp(COL_SHADOW, 1 - g);
  }
  return _scratchCol.copy(COL_GUARDIAN);
}

// --- State (interpolated targets) ---
const target = {
  zone: 'guardian',
  tension: 0,        // -1..+1
  dominance: 0,      // -1..+1
  presence: 1.0,
  voice_intensity: 0.3,
  language_sharpness: 0.0,
  guardian_influence: 1.0,
  shadow_influence: 0.0,
  berserker_active: false,
  watchdog_warn: false
};
const live = JSON.parse(JSON.stringify(target));

// --- Animation loop ---
let frameCount = 0;
let lastFpsT = performance.now();
let renderFps = 60;
function lerp(a,b,t){return a+(b-a)*t}

function tick() {
  // Smooth all numeric values toward target
  const k = 0.08;
  live.tension = lerp(live.tension, target.tension, k);
  live.dominance = lerp(live.dominance, target.dominance, k);
  live.presence = lerp(live.presence, target.presence, k);
  live.voice_intensity = lerp(live.voice_intensity, target.voice_intensity, k);
  live.language_sharpness = lerp(live.language_sharpness, target.language_sharpness, k);
  live.guardian_influence = lerp(live.guardian_influence, target.guardian_influence, k);
  live.shadow_influence = lerp(live.shadow_influence, target.shadow_influence, k);
  live.zone = target.zone; // discrete
  live.berserker_active = target.berserker_active;
  live.watchdog_warn = target.watchdog_warn;

  const now = performance.now();
  const t = now * 0.001;

  // Mood-Color (skull = darker variant, iris = saturated variant)
  const col = zoneColor(live.zone, live.guardian_influence, live.shadow_influence);
  if (live.berserker_active) col.lerp(COL_BERSERKER, 0.6);

  // Pulse: voice_intensity drives speed (visible in glow + breathing scale)
  const pulseSpeed = 0.6 + live.voice_intensity * 4.0;
  const pulse = 0.5 + 0.5 * Math.sin(t * pulseSpeed);

  // Skull material (slightly desaturated)
  skullMat.color.copy(col).multiplyScalar(0.7);
  skullMat.emissive.copy(col).multiplyScalar(0.15 + 0.35 * pulse * live.presence);
  skullMat.emissiveIntensity = 0.3 + 0.5 * pulse * live.presence;

  // Iris colors (more saturated than skull, glowing)
  leftIrisMat.color.copy(col);
  rightIrisMat.color.copy(col);

  // Pupil dilation: high abs(tension) -> bigger pupil (stress + extreme calm both dilate)
  const pupilSize = 0.6 + 0.7 * Math.abs(live.tension);  // 0.6 to 1.3
  leftPupil.scale.setScalar(pupilSize);
  rightPupil.scale.setScalar(pupilSize);

  // Particle aura color follows mood
  pmat.color.copy(col);

  // Mouth: tension makes it longer/sharper, dominance gives smile/frown curl
  const mouthScaleY = 1.0 + 1.5 * Math.max(0, live.tension);
  mouth.scale.set(1, mouthScaleY, 1);
  mouth.rotation.z = -live.dominance * 0.18;  // pos dominance = corners up
  mouth.position.y = -0.45 - 0.05 * Math.max(0, live.tension);  // drops a hair when tense

  // Head: NO auto-rotation. Subtle dominance-tilt + soft tension breath, that's it.
  head.rotation.x = live.dominance * -0.10;             // pos dominance = head up
  head.rotation.z = -live.dominance * 0.05;
  head.rotation.y = Math.sin(t * 0.3) * 0.04;           // micro left/right glance, no spin

  // Breathing scale: subtle, follows pulse + tension
  const breathing = 1.0 + 0.025 * Math.sin(t * 1.2);
  const tensionScale = 1.0 + 0.06 * Math.max(0, live.tension);
  head.scale.setScalar(breathing * tensionScale);

  // Skull vertex breathing (subtle wobble — much gentler than before, only in tension)
  const pos = skullGeo.attributes.position;
  const wild = 0.02 + 0.05 * Math.max(0, live.tension);
  for (let i=0; i<pos.count; i++) {
    const ix = i*3;
    const nx = basePositions[ix], ny = basePositions[ix+1], nz = basePositions[ix+2];
    const noise = Math.sin(t * 1.5 + ix * 0.7) * wild;
    pos.array[ix]   = nx * (1 + noise * 0.02);
    pos.array[ix+1] = ny * (1 + noise * 0.02);
    pos.array[ix+2] = nz * (1 + noise * 0.02);
  }
  pos.needsUpdate = true;

  // Particle aura: slow orbit, no aggressive spin
  particles.rotation.y += 0.001;
  particles.rotation.x = Math.sin(t * 0.15) * 0.04;
  pmat.size = 0.04 + 0.025 * pulse;
  pmat.opacity = 0.30 + 0.35 * live.presence;

  // Watchdog flash
  document.getElementById('flash').classList.toggle('on', live.watchdog_warn);

  renderer.render(scene, camera);

  frameCount++;
  if (now - lastFpsT > 1000) {
    renderFps = Math.round(frameCount * 1000 / (now - lastFpsT));
    frameCount = 0; lastFpsT = now;
  }

  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// --- Polling ---
function fmtPct(v){return (typeof v==='number')?(v.toFixed(0)+'%'):'—'}
function fix(v,n){return (typeof v==='number')?v.toFixed(n):'—'}

async function poll() {
  try {
    const r = await fetch('/api/state');
    const s = await r.json();
    const pi = s.pi.state;
    if (!pi) {
      document.getElementById('status').textContent = 'PI OFFLINE — Tunnel down?';
      target.watchdog_warn = true;
      return;
    }
    const sys = pi.system || {};
    const pers = pi.personality || {};
    const core = pers.core || {};
    const eff = core.effects || {};
    const wd = sys.watchdog || {};
    const llm = pi.llm || {};

    // /state_full.system.fps is a dict {scrfd, arcface, yolov8m, total} — pick total
    const fpsRaw = sys.fps;
    const fpsNum = (typeof fpsRaw === 'number') ? fpsRaw
                 : (fpsRaw && typeof fpsRaw.total === 'number') ? fpsRaw.total : 0;

    // Update animation targets (alle defensiv typed)
    target.zone = (typeof core.zone === 'string' && core.zone) ? core.zone : 'guardian';
    target.tension = (typeof core.tension === 'number') ? core.tension : 0;
    target.dominance = (typeof core.dominance === 'number') ? core.dominance : 0;
    target.presence = (typeof core.presence === 'number') ? core.presence : 1.0;
    target.voice_intensity = (typeof eff.voice_intensity === 'number') ? eff.voice_intensity : 0.3;
    target.language_sharpness = (typeof eff.language_sharpness === 'number') ? eff.language_sharpness : 0;
    target.guardian_influence = (typeof eff.guardian_influence === 'number') ? eff.guardian_influence : 1.0;
    target.shadow_influence = (typeof eff.shadow_influence === 'number') ? eff.shadow_influence : 0;
    target.berserker_active = !!core.berserker_active;
    target.watchdog_warn = !!wd.throttled || (Array.isArray(wd.warnings) && wd.warnings.length > 0);

    // HUD updates
    document.getElementById('hud-tc').className = 'hud ' + target.zone;
    document.getElementById('hud-tc').firstElementChild.textContent = 'ZONE ' + target.zone.toUpperCase();

    const ramP = (typeof wd.ram_percent === 'number') ? wd.ram_percent : 0;
    const cpuT = (typeof wd.cpu_temp === 'number') ? wd.cpu_temp : 0;
    const cpuPct = Math.min(100, cpuT * 100 / 80); // 80°C = 100%
    document.getElementById('ram').textContent = fix(ramP, 0) + '%';
    document.getElementById('bar-ram').style.width = fix(ramP, 0) + '%';
    document.getElementById('bar-ram').parentElement.className = 'bar' + (ramP > 80 ? ' err' : ramP > 60 ? ' warn' : '');
    document.getElementById('cpu').textContent = fix(cpuT, 1) + '°C';
    document.getElementById('bar-cpu').style.width = fix(cpuPct, 0) + '%';
    document.getElementById('bar-cpu').parentElement.className = 'bar' + (cpuT > 70 ? ' err' : cpuT > 60 ? ' warn' : '');
    document.getElementById('tension').textContent = fix(target.tension, 2);
    document.getElementById('dominance').textContent = fix(target.dominance, 2);
    document.getElementById('presence').textContent = fix(target.presence, 2);

    const fpsPct = Math.min(100, fpsNum * 100 / 25);
    document.getElementById('fps').textContent = fix(fpsNum, 1);
    document.getElementById('bar-fps').style.width = fix(fpsPct, 0) + '%';
    document.getElementById('bar-fps').parentElement.className = 'bar' + (fpsNum < 15 ? ' err' : fpsNum < 20 ? ' warn' : '');
    document.getElementById('lat').textContent = s.pi.latency_ms;
    document.getElementById('adapter').textContent = (s.pc.health || {}).adapter || '(none)';
    document.getElementById('provider').textContent = llm.provider || '—';

    // Worker pills
    const ph = (pi.pipeline || {}).worker_health || {};
    const wEl = document.getElementById('workers');
    wEl.innerHTML = '';
    for (const [name, w] of Object.entries(ph)) {
      if (name.startsWith('_') || !w) continue;
      const ok = w.running && w.total_errors === 0;
      const span = document.createElement('span');
      span.className = 'pill ' + (ok ? 'on' : 'off');
      span.textContent = name.replace('Worker', '');
      wEl.appendChild(span);
    }

    // Watchdog warnings as toasts
    const wEl2 = document.getElementById('warnings');
    wEl2.innerHTML = '';
    if (wd.throttled) {
      const t = document.createElement('div'); t.className = 'toast'; t.textContent = '⚠ THROTTLED';
      wEl2.appendChild(t);
    }
    for (const w of (wd.warnings || []).slice(0, 3)) {
      const t = document.createElement('div'); t.className = 'toast'; t.textContent = '⚠ ' + w;
      wEl2.appendChild(t);
    }

    document.getElementById('status').textContent =
      'MOLOCH AVATAR · :11800 · render '+renderFps+'fps · Pi '+s.pi.latency_ms+'ms';
  } catch (e) {
    document.getElementById('status').textContent = 'POLL ERROR: ' + e.message;
    target.watchdog_warn = true;
  }
}
poll();
setInterval(poll, 1000);
</script></body></html>
"""


@app.get("/")
def root():
    return HTMLResponse(HTML_PAGE)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
