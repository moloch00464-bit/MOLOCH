// MOLOCH Avatar 2.0 — State-Polling (HTTP, 200ms)
// Pi-State-Source: /api/state via FastAPI-Server (proxied von Pi /api/state/current)
// Phase 3: Websocket-Migration falls Polling-Latency stoert
//
// IMPORTANT FIX (Code-Review #4): recursive setTimeout statt setInterval.
// Verhindert Request-Stacking wenn /api/state >200ms braucht (z.B. Pi-Offline-2s-Timeout).
// Naechster Poll wird erst geschedulet wenn aktueller fertig ist.

const POLL_INTERVAL_MS = 200;
const ENDPOINT = '/api/state';

let _running = false;
let _lastSuccess = 0;
let _onState = null;

async function _pollOnce() {
  try {
    const r = await fetch(ENDPOINT, { cache: 'no-store' });
    if (r.ok) {
      const data = await r.json();
      _lastSuccess = performance.now();
      try { _onState && _onState(data); } catch (e) { console.error('state-callback error', e); }
    }
  } catch (e) {
    // Stiller Fehler — Polling laeuft weiter, FPS-Loop bleibt stabil
  }
}

async function _pollLoop() {
  while (_running) {
    await _pollOnce();
    if (!_running) break;
    await new Promise((res) => setTimeout(res, POLL_INTERVAL_MS));
  }
}

/**
 * Startet Polling. Callback wird mit aggregiertem State aufgerufen.
 * State-Format: {ts, pi: {online, latency_ms, state}, pc: {online, latency_ms, health}}
 */
export function startStatePolling(onState) {
  if (_running) return;
  _onState = onState;
  _running = true;
  _pollLoop();
}

export function stopStatePolling() {
  _running = false;
}

export function getLastSuccessTs() { return _lastSuccess; }
