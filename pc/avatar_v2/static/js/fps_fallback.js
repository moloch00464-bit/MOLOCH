// MOLOCH Avatar 2.0 — FPS-Fallback
// Wenn FPS dauerhaft <threshold (default 15) faellt: switch zu SVG-Fallback
// Verhindert dass GTX 760 2GB sich aufhaengt bei zu komplexer Scene

import { onFpsUpdate, getCurrentFps } from '/static/js/scene.js';

let _fallbackActive = false;
let _consecutiveBadFrames = 0;
const REQUIRED_BAD_FRAMES = 5;  // 5x in Folge unter Threshold = Fallback aktiv

/**
 * options:
 *   threshold: FPS-Mindestwert (default 15)
 *   onFallback: Callback wenn Fallback triggert
 *   onRecover: Callback wenn FPS wieder ueber threshold (NICHT auto-recovered, manuell)
 */
export function initFpsFallback(sceneObjects, options) {
  const threshold = options?.threshold ?? 15;
  const onFallback = options?.onFallback || (() => {});

  onFpsUpdate((fps) => {
    if (_fallbackActive) return;  // Einmal aktiv, bleibt aktiv (no flicker)

    if (fps < threshold) {
      _consecutiveBadFrames++;
      console.warn('[fps-fallback] low FPS detected:', fps, '(', _consecutiveBadFrames, '/', REQUIRED_BAD_FRAMES, ')');
      if (_consecutiveBadFrames >= REQUIRED_BAD_FRAMES) {
        console.warn('[fps-fallback] activating SVG fallback (FPS below', threshold, 'for', REQUIRED_BAD_FRAMES, 'consecutive seconds)');
        _fallbackActive = true;
        try { onFallback(); } catch (e) { console.error('fallback-callback error', e); }
      }
    } else {
      _consecutiveBadFrames = 0;
    }
  });
}

export function isFallbackActive() { return _fallbackActive; }
