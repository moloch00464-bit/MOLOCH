// MOLOCH Avatar 2.0 — Idle-Animations: Breathing, Blinking, Scanning
// Plus: zentraler animate-Loop der state_morph + effects + glitch_shader tickt
// Phase 1: einfache mathematische Idle-Bewegungen (sin/cos)

import { tickStateMorph } from '/static/js/state_morph.js';
import { tickEffects } from '/static/js/effects.js';
import { tickGlitchShader } from '/static/js/glitch_shader.js';

let _running = false;
let _baselineY = 0;
let _baselinePosX = 0;
let _baselinePosY = 0;
let _baselinePosZ = 0;

/**
 * Idle-Patterns:
 *   Breathing: subtile vertikale Skalierung + Position-Wiggle (3s Periode)
 *   Blinking: emissive-Drop alle 4-7s (Random)
 *   Scanning: leichte Particle-Rotation (Aura dreht sich langsam)
 *   Head-Turn (Phase 3 mit GLB-Modell): Avatar dreht Kopf zu Person-Position
 */
export function startIdleAnimations(sceneObjects) {
  if (_running) return;
  _running = true;

  if (sceneObjects.avatar) {
    _baselineY = sceneObjects.avatar.scale.y;
    _baselinePosX = sceneObjects.avatar.position.x;
    _baselinePosY = sceneObjects.avatar.position.y;
    _baselinePosZ = sceneObjects.avatar.position.z;
  }

  let nextBlinkAt = performance.now() + 4000 + Math.random() * 3000;
  let blinkActive = false;
  let blinkStartedAt = 0;
  const blinkDuration = 100;  // ms

  function tick() {
    if (!_running) return;
    requestAnimationFrame(tick);

    const dt = Math.min(sceneObjects.clock.getDelta(), 0.1);
    const elapsed = sceneObjects.clock.elapsedTime;
    const now = performance.now();

    // 1. State-Morph (Lerp + Apply)
    tickStateMorph(sceneObjects, dt);

    // 2. Effects (Color-Lerp + Flicker + Pulse)
    tickEffects(sceneObjects, dt);

    // 3. Glitch-Shader (wenn aktiv)
    tickGlitchShader(sceneObjects, dt, elapsed);

    // 4. Breathing: subtile vertical scale-modulation + Position-Wiggle
    // CRITICAL FIX (Code-Review #2): breathScale wird jetzt auf scale.y angewendet
    // (war vorher dead code, tickStateMorph setzt scale neu jeden Frame, aber
    // tickStateMorph ist VOR Breathing aufgerufen → wir multiplizieren oben drauf)
    const breathPhase = elapsed * 1.0;  // langsam (1 Hz)
    const breathScale = 1.0 + 0.02 * Math.sin(breathPhase);
    if (sceneObjects.avatar) {
      sceneObjects.avatar.scale.y *= breathScale;  // additive Modulation
      sceneObjects.avatar.position.y = _baselinePosY + 0.05 * Math.sin(breathPhase);
    }

    // 5. Blinking: kurzer emissive-Drop (additive Modulation, kein Multi-Drift)
    if (sceneObjects.material && sceneObjects.material.emissiveIntensity !== undefined) {
      if (now >= nextBlinkAt && !blinkActive) {
        blinkActive = true;
        blinkStartedAt = now;
      }
      if (blinkActive) {
        const blinkProgress = (now - blinkStartedAt) / blinkDuration;
        if (blinkProgress >= 1.0) {
          blinkActive = false;
          nextBlinkAt = now + 4000 + Math.random() * 3000;
        } else {
          // IMPORTANT FIX (Code-Review #5): einmalige Drop-Modulation, kein dauerhaftes *=.
          // tickStateMorph setzt Base-Emissive neu jeden Frame, wir multiplizieren
          // nur fuer die kurze Blink-Dauer drauf (nicht akkumulativ).
          const dampFactor = 1.0 - 0.6 * Math.sin(blinkProgress * Math.PI);
          sceneObjects.material.emissiveIntensity *= dampFactor;
        }
      }
    }

    // 6. Scanning: Particle-Rotation
    if (sceneObjects.particles) {
      sceneObjects.particles.rotation.y += dt * 0.05;
      sceneObjects.particles.rotation.x += dt * 0.02;
    }
  }
  tick();
}

export function stopIdleAnimations() {
  _running = false;
}
