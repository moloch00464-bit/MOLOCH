// MOLOCH Avatar 2.0 — Effects: Material-Color + Lighting je Zone
// Gemini-Vision:
//   guardian   -> Cyan/Magenta Neon, steady
//   shadow     -> Deep Purple, flickering
//   berserker  -> Pulsating Crimson Red

const ZONE_PALETTE = {
  guardian: {
    base: 0x66aaff,         // Cyan
    emissive: 0x113355,
    keyLight: 0x66ccff,     // Cyan-Spot
    rimLight: 0xff66ee,     // Magenta-Rim
    flicker: false,
    pulse: false,
  },
  shadow: {
    base: 0x9966cc,         // Deep Purple
    emissive: 0x331144,
    keyLight: 0xaa66ff,
    rimLight: 0x6622aa,
    flicker: true,
    pulse: false,
  },
  berserker: {
    base: 0xff3344,         // Crimson Red
    emissive: 0x550011,
    keyLight: 0xff5566,
    rimLight: 0xff8866,
    flicker: false,
    pulse: true,
  },
};

let _currentZone = 'guardian';
let _flickerSeed = 0;
let _pulsePhase = 0;

const COLOR_LERP = 0.04;

// CRITICAL FIX (Code-Review #1): THREE.Color lazy-init.
// ES-Module-Evaluation laeuft bevor inline <script type="module"> garantiert
// nach dem global <script src=three.min.js> kommt. THREE moeglicherweise undefined.
let _baseTarget = null;
let _emissiveTarget = null;

function _ensureColors() {
  if (_baseTarget === null) {
    _baseTarget = new THREE.Color(0x66aaff);
    _emissiveTarget = new THREE.Color(0x113355);
  }
}

export function applyEffects(sceneObjects, state) {
  _ensureColors();

  const piState = state?.pi?.state || {};
  const zone = piState.zone || 'guardian';

  if (zone !== _currentZone && ZONE_PALETTE[zone]) {
    _currentZone = zone;
    const p = ZONE_PALETTE[zone];
    _baseTarget.setHex(p.base);
    _emissiveTarget.setHex(p.emissive);

    if (sceneObjects.keyLight) {
      sceneObjects.keyLight.color.setHex(p.keyLight);
    }
    if (sceneObjects.rimLight) {
      sceneObjects.rimLight.color.setHex(p.rimLight);
    }
  }
}

/**
 * Per-Frame-Effects-Tick (Color-Lerp, Flicker, Pulse).
 * Aufgerufen von idle_animations.js im animate-Loop NACH tickStateMorph.
 *
 * IMPORTANT FIX (Code-Review #5): Kein Multiplikations-Drift mehr auf
 * material.emissiveIntensity. tickStateMorph setzt einen "base" Wert, wir
 * speichern den als _baseEmissive und modulieren additiv pro Frame.
 */
export function tickEffects(sceneObjects, dt) {
  if (!sceneObjects || !sceneObjects.material) return;
  _ensureColors();

  const palette = ZONE_PALETTE[_currentZone] || ZONE_PALETTE.guardian;

  // Smooth color lerp
  sceneObjects.material.color.lerp(_baseTarget, COLOR_LERP);
  sceneObjects.material.emissive.lerp(_emissiveTarget, COLOR_LERP);

  // Snapshot des "morph-base"-Werts (von tickStateMorph soeben gesetzt)
  const baseEmissive = sceneObjects.material.emissiveIntensity ?? 0.4;

  // Shadow-Flicker: zufaellige Helligkeits-Drops
  if (palette.flicker) {
    _flickerSeed += dt;
    if (_flickerSeed > 0.15 + Math.random() * 0.3) {
      _flickerSeed = 0;
      const drop = 0.4 + Math.random() * 0.3;
      if (sceneObjects.keyLight) {
        sceneObjects.keyLight.intensity = drop;
      }
    } else {
      if (sceneObjects.keyLight) {
        sceneObjects.keyLight.intensity = Math.min(
          sceneObjects.keyLight.intensity + dt * 4,
          1.5
        );
      }
    }
  }

  // Berserker-Pulse: rhythmischer Glow-Puls — ADDITIV auf baseEmissive
  if (palette.pulse) {
    _pulsePhase += dt * 4;
    const pulseFactor = 0.6 + 0.4 * Math.abs(Math.sin(_pulsePhase));
    if (sceneObjects.keyLight) {
      sceneObjects.keyLight.intensity = pulseFactor * 1.5;
    }
    // Additiv: baseEmissive bleibt stabil, pulseFactor moduliert obendrauf
    sceneObjects.material.emissiveIntensity = baseEmissive * (0.8 + pulseFactor * 0.4);
  }
}
