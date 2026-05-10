// MOLOCH Avatar 2.0 — State-Morph: Mesh-Form je State
// 6 States: idle/observing/engaged/overloaded/withdrawing/offline_anchor
// Phase 1 Placeholder: Skalierung + Verformung der Icosahedron
// Phase 3: echte Morph-Targets via .glb mit benannten Targets

/**
 * State-Form-Profile.
 * scale: Mesh-Skalierung
 * deform: Vertex-Distortion-Faktor (geht in glitch_shader)
 * rotation_speed: Eigen-Rotation
 * emissive_intensity: Emissive-Glow
 */
const STATE_MORPH = {
  idle:         { scale: 1.00, deform: 0.05, rotation_speed: 0.05, emissive_intensity: 0.2 },
  observing:    { scale: 1.05, deform: 0.10, rotation_speed: 0.10, emissive_intensity: 0.3 },
  engaged:      { scale: 1.15, deform: 0.20, rotation_speed: 0.15, emissive_intensity: 0.5 },
  overloaded:   { scale: 1.25, deform: 0.45, rotation_speed: 0.35, emissive_intensity: 0.8 },
  withdrawing:  { scale: 0.85, deform: 0.08, rotation_speed: 0.03, emissive_intensity: 0.15 },
  offline_anchor: { scale: 0.70, deform: 0.02, rotation_speed: 0.01, emissive_intensity: 0.10 },
};

const TRANSITION_LERP = 0.05;  // smooth state-Morph (per frame)
const BEAT_PULSE_AMPLITUDE = 0.4;  // Max emissive-Boost on Beat-Onset

let _currentMorph = { ...STATE_MORPH.idle };
let _targetMorph = { ...STATE_MORPH.idle };

// Music-Beat-Sync (Pi liefert music_beat_phase 0..1 + music_bpm via Sprint 2 Fix 4)
let _musicBeatPhase = 0;     // letzter bekannter beat-phase aus Pi-state
let _musicBpm = 0;            // aktueller BPM (0 = keine Musik)
let _beatPhaseUpdateMs = 0;  // performance.now() beim letzten Pi-state-update

/**
 * Setzt Ziel-Morph basierend auf state-Object.
 * state.pi.state.current_state oder fallback state.pi.state.zone.
 */
export function applyStateMorph(sceneObjects, state) {
  const piState = state?.pi?.state || {};
  const currentState = piState.current_state || piState.zone || 'idle';

  const target = STATE_MORPH[currentState] || STATE_MORPH.idle;
  _targetMorph = { ...target };

  // Tension overlay: hohe Tension intensiviert deform unabhaengig vom State
  const tension = piState.tension ?? 0;
  if (tension > 0.5) {
    _targetMorph.deform += (tension - 0.5) * 0.4;
    _targetMorph.emissive_intensity += (tension - 0.5) * 0.4;
  }

  // Music-Beat-Sync: cache phase + bpm aus Pi-state (Sprint 2 Fix 4 Felder)
  // Wir extrapolieren in tickStateMorph zwischen Pi-state-updates (200ms Polling
  // ist viel laenger als Frame-Rate, sonst kein gleichmaessiger Pulse).
  if (typeof piState.music_bpm === 'number') {
    _musicBpm = piState.music_bpm;
    _musicBeatPhase = piState.music_beat_phase ?? 0;
    _beatPhaseUpdateMs = performance.now();
  }
}

/**
 * Per-Frame-Update: lerpt _currentMorph zu _targetMorph + applies auf Mesh.
 * Wird von idle_animations.js im animate-Loop aufgerufen.
 */
export function tickStateMorph(sceneObjects, dt) {
  if (!sceneObjects || !sceneObjects.avatar) return;

  // Lerp aktuelle Werte zu Ziel
  for (const k of Object.keys(_targetMorph)) {
    _currentMorph[k] += (_targetMorph[k] - _currentMorph[k]) * TRANSITION_LERP;
  }

  // Apply Scale
  sceneObjects.avatar.scale.set(
    _currentMorph.scale,
    _currentMorph.scale,
    _currentMorph.scale,
  );

  // Apply Rotation (akkumuliert)
  sceneObjects.avatar.rotation.x += _currentMorph.rotation_speed * dt * 0.5;
  sceneObjects.avatar.rotation.y += _currentMorph.rotation_speed * dt;

  // Music-Beat-Pulse (overlay on emissive). Sprint 2 Fix 4 liefert phase+bpm.
  // Extrapoliere phase basierend auf elapsed time seit letztem Pi-state-update.
  // Cosinus-Pulse: peak=BEAT_PULSE_AMPLITUDE am Beat-Onset (phase=0), 0 dazwischen.
  // Pi-Stall-Guard: bei >2s ohne Pi-state-update keine Pulse-Halluzination.
  let beatBoost = 0;
  if (_musicBpm > 0) {
    const elapsedS = (performance.now() - _beatPhaseUpdateMs) / 1000;
    if (elapsedS <= 2.0) {  // Schutz gegen Pi-down: simuliere keine Musik die laengst aus ist
      const beatPeriodS = 60.0 / _musicBpm;
      const extrapolatedPhase = (_musicBeatPhase + elapsedS / beatPeriodS) % 1.0;
      beatBoost = BEAT_PULSE_AMPLITUDE * Math.max(0, Math.cos(extrapolatedPhase * Math.PI * 2));
    }
  }

  // Apply Emissive (state-morph base + music-beat overlay)
  if (sceneObjects.material && sceneObjects.material.emissiveIntensity !== undefined) {
    sceneObjects.material.emissiveIntensity = _currentMorph.emissive_intensity + beatBoost;
  }
}

export function getCurrentMorph() { return { ..._currentMorph }; }
export function getDeformFactor() { return _currentMorph.deform; }
