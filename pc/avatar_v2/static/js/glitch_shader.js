// MOLOCH Avatar 2.0 — Glitch-Shader fuer berserker / overloaded
// Vertex-Distortion via Custom-Shader-Material (Three.js ShaderMaterial)
// Phase 1: einfache sin/noise-Distortion entlang Normalen
// Phase 3: full GLSL noise + RGB-Split + scanlines

import { getDeformFactor } from '/static/js/state_morph.js';

const GLITCH_VS = `
  uniform float uTime;
  uniform float uDeform;
  varying vec3 vNormal;
  varying vec2 vUv;

  // Pseudo-random hash
  float hash(vec3 p) {
    p = fract(p * vec3(443.8975, 397.2973, 491.1871));
    p += dot(p, p.yxz + 19.19);
    return fract((p.x + p.y) * p.z);
  }

  void main() {
    vNormal = normal;
    vUv = uv;

    vec3 pos = position;

    // Sinus-Distortion entlang Normale
    float wave = sin(pos.y * 8.0 + uTime * 3.0) * uDeform * 0.15;
    // Noise-Distortion fuer Glitch-Sprenkler
    float noise = (hash(pos + vec3(uTime)) - 0.5) * uDeform * 0.4;

    pos += normal * (wave + noise);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
  }
`;

const GLITCH_FS = `
  uniform vec3 uBaseColor;
  uniform vec3 uEmissive;
  uniform float uEmissiveIntensity;
  uniform float uTime;
  varying vec3 vNormal;
  varying vec2 vUv;

  void main() {
    // Simple Lambert-ish shading
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.8));
    float diff = max(dot(vNormal, lightDir), 0.0);
    vec3 col = uBaseColor * (0.4 + 0.6 * diff);

    // Emissive
    col += uEmissive * uEmissiveIntensity;

    // RGB-Split-Glitch (zeitabhaengig, leicht)
    float glitchAmount = 0.02 * sin(uTime * 7.3);
    col.r += glitchAmount;
    col.b -= glitchAmount;

    gl_FragColor = vec4(col, 1.0);
  }
`;

let _glitchMaterial = null;
let _enabled = false;

/**
 * Initialisiert ShaderMaterial. Wird statt MeshStandardMaterial gesetzt
 * sobald glitch-mode aktiv (overloaded/berserker mit Tension>0.7).
 *
 * Aktuell deaktiviert per Default — Standard-Material aus scene.js wird genutzt.
 * Aktivieren via toggleGlitch(true).
 */
export function initGlitchShader(sceneObjects) {
  _glitchMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0.0 },
      uDeform: { value: 0.0 },
      uBaseColor: { value: new THREE.Color(0xff3344) },
      uEmissive: { value: new THREE.Color(0x550011) },
      uEmissiveIntensity: { value: 0.6 },
    },
    vertexShader: GLITCH_VS,
    fragmentShader: GLITCH_FS,
  });
  return _glitchMaterial;
}

export function toggleGlitch(sceneObjects, enable) {
  if (!_glitchMaterial) initGlitchShader(sceneObjects);
  if (enable && !_enabled) {
    sceneObjects.avatar.material = _glitchMaterial;
    _enabled = true;
  } else if (!enable && _enabled) {
    sceneObjects.avatar.material = sceneObjects.material;
    _enabled = false;
  }
}

/** Per-Frame-Tick: Time-Update + Deform aus state_morph. */
export function tickGlitchShader(sceneObjects, dt, elapsed) {
  if (!_enabled || !_glitchMaterial) return;
  _glitchMaterial.uniforms.uTime.value = elapsed;
  _glitchMaterial.uniforms.uDeform.value = getDeformFactor();
}

/** Auto-Toggle basierend auf state. */
export function autoToggleGlitch(sceneObjects, state) {
  const piState = state?.pi?.state || {};
  const currentState = piState.current_state || piState.zone || 'idle';
  const tension = piState.tension ?? 0;

  const shouldGlitch =
    currentState === 'overloaded' ||
    currentState === 'berserker' ||
    (currentState === 'engaged' && tension > 0.7);

  toggleGlitch(sceneObjects, shouldGlitch);
}
