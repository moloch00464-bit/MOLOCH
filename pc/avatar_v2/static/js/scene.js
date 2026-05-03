// MOLOCH Avatar 2.0 — Three.js Scene Setup
// Phase 1: Placeholder-Geometry (IcosahedronGeometry).
// Phase 3: MOLOCH_v1.glb Modell-Loader (TODO).

export function initScene(canvas) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000010);
  scene.fog = new THREE.Fog(0x000010, 5, 25);

  const aspect = window.innerWidth / window.innerHeight;
  const camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 100);
  camera.position.set(0, 0, 7);
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    antialias: true,
    alpha: false,
  });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  // ===== Lighting =====
  const ambient = new THREE.AmbientLight(0x223355, 0.6);
  scene.add(ambient);

  const keyLight = new THREE.PointLight(0x66ccff, 1.5, 20);
  keyLight.position.set(3, 4, 5);
  scene.add(keyLight);

  const rimLight = new THREE.PointLight(0xff66ee, 1.0, 20);
  rimLight.position.set(-3, -2, 3);
  scene.add(rimLight);

  // ===== Avatar-Mesh (Phase-1 Placeholder) =====
  // Spaeter Modell-Load: const loader = new GLTFLoader(); loader.load('MOLOCH_v1.glb', ...);
  // Jetzt: Icosahedron als Stand-In, mit eigenem Shader-Material
  const geometry = new THREE.IcosahedronGeometry(1.5, 4);
  const material = new THREE.MeshStandardMaterial({
    color: 0x66aaff,
    emissive: 0x113355,
    emissiveIntensity: 0.4,
    roughness: 0.3,
    metalness: 0.7,
    flatShading: true,
  });
  const avatar = new THREE.Mesh(geometry, material);
  scene.add(avatar);

  // ===== Particle Aura (Phase 1: Sterne) =====
  const particleCount = 600;
  const positions = new Float32Array(particleCount * 3);
  for (let i = 0; i < particleCount; i++) {
    positions[i*3 + 0] = (Math.random() - 0.5) * 20;
    positions[i*3 + 1] = (Math.random() - 0.5) * 20;
    positions[i*3 + 2] = (Math.random() - 0.5) * 20;
  }
  const particleGeometry = new THREE.BufferGeometry();
  particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const particleMaterial = new THREE.PointsMaterial({
    color: 0x6688aa,
    size: 0.05,
    transparent: true,
    opacity: 0.6,
  });
  const particles = new THREE.Points(particleGeometry, particleMaterial);
  scene.add(particles);

  // Resize handler
  window.addEventListener('resize', () => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });

  return {
    scene, camera, renderer,
    avatar, material, geometry,
    particles, particleMaterial,
    keyLight, rimLight, ambient,
    clock: new THREE.Clock(),
  };
}

let _renderer = null;
export function getRenderer() { return _renderer; }

let _frameCount = 0;
let _lastFpsUpdate = 0;
let _currentFps = 0;
const _fpsListeners = [];

export function onFpsUpdate(cb) { _fpsListeners.push(cb); }
export function getCurrentFps() { return _currentFps; }

export function renderLoop(sceneObjects) {
  _renderer = sceneObjects.renderer;
  function tick() {
    requestAnimationFrame(tick);
    sceneObjects.renderer.render(sceneObjects.scene, sceneObjects.camera);

    _frameCount++;
    const now = performance.now();
    if (now - _lastFpsUpdate >= 1000) {
      _currentFps = Math.round((_frameCount * 1000) / (now - _lastFpsUpdate));
      _frameCount = 0;
      _lastFpsUpdate = now;
      const fpsEl = document.getElementById('hud-fps');
      if (fpsEl) fpsEl.textContent = 'FPS: ' + _currentFps;
      for (const cb of _fpsListeners) cb(_currentFps);
    }
  }
  _lastFpsUpdate = performance.now();
  tick();
}
