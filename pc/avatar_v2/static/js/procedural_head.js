// MOLOCH Avatar 2.0 — Stufe 2: Procedural Low-Poly Head
// Inspired by Markus' Max-Headroom-Bild: Pompadour-Hair + facettiertes Gesicht
// + leuchtende Cyan-Augen + Lacheln + Rolli-Neck. Alles Three.js Primitives,
// die GPU rendert - genau das was Markus wollte (3D-Spiel-Feeling).
//
// Export: createMolochHead(palette) -> { group, material, eyeMaterials, mouthMaterial }
//   palette = optional {face, hair, eye, neck, ...} fuer Override
// State-Morph wirkt auf group.scale + material.emissiveIntensity (wie vorher).

const DEFAULT_PALETTE = {
  face:    0xffd9b8,  // hautfarbe
  faceShadow: 0x9a7a60,
  hair:    0xc8a8d0,  // lila-rosa Pompadour
  hairHi:  0xf5d8e8,  // hair highlights
  eye:     0x33ccff,  // cyan emissive
  eyeWhite: 0xffffff,
  mouth:   0xa04050,  // lippen
  mouthHi: 0xdd6677,
  neck:    0x222244,  // dunkler Rolli
  brow:    0x3a2010,  // Augenbrauen
};

/**
 * Erstellt einen Low-Poly Avatar-Kopf als THREE.Group.
 * Returns:
 *   group         - THREE.Group, fuegt sich in scene
 *   material      - "Hauptmaterial" (Face) - State-Morph ueberschreibt emissive
 *   eyeMaterials  - Array von Eye-Materialien (separates emissive je State)
 *   mouthMaterial - Mouth-Material
 */
export function createMolochHead(paletteOverride) {
  const p = Object.assign({}, DEFAULT_PALETTE, paletteOverride || {});

  const group = new THREE.Group();

  // ============ FACE ============
  // Low-Poly Sphere (IcosahedronGeometry mit detail=2 fuer facettiertes Aussehen)
  const faceGeo = new THREE.IcosahedronGeometry(1.0, 2);
  // Etwas zu Ei-Form ziehen
  const facePos = faceGeo.attributes.position;
  for (let i = 0; i < facePos.count; i++) {
    const y = facePos.getY(i);
    facePos.setY(i, y * 1.15);  // vertikal strecken
    const x = facePos.getX(i);
    facePos.setX(i, x * 0.95);  // horizontal etwas dünner
  }
  faceGeo.computeVertexNormals();

  // Markus' Reference-Bild als Texture aufs Low-Poly Face-Mesh.
  // Kombi: 3D-Form vom Procedural + Texture-Look vom echten Bild.
  // GTX 760 Tuning: Texture-Sampling ist guenstiger als komplexer per-pixel emissive.
  const _texLoader = new THREE.TextureLoader();
  const _faceTexture = _texLoader.load('/static/img/moloch_reference.png');
  _faceTexture.colorSpace = THREE.SRGBColorSpace;
  const faceMat = new THREE.MeshStandardMaterial({
    map: _faceTexture,         // Bild ueberzieht das Mesh
    color: 0xffffff,           // neutral, laesst Texture durch
    emissive: 0x221133,        // dezenter Magenta-Glow (halbiert vs Plan: bei state_morph IntensitÃ¤t 0.96 wuerde 0x442266 das Bild ueberlagern)
    emissiveIntensity: 0.4,
    roughness: 0.6,
    metalness: 0.1,
    flatShading: true,         // Low-Poly-Look bleibt (facettiertes Bild)
  });
  const face = new THREE.Mesh(faceGeo, faceMat);
  face.name = 'face';
  group.add(face);

  // ============ HAIR (Pompadour) ============
  // Cone + Capsule fuer Pompadour-Wave
  const hairMat = new THREE.MeshStandardMaterial({
    color: p.hair,
    emissive: 0x331144,
    emissiveIntensity: 0.4,
    roughness: 0.6,
    metalness: 0.2,
    flatShading: true,
  });

  // Hair-Cap (Halbkugel oben)
  const hairCapGeo = new THREE.SphereGeometry(0.95, 12, 8, 0, Math.PI * 2, 0, Math.PI * 0.55);
  const hairCap = new THREE.Mesh(hairCapGeo, hairMat);
  hairCap.position.set(0, 0.18, -0.05);
  hairCap.scale.set(1.05, 1.1, 1.05);
  hairCap.name = 'hairCap';
  group.add(hairCap);

  // Pompadour-Wave (oben aufgebauschter Kegel)
  const pompGeo = new THREE.ConeGeometry(0.55, 0.85, 6, 1);
  const pomp = new THREE.Mesh(pompGeo, hairMat);
  pomp.position.set(0, 1.15, -0.05);
  pomp.rotation.set(-0.15, 0, 0);
  pomp.scale.set(1.0, 1.0, 0.7);
  pomp.name = 'pomp';
  group.add(pomp);

  // Hair-Highlight-Wisps (heller, glaenzender)
  const wispMat = new THREE.MeshStandardMaterial({
    color: p.hairHi,
    emissive: 0x553366,
    emissiveIntensity: 0.5,
    roughness: 0.3,
    metalness: 0.4,
    flatShading: true,
  });
  for (let i = 0; i < 3; i++) {
    const wispGeo = new THREE.ConeGeometry(0.08, 0.4, 4, 1);
    const wisp = new THREE.Mesh(wispGeo, wispMat);
    const angle = -0.3 + i * 0.3;
    wisp.position.set(Math.sin(angle) * 0.3, 1.3, -0.1);
    wisp.rotation.set(-0.2, angle, 0);
    group.add(wisp);
  }

  // ============ EYES (Cyan emissive) ============
  const eyeMat = new THREE.MeshStandardMaterial({
    color: p.eye,
    emissive: p.eye,
    emissiveIntensity: 0.9,
    roughness: 0.2,
    metalness: 0.1,
  });
  const eyeWhiteMat = new THREE.MeshStandardMaterial({
    color: p.eyeWhite,
    emissive: 0x222244,
    emissiveIntensity: 0.2,
    roughness: 0.4,
  });

  const eyeMaterials = [];
  // Iris + Augapfel (links + rechts)
  for (const xSign of [-1, 1]) {
    // Augapfel-White (kleine Sphere)
    const eyeWhiteGeo = new THREE.SphereGeometry(0.12, 10, 8);
    const eyeWhite = new THREE.Mesh(eyeWhiteGeo, eyeWhiteMat);
    eyeWhite.position.set(xSign * 0.3, 0.18, 0.78);
    eyeWhite.scale.set(1.0, 0.85, 0.7);
    group.add(eyeWhite);

    // Iris (cyan, leuchtend)
    const irisGeo = new THREE.SphereGeometry(0.06, 10, 8);
    const irisMat = eyeMat.clone();
    const iris = new THREE.Mesh(irisGeo, irisMat);
    iris.position.set(xSign * 0.3, 0.16, 0.86);
    iris.scale.set(1.0, 0.95, 0.5);
    iris.name = (xSign === -1) ? 'eyeL' : 'eyeR';
    group.add(iris);
    eyeMaterials.push(irisMat);

    // Pupille (schwarz, kleiner Sphere innen)
    const pupGeo = new THREE.SphereGeometry(0.025, 8, 6);
    const pupMat = new THREE.MeshBasicMaterial({ color: 0x000000 });
    const pup = new THREE.Mesh(pupGeo, pupMat);
    pup.position.set(xSign * 0.3, 0.16, 0.91);
    group.add(pup);
  }

  // ============ EYEBROWS (Low-Poly Box) ============
  const browMat = new THREE.MeshStandardMaterial({
    color: p.brow,
    roughness: 0.8,
    flatShading: true,
  });
  for (const xSign of [-1, 1]) {
    const browGeo = new THREE.BoxGeometry(0.22, 0.05, 0.08);
    const brow = new THREE.Mesh(browGeo, browMat);
    brow.position.set(xSign * 0.3, 0.38, 0.82);
    brow.rotation.set(0, 0, xSign * -0.15);
    group.add(brow);
  }

  // ============ NOSE (Low-Poly Pyramide) ============
  const noseGeo = new THREE.ConeGeometry(0.15, 0.4, 4, 1);
  const noseMat = new THREE.MeshStandardMaterial({
    color: p.face,
    emissive: 0x110a05,
    emissiveIntensity: 0.2,
    roughness: 0.5,
    flatShading: true,
  });
  const nose = new THREE.Mesh(noseGeo, noseMat);
  nose.position.set(0, -0.05, 0.85);
  nose.rotation.set(-Math.PI / 2 + 0.3, 0, 0);
  nose.scale.set(0.7, 0.7, 1.0);
  group.add(nose);

  // ============ MOUTH (Lacheln) ============
  // Halbkreis-Torus als Smile
  const mouthGeo = new THREE.TorusGeometry(0.18, 0.04, 6, 12, Math.PI);
  const mouthMat = new THREE.MeshStandardMaterial({
    color: p.mouth,
    emissive: p.mouthHi,
    emissiveIntensity: 0.4,
    roughness: 0.4,
    flatShading: true,
  });
  const mouth = new THREE.Mesh(mouthGeo, mouthMat);
  mouth.position.set(0, -0.32, 0.78);
  mouth.rotation.set(0, 0, Math.PI);  // Halbkreis nach unten = Lacheln
  mouth.scale.set(1.0, 0.7, 1.0);
  mouth.name = 'mouth';
  group.add(mouth);

  // ============ NECK (Rolli) ============
  const neckGeo = new THREE.CylinderGeometry(0.55, 0.7, 0.5, 12, 1);
  const neckMat = new THREE.MeshStandardMaterial({
    color: p.neck,
    emissive: 0x111133,
    emissiveIntensity: 0.3,
    roughness: 0.7,
    metalness: 0.3,
    flatShading: true,
  });
  const neck = new THREE.Mesh(neckGeo, neckMat);
  neck.position.set(0, -1.0, 0.0);
  group.add(neck);

  // Rolli-Detail-Ringe
  for (let i = 0; i < 4; i++) {
    const ringGeo = new THREE.TorusGeometry(0.6, 0.025, 4, 16);
    const ring = new THREE.Mesh(ringGeo, neckMat);
    ring.position.set(0, -0.85 - i * 0.07, 0);
    ring.rotation.set(Math.PI / 2, 0, 0);
    group.add(ring);
  }

  // Slight Y-rotation for character
  group.rotation.y = -0.05;

  return {
    group,
    material: faceMat,
    eyeMaterials,
    mouthMaterial: mouthMat,
    hairMaterial: hairMat,
  };
}
