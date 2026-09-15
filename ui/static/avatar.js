// Companion avatar: a stylized 3D head built from primitives in three.js (no model files).
// It floats, blinks and glances around; its mouth bars follow the real playback loudness; it leans in while the user
// talks and looks up while thinking. Its face, colour and motion follow the AI's emotion tag: eyebrows, eye shape,
// mouth curve, head tilt, bob, glow and a "?" or "!" when confused or surprised.

import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { EMOTIONS } from "./emotions.js";

const HEAD_SCALE = new THREE.Vector3(1.12, 0.95, 0.95);
const VISOR_R = 1.075;
const GLOW = 3.2; // HDR multiplier so glowing parts bloom
const FOCUS_COLORS = {
  good: new THREE.Color("#46e6a6"),
  warn: new THREE.Color("#ffb347"),
  bad: new THREE.Color("#ff5c4d"),
  nocam: new THREE.Color("#6f7d99"),
};
const EMOTION_COLORS = Object.fromEntries(Object.entries(EMOTIONS).map(([name, e]) => [name, new THREE.Color(e.color)]));
const BAR_SHAPE = [0.42, 0.7, 0.9, 1.0, 0.9, 0.7, 0.42];
const PANEL_MAX = 470;
const PANEL_FRACTION = 0.38;
const GAP = 24;
const AVATAR_HALF_WIDTH = 1.75; // outer halo radius plus a margin, in world units

// Face shape per emotion. brow: + raises the inner ends (sad, worried), - lowers them (angry). eyeTilt: + leans the
// eye tops outward (fierce), - inward (sad). arc: left/right eye as a happy "∩" arc. smile: mouth curve -1..1.
// slant/wave: crooked or wobbly mouth. tilt/pitch: head roll and nod (+ = down). bob/speed: floating motion.
const NEUTRAL_FACE = {
  brow: 0, browLift: 0, browAsym: 0, eyeOpen: 1, eyeTilt: 0, arcL: 0, arcR: 0, smile: 0.35, mouthOpen: 0, slant: 0,
  wave: 0, tilt: 0, pitch: 0, sink: 0, lookX: 0, lookY: 0, bob: 1, speed: 1, tremble: 0, glow: 1, glyph: "",
};
const FACE = {
  neutral: {},
  happy: { brow: 0.08, browLift: 0.03, arcL: 1, arcR: 1, smile: 1, mouthOpen: 0.1, tilt: 0.06, pitch: -0.02, bob: 1.3, speed: 1.3, glow: 1.15 },
  excited: { brow: 0.05, browLift: 0.07, eyeOpen: 1.25, smile: 1, mouthOpen: 0.45, pitch: -0.05, bob: 1.9, speed: 1.9, glow: 1.3 },
  proud: { brow: -0.05, browLift: 0.02, arcL: 1, arcR: 1, smile: 0.85, pitch: -0.13, bob: 1.1, glow: 1.15 },
  playful: { browLift: 0.01, browAsym: 0.07, arcR: 1, smile: 0.8, slant: 0.35, tilt: 0.14, bob: 1.4, speed: 1.4, glow: 1.1 },
  caring: { brow: 0.22, browLift: 0.01, eyeOpen: 0.85, eyeTilt: -0.08, smile: 0.55, tilt: 0.1, pitch: 0.05, bob: 0.8, speed: 0.75, glow: 0.95 },
  thoughtful: { brow: 0.06, browLift: 0.02, browAsym: 0.05, eyeOpen: 0.8, smile: 0.1, slant: 0.25, tilt: -0.1, pitch: -0.12, lookX: 0.045, lookY: 0.04, bob: 0.7, speed: 0.7, glow: 0.95 },
  confused: { brow: -0.05, browLift: 0.02, browAsym: 0.1, eyeOpen: 1.05, smile: -0.15, slant: 0.5, wave: 1, tilt: 0.22, bob: 0.9, speed: 0.9, glyph: "?" },
  surprised: { browLift: 0.1, eyeOpen: 1.4, smile: 0, mouthOpen: 1, pitch: -0.06, bob: 1.2, speed: 1.2, glow: 1.25, glyph: "!" },
  sad: { brow: 0.38, browLift: -0.01, eyeOpen: 0.75, eyeTilt: -0.18, smile: -0.9, tilt: -0.06, pitch: 0.2, sink: 0.1, bob: 0.5, speed: 0.55, glow: 0.6 },
  annoyed: { brow: -0.25, browLift: -0.02, browAsym: 0.02, eyeOpen: 0.62, eyeTilt: 0.14, smile: -0.4, slant: 0.2, tilt: -0.05, pitch: 0.06, bob: 0.6, speed: 0.8 },
  angry: { brow: -0.5, browLift: -0.035, eyeOpen: 0.75, eyeTilt: 0.3, smile: -1, mouthOpen: 0.2, pitch: 0.12, bob: 0.35, speed: 0.6, tremble: 0.022, glow: 1.35 },
};
const faceFor = (name) => ({ ...NEUTRAL_FACE, ...(FACE[name] || {}) });

const damp = (current, target, rate, dt) => current + (target - current) * (1 - Math.exp(-rate * dt));

// A pivot sitting on the visor surface, facing outward, so features can move in its local plane.
function pivotOnVisor(parent, x, y, lift = 0.012) {
  const nx = x / HEAD_SCALE.x;
  const ny = y / HEAD_SCALE.y;
  const nz = Math.sqrt(Math.max(0, VISOR_R * VISOR_R - nx * nx - ny * ny));
  const normal = new THREE.Vector3(nx / HEAD_SCALE.x, ny / HEAD_SCALE.y, nz / HEAD_SCALE.z).normalize();
  const pivot = new THREE.Group();
  pivot.position.set(x, y, nz * HEAD_SCALE.z).addScaledVector(normal, lift);
  pivot.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
  parent.add(pivot);
  return pivot;
}

const glow = (color, strength = GLOW) => new THREE.MeshBasicMaterial({ color: color.clone().multiplyScalar(strength) });

function glyphSprite(char) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 128;
  const g = canvas.getContext("2d");
  g.fillStyle = "#ffffff";
  g.font = "800 104px -apple-system, 'Segoe UI', Arial, sans-serif";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(char, 64, 72);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false }));
  sprite.scale.setScalar(0.001);
  return sprite;
}

export function createAvatar(canvas) {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const motion = reducedMotion ? 0.35 : 1;

  // ---------------------------------------------------------------- renderer, scene, camera
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.9;

  const scene = new THREE.Scene();
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.55;
  pmrem.dispose();

  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
  camera.position.set(0, 0.15, 8);

  // Backdrop: deep blue radial glow, tinted by the current emotion.
  const backdropUniforms = { uTint: { value: EMOTION_COLORS.neutral.clone() }, uTintAmount: { value: 0.3 } };
  const backdrop = new THREE.Mesh(
    new THREE.PlaneGeometry(44, 26),
    new THREE.ShaderMaterial({
      uniforms: backdropUniforms,
      depthWrite: false,
      vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
      fragmentShader: `
        varying vec2 vUv;
        uniform vec3 uTint;
        uniform float uTintAmount;
        void main() {
          vec2 p = vUv - vec2(0.5, 0.52);
          p.x *= 1.7;
          float d = length(p);
          vec3 base = mix(vec3(0.020, 0.028, 0.055), vec3(0.002, 0.003, 0.007), smoothstep(0.02, 0.40, d));
          float halo = exp(-d * d * 55.0);
          vec3 col = base + vec3(0.030, 0.050, 0.110) * halo + uTint * uTintAmount * halo * 0.22;
          gl_FragColor = vec4(col, 1.0);
        }`,
    }),
  );
  backdrop.position.z = -8;
  scene.add(backdrop);

  scene.add(new THREE.HemisphereLight(0x9fb4ff, 0x080b14, 0.35));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(2.5, 3.5, 5);
  scene.add(key);
  const rimCyan = new THREE.PointLight(0x4fd4ff, 22, 14, 2);
  rimCyan.position.set(-3.2, 1.4, -1.2);
  scene.add(rimCyan);
  const rimViolet = new THREE.PointLight(0x8f6bff, 20, 14, 2);
  rimViolet.position.set(3.2, -0.6, -1.4);
  scene.add(rimViolet);

  // ---------------------------------------------------------------- the character
  const avatar = new THREE.Group();
  scene.add(avatar);
  const head = new THREE.Group();
  avatar.add(head);

  const shell = new THREE.Mesh(
    new THREE.SphereGeometry(1.05, 96, 64),
    new THREE.MeshPhysicalMaterial({
      color: 0xc3cbd9, roughness: 0.38, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.22,
      sheen: 0.25, sheenColor: new THREE.Color(0xa8c0ff), sheenRoughness: 0.6,
    }),
  );
  shell.scale.copy(HEAD_SCALE);
  head.add(shell);

  const visor = new THREE.Mesh(
    new THREE.SphereGeometry(VISOR_R, 96, 48, Math.PI / 2 - 0.74, 1.48, 1.02, 1.12),
    new THREE.MeshPhysicalMaterial({
      color: 0x03050b, roughness: 0.18, metalness: 0.2, clearcoat: 1, clearcoatRoughness: 0.05, envMapIntensity: 0.35,
    }),
  );
  visor.scale.copy(HEAD_SCALE);
  head.add(visor);

  const featureColor = EMOTION_COLORS.neutral.clone();
  const eyeMat = glow(featureColor);
  const eyeGeo = new THREE.CapsuleGeometry(0.075, 0.13, 8, 20);
  const arcGeo = new THREE.TorusGeometry(0.085, 0.024, 10, 32, Math.PI); // "∩": a closed, smiling eye
  const browMat = glow(featureColor, GLOW * 0.8);
  const browGeo = new THREE.CapsuleGeometry(0.02, 0.17, 6, 12);
  const eyes = [-0.34, 0.34].map((x) => {
    const side = Math.sign(x);
    const eye = new THREE.Mesh(eyeGeo, eyeMat);
    eye.scale.z = 0.35;
    pivotOnVisor(head, x, 0.13).add(eye);
    const arc = new THREE.Mesh(arcGeo, eyeMat);
    arc.position.y = -0.03;
    arc.scale.setScalar(0.001);
    pivotOnVisor(head, x, 0.13).add(arc);
    const brow = new THREE.Group();
    pivotOnVisor(head, x, 0.37).add(brow);
    const browBar = new THREE.Mesh(browGeo, browMat);
    browBar.rotation.z = Math.PI / 2;
    browBar.scale.z = 0.4;
    brow.add(browBar);
    return { eye, arc, brow, side };
  });

  const mouth = pivotOnVisor(head, 0, -0.27);
  const barMat = glow(featureColor, GLOW * 0.9);
  const barGeo = new THREE.CapsuleGeometry(0.022, 0.16, 6, 12); // ~0.204 tall
  const bars = BAR_SHAPE.map((shape, i) => {
    const bar = new THREE.Mesh(barGeo, barMat);
    bar.position.x = (i - 3) * 0.062;
    bar.scale.set(1, 0.1, 0.4);
    mouth.add(bar);
    return { bar, shape, phase: i * 1.37, speed: 10 + i * 1.9 };
  });

  const trimMat = new THREE.MeshPhysicalMaterial({ color: 0xcfd7e6, roughness: 0.35, metalness: 0.25, clearcoat: 0.6 });
  const earRingMat = glow(featureColor, 1.4);
  for (const side of [-1, 1]) {
    const ear = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.26, 0.2, 48), trimMat);
    ear.rotation.z = Math.PI / 2;
    ear.position.set(side * 1.16, 0.02, 0);
    head.add(ear);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.17, 0.02, 12, 64), earRingMat);
    ring.rotation.y = Math.PI / 2;
    ring.position.set(side * 1.27, 0.02, 0);
    head.add(ring);
  }

  const stalk = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.035, 0.42, 16), trimMat);
  stalk.position.set(0, 1.12, 0);
  head.add(stalk);
  const tipMat = glow(new THREE.Color("#b48cff"), 1.6);
  const tip = new THREE.Mesh(new THREE.SphereGeometry(0.075, 24, 16), tipMat);
  tip.position.set(0, 1.36, 0);
  head.add(tip);

  const neckMat = glow(featureColor, 2);
  const neck = new THREE.Mesh(new THREE.TorusGeometry(0.34, 0.03, 16, 64), neckMat);
  neck.rotation.x = Math.PI / 2;
  neck.position.y = -1.08;
  avatar.add(neck);

  const haloMat = new THREE.MeshBasicMaterial({ color: featureColor.clone().multiplyScalar(GLOW), transparent: true, opacity: 0.95 });
  const halo = new THREE.Mesh(new THREE.TorusGeometry(1.25, 0.018, 12, 160), haloMat);
  halo.rotation.x = Math.PI / 2;
  halo.position.y = -1.5;
  avatar.add(halo);
  // The thin outer ring keeps showing the user's focus (tutor mode), independent of the AI's mood.
  const outerHaloMat = new THREE.MeshBasicMaterial({ color: FOCUS_COLORS.nocam.clone(), transparent: true, opacity: 0.4 });
  const outerHalo = new THREE.Mesh(new THREE.TorusGeometry(1.6, 0.008, 8, 160), outerHaloMat);
  outerHalo.rotation.x = Math.PI / 2;
  outerHalo.position.y = -1.52;
  avatar.add(outerHalo);

  const glyphs = { "?": glyphSprite("?"), "!": glyphSprite("!") };
  for (const sprite of Object.values(glyphs)) {
    sprite.position.set(1.12, 1.3, 0.3);
    avatar.add(sprite);
  }

  const dustCount = reducedMotion ? 120 : 420;
  const dustPositions = new Float32Array(dustCount * 3);
  for (let i = 0; i < dustCount; i++) {
    const r = 1.8 + Math.random() * 1.7;
    const a = Math.random() * Math.PI * 2;
    dustPositions.set([Math.cos(a) * r, (Math.random() - 0.5) * 2.8, Math.sin(a) * r], i * 3);
  }
  const dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute("position", new THREE.BufferAttribute(dustPositions, 3));
  const dustBase = new THREE.Color(0x8fb6ff);
  const dustMat = new THREE.PointsMaterial({
    color: dustBase.clone(), size: 0.03, transparent: true, opacity: 0.45, depthWrite: false, blending: THREE.AdditiveBlending,
  });
  const dust = new THREE.Points(dustGeo, dustMat);
  avatar.add(dust);

  // ---------------------------------------------------------------- post-processing + layout
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  // Threshold above normally lit surfaces, so only the HDR glow parts (eyes, mouth, rings) bloom.
  const bloom = new UnrealBloomPass(new THREE.Vector2(512, 512), 0.6, 0.4, 1.05);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  function resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h, false);
    composer.setSize(w, h);
    camera.aspect = w / h;
    const wide = w > 900;
    // Centre the character in the space left of the conversation panel (or above it on narrow screens), and back
    // the camera off until the whole character (halo included) fits that space's width.
    const panel = wide ? Math.min(PANEL_MAX, w * PANEL_FRACTION) + GAP : 0;
    const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
    const fitDistance = AVATAR_HALF_WIDTH / (Math.tan(halfFov) * ((w - panel) / h));
    camera.position.z = Math.max(wide ? 9.6 : 14, fitDistance);
    camera.setViewOffset(w, h, panel / 2, wide ? 0 : h * 0.2, w, h);
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  // Adaptive quality: slower GPUs (common on Windows laptops) step down to a lower resolution, then no glow, instead
  // of stuttering and taking CPU from the voice. Calm moments also render at 30 fps instead of 60.
  const QUALITY = [
    { ratio: Math.min(window.devicePixelRatio || 1, 1.75), bloom: true },
    { ratio: 1, bloom: true },
    { ratio: 1, bloom: false },
  ];
  let quality = 0;
  let lastRender = 0;
  let sampled = 0;
  let slowFrames = 0;
  function trackSpeed(intervalMs, calm) {
    sampled++;
    if (intervalMs > (calm ? 45 : 30)) slowFrames++;
    if (sampled < 90) return;
    if (slowFrames > sampled / 2 && quality < QUALITY.length - 1) {
      quality++;
      renderer.setPixelRatio(QUALITY[quality].ratio);
      composer.setPixelRatio(QUALITY[quality].ratio);
      bloom.enabled = QUALITY[quality].bloom;
      resize();
      console.info(`avatar: lowered quality to level ${quality} (${QUALITY[quality].ratio}x, glow ${QUALITY[quality].bloom ? "on" : "off"})`);
    }
    sampled = slowFrames = 0;
  }

  const pointer = new THREE.Vector2();
  window.addEventListener("pointermove", (e) => {
    pointer.set((e.clientX / window.innerWidth) * 2 - 1, (e.clientY / window.innerHeight) * 2 - 1);
  });

  // ---------------------------------------------------------------- behaviour
  const s = {
    mode: "listening", focus: "nocam", emotion: "neutral", tutor: 0, user: 0, tutorS: 0, userS: 0,
    blink: 0, nextBlink: 1.5, look: new THREE.Vector2(), glance: new THREE.Vector2(), nextGlance: 1,
    react: null, reactT: 0, floatPhase: 0,
  };
  const face = faceFor("neutral");
  const lookTarget = new THREE.Vector2();
  const clock = new THREE.Clock();

  function frame() {
    const nowMs = performance.now();
    const calm = s.tutor < 0.01 && s.mode !== "user" && !s.react;
    if (calm && nowMs - lastRender < 32) return;
    if (lastRender) trackSpeed(nowMs - lastRender, calm);
    lastRender = nowMs;
    const dt = Math.min(clock.getDelta(), 0.05);
    const t = clock.elapsedTime;
    s.tutorS = damp(s.tutorS, s.tutor, 22, dt);
    s.userS = damp(s.userS, s.user, 12, dt);
    const speaking = s.mode === "speaking";
    const thinking = s.mode === "thinking";
    const hearing = s.mode === "user";

    // Ease every face parameter toward the current emotion's shape.
    const target = faceFor(s.emotion);
    for (const k of Object.keys(NEUTRAL_FACE)) {
      if (k !== "glyph") face[k] = damp(face[k], target[k], 5, dt);
    }
    featureColor.lerp(EMOTION_COLORS[s.emotion] || EMOTION_COLORS.neutral, 1 - Math.exp(-3.5 * dt));

    // Reactions to one-off events.
    let bounce = 0;
    let shake = 0;
    if (s.react) {
      s.reactT += dt;
      const k = s.reactT;
      if (s.react === "nudge" || s.react === "hello" || s.react === "pop") bounce = Math.sin(k * 12) * 0.13 * Math.exp(-k * 2.6);
      if (s.react === "interrupt" || s.react === "stomp") {
        shake = Math.sin(k * 38) * 0.06 * Math.exp(-k * 7);
        if (s.react === "interrupt" && k < 0.06) s.blink = 1;
      }
      if (k > 1.6) s.react = null;
    }

    // Body float and head pose.
    s.floatPhase += dt * 1.15 * face.speed;
    avatar.position.y = 0.22 - face.sink + Math.sin(s.floatPhase) * 0.07 * face.bob * motion + bounce;
    avatar.rotation.y = Math.sin(t * 0.35) * 0.05 * motion;
    const nod = Math.sin(t * 8.5) * s.tutorS * 0.04;
    head.rotation.x = damp(head.rotation.x, -pointer.y * 0.1 + (thinking ? -0.16 : 0) + (hearing ? 0.07 : 0) + face.pitch + nod, 5, dt);
    head.rotation.y = damp(head.rotation.y, pointer.x * 0.3 + (thinking ? 0.22 : 0), 5, dt) + shake;
    head.rotation.z = damp(head.rotation.z, (hearing ? 0.1 : 0) + face.tilt + Math.sin(t * 0.8) * 0.02 * motion, 4, dt);
    head.position.x = Math.sin(t * 55) * face.tremble * motion;
    head.scale.setScalar(1 + Math.sin(t * 2.2) * 0.008 * motion + s.tutorS * 0.015);

    // Eyes: blink, glance around, emotion shape.
    s.nextBlink -= dt;
    if (s.nextBlink <= 0) {
      s.blink = 1;
      s.nextBlink = 2.2 + Math.random() * 3.2;
    }
    s.blink = Math.max(0, s.blink - dt * 6.5);
    const lid = 1 - Math.sin(Math.min(1, s.blink) * Math.PI);
    s.nextGlance -= dt;
    if (s.nextGlance <= 0) {
      s.nextGlance = 1.2 + Math.random() * 2.8;
      s.glance.set((Math.random() - 0.5) * 0.06, (Math.random() - 0.5) * 0.035);
    }
    if (thinking) lookTarget.set(0.045, 0.045);
    else if (hearing || speaking) lookTarget.set(face.lookX, face.lookY);
    else lookTarget.set(s.glance.x + face.lookX, s.glance.y + face.lookY);
    s.look.x = damp(s.look.x, lookTarget.x + pointer.x * 0.02, 10, dt);
    s.look.y = damp(s.look.y, lookTarget.y - pointer.y * 0.012, 10, dt);
    const open = (hearing ? 1.18 : thinking ? 0.8 : speaking ? 0.92 + s.tutorS * 0.1 : 1) * face.eyeOpen;
    for (const { eye, arc, brow, side } of eyes) {
      const arcAmount = side < 0 ? face.arcL : face.arcR;
      eye.position.set(s.look.x, s.look.y, 0);
      eye.visible = arcAmount < 0.97;
      eye.scale.y = Math.max(0.05, lid * open * (1 - arcAmount));
      eye.rotation.z = -side * face.eyeTilt;
      arc.visible = arcAmount > 0.03;
      arc.position.set(s.look.x, s.look.y - 0.03, 0);
      arc.scale.set(arcAmount, Math.max(0.05, arcAmount * lid), 0.35 * arcAmount);
      // Brows sit above the eyes; the left one also lifts for a quizzical look.
      brow.position.set(s.look.x * 0.5, face.browLift + (side < 0 ? face.browAsym : 0) + (hearing ? 0.015 : 0), 0);
      brow.rotation.z = -side * face.brow + (side < 0 ? face.browAsym * 1.5 : 0);
    }

    const glowBoost = face.glow;
    eyeMat.color.copy(featureColor).multiplyScalar(GLOW * glowBoost);
    browMat.color.copy(featureColor).multiplyScalar(GLOW * 0.8 * glowBoost);
    barMat.color.copy(featureColor).multiplyScalar(GLOW * 0.9 * glowBoost);
    neckMat.color.copy(featureColor).multiplyScalar((1.6 + s.tutorS * 1.5) * glowBoost);
    earRingMat.color.copy(featureColor).multiplyScalar((1.1 + (hearing ? s.userS * 3.5 : 0) + s.tutorS * 1.5) * glowBoost);

    // Mouth: bars bounce with playback loudness; the emotion curves, opens, slants or wobbles them.
    const quiet = s.tutorS < 0.02;
    bars.forEach(({ bar, shape, phase, speed }, i) => {
      const wobble = 0.62 + 0.38 * Math.sin(t * speed + phase);
      const rest = 0.1 + face.mouthOpen * shape * 0.9;
      bar.scale.y = damp(bar.scale.y, quiet ? rest : 0.16 + s.tutorS * 1.9 * shape * wobble + face.mouthOpen * 0.3 * shape, 30, dt);
      const edge = (i - 3) / 3;
      const curve = edge * edge * (quiet ? 0.035 : 0.02) * face.smile;
      const crooked = edge * face.slant * 0.03 + Math.sin(i * 1.6 + t * 4) * face.wave * 0.012;
      bar.position.y = damp(bar.position.y, curve + crooked, 12, dt);
    });

    // Glyph: a "?" when confused, a "!" when surprised.
    for (const [char, sprite] of Object.entries(glyphs)) {
      const shown = FACE[s.emotion]?.glyph === char;
      sprite.scale.setScalar(damp(sprite.scale.x, shown ? 0.5 : 0.001, 7, dt));
      sprite.visible = sprite.scale.x > 0.01;
      sprite.position.y = 1.3 + Math.sin(t * 3) * 0.05 * motion;
      sprite.material.rotation = Math.sin(t * 2) * 0.15 * motion;
      sprite.material.color.copy(featureColor).multiplyScalar(2.2);
    }

    // Antenna, halo, backdrop, dust, rim light.
    const angry = s.emotion === "angry" || s.emotion === "annoyed";
    const tipPulse = thinking ? 2.4 + 2.6 * (0.5 + 0.5 * Math.sin(t * 9))
      : angry ? 2.4 + 2.2 * (0.5 + 0.5 * Math.sin(t * 14)) : 2.2 + Math.sin(t * 2) * 0.4;
    tipMat.color.copy(featureColor).multiplyScalar(tipPulse * glowBoost);
    haloMat.color.copy(featureColor).multiplyScalar(GLOW * (0.75 + s.tutorS * 0.9) * glowBoost);
    halo.scale.setScalar(1 + s.tutorS * 0.06 + Math.sin(t * 1.6) * 0.01 * motion);
    outerHaloMat.color.lerp(FOCUS_COLORS[s.focus] || FOCUS_COLORS.nocam, 1 - Math.exp(-4 * dt));
    outerHalo.rotation.z += dt * 0.15 * motion;
    backdropUniforms.uTint.value.copy(featureColor);
    backdropUniforms.uTintAmount.value = damp(backdropUniforms.uTintAmount.value, 0.35 + Math.max(0, glowBoost - 1) * 1.6, 2, dt);
    dust.rotation.y += dt * (thinking ? 0.55 : 0.08 * face.speed) * motion;
    dustMat.color.copy(dustBase).lerp(featureColor, 0.55);
    dustMat.opacity = 0.35 + s.tutorS * 0.4 + (thinking ? 0.2 : 0);
    rimCyan.intensity = 22 + s.tutorS * 30;

    composer.render();
  }
  renderer.setAnimationLoop(frame);

  canvas.addEventListener("webglcontextlost", (e) => {
    e.preventDefault();
    console.warn("WebGL context lost; the avatar will resume when the browser restores it");
  });

  return {
    setLevels(tutor, user) {
      s.tutor = Number(tutor) || 0;
      s.user = Number(user) || 0;
    },
    setMode(mode) { s.mode = mode; },
    setFocus(kind) { s.focus = kind; },
    setEmotion(name) {
      if (!FACE[name] || name === s.emotion) return;
      s.emotion = name;
      if (name === "surprised" || name === "excited") this.react("pop");
      if (name === "angry") this.react("stomp");
    },
    react(kind) {
      s.react = kind;
      s.reactT = 0;
    },
  };
}
