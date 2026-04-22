/**
 * embedding_3d.js — Three.js 3D embedding visualisation.
 *
 * Reduces 64-dim user-state embeddings to 3D via a fixed,
 * deterministic random projection (seeded xorshift). Each
 * incoming vector is pushed to a rolling buffer (max 200). The
 * most recent point is a pulsing accent-coloured mesh; older
 * points fade to muted along a trail.
 *
 * Soft-fails: if WebGL is unavailable *or* Three.js failed to
 * load (CDN blocked), we draw a static 2D projection on a plain
 * canvas so the panel never appears broken.
 */

import { showError } from "./loading_states.js";

const EMBED_DIM = 64;

/* Deterministic seeded random so the 64x3 projection is stable
   across page reloads. */
function xorshift(seed) {
  let s = seed >>> 0;
  return function () {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;  s >>>= 0;
    return (s >>> 0) / 4294967296;
  };
}
function buildProjection(dim = EMBED_DIM, out = 3) {
  const rnd = xorshift(0x13ADEAD);
  const P = new Float32Array(dim * out);
  for (let i = 0; i < P.length; i++) P[i] = (rnd() * 2 - 1) * 0.5;
  return P;
}
function project(vec, P) {
  const out = [0, 0, 0];
  const n = Math.min(vec.length, EMBED_DIM);
  for (let j = 0; j < 3; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += vec[i] * P[i * 3 + j];
    out[j] = s;
  }
  return out;
}

export function mountEmbedding3D({ root, bus, getState }) {
  const P = buildProjection();

  const THREE = window.THREE;
  const webglOK = (() => {
    try {
      const c = document.createElement("canvas");
      return !!(window.WebGLRenderingContext && (c.getContext("webgl") || c.getContext("experimental-webgl")));
    } catch { return false; }
  })();

  if (!THREE || !webglOK) {
    return mountCanvas2D(root, bus, P);
  }

  /* ---- 3D path ---- */
  const w = root.clientWidth || 400;
  const h = root.clientHeight || 300;

  const scene = new THREE.Scene();
  scene.background = null;

  const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 100);
  camera.position.set(0, 0, 4);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(w, h, false);
  root.appendChild(renderer.domElement);

  // Ambient sphere reference (muted wireframe).
  const refGeo = new THREE.SphereGeometry(1.4, 24, 18);
  const refMat = new THREE.MeshBasicMaterial({ color: 0xa0a0b0, wireframe: true, transparent: true, opacity: 0.18 });
  const refMesh = new THREE.Mesh(refGeo, refMat);
  scene.add(refMesh);

  // Trail points — buffered geometry updated in place.
  const MAX = 200;
  const positions = new Float32Array(MAX * 3);
  const colors    = new Float32Array(MAX * 3);
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setAttribute("color",    new THREE.BufferAttribute(colors, 3));
  const pointsMat = new THREE.PointsMaterial({
    size: 0.05, sizeAttenuation: true, vertexColors: true, transparent: true, opacity: 0.9,
  });
  const points = new THREE.Points(geom, pointsMat);
  scene.add(points);

  // Current-state glowing mesh.
  const curMat = new THREE.MeshBasicMaterial({ color: 0xe94560 });
  const curMesh = new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 12), curMat);
  scene.add(curMesh);

  /* ---- orbital controls: simple drag + wheel ---- */
  let yaw = 0.4, pitch = 0.15, dist = 4;
  let dragging = false, lastX = 0, lastY = 0;
  renderer.domElement.addEventListener("mousedown", (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener("mouseup", () => { dragging = false; });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    yaw   += (e.clientX - lastX) * 0.005;
    pitch += (e.clientY - lastY) * 0.005;
    pitch = Math.max(-1.2, Math.min(1.2, pitch));
    lastX = e.clientX; lastY = e.clientY;
  });
  renderer.domElement.addEventListener("wheel", (e) => {
    e.preventDefault();
    dist = Math.max(2, Math.min(12, dist + e.deltaY * 0.004));
  }, { passive: false });

  const accent = new THREE.Color(0xe94560);
  const active = new THREE.Color(0xf0f0f0);
  const muted  = new THREE.Color(0xa0a0b0);

  function writePoints() {
    const buf = getState().embeddingBuffer || [];
    const n = buf.length;
    const posAttr = geom.getAttribute("position");
    const colAttr = geom.getAttribute("color");
    for (let i = 0; i < MAX; i++) {
      if (i < n) {
        const [x, y, z] = project(buf[i].vec, P);
        posAttr.array[i * 3 + 0] = x;
        posAttr.array[i * 3 + 1] = y;
        posAttr.array[i * 3 + 2] = z;

        // Fade older points toward muted. Latest point: accent.
        const t = i / Math.max(1, n - 1);
        const c = (i === n - 1) ? accent : muted.clone().lerp(active, t * 0.4);
        colAttr.array[i * 3 + 0] = c.r;
        colAttr.array[i * 3 + 1] = c.g;
        colAttr.array[i * 3 + 2] = c.b;
      } else {
        posAttr.array[i * 3 + 0] = 999;
        posAttr.array[i * 3 + 1] = 999;
        posAttr.array[i * 3 + 2] = 999;
      }
    }
    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
    if (n > 0) {
      const [x, y, z] = project(buf[n - 1].vec, P);
      curMesh.position.set(x, y, z);
    }
    geom.setDrawRange(0, Math.min(n, MAX));
    document.getElementById("embed-count").textContent = `${n} / 200`;
  }

  function onResize() {
    const w2 = root.clientWidth;
    const h2 = root.clientHeight;
    if (!w2 || !h2) return;
    camera.aspect = w2 / h2;
    camera.updateProjectionMatrix();
    renderer.setSize(w2, h2, false);
  }
  const ro = new ResizeObserver(onResize);
  ro.observe(root);

  let t0 = performance.now();
  function tick() {
    const dt = (performance.now() - t0) / 1000;
    // Orbital camera.
    camera.position.x = Math.sin(yaw) * Math.cos(pitch) * dist;
    camera.position.y = Math.sin(pitch) * dist;
    camera.position.z = Math.cos(yaw) * Math.cos(pitch) * dist;
    camera.lookAt(0, 0, 0);
    refMesh.rotation.y += 0.0008;

    // Pulse current mesh.
    const s = 1 + 0.15 * Math.sin(dt * 3.2);
    curMesh.scale.setScalar(s);

    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  bus.addEventListener("embedding", writePoints);
  writePoints();

  return { dispose() { ro.disconnect(); renderer.dispose(); } };
}

/* ---- 2D fallback ---- */
function mountCanvas2D(root, bus, P) {
  const canvas = document.createElement("canvas");
  canvas.style.width = "100%"; canvas.style.height = "100%";
  root.appendChild(canvas);
  const ctx = canvas.getContext("2d");
  function fit() {
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(r.width * devicePixelRatio));
    canvas.height = Math.max(1, Math.floor(r.height * devicePixelRatio));
  }
  fit();
  new ResizeObserver(fit).observe(canvas);

  function draw(buf) {
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#16213e";
    ctx.fillRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2;
    const s = Math.min(w, h) * 0.35;
    for (let i = 0; i < buf.length; i++) {
      const [x, y] = project(buf[i].vec, P);
      const px = cx + x * s;
      const py = cy + y * s;
      const alpha = 0.25 + 0.75 * (i / Math.max(1, buf.length - 1));
      ctx.fillStyle = (i === buf.length - 1) ? "#e94560" : `rgba(240,240,240,${alpha})`;
      ctx.beginPath();
      ctx.arc(px, py, (i === buf.length - 1) ? 5 : 2, 0, Math.PI * 2);
      ctx.fill();
    }
    document.getElementById("embed-count").textContent = `${buf.length} / 200 (2D)`;
  }
  bus.addEventListener("embedding", (ev) => draw(ev.detail.buf));
  draw([]);
  return {};
}
