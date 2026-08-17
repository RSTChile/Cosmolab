/**
 * CG001 Viewer 3D — Three.js (protocolo §151-154)
 * Color ← H · Tamaño ← S · Brillo ← S · Estelas por partícula
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

function displayPos(pos, gridSize) {
  const g2 = gridSize * 0.5;
  return [(pos[0] || 0) + g2, (pos[1] || 0) + g2, (pos[2] || 0) + g2];
}

function entityColor(e, smax, hmax) {
  const hVal = e.H || 0;
  const sVal = Math.max(0, e.S || 0);
  const hNorm = hVal / (hmax + 1e-9);
  const sNorm = sVal / (smax + 1e-9);
  const hue = (0.50 + hNorm * 0.50) % 1;
  const sat = 0.55 + 0.35 * Math.min(1, hNorm);
  const lit = 0.20 + 0.70 * Math.min(1, sNorm);
  return new THREE.Color().setHSL(hue, sat, lit);
}

export class CG001Viewer3D {
  constructor(container, opts = {}) {
    this.container = container;
    this.gridSize = opts.gridSize || 64;
    this.followEpsilon = false;
    this.showTrails = opts.showTrails !== false;
    this.maxTrail = 48;
    this.maxTrailParticles = 200;

    this.trails = new Map();
    this.entityMap = new Map();
    this.targetMap = new Map();
    this.blendStart = performance.now();
    this.blendMs = 120;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x060a14);
    this.scene.fog = new THREE.FogExp2(0x060a14, 0.012);

    const w = container.clientWidth || 800;
    const h = container.clientHeight || 480;
    this.camera = new THREE.PerspectiveCamera(52, w / h, 0.1, 500);
    this.camera.position.set(this.gridSize * 0.9, this.gridSize * 0.7, this.gridSize * 1.1);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.target.set(this.gridSize / 2, this.gridSize / 2, this.gridSize / 2);

    const amb = new THREE.AmbientLight(0x334466, 0.6);
    this.scene.add(amb);
    const key = new THREE.DirectionalLight(0x9adfff, 0.9);
    key.position.set(40, 60, 30);
    this.scene.add(key);

    const vol = new THREE.Mesh(
      new THREE.BoxGeometry(this.gridSize, this.gridSize, this.gridSize),
      new THREE.MeshBasicMaterial({ visible: false }),
    );
    vol.position.set(this.gridSize / 2, this.gridSize / 2, this.gridSize / 2);
    this.scene.add(vol);
    this.scene.add(new THREE.BoxHelper(vol, 0x2a3f6a));

    const grid = new THREE.GridHelper(this.gridSize, 16, 0x1e3058, 0x121c33);
    grid.position.set(this.gridSize / 2, 0, this.gridSize / 2);
    this.scene.add(grid);

    this.points = null;
    this.trailGroup = new THREE.Group();
    this.scene.add(this.trailGroup);

    this._boundResize = () => this.resize();
    window.addEventListener('resize', this._boundResize);
    this._animate();
  }

  resize() {
    const w = this.container.clientWidth || 800;
    const h = this.container.clientHeight || 480;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  setFollowEpsilon(on) {
    this.followEpsilon = !!on;
  }

  setTargets(entities, meta = {}) {
    if (meta.grid_size) this.gridSize = meta.grid_size;
    if (!entities || !entities.length) {
      this.targetMap.clear();
      return;
    }

    const now = performance.now();
    const sorted = [...entities].sort((a, b) => a.id - b.id);
    const newTargets = new Map();

    for (const e of sorted) {
      newTargets.set(e.id, e);
      const disp = displayPos(e.pos || [0, 0, 0], this.gridSize);
      if (this.entityMap.has(e.id)) {
        const cur = this.entityMap.get(e.id);
        this.entityMap.set(e.id, { ...cur, from: [cur.x, cur.y, cur.z], to: disp });
      } else {
        this.entityMap.set(e.id, { x: disp[0], y: disp[1], z: disp[2], from: disp, to: disp, entity: e });
      }
    }
    this.targetMap = newTargets;
    this.blendStart = now;
  }

  _blendedList() {
    const t = Math.min(1, (performance.now() - this.blendStart) / this.blendMs);
    const out = [];
    for (const [id, state] of this.entityMap) {
      if (!this.targetMap.has(id)) continue;
      const e = this.targetMap.get(id);
      const from = state.from || [state.x, state.y, state.z];
      const to = state.to || from;
      const x = from[0] * (1 - t) + to[0] * t;
      const y = from[1] * (1 - t) + to[1] * t;
      const z = from[2] * (1 - t) + to[2] * t;
      state.x = x;
      state.y = y;
      state.z = z;
      out.push({ ...e, pos: [x, y, z] });
    }
    return out.sort((a, b) => a.id - b.id);
  }

  _renderPoints(entities) {
    if (!entities.length) {
      if (this.points) {
        this.scene.remove(this.points);
        this.points.geometry.dispose();
        this.points.material.dispose();
        this.points = null;
      }
      return;
    }

    let smax = 1e-9;
    let hmax = 1e-9;
    for (const e of entities) {
      smax = Math.max(smax, e.S || 0);
      hmax = Math.max(hmax, e.H || 0);
    }

    const n = entities.length;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const sizes = new Float32Array(n);

    let epsEntity = null;
    for (let i = 0; i < n; i++) {
      const e = entities[i];
      pos[i * 3] = e.pos[0];
      pos[i * 3 + 1] = e.pos[1];
      pos[i * 3 + 2] = e.pos[2];

      const c = entityColor(e, smax, hmax);
      col[i * 3] = c.r;
      col[i * 3 + 1] = c.g;
      col[i * 3 + 2] = c.b;

      const sNorm = Math.max(0, e.S || 0) / (smax + 1e-9);
      sizes[i] = 0.2 + 1.6 * Math.min(1, sNorm);  // puntitos: base pequena, solo la persistencia agranda

      if (e.id === 0) epsEntity = e;
      if (this.showTrails) this._pushTrail(e.id, e.pos, e.H || 0);
    }

    if (!this.points) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
      geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
      const mat = new THREE.ShaderMaterial({
        uniforms: {},
        vertexShader: `
          attribute float size;
          attribute vec3 color;
          varying vec3 vColor;
          void main() {
            vColor = color;
            vec4 mv = modelViewMatrix * vec4(position, 1.0);
            gl_PointSize = clamp(size * (200.0 / -mv.z), 1.0, 9.0);
            gl_Position = projectionMatrix * mv;
          }`,
        fragmentShader: `
          varying vec3 vColor;
          void main() {
            vec2 uv = gl_PointCoord - vec2(0.5);
            float d = length(uv);
            if (d > 0.5) discard;
            float glow = 1.0 - smoothstep(0.2, 0.5, d);
            gl_FragColor = vec4(vColor * (0.75 + glow * 0.5), 0.95);
          }`,
        transparent: true,
        depthWrite: true,
      });
      this.points = new THREE.Points(geo, mat);
      this.scene.add(this.points);
    } else {
      const geo = this.points.geometry;
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
      geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
      geo.attributes.position.needsUpdate = true;
      geo.attributes.color.needsUpdate = true;
      geo.attributes.size.needsUpdate = true;
    }

    this._drawTrails();
    if (this.followEpsilon && epsEntity) {
      const p = epsEntity.pos;
      this.controls.target.lerp(new THREE.Vector3(p[0], p[1], p[2]), 0.08);
    }
  }

  updateEntities(entities, meta = {}) {
    this.setTargets(entities, meta);
    this._renderPoints(this._blendedList());
  }

  _pushTrail(id, pos, hVal) {
    if (this.trails.size >= this.maxTrailParticles && !this.trails.has(id)) return;
    const p = [pos[0], pos[1], pos[2]];
    if (!this.trails.has(id)) this.trails.set(id, []);
    const arr = this.trails.get(id);
    if (arr.length) {
      const last = arr[arr.length - 1];
      const dx = p[0] - last[0];
      const dy = p[1] - last[1];
      const dz = p[2] - last[2];
      if (dx * dx + dy * dy + dz * dz < 0.0004) return;
    }
    arr.push(p);
    if (arr.length > this.maxTrail) arr.shift();
  }

  _drawTrails() {
    while (this.trailGroup.children.length) {
      const ch = this.trailGroup.children[0];
      ch.geometry.dispose();
      ch.material.dispose();
      this.trailGroup.remove(ch);
    }
    for (const [id, pts] of this.trails) {
      if (pts.length < 2) continue;
      const flat = new Float32Array(pts.length * 3);
      for (let i = 0; i < pts.length; i++) {
        flat[i * 3] = pts[i][0];
        flat[i * 3 + 1] = pts[i][1];
        flat[i * 3 + 2] = pts[i][2];
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(flat, 3));
      const mat = new THREE.LineBasicMaterial({
        color: id === 0 ? 0xffcc66 : 0x5588cc,
        transparent: true,
        opacity: id === 0 ? 0.9 : 0.35,
      });
      this.trailGroup.add(new THREE.Line(geo, mat));
    }
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    if (this.targetMap.size) {
      this._renderPoints(this._blendedList());
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  destroy() {
    window.removeEventListener('resize', this._boundResize);
    this.renderer.dispose();
  }
}