// Extracción manual del motor físico de EIT3_Termico_kappaH_v7.3.html, sin DOM.
// Fiel línea a línea a las funciones del HTML (mismo orden de operaciones,
// misma fórmula, mismo orden de consumo del rng seedeado) para que a
// igualdad de semilla y parámetros reproduzca exactamente la trayectoria
// del <script> real (ver test_cruzado.mjs). Solo cubre la rama día/noche
// APAGADO (pasoFisica no-daynight) — la única que usan los 3 experimentos.
// clamp/lerp/gauss son funciones flecha en el original: aquí también.

export const gridSize = 64;
export const HCFG = { win: 120, bins: 24, lo: 0, hi: 1.2, margin: 0.05 };

export const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const gauss = (x, m, s) => Math.exp(-0.5 * Math.pow((x - m) / Math.max(s, 1e-6), 2));

export function pseudoNoise(x, y, t) {
  const s = Math.sin(x * 12.9898 + y * 78.233 + t * 0.021) * 43758.5453;
  return (s - Math.floor(s)) * 2 - 1;
}

export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const DEFAULTS = {
  running: false, auto: false, dayNightMode: false, step: 0, tick: 0, dt: 5,
  dayNightSpeed: 1 / 60, maxCycles: 1000, cyclesVivo: 0, isVivo: false, lambdaHistory: [],
  powerBase: 0.47, powerLive: 0.47, beta: 0.94, sigma: 6.8, noise: 0.0079, band: 1.105,
  luminosity: 0.94, tOpt: 25, ptcTc: 25, ptcSharp: 8, minTemp: -6, maxTemp: 25,
  ptcR: 1, ptcOut: 1, mult: 0, _A_prev: 0, H_at: 0, H_noise: 0, H_rel: 0, absBand: 0, absFloor: 0,
  bioticTf: 24.6, abioticTf: 25, bioticFootprint: 0, Tf: 24.6, Tc: 25, Th: 28, delta: 3.4,
  LF: 0, err: 0, Lambda: 0.1, deltaStruct: 0, A_sys_env: 0, LF_exp: 0, err_exp: 0, Lambda_exp: 0,
  fertileScore: 0, action: 'dejar oscilar', regime: 'RÍGIDO', omega: 0, envTemp: 24.6,
  externalStress: 0, envDrift: 0, history: [], black: 0.18, white: 0.14, bare: 0.68,
  albedoBlack: 0.25, albedoWhite: 0.75, albedoBare: 0.5, seed: 1,
};

// Reset "parcial" (equivalente a resetSimulation() del HTML): mismos campos,
// mismo conjunto — NO toca beta/sigma/noise/band/luminosity/tOpt/ptcTc/ptcSharp
// (esos vienen de syncStateFromUI en el original; acá se fijan aparte, ver
// Motor.setParams). Defecto de reproducibilidad documentado en
// defectos_encontrados.md: el runSweep() original NUNCA llama a esto — motor.mjs
// SÍ lo hace, una vez por barrido completo, para que la batería sea reproducible.
const RESET_KEYS = [
  'step', 'tick', 'cyclesVivo', 'isVivo', 'lambdaHistory', 'powerBase', 'powerLive',
  'ptcR', 'ptcOut', 'Tf', 'Tc', 'Th', 'delta', 'LF', 'err', 'Lambda', 'deltaStruct',
  'A_sys_env', 'LF_exp', 'err_exp', 'Lambda_exp', 'fertileScore', 'regime', 'action',
  'omega', 'envTemp', 'externalStress', 'envDrift', 'history', 'black', 'white', 'bare',
];

export class Motor {
  constructor() {
    this.state = { ...DEFAULTS };
    this.field = Array.from({ length: gridSize }, () => Array.from({ length: gridSize }, () => 24.5));
    this.aBuf = [];
    this.noiseEchoBuf = [];
    this._Awin = [];
    this.rng = Math.random;
  }

  setSeed(seed) {
    this.state.seed = seed;
    this.rng = mulberry32(seed >>> 0);
  }

  // Aplica los parámetros "de UI" de una corrida (equivalente a mover los
  // sliders + syncStateFromUI()).
  setParams(p) {
    for (const k of ['powerBase', 'beta', 'sigma', 'noise', 'band', 'luminosity', 'tOpt', 'ptcTc', 'ptcSharp', 'minTemp', 'maxTemp']) {
      if (p[k] !== undefined) this.state[k] = p[k];
    }
  }

  resetField() {
    this.field = Array.from({ length: gridSize }, () => Array.from({ length: gridSize }, () => 24.5));
  }

  // Reset completo: full reset de física + campo + buffers, preservando la
  // semilla y los parámetros de la corrida (se llama DESPUÉS de setSeed/setParams
  // en el flujo normal, pero solo toca los campos de RESET_KEYS + buffers).
  resetSimulation() {
    const defaultsForKeys = {};
    for (const k of RESET_KEYS) defaultsForKeys[k] = Array.isArray(DEFAULTS[k]) ? [] : DEFAULTS[k];
    Object.assign(this.state, defaultsForKeys);
    this.aBuf = [];
    this.noiseEchoBuf = [];
    this.state._A_prev = 0;
    this.resetField();
  }

  shannonEntropy(samples, bins, lo, hi) {
    if (samples.length < 8) return 0;
    const counts = new Array(bins).fill(0);
    const span = Math.max(1e-9, hi - lo);
    for (const v of samples) {
      let idx = Math.floor(((v - lo) / span) * bins);
      if (idx < 0) idx = 0; if (idx >= bins) idx = bins - 1;
      counts[idx]++;
    }
    const n = samples.length;
    let H = 0;
    for (const c of counts) { if (c > 0) { const p = c / n; H -= p * Math.log2(p); } }
    return H;
  }

  entropyLocalAbs(samples, floorSamples, bins, margin) {
    if (samples.length < 8) return { H: 0, lo: 0, hi: 0, band: 0, floor: 0 };
    let pLo = Infinity, pHi = -Infinity; for (const v of samples) { if (v < pLo) pLo = v; if (v > pHi) pHi = v; }
    let fLo = Infinity, fHi = -Infinity; for (const v of floorSamples) { if (v < fLo) fLo = v; if (v > fHi) fHi = v; }
    const band = pHi - pLo;
    const floor = (floorSamples.length >= 8) ? (fHi - fLo) : 0;
    const center = (pLo + pHi) / 2;
    let width = Math.max(band, floor) * (1 + 2 * margin);
    if (!(width > 0)) width = 1e-9;
    const lo = center - width / 2, hi = center + width / 2;
    return { H: this.shannonEntropy(samples, bins, lo, hi), lo, hi, band, floor };
  }

  entropyAtWidth(samples, bins, width) {
    if (samples.length < 8 || !(width > 0)) return 0;
    let lo = Infinity, hi = -Infinity;
    for (const v of samples) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const c = (lo + hi) / 2;
    return this.shannonEntropy(samples, bins, c - width / 2, c + width / 2);
  }

  entropyRel(samples, bins) {
    if (samples.length < 8) return 0;
    let lo = Infinity, hi = -Infinity; for (const v of samples) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (hi - lo < 1e-9) return 0;
    return this.shannonEntropy(samples, bins, lo, hi);
  }

  passiveNoiseSample() {
    return this.state.powerBase + (this.rng() - 0.5) * this.state.noise * 10;
  }

  updateBehavioralEntropy() {
    const s = this.state;
    this.aBuf.push(s.powerLive);
    if (this.aBuf.length > HCFG.win) this.aBuf.shift();
    this.noiseEchoBuf.push(this.passiveNoiseSample());
    if (this.noiseEchoBuf.length > HCFG.win) this.noiseEchoBuf.shift();
    const la = this.entropyLocalAbs(this.aBuf, this.noiseEchoBuf, HCFG.bins, HCFG.margin);
    s.H_at = la.H;
    s.H_rel = this.entropyRel(this.aBuf, HCFG.bins);
    s.H_noise = this.entropyAtWidth(this.noiseEchoBuf, HCFG.bins, la.hi - la.lo);
    s.absBand = la.band; s.absFloor = la.floor;
  }

  abioticTf() {
    const s = this.state;
    return 12 + 34 * s.luminosity * (1 - s.albedoBare);
  }

  computeDeltaStruct(field) {
    let sum = 0, sumSq = 0, n = 0;
    for (let y = 0; y < field.length; y++) for (let x = 0; x < field[0].length; x++) { const v = field[y][x]; sum += v; sumSq += v * v; n++; }
    const mean = sum / n;
    return Math.sqrt(Math.max(sumSq / n - mean * mean, 0));
  }

  computeCoupling(Tf, targetTf) {
    const diff = Math.abs(Tf - targetTf);
    return Math.max(0, 1 - diff / 8.0);
  }

  ptcResponse(temp) {
    const s = this.state;
    const ratio = clamp(temp / Math.max(0.1, s.ptcTc), 0.2, 3);
    s.ptcR = Math.max(0.15, Math.pow(ratio, s.ptcSharp));
    s.ptcOut = clamp(1 / s.ptcR, 0.05, 1.2);
    return s.ptcOut;
  }

  computeDaisyworld() {
    const s = this.state;
    s.bare = clamp(1 - s.black - s.white, 0, 1);
    const albedo = s.black * s.albedoBlack + s.white * s.albedoWhite + s.bare * s.albedoBare;
    const absorbed = s.luminosity * (1 - albedo);
    const Tplanet = 12 + 34 * absorbed;
    const localBlack = Tplanet + (albedo - s.albedoBlack) * 14;
    const localWhite = Tplanet + (albedo - s.albedoWhite) * 14;
    const growthBlack = clamp(1 - 0.003265 * Math.pow(s.tOpt - localBlack, 2), 0, 1);
    const growthWhite = clamp(1 - 0.003265 * Math.pow(s.tOpt - localWhite, 2), 0, 1);
    const death = 0.28 + s.noise * 10;
    const spawn = Math.max(0, s.bare);
    s.black = clamp(s.black + (s.black * (growthBlack * spawn - death)) * 0.08, 0, 0.9);
    s.white = clamp(s.white + (s.white * (growthWhite * spawn - death)) * 0.08, 0, 0.9);
    s.bare = clamp(1 - s.black - s.white, 0, 1);
    return {
      albedo: s.black * s.albedoBlack + s.white * s.albedoWhite + s.bare * s.albedoBare,
      targetTf: 12 + 34 * s.luminosity * (1 - (s.black * s.albedoBlack + s.white * s.albedoWhite + s.bare * s.albedoBare)),
    };
  }

  classifyRegime() {
    const s = this.state;
    if (s.deltaStruct < 0.2 && s.LF < 0.15) return ['CERRADO', 'dejar oscilar'];
    if (s.deltaStruct >= 0.2 && s.deltaStruct < 1.2 && s.fertileScore >= 0.05) return ['TRANSICIÓN', 'dejar oscilar'];
    if (s.deltaStruct > 1.4 || s.err > 2.5) return ['SOBRECARGA', 'enfriar borde'];
    return ['RÍGIDO', 'reanclar'];
  }

  computeLFandErr(daisiesTargetTf) {
    const s = this.state;
    const targetPower = s.powerBase * this.ptcResponse(s.Tf);
    s.powerLive = lerp(s.powerLive, targetPower, 0.08);
    const deviation = Math.abs(s.powerLive - s.powerBase);
    const inertiaPenalty = Math.exp(-deviation * 4);
    s.mult = clamp(deviation * (1 - inertiaPenalty), 0, 1);
    s.LF = s.mult;
    const dA = s.A_sys_env - s._A_prev;
    s.err = Math.max(0, -dA);
    s._A_prev = s.A_sys_env;
    s.err_exp = s.externalStress;
  }

  evolveField(albedo) {
    const s = this.state;
    const next = this.field.map(r => r.slice());
    const cx = gridSize / 2, cy = gridSize / 2;
    for (let y = 0; y < gridSize; y++) for (let x = 0; x < gridSize; x++) {
      const dx = (x - cx) / cx, dy = (y - cy) / cy, r = Math.sqrt(dx * dx + dy * dy);
      const edge = gauss(r, 0.72, 0.11 + s.band * 0.03);
      const daisyMix = s.black * (1 - r) + s.white * r;
      const noise = (pseudoNoise(x, y, s.tick + s.seed * 1013.9) - 0.5) * s.noise * 16;
      const target = s.Tf + edge * s.delta * 1.8 + daisyMix * 1.2 - albedo * 1.8 + noise;
      const smooth = clamp(0.05 + s.sigma * 0.01, 0.04, 0.16);
      let n = 0, c = 0;
      for (let oy = -1; oy <= 1; oy++) for (let ox = -1; ox <= 1; ox++) {
        const nx = x + ox, ny = y + oy;
        if (nx >= 0 && nx < gridSize && ny >= 0 && ny < gridSize) { n += this.field[ny][nx]; c++; }
      }
      next[y][x] = lerp(n / c, target, smooth);
    }
    this.field = next;
  }

  errRatePush(A) { this._Awin.push(A); if (this._Awin.length > 12) this._Awin.shift(); }
  errRate() {
    if (this._Awin.length < 3) return 0;
    let loss = 0, c = 0;
    for (let i = 1; i < this._Awin.length; i++) { const dA = this._Awin[i] - this._Awin[i - 1]; if (dA < 0) loss += -dA; c++; }
    return c ? loss / c : 0;
  }

  // pasoFisica() — SOLO rama día/noche apagado (única usada por los 3 experimentos).
  paso() {
    const s = this.state;
    const daisies = this.computeDaisyworld();
    this.computeLFandErr(daisies.targetTf);
    const thermalDrive = 8.2 * s.powerLive + 0.46 * (daisies.targetTf - s.Tf);
    const damping = 0.09 + (1 - s.beta) * 0.65;
    const stochastic = (this.rng() - 0.5) * s.noise * 14;
    s.Tf = s.Tf + thermalDrive * 0.12 - damping * (s.Tf - s.tOpt) * 0.05 + stochastic;
    const edgeBias = (s.black - s.white) * 4.2;
    s.Tc = lerp(s.Tc, s.Tf + edgeBias * 0.18, 0.12);
    s.Th = lerp(s.Th, s.Tf + 0.65 + s.powerLive * 1.8 + edgeBias * 0.32, 0.10);
    s.delta = s.Th - s.Tf;
    s.Lambda = (s.deltaStruct * s.LF) / Math.max(s.err, 1e-6) * s.A_sys_env;
    s.Lambda_exp = s.Lambda;
    s.fertileScore = clamp(s.LF * clamp(1 - Math.abs(s.delta - 1.1) / 1.6, 0, 1) * (0.4 + Math.abs(s.black - s.white) + s.Lambda * 0.12), 0, 1);
    s.omega = (s.black - s.white) * s.powerLive * 0.12;
    this.evolveField(daisies.albedo);
    s.deltaStruct = this.computeDeltaStruct(this.field);
    s.A_sys_env = this.computeCoupling(s.Tf, daisies.targetTf);
    s.abioticTf = this.abioticTf();
    s.bioticTf = s.Tf;
    s.bioticFootprint = Math.abs(s.Tf - s.abioticTf);
    this.updateBehavioralEntropy();
    s.LF_exp = clamp(Math.abs(s.powerLive - s.powerBase), 0, 1);
    s.lambdaHistory.push(s.Lambda_exp);
    if (s.lambdaHistory.length > 50) s.lambdaHistory.shift();

    const rr = this.classifyRegime();
    s.regime = rr[0]; s.action = rr[1];

    s.step += s.dt; s.tick += 1;
  }
}
