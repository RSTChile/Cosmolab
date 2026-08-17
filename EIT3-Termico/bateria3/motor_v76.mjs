// Extracción manual del motor físico de ET3-Termico_v7.6.1.html, sin DOM.
// La física de pasoFisica/evolveField/computeDaisyworld/etc. es idéntica a
// v7.5 (confirmado leyendo el archivo) — lo que cambió es el generador de
// azar: dos flujos (rngTf, rngEco) resembrados por fase vía sembrarFase(), más
// asentarHastaEquilibrio()/instantanea()/restaurarInstantanea() nuevos.
// Standalone en vez de extender bateria2/motor_v75.mjs porque el punto de
// consumo del rng cambia en varios métodos (paso, passiveNoiseSample, setSeed).
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

// Hash FNV-1a de 32 bits — idéntico al claveSemilla() del HTML.
export function claveSemilla(...partes) {
  const s = partes.join('|');
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return h >>> 0;
}

const DEFAULTS = {
  running: false, auto: false, dayNightMode: false, step: 0, tick: 0, dt: 5,
  dayNightSpeed: 1 / 60, maxCycles: 1000, cyclesVivo: 0, isVivo: false, lambdaHistory: [],
  powerBase: 0.47, powerLive: 0.47, beta: 0.94, sigma: 6.8, noise: 0.0079, band: 1.105,
  luminosity: 0.94, tOpt: 25, ptcTc: 25, ptcSharp: 4, minTemp: -6, maxTemp: 25,
  ptcR: 1, ptcOut: 1, mult: 0, _A_prev: 0, H_at: 0, H_noise: 0, H_rel: 0, absBand: 0, absFloor: 0,
  bioticTf: 24.6, abioticTf: 25, bioticFootprint: 0, Tf: 24.6, Tc: 25, Th: 28, delta: 3.4,
  LF: 0, err: 0, Lambda: 0.1, deltaStruct: 0, A_sys_env: 0, LF_exp: 0, err_exp: 0, Lambda_exp: 0,
  fertileScore: 0, action: 'dejar oscilar', regime: 'RÍGIDO', omega: 0, envTemp: 24.6,
  externalStress: 0, envDrift: 0, history: [], black: 0.18, white: 0.14, bare: 0.68,
  albedoBlack: 0.25, albedoWhite: 0.75, albedoBare: 0.5, seed: 1,
};

const SILENCIO_KEYS = [
  'step', 'tick', 'cyclesVivo', 'isVivo', 'lambdaHistory',
  'ptcR', 'ptcOut', 'Tf', 'Tc', 'Th', 'delta', 'LF', 'err', 'Lambda', 'deltaStruct',
  'A_sys_env', 'LF_exp', 'err_exp', 'Lambda_exp', 'fertileScore', 'regime', 'action',
  'omega', 'externalStress', 'envDrift', 'black', 'white', 'bare',
];
const RESET_KEYS = [
  'step', 'tick', 'cyclesVivo', 'isVivo', 'lambdaHistory', 'powerBase', 'powerLive',
  'ptcR', 'ptcOut', 'Tf', 'Tc', 'Th', 'delta', 'LF', 'err', 'Lambda', 'deltaStruct',
  'A_sys_env', 'LF_exp', 'err_exp', 'Lambda_exp', 'fertileScore', 'regime', 'action',
  'omega', 'envTemp', 'externalStress', 'envDrift', 'history', 'black', 'white', 'bare',
];

export class MotorV76 {
  constructor() {
    this.state = { ...DEFAULTS };
    this.field = Array.from({ length: gridSize }, () => Array.from({ length: gridSize }, () => 24.5));
    this.aBuf = [];
    this.noiseEchoBuf = [];
    this._Awin = [];
    this.rngTf = mulberry32(claveSemilla(1, 'libre', 0, 'libre', 'Tf'));
    this.rngEco = mulberry32(claveSemilla(1, 'libre', 0, 'libre', 'eco'));
  }

  setSeed(seed) {
    this.state.seed = seed >>> 0;
    this.rngTf = mulberry32(claveSemilla(this.state.seed, 'libre', 0, 'libre', 'Tf'));
    this.rngEco = mulberry32(claveSemilla(this.state.seed, 'libre', 0, 'libre', 'eco'));
  }

  // sembrarFase(eje,punto,fase) — resiembra ambos flujos, sin tocar el estado físico.
  sembrarFase(eje, punto, fase) {
    this.rngTf = mulberry32(claveSemilla(this.state.seed, eje, punto, fase, 'Tf'));
    this.rngEco = mulberry32(claveSemilla(this.state.seed, eje, punto, fase, 'eco'));
  }

  setParams(p) {
    for (const k of ['powerBase', 'beta', 'sigma', 'noise', 'band', 'luminosity', 'tOpt', 'ptcTc', 'ptcSharp', 'minTemp', 'maxTemp']) {
      if (p[k] !== undefined) this.state[k] = p[k];
    }
  }

  resetField() {
    this.field = Array.from({ length: gridSize }, () => Array.from({ length: gridSize }, () => 24.5));
  }

  resetSimulation() {
    const defaultsForKeys = {};
    for (const k of RESET_KEYS) defaultsForKeys[k] = Array.isArray(DEFAULTS[k]) ? [] : DEFAULTS[k];
    Object.assign(this.state, defaultsForKeys);
    this.aBuf = []; this.noiseEchoBuf = []; this._Awin = []; this.state._A_prev = 0;
    this.resetField();
  }

  reiniciarSilencioso() {
    const s = this.state;
    const defaultsForKeys = {};
    for (const k of SILENCIO_KEYS) defaultsForKeys[k] = Array.isArray(DEFAULTS[k]) ? [] : DEFAULTS[k];
    Object.assign(s, defaultsForKeys);
    s.powerLive = s.powerBase;
    this.aBuf = []; this.noiseEchoBuf = []; this._Awin = []; s._A_prev = 0;
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

  varianzaYAutocorr(xs0) {
    const n = xs0.length;
    if (n < 3) return { varianza: 0, autocorr1: 0 };
    let sx = 0, sy = 0, sxy = 0, sxx = 0;
    for (let i = 0; i < n; i++) { sx += i; sy += xs0[i]; sxy += i * xs0[i]; sxx += i * i; }
    const den = n * sxx - sx * sx;
    const b = den !== 0 ? (n * sxy - sx * sy) / den : 0;
    const a = sy / n - b * sx / n;
    const xs = xs0.map((y, i) => y - (a + b * i));
    let m = 0; for (const x of xs) m += x; m /= n;
    let s2 = 0; for (const x of xs) s2 += (x - m) * (x - m); s2 /= n;
    if (s2 <= 0) return { varianza: 0, autocorr1: 0 };
    let c = 0; for (let i = 1; i < n; i++) c += (xs[i] - m) * (xs[i - 1] - m);
    return { varianza: s2, autocorr1: (c / (n - 1)) / s2 };
  }

  passiveNoiseSample() {
    return this.state.powerBase + (this.rngEco() - 0.5) * this.state.noise * 10;
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

  // ptcResponse: se omite registrarSaturacion() a propósito — solo alimenta
  // state.satFrac/satCiego (aviso de pantalla), que no entra en ninguna
  // columna exportada (ptcSat se calcula aparte, ver paso()/correr_barrido).
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

  // pasoFisica() — SOLO rama día/noche apagado (única usada por los experimentos).
  paso() {
    const s = this.state;
    const daisies = this.computeDaisyworld();
    this.computeLFandErr(daisies.targetTf);
    const thermalDrive = 8.2 * s.powerLive + 0.46 * (daisies.targetTf - s.Tf);
    const damping = 0.09 + (1 - s.beta) * 0.65;
    const stochastic = (this.rngTf() - 0.5) * s.noise * 14;
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

  // ── v7.6.1: instantánea/restauración + asentamiento hasta equilibrio + recuperación ──

  instantanea() {
    return {
      black: this.state.black, white: this.state.white, bare: this.state.bare,
      Tf: this.state.Tf, Tc: this.state.Tc, Th: this.state.Th, delta: this.state.delta,
      powerLive: this.state.powerLive, ptcR: this.state.ptcR, ptcOut: this.state.ptcOut,
      campo: this.field.map(f => f.slice()),
    };
  }

  restaurarInstantanea(s) {
    this.state.black = s.black; this.state.white = s.white; this.state.bare = s.bare;
    this.state.Tf = s.Tf; this.state.Tc = s.Tc; this.state.Th = s.Th; this.state.delta = s.delta;
    this.state.powerLive = s.powerLive; this.state.ptcR = s.ptcR; this.state.ptcOut = s.ptcOut;
    this.field = s.campo.map(f => f.slice());
  }

  asentarHastaEquilibrio(tope, tol) {
    const V = 50, NECESARIAS = 3;
    let prev = this.state.black, quietas = 0, pasos = 0;
    while (pasos < tope) {
      for (let s = 0; s < V; s++) this.paso();
      pasos += V;
      const d = Math.abs(this.state.black - prev); prev = this.state.black;
      if (d < tol) { quietas++; if (quietas >= NECESARIAS) return { pasos, asentado: 1 }; }
      else quietas = 0;
    }
    return { pasos, asentado: 0 };
  }

  medirRecuperacion(golpe, tope) {
    const REPS = 5, UMBRAL = 0.2;
    const base = this.instantanea();
    let suma = 0, fallos = 0; const reps = [];
    for (let r = 0; r < REPS; r++) {
      this.restaurarInstantanea(base);
      const bB = base.black;
      this.state.black = clamp(bB + golpe, 0, 0.9);
      this.state.white = clamp(base.white - golpe * 0.5, 0, 0.9);
      let i = 1;
      for (; i <= tope; i++) {
        this.paso();
        if (Math.abs(this.state.black - bB) <= golpe * UMBRAL) break;
      }
      if (i > tope) fallos++;
      reps.push(Math.min(i, tope));
      suma += Math.min(i, tope);
    }
    this.restaurarInstantanea(base);
    const ord = reps.slice().sort((a, b) => a - b);
    return { pasos: suma / REPS, convergio: fallos === 0 ? 1 : 0, reps, topes: fallos, mediana: ord[Math.floor(REPS / 2)] };
  }
}
