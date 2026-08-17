// Tarea 3: verificación obligatoria — con la misma semilla, la trayectoria
// updateSimulation() (física+dibujo) vs stepHeadless() (solo física) tiene
// que ser idéntica bit a bit. Corre sobre el <script> REAL del HTML (vía
// shim_html.mjs), no sobre una reimplementación.
import { buildSandbox } from './shim_html.mjs';

const SEED = 42;
const N_STEPS = 2000;

function setParams(api, over = {}) {
  const p = {
    powerBase: 0.47, beta: 0.94, sigma: 6.8, noise: 0.0079, band: 1.105,
    luminosity: 0.9, tOpt: 25, ptcTc: 18, ptcSharp: 4.1,
    minTemp: -6, maxTemp: 25,
    ...over,
  };
  for (const [k, v] of Object.entries(p)) api.els[k].value = String(v);
  api.els.dayNightToggle.checked = false;
  api.updateLabels();
  api.syncStateFromUI();
}

function snapshot(api) {
  const s = api.getState();
  const f = api.getField();
  const flat = {};
  for (const k of Object.keys(s)) {
    const v = s[k];
    if (typeof v === 'number') flat[k] = v;
  }
  let fieldSum = 0, fieldChecksum = '';
  for (let y = 0; y < f.length; y++) for (let x = 0; x < f[0].length; x++) fieldSum += f[y][x];
  fieldChecksum = f.map(r => r.join(',')).join('|');
  return { flat, fieldSum, fieldChecksum };
}

function run(mode) {
  const api = buildSandbox();
  api.setSeed(SEED);
  api.resetSimulation();
  setParams(api);
  for (let i = 0; i < N_STEPS; i++) {
    if (mode === 'conDibujo') api.updateSimulation();
    else api.stepHeadless();
  }
  return snapshot(api);
}

const a = run('conDibujo');
const b = run('sinDibujo');

let mismatches = [];
for (const k of Object.keys(a.flat)) {
  if (!Object.is(a.flat[k], b.flat[k])) mismatches.push(`${k}: ${a.flat[k]} vs ${b.flat[k]}`);
}
const fieldMatch = a.fieldChecksum === b.fieldChecksum;

console.log(JSON.stringify({
  seed: SEED, pasos: N_STEPS,
  estadoIdentico: mismatches.length === 0,
  mismatches,
  campoIdentico: fieldMatch,
  fieldSumA: a.fieldSum, fieldSumB: b.fieldSum,
}, null, 2));

if (mismatches.length === 0 && fieldMatch) {
  console.log('RESULTADO: BIT-IDÉNTICO — OK');
  process.exit(0);
} else {
  console.log('RESULTADO: DIVERGENCIA — DEFECTO BLOQUEANTE');
  process.exit(1);
}
