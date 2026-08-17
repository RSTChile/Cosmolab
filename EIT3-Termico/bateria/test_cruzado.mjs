// Tarea 4: valida motor.mjs (extracción limpia) contra shim_html.mjs (script
// real del HTML ejecutado en Node). Misma semilla, mismos parámetros, mismo
// número de pasos -> debe coincidir bit a bit.
import { buildSandbox } from './shim_html.mjs';
import { Motor } from './motor.mjs';

const CASES = [
  { seed: 1, luminosity: 0.6, steps: 1200 },
  { seed: 1, luminosity: 1.2, steps: 1200 },
  { seed: 17, luminosity: 0.6, steps: 1200 },
];

const COMMON = { powerBase: 0.47, beta: 0.94, sigma: 6.8, noise: 0.0079, band: 1.105, tOpt: 25, ptcTc: 18, ptcSharp: 4.1, minTemp: -6, maxTemp: 25 };

function runShim({ seed, luminosity, steps }) {
  const api = buildSandbox();
  api.setSeed(seed);
  api.resetSimulation();
  for (const [k, v] of Object.entries({ ...COMMON, luminosity })) api.els[k].value = String(v);
  api.els.dayNightToggle.checked = false;
  api.updateLabels();
  api.syncStateFromUI();
  for (let i = 0; i < steps; i++) api.stepHeadless();
  return { state: api.getState(), field: api.getField() };
}

function runMotor({ seed, luminosity, steps }) {
  const m = new Motor();
  m.setSeed(seed);
  m.resetSimulation();
  m.setParams({ ...COMMON, luminosity });
  for (let i = 0; i < steps; i++) m.paso();
  return { state: m.state, field: m.field };
}

function fieldChecksum(f) { return f.map(r => r.join(',')).join('|'); }

let allOk = true;
const report = [];
for (const c of CASES) {
  const a = runShim(c);
  const b = runMotor(c);
  const mismatches = [];
  for (const k of Object.keys(a.state)) {
    if (typeof a.state[k] === 'number' && !Object.is(a.state[k], b.state[k])) {
      mismatches.push(`${k}: shim=${a.state[k]} motor=${b.state[k]}`);
    }
  }
  const fieldOk = fieldChecksum(a.field) === fieldChecksum(b.field);
  const ok = mismatches.length === 0 && fieldOk;
  if (!ok) allOk = false;
  report.push({ caso: c, ok, mismatches, fieldOk });
}

console.log(JSON.stringify(report, null, 2));
console.log(allOk ? 'RESULTADO: motor.mjs == shim_html.mjs bit a bit — OK' : 'RESULTADO: DIVERGENCIA — motor.mjs mal, defecto bloqueante');
process.exit(allOk ? 0 : 1);
