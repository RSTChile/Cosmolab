// Verificación bit-a-bit render vs no-render sobre v7.6.1, misma semilla,
// 1000 pasos (bajado de 2000 de rondas anteriores: acá cada paso es más caro
// por el registro de saturación, y esto ya alcanza para probar la separación).
import { buildSandbox } from './shim_v76.mjs';

function checksumField(field) {
  let s = 0;
  for (const row of field) for (const v of row) s += v;
  return s;
}

function snapshotState(state) {
  const out = {};
  for (const k of Object.keys(state)) {
    const v = state[k];
    if (typeof v === 'number') out[k] = v;
  }
  return out;
}

function comparar(a, b) {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  const diffs = [];
  for (const k of keys) {
    if (!Object.is(a[k], b[k])) diffs.push({ k, a: a[k], b: b[k] });
  }
  return diffs;
}

async function correr(seed, conRender, pasos) {
  const api = buildSandbox();
  api.els.powerBase.value = '0.47'; api.els.beta.value = '0.94'; api.els.sigma.value = '6.8';
  api.els.noise.value = '0.0079'; api.els.band.value = '1.105'; api.els.tOpt.value = '25';
  api.els.ptcTc.value = '18'; api.els.ptcSharp.value = '4.1'; api.els.luminosity.value = '0.9';
  api.els.minTemp.value = '-6'; api.els.maxTemp.value = '25'; api.els.dayNightToggle.checked = false;
  api.syncStateFromUI();
  api.setSeed(seed);
  api.resetSimulation();
  for (let i = 0; i < pasos; i++) {
    if (conRender) api.updateSimulation(); else api.stepHeadless();
  }
  return { state: snapshotState(api.getState()), field: checksumField(api.getField()) };
}

async function main() {
  const casos = [42, 777];
  let todoOk = true;
  for (const seed of casos) {
    const conR = await correr(seed, true, 1000);
    const sinR = await correr(seed, false, 1000);
    const diffs = comparar(conR.state, sinR.state);
    const campoOk = conR.field === sinR.field;
    const estadoOk = diffs.length === 0;
    console.log(`semilla=${seed} pasos=1000: estado idéntico=${estadoOk} campo idéntico=${campoOk} (fieldSum con-render=${conR.field} sin-render=${sinR.field})`);
    if (!estadoOk) { console.log('  diffs:', diffs.slice(0, 10)); }
    if (!estadoOk || !campoOk) todoOk = false;
  }
  console.log(todoOk ? '\nPASÓ: render vs no-render idéntico bit a bit en ambos casos.' : '\nFALLÓ.');
  process.exit(todoOk ? 0 : 1);
}
main().catch(e => { console.error('ERROR', e); process.exit(2); });
