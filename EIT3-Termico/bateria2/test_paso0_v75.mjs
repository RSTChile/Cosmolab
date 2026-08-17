// Rehace el Paso 0 (verificación obligatoria) específicamente contra v7.5.html
// — no se asume que hereda la validación de v7.4.1, aunque comparta motor.
import { buildSandbox } from './shim_v75.mjs';
import crypto from 'node:crypto';

function sha256(text) { return crypto.createHash('sha256').update(text).digest('hex'); }

function fijarParametrosFisicos(api, p) {
  api.els.powerBase.value = String(p.powerBase);
  api.els.beta.value = String(p.beta);
  api.els.sigma.value = String(p.sigma);
  api.els.noise.value = String(p.noise);
  api.els.band.value = String(p.band);
  api.els.tOpt.value = String(p.tOpt);
  api.els.ptcTc.value = String(p.ptcTc);
  api.els.ptcSharp.value = String(p.ptcSharp);
  api.els.minTemp.value = '-6';
  api.els.maxTemp.value = '25';
  api.els.dayNightToggle.checked = false;
}

function fijarBarrido(api, { desde, hasta, puntos, settle, measure, trazas, modo, semilla }) {
  api.getEl('sweepAxis').value = 'luminosity';
  api.getEl('sweepFrom').value = String(desde);
  api.getEl('sweepTo').value = String(hasta);
  api.getEl('sweepSteps').value = String(puntos);
  api.getEl('sweepSettle').value = String(settle);
  api.getEl('sweepMeasure').value = String(measure);
  api.getEl('sweepNoise').value = '0';
  api.getEl('sweepTraceN').value = String(trazas);
  api.getEl('sweepReset').value = modo;
  api.getEl('seedInput').value = String(semilla);
}

async function correrBarrido(api, params, barrido) {
  fijarParametrosFisicos(api, params);
  fijarBarrido(api, barrido);
  api.limpiarDescargas();
  await api.runSweep();
  const descargas = api.getDescargas();
  const csv = descargas.find(d => d.nombre === 'EIT3_kappaH_barrido.csv');
  if (!csv) throw new Error('no se capturó EIT3_kappaH_barrido.csv');
  return csv.contenido;
}

// Barrido chico a propósito, y en un tramo del eje (1.2->1.4) que la
// referencia de Alexis muestra como de recuperación RÁPIDA (lum=1.4 -> 70
// pasos), para no pagar el costo de medirRecuperacion (TOPE_REC=20000 fijo,
// no depende de settle) en la zona lenta — Paso 0 solo necesita confirmar
// reproducibilidad de semilla/modo, no medir recuperación.
const PARAMS = { powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:25, ptcTc:18, ptcSharp:4.1 };
const BASE_BARRIDO = { desde:1.2, hasta:1.4, puntos:3, settle:40, measure:30, trazas:0 };

async function main() {
  console.log('=== Paso 0 (v7.5) — verificación obligatoria ===');
  const resultados = {};

  {
    const api = buildSandbox();
    resultados.X = await correrBarrido(api, PARAMS, { ...BASE_BARRIDO, modo:'parada', semilla:7 });
    console.log('1) semilla=7 modo=parada -> capturado, sha256=', sha256(resultados.X).slice(0,16));
  }
  {
    const api = buildSandbox();
    const csv2 = await correrBarrido(api, PARAMS, { ...BASE_BARRIDO, modo:'parada', semilla:7 });
    const identico = csv2 === resultados.X;
    console.log('2) semilla=7 modo=parada (repetido) -> ¿idéntico a X?', identico, 'sha256=', sha256(csv2).slice(0,16));
    resultados.identico2 = identico;
  }
  {
    const api = buildSandbox();
    const csv99 = await correrBarrido(api, PARAMS, { ...BASE_BARRIDO, modo:'parada', semilla:99 });
    const distinto = csv99 !== resultados.X;
    console.log('3) semilla=99 modo=parada -> ¿distinto de X?', distinto, 'sha256=', sha256(csv99).slice(0,16));
    resultados.distinto99 = distinto;
  }
  {
    const api = buildSandbox();
    const csvInicio = await correrBarrido(api, PARAMS, { ...BASE_BARRIDO, modo:'inicio', semilla:7 });
    const distinto = csvInicio !== resultados.X;
    console.log('4) semilla=7 modo=inicio -> ¿distinto de X?', distinto, 'sha256=', sha256(csvInicio).slice(0,16));
    resultados.distintoInicio = distinto;
  }

  const ok = resultados.identico2 && resultados.distinto99 && resultados.distintoInicio;
  console.log('\nPaso 0 (v7.5) ' + (ok ? 'PASÓ.' : 'FALLÓ.'));
  process.exit(ok ? 0 : 1);
}
main().catch(e => { console.error('ERROR:', e); process.exit(2); });
