// Tarea 2 + Tarea 3 del encargo v2: verificación obligatoria (Paso 0) y chequeo
// contra la referencia numérica de Alexis, usando el shim del script REAL.
import { buildSandbox } from './shim_html2.mjs';
import crypto from 'node:crypto';

function sha256(text) {
  return crypto.createHash('sha256').update(text).digest('hex');
}

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

function parseCSV(csv) {
  const [headerLine, ...lines] = csv.trim().split('\n');
  const header = headerLine.split(',');
  return lines.map(line => {
    const cols = line.split(',');
    const row = {};
    header.forEach((h, i) => { row[h] = cols[i]; });
    return row;
  });
}

function filaMinimaHuella(csv) {
  const rows = parseCSV(csv);
  let best = null, bestK = -1;
  rows.forEach((r, k) => {
    const f = Number(r.footprint);
    if (best === null || f < best) { best = f; bestK = k; }
  });
  return { k: bestK, x: Number(rows[bestK].x), footprint: best };
}

const PARAMS_INTENTO_1 = { powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:25, ptcTc:18, ptcSharp:4.1 };
const PARAMS_INTENTO_2 = { powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:25, ptcTc:25, ptcSharp:8 };

const BASE_BARRIDO = { desde:0.6, hasta:1.4, puntos:8, settle:40, measure:30, trazas:0 };

async function main() {
  console.log('=== Tarea 2: Paso 0 — verificación obligatoria ===');
  const resultados = {};

  {
    const api = buildSandbox();
    const csvX1 = await correrBarrido(api, PARAMS_INTENTO_1, { ...BASE_BARRIDO, modo:'parada', semilla:7 });
    resultados.X = csvX1;
    console.log('1) semilla=7 modo=parada -> capturado, sha256=', sha256(csvX1).slice(0,16));
  }
  {
    const api = buildSandbox();
    const csvX2 = await correrBarrido(api, PARAMS_INTENTO_1, { ...BASE_BARRIDO, modo:'parada', semilla:7 });
    const identico = csvX2 === resultados.X;
    console.log('2) semilla=7 modo=parada (repetido, proceso NUEVO) -> ¿idéntico a X byte a byte?', identico, 'sha256=', sha256(csvX2).slice(0,16));
    resultados.identico2 = identico;
    resultados.X2 = csvX2;
  }
  {
    const api = buildSandbox();
    const csv99 = await correrBarrido(api, PARAMS_INTENTO_1, { ...BASE_BARRIDO, modo:'parada', semilla:99 });
    const distinto = csv99 !== resultados.X;
    console.log('3) semilla=99 modo=parada -> ¿distinto de X?', distinto, 'sha256=', sha256(csv99).slice(0,16));
    resultados.distinto99 = distinto;
  }
  {
    const api = buildSandbox();
    const csvInicio = await correrBarrido(api, PARAMS_INTENTO_1, { ...BASE_BARRIDO, modo:'inicio', semilla:7 });
    const distinto = csvInicio !== resultados.X;
    console.log('4) semilla=7 modo=inicio -> ¿distinto de X?', distinto, 'sha256=', sha256(csvInicio).slice(0,16));
    resultados.distintoInicio = distinto;
    resultados.csvInicio = csvInicio;
  }

  const paso0_ok = resultados.identico2 && resultados.distinto99 && resultados.distintoInicio;
  console.log('\nPaso 0 ' + (paso0_ok ? 'PASÓ.' : 'FALLÓ — hay una fuente de azar sin sembrar o el modo/semilla no afecta el resultado.'));

  if (!paso0_ok) {
    console.log('DETENIDO: no se sigue a la Tarea 3.');
    process.exit(1);
  }

  console.log('\n=== Tarea 3: referencia numérica de Alexis ===');
  const minParada = filaMinimaHuella(resultados.X);
  const minInicio = filaMinimaHuella(resultados.csvInicio);
  console.log('modo=parada: mínimo footprint en k=' + minParada.k + ' x=' + minParada.x.toFixed(4) + ' footprint=' + minParada.footprint.toFixed(4));
  console.log('modo=inicio: mínimo footprint en k=' + minInicio.k + ' x=' + minInicio.x.toFixed(4) + ' footprint=' + minInicio.footprint.toFixed(4));
  console.log('esperado (intento 1, ptcTc=18 ptcSharp=4.1): parada k=5 x≈1.171 · inicio k=4 x≈1.057');

  const coincideIntento1 = minParada.k === 5 && minInicio.k === 4;
  console.log('¿coincide con intento 1?', coincideIntento1);

  if (!coincideIntento1) {
    console.log('\nProbando intento 2 (ptcTc=25, ptcSharp=8, valores por defecto del HTML)...');
    const apiP = buildSandbox();
    const csvParada2 = await correrBarrido(apiP, PARAMS_INTENTO_2, { ...BASE_BARRIDO, modo:'parada', semilla:7 });
    const apiI = buildSandbox();
    const csvInicio2 = await correrBarrido(apiI, PARAMS_INTENTO_2, { ...BASE_BARRIDO, modo:'inicio', semilla:7 });
    const minParada2 = filaMinimaHuella(csvParada2);
    const minInicio2 = filaMinimaHuella(csvInicio2);
    console.log('modo=parada (intento2): k=' + minParada2.k + ' x=' + minParada2.x.toFixed(4));
    console.log('modo=inicio (intento2): k=' + minInicio2.k + ' x=' + minInicio2.x.toFixed(4));
    const coincideIntento2 = minParada2.k === 5 && minInicio2.k === 4;
    console.log('¿coincide con intento 2?', coincideIntento2);
    if (!coincideIntento2) {
      console.log('\nNINGÚN intento coincidió con la referencia. DETENIDO — reportar en defectos_encontrados2.md.');
      process.exit(2);
    } else {
      console.log('\nIntento 2 coincide. Usar ptcTc=25, ptcSharp=8 en Paso 1 y en adelante.');
    }
  } else {
    console.log('\nIntento 1 coincide. Usar ptcTc=18, ptcSharp=4.1 (los valores del encargo) en Paso 1 y en adelante.');
  }
}

main().catch(e => { console.error('ERROR:', e); process.exit(3); });
