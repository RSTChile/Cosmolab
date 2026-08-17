import { correrBarridoV75 } from './correr_barrido_v75.mjs';

async function correr(params, settle, measure) {
  const t0 = Date.now();
  const rows = correrBarridoV75({ seed:7, modo:'parada', from:0.6, to:1.4, steps:24, settle, measure, params });
  console.log(`  (${((Date.now()-t0)/1000).toFixed(1)}s)`);
  return rows;
}

const PARAMS1 = { powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:25, ptcTc:18, ptcSharp:4.1, minTemp:-6, maxTemp:25 };
const PARAMS2 = { powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:25, ptcTc:25, ptcSharp:8, minTemp:-6, maxTemp:25 };

function mostrar(rows, etiqueta) {
  console.log(`\n=== ${etiqueta} ===`);
  console.log('k  x       pasos_recup  convergio');
  rows.forEach((r,k)=>console.log(String(k).padStart(2), r.luminosidad.toFixed(4), String(r.pasos_recuperacion.toFixed(1)).padStart(10), r.recuperaron_todos));
  console.log('referencia: k0-4 (x0.6-0.739) NO convergen ~4000-5100 | k5(x0.774)=198 | k12(x1.017)=48(min) | k23(x1.4)=70');
}

console.log('Intento 1: ptcTc=18, ptcSharp=4.1, settle=300, measure=120');
const r1 = await correr(PARAMS1, 300, 120);
mostrar(r1, 'intento 1');
