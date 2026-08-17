// Corre un barrido completo (60 puntos, eje 0.60->1.40) con los parámetros
// dados por argv y vuelca las filas COMPLETAS (incluye asent_pasos y rec_1..5
// crudos, no solo resumen) a un JSON, para analizar la distribución real de
// asentarHastaEquilibrio/medirRecuperacion antes de decidir si achicar los topes.
import fs from 'node:fs';
import { correrBarridoV76 } from './correr_barrido_v76.mjs';

const [, , outPath, paramsJSON] = process.argv;
const params = JSON.parse(paramsJSON);
const t0 = Date.now();
const rows = correrBarridoV76({ seed: 1, modo: 'parada', from: 0.60, to: 1.40, steps: 60, settle: 300, measure: 120, params });
const seg = (Date.now() - t0) / 1000;
fs.writeFileSync(outPath, JSON.stringify({ params, seg, rows }));
console.error(`listo ${outPath} en ${seg.toFixed(1)}s`);
