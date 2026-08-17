// Cierre de investigación sobre datos YA EXISTENTES (bateria/, bateria2/,
// bateria3/) — no corre ninguna simulación nueva.
//  Tarea 1: ¿huella y acoplamiento son casi-identidad, o se separan en algún
//           régimen? + candidato de variable derivada menos redundante.
//  Tarea 2: distribución de H_absLocal/H_rel/H_noiseLocal en zonas "vivo" vs
//           "colapsado", material descriptivo para que Alexis defina κ_H.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { pearson, mean, std, writeCsv } from '../bateria/csv_util.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TEXTO = new Set(['modo', 'diagnostico']);

function readCsv(p) {
  const lines = fs.readFileSync(p, 'utf8').trim().split('\n');
  const header = lines[0].split(',');
  return lines.slice(1).map(line => {
    const vals = line.split(',');
    const row = {};
    header.forEach((h, i) => { row[h] = TEXTO.has(h) ? vals[i] : Number(vals[i]); });
    return row;
  });
}

// Cada archivo con su forma de agrupar en "series" (una serie = un barrido
// completo a lo largo del eje de luminosidad), igual criterio que los
// análisis de C en cada batería.
const ARCHIVOS = [
  { bateria: 'bateria', tag: 'A', file: 'bateria/experimento_A_repeticion.csv', clave: r => `${r.semilla}` },
  { bateria: 'bateria', tag: 'B', file: 'bateria/experimento_B_multivariable.csv', clave: r => `${r.beta}|${r.sigma}|${r.potencia_base}|${r.semilla}` },
  { bateria: 'bateria2', tag: 'D', file: 'bateria2/experimento_D_reinicio.csv', clave: r => `${r.modo}|${r.semilla}` },
  { bateria: 'bateria2', tag: 'Aprima', file: 'bateria2/experimento_Aprima_repeticion.csv', clave: r => `${r.semilla}` },
  { bateria: 'bateria2', tag: 'Bprima', file: 'bateria2/experimento_Bprima_multivariable.csv', clave: r => `${r.beta}|${r.t_optima}|${r.exponente_ptc}|${r.potencia_base}|${r.semilla}` },
  { bateria: 'bateria3', tag: 'D', file: 'bateria3/experimento_D_reinicio.csv', clave: r => `${r.modo}|${r.semilla}` },
  { bateria: 'bateria3', tag: 'Aprima', file: 'bateria3/experimento_Aprima_repeticion.csv', clave: r => `${r.semilla}` },
  { bateria: 'bateria3', tag: 'Bprima', file: 'bateria3/experimento_Bprima_multivariable.csv', clave: r => `${r.beta}|${r.t_optima}|${r.exponente_ptc}|${r.potencia_base}|${r.semilla}` },
];

const raiz = path.join(__dirname, '..');

// ══════════════════════════ TAREA 1 ══════════════════════════
const seriesInfo = []; // {bateria, tag, clave, params, r_huella_acoplamiento, r_huella_brecha, n}

for (const spec of ARCHIVOS) {
  const rows = readCsv(path.join(raiz, spec.file));
  const g = new Map();
  for (const r of rows) { const k = spec.clave(r); if (!g.has(k)) g.set(k, []); g.get(k).push(r); }
  for (const [k, serie] of g) {
    serie.sort((a, b) => a.luminosidad - b.luminosidad);
    const huella = serie.map(r => r.huella);
    const acoplamiento = serie.map(r => r.acoplamiento);
    // brecha = huella - 8*(1-acoplamiento) = |Tf-abioticTf| - |Tf-targetTf|
    // (magnitud, no la diferencia de referencias con signo — ver nota en el .md)
    const brecha = serie.map(r => r.huella - 8 * (1 - r.acoplamiento));
    const rHA = pearson(huella, acoplamiento);
    const rHB = pearson(huella, brecha);
    seriesInfo.push({
      bateria: spec.bateria, tag: spec.tag, clave: k,
      potencia_base: serie[0].potencia_base, beta: serie[0].beta,
      t_optima: serie[0].t_optima, exponente_ptc: serie[0].exponente_ptc,
      sigma: serie[0].sigma, modo: serie[0].modo ?? 'parada',
      n: serie.length, r_huella_acoplamiento: rHA, r_huella_brecha: rHB,
    });
  }
}

const rsHA = seriesInfo.map(s => s.r_huella_acoplamiento);
const rsHB = seriesInfo.map(s => s.r_huella_brecha);

console.log(`=== TAREA 1: huella vs acoplamiento, ${seriesInfo.length} series (barridos) en total ===`);
console.log(`r(huella,acoplamiento): media=${mean(rsHA).toFixed(4)} desv=${std(rsHA).toFixed(4)} min=${Math.min(...rsHA).toFixed(4)} max=${Math.max(...rsHA).toFixed(4)}`);
console.log(`  |r|<0.98: ${rsHA.filter(r => Math.abs(r) < 0.98).length}/${rsHA.length}`);
console.log(`  |r|<0.90: ${rsHA.filter(r => Math.abs(r) < 0.90).length}/${rsHA.length}`);
console.log(`  |r|<0.70: ${rsHA.filter(r => Math.abs(r) < 0.70).length}/${rsHA.length}`);
console.log(`  |r|<0.50: ${rsHA.filter(r => Math.abs(r) < 0.50).length}/${rsHA.length}`);

console.log(`\nr(huella, brecha=huella-8*(1-acoplamiento)): media=${mean(rsHB).toFixed(4)} desv=${std(rsHB).toFixed(4)} min=${Math.min(...rsHB).toFixed(4)} max=${Math.max(...rsHB).toFixed(4)}`);
console.log(`  |r|<0.98: ${rsHB.filter(r => Math.abs(r) < 0.98).length}/${rsHB.length}`);
console.log(`  |r|<0.90: ${rsHB.filter(r => Math.abs(r) < 0.90).length}/${rsHB.length}`);
console.log(`  |r|<0.70: ${rsHB.filter(r => Math.abs(r) < 0.70).length}/${rsHB.length}`);
console.log(`  |r|<0.50: ${rsHB.filter(r => Math.abs(r) < 0.50).length}/${rsHB.length}`);

const ordenadoHA = seriesInfo.slice().sort((a, b) => Math.abs(a.r_huella_acoplamiento) - Math.abs(b.r_huella_acoplamiento));
console.log('\n20 series con |r(huella,acoplamiento)| MÁS BAJO (donde más se separan):');
for (const s of ordenadoHA.slice(0, 20)) {
  console.log(`  ${s.bateria}/${s.tag} clave=${s.clave} r=${s.r_huella_acoplamiento.toFixed(4)} (pB=${s.potencia_base} β=${s.beta} tOpt=${s.t_optima} sharp=${s.exponente_ptc} sigma=${s.sigma} modo=${s.modo})`);
}
console.log('\n5 series con |r(huella,acoplamiento)| MÁS ALTO (casi-identidad):');
for (const s of ordenadoHA.slice(-5)) {
  console.log(`  ${s.bateria}/${s.tag} clave=${s.clave} r=${s.r_huella_acoplamiento.toFixed(4)}`);
}

writeCsv(path.join(__dirname, 'confound_huella_acoplamiento_por_serie.csv'), seriesInfo);

// ══════════════════════════ TAREA 2 ══════════════════════════
console.log('\n\n=== TAREA 2: distribución de H_absLocal/H_rel/H_noiseLocal, vivo vs colapsado ===');

function pct(arr, p) {
  const s = arr.slice().sort((a, b) => a - b);
  return s[Math.max(0, Math.min(s.length - 1, Math.floor(s.length * p)))];
}
function resumen(arr) {
  if (!arr.length) return null;
  return { n: arr.length, media: mean(arr), desv: std(arr), min: Math.min(...arr), p25: pct(arr, 0.25), mediana: pct(arr, 0.5), p75: pct(arr, 0.75), max: Math.max(...arr) };
}

const filasTarea2 = [];
for (const spec of ARCHIVOS) {
  const rows = readCsv(path.join(raiz, spec.file));
  const g = new Map();
  for (const r of rows) { const k = spec.clave(r); if (!g.has(k)) g.set(k, []); g.get(k).push(r); }

  const vivoH = { abs: [], rel: [], noise: [] };
  const colapsoH = { abs: [], rel: [], noise: [] };

  for (const [k, serie] of g) {
    const huellas = serie.map(r => r.huella);
    const hMin = Math.min(...huellas), hMax = Math.max(...huellas);
    const rango = hMax - hMin;
    if (rango <= 1e-9) continue;
    for (const r of serie) {
      const pos = (r.huella - hMin) / rango; // 0 = en el mínimo (colapso), 1 = en el máximo de esa serie (más vivo)
      if (pos <= 0.10) {
        colapsoH.abs.push(r.entropia_abs_local); colapsoH.rel.push(r.entropia_rel); colapsoH.noise.push(r.entropia_piso_local);
      } else if (pos >= 0.50) {
        vivoH.abs.push(r.entropia_abs_local); vivoH.rel.push(r.entropia_rel); vivoH.noise.push(r.entropia_piso_local);
      }
    }
  }

  const resVivo = { abs: resumen(vivoH.abs), rel: resumen(vivoH.rel), noise: resumen(vivoH.noise) };
  const resColapso = { abs: resumen(colapsoH.abs), rel: resumen(colapsoH.rel), noise: resumen(colapsoH.noise) };
  console.log(`\n-- ${spec.bateria}/${spec.tag} --`);
  console.log(`  VIVO (huella en el 50% superior de su serie, n=${resVivo.abs?.n}):`);
  console.log(`    H_absLocal: media=${resVivo.abs?.media.toFixed(4)} mediana=${resVivo.abs?.mediana.toFixed(4)} [${resVivo.abs?.min.toFixed(4)}, ${resVivo.abs?.max.toFixed(4)}]`);
  console.log(`    H_rel:      media=${resVivo.rel?.media.toFixed(4)} mediana=${resVivo.rel?.mediana.toFixed(4)} [${resVivo.rel?.min.toFixed(4)}, ${resVivo.rel?.max.toFixed(4)}]`);
  console.log(`    H_noiseLocal: media=${resVivo.noise?.media.toFixed(4)} mediana=${resVivo.noise?.mediana.toFixed(4)} [${resVivo.noise?.min.toFixed(4)}, ${resVivo.noise?.max.toFixed(4)}]`);
  console.log(`  COLAPSADO (huella en el 10% inferior de su serie, n=${resColapso.abs?.n}):`);
  console.log(`    H_absLocal: media=${resColapso.abs?.media.toFixed(4)} mediana=${resColapso.abs?.mediana.toFixed(4)} [${resColapso.abs?.min.toFixed(4)}, ${resColapso.abs?.max.toFixed(4)}]`);
  console.log(`    H_rel:      media=${resColapso.rel?.media.toFixed(4)} mediana=${resColapso.rel?.mediana.toFixed(4)} [${resColapso.rel?.min.toFixed(4)}, ${resColapso.rel?.max.toFixed(4)}]`);
  console.log(`    H_noiseLocal: media=${resColapso.noise?.media.toFixed(4)} mediana=${resColapso.noise?.mediana.toFixed(4)} [${resColapso.noise?.min.toFixed(4)}, ${resColapso.noise?.max.toFixed(4)}]`);

  for (const [zona, res] of [['vivo', resVivo], ['colapsado', resColapso]]) {
    for (const [variable, r] of [['H_absLocal', res.abs], ['H_rel', res.rel], ['H_noiseLocal', res.noise]]) {
      if (!r) continue;
      filasTarea2.push({ bateria: spec.bateria, experimento: spec.tag, zona, variable, n: r.n, media: r.media, desv: r.desv, min: r.min, p25: r.p25, mediana: r.mediana, p75: r.p75, max: r.max });
    }
  }
}

writeCsv(path.join(__dirname, 'distribucion_H_vivo_vs_colapsado.csv'), filasTarea2);
console.log('\n\nEscrito: confound_huella_acoplamiento_por_serie.csv, distribucion_H_vivo_vs_colapsado.csv');
