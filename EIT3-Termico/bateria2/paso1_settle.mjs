// Paso 1 del encargo v2: sensibilidad al settle. Barrido de 20 puntos,
// luminosidad 0.25->1.95, modo=parada, measure=120, semilla=1, con settle en
// 150/300/600/1200/2400. Compara huella y entropía punto a punto entre niveles
// consecutivos.
import { correrBarrido2 } from './correr_barrido2.mjs';
import fs from 'node:fs';

const PARAMS = { powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:25, ptcTc:18, ptcSharp:4.1, minTemp:-6, maxTemp:25 };
const SEED = 1;
const NIVELES = [150, 300, 600, 1200, 2400];
const PUNTOS = 20, DESDE = 0.25, HASTA = 1.95, MEASURE = 120;

function correr(settle) {
  const t0 = Date.now();
  const rows = correrBarrido2({ seed: SEED, modo: 'parada', from: DESDE, to: HASTA, steps: PUNTOS, settle, measure: MEASURE, params: PARAMS });
  const seg = (Date.now() - t0) / 1000;
  return { rows, seg };
}

function diffNiveles(rowsA, rowsB) {
  let sumAbsFoot = 0, maxAbsFoot = 0, sumAbsH = 0, maxAbsH = 0;
  let sumRelFoot = 0, maxRelFoot = 0, sumRelH = 0, maxRelH = 0;
  for (let i = 0; i < rowsA.length; i++) {
    const fa = rowsA[i].huella, fb = rowsB[i].huella;
    const ha = rowsA[i].entropia_abs_local, hb = rowsB[i].entropia_abs_local;
    const dFoot = Math.abs(fb - fa), dH = Math.abs(hb - ha);
    sumAbsFoot += dFoot; maxAbsFoot = Math.max(maxAbsFoot, dFoot);
    sumAbsH += dH; maxAbsH = Math.max(maxAbsH, dH);
    const relFoot = dFoot / Math.max(1e-9, Math.abs(fa));
    const relH = dH / Math.max(1e-9, Math.abs(ha));
    sumRelFoot += relFoot; maxRelFoot = Math.max(maxRelFoot, relFoot);
    sumRelH += relH; maxRelH = Math.max(maxRelH, relH);
  }
  const n = rowsA.length;
  return {
    huella_dif_abs_media: sumAbsFoot / n, huella_dif_abs_max: maxAbsFoot,
    huella_dif_rel_max_pct: maxRelFoot * 100,
    entropia_dif_abs_media: sumAbsH / n, entropia_dif_abs_max: maxAbsH,
    entropia_dif_rel_max_pct: maxRelH * 100,
  };
}

function main() {
  console.log('=== Paso 1: sensibilidad al settle ===');
  const resultadosPorNivel = {};
  const tiempos = {};
  for (const settle of NIVELES) {
    const { rows, seg } = correr(settle);
    resultadosPorNivel[settle] = rows;
    tiempos[settle] = seg;
    console.log(`settle=${settle}: ${seg.toFixed(2)}s (${(20 * (Math.min(80, Math.max(20, Math.round(settle / 2) || 40)) + settle + MEASURE) / seg).toFixed(0)} pasos/s aprox)`);
  }

  const filas = [];
  for (let i = 1; i < NIVELES.length; i++) {
    const prev = NIVELES[i - 1], cur = NIVELES[i];
    const d = diffNiveles(resultadosPorNivel[prev], resultadosPorNivel[cur]);
    filas.push({ transicion: `${prev}->${cur}`, ...d });
  }

  console.log('\ntransicion | huella_dif_abs_media | huella_dif_abs_max | huella_dif_rel_max_% | entropia_dif_abs_media | entropia_dif_abs_max | entropia_dif_rel_max_%');
  for (const f of filas) {
    console.log(`${f.transicion} | ${f.huella_dif_abs_media.toFixed(5)} | ${f.huella_dif_abs_max.toFixed(5)} | ${f.huella_dif_rel_max_pct.toFixed(3)} | ${f.entropia_dif_abs_media.toFixed(5)} | ${f.entropia_dif_abs_max.toFixed(5)} | ${f.entropia_dif_rel_max_pct.toFixed(3)}`);
  }

  // Criterio: cambio relativo MÁXIMO (sobre los 20 puntos) < 1% en huella Y en
  // entropía respecto al nivel anterior, sostenido también en la transición
  // siguiente (para no confiar en una casualidad de un solo escalón).
  const UMBRAL = 1.0; // %
  let settleEstable = null;
  for (let i = 0; i < filas.length; i++) {
    const cumpleEste = filas[i].huella_dif_rel_max_pct < UMBRAL && filas[i].entropia_dif_rel_max_pct < UMBRAL;
    const cumpleSiguiente = (i + 1 < filas.length)
      ? (filas[i + 1].huella_dif_rel_max_pct < UMBRAL && filas[i + 1].entropia_dif_rel_max_pct < UMBRAL)
      : null; // última transición no tiene siguiente para confirmar
    if (cumpleEste && (cumpleSiguiente === true)) {
      settleEstable = NIVELES[i + 1]; // el nivel DESTINO de esta transición es el primero "estable"
      break;
    }
  }

  console.log('\ncriterio: cambio relativo máximo (20 puntos) < 1% en huella Y entropía, sostenido en la transición siguiente');
  if (settleEstable) {
    console.log(`settle recomendado: ${settleEstable}`);
  } else {
    const ultima = filas[filas.length - 1];
    const ultimaCumple = ultima.huella_dif_rel_max_pct < UMBRAL && ultima.entropia_dif_rel_max_pct < UMBRAL;
    if (ultimaCumple) {
      console.log(`settle recomendado: 2400 (única transición que cumple es la última, sin siguiente para confirmar — se acepta con nota)`);
      settleEstable = 2400;
    } else {
      console.log('NO ESTABILIZÓ ni siquiera en la última transición (1200->2400). No se recomienda settle — replantear protocolo.');
    }
  }

  fs.writeFileSync('paso1_tabla.json', JSON.stringify({ filas, tiempos, settleEstable }, null, 2));
  console.log('\ntiempos por settle (s, 20 barridos... no, 20 PUNTOS de 1 barrido cada nivel):', tiempos);
}
main();
