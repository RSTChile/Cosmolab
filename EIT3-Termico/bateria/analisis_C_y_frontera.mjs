// Experimento C (barajado) + Tarea 9 (frontera de la huella + meseta >1.40).
// Solo estadística descriptiva: NO concluye si hay hallazgo o no.
import { readCsv, writeCsv, pearson, mulberry32, shuffleWith, mean, std, percentileOfValue } from './csv_util.mjs';

const N_SHUFFLES = 1000;

function grupoPorSemilla(rows) {
  const g = new Map();
  for (const r of rows) {
    if (!g.has(r.semilla)) g.set(r.semilla, []);
    g.get(r.semilla).push(r);
  }
  for (const arr of g.values()) arr.sort((a, b) => a.luminosidad - b.luminosidad);
  return g;
}

// Experimento C: para una serie ordenada por el eje, baraja la correspondencia
// eje<->huella (manteniendo los valores de huella y de entropía intactos) y
// recalcula la correlación. Repite N_SHUFFLES veces.
function barajadoDeUnaSerie(huella, entropiaAbs, seedForShuffle) {
  const rReal = pearson(huella, entropiaAbs);
  const rng = mulberry32(seedForShuffle >>> 0);
  const nulos = [];
  for (let i = 0; i < N_SHUFFLES; i++) {
    const huellaBarajada = shuffleWith(rng, huella);
    nulos.push(pearson(huellaBarajada, entropiaAbs));
  }
  const percentil = percentileOfValue(rReal, nulos);
  return { rReal, mediaNula: mean(nulos), desviacionNula: std(nulos), percentil, nulos };
}

// Tarea 9: frontera = luminosidad donde la huella toca su mínimo global,
// EXCLUYENDO puntos con sensor saturado (saturacion_sensor=1): esos puntos
// caen en el borde de arriba del eje por saturación del PTC, no por colapso
// biótico, y contaminan un criterio de "pendiente más negativa" (ver
// defectos_encontrados.md — primer criterio probado, descartado).
// La huella real muestra un mínimo local en forma de V (cae y luego se
// recupera) alrededor de luminosidad~1.0, distinto del artefacto de borde.
function frontera(rows) {
  const candidatos = rows.filter(r => r.saturacion_sensor === 0);
  let min = Infinity, xMin = null;
  for (const r of candidatos) { if (r.huella < min) { min = r.huella; xMin = r.luminosidad; } }
  return { xFrontera: xMin, huellaMinima: min };
}

function mesetaStats(rows) {
  const meseta = rows.filter(r => r.luminosidad > 1.40);
  const resto = rows.filter(r => r.luminosidad <= 1.40);
  const rango = (arr, campo) => { const vs = arr.map(r => r[campo]); return Math.max(...vs) - Math.min(...vs); };
  return {
    n_meseta: meseta.length, n_resto: resto.length,
    rango_huella_meseta: meseta.length ? rango(meseta, 'huella') : null,
    rango_huella_resto: resto.length ? rango(resto, 'huella') : null,
    rango_mult_meseta: meseta.length ? rango(meseta, 'multiplicidad') : null,
    rango_mult_resto: resto.length ? rango(resto, 'multiplicidad') : null,
    rango_acoplamiento_meseta: meseta.length ? rango(meseta, 'acoplamiento') : null,
    rango_acoplamiento_resto: resto.length ? rango(resto, 'acoplamiento') : null,
  };
}

function procesarExperimentoA(inPath, outCsvPath) {
  const rows = readCsv(inPath);
  const porSemilla = grupoPorSemilla(rows);
  const filasC = [];
  const fronteras = [];
  for (const [semilla, serie] of porSemilla) {
    const huella = serie.map(r => r.huella);
    const entropia = serie.map(r => r.entropia_abs_local);
    const res = barajadoDeUnaSerie(huella, entropia, 10000 + semilla);
    filasC.push({ semilla, r_real: res.rReal, percentil_r_real_en_nula: res.percentil, media_nula: res.mediaNula, desviacion_nula: res.desviacionNula });
    const f = frontera(serie);
    fronteras.push({ semilla, x_frontera: f.xFrontera, huella_minima: f.huellaMinima });
  }
  writeCsv(outCsvPath, filasC);

  const rs = filasC.map(f => f.r_real);
  const pcts = filasC.map(f => f.percentil_r_real_en_nula);
  const xs = fronteras.map(f => f.x_frontera);
  const fueraDe95 = pcts.filter(p => p <= 2.5 || p >= 97.5).length;
  const fueraDe99 = pcts.filter(p => p <= 0.5 || p >= 99.5).length;

  const meseta = mesetaStats(rows);

  return {
    filasC, fronteras,
    resumen: {
      n_semillas: filasC.length,
      r_media: mean(rs), r_desv: std(rs), r_min: Math.min(...rs), r_max: Math.max(...rs),
      percentil_media: mean(pcts), percentil_mediana: pcts.slice().sort((a, b) => a - b)[Math.floor(pcts.length / 2)],
      percentil_min: Math.min(...pcts), percentil_max: Math.max(...pcts),
      fuera_de_95_de_30: fueraDe95, fuera_de_99_de_30: fueraDe99,
      frontera_media: mean(xs), frontera_desv: std(xs), frontera_min: Math.min(...xs), frontera_max: Math.max(...xs),
      meseta,
    },
  };
}

function procesarExperimentoB(inPath, outCsvPath) {
  const rows = readCsv(inPath);
  // agrupar por combinación (beta,sigma,potencia_base,semilla) para C
  const clave = r => `${r.beta}|${r.sigma}|${r.potencia_base}|${r.semilla}`;
  const g = new Map();
  for (const r of rows) { const k = clave(r); if (!g.has(k)) g.set(k, []); g.get(k).push(r); }
  for (const arr of g.values()) arr.sort((a, b) => a.luminosidad - b.luminosidad);

  const filasC = [];
  let seedCounter = 20000;
  const fronterasPorCombo = new Map(); // clave combo (sin semilla) -> [xFrontera,...]
  for (const [key, serie] of g) {
    const huella = serie.map(r => r.huella);
    const entropia = serie.map(r => r.entropia_abs_local);
    const res = barajadoDeUnaSerie(huella, entropia, seedCounter++);
    const { beta, sigma, potencia_base, semilla } = serie[0];
    filasC.push({ beta, sigma, potencia_base, semilla, r_real: res.rReal, percentil_r_real_en_nula: res.percentil, media_nula: res.mediaNula, desviacion_nula: res.desviacionNula });
    const f = frontera(serie);
    const comboKey = `${beta}|${sigma}|${potencia_base}`;
    if (!fronterasPorCombo.has(comboKey)) fronterasPorCombo.set(comboKey, []);
    fronterasPorCombo.get(comboKey).push(f.xFrontera);
  }
  writeCsv(outCsvPath, filasC);

  // tabla descriptiva: frontera media por nivel de cada parámetro (promediando sobre los otros factores y semillas)
  function porNivel(param) {
    const niveles = new Map();
    for (const [comboKey, xs] of fronterasPorCombo) {
      const [beta, sigma, potencia_base] = comboKey.split('|').map(Number);
      const val = { beta, sigma, potencia_base }[param];
      if (!niveles.has(val)) niveles.set(val, []);
      niveles.get(val).push(...xs);
    }
    return [...niveles.entries()].sort((a, b) => a[0] - b[0]).map(([nivel, xs]) => ({ parametro: param, nivel, frontera_media: mean(xs), frontera_desv: std(xs), n: xs.length }));
  }
  const tablaFrontera = [...porNivel('beta'), ...porNivel('sigma'), ...porNivel('potencia_base')];

  const meseta = mesetaStats(rows);
  const rs = filasC.map(f => f.r_real);
  const pcts = filasC.map(f => f.percentil_r_real_en_nula);

  return {
    filasC, tablaFrontera,
    resumen: {
      n_combos_semilla: filasC.length,
      r_media: mean(rs), r_desv: std(rs),
      percentil_media: mean(pcts), percentil_desv: std(pcts),
      meseta,
    },
  };
}

const modo = process.argv[2]; // 'A' o 'B'
const inPath = process.argv[3];
const outCsvPath = process.argv[4];
const outJsonPath = process.argv[5];

const resultado = modo === 'A' ? procesarExperimentoA(inPath, outCsvPath) : procesarExperimentoB(inPath, outCsvPath);
if (outJsonPath) {
  const { fs } = await import('node:fs').then(m => ({ fs: m.default }));
  fs.writeFileSync(outJsonPath, JSON.stringify(resultado, (k, v) => k === 'nulos' ? undefined : v, 2));
}
console.log(JSON.stringify(resultado.resumen, null, 2));
