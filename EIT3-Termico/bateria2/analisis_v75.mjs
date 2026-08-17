// Análisis descriptivo de D, A', B' + Experimento C (barajado) para la
// segunda batería (v7.5). Reusa las utilidades de la primera batería
// (bateria/csv_util.mjs: pearson, mulberry32, shuffleWith, mean, std,
// percentileOfValue) pero con un lector de CSV propio porque estos archivos
// tienen columnas de texto nuevas ('modo', además de 'diagnostico').
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { pearson, mulberry32, shuffleWith, mean, std, percentileOfValue, writeCsv } from '../bateria/csv_util.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TEXTO = new Set(['modo', 'diagnostico']);

function readCsvV75(p) {
  const lines = fs.readFileSync(p, 'utf8').trim().split('\n');
  const header = lines[0].split(',');
  return lines.slice(1).map(line => {
    const vals = line.split(',');
    const row = {};
    header.forEach((h, i) => { row[h] = TEXTO.has(h) ? vals[i] : Number(vals[i]); });
    return row;
  });
}

const N_SHUFFLES = 1000;

function barajadoDeUnaSerie(huella, entropiaAbs, seedForShuffle) {
  const rReal = pearson(huella, entropiaAbs);
  const rng = mulberry32(seedForShuffle >>> 0);
  const nulos = [];
  for (let i = 0; i < N_SHUFFLES; i++) {
    const huellaBarajada = shuffleWith(rng, huella);
    nulos.push(pearson(huellaBarajada, entropiaAbs));
  }
  return { rReal, mediaNula: mean(nulos), desviacionNula: std(nulos), percentil: percentileOfValue(rReal, nulos) };
}

function frontera(rows) {
  const candidatos = rows.filter(r => r.saturacion_sensor === 0);
  let min = Infinity, xMin = null;
  for (const r of candidatos) { if (r.huella < min) { min = r.huella; xMin = r.luminosidad; } }
  return { xFrontera: xMin, huellaMinima: min };
}

// ── Experimento D: agrupar por (modo, semilla) ──────────────────────────────
function analizarD() {
  const rows = readCsvV75(path.join(__dirname, 'experimento_D_reinicio.csv'));
  const g = new Map();
  for (const r of rows) {
    const k = `${r.modo}|${r.semilla}`;
    if (!g.has(k)) g.set(k, []);
    g.get(k).push(r);
  }
  for (const arr of g.values()) arr.sort((a, b) => a.luminosidad - b.luminosidad);

  const porModo = { parada: [], inicio: [] };
  for (const [key, serie] of g) {
    const [modo, semilla] = key.split('|');
    const huella = serie.map(r => r.huella);
    const entropia = serie.map(r => r.entropia_abs_local);
    const r = pearson(huella, entropia);
    const f = frontera(serie);
    porModo[modo].push({ semilla: Number(semilla), r, xFrontera: f.xFrontera, huellaMinima: f.huellaMinima });
  }
  const resumen = {};
  for (const modo of ['parada', 'inicio']) {
    const arr = porModo[modo];
    const rs = arr.map(a => a.r), xs = arr.map(a => a.xFrontera);
    resumen[modo] = {
      n_semillas: arr.length,
      r_media: mean(rs), r_desv: std(rs),
      frontera_media: mean(xs), frontera_desv: std(xs), frontera_min: Math.min(...xs), frontera_max: Math.max(...xs),
    };
  }
  return { resumen, detalle: porModo };
}

// ── Experimento A': por semilla, igual que la batería anterior ─────────────
function analizarAprima() {
  const rows = readCsvV75(path.join(__dirname, 'experimento_Aprima_repeticion.csv'));
  const g = new Map();
  for (const r of rows) { if (!g.has(r.semilla)) g.set(r.semilla, []); g.get(r.semilla).push(r); }
  for (const arr of g.values()) arr.sort((a, b) => a.luminosidad - b.luminosidad);

  const filasC = [];
  const fronteras = [];
  for (const [semilla, serie] of g) {
    const huella = serie.map(r => r.huella);
    const entropia = serie.map(r => r.entropia_abs_local);
    const res = barajadoDeUnaSerie(huella, entropia, 30000 + semilla);
    filasC.push({ semilla, r_real: res.rReal, percentil_r_real_en_nula: res.percentil, media_nula: res.mediaNula, desviacion_nula: res.desviacionNula });
    const f = frontera(serie);
    fronteras.push({ semilla, x_frontera: f.xFrontera, huella_minima: f.huellaMinima });
  }
  writeCsv(path.join(__dirname, 'experimento_C_barajado.csv'), filasC);

  const rs = filasC.map(f => f.r_real);
  const pcts = filasC.map(f => f.percentil_r_real_en_nula);
  const xs = fronteras.map(f => f.x_frontera);
  const fueraDe95 = pcts.filter(p => p <= 2.5 || p >= 97.5).length;
  const fueraDe99 = pcts.filter(p => p <= 0.5 || p >= 99.5).length;

  return {
    resumen: {
      n_semillas: filasC.length,
      r_media: mean(rs), r_desv: std(rs), r_min: Math.min(...rs), r_max: Math.max(...rs),
      percentil_media: mean(pcts), fuera_de_95: fueraDe95, fuera_de_99: fueraDe99,
      frontera_media: mean(xs), frontera_desv: std(xs), frontera_min: Math.min(...xs), frontera_max: Math.max(...xs),
    },
  };
}

// ── Experimento B': por combinación (beta,tOpt,ptcSharp,potencia_base,semilla) ──
function analizarBprima() {
  const rows = readCsvV75(path.join(__dirname, 'experimento_Bprima_multivariable.csv'));
  const clave = r => `${r.beta}|${r.t_optima}|${r.exponente_ptc}|${r.potencia_base}|${r.semilla}`;
  const g = new Map();
  for (const r of rows) { const k = clave(r); if (!g.has(k)) g.set(k, []); g.get(k).push(r); }
  for (const arr of g.values()) arr.sort((a, b) => a.luminosidad - b.luminosidad);

  const filasC = [];
  let seedCounter = 40000;
  // saturación por combinación (sin semilla, agregando las 10 semillas juntas)
  const porCombo = new Map(); // comboKey (sin semilla) -> {puntos: [], fronteras: []}
  for (const [key, serie] of g) {
    const huella = serie.map(r => r.huella);
    const entropia = serie.map(r => r.entropia_abs_local);
    const res = barajadoDeUnaSerie(huella, entropia, seedCounter++);
    const { beta, t_optima, exponente_ptc, potencia_base, semilla } = serie[0];
    filasC.push({ beta, t_optima, exponente_ptc, potencia_base, semilla, r_real: res.rReal, percentil_r_real_en_nula: res.percentil, media_nula: res.mediaNula, desviacion_nula: res.desviacionNula });
    const f = frontera(serie);
    const comboKey = `${beta}|${t_optima}|${exponente_ptc}|${potencia_base}`;
    if (!porCombo.has(comboKey)) porCombo.set(comboKey, { fronteras: [], saturados: 0, total: 0, rs: [] });
    const c = porCombo.get(comboKey);
    c.fronteras.push(f.xFrontera);
    c.rs.push(res.rReal);
    c.total += serie.length;
    c.saturados += serie.filter(r => r.saturacion_sensor === 1).length;
  }
  writeCsv(path.join(__dirname, 'experimento_C_barajado_Bprima.csv'), filasC);

  // tabla de saturación + frontera por combinación
  const tablaCombos = [];
  for (const [comboKey, c] of porCombo) {
    const [beta, t_optima, exponente_ptc, potencia_base] = comboKey.split('|').map(Number);
    tablaCombos.push({
      beta, t_optima, exponente_ptc, potencia_base,
      fraccion_saturada: c.saturados / c.total,
      frontera_media: mean(c.fronteras), frontera_desv: std(c.fronteras),
      r_media: mean(c.rs), r_desv: std(c.rs),
    });
  }
  writeCsv(path.join(__dirname, 'Bprima_resumen_por_combinacion.csv'), tablaCombos);

  const comboAltasSaturadas = tablaCombos.filter(c => c.fraccion_saturada > 0.10);

  // frontera media por NIVEL de cada parámetro (promediando sobre los otros factores)
  function porNivel(param) {
    const niveles = new Map();
    for (const c of tablaCombos) {
      const val = c[param];
      if (!niveles.has(val)) niveles.set(val, []);
      niveles.get(val).push(c.frontera_media);
    }
    return [...niveles.entries()].sort((a, b) => a[0] - b[0]).map(([nivel, xs]) => ({ parametro: param, nivel, frontera_media: mean(xs), frontera_desv: std(xs), n_combos: xs.length }));
  }
  const tablaFronteraPorNivel = [...porNivel('beta'), ...porNivel('t_optima'), ...porNivel('exponente_ptc'), ...porNivel('potencia_base')];

  const rs = filasC.map(f => f.r_real);
  const pcts = filasC.map(f => f.percentil_r_real_en_nula);
  const fueraDe95 = pcts.filter(p => p <= 2.5 || p >= 97.5).length;
  const fueraDe99 = pcts.filter(p => p <= 0.5 || p >= 99.5).length;

  // promedios EXCLUYENDO combinaciones con >10% de puntos saturados (el
  // encargo pide descartarlas de los promedios, no de las filas crudas).
  const comboKeysAltasSaturadas = new Set(comboAltasSaturadas.map(c => `${c.beta}|${c.t_optima}|${c.exponente_ptc}|${c.potencia_base}`));
  const filasCLimpio = filasC.filter(f => !comboKeysAltasSaturadas.has(`${f.beta}|${f.t_optima}|${f.exponente_ptc}|${f.potencia_base}`));
  const tablaCombosLimpio = tablaCombos.filter(c => c.fraccion_saturada <= 0.10);
  const rsLimpio = filasCLimpio.map(f => f.r_real);
  const pctsLimpio = filasCLimpio.map(f => f.percentil_r_real_en_nula);

  function porNivelLimpio(param) {
    const niveles = new Map();
    for (const c of tablaCombosLimpio) {
      const val = c[param];
      if (!niveles.has(val)) niveles.set(val, []);
      niveles.get(val).push(c.frontera_media);
    }
    return [...niveles.entries()].sort((a, b) => a[0] - b[0]).map(([nivel, xs]) => ({ parametro: param, nivel, frontera_media: mean(xs), frontera_desv: std(xs), n_combos: xs.length }));
  }
  const tablaFronteraPorNivelLimpio = [...porNivelLimpio('beta'), ...porNivelLimpio('t_optima'), ...porNivelLimpio('exponente_ptc'), ...porNivelLimpio('potencia_base')];

  return {
    tablaFronteraPorNivel, tablaFronteraPorNivelLimpio,
    resumen: {
      n_combos_semilla: filasC.length, n_combos: tablaCombos.length,
      r_media: mean(rs), r_desv: std(rs),
      percentil_media: mean(pcts), fuera_de_95: fueraDe95, fuera_de_99: fueraDe99,
      combos_con_mas_10pct_saturado: comboAltasSaturadas.length,
      combos_con_mas_10pct_saturado_detalle: comboAltasSaturadas,
      SIN_COMBOS_SATURADOS: {
        n_combos_semilla: filasCLimpio.length, n_combos: tablaCombosLimpio.length,
        r_media: mean(rsLimpio), r_desv: std(rsLimpio),
        percentil_media: mean(pctsLimpio),
        fuera_de_95: pctsLimpio.filter(p => p <= 2.5 || p >= 97.5).length,
        fuera_de_99: pctsLimpio.filter(p => p <= 0.5 || p >= 99.5).length,
      },
    },
  };
}

console.log('=== Experimento D ===');
const D = analizarD();
console.log(JSON.stringify(D.resumen, null, 2));

console.log('\n=== Experimento A prima ===');
const A = analizarAprima();
console.log(JSON.stringify(A.resumen, null, 2));

console.log('\n=== Experimento B prima ===');
const B = analizarBprima();
console.log(JSON.stringify({ ...B.resumen, combos_con_mas_10pct_saturado_detalle: undefined }, null, 2));
console.log('\ntabla frontera por nivel (todas las combinaciones):');
console.log(JSON.stringify(B.tablaFronteraPorNivel, null, 2));
console.log('\ntabla frontera por nivel (excluyendo combinaciones >10% saturadas):');
console.log(JSON.stringify(B.tablaFronteraPorNivelLimpio, null, 2));

fs.writeFileSync(path.join(__dirname, 'analisis_v75_completo.json'), JSON.stringify({ D, A, B }, null, 2));
