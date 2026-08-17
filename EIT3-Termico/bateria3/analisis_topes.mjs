import fs from 'node:fs';
import path from 'node:path';

const NOMBRES = ['baseline','tOpt22','tOpt28','ptcSharp3','ptcSharp6','beta080','beta098','extremo_combo'];
const combos = NOMBRES.map(n => JSON.parse(fs.readFileSync(path.join('muestras_topes', `${n}.json`), 'utf8')));

// ── ASENTAMIENTO (TOPE_EQ) ──
let todosAsentPasos = [];
let asentNoOk = 0, asentTotal = 0;
for (const c of combos) {
  for (const r of c.rows) {
    todosAsentPasos.push({ combo: c.params, x: r.luminosidad, pasos: r.asent_pasos, ok: r.asent_ok });
    asentTotal++;
    if (r.asent_ok === 0) asentNoOk++;
  }
}
console.log(`=== ASENTAMIENTO (asentarHastaEquilibrio), ${asentTotal} puntos en 8 combinaciones ===`);
console.log(`asent_ok=0 (no asentó ni con tope=20000) actual: ${asentNoOk}/${asentTotal}`);
const pasosOk = todosAsentPasos.filter(a => a.ok === 1).map(a => a.pasos);
console.log(`asent_pasos (solo los que SÍ asentaron): min=${Math.min(...pasosOk)} max=${Math.max(...pasosOk)} media=${(pasosOk.reduce((a,b)=>a+b,0)/pasosOk.length).toFixed(0)}`);
// percentiles
const ordenado = pasosOk.slice().sort((a,b)=>a-b);
function pct(p){ return ordenado[Math.floor(ordenado.length*p)]; }
console.log(`percentiles: p50=${pct(0.5)} p90=${pct(0.9)} p95=${pct(0.95)} p99=${pct(0.99)} max=${ordenado[ordenado.length-1]}`);

console.log('\ncandidato_TOPE_EQ | reclasificados (lento-pero-convergente > tope) | ya fallaban a 20000 (sin cambio)');
for (const T of [3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000]) {
  const reclas = todosAsentPasos.filter(a => a.ok === 1 && a.pasos > T).length;
  console.log(`${T} | ${reclas}/${asentTotal} (${(100*reclas/asentTotal).toFixed(1)}%) | ${asentNoOk}`);
}

// detalle por combinación de cuál tiene los peores asent_pasos
console.log('\nmax asent_pasos (asentado=1) por combinación:');
for (const c of combos) {
  const ok = c.rows.filter(r=>r.asent_ok===1).map(r=>r.asent_pasos);
  const noOk = c.rows.filter(r=>r.asent_ok===0).length;
  console.log(`  ${JSON.stringify({tOpt:c.params.tOpt,ptcSharp:c.params.ptcSharp,beta:c.params.beta,powerBase:c.params.powerBase})}: max=${ok.length?Math.max(...ok):'-'} asent_no_ok=${noOk}/60 (${c.seg.toFixed(0)}s)`);
}

// ── RECUPERACIÓN (TOPE_REC) ──
console.log('\n\n=== RECUPERACIÓN (medirRecuperacion), reps individuales ===');
let todosReps = [];
for (const c of combos) {
  for (const r of c.rows) {
    for (const rep of [r.rec_1, r.rec_2, r.rec_3, r.rec_4, r.rec_5]) {
      todosReps.push({ combo: c.params, x: r.luminosidad, rep });
    }
  }
}
const totalReps = todosReps.length;
const topeados = todosReps.filter(r => r.rep >= 20000).length;
console.log(`reps totales: ${totalReps} (${combos.length} combos × 60 puntos × 5 reps)`);
console.log(`reps que YA tocan el tope actual (=20000): ${topeados}/${totalReps} (${(100*topeados/totalReps).toFixed(1)}%) — esto es dato válido de bifurcación, no error`);
const repsNoTope = todosReps.filter(r => r.rep < 20000).map(r => r.rep);
const repsOrdenado = repsNoTope.slice().sort((a,b)=>a-b);
function pctR(p){ return repsOrdenado[Math.floor(repsOrdenado.length*p)]; }
console.log(`reps que SÍ convergen (<20000): min=${Math.min(...repsNoTope)} max=${Math.max(...repsNoTope)} media=${(repsNoTope.reduce((a,b)=>a+b,0)/repsNoTope.length).toFixed(0)}`);
console.log(`percentiles (solo convergentes): p50=${pctR(0.5)} p90=${pctR(0.9)} p95=${pctR(0.95)} p99=${pctR(0.99)} max=${repsOrdenado[repsOrdenado.length-1]}`);

console.log('\ncandidato_TOPE_REC | reclasificados (convergente genuino > tope, se volvería "topó" falsamente) | ya topaban a 20000 (sin cambio, dato válido)');
for (const T of [3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000]) {
  const reclas = todosReps.filter(r => r.rep < 20000 && r.rep > T).length;
  console.log(`${T} | ${reclas}/${totalReps} (${(100*reclas/totalReps).toFixed(2)}%) | ${topeados} (${(100*topeados/totalReps).toFixed(1)}%)`);
}

fs.writeFileSync('analisis_topes_resultado.json', JSON.stringify({ asentTotal, asentNoOk, pasosOk: ordenado, totalReps, topeados, repsNoTope: repsOrdenado }, null, 2));

console.log('\n\n=== resolución fina ===');
console.log('TOPE_EQ candidato | reclasificados');
for (const T of [1500,2000,2500,3000,3500,4000,4500,5000]) {
  const reclas = todosAsentPasos.filter(a => a.ok === 1 && a.pasos > T).length;
  console.log(`${T} | ${reclas}/${asentTotal}`);
}
console.log('TOPE_REC candidato | reclasificados');
for (const T of [1500,1800,2000,2101,2200,2500,3000]) {
  const reclas = todosReps.filter(r => r.rep < 20000 && r.rep > T).length;
  console.log(`${T} | ${reclas}/${totalReps}`);
}

console.log('\n\n=== impacto en pasos totales (muestra de 8 combos x 60 puntos) ===');
const T_EQ_NUEVO = 6000, T_REC_NUEVO = 3000;

let pasosEqActual=0, pasosEqNuevo=0;
for (const a of todosAsentPasos) {
  pasosEqActual += a.pasos; // ya es min(pasos,20000) por construcción del motor
  pasosEqNuevo += Math.min(a.pasos, T_EQ_NUEVO); // ninguno excede 6000 en la muestra, así que == a.pasos
}
let pasosRecActual=0, pasosRecNuevo=0;
for (const r of todosReps) {
  pasosRecActual += r.rep;
  pasosRecNuevo += Math.min(r.rep, T_REC_NUEVO);
}
console.log(`asentamiento: actual=${pasosEqActual} pasos, con TOPE_EQ=${T_EQ_NUEVO} -> ${pasosEqNuevo} pasos (${(100*pasosEqNuevo/pasosEqActual).toFixed(1)}% del actual)`);
console.log(`recuperación: actual=${pasosRecActual} pasos, con TOPE_REC=${T_REC_NUEVO} -> ${pasosRecNuevo} pasos (${(100*pasosRecNuevo/pasosRecActual).toFixed(1)}% del actual)`);

const pasosTotalActual = pasosEqActual + pasosRecActual;
const pasosTotalNuevo = pasosEqNuevo + pasosRecNuevo;
console.log(`combinado (solo asent+recuperación, sin contar settle/measure/calibración): ${(100*pasosTotalNuevo/pasosTotalActual).toFixed(1)}% del costo actual`);

// tiempo real medido en la muestra (8 combos) vs estimado con topes nuevos,
// asumiendo que settle+measure+calibración (no afectados por los topes) son
// una fracción fija del tiempo total, estimada del baseline sin throttling
// (630s con TOPE_EQ/TOPE_REC actuales, de la ronda anterior).
const segActualMuestra = combos.reduce((a,c)=>a+c.seg,0);
console.log(`\ntiempo real medido en esta muestra (8 barridos, con throttling): ${segActualMuestra.toFixed(0)}s total`);
