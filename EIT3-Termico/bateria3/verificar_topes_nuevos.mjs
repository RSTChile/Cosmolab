import { correrBarridoV76 } from './correr_barrido_v76.mjs';
const PARAMS = { powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:25, ptcTc:18, ptcSharp:4.1, minTemp:-6, maxTemp:25 };
const BARRIDO = { desde:0.93, hasta:1.4, puntos:2, settle:40, measure:30 };

function correr(seed, modo) {
  return correrBarridoV76({ seed, modo, from:BARRIDO.desde, to:BARRIDO.hasta, steps:BARRIDO.puntos, settle:BARRIDO.settle, measure:BARRIDO.measure, params:PARAMS });
}

console.log('esperado (con topes viejos, de validacion3.md):');
console.log('  seed=7 parada: x=0.93 asent_pasos=2900 rec_mediana=948 | x=1.4 asent_pasos=750 rec_mediana=201');
console.log('  seed=7 inicio: x=0.93 asent_pasos=2750 rec_mediana=944 | x=1.4 asent_pasos=550 rec_mediana=203');
console.log('  seed=17 parada: x=0.93 asent_pasos=2900 rec_mediana=948 | x=1.4 asent_pasos=750 rec_mediana=201');
console.log('\ncon topes nuevos (TOPE_EQ=6000, TOPE_REC=3000):');
for (const [seed,modo] of [[7,'parada'],[7,'inicio'],[17,'parada']]) {
  const rows = correr(seed, modo);
  console.log(`  seed=${seed} ${modo}: ` + rows.map(r=>`x=${r.luminosidad.toFixed(2)} asent_pasos=${r.asent_pasos} rec_mediana=${r.rec_mediana}`).join(' | '));
}
