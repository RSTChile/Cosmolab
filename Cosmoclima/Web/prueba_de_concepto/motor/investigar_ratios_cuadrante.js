// Investigación de los 4 ratios del cuadrante (11-ago-2026), a partir de la
// definición que dio Alexis: "Cierre debería ser lo contrario a Jardín Fértil
// en el sentido de que el sistema se estabilizó sin cambios, punto de
// equilibrio sin cambios".
//
// Esa definición da un TEST NUEVO, mucho más directo que los 4 criterios
// anteriores: en MEGASEQUÍA (2019-2025) el desierto real está quieto
// (semillas latentes esperando lluvia) => debería DOMINAR CIERRE.
// Lo medido va al revés: CIERRE cae de 40% global a 10.7% en megasequía,
// mientras SELVA_HOSTIL (activo y NO viable) sube de 19.7% a 49.4%. O sea
// el instrumento cree que el desierto en sequía está MUY ACTIVO.
//
// clasificarCierre() decide con xRatio=min(ratioLF,ratioDelta) [activación] e
// yRatio=min(ratioA,ratioE) [viabilidad]. Un min() lo decide UNO SOLO de los
// dos términos: este script registra, por año, los 4 ratios y CUÁL manda en
// cada min(), para saber si la "activación" en sequía la produce LF (la
// libertad, ligada a floración real) o Δ_struct (la textura del campo
// espacial, que puede no tener nada que ver con la lluvia).
// No cambia la física: solo observa.
'use strict';
const fs = require('fs');
const path = require('path');
const motor = require('./motor_fisico.generado.js');

const PARAMETROS_FABRICA = {
  powerBase: 0.47, beta: 0.94, noise: 0.0079, tOpt: 13.0, ptcTc: 16.0,
  ptcSharp: 1.0, luminosity: 0.94, umbralGerminacion: 15, rezagoGyriosomus: 30,
};

motor.rngTf = motor.mulberry32(motor.claveSemilla('regimen1966-2027', 'Tf'));
motor.rngEco = motor.mulberry32(motor.claveSemilla('regimen1966-2027', 'eco'));
Object.assign(motor.state, PARAMETROS_FABRICA, { dayNightMode: true, seasonMode: true });
motor.state.tick = 0; motor.state.step = 0;
motor.state.Tf = 24.6; motor.state.Tc = 25; motor.state.Th = 28;
motor.state.floracion = 0; motor.state.gyriosomus = 0; motor.state.sueloDesnudo = 1;
motor.state.floracionHistorial = [];
motor.state.powerLive = motor.state.powerBase; motor.state._A_prev = 0;
motor.resetField(); motor.resetBuffers();

const DIAS_ASENTAMIENTO = 60;
for (let i = 0; i < DIAS_ASENTAMIENTO * motor.TICKS_POR_DIA; i++) motor.pasoFisica(false);
motor.state.tick = 0; motor.state.step = 0;

const kLF = motor.KAPPA_LF_POR_SEMANA, kD = motor.KAPPA_DELTA_POR_SEMANA;
const kV = motor.KAPPA_V_POR_SEMANA, kO = motor.KAPPA_O_POR_SEMANA;
const semanaDe = (dia) => Math.floor((dia % motor.DIAS_POR_ANIO_CAL) / 7);

const ticksTotales = (motor.TOPE_DIA_CALENDARIO + 1) * motor.TICKS_POR_DIA;
const porAnio = {};
const t0 = Date.now();
for (let i = 0; i < ticksTotales; i++) {
  motor.pasoFisica(false);
  const dia = motor.diaCalendarioActual();
  const s = semanaDe(dia);
  const a = motor.fechaDesdeDiaCalendario(dia).anio;
  if (!porAnio[a]) porAnio[a] = { n: 0, rLF: 0, rD: 0, rA: 0, rE: 0, mandaLF: 0, mandaD: 0, mandaA: 0, mandaE: 0, LF: 0, dStruct: 0, floracion: 0 };
  const o = porAnio[a];
  const rLF = motor.state.LF / kLF[s], rD = motor.state.deltaStruct / kD[s];
  const rA = motor.state.A_sys_env / kV[s], rE = 2 - (motor.state.err / kO[s]);
  o.n++;
  o.rLF += rLF; o.rD += rD; o.rA += rA; o.rE += rE;
  if (rLF <= rD) o.mandaLF++; else o.mandaD++;
  if (rA <= rE) o.mandaA++; else o.mandaE++;
  o.LF += motor.state.LF; o.dStruct += motor.state.deltaStruct; o.floracion += motor.state.floracion;
  if (i % 300000 === 0) process.stderr.write(`${(i / ticksTotales * 100).toFixed(1)}% · ${((Date.now() - t0) / 1000).toFixed(0)}s\n`);
}

const FLOR = [1983, 1991, 1997, 2000, 2002, 2005, 2011, 2012, 2015, 2017, 2021, 2022, 2024];
const SEQ = [2019, 2020, 2021, 2022, 2023, 2024, 2025];
const anios = Object.keys(porAnio).map(Number).sort((a, b) => a - b);
const media = (a, c) => porAnio[a][c] / porAnio[a].n;
const prom = (lista, c) => { const v = lista.filter((a) => porAnio[a]); return v.reduce((s, a) => s + media(a, c), 0) / v.length; };
const global = (c) => anios.reduce((s, a) => s + media(a, c), 0) / anios.length;

console.log(`\nRatios promedio (>=1 significa que esa condición SE CUMPLE):\n`);
console.log(`  ${'ratio'.padEnd(12)} ${'global'.padStart(8)} ${'megasequía'.padStart(11)} ${'floración'.padStart(10)}`);
for (const c of ['rLF', 'rD', 'rA', 'rE']) {
  console.log(`  ${c.padEnd(12)} ${global(c).toFixed(3).padStart(8)} ${prom(SEQ, c).toFixed(3).padStart(11)} ${prom(FLOR, c).toFixed(3).padStart(10)}`);
}
console.log(`\n¿Quién MANDA en cada min()? (fracción de ticks en que ese término es el menor)\n`);
console.log(`  ${'quién'.padEnd(12)} ${'global'.padStart(8)} ${'megasequía'.padStart(11)} ${'floración'.padStart(10)}`);
for (const c of ['mandaLF', 'mandaD', 'mandaA', 'mandaE']) {
  const g = anios.reduce((s, a) => s + porAnio[a][c] / porAnio[a].n, 0) / anios.length;
  const ms = SEQ.filter((a) => porAnio[a]).reduce((s, a) => s + porAnio[a][c] / porAnio[a].n, 0) / SEQ.filter((a) => porAnio[a]).length;
  const fl = FLOR.filter((a) => porAnio[a]).reduce((s, a) => s + porAnio[a][c] / porAnio[a].n, 0) / FLOR.filter((a) => porAnio[a]).length;
  console.log(`  ${c.padEnd(12)} ${(g * 100).toFixed(1).padStart(7)}% ${(ms * 100).toFixed(1).padStart(10)}% ${(fl * 100).toFixed(1).padStart(9)}%`);
}
console.log(`\nCrudos:\n`);
for (const c of ['LF', 'dStruct', 'floracion']) {
  console.log(`  ${c.padEnd(12)} ${global(c).toFixed(4).padStart(8)} ${prom(SEQ, c).toFixed(4).padStart(11)} ${prom(FLOR, c).toFixed(4).padStart(10)}`);
}

fs.writeFileSync(path.join(__dirname, 'investigacion_ratios_cuadrante.json'), JSON.stringify(porAnio, null, 2));
console.log(`\nEscrito investigacion_ratios_cuadrante.json`);
