// Investigación de A_sys_env (11-ago-2026, a pedido de Alexis: "Sí,
// investiga A_sys_env"). Tras adoptar el reemplazo de PTC (modo 'B'), los
// criterios (c) megasequía y (d) correlación con lluvia real SIGUEN
// fallando en ambas opciones y con el mismo signo -- lo que apunta al otro
// lado del clasificador: la VIABILIDAD (A_sys_env/e_R), que quedó
// anticorrelacionada con la lluvia real desde antes de esta ronda.
//
// A_sys_env hoy = computeCoupling(Tf, ref) = max(0, 1 - |Tf-ref|/8), con
// ref = lerp(envTemp, targetTf, 0.5), promediado por semana real.
// Es decir: mide QUÉ TAN BIEN LA TEMPERATURA DEL PLANETA SIGUE A UNA
// REFERENCIA TÉRMICA -- un error de seguimiento de termostato.
//
// HIPÓTESIS a falsar: el propio término de regulación interglacial
// (targetTf baja 9,4 °C por unidad de floración) hace que en años de
// floración la referencia se aleje MUCHO de la temperatura real del
// entorno, que casi no baja. Si Tf se queda en el medio, |Tf-ref| CRECE
// justo en los años lluviosos => A_sys_env baja cuando llueve => signo
// invertido. Se prueba midiendo, en la MISMA corrida, 4 acoplamientos
// alternativos y correlacionando cada uno con la lluvia real independiente:
//   actual         -- ref = lerp(envTemp, targetTf, 0.5)          [el que se usa]
//   soloEnv        -- ref = envTemp                                [entorno real puro]
//   sinRegulacion  -- ref = lerp(envTemp, targetTfSinRegulacion, 0.5)
//   soloTarget     -- ref = targetTf
// No cambia nada de la física: solo observa.
'use strict';
const fs = require('fs');
const path = require('path');
const motor = require('./motor_fisico.generado.js');

const DB_LLUVIA = path.join(__dirname, '..', '..', '..', 'investigacion', 'fuentes', 'pluviosidad_diaria_consolidada.sqlite');

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

const acopl = (Tf, ref) => Math.max(0, 1 - Math.abs(Tf - ref) / 8.0);
const lerp = (a, b, t) => a + (b - a) * t;

const DIAS_ASENTAMIENTO = 60;
for (let i = 0; i < DIAS_ASENTAMIENTO * motor.TICKS_POR_DIA; i++) motor.pasoFisica(false);
motor.state.tick = 0; motor.state.step = 0;

const ticksTotales = (motor.TOPE_DIA_CALENDARIO + 1) * motor.TICKS_POR_DIA;
const porAnio = {}; // anio -> acumuladores
const t0 = Date.now();
for (let i = 0; i < ticksTotales; i++) {
  motor.pasoFisica(false);
  const f = motor.fechaDesdeDiaCalendario(motor.diaCalendarioActual());
  const a = f.anio;
  if (!porAnio[a]) porAnio[a] = { n: 0, A_pub: 0, err_pub: 0, floracion: 0, Tf: 0, envTemp: 0, targetTf: 0, actual: 0, soloEnv: 0, sinReg: 0, soloTarget: 0, gapActual: 0 };
  const o = porAnio[a];
  const Tf = motor.state.Tf, envTemp = motor.state.envTemp;
  const targetTf = motor.state.targetTf, targetSinReg = motor.state.targetTfSinRegulacion;
  const refActual = lerp(envTemp, targetTf, 0.5);
  o.n++;
  o.A_pub += motor.state.A_sys_env;
  o.err_pub += motor.state.err;
  o.floracion += motor.state.floracion;
  o.Tf += Tf; o.envTemp += envTemp; o.targetTf += targetTf;
  o.actual += acopl(Tf, refActual);
  o.soloEnv += acopl(Tf, envTemp);
  o.sinReg += acopl(Tf, lerp(envTemp, targetSinReg, 0.5));
  o.soloTarget += acopl(Tf, targetTf);
  o.gapActual += Math.abs(Tf - refActual);
  if (i % 300000 === 0) process.stderr.write(`${(i / ticksTotales * 100).toFixed(1)}% · ${((Date.now() - t0) / 1000).toFixed(0)}s\n`);
}

// --- lluvia real independiente (misma consulta que evaluar_contra_ground_truth.js) ---
const { DatabaseSync } = require('node:sqlite');
const db = new DatabaseSync(DB_LLUVIA, { readOnly: true });
const filas = db.prepare(`
  SELECT substr(fecha,1,4) AS anio, COUNT(DISTINCT localidad) AS n_localidades, SUM(lluvia_mm) AS suma_total
  FROM pluviosidad_diaria WHERE substr(fecha,1,4) BETWEEN '1966' AND '2027' GROUP BY anio ORDER BY anio;`).all();
db.close();
const lluviaPorAnio = new Map();
for (const f of filas) {
  const anio = parseInt(f.anio, 10);
  if (!Number.isFinite(anio) || !f.n_localidades) continue;
  lluviaPorAnio.set(anio, f.suma_total / f.n_localidades);
}

function rangos(v) {
  const idx = v.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]);
  const r = new Array(v.length);
  let i = 0;
  while (i < idx.length) {
    let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    const promedio = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) r[idx[k][1]] = promedio;
    i = j + 1;
  }
  return r;
}
function pearson(x, y) {
  const n = x.length, mx = x.reduce((s, v) => s + v, 0) / n, my = y.reduce((s, v) => s + v, 0) / n;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) { const a = x[i] - mx, b = y[i] - my; num += a * b; dx += a * a; dy += b * b; }
  return num / Math.sqrt(dx * dy);
}
const spearman = (x, y) => (x.length < 2 ? null : pearson(rangos(x), rangos(y)));

const anios = Object.keys(porAnio).map(Number).sort((a, b) => a - b).filter((a) => lluviaPorAnio.has(a));
const lluvia = anios.map((a) => lluviaPorAnio.get(a));
const media = (a, campo) => porAnio[a][campo] / porAnio[a].n;

const VARIANTES = ['A_pub', 'actual', 'soloEnv', 'sinReg', 'soloTarget', 'err_pub', 'floracion', 'Tf', 'envTemp', 'targetTf', 'gapActual'];
console.log(`\nCorrelación de Spearman contra la lluvia real anual (n=${anios.length} años):\n`);
const resultados = {};
for (const v of VARIANTES) {
  const serie = anios.map((a) => media(a, v));
  const rho = spearman(serie, lluvia);
  resultados[v] = rho;
  console.log(`  ${v.padEnd(16)} rho = ${rho >= 0 ? ' ' : ''}${rho.toFixed(4)}`);
}

// Bloom (13 documentados) vs control (10) -- mismos años del evaluador
const FLOR = [1983, 1991, 1997, 2000, 2002, 2005, 2011, 2012, 2015, 2017, 2021, 2022, 2024];
const CTRL = [1989, 1990, 1996, 2003, 2008, 2013, 2016, 2018, 2019, 2020];
console.log(`\nMedia por grupo (floración documentada vs control):\n`);
console.log(`  ${'variable'.padEnd(16)} ${'floración'.padStart(10)} ${'control'.padStart(10)}   dirección`);
for (const v of VARIANTES) {
  const mf = FLOR.filter((a) => porAnio[a]).reduce((s, a) => s + media(a, v), 0) / FLOR.filter((a) => porAnio[a]).length;
  const mc = CTRL.filter((a) => porAnio[a]).reduce((s, a) => s + media(a, v), 0) / CTRL.filter((a) => porAnio[a]).length;
  console.log(`  ${v.padEnd(16)} ${mf.toFixed(4).padStart(10)} ${mc.toFixed(4).padStart(10)}   ${mf > mc ? 'floración MAYOR' : 'floración menor'}`);
}

fs.writeFileSync(path.join(__dirname, 'investigacion_asysenv.json'), JSON.stringify({ rhoPorVariante: resultados, porAnio }, null, 2));
console.log(`\nEscrito investigacion_asysenv.json`);
