// Recalibración de κ contra la lluvia DIARIA real (Fase B, 08-ago-2026).
// Misma config que "Experimento Completo" del HTML (parámetros de fábrica,
// Día/Noche+Estaciones on, semilla 'regimen1966-2027', 60 días de
// asentamiento, 1966-2027 completo) pero AHORA con la lluvia diaria/suma
// móvil de 30 días en vez del pico mensual anual congelado -- para ver si
// los κ recién calibrados (contra la lluvia vieja, mucho menos ruidosa)
// siguen sirviendo, y recalibrar con el mismo método (mediana real para
// κ_V/κ_LF/κ_Δ, p90 de e_R para κ_O) si no.
// Correr: node recalibrar_con_lluvia_diaria.js
'use strict';
const fs = require('fs');
const path = require('path');
const motor = require('./motor_fisico.generado.js');

const ZONAS = ['JARDIN_FERTIL', 'CIERRE', 'SELVA_HOSTIL', 'COLAPSO'];
const PARAMETROS_FABRICA = {
  powerBase: 0.47, beta: 0.94, noise: 0.0079, tOpt: 25.0, ptcTc: 16.0,
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
const ticksAsentamiento = DIAS_ASENTAMIENTO * motor.TICKS_POR_DIA;
const ticksTotales = (motor.TOPE_DIA_CALENDARIO + 1) * motor.TICKS_POR_DIA;

const t0 = Date.now();
for (let i = 0; i < ticksAsentamiento; i++) motor.pasoFisica(false);
motor.state.tick = 0; motor.state.step = 0;

const arrLF = new Float64Array(ticksTotales);
const arrDelta = new Float64Array(ticksTotales);
const arrA = new Float64Array(ticksTotales);
const arrErr = new Float64Array(ticksTotales);
const conteoGlobal = { JARDIN_FERTIL: 0, CIERRE: 0, SELVA_HOSTIL: 0, COLAPSO: 0 };
const conteoPorAnio = {};

for (let i = 0; i < ticksTotales; i++) {
  motor.pasoFisica(false);
  const dia = motor.diaCalendarioActual();
  const anio = motor.ANIO_CERO + Math.floor(dia / motor.DIAS_POR_ANIO_CAL);
  const zona = motor.clasificarCierre().zona;
  conteoGlobal[zona]++;
  if (!conteoPorAnio[anio]) conteoPorAnio[anio] = { JARDIN_FERTIL: 0, CIERRE: 0, SELVA_HOSTIL: 0, COLAPSO: 0 };
  conteoPorAnio[anio][zona]++;
  arrLF[i] = motor.state.LF; arrDelta[i] = motor.state.deltaStruct;
  arrA[i] = motor.state.A_sys_env; arrErr[i] = motor.state.err;
  if (i % 150000 === 0) {
    const transcurrido = (Date.now() - t0) / 1000;
    const ritmo = (i + 1) / transcurrido;
    process.stderr.write(`${(i / ticksTotales * 100).toFixed(1)}% · ${Math.round(ritmo)} ticks/s · faltan ~${Math.round((ticksTotales - i) / ritmo / 60)} min\n`);
  }
}
const t1 = Date.now();

const PS = [10, 25, 50, 75, 90, 95, 100];
function percentiles(arr) {
  arr.sort();
  const out = {};
  PS.forEach((p) => { const idx = Math.min(arr.length - 1, Math.floor(p / 100 * arr.length)); out['p' + p] = arr[idx]; });
  return out;
}
const pLF = percentiles(arrLF), pDelta = percentiles(arrDelta), pA = percentiles(arrA), pErr = percentiles(arrErr);

const anios = Object.keys(conteoPorAnio).map(Number).sort((a, b) => a - b);
const header = ['anio', 'JARDIN_FERTIL_pct', 'CIERRE_pct', 'SELVA_HOSTIL_pct', 'COLAPSO_pct', 'dominante'];
const rows = anios.map((a) => {
  const c = conteoPorAnio[a];
  const total = ZONAS.reduce((s, z) => s + c[z], 0);
  const dominante = ZONAS.reduce((mx, z) => (c[z] > c[mx] ? z : mx), ZONAS[0]);
  return [a, ...ZONAS.map((z) => (c[z] / total * 100).toFixed(2)), dominante];
});
const csv = [header.join(','), ...rows.map((r) => r.join(','))].join('\n');
fs.writeFileSync(path.join(__dirname, 'regimen_lluvia_diaria_por_anio.csv'), csv);

const resumen = {
  ticksTotales,
  minutos: ((t1 - t0) / 60000).toFixed(1),
  conteoGlobalPct: Object.fromEntries(ZONAS.map((z) => [z, +(conteoGlobal[z] / ticksTotales * 100).toFixed(2)])),
  percentiles: { LF: pLF, deltaStruct: pDelta, A_sys_env: pA, err: pErr },
  kappaActuales: { V: 0.92, O: 0.0099, LF: 0.069, DELTA: 0.51 },
  kappaPropuestos: { V: +pA.p50.toFixed(4), O: +pErr.p90.toFixed(4), LF: +pLF.p50.toFixed(4), DELTA: +pDelta.p50.toFixed(4) },
};
fs.writeFileSync(path.join(__dirname, 'recalibracion_lluvia_diaria_resumen.json'), JSON.stringify(resumen, null, 2));
console.error(`Listo en ${resumen.minutos} min.`);
console.log(JSON.stringify(resumen, null, 2));
