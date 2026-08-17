// Corrida completa 1966-2027 (parámetros de fábrica, misma semilla que
// "Experimento Completo" del HTML) que junta en UNA sola pasada lo que antes
// eran dos scripts separados: el CSV por año (verificar_experimento_completo.js)
// Y los percentiles reales de LF/Δ_struct/A_sys_env/e_R que hacen falta para
// recalibrar κ_LF/κ_Δ tras el Nivel 1 (activación mensual real, 10-ago-2026).
// Además junta percentiles de LF/Δ_struct EN 53 BALDES POR SEMANA-CALENDARIO
// (no solo global) -- es la materia prima del Nivel 3 (anomalía por semana,
// 10-ago-2026 ronda 4, a pedido de Alexis: "subamos la resolución a nivel
// semanal"), para no tener que correr la simulación completa una vez más
// solo para recalibrar. Correr: node calcular_percentiles_y_regimen.js [sufijo]
'use strict';
const fs = require('fs');
const path = require('path');
const motor = require('./motor_fisico.generado.js');

const sufijo = process.argv[2] || 'nivel1';
// Ronda 12 (11-ago-2026): 3er argumento opcional selecciona el modo de
// reemplazo de PTC ('A' velocidad de transición, 'B' amplitud de
// alternativas) -- ver PTC_REEMPLAZO_MODO en el HTML/generador.
const modoPTC = process.argv[3];
if (modoPTC) motor.PTC_REEMPLAZO_MODO = modoPTC;
// Ronda 13 (11-ago-2026): banderas de las dos correcciones por variable de
// entorno, para poder medir C1 y C2 por separado sin regenerar el motor.
if (process.env.LF_SIN_V !== undefined) motor.LF_SIN_V = process.env.LF_SIN_V === 'true';
if (process.env.KAPPA_CANONICOS !== undefined) motor.KAPPA_CANONICOS = process.env.KAPPA_CANONICOS === 'true';
if (process.env.KAPPA_LF_INFIMO) motor.KAPPA_LF_INFIMO = Number(process.env.KAPPA_LF_INFIMO);
if (process.env.KAPPA_DELTA_INFIMO) motor.KAPPA_DELTA_INFIMO = Number(process.env.KAPPA_DELTA_INFIMO);
console.error(`config: PTC=${motor.PTC_REEMPLAZO_MODO} LF_SIN_V=${motor.LF_SIN_V} KAPPA_CANONICOS=${motor.KAPPA_CANONICOS}`);
const ZONAS = ['JARDIN_FERTIL', 'CIERRE', 'SELVA_HOSTIL', 'COLAPSO'];
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
const ticksAsentamiento = DIAS_ASENTAMIENTO * motor.TICKS_POR_DIA;
const ticksTotales = (motor.TOPE_DIA_CALENDARIO + 1) * motor.TICKS_POR_DIA;

const t0 = Date.now();
for (let i = 0; i < ticksAsentamiento; i++) motor.pasoFisica(false);

motor.state.tick = 0; motor.state.step = 0;

const conteoPorAnio = {};
// Ronda 6 (10-ago-2026): cobertura de temperatura real por año -- la
// temperatura real (TEMPERATURA_DIARIA_ZHCS) solo cubre 1981-2026-08, así
// que ~24% de los 62 años (1966-1980 y la cola 2026-08+) siguen con el
// vaivén sintético de siempre. Se cuenta acá para poder separar el
// resultado por cobertura real vs sintética en la evaluación, no solo
// reportar un agregado que mezcla ambos.
const ticksConTempRealPorAnio = {};
const ticksTotalPorAnio = {};
const arrLF = new Float64Array(ticksTotales);
const arrDelta = new Float64Array(ticksTotales);
const arrA = new Float64Array(ticksTotales);
const arrErr = new Float64Array(ticksTotales);
// 53 baldes por semana-calendario (0-52, semana 52 es corta: 1 día) --
// Nivel 3. Tamaño dinámico (push), no Float64Array fijo, porque no sabemos
// de antemano cuántos ticks caen en cada semana exacta.
const porSemanaLF = Array.from({ length: 53 }, () => []);
const porSemanaDelta = Array.from({ length: 53 }, () => []);
// Ronda 5 (10-ago-2026): también A_sys_env/err por semana -- viabilidad
// sube a semanal, hace falta recalibrar κ_V/κ_O por semana igual que κ_LF/κ_Δ.
const porSemanaA = Array.from({ length: 53 }, () => []);
const porSemanaErr = Array.from({ length: 53 }, () => []);

for (let i = 0; i < ticksTotales; i++) {
  motor.pasoFisica(false);
  const dia = motor.diaCalendarioActual();
  const anio = motor.ANIO_CERO + Math.floor(dia / motor.DIAS_POR_ANIO_CAL);
  const zona = motor.clasificarCierre().zona;
  if (!conteoPorAnio[anio]) conteoPorAnio[anio] = { JARDIN_FERTIL: 0, CIERRE: 0, SELVA_HOSTIL: 0, COLAPSO: 0 };
  conteoPorAnio[anio][zona]++;
  ticksTotalPorAnio[anio] = (ticksTotalPorAnio[anio] || 0) + 1;
  if (motor.state.coberturaTempReal) ticksConTempRealPorAnio[anio] = (ticksConTempRealPorAnio[anio] || 0) + 1;
  arrLF[i] = motor.state.LF; arrDelta[i] = motor.state.deltaStruct;
  arrA[i] = motor.state.A_sys_env; arrErr[i] = motor.state.err;
  const semana = Math.floor((dia % motor.DIAS_POR_ANIO_CAL) / 7); // misma fórmula que semanaDelAnioReal() del HTML
  porSemanaLF[semana].push(motor.state.LF); porSemanaDelta[semana].push(motor.state.deltaStruct);
  porSemanaA[semana].push(motor.state.A_sys_env); porSemanaErr[semana].push(motor.state.err);
  if (i % 200000 === 0) {
    const transcurrido = (Date.now() - t0) / 1000;
    const ritmo = (i + 1) / transcurrido;
    process.stderr.write(`${(i / ticksTotales * 100).toFixed(1)}% · ${Math.round(ritmo)} ticks/s · faltan ~${Math.round((ticksTotales - i) / ritmo / 60)} min\n`);
  }
}
const t1 = Date.now();

const PS = [10, 25, 50, 75, 90, 95, 100];
function calcPercentiles(arr) {
  const a = Array.from(arr).sort((x, y) => x - y);
  const out = {};
  for (const p of PS) out['p' + p] = a[Math.min(a.length - 1, Math.floor(p / 100 * a.length))];
  return out;
}
// Ronda 8 (10-ago-2026, a pedido de Alexis: "calibremos fino... ¿cuál es la
// varianza? porque quizá lo que haya que hacer es ponderar contra eso").
// La mediana separa cada semana ~50/50 casi por definición -- con Jardín
// Fértil exigiendo 2 condiciones a la vez, eso da ~20-25% de "activo"/
// "viable" por puro azar, sin que la lluvia real tenga que intervenir.
// Media+desviación estándar por semana permite exigir que el año esté
// GENUINAMENTE por encima de lo típico esa semana (no solo del lado de
// arriba de la mitad), ponderado por cuánto varía esa semana en realidad.
function calcMediaStd(arr) {
  const n = arr.length;
  const media = arr.reduce((s, v) => s + v, 0) / n;
  const varianza = arr.reduce((s, v) => s + (v - media) * (v - media), 0) / n;
  return { media, std: Math.sqrt(varianza), n };
}

const percentilesGlobal = { LF: calcPercentiles(arrLF), deltaStruct: calcPercentiles(arrDelta), A_sys_env: calcPercentiles(arrA), err: calcPercentiles(arrErr) };
const percentilesPorSemana = porSemanaLF.map((_, semana) => ({
  semana, LF: calcPercentiles(porSemanaLF[semana]), deltaStruct: calcPercentiles(porSemanaDelta[semana]),
  A_sys_env: calcPercentiles(porSemanaA[semana]), err: calcPercentiles(porSemanaErr[semana]),
  n: porSemanaLF[semana].length,
}));
const mediaStdPorSemana = porSemanaLF.map((_, semana) => ({
  semana,
  LF: calcMediaStd(porSemanaLF[semana]),
  deltaStruct: calcMediaStd(porSemanaDelta[semana]),
  A_sys_env: calcMediaStd(porSemanaA[semana]),
}));

const anios = Object.keys(conteoPorAnio).map(Number).sort((a, b) => a - b);
const header = ['anio', 'JARDIN_FERTIL_pct', 'CIERRE_pct', 'SELVA_HOSTIL_pct', 'COLAPSO_pct', 'dominante', 'coberturaTempReal_pct'];
const rows = anios.map((a) => {
  const c = conteoPorAnio[a];
  const total = ZONAS.reduce((s, z) => s + c[z], 0);
  const dominante = ZONAS.reduce((mx, z) => (c[z] > c[mx] ? z : mx), ZONAS[0]);
  const coberturaPct = ((ticksConTempRealPorAnio[a] || 0) / ticksTotalPorAnio[a] * 100).toFixed(1);
  return [a, ...ZONAS.map((z) => (c[z] / total * 100).toFixed(2)), dominante, coberturaPct];
});
const csv = [header.join(','), ...rows.map((r) => r.join(','))].join('\n');
fs.writeFileSync(path.join(__dirname, `regimen_${sufijo}_por_anio.csv`), csv);
fs.writeFileSync(path.join(__dirname, `regimen_${sufijo}_percentiles.json`), JSON.stringify({ percentilesGlobal, percentilesPorSemana, mediaStdPorSemana, kappaUsados: { KAPPA_V: motor.KAPPA_V, KAPPA_O: motor.KAPPA_O, KAPPA_LF: motor.KAPPA_LF, KAPPA_DELTA: motor.KAPPA_DELTA } }, null, 2));

console.error(`Listo en ${((t1 - t0) / 60000).toFixed(1)} min. Escrito regimen_${sufijo}_por_anio.csv y regimen_${sufijo}_percentiles.json`);
