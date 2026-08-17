// Proceso hijo de una corrida de batería (Fase A.4 del plan de
// granularidad, 08-ago-2026). Un proceso de sistema operativo por
// configuración (no worker_threads): si una config hace explotar la
// física, no tumba la batería entera, y es inspeccionable en el Monitor
// de Actividad sin saber de threads. Recibe la config por argv[2] (JSON),
// corre el mismo asentamiento de 60 días + rango completo 1966-2027 que
// "Experimento Completo" del HTML, y escribe resultados/<id>.json +
// resultados/<id>_por_anio.csv (mismo esquema que
// descargarCSVRegimenPorAnio del HTML, para que sea diffable).
//
// NOTA (08-ago-2026): resolucionLluvia todavía solo soporta "mensual" --
// la Fase B (lluvia diaria real 1966-2017) agrega la otra opción acá
// mismo, en el bloque marcado abajo, sin tocar el resto de este archivo.
'use strict';
const fs = require('fs');
const path = require('path');
const motor = require('../motor_fisico.generado.js');

const config = JSON.parse(process.argv[2]);
const { id, overrides = {}, kappa = {}, semilla = id } = config;

if (kappa.V !== undefined) motor.KAPPA_V = kappa.V;
if (kappa.O !== undefined) motor.KAPPA_O = kappa.O;
if (kappa.LF !== undefined) motor.KAPPA_LF = kappa.LF;
if (kappa.DELTA !== undefined) motor.KAPPA_DELTA = kappa.DELTA;
// Ronda 5 (10-ago-2026): clasificarCierre() ya NO lee KAPPA_V/KAPPA_O/
// KAPPA_LF/KAPPA_DELTA (los escalares de arriba) -- lee las 4 versiones
// POR_SEMANA (53 valores cada una) -- ver comentario junto a
// clasificarCierre() en el HTML. kappa.V/O/LF/DELTA (escalares) quedan por
// compatibilidad -- si vienen, se difunden a las 53 semanas por igual
// (mismo efecto que un κ global). kappa.*_POR_SEMANA (arrays de 53) los
// pisan si vienen, para baterías genuinas de κ variable por semana.
if (kappa.V !== undefined) motor.KAPPA_V_POR_SEMANA = Array(53).fill(kappa.V);
if (kappa.O !== undefined) motor.KAPPA_O_POR_SEMANA = Array(53).fill(kappa.O);
if (kappa.LF !== undefined) motor.KAPPA_LF_POR_SEMANA = Array(53).fill(kappa.LF);
if (kappa.DELTA !== undefined) motor.KAPPA_DELTA_POR_SEMANA = Array(53).fill(kappa.DELTA);
if (kappa.V_POR_SEMANA !== undefined) motor.KAPPA_V_POR_SEMANA = kappa.V_POR_SEMANA;
if (kappa.O_POR_SEMANA !== undefined) motor.KAPPA_O_POR_SEMANA = kappa.O_POR_SEMANA;
if (kappa.LF_POR_SEMANA !== undefined) motor.KAPPA_LF_POR_SEMANA = kappa.LF_POR_SEMANA;
if (kappa.DELTA_POR_SEMANA !== undefined) motor.KAPPA_DELTA_POR_SEMANA = kappa.DELTA_POR_SEMANA;

motor.rngTf = motor.mulberry32(motor.claveSemilla(semilla, 'Tf'));
motor.rngEco = motor.mulberry32(motor.claveSemilla(semilla, 'eco'));
Object.assign(motor.state, {
  powerBase: 0.47, beta: 0.94, noise: 0.0079, tOpt: 13.0, ptcTc: 16.0,
  ptcSharp: 1.0, luminosity: 0.94, umbralGerminacion: 15, rezagoGyriosomus: 30,
  dayNightMode: true, seasonMode: true,
}, overrides);
motor.state.tick = 0; motor.state.step = 0;
motor.state.Tf = 24.6; motor.state.Tc = 25; motor.state.Th = 28;
motor.state.floracion = 0; motor.state.gyriosomus = 0; motor.state.sueloDesnudo = 1;
motor.state.floracionHistorial = [];
motor.state.powerLive = motor.state.powerBase; motor.state._A_prev = 0;
motor.resetField(); motor.resetBuffers();

// --- BLOQUE resolucionLluvia (Fase B lo extiende acá) ---
if (config.resolucionLluvia && config.resolucionLluvia !== 'mensual') {
  console.error(`[${id}] resolucionLluvia=${config.resolucionLluvia} todavía no está implementada (pendiente Fase B) -- corriendo con mensual.`);
}
// --- fin bloque resolucionLluvia ---

const ZONAS = ['JARDIN_FERTIL', 'CIERRE', 'SELVA_HOSTIL', 'COLAPSO'];
const DIAS_ASENTAMIENTO = 60;
const ticksAsentamiento = DIAS_ASENTAMIENTO * motor.TICKS_POR_DIA;
const ticksTotales = (motor.TOPE_DIA_CALENDARIO + 1) * motor.TICKS_POR_DIA;

for (let i = 0; i < ticksAsentamiento; i++) motor.pasoFisica(false);
motor.state.tick = 0; motor.state.step = 0;

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
}

const resultadosDir = path.join(__dirname, '..', 'resultados');
fs.mkdirSync(resultadosDir, { recursive: true });

fs.writeFileSync(
  path.join(resultadosDir, `${id}.json`),
  JSON.stringify({ id, config, conteoGlobal, conteoPorAnio, ticksTotales }, null, 2),
);

const anios = Object.keys(conteoPorAnio).map(Number).sort((a, b) => a - b);
const header = ['anio', 'JARDIN_FERTIL_pct', 'CIERRE_pct', 'SELVA_HOSTIL_pct', 'COLAPSO_pct', 'dominante'];
const rows = anios.map((a) => {
  const c = conteoPorAnio[a];
  const total = ZONAS.reduce((s, z) => s + c[z], 0);
  const dominante = ZONAS.reduce((mx, z) => (c[z] > c[mx] ? z : mx), ZONAS[0]);
  return [a, ...ZONAS.map((z) => (c[z] / total * 100).toFixed(2)), dominante];
});
const csv = [header.join(','), ...rows.map((r) => r.join(','))].join('\n');
fs.writeFileSync(path.join(resultadosDir, `${id}_por_anio.csv`), csv);

// Mensaje al padre para el resumen final (no imprescindible, pero evita
// que el padre tenga que releer el JSON que este mismo proceso ya escribió).
if (process.send) {
  process.send({ id, conteoGlobal, ticksTotales });
}
