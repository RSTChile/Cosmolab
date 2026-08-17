// Ronda 12 (11-ago-2026) -- segunda pasada, ya con los κ recalibrados
// (recalibrar_kappa_semanal.js) puestos en el motor ANTES de correr, para
// que clasificarCierre() clasifique cada tick con el umbral correcto (la
// primera pasada, calcular_percentiles_y_regimen.js, solo sirve para medir
// la distribución real y no puede auto-clasificarse con κ que todavía no
// existían). Mismo patrón que las rondas 5-9 de esta sesión.
// Uso: node correr_final_con_kappa.js <sufijo> <modoPTC A|B> kappa_reemplazo<X>.txt
'use strict';
const fs = require('fs');
const path = require('path');
const motor = require('./motor_fisico.generado.js');

const sufijo = process.argv[2];
const modoPTC = process.argv[3];
const rutaKappa = process.argv[4];
if (!sufijo || !modoPTC || !rutaKappa) {
  console.error('Uso: node correr_final_con_kappa.js <sufijo> <A|B> kappa_reemplazo<X>.txt');
  process.exit(1);
}
motor.PTC_REEMPLAZO_MODO = modoPTC;
// Ronda 13 (11-ago-2026): mismas banderas por variable de entorno que
// calcular_percentiles_y_regimen.js, para correr C1/C2/C3 sin regenerar.
if (process.env.LF_SIN_V !== undefined) motor.LF_SIN_V = process.env.LF_SIN_V === 'true';
if (process.env.KAPPA_CANONICOS !== undefined) motor.KAPPA_CANONICOS = process.env.KAPPA_CANONICOS === 'true';
if (process.env.KAPPA_LF_INFIMO) motor.KAPPA_LF_INFIMO = Number(process.env.KAPPA_LF_INFIMO);
if (process.env.KAPPA_DELTA_INFIMO) motor.KAPPA_DELTA_INFIMO = Number(process.env.KAPPA_DELTA_INFIMO);
console.error(`config: PTC=${motor.PTC_REEMPLAZO_MODO} LF_SIN_V=${motor.LF_SIN_V} KAPPA_CANONICOS=${motor.KAPPA_CANONICOS} kLFinf=${motor.KAPPA_LF_INFIMO} kDinf=${motor.KAPPA_DELTA_INFIMO}`);
const textoKappa = fs.readFileSync(rutaKappa, 'utf-8');
// Parseo por regex (no eval -- en 'use strict' un eval directo no filtra sus
// `let` al scope que lo llama): el archivo solo contiene 4 líneas
// `let KAPPA_..._POR_SEMANA = [...]` generadas por recalibrar_kappa_semanal.js.
function extraerArray(nombre) {
  const m = textoKappa.match(new RegExp(`let ${nombre} = (\\[[^\\]]*\\]);`));
  if (!m) throw new Error(`no se encontró ${nombre} en ${rutaKappa}`);
  return JSON.parse(m[1]);
}
motor.KAPPA_LF_POR_SEMANA = extraerArray('KAPPA_LF_POR_SEMANA');
motor.KAPPA_DELTA_POR_SEMANA = extraerArray('KAPPA_DELTA_POR_SEMANA');
motor.KAPPA_V_POR_SEMANA = extraerArray('KAPPA_V_POR_SEMANA');
motor.KAPPA_O_POR_SEMANA = extraerArray('KAPPA_O_POR_SEMANA');

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
const ticksConTempRealPorAnio = {};
const ticksTotalPorAnio = {};

for (let i = 0; i < ticksTotales; i++) {
  motor.pasoFisica(false);
  const dia = motor.diaCalendarioActual();
  const f = motor.fechaDesdeDiaCalendario(dia);
  const anio = f.anio;
  const zona = motor.clasificarCierre().zona;
  if (!conteoPorAnio[anio]) conteoPorAnio[anio] = { JARDIN_FERTIL: 0, CIERRE: 0, SELVA_HOSTIL: 0, COLAPSO: 0 };
  conteoPorAnio[anio][zona]++;
  ticksTotalPorAnio[anio] = (ticksTotalPorAnio[anio] || 0) + 1;
  if (motor.state.coberturaTempReal) ticksConTempRealPorAnio[anio] = (ticksConTempRealPorAnio[anio] || 0) + 1;
  if (i % 300000 === 0) {
    const transcurrido = (Date.now() - t0) / 1000;
    process.stderr.write(`${(i / ticksTotales * 100).toFixed(1)}% · ${transcurrido.toFixed(0)}s\n`);
  }
}
const t1 = Date.now();

const anios = Object.keys(conteoPorAnio).map(Number).sort((a, b) => a - b);
const header = ['anio', 'JARDIN_FERTIL_pct', 'CIERRE_pct', 'SELVA_HOSTIL_pct', 'COLAPSO_pct', 'dominante', 'coberturaTempReal_pct'];
const rows = anios.map((a) => {
  const c = conteoPorAnio[a];
  const total = ZONAS.reduce((s, z) => s + c[z], 0);
  const dominante = ZONAS.reduce((mx, z) => (c[z] > c[mx] ? z : mx), ZONAS[0]);
  const coberturaPct = (100 * (ticksConTempRealPorAnio[a] || 0) / ticksTotalPorAnio[a]).toFixed(1);
  return [a, ...ZONAS.map((z) => (c[z] / total * 100).toFixed(2)), dominante, coberturaPct];
});
const csv = [header.join(','), ...rows.map((r) => r.join(','))].join('\n');
fs.writeFileSync(path.join(__dirname, `regimen_${sufijo}_final_por_anio.csv`), csv);
console.error(`Listo en ${((t1 - t0) / 60000).toFixed(1)} min. Escrito regimen_${sufijo}_final_por_anio.csv`);
