// Verificación end-to-end del motor portado (Fase A.2 del plan de
// granularidad, 08-ago-2026): corre la MISMA configuración que el botón
// "▶ Experimento Completo" del HTML (parámetros de fábrica, Día/Noche +
// Estaciones prendidos, semilla 'regimen1966-2027', 60 días de asentamiento
// térmico, luego 1966-2027 completo) y escribe un CSV con el mismo esquema
// exacto que descargarCSVRegimenPorAnio() del HTML -- para poder compararlo
// con `diff` contra un CSV que el propio botón del HTML ya descargó.
// Correr: node verificar_experimento_completo.js
'use strict';
const fs = require('fs');
const motor = require('./motor_fisico.generado.js');

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
for (let i = 0; i < ticksTotales; i++) {
  motor.pasoFisica(false);
  const dia = motor.diaCalendarioActual();
  const anio = motor.ANIO_CERO + Math.floor(dia / motor.DIAS_POR_ANIO_CAL);
  const zona = motor.clasificarCierre().zona;
  if (!conteoPorAnio[anio]) conteoPorAnio[anio] = { JARDIN_FERTIL: 0, CIERRE: 0, SELVA_HOSTIL: 0, COLAPSO: 0 };
  conteoPorAnio[anio][zona]++;
  if (i % 100000 === 0) {
    const transcurrido = (Date.now() - t0) / 1000;
    const ritmo = (i + 1) / transcurrido;
    process.stderr.write(`${(i / ticksTotales * 100).toFixed(1)}% · ${Math.round(ritmo)} ticks/s · faltan ~${Math.round((ticksTotales - i) / ritmo / 60)} min\n`);
  }
}
const t1 = Date.now();

const anios = Object.keys(conteoPorAnio).map(Number).sort((a, b) => a - b);
const header = ['anio', 'JARDIN_FERTIL_pct', 'CIERRE_pct', 'SELVA_HOSTIL_pct', 'COLAPSO_pct', 'dominante'];
const rows = anios.map((a) => {
  const c = conteoPorAnio[a];
  const total = ZONAS.reduce((s, z) => s + c[z], 0);
  const dominante = ZONAS.reduce((mx, z) => (c[z] > c[mx] ? z : mx), ZONAS[0]);
  return [a, ...ZONAS.map((z) => (c[z] / total * 100).toFixed(2)), dominante];
});
const csv = [header.join(','), ...rows.map((r) => r.join(','))].join('\n');
fs.writeFileSync(require('path').join(__dirname, 'verificacion_experimento_completo_node.csv'), csv);
console.error(`Listo en ${((t1 - t0) / 60000).toFixed(1)} min. Escrito verificacion_experimento_completo_node.csv`);
