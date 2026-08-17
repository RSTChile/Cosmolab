// Ronda 12 (11-ago-2026) -- diagnóstico rápido y barato (sin correr los 62
// años) del reemplazo de PTC: confirma que powerLive/LF/H_at YA NO están
// planos (la falla de Ronda 11, remover PTC sin reemplazo) y compara 1997
// (floración real documentada, mega Niño) vs 2019 (megasequía, control) para
// las 2 opciones ('A' velocidad de transición, 'B' amplitud de alternativas).
'use strict';
const motor = require('./motor_fisico.generado.js');

const PARAMETROS_FABRICA = {
  powerBase: 0.47, beta: 0.94, noise: 0.0079, tOpt: 13.0, ptcTc: 16.0,
  ptcSharp: 1.0, luminosity: 0.94, umbralGerminacion: 15, rezagoGyriosomus: 30,
};

function correr(modo) {
  motor.PTC_REEMPLAZO_MODO = modo;
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

  const ticksTotales = (motor.TOPE_DIA_CALENDARIO + 1) * motor.TICKS_POR_DIA;
  const objetivo = new Set(['1997-06-25','1997-07-25','1997-08-24','2019-06-25','2019-07-25','2019-08-24']);
  const muestras = [];
  const powerLiveSerie = []; // para varianza global (no-degeneración)
  let ultimoDia = -1;
  for (let i = 0; i < ticksTotales; i++) {
    motor.pasoFisica(false);
    powerLiveSerie.push(motor.state.powerLive);
    const dia = motor.diaCalendarioActual();
    if (dia !== ultimoDia) {
      ultimoDia = dia;
      const f = motor.fechaDesdeDiaCalendario(dia);
      const pad = (n) => String(n).padStart(2, '0');
      const clave = f.anio + '-' + pad(f.mes + 1) + '-' + pad(f.diaMes);
      if (objetivo.has(clave)) {
        muestras.push({ clave, powerLive: motor.state.powerLive, LF: motor.state.LF, H_at: motor.state.H_at, floracion: motor.state.floracion, A_sys_env: motor.state.A_sys_env });
      }
    }
  }
  let suma = 0, min = Infinity, max = -Infinity;
  for (const v of powerLiveSerie) { suma += v; if (v < min) min = v; if (v > max) max = v; }
  const media = suma / powerLiveSerie.length;
  let sumaSqDesv = 0;
  for (const v of powerLiveSerie) sumaSqDesv += (v - media) ** 2;
  const varianza = sumaSqDesv / powerLiveSerie.length;
  return { muestras, media, varianza, min, max };
}

for (const modo of ['A', 'B']) {
  console.log(`\n=== MODO ${modo} ===`);
  const r = correr(modo);
  console.log(`powerLive global: media=${r.media.toFixed(4)} var=${r.varianza.toFixed(6)} min=${r.min.toFixed(4)} max=${r.max.toFixed(4)}`);
  for (const m of r.muestras) {
    console.log(`  ${m.clave}: powerLive=${m.powerLive.toFixed(4)} LF=${m.LF.toFixed(4)} H_at=${m.H_at.toFixed(4)} floracion=${m.floracion.toFixed(4)} A_sys_env=${m.A_sys_env.toFixed(4)}`);
  }
}
