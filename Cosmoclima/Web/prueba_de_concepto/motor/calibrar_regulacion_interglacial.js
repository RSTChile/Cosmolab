// Ronda 10 (10-ago-2026) -- calibración real de KAPPA_REGULACION_INTERGLACIAL,
// el término que le falta a computeFloracion() para que la floración
// enfríe el planeta (regulación interglacial) en vez de solo calentarlo
// débilmente vía el albedo real (He et al. 2017, efecto chico y de signo
// contrario al que ya medimos con datos reales esta sesión).
//
// Método: correr el motor TAL CUAL ESTÁ (sin el término nuevo -- floración
// no depende de Tf/targetTf, así que no hay circularidad), registrar
// (día real, state.floracion) para los 62 años, y cruzarlo contra la
// temperatura MÁXIMA real (TEMPERATURA_DIARIA_ZHCS, 1981-2026, la misma
// que ya conectamos a la física) del mismo día. La pendiente de la
// regresión lineal simple tmax_real ~ floracion (sobre los días con dato
// real) es el número: cuántos °C baja la temperatura real por cada unidad
// de floración -- un dato medido, no un valor puesto a ojo para que el
// resultado final "se vea bien".
'use strict';
const fs = require('fs');
const path = require('path');
const motor = require('./motor_fisico.generado.js');

const htmlPath = path.join(__dirname, '..', 'sim-cosmoclima.html');
const html = fs.readFileSync(htmlPath, 'utf-8');
const m = html.match(/const TEMPERATURA_DIARIA_ZHCS = (\{[^;]+\});/);
const TEMP_REAL = JSON.parse(m[1]);

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

function claveFecha(dia) {
  const f = motor.fechaDesdeDiaCalendario(dia);
  const pad = (n) => String(n).padStart(2, '0');
  return f.anio + '-' + pad(f.mes + 1) + '-' + pad(f.diaMes);
}

const pares = []; // {floracion, tmax}
let ultimoDia = -1;
for (let i = 0; i < ticksTotales; i++) {
  motor.pasoFisica(false);
  const dia = motor.diaCalendarioActual();
  if (dia !== ultimoDia) {
    ultimoDia = dia;
    const clave = claveFecha(dia);
    const real = TEMP_REAL[clave];
    if (real && Number.isFinite(real.tmax)) {
      pares.push({ floracion: motor.state.floracion, tmax: real.tmax });
    }
  }
  if (i % 300000 === 0) {
    const transcurrido = (Date.now() - t0) / 1000;
    process.stderr.write(`${(i / ticksTotales * 100).toFixed(1)}% · ${transcurrido.toFixed(0)}s\n`);
  }
}
const t1 = Date.now();

// Regresión lineal simple: tmax = a + b*floracion
const n = pares.length;
const mediaF = pares.reduce((s, p) => s + p.floracion, 0) / n;
const mediaT = pares.reduce((s, p) => s + p.tmax, 0) / n;
let num = 0, den = 0;
for (const p of pares) { num += (p.floracion - mediaF) * (p.tmax - mediaT); den += (p.floracion - mediaF) ** 2; }
const b = num / den; // pendiente: °C por unidad de floracion
const a = mediaT - b * mediaF;

console.log(`Listo en ${((t1 - t0) / 60000).toFixed(1)} min. n=${n} días con dato real (1981-2026).`);
console.log(`floración: media=${mediaF.toFixed(4)}, rango observado en estos pares: [${Math.min(...pares.map(p=>p.floracion)).toFixed(3)}, ${Math.max(...pares.map(p=>p.floracion)).toFixed(3)}]`);
console.log(`tmax real: media=${mediaT.toFixed(2)}°C`);
console.log(`Regresión tmax_real = ${a.toFixed(3)} + ${b.toFixed(4)} * floracion`);
console.log(`KAPPA_REGULACION_INTERGLACIAL (pendiente, signo esperado negativo) = ${b.toFixed(4)}`);

fs.writeFileSync(path.join(__dirname, 'calibracion_regulacion_interglacial.json'), JSON.stringify({ n, mediaF, mediaT, a, b, KAPPA_REGULACION_INTERGLACIAL: b }, null, 2));
