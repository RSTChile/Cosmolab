// Benchmark honesto del motor portado (Fase A.3 del plan de granularidad,
// 08-ago-2026). Corre 200.000 ticks SIN yields (nada que atender, es un
// proceso Node sin UI) y reporta el ritmo real medido en ESTA máquina --
// no un número prometido de antemano. Correr: node benchmark_motor.js
'use strict';
const motor = require('./motor_fisico.generado.js');

motor.rngTf = motor.mulberry32(motor.claveSemilla('bench', 'Tf'));
motor.rngEco = motor.mulberry32(motor.claveSemilla('bench', 'eco'));
Object.assign(motor.state, {
  dayNightMode: true, seasonMode: true, powerBase: 0.47, beta: 0.94,
  noise: 0.0079, tOpt: 25, ptcTc: 16, ptcSharp: 1, luminosity: 0.94,
  umbralGerminacion: 15, rezagoGyriosomus: 30,
});
motor.state.tick = 0; motor.state.step = 0;
motor.state.Tf = 24.6; motor.state.Tc = 25; motor.state.Th = 28;
motor.state.floracion = 0; motor.state.gyriosomus = 0; motor.state.sueloDesnudo = 1;
motor.state.floracionHistorial = [];
motor.state.powerLive = motor.state.powerBase; motor.state._A_prev = 0;
motor.resetField(); motor.resetBuffers();

const N = 200000;
const t0 = Date.now();
for (let i = 0; i < N; i++) motor.pasoFisica(false);
const t1 = Date.now();
const ticksPorSeg = Math.round(N / ((t1 - t0) / 1000));
const ticksTotales1966_2027 = 1357800;
const minutosPorCorridaCompleta = (ticksTotales1966_2027 / ticksPorSeg / 60).toFixed(1);
console.log(JSON.stringify({ N, ms: t1 - t0, ticksPorSeg, minutosPorCorridaCompleta1966_2027: Number(minutosPorCorridaCompleta) }, null, 2));
