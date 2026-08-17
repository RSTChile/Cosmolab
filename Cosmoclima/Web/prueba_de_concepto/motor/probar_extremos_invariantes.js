// ¿Son e_R y Δ_struct decorativos, o simplemente este sistema nunca los viola?
// (12-ago-2026, a raíz de la observación de Claude Web verificada sobre la
// corrida publicada: en 62 años e_R llega como máximo a 0.1788 contra un
// umbral de 0.20, y Δ_struct nunca baja de 0.611 contra un umbral de 0.510.
// O sea: el Plano Cierre opera de hecho con DOS de sus cuatro invariantes.)
//
// LA PREGUNTA CORRECTA no es "¿cómo hacemos que los cuatro discriminen?"
// -- bajar los umbrales a un percentil los haría discriminar por construcción,
// que es exactamente el error de plano (O-N16.2d) corregido el 11-ago. La
// pregunta es si EXISTE algún régimen donde el sistema los cruce.
//
// MÉTODO: correr el mismo motor en tres regímenes, sin tocar los umbrales.
// La palanca es `umbralGerminacion` (mm de lluvia necesarios para que la
// semilla germine), porque permite llevar la floración a sus dos extremos sin
// inventar datos climáticos falsos:
//   · SEQUÍA TOTAL   umbral 500 mm -> ninguna lluvia real lo alcanza nunca:
//                    floración 0 permanente, el desierto nunca despierta.
//   · REAL           umbral 15 mm, el valor de la literatura (control).
//   · LLUVIA SIEMPRE umbral 0 mm -> cualquier gota dispara floración plena.
// Si ni siquiera en los extremos e_R cruza 0.20 ni Δ_struct baja de 0.5102,
// entonces para este modelo son condiciones satisfechas SIEMPRE, y eso se
// declara. Si se cruzan, los cuatro ejes son reales y el problema era de
// calibración.
'use strict';
const motor = require('./motor_fisico.generado.js');

const KAPPA_O = 0.20, KAPPA_D = 0.5102, KAPPA_V = 0.70, KAPPA_LF = 0.35;

// VENTANA REDUCIDA A 20 AÑOS (decidido a mitad de la corrida, 12-ago): con las
// simulaciones de Phantom ocupando tres nucleos, los 62 años x 3 regimenes
// pedian ~3 horas de reloj. Los dos regimenes extremos son CONSTANTES por
// construccion -- el umbral de germinacion los vuelve insensibles a que llueva
// mas o menos-- asi que el sistema llega a su estado estacionario y despues
// solo repite. Para responder "¿cruza el umbral ALGUNA vez?" no hace falta
// repetir el estacionario 62 veces. Se reporta la primera mitad contra la
// segunda: si coinciden, la ventana basto; si no, hay que alargarla.
const ANIOS_PRUEBA = 20;

const PARAMETROS = {
  powerBase: 0.47, beta: 0.94, noise: 0.0079, tOpt: 13.0, ptcTc: 16.0,
  ptcSharp: 1.0, luminosity: 0.94, rezagoGyriosomus: 30,
};

function correr(umbralGerminacion, etiqueta) {
  motor.rngTf = motor.mulberry32(motor.claveSemilla('regimen1966-2027', 'Tf'));
  motor.rngEco = motor.mulberry32(motor.claveSemilla('regimen1966-2027', 'eco'));
  Object.assign(motor.state, PARAMETROS, { umbralGerminacion, dayNightMode: true, seasonMode: true });
  motor.state.tick = 0; motor.state.step = 0;
  motor.state.Tf = 24.6; motor.state.Tc = 25; motor.state.Th = 28;
  motor.state.floracion = 0; motor.state.gyriosomus = 0; motor.state.sueloDesnudo = 1;
  motor.state.floracionHistorial = [];
  motor.state.powerLive = motor.state.powerBase; motor.state._A_prev = 0;
  motor.resetField(); motor.resetBuffers();

  for (let i = 0; i < 60 * motor.TICKS_POR_DIA; i++) motor.pasoFisica(false);
  motor.state.tick = 0; motor.state.step = 0;

  const ticks = ANIOS_PRUEBA * motor.DIAS_POR_ANIO_CAL * motor.TICKS_POR_DIA;
  const mitad = Math.floor(ticks / 2);
  // Se acumula por mitad de la ventana para poder mostrar que el sistema ya
  // esta en estado estacionario y que alargar la corrida no cambiaria nada.
  const vacio = () => ({ errMax: 0, dMin: Infinity, aMin: Infinity,
                         lfMin: Infinity, lfMax: 0, florMax: 0,
                         cruzaER: 0, cruzaD: 0, cruzaV: 0, n: 0 });
  const mitades = [vacio(), vacio()];
  for (let i = 0; i < ticks; i++) {
    motor.pasoFisica(false);
    const m = mitades[i < mitad ? 0 : 1];
    const e = motor.state.err, d = motor.state.deltaStruct;
    const a = motor.state.A_sys_env, lf = motor.state.LF;
    if (e > m.errMax) m.errMax = e;
    if (d < m.dMin) m.dMin = d;
    if (a < m.aMin) m.aMin = a;
    if (lf < m.lfMin) m.lfMin = lf;
    if (lf > m.lfMax) m.lfMax = lf;
    if (motor.state.floracion > m.florMax) m.florMax = motor.state.floracion;
    if (e >= KAPPA_O) m.cruzaER++;
    if (d < KAPPA_D) m.cruzaD++;
    if (a < KAPPA_V) m.cruzaV++;
    m.n++;
  }
  const [m1, m2] = mitades;
  const errMax = Math.max(m1.errMax, m2.errMax);
  const dMin = Math.min(m1.dMin, m2.dMin);
  const aMin = Math.min(m1.aMin, m2.aMin);
  const lfMin = Math.min(m1.lfMin, m2.lfMin), lfMax = Math.max(m1.lfMax, m2.lfMax);
  const florMax = Math.max(m1.florMax, m2.florMax);
  const cruzaER = m1.cruzaER + m2.cruzaER, cruzaD = m1.cruzaD + m2.cruzaD;
  const cruzaV = m1.cruzaV + m2.cruzaV, n = m1.n + m2.n;

  const pct = (x) => (100 * x / n).toFixed(2) + '%';
  console.log(`\n${etiqueta}  (umbral de germinación ${umbralGerminacion} mm)`);
  console.log(`  floración máxima alcanzada: ${florMax.toFixed(3)}`);
  console.log(`  e_R        máx ${errMax.toFixed(4)}  vs κ_O ${KAPPA_O}   -> lo cruza el ${pct(cruzaER)} del tiempo`);
  console.log(`  Δ_struct   mín ${dMin.toFixed(4)}  vs κ_Δ ${KAPPA_D}   -> baja de él el ${pct(cruzaD)} del tiempo`);
  console.log(`  A_sys_env  mín ${aMin.toFixed(4)}  vs κ_V ${KAPPA_V}   -> baja de él el ${pct(cruzaV)} del tiempo`);
  console.log(`  LF         rango ${lfMin.toFixed(4)} – ${lfMax.toFixed(4)}  vs κ_LF ${KAPPA_LF}`);
  console.log(`  estacionario: e_R máx ${m1.errMax.toFixed(4)} / ${m2.errMax.toFixed(4)}` +
              `   Δ_struct mín ${m1.dMin.toFixed(4)} / ${m2.dMin.toFixed(4)}` +
              `   (1ª mitad / 2ª mitad de la ventana)`);
  return { errMax, dMin, cruzaER, cruzaD };
}

console.log('¿Existe algún régimen donde e_R o Δ_struct se violen?');
console.log('Mismos umbrales en los tres casos; solo cambia el régimen del sistema.');
const seca = correr(500, 'SEQUÍA TOTAL — el desierto nunca florece');
const real = correr(15, 'RÉGIMEN REAL — control (valor de la literatura)');
const humeda = correr(0, 'LLUVIA SIEMPRE — floración permanente');

console.log('\n' + '='.repeat(66));
const nunca = (seca.cruzaER + real.cruzaER + humeda.cruzaER) === 0;
const nuncaD = (seca.cruzaD + real.cruzaD + humeda.cruzaD) === 0;
console.log(`e_R      ${nunca ? 'NO se viola en NINGUNO de los tres regímenes' : 'SÍ se viola en algún régimen'}`);
console.log(`Δ_struct ${nuncaD ? 'NO se viola en NINGUNO de los tres regímenes' : 'SÍ se viola en algún régimen'}`);
console.log('\nLectura: si no se violan ni en los extremos, para ESTE modelo son');
console.log('condiciones satisfechas siempre — y el Plano Cierre opera con dos ejes.');
