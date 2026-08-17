// RONDA 17 (13-ago-2026) — ¿la conclusión del experimento depende de qué
// serie de lluvia se elija para 2019-2026?
//
// EL PROBLEMA. Llegaron 38 estaciones de Coquimbo y 12 caen dentro de 60 km
// del punto-reloj cubriendo 2019-2026, el tramo que hasta ayer se rellenaba
// con reanálisis ERA5 corregido. Validando ambos métodos contra Huintil
// MEDIDO (631 meses, dejando-un-año-fuera), la estación encadenada gana
// claro: MAE 8.8 mm contra 14.9, y sobre todo fabrica 9 floraciones falsas
// contra 39. Pero la cadena arrastra un supuesto que NO se puede validar
// -- que la estación DMC nueva mide igual que la histórica co-localizada --
// y medido contra ERA5 como vara común hay un escalón de 1.00x a 1.44x
// (mediana 1.266) al cruzar 2018. Nadie puede resolverlo: ninguna estación
// cruza el corte.
//
// LA SALIDA HONESTA no es elegir la que más guste, sino correr las TRES y
// ver si el resultado del experimento aguanta en todo el rango:
//   era5      lo publicado (reanálisis corregido por sesgo)
//   cruda     estaciones encadenadas, sin tocar el escalón
//   ajustada  estaciones encadenadas, descontando el escalón mediano (÷1.266)
// 'cruda' y 'ajustada' son los dos extremos de la incertidumbre. Si las tres
// dan la misma lectura, la conclusión no depende de esta elección.
//
// La física se corre entera en cada variante porque la lluvia SÍ realimenta
// (a diferencia de e_R en la ronda 16): cambia floración, albedo y Tf.
//
// Uso: node correr_variante_lluvia_r17.js <era5|cruda|ajustada>
'use strict';
const fs = require('fs');
const path = require('path');
const motor = require('./motor_fisico.generado.js');

const variante = process.argv[2];
if (!['era5', 'cruda', 'ajustada'].includes(variante)) {
  console.error('Uso: node correr_variante_lluvia_r17.js <era5|cruda|ajustada>');
  process.exit(1);
}

const FUENTES = path.join(__dirname, '..', '..', '..', 'investigacion', 'fuentes');
const ARCHIVO = {
  cruda: 'lluvia_mensual_zhcs_reconstruida_estaciones_cruda.csv',
  ajustada: 'lluvia_mensual_zhcs_reconstruida_estaciones_ajustada.csv',
};

// La serie base del motor ya trae 1966-2018 de estación real (CR2 Huintil) y
// 2019+ de ERA5 corregido. Para las variantes de estación sólo se PISAN los
// meses de 2019 en adelante; el tramo histórico no se toca en ninguna.
let reemplazados = 0;
if (variante !== 'era5') {
  const texto = fs.readFileSync(path.join(FUENTES, ARCHIVO[variante]), 'utf-8').trim();
  const lineas = texto.split(/\r?\n/).slice(1);
  for (const l of lineas) {
    const [ym, mm] = l.split(',');
    if (ym >= '2019-01' && Object.prototype.hasOwnProperty.call(motor.PLUVIOSIDAD_MENSUAL, ym)) {
      motor.PLUVIOSIDAD_MENSUAL[ym] = Number(mm);
      reemplazados++;
    }
  }
}
console.log(`[${variante}] meses de 2019+ reemplazados: ${reemplazados}`);

const ZONAS = ['JARDIN_FERTIL', 'CIERRE', 'SELVA_HOSTIL', 'COLAPSO'];
const PARAMETROS = {
  powerBase: 0.47, beta: 0.94, noise: 0.0079, tOpt: 13.0, ptcTc: 16.0,
  ptcSharp: 1.0, luminosity: 0.94, umbralGerminacion: 15, rezagoGyriosomus: 30,
};
motor.rngTf = motor.mulberry32(motor.claveSemilla('regimen1966-2027', 'Tf'));
motor.rngEco = motor.mulberry32(motor.claveSemilla('regimen1966-2027', 'eco'));
Object.assign(motor.state, PARAMETROS, { dayNightMode: true, seasonMode: true });
motor.state.tick = 0; motor.state.step = 0;
motor.state.Tf = 24.6; motor.state.Tc = 25; motor.state.Th = 28;
motor.state.floracion = 0; motor.state.gyriosomus = 0; motor.state.sueloDesnudo = 1;
motor.state.floracionHistorial = [];
motor.state.powerLive = motor.state.powerBase; motor.state._A_prev = 0;
motor.resetField(); motor.resetBuffers();

for (let i = 0; i < 60 * motor.TICKS_POR_DIA; i++) motor.pasoFisica(false);
motor.state.tick = 0; motor.state.step = 0;

const N = (motor.TOPE_DIA_CALENDARIO + 1) * motor.TICKS_POR_DIA;
const conteo = {};
const sLF = [], sD = [], sA = [], sE = [];
const t0 = Date.now();
for (let i = 0; i < N; i++) {
  motor.pasoFisica(false);
  const dia = motor.diaCalendarioActual();
  const anio = motor.ANIO_CERO + Math.floor(dia / motor.DIAS_POR_ANIO_CAL);
  const zona = motor.clasificarCierre().zona;
  if (!conteo[anio]) conteo[anio] = { JARDIN_FERTIL: 0, CIERRE: 0, SELVA_HOSTIL: 0, COLAPSO: 0 };
  conteo[anio][zona]++;
  if (i % 60 === 0) {   // una muestra por día: suficiente para percentiles
    sLF.push(motor.state.LF); sD.push(motor.state.deltaStruct);
    sA.push(motor.state.A_sys_env); sE.push(motor.state.err);
  }
}
console.log(`[${variante}] física en ${((Date.now() - t0) / 60000).toFixed(1)} min`);

// LA TABLA DE INVARIANTES — el control que faltaba antes de publicar (r16):
// cada invariante contra su umbral, ANTES de mirar los cinco criterios.
function pct(a, q) { const c = a.slice().sort((x, y) => x - y); return c[Math.floor(q * c.length)]; }
console.log(`\n[${variante}] invariantes contra su umbral`);
console.log('    inv          p10      p50      p90      máx   umbral');
for (const [nom, arr, u] of [['LF', sLF, 0.35], ['Δ_struct', sD, 0], ['A_sys_env', sA, 0.70], ['|e_R|', sE, 0.20]]) {
  const f = (x) => x.toFixed(4).padStart(8);
  console.log(`    ${nom.padEnd(10)}${f(pct(arr, .1))} ${f(pct(arr, .5))} ${f(pct(arr, .9))} ${f(Math.max(...arr))} ${u.toFixed(4).padStart(8)}`);
}
console.log(`    e_R = 0: ${(100 * sE.filter((v) => !(v > 0)).length / sE.length).toFixed(2)}% del tiempo`);

const anios = Object.keys(conteo).map(Number).sort((a, b) => a - b);
const filas = anios.map((a) => {
  const c = conteo[a], tot = ZONAS.reduce((s, z) => s + c[z], 0);
  const dom = ZONAS.reduce((mx, z) => (c[z] > c[mx] ? z : mx), ZONAS[0]);
  return [a, ...ZONAS.map((z) => (c[z] / tot * 100).toFixed(2)), dom].join(',');
});
const ruta = path.join(__dirname, `r17_${variante}_por_anio.csv`);
fs.writeFileSync(ruta, ['anio,JARDIN_FERTIL_pct,CIERRE_pct,SELVA_HOSTIL_pct,COLAPSO_pct,dominante',
                        ...filas].join('\n'));
console.log(`[${variante}] escrito ${path.basename(ruta)}`);
